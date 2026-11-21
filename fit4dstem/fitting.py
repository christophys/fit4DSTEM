# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Fitting of thickness, tilt, Ti displacement (and the BaTiO3/SrTiO3 fraction) pattern by pattern (tab 3).

Model of one pattern (counts) on the disc pixels:

    M = s * sum_materials f_m * I_g(k_pixel - tau; t, u, w) * A(|q|) + b * B

* I_g: rocking curves of the library (Catmull-Rom in thickness and displacement, bilinear in k),
* k_pixel: incident direction of the pixel inside its disc (from the calibration: undistorted position, per-pattern
  lattice, crystal assignment), tau: crystal tilt, A: aperture edge (error function, width from tab 2),
* B: background interpolated from the pixels between the discs (tab 2), s and b: linear (solved in every step).

Parameters: thickness t (nm), tilt (tau_x, tau_y) (mrad, projection axes), Ti displacement (u, w) (pm, projection
axes x = in-plane, y = growth), optionally the SrTiO3 fraction f (linear mixture of the two libraries; a mixed
column at an interface is approximated by an incoherent mixture).  The objective is the sum of squared Anscombe
residuals 2 (sqrt(y + 3/8) - sqrt(M + 3/8)) (variance-stabilized Poisson), minimized with scipy's trust-region
least squares; uncertainties from the Jacobian at the solution.

Maps: neighbouring patterns have similar parameters, so the map is processed in snake order with the solutions of
already fitted neighbours (and of the coarse seed fits) as starting points; every pattern is started from the best
of these candidates.  A second pass in the opposite direction re-fits a pattern from its neighbours' solutions when
they explain the pattern better (escaping local minima).  The patterns are reduced to their disc pixels (plus the
background there) once and kept in memory (or a memory-mapped cache), so the passes do not re-read the data.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import erfc

from . import crystal as cr

PARAMS = ("thickness_nm", "tilt_x_mrad", "tilt_y_mrad", "u_pm", "w_pm", "f_STO")


# ====================================================================== geometry ==


@dataclass
class Geometry:
    """Disc pixels of one pattern: flat pixel indices, library reflection index, incident direction (mrad, crystal
    projection frame) and aperture weight."""
    pix: np.ndarray
    gi: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    ap: np.ndarray


class DetectorModel:
    """Maps library rocking curves onto the detector pixels of a pattern."""

    def __init__(self, libs, cal, edge_factor=2.5):
        self.libs = dict(libs)                      # name -> multislice.Library
        self.names = list(self.libs)
        self.cal = cal
        lib0 = self.libs[self.names[0]]
        self.lam = cr.wavelength_A(lib0.meta["energy_eV"])
        # common reflections of all libraries (indices of the zone cell); the detector positions follow the lattice
        # measured in each pattern, the crystal vectors of the first library define the assignment
        common = [tuple(h) for h in lib0.hk]
        for nm in self.names[1:]:
            if self.libs[nm].meta.get("zone") != lib0.meta.get("zone"):
                raise ValueError("the libraries must be computed for the same zone axis")
            other = set(map(tuple, self.libs[nm].hk))
            common = [h for h in common if h in other]
        self.sel = {}
        for nm, lib in self.libs.items():
            idx = {tuple(h): i for i, h in enumerate(lib.hk)}
            self.sel[nm] = np.array([idx[h] for h in common], int)
        self.hk = np.array(common, int)
        self.g = lib0.g[self.sel[self.names[0]]]    # (ng, 2) 1/A, projection frame
        self.edge_factor = edge_factor
        self.U = cal.ensure_U()
        self.Uflat = self.U.reshape(-1, 2)

    def mapping(self, G):
        """M (px per 1/A) of a pattern from its detector lattice G and the crystal assignment."""
        return np.asarray(G) @ np.linalg.inv(self.cal.B_cry)

    def geometry(self, c, G, pix=None):
        """Disc pixels of a pattern with lattice (c, G).  With ``pix`` the pixel set is kept (lattice refinement)."""
        cal = self.cal
        M = self.mapping(G)
        p = np.asarray(c) + self.g @ M.T                         # disc centres (undistorted px)
        r = cal.radius_px
        rr = r + self.edge_factor * max(cal.edge_px, 0.5) + 1.5
        ny, nx = cal.det_shape
        inside = (p[:, 0] > -rr) & (p[:, 1] > -rr) & (p[:, 0] < ny - 1 + rr) & (p[:, 1] < nx - 1 + rr)
        tree = cKDTree(p[inside])
        if pix is None:
            d, j = tree.query(self.Uflat, k=1, distance_upper_bound=rr)
            ok = np.isfinite(d)
            pix = np.nonzero(ok)[0]
            gi = np.nonzero(inside)[0][j[ok]]
        else:
            d, j = tree.query(self.Uflat[pix], k=1)
            gi = np.nonzero(inside)[0][j]
        q = self.Uflat[pix] - p[gi]                              # px
        Minv = np.linalg.inv(M)
        k = (q @ Minv.T) * self.lam * 1e3                       # mrad (projection frame)
        rq = np.hypot(q[:, 0], q[:, 1])
        ap = 0.5 * erfc((rq - r) / (math.sqrt(2) * max(cal.edge_px, 0.3)))
        return Geometry(pix, gi, k[:, 0], k[:, 1], ap.astype(np.float32))

    def curves(self, t_nm, u, w):
        out = {}
        for nm, lib in self.libs.items():
            c = lib.curves(t_nm * 10.0 / lib.Lz, u, w)
            sel = self.sel[nm]
            out[nm] = c if (len(sel) == len(c) and np.all(sel == np.arange(len(c)))) else c[sel]
        return out

    def render(self, curves, geo, tilt, frac=None):
        """Intensity (fraction of the incident beam, before scaling) on the pixels of geo."""
        kx = geo.kx - tilt[0]
        ky = geo.ky - tilt[1]
        if len(self.names) == 1 or frac is None:
            lib = self.libs[self.names[0]]
            return lib.sample(curves[self.names[0]], geo.gi, kx, ky) * geo.ap
        f = float(np.clip(frac, 0, 1))
        a = self.libs[self.names[0]].sample(curves[self.names[0]], geo.gi, kx, ky)
        b = self.libs[self.names[1]].sample(curves[self.names[1]], geo.gi, kx, ky)
        return ((1 - f) * a + f * b) * geo.ap

    def limits(self):
        lib = self.libs[self.names[0]]
        tmax = min(l.thickness_nm[-1] for l in self.libs.values())
        tmin = max(l.thickness_nm[0] for l in self.libs.values())
        tilt_max = max(0.0, lib.k[-1] - self.cal.radius_px * (self.cal.mrad_per_px or 0) - 0.5)
        return dict(t=(tmin, tmax), u=(lib.u[0], lib.u[-1]), w=(lib.w[0], lib.w[-1]), tilt=tilt_max)


# ====================================================================== reduced data ==


@dataclass
class Reduced:
    """One pattern reduced to its disc pixels: counts y, background B, geometry."""
    y: np.ndarray
    B: np.ndarray
    geo: Geometry
    counts: float = 0.0
    c: np.ndarray = None
    G: np.ndarray = None


def reduce_frame(frame, c, G, model: DetectorModel, gain=1.0):
    geo = model.geometry(c, G)
    B = model.cal.background(frame, c, G)
    y = frame.ravel()[geo.pix] / gain
    return Reduced(y.astype(np.float32), (B.ravel()[geo.pix] / gain).astype(np.float32), geo, float(frame.sum() / gain),
                   np.asarray(c, float), np.asarray(G, float))


# ====================================================================== single fit ==


def _anscombe(y):
    return 2.0 * np.sqrt(np.maximum(y, 0) + 0.375)


def linear_scale(I, B, y, fit_background=True, n_iter=3):
    """Poisson-weighted least squares of y ~ s I + b B (s, b >= 0)."""
    M = np.maximum(I.mean() * y.sum() / max(I.sum(), 1e-12) * np.ones_like(y), 1.0)
    s, b = y.sum() / max(I.sum(), 1e-12), 1.0
    for _ in range(n_iter):
        w = 1.0 / np.maximum(M, 1.0)
        if fit_background and B is not None and np.any(B > 0):
            A = np.array([[np.sum(w * I * I), np.sum(w * I * B)], [np.sum(w * I * B), np.sum(w * B * B)]])
            r = np.array([np.sum(w * I * y), np.sum(w * B * y)])
            try:
                s, b = np.linalg.solve(A, r)
            except np.linalg.LinAlgError:
                pass
            s, b = max(s, 1e-12), max(b, 0.0)
        else:
            Bv = 0 if B is None else B
            s = max(np.sum(w * I * (y - Bv)) / max(np.sum(w * I * I), 1e-30), 1e-12)
            b = 1.0
        M = s * I + b * (0 if B is None else B)
    return s, b, M


@dataclass
class FitOptions:
    mixture: bool = False          # fit the SrTiO3 fraction (needs two libraries)
    fit_background: bool = True   # free background scale b
    fix_tilt: bool = False
    max_nfev: int = 60
    t_grid_step_nm: float = 1.0
    seed_uw: tuple = (-25.0, 0.0, 25.0)
    seed_tilt: tuple = (-1.0, 0.0, 1.0)
    gain: float = 1.0              # detector counts per electron
    refine_lattice: bool = True    # model-based refinement of beam position and disc lattice (strain, field)


class PatternFitter:
    def __init__(self, model: DetectorModel, options: FitOptions = None):
        self.model = model
        self.opt = options or FitOptions()
        self.lim = model.limits()
        self.nparam = 6 if (self.opt.mixture and len(model.names) > 1) else 5

    # ---------------------------------------------------------------- objective
    def predict(self, red: Reduced, p, curves=None):
        t, tx, ty, u, w = p[:5]
        f = p[5] if self.nparam == 6 else None
        if curves is None:
            curves = self.model.curves(t, u, w)
        I = self.model.render(curves, red.geo, (tx, ty), f)
        s, b, M = linear_scale(I, red.B, red.y, self.opt.fit_background)
        return M, s, b

    def cost(self, red, p):
        M, _, _ = self.predict(red, p)
        r = _anscombe(red.y) - _anscombe(M)
        return float(r @ r)

    def residuals(self, red, p, curves=None):
        M, _, _ = self.predict(red, p, curves)
        return _anscombe(red.y) - _anscombe(M)

    def _bounds(self):
        L = self.lim
        lo = [L["t"][0], -L["tilt"], -L["tilt"], L["u"][0], L["w"][0]]
        hi = [L["t"][1], L["tilt"], L["tilt"], L["u"][1], L["w"][1]]
        if self.nparam == 6:
            lo.append(0.0)
            hi.append(1.0)
        return np.array(lo, float), np.array(hi, float)

    def _jac_fun(self, red):
        steps = np.array([0.05, 0.02, 0.02, 0.3, 0.3, 0.02])[:self.nparam]
        lo, hi = self._bounds()

        def fun(p):
            return self.residuals(red, p)

        def jac(p):
            curves0 = self.model.curves(p[0], p[3], p[4])
            r0 = self.residuals(red, p, curves0)
            J = np.empty((len(r0), self.nparam))
            for i in range(self.nparam):
                h = steps[i]
                q = p.copy()
                if q[i] + h > hi[i]:
                    h = -h
                q[i] += h
                cv = curves0 if i in (1, 2, 5) else None       # tilt and fraction re-use the rocking curves
                J[:, i] = (self.residuals(red, q, cv) - r0) / h
            return J
        return fun, jac

    def local_fit(self, red, p0):
        lo, hi = self._bounds()
        p0 = np.clip(np.asarray(p0, float)[:self.nparam], lo + 1e-9, hi - 1e-9)
        fun, jac = self._jac_fun(red)
        if self.opt.fix_tilt:
            lo[1:3] = p0[1:3] - 1e-9
            hi[1:3] = p0[1:3] + 1e-9
        try:
            r = least_squares(fun, p0, jac=jac, bounds=(lo, hi), method="trf", x_scale=np.array(
                [1.0, 0.3, 0.3, 5.0, 5.0, 0.2])[:self.nparam], max_nfev=self.opt.max_nfev, xtol=1e-6, ftol=1e-8)
        except Exception:  # noqa: BLE001
            return p0, self.cost(red, p0), None
        J = r.jac
        try:
            cov = np.linalg.pinv(J.T @ J) * max(1.0, 2 * r.cost / max(len(r.fun) - self.nparam, 1))
            sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        except Exception:  # noqa: BLE001
            sig = None
        return r.x, float(2 * r.cost), sig

    def global_search(self, red, n_best=3, tilt_guess=(0.0, 0.0)):
        """Grid search over thickness, displacement and tilt (curves computed once per (t, u, w))."""
        L = self.lim
        ts = np.arange(L["t"][0], L["t"][1] + 1e-9, self.opt.t_grid_step_nm)
        uws = [(u, w) for u in self.opt.seed_uw for w in self.opt.seed_uw]
        tilts = [(tilt_guess[0] + a, tilt_guess[1] + b) for a in self.opt.seed_tilt for b in self.opt.seed_tilt]
        fr = [None] if self.nparam == 5 else [0.0, 1.0]
        res = []
        y_a = _anscombe(red.y)
        for t in ts:
            for u, w in uws:
                cv = self.model.curves(t, u, w)
                for tx, ty in tilts:
                    for f in fr:
                        I = self.model.render(cv, red.geo, (tx, ty), f)
                        _, _, M = linear_scale(I, red.B, red.y, self.opt.fit_background)
                        r = y_a - _anscombe(M)
                        p = [t, tx, ty, u, w] + ([] if f is None else [f])
                        res.append((float(r @ r), p))
        res.sort(key=lambda a: a[0])
        return [np.array(p) for _, p in res[:n_best]]

    def refine_lattice(self, red, p, c, G, max_nfev=30):
        """Model-based refinement of the beam position c and the lattice G of a pattern (structural parameters p
        fixed): the disc edges and the intensity distribution inside the discs are used, which is unbiased and
        more precise than the edge correlation of tab 2.  Updates red.geo; returns (c, G, sigma (6))."""
        t, tx, ty, u, w = p[:5]
        f = p[5] if self.nparam == 6 else None
        curves = self.model.curves(t, u, w)
        y_a = _anscombe(red.y)
        pix = red.geo.pix
        c0, G0 = np.asarray(c, float), np.asarray(G, float)

        def res(x):
            geo = self.model.geometry(c0 + x[:2], G0 + x[2:].reshape(2, 2), pix=pix)
            I = self.model.render(curves, geo, (tx, ty), f)
            _, _, M = linear_scale(I, red.B, red.y, self.opt.fit_background)
            return y_a - _anscombe(M)
        try:
            r = least_squares(res, np.zeros(6), diff_step=np.full(6, 1e-3), x_scale=np.array([0.1, 0.1] + [0.02] * 4),
                              bounds=(np.array([-3, -3] + [-1] * 4), np.array([3, 3] + [1] * 4)), max_nfev=max_nfev)
        except Exception:  # noqa: BLE001
            return c0, G0, None
        J = r.jac
        try:
            cov = np.linalg.pinv(J.T @ J) * max(1.0, 2 * r.cost / max(len(r.fun) - 6, 1))
            sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        except Exception:  # noqa: BLE001
            sig = None
        c1, G1 = c0 + r.x[:2], G0 + r.x[2:].reshape(2, 2)
        red.geo = self.model.geometry(c1, G1, pix=pix)
        return c1, G1, sig

    def fit(self, red, starts=None, global_search=False, n_local=2):
        """Best local fit over the start points (and a global grid search if requested)."""
        cands = [np.asarray(s, float)[:self.nparam] for s in (starts or [])]
        if self.nparam == 6:
            cands = [np.append(c, 0.5) if len(c) == 5 else c for c in cands]
        if global_search or not cands:
            cands += self.global_search(red, n_best=3)
        costs = [self.cost(red, c) for c in cands]
        order = np.argsort(costs)
        best = None
        for i in order[:n_local]:
            p, c, s = self.local_fit(red, cands[i])
            if best is None or c < best[1]:
                best = (p, c, s)
        return best


# ====================================================================== maps ==


@dataclass
class MapResult:
    scan_shape: tuple
    params: np.ndarray                 # (sy, sx, nparam)
    sigma: np.ndarray                  # (sy, sx, nparam)
    cost: np.ndarray                   # (sy, sx) sum of squared Anscombe residuals
    npix: np.ndarray                   # (sy, sx)
    scale: np.ndarray                  # (sy, sx)  s
    bg_scale: np.ndarray               # (sy, sx)  b
    done: np.ndarray                   # (sy, sx) bool
    names: tuple = PARAMS[:5]
    centre: np.ndarray = None          # (sy, sx, 2) refined beam position (undistorted px)
    lattice: np.ndarray = None         # (sy, sx, 2, 2) refined disc lattice
    centre_sigma: np.ndarray = None    # (sy, sx, 2)
    bin: int = 1
    info: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(self.scan_shape)

    def map(self, name):
        return self.params[..., self.names.index(name)]

    def reduced_chi2(self):
        return self.cost / np.maximum(self.npix - len(self.names), 1)


class MapFitter:
    """Fits all patterns of a map (optionally binned in the scan) with neighbour seeding.

    ds: io.Dataset4D; cal: calibration (centre_map and G_map: full pass of tab 2, or interpolated coarse maps);
    model: DetectorModel; seeds: dict {(iy, ix): params} of coarse fits (scan coordinates of the binned map)."""

    def __init__(self, ds, cal, model, options=None, scan_bin=1, roi=None, progress=None, log=None, stop=None,
                 n_threads=1):
        self.ds, self.cal, self.model = ds, cal, model
        self.opt = options or FitOptions()
        self.fitter = PatternFitter(model, self.opt)
        self.bin = int(scan_bin)
        sy, sx = ds.scan_shape
        self.roi = roi or (0, sy, 0, sx)
        y0, y1, x0, x1 = self.roi
        self.shape = ((y1 - y0) // self.bin, (x1 - x0) // self.bin)
        self.progress = progress or (lambda f, msg="": None)
        self.log = log or (lambda m: None)
        self.stop = stop or (lambda: False)
        self.reduced = {}
        self.n_threads = max(1, int(n_threads))
        n = self.fitter.nparam
        Y, X = self.shape
        self.result = MapResult(self.shape, np.full((Y, X, n), np.nan), np.full((Y, X, n), np.nan),
                                np.full((Y, X), np.nan), np.zeros((Y, X), int), np.full((Y, X), np.nan),
                                np.full((Y, X), np.nan), np.zeros((Y, X), bool), PARAMS[:n], bin=self.bin,
                                centre=np.full((Y, X, 2), np.nan), lattice=np.full((Y, X, 2, 2), np.nan),
                                centre_sigma=np.full((Y, X, 2), np.nan))

    # ---------------------------------------------------------------- data
    def scan_index(self, iy, ix):
        """Raw scan positions of binned map pixel (iy, ix)."""
        y0, _, x0, _ = self.roi
        b = self.bin
        return [(y0 + iy * b + a, x0 + ix * b + c) for a in range(b) for c in range(b)]

    def _calib_at(self, iy, ix):
        pos = self.scan_index(iy, ix)
        cal = self.cal
        if cal.centre_map is not None:
            c = np.mean([cal.centre_map[p] for p in pos], 0)
            G = np.mean([cal.G_map[p] for p in pos], 0)
        else:
            c, G = cal.centre0, cal.G0
        return c, G

    def load(self, rows=None):
        """Read the patterns (row blocks), bin them in the scan, reduce them to the disc pixels."""
        Y, X = self.shape
        b = self.bin
        y0, _, x0, _ = self.roi
        rows = range(Y) if rows is None else rows
        t0 = time.time()
        for k, iy in enumerate(rows):
            block = self.ds.rows(y0 + iy * b, y0 + (iy + 1) * b, x0, x0 + X * b)
            block = block.reshape(b, X, b, *block.shape[2:]).sum(axis=(0, 2)) if b > 1 else block[0]
            for ix in range(X):
                if b > 1:
                    # descan differs between the binned positions: the mean centre is used (small for small bins)
                    pass
                c, G = self._calib_at(iy, ix)
                self.reduced[(iy, ix)] = reduce_frame(block[ix], c, G, self.model, self.opt.gain)
            self.progress((k + 1) / len(rows), f"reading row {k + 1}/{len(rows)}")
            if self.stop():
                raise InterruptedError("stopped")
        self.log(f"read and reduced {len(self.reduced)} patterns in {time.time() - t0:.0f} s")

    # ---------------------------------------------------------------- fitting
    def _store(self, iy, ix, p, c, s, red):
        R = self.result
        n = self.fitter.nparam
        R.params[iy, ix, :n] = p[:n]
        if s is not None:
            R.sigma[iy, ix, :n] = s[:n]
        R.cost[iy, ix] = c
        R.npix[iy, ix] = len(red.y)
        _, sc, bs = self.fitter.predict(red, p)
        R.scale[iy, ix], R.bg_scale[iy, ix] = sc, bs
        R.done[iy, ix] = True
        if red.c is not None:
            R.centre[iy, ix] = red.c
            R.lattice[iy, ix] = red.G

    def _refine_geometry(self, red, p):
        """Model-based lattice refinement, then a final local fit of the structural parameters."""
        if not self.opt.refine_lattice or red.c is None:
            return p, None, None
        c, G, sg = self.fitter.refine_lattice(red, p, red.c, red.G)
        red.c, red.G = c, G
        p2, c2, s2 = self.fitter.local_fit(red, p)
        iy_ix = None
        return p2, c2, (s2, sg)

    def fit_seeds(self, positions):
        """Global fits at seed positions (binned map coordinates)."""
        t0 = time.time()
        for k, (iy, ix) in enumerate(positions):
            red = self.reduced[(iy, ix)]
            p, c, s = self.fitter.fit(red, global_search=True, n_local=3)
            p, c, s = self._finish(red, p, c, s, iy, ix)
            self._store(iy, ix, p, c, s, red)
            self.progress((k + 1) / len(positions), f"seed fit {k + 1}/{len(positions)}")
            if self.stop():
                raise InterruptedError("stopped")
        self.seeds = list(positions)
        self.log(f"{len(positions)} seed fits (global search) in {time.time() - t0:.0f} s")

    def _neighbour_starts(self, iy, ix, radius=1):
        R = self.result
        Y, X = self.shape
        out = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                a, b = iy + dy, ix + dx
                if 0 <= a < Y and 0 <= b < X and R.done[a, b]:
                    out.append(R.params[a, b].copy())
        # nearest seed as a fall-back
        if getattr(self, "seeds", None):
            d = [abs(a - iy) + abs(b - ix) for a, b in self.seeds]
            a, b = self.seeds[int(np.argmin(d))]
            if self.result.done[a, b]:
                out.append(self.result.params[a, b].copy())
        # remove near-duplicates
        uniq = []
        for p in out:
            if not any(np.allclose(p, q, atol=1e-3) for q in uniq):
                uniq.append(p)
        return uniq

    def _fit_one(self, iy, ix):
        red = self.reduced[(iy, ix)]
        starts = self._neighbour_starts(iy, ix)
        p, c, s = self.fitter.fit(red, starts, global_search=not starts, n_local=1)
        p, c, s = self._finish(red, p, c, s, iy, ix)
        self._store(iy, ix, p, c, s, red)

    def _finish(self, red, p, c, s, iy, ix):
        if self.opt.refine_lattice and red.c is not None and not getattr(red, "_refined", False):
            p2, c2, ss = self._refine_geometry(red, p)
            red._refined = True
            if ss is not None:
                if ss[1] is not None:
                    self.result.centre_sigma[iy, ix] = ss[1][:2]
                if c2 <= c * 1.001:
                    return p2, c2, ss[0]
        return p, c, s

    def first_pass(self):
        """Snake order through the map: each pattern starts from its fitted neighbours (and the nearest seed)."""
        Y, X = self.shape
        order = [(iy, ix) for iy in range(Y) for ix in (range(X) if iy % 2 == 0 else range(X - 1, -1, -1))]
        order = [p for p in order if not self.result.done[p]]
        t0 = time.time()
        if self.n_threads > 1:
            self._parallel_strips(order)
        else:
            for k, (iy, ix) in enumerate(order):
                self._fit_one(iy, ix)
                if k % 5 == 0:
                    el = time.time() - t0
                    self.progress((k + 1) / len(order), f"fitting {k + 1}/{len(order)}, "
                                  f"about {el / (k + 1) * (len(order) - k - 1):.0f} s left")
                if self.stop():
                    raise InterruptedError("stopped")
        self.log(f"first pass: {len(order)} patterns in {time.time() - t0:.0f} s")

    def _parallel_strips(self, order):
        Y, X = self.shape
        n = self.n_threads
        edges = np.linspace(0, X, n + 1).astype(int)
        done = [0]
        lock = threading.Lock()
        t0 = time.time()
        total = len(order)

        def work(x0, x1):
            for iy in range(Y):
                xs = range(x0, x1) if iy % 2 == 0 else range(x1 - 1, x0 - 1, -1)
                for ix in xs:
                    if self.stop():
                        return
                    if not self.result.done[iy, ix]:
                        self._fit_one(iy, ix)
                        with lock:
                            done[0] += 1
                            if done[0] % 5 == 0:
                                el = time.time() - t0
                                self.progress(done[0] / total, f"fitting {done[0]}/{total}, about "
                                              f"{el / done[0] * (total - done[0]):.0f} s left")
        th = [threading.Thread(target=work, args=(edges[i], edges[i + 1])) for i in range(n)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if self.stop():
            raise InterruptedError("stopped")

    def refine_pass(self, max_passes=2, rel_improvement=1e-3):
        """Re-fit patterns from their neighbours' solutions when those explain the pattern better (reverse snake
        order, then forward); stops when fewer than 0.5 % of the patterns change."""
        Y, X = self.shape
        R = self.result
        for ps in range(max_passes):
            changed = 0
            order = [(iy, ix) for iy in range(Y - 1, -1, -1)
                     for ix in (range(X - 1, -1, -1) if iy % 2 == 0 else range(X))]
            if ps % 2 == 1:
                order = order[::-1]
            t0 = time.time()
            for k, (iy, ix) in enumerate(order):
                red = self.reduced[(iy, ix)]
                c_now = R.cost[iy, ix]
                cands = self._neighbour_starts(iy, ix)
                best = None
                for p in cands:
                    c = self.fitter.cost(red, p)
                    if c < c_now * (1 - rel_improvement) and (best is None or c < best[0]):
                        best = (c, p)
                if best is not None:
                    p, c, s = self.fitter.local_fit(red, best[1])
                    if c < c_now:
                        self._store(iy, ix, p, c, s, red)
                        changed += 1
                if k % 10 == 0:
                    self.progress((k + 1) / len(order), f"refinement pass {ps + 1}: {k + 1}/{len(order)}, "
                                  f"{changed} improved")
                if self.stop():
                    raise InterruptedError("stopped")
            self.log(f"refinement pass {ps + 1}: {changed} of {len(order)} patterns improved "
                     f"({time.time() - t0:.0f} s)")
            if changed < 0.005 * len(order):
                break

    def run(self, seed_positions, refine_passes=2):
        if not self.reduced:
            self.load()
        self.fit_seeds(seed_positions)
        self.first_pass()
        if refine_passes:
            self.refine_pass(refine_passes)
        return self.result


def seed_grid(shape, n=6):
    """About n x n seed positions over a (binned) map."""
    sy, sx = shape
    ys = np.unique(np.round(np.linspace(0, sy - 1, min(n, sy))).astype(int))
    xs = np.unique(np.round(np.linspace(0, sx - 1, min(n, sx))).astype(int))
    return [(int(y), int(x)) for y in ys for x in xs]


# ====================================================================== smoothing ==


def smooth_map(values, sigma, mask=None, weights=None, segments=None):
    """Gaussian smoothing of a map (normalized convolution, NaN-aware).

    segments: integer label map (e.g. 0 = BaTiO3, 1 = SrTiO3): each segment is smoothed separately, so that sharp
    interfaces are not blurred.  weights: e.g. 1 / sigma^2 of the fit."""
    v = np.asarray(values, float)
    if sigma <= 0:
        return v.copy()
    w = np.isfinite(v).astype(float)
    if mask is not None:
        w *= mask
    if weights is not None:
        w *= np.where(np.isfinite(weights), weights, 0)
    v0 = np.where(np.isfinite(v), v, 0)
    if segments is None:
        num = ndimage.gaussian_filter(v0 * w, sigma, mode="nearest")
        den = ndimage.gaussian_filter(w, sigma, mode="nearest")
        return np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)
    out = np.full_like(v, np.nan)
    for lab in np.unique(segments[np.isfinite(segments)] if segments.dtype.kind == "f" else segments):
        m = (segments == lab)
        ww = w * m
        num = ndimage.gaussian_filter(v0 * ww, sigma, mode="nearest")
        den = ndimage.gaussian_filter(ww, sigma, mode="nearest")
        sm = np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)
        out[m] = sm[m]
    return out


def segment_layers(f_map, threshold=0.5):
    """Material segmentation from the SrTiO3 fraction (0 = BaTiO3, 1 = SrTiO3)."""
    return (np.nan_to_num(f_map, nan=0.0) > threshold).astype(int)


# ====================================================================== model-based disc positions ==


def model_disc_positions(fitter, frame, c, G, p, min_pixels=12, h=0.25):
    """Positions of the individual discs from the fitted model (unbiased by the intensity inside the discs).

    For every disc the shift (dy, dx) that best moves the model onto the data is found by one weighted
    Gauss-Newton step with the gradient of the model.  Returns (q (n, 2) disc centres in raw (distorted) detector
    coordinates, hk (n, 2) indices in the detector lattice basis (float), weights (n))."""
    model = fitter.model
    red = reduce_frame(frame, c, G, model, fitter.opt.gain)
    t, tx, ty, u, w = p[:5]
    f = p[5] if fitter.nparam == 6 else None
    curves = model.curves(t, u, w)
    I = model.render(curves, red.geo, (tx, ty), f)
    s, b, M = linear_scale(I, red.B, red.y, fitter.opt.fit_background)
    grads = []
    for d in ((h, 0.0), (0.0, h)):
        cp = np.asarray(c, float) - np.array(d)          # moving the lattice by -d moves the model image by -d ...
        gp = model.geometry(cp, G, pix=red.geo.pix)
        cm = np.asarray(c, float) + np.array(d)
        gm = model.geometry(cm, G, pix=red.geo.pix)
        Ip = model.render(curves, gp, (tx, ty), f)
        Im = model.render(curves, gm, (tx, ty), f)
        grads.append(s * (Im - Ip) / (2 * h))             # d M / d (disc shift) along d
    Jy, Jx = grads
    wts = 1.0 / np.maximum(M, 1.0)
    r = red.y - M
    Mmap = model.mapping(G)
    pc = np.asarray(c) + model.g @ Mmap.T
    out_q, out_hk, out_w = [], [], []
    Ginv = np.linalg.inv(G)
    D = model.cal.distortion
    for gi in np.unique(red.geo.gi):
        m = red.geo.gi == gi
        if m.sum() < min_pixels:
            continue
        A = np.array([[np.sum(wts[m] * Jy[m] * Jy[m]), np.sum(wts[m] * Jy[m] * Jx[m])],
                      [np.sum(wts[m] * Jy[m] * Jx[m]), np.sum(wts[m] * Jx[m] * Jx[m])]])
        rhs = np.array([np.sum(wts[m] * Jy[m] * r[m]), np.sum(wts[m] * Jx[m] * r[m])])
        try:
            d = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(d)) or np.hypot(*d) > 3:
            continue
        info = float(np.sqrt(max(np.linalg.det(A), 0)))
        u_pos = pc[gi] + d
        out_q.append(D.forward(u_pos[None])[0])
        out_hk.append(Ginv @ (pc[gi] - np.asarray(c)))
        out_w.append(info)
    return np.array(out_q), np.array(out_hk), np.array(out_w)


def refine_distortion(fitter_factory, cal, frames, lattices, params, order=None, n_iter=2, log=None):
    """Model-based distortion: disc positions from the fitted model of a few patterns -> polynomial distortion
    (fit_distortion) -> repeat.  ``fitter_factory()`` must return a PatternFitter for the current calibration."""
    from . import calibration as calm
    order = order or max(cal.distortion.order, 2)
    for it in range(n_iter):
        fitter = fitter_factory()
        meas = []
        new_lat = []
        for f, (c, G), p in zip(frames, lattices, params):
            q, hk, w = model_disc_positions(fitter, f, c, G, p)
            if len(q) >= 6:
                meas.append(dict(q=q, hk=hk, w=w / max(w.max(), 1e-12)))
        if not meas:
            break
        D, lat, rms = calm.fit_distortion(meas, order=order, x0=cal.distortion.x0 if cal.distortion.order else None,
                                          scale=cal.distortion.scale if cal.distortion.order else None)
        cal.distortion = D
        cal.U = None
        fitter = fitter_factory()
        lattices = [(c, G) for c, G in lat]
        if log:
            log(f"model-based distortion {it + 1}: rms {rms:.3f} px, non-linear part up to "
                f"{D.magnitude(cal.det_shape):.2f} px")
    return cal.distortion, lattices
