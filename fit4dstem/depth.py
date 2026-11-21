# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Variation of the Ti displacement along the beam (depth) from a group of patterns (tab 3, advanced).

The crystal is divided into N slabs along the beam (equal thicknesses, or for N = 2 a free boundary), each with its
own Ti displacement (u_i, w_i).  Because the rocking curves of a stack are not a function of the rocking curves of
its parts, the model is computed with the plane-wave multislice (:class:`multislice.Engine`) for every parameter
set; perturbations of a lower slab restart from the cached wave at its top (the Jacobian needs only partial
propagations).  All patterns of a small region (e.g. one binned map pixel or a user-defined area) are fitted jointly
with their own scale, background, thickness and tilt (from the single-slab maps).  A weak regularization pulls the
slabs towards their mean; the Cramer-Rao bounds and correlations of the slab displacements tell how well the depth
profile is determined (strong anticorrelation = only the mean is determined).

The sensitivity to the depth profile is limited: it comes from dynamical scattering (the order in which the electrons
meet the slabs), and is best for thicknesses of one to a few extinction distances (see the CBED notebooks: two
opposite domains on top of each other are distinguished at >100 sigma at 10^6 electrons, but the slab displacements
are less precise than the single-slab mean by about one order of magnitude).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import crystal as cr
from . import fitting as fm
from . import multislice as ms


class SlabPropagator:
    """Rocking curves of a slab stack on the k grid of a library, with prefix caching."""

    def __init__(self, engine: ms.Engine, k_axis, hk, radius_mrad, cache=16):
        self.eng = engine
        self.k = np.asarray(k_axis, float)
        KX, KY = np.meshgrid(self.k, self.k, indexing="ij")
        self.mask = KX ** 2 + KY ** 2 <= (radius_mrad + 1.5 * (self.k[1] - self.k[0])) ** 2
        self.dirs = np.stack([KX[self.mask], KY[self.mask]], 1)
        self.hk = np.asarray(hk, int)
        self.ix, self.iy = engine.reflection_index(self.hk)
        self.P = engine._propagators(self.dirs)
        self.cache = {}
        self.cache_size = cache
        self.n_runs = 0
        self.cells_propagated = 0

    def _propagate(self, psi, n, u, w):
        eng, xp = self.eng, self.eng.xp
        T = eng.transmission(u, w)
        sf = ms._fft() if xp is np else None
        for _ in range(int(n)):
            for s in range(eng.ns):
                psi = psi * T[s][None]
                if sf is not None:
                    psi = sf.ifft2(sf.fft2(psi, axes=(1, 2), workers=-1) * self.P, axes=(1, 2), workers=-1)
                else:
                    psi = xp.fft.ifft2(xp.fft.fft2(psi, axes=(1, 2)) * self.P, axes=(1, 2))
        self.cells_propagated += int(n)
        return psi

    def curves(self, segments):
        """segments: list of (n_cells, u, w).  Returns rocking curves [ng, nk, nk]."""
        eng, xp = self.eng, self.eng.xp
        key = tuple((int(n), round(float(u), 5), round(float(w), 5)) for n, u, w in segments)
        start, psi = 0, None
        for j in range(len(key), 0, -1):
            if key[:j] in self.cache:
                start, psi = j, self.cache[key[:j]]
                break
        if psi is None:
            psi = xp.ones((len(self.dirs), eng.nx, eng.ny), np.complex64)
        for j in range(start, len(key)):
            n, u, w = key[j]
            psi = self._propagate(psi, n, u, w)
            if j < len(key) - 1:
                self.cache[key[:j + 1]] = psi
                while len(self.cache) > self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
        F = xp.fft.fft2(psi, axes=(1, 2))
        amp = F[:, self.ix, self.iy] / (eng.nx * eng.ny)
        I = amp.real ** 2 + amp.imag ** 2
        I = I.get() if hasattr(I, "get") else np.asarray(I)
        nk = len(self.k)
        out = np.zeros((len(self.hk), nk, nk), np.float32)
        out[:, self.mask] = I.T
        self.n_runs += 1
        return out


@dataclass
class DepthResult:
    n_slabs: int
    params: np.ndarray            # (N, 2) u, w per slab (pm), slab 0 = beam entrance
    sigma: np.ndarray             # (N, 2)
    boundaries_cells: list
    corr: np.ndarray
    cost: float
    cost_single: float
    n_pix: int
    polarization: np.ndarray = None   # (N, 2) uC/cm^2
    info: dict = field(default_factory=dict)

    def significance(self):
        """sqrt of the cost reduction relative to a single slab (approximate sigma)."""
        return float(np.sqrt(max(self.cost_single - self.cost, 0)))


def slab_segments(n_total, params, fractions):
    edges = np.round(np.concatenate([[0], np.cumsum(fractions)]) / np.sum(fractions) * n_total).astype(int)
    segs = []
    for i, (u, w) in enumerate(params):
        n = edges[i + 1] - edges[i]
        if n > 0:
            segs.append((int(n), float(u), float(w)))
    return segs, edges


def fit_depth(patterns, model: fm.DetectorModel, material, n_slabs=2, thickness_nm=None, tilts=None,
              start=None, regularization=0.02, free_boundary=False, sampling=0.08, slices_per_cell=None,
              absorption=None, B=None, device="cpu", max_nfev=12, log=None, progress=None, stop=None,
              fit_background=True):
    """Joint slab fit of a group of reduced patterns (fitting.Reduced) of one material.

    thickness_nm: common thickness; tilts: list of (tx, ty) per pattern; start: (u, w) start for all slabs."""
    lib = model.libs[material] if material in model.libs else model.libs[model.names[0]]
    meta = lib.meta
    mat = cr.Material(**meta["material"])
    eng = ms.Engine(mat, meta["zone"], meta["energy_eV"], sampling, slices_per_cell or meta.get("slices_per_cell"),
                    B if B is not None else meta.get("debye_waller_B"),
                    absorption if absorption is not None else meta.get("absorption", 0.0), device)
    radius = float(lib.k[-1])
    prop = SlabPropagator(eng, lib.k, model.hk, radius)
    n_total = int(round(thickness_nm * 10 / eng.Lz))
    tilts = tilts or [(0.0, 0.0)] * len(patterns)
    u0, w0 = start if start is not None else (0.0, 0.0)
    N = int(n_slabs)
    p0 = np.tile([u0, w0], N).astype(float)
    if free_boundary and N == 2:
        p0 = np.append(p0, 0.5)
    y_a = [fm._anscombe(r.y) for r in patterns]
    t0 = time.time()

    def unpack(p):
        par = p[:2 * N].reshape(N, 2)
        fr = [p[2 * N], 1 - p[2 * N]] if (free_boundary and N == 2) else [1.0] * N
        return par, fr

    def residual(p):
        par, fr = unpack(p)
        segs, _ = slab_segments(n_total, par, fr)
        C = prop.curves(segs)
        res = []
        for red, ya, (tx, ty) in zip(patterns, y_a, tilts):
            I = ms.sample_curves(C, lib.k, red.geo.gi, red.geo.kx - tx, red.geo.ky - ty) * red.geo.ap
            _, _, M = fm.linear_scale(I, red.B, red.y, fit_background)
            res.append(ya - fm._anscombe(M))
        mean = par.mean(0)
        reg = np.sqrt(regularization) * (par - mean).ravel() / 1.0
        if progress:
            progress(min(0.99, prop.n_runs / (max_nfev * (len(p) + 1))), f"depth fit: {prop.n_runs} multislice runs, "
                     f"{time.time() - t0:.0f} s")
        if stop and stop():
            raise InterruptedError("stopped")
        return np.concatenate(res + [reg])

    lo = np.full(len(p0), -60.0)
    hi = np.full(len(p0), 60.0)
    if free_boundary and N == 2:
        lo[-1], hi[-1] = 0.1, 0.9

    # stage 1: one slab (common displacement) -> reference cost and start point
    def residual1(q):
        p = np.tile(q, N)
        if free_boundary and N == 2:
            p = np.append(p, 0.5)
        return residual(p)
    s1 = least_squares(residual1, p0[:2], bounds=(lo[:2], hi[:2]), diff_step=np.full(2, 0.02), x_scale=5.0,
                       max_nfev=max_nfev, method="trf")
    cost_single = float(2 * s1.cost)
    p0[:2 * N] = np.tile(s1.x, N)
    sol = least_squares(residual, p0, bounds=(lo, hi), diff_step=np.full(len(p0), 0.02), x_scale=5.0,
                        max_nfev=max_nfev, method="trf")
    J = sol.jac
    npix = sum(len(r.y) for r in patterns)
    chi = 2 * sol.cost / max(npix - len(p0), 1)
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    corr = cov / np.outer(d, d)
    par, fr = unpack(sol.x)
    _, edges = slab_segments(n_total, par, fr)
    P = cr.polarization(mat, meta["zone"], par[:, 0], par[:, 1])
    res = DepthResult(N, par, sig[:2 * N].reshape(N, 2), edges.tolist(), corr, float(2 * sol.cost), cost_single,
                      npix, P, dict(thickness_nm=thickness_nm, n_cells=n_total, multislice_runs=prop.n_runs,
                                    seconds=time.time() - t0, regularization=regularization,
                                    boundary_fraction=float(fr[0]) if free_boundary and N == 2 else None))
    if log:
        log(f"depth fit ({N} slabs, {len(patterns)} patterns): {prop.n_runs} multislice runs, "
            f"{time.time() - t0:.0f} s; slab displacements (pm): "
            + "; ".join(f"({a:.1f}+-{sa:.1f}, {b:.1f}+-{sb:.1f})" for (a, b), (sa, sb) in zip(par, res.sigma)))
    return res
