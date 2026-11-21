# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Calibration of the diffraction patterns (tab 2): disc radius and edge width, positions of the discs, the lattice
of the discs, polynomial distortions (e.g. of an energy filter), the background between the discs, the position of the
undiffracted beam, and the assignment of the disc lattice to the crystal zone.

Discs are located by correlating the gradient magnitude of sqrt(I) (edges of the discs, independent of the strongly
varying intensity inside the discs) with a ring template.  The lattice of each pattern is

    p(h, k) = c + G (h, k)^T       (undistorted detector coordinates, pixels)

and the measured positions are q = p + D(p - x0) with a polynomial distortion D of orders 2..N (no constant and linear
terms: those are indistinguishable from the beam position and the lattice; the distortion therefore vanishes at the
centre x0 of the pattern and grows towards the edges).  D is common to all patterns; c and G are fitted per pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# ====================================================================== templates / correlation ==


def edge_image(frame, smooth=1.0, normalize=8.0):
    """Gradient magnitude of sqrt(frame) (edges of the discs), divided by its local mean (scale ``normalize``
    pixels) so that weak and strong discs give comparable edges."""
    a = np.sqrt(np.clip(np.asarray(frame, np.float32), 0, None))
    if smooth:
        a = ndimage.gaussian_filter(a, smooth)
    gy = ndimage.sobel(a, 0)
    gx = ndimage.sobel(a, 1)
    e = np.hypot(gx, gy)
    if normalize:
        loc = ndimage.gaussian_filter(e, normalize)
        e = e / (loc + 0.05 * e.mean() + 1e-12)
    return e


def ring_template(shape, r, width=1.0):
    """Ring of radius r (Gaussian profile of the given width) centred at pixel (0, 0) (FFT order)."""
    ky, kx = shape
    y = np.fft.fftfreq(ky) * ky
    x = np.fft.fftfreq(kx) * kx
    R = np.hypot(y[:, None], x[None, :])
    t = np.exp(-0.5 * ((R - r) / max(width, 0.5)) ** 2)
    t -= t.mean()
    return t / np.sqrt((t ** 2).sum())


def ring_correlation(frame, r, width=1.0, edge=None):
    """Correlation of the edge image with a ring of radius r (peaks at disc centres)."""
    e = edge_image(frame) if edge is None else edge
    T = np.fft.rfft2(ring_template(e.shape, r, width))
    return np.fft.irfft2(np.fft.rfft2(e - e.mean()) * np.conj(T), s=e.shape)


def local_maxima(cmap, min_distance, threshold_rel=0.2, n_max=2000):
    from skimage.feature import peak_local_max
    pk = peak_local_max(cmap, min_distance=max(1, int(min_distance)), threshold_rel=threshold_rel,
                        num_peaks=n_max, exclude_border=False)
    return pk.astype(float)


def subpixel(cmap, pts):
    """Parabolic sub-pixel refinement of maxima; pts (n, 2) integer (row, col).  Returns (n, 2) and peak values."""
    pts = np.round(np.asarray(pts)).astype(int)
    ny, nx = cmap.shape
    y = np.clip(pts[:, 0], 1, ny - 2)
    x = np.clip(pts[:, 1], 1, nx - 2)
    c0 = cmap[y, x]
    dy = 0.5 * (cmap[y - 1, x] - cmap[y + 1, x]) / np.where(
        (d := cmap[y - 1, x] - 2 * c0 + cmap[y + 1, x]) < 0, d, -1e-12)
    dx = 0.5 * (cmap[y, x - 1] - cmap[y, x + 1]) / np.where(
        (e := cmap[y, x - 1] - 2 * c0 + cmap[y, x + 1]) < 0, e, -1e-12)
    return np.stack([y + np.clip(dy, -1, 1), x + np.clip(dx, -1, 1)], 1), c0


def sample_max_near(cmap, pts, radius=2):
    """Integer position of the maximum of cmap within +-radius of each predicted point (n, 2)."""
    ny, nx = cmap.shape
    out = []
    for py, px in np.asarray(pts):
        y0, y1 = int(max(0, round(py) - radius)), int(min(ny, round(py) + radius + 1))
        x0, x1 = int(max(0, round(px) - radius)), int(min(nx, round(px) + radius + 1))
        if y1 <= y0 or x1 <= x0:
            out.append((np.nan, np.nan))
            continue
        w = cmap[y0:y1, x0:x1]
        iy, ix = np.unravel_index(np.argmax(w), w.shape)
        out.append((y0 + iy, x0 + ix))
    return np.array(out, float)


# ====================================================================== radius / edge ==


def estimate_radius(frame, r_min=3, r_max=None, n_peaks=20):
    """Disc radius (pixels) maximizing the summed ring-correlation peaks of the strongest discs."""
    e = edge_image(frame)
    r_max = r_max or min(frame.shape) / 6

    def score(r):
        c = ring_correlation(frame, r, 1.0, e)
        return np.sort(c.ravel())[-n_peaks * 4:].mean()
    best = max((score(r), r) for r in np.arange(r_min, r_max, 0.5))
    r0 = best[1]
    fine = max((score(r), r) for r in np.arange(r0 - 0.5, r0 + 0.51, 0.1))
    return float(fine[1])


def edge_profile(frame, centres, r_guess, r_out=None, n_bins=60):
    """Mean normalized radial profile around disc centres and a fit of the edge: (radius, edge width (sigma), r, prof).

    Each disc is normalized by its mean intensity inside 0.7 r, so that weak and strong discs contribute equally."""
    from scipy.optimize import curve_fit
    from scipy.special import erfc
    r_out = r_out or 1.6 * r_guess
    ny, nx = frame.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    edges = np.linspace(0, r_out, n_bins + 1)
    for cy, cx in centres:
        y0, y1 = int(max(0, cy - r_out - 1)), int(min(ny, cy + r_out + 2))
        x0, x1 = int(max(0, cx - r_out - 1)), int(min(nx, cx + r_out + 2))
        sub = frame[y0:y1, x0:x1]
        R = np.hypot(Y[y0:y1, x0:x1] - cy, X[y0:y1, x0:x1] - cx)
        inner = sub[R < 0.7 * r_guess]
        if inner.size == 0 or inner.mean() <= 0:
            continue
        idx = np.digitize(R.ravel(), edges) - 1
        ok = (idx >= 0) & (idx < n_bins)
        np.add.at(acc, idx[ok], sub.ravel()[ok] / inner.mean())
        np.add.at(cnt, idx[ok], 1)
    rr = 0.5 * (edges[1:] + edges[:-1])
    prof = acc / np.maximum(cnt, 1)

    def model(r, R0, w, a, b):
        return a * 0.5 * erfc((r - R0) / (np.sqrt(2) * w)) + b

    ok = cnt > 0
    try:
        p, _ = curve_fit(model, rr[ok], prof[ok], p0=(r_guess, 0.8, 1.0, 0.05), sigma=1 / np.sqrt(cnt[ok]),
                         maxfev=5000)
        return float(p[0]), float(abs(p[1])), rr, prof
    except Exception:  # noqa: BLE001
        return float(r_guess), 1.0, rr, prof


# ====================================================================== lattice ==


def reduce_basis(a, b):
    """Gauss (Lagrange) reduction of a 2D lattice basis."""
    a, b = np.array(a, float), np.array(b, float)
    for _ in range(50):
        if np.dot(a, a) > np.dot(b, b):
            a, b = b, a
        m = round(np.dot(a, b) / np.dot(a, a))
        if m == 0:
            break
        b = b - m * a
    if np.dot(a, a) > np.dot(b, b):
        a, b = b, a
    return a, b


def find_basis(points, centre, weights=None, min_len=None):
    """Lattice basis (2 shortest non-collinear vectors) from peak positions around a centre."""
    d = np.asarray(points, float) - np.asarray(centre, float)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > (min_len or 1e-6)
    d, L = d[keep], L[keep]
    if len(d) < 2:
        raise ValueError("not enough discs to determine the lattice")
    order = np.argsort(L)
    a = d[order[0]]
    for i in order[1:]:
        v = d[i]
        cross = abs(a[0] * v[1] - a[1] * v[0]) / (np.linalg.norm(a) * np.linalg.norm(v))
        if cross > 0.3:
            return reduce_basis(a, v)
    raise ValueError("all discs are collinear")


def index_points(points, centre, G):
    """Integer lattice indices (h, k) of points for the lattice p = c + G (h, k)."""
    hk = np.linalg.solve(G, (np.asarray(points, float) - centre).T).T
    return np.round(hk).astype(int), hk


def fit_lattice(points, hk, weights=None):
    """Least-squares c (2) and G (2x2, columns = basis vectors) of p = c + G hk."""
    hk = np.asarray(hk, float)
    A = np.hstack([np.ones((len(hk), 1)), hk])          # [1, h, k]
    w = np.ones(len(hk)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(A * sw, np.asarray(points, float) * sw, rcond=None)
    c = sol[0]
    G = sol[1:].T                                        # columns: d p / d h, d p / d k
    res = points - (c + hk @ G.T)
    return c, G, res


# ====================================================================== distortion ==


@dataclass
class Distortion:
    """q = u + D(u): polynomial in s = (u - x0) / scale with orders 2..order (no constant / linear terms)."""
    order: int = 0
    x0: tuple = (0.0, 0.0)
    scale: float = 100.0
    cy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cx: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def terms(self):
        return [(i, j) for n in range(2, self.order + 1) for i in range(n + 1) for j in [n - i]]

    def basis(self, u):
        s = (np.asarray(u, float) - np.asarray(self.x0)) / self.scale
        if self.order < 2:
            return np.zeros((len(s), 0))
        return np.stack([s[:, 0] ** i * s[:, 1] ** j for i, j in self.terms()], 1)

    def forward(self, u):
        u = np.asarray(u, float).reshape(-1, 2)
        if self.order < 2 or len(self.cy) == 0:
            return u.copy()
        B = self.basis(u)
        return u + np.stack([B @ self.cy, B @ self.cx], 1)

    def inverse(self, q, n_iter=30):
        q = np.asarray(q, float).reshape(-1, 2)
        if self.order < 2 or len(self.cy) == 0:
            return q.copy()
        u = q.copy()
        for _ in range(n_iter):
            u = q - (self.forward(u) - u)
        return u

    def magnitude(self, shape, n=9, nonlinear=True):
        """Largest displacement (pixels) on a grid over the detector; with ``nonlinear`` the best affine part (which
        cannot be separated from the beam position and the lattice) is removed first."""
        ys, xs = np.meshgrid(np.linspace(0, shape[0] - 1, n), np.linspace(0, shape[1] - 1, n), indexing="ij")
        u = np.stack([ys.ravel(), xs.ravel()], 1)
        d = self.forward(u) - u
        if nonlinear:
            A = np.hstack([np.ones((len(u), 1)), u])
            coef, *_ = np.linalg.lstsq(A, d, rcond=None)
            d = d - A @ coef
        return float(np.abs(d).max())

    def to_dict(self):
        return dict(order=self.order, x0=[float(v) for v in self.x0], scale=float(self.scale),
                    terms=self.terms(), coeff_y=[float(v) for v in self.cy], coeff_x=[float(v) for v in self.cx])

    @classmethod
    def from_dict(cls, d):
        return cls(int(d["order"]), tuple(d["x0"]), float(d["scale"]), np.array(d["coeff_y"]),
                   np.array(d["coeff_x"]))


def fit_distortion(measurements, order=3, x0=None, scale=None, n_iter=8, weights_key="w"):
    """Common polynomial distortion of several patterns.

    measurements: list of dict(q=(n, 2) measured disc positions, hk=(n, 2) indices, w=(n,) weights).
    Returns (Distortion, list of (c, G) per pattern, rms residual in pixels)."""
    qs = [np.asarray(m["q"], float) for m in measurements]
    if x0 is None:
        x0 = tuple(np.mean([np.average(q, axis=0) for q in qs], 0))
    if scale is None:
        allq = np.vstack(qs)
        scale = float(np.max(np.abs(allq - np.asarray(x0)))) or 1.0
    D = Distortion(order, tuple(x0), scale)
    nterm = len(D.terms())
    D.cy, D.cx = np.zeros(nterm), np.zeros(nterm)
    lat = []
    # robust (Huber) weights per disc, updated every iteration
    rw = [np.ones(len(q)) for q in qs]
    for it in range(n_iter):
        lat = []
        rows_B, rows_r, rows_w = [], [], []
        for k, (m, q) in enumerate(zip(measurements, qs)):
            u = D.inverse(q)
            w0 = np.ones(len(q)) if m.get(weights_key) is None else np.asarray(m.get(weights_key), float)
            w = w0 * rw[k]
            c, G, res = fit_lattice(u, m["hk"], w)
            lat.append((c, G))
            ideal = c + np.asarray(m["hk"], float) @ G.T
            r = np.hypot(*(D.forward(ideal) - q).T)
            sc = 1.4826 * np.median(r) + 1e-3
            rw[k] = np.minimum(1.0, 1.5 * sc / np.maximum(r, 1e-9))
            if order >= 2:
                rows_B.append(D.basis(ideal))
                rows_r.append(q - ideal)
                rows_w.append(w0 * rw[k])
        if order < 2:
            break
        B = np.vstack(rows_B)
        R = np.vstack(rows_r)
        W = np.sqrt(np.concatenate(rows_w))[:, None]
        coef, *_ = np.linalg.lstsq(B * W, R * W, rcond=None)
        D.cy, D.cx = coef[:, 0], coef[:, 1]
    res = []
    for m, q, (c, G) in zip(measurements, qs, lat):
        ideal = c + np.asarray(m["hk"], float) @ G.T
        res.append(D.forward(ideal) - q)
    rms = float(np.sqrt(np.mean(np.vstack(res) ** 2))) if res else 0.0
    return D, lat, rms


def undistorted_pixel_grid(shape, D):
    """Undistorted coordinates (ky, kx, 2) of all detector pixels (inverse of the distortion)."""
    ny, nx = shape
    Y, X = np.mgrid[0:ny, 0:nx]
    q = np.stack([Y.ravel(), X.ravel()], 1).astype(float)
    return D.inverse(q).reshape(ny, nx, 2)


# ====================================================================== background ==


def disc_mask(shape, centre, G, radius, margin, U=None, hk_max=60):
    """Pixels within radius + margin of any lattice point (in undistorted coordinates U if given)."""
    ny, nx = shape
    if U is None:
        Y, X = np.mgrid[0:ny, 0:nx]
        U = np.stack([Y, X], -1).astype(float)
    hk = np.linalg.solve(G, (U.reshape(-1, 2) - centre).T).T
    near = np.round(hk)
    d = (U.reshape(-1, 2) - centre) - near @ G.T
    r = np.hypot(d[:, 0], d[:, 1])
    return (r <= radius + margin).reshape(ny, nx)


def background_smooth(frame, mask_discs, sigma):
    """Normalized convolution: smooth estimate of the background from the pixels between the discs."""
    m = (~mask_discs).astype(np.float32)
    num = ndimage.gaussian_filter(frame * m, sigma, mode="nearest")
    den = ndimage.gaussian_filter(m, sigma, mode="nearest")
    return num / np.maximum(den, 1e-6)


def background_radial(frame, mask_discs, centre, n_harm=2, n_knots=12):
    """Background model B(r, phi) = spline(r) * (1 + sum_m a_m cos(m phi) + b_m sin(m phi)) fitted to the pixels
    between the discs (linear least squares in log-free form)."""
    ny, nx = frame.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    dy, dx = Y - centre[0], X - centre[1]
    R = np.hypot(dy, dx)
    P = np.arctan2(dy, dx)
    rmax = R.max()
    knots = np.linspace(0, rmax, n_knots)
    # piecewise-linear radial basis (hat functions) times harmonics
    hats = []
    for i, k in enumerate(knots):
        lo = knots[i - 1] if i > 0 else k - 1e-9
        hi = knots[i + 1] if i < len(knots) - 1 else k + 1e-9
        h = np.where(R <= k, (R - lo) / max(k - lo, 1e-9), (hi - R) / max(hi - k, 1e-9))
        hats.append(np.clip(h, 0, 1))
    cols = []
    for h in hats:
        cols.append(h)
        for m in range(1, n_harm + 1):
            cols.append(h * np.cos(m * P))
            cols.append(h * np.sin(m * P))
    A = np.stack([c.ravel() for c in cols], 1)
    sel = (~mask_discs).ravel()
    w = 1 / np.sqrt(np.maximum(frame.ravel()[sel], 1.0))
    coef, *_ = np.linalg.lstsq(A[sel] * w[:, None], frame.ravel()[sel] * w, rcond=None)
    return np.clip((A @ coef).reshape(ny, nx), 0, None)


def background(frame, mask_discs, method="smooth", sigma=6.0, centre=None):
    if method == "radial":
        return background_radial(frame, mask_discs, centre if centre is not None else
                                 (frame.shape[0] / 2, frame.shape[1] / 2))
    if method == "none":
        return np.zeros_like(frame)
    return background_smooth(frame, mask_discs, sigma)


def choose_background_sigma(frames, masks, sigmas=(2, 3, 4, 6, 8, 12, 16), holdout_px=2.0, dist_maps=None):
    """Cross-validated smoothing width: pixels just outside the discs (a ring of holdout_px) are left out and
    predicted from the others; the Poisson deviance of the prediction is minimized."""
    best = None
    scores = []
    for s in sigmas:
        tot = 0.0
        for f, m, dist in zip(frames, masks, dist_maps):
            hold = (~m) & (dist <= holdout_px)
            B = background_smooth(f, m | hold, s)
            y = f[hold]
            mu = np.maximum(B[hold], 1e-3)
            tot += float(np.sum(2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - (y - mu))))
        scores.append(tot)
        if best is None or tot < best[0]:
            best = (tot, s)
    return float(best[1]), list(zip(sigmas, scores))


def distance_outside_discs(shape, centre, G, radius, U=None):
    """Distance (pixels) of each pixel from the edge of the nearest disc (0 inside)."""
    ny, nx = shape
    if U is None:
        Y, X = np.mgrid[0:ny, 0:nx]
        U = np.stack([Y, X], -1).astype(float)
    hk = np.linalg.solve(G, (U.reshape(-1, 2) - centre).T).T
    best = np.full(len(hk), np.inf)
    base = np.floor(hk)
    for a in (0, 1):
        for b in (0, 1):
            n = base + np.array([a, b])
            d = (U.reshape(-1, 2) - centre) - n @ G.T
            best = np.minimum(best, np.hypot(d[:, 0], d[:, 1]))
    return np.clip(best - radius, 0, None).reshape(ny, nx)


# ====================================================================== per-pattern measurement ==


@dataclass
class Calibration:
    """Result of tab 2 (all positions in detector pixels of the data set as read, i.e. after detector binning)."""
    det_shape: tuple = (0, 0)
    radius_px: float = 0.0
    edge_px: float = 1.0
    distortion: Distortion = field(default_factory=Distortion)
    centre0: np.ndarray = field(default_factory=lambda: np.zeros(2))     # mean undistorted beam position
    G0: np.ndarray = field(default_factory=lambda: np.eye(2))           # mean lattice (columns: basis vectors)
    hk_max: int = 8
    g_max_px: float = 0.0                                                # discs used: |p - c| <= g_max_px
    bg_method: str = "smooth"
    bg_sigma: float = 6.0
    bg_margin: float = 1.5
    # crystal assignment: detector basis vector i = sum_j B_cry[j, i] * crystal reciprocal axis j (1/A)
    B_cry: np.ndarray = field(default_factory=lambda: np.eye(2))
    mrad_per_px: float = 0.0
    zone_assignment: dict = field(default_factory=dict)
    # maps over the scan (coarse -> interpolated, refined by the full pass)
    coarse_positions: list = field(default_factory=list)
    coarse_centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    coarse_G: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))
    centre_map: np.ndarray = None          # (sy, sx, 2)  undistorted beam position
    G_map: np.ndarray = None               # (sy, sx, 2, 2)
    quality_map: np.ndarray = None         # rms residual of the lattice fit (px)
    bg_fraction_map: np.ndarray = None     # fraction of the counts in the background
    counts_map: np.ndarray = None          # total counts
    centre_map_source: str = "none"
    U: np.ndarray = None                   # undistorted coordinates of all pixels

    def summary(self):
        d = self.distortion
        s = [f"disc radius {self.radius_px:.2f} px, edge width {self.edge_px:.2f} px"]
        if self.mrad_per_px:
            s.append(f"{self.mrad_per_px:.4f} mrad/px (alpha = {self.radius_px * self.mrad_per_px:.2f} mrad)")
        s.append(f"lattice |g1| = {np.linalg.norm(self.G0[:, 0]):.2f} px, |g2| = {np.linalg.norm(self.G0[:, 1]):.2f} px")
        if d.order >= 2:
            s.append(f"distortion order {d.order}: non-linear part up to {d.magnitude(self.det_shape):.2f} px")
        s.append(f"background: {self.bg_method}" + (f", sigma {self.bg_sigma:.1f} px" if self.bg_method == "smooth"
                                                    else ""))
        return "; ".join(s)

    def to_dict(self):
        return dict(det_shape=list(self.det_shape), radius_px=self.radius_px, edge_px=self.edge_px,
                    distortion=self.distortion.to_dict(), centre0=self.centre0.tolist(), G0=self.G0.tolist(),
                    g_max_px=self.g_max_px, bg_method=self.bg_method, bg_sigma=self.bg_sigma,
                    bg_margin=self.bg_margin, B_cry=np.asarray(self.B_cry).tolist(), mrad_per_px=self.mrad_per_px,
                    zone_assignment=self.zone_assignment, centre_map_source=self.centre_map_source)

    def ensure_U(self):
        if self.U is None or self.U.shape[:2] != tuple(self.det_shape):
            self.U = undistorted_pixel_grid(self.det_shape, self.distortion)
        return self.U

    # ---------------------------------------------------------------- per frame
    def lattice_points(self, c, G):
        """Indices and undistorted positions of the discs used (inside the detector, |p - c| <= g_max_px)."""
        n = self.hk_max
        h, k = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
        hk = np.stack([h.ravel(), k.ravel()], 1)
        p = c + hk @ G.T
        r = self.radius_px + 2
        inside = (p[:, 0] >= r) & (p[:, 1] >= r) & (p[:, 0] <= self.det_shape[0] - 1 - r) & \
                 (p[:, 1] <= self.det_shape[1] - 1 - r)
        if self.g_max_px:
            inside &= np.hypot(*(p - c).T) <= self.g_max_px
        return hk[inside], p[inside]

    def measure(self, frame, c_guess=None, G_guess=None, search=None):
        """Lattice (c, G) of one pattern in undistorted coordinates: disc centres from the ring correlation near the
        predicted positions, distortion removed, least squares.  Returns dict(c, G, rms, n, q, hk, w)."""
        c = np.asarray(self.centre0 if c_guess is None else c_guess, float)
        G = np.asarray(self.G0 if G_guess is None else G_guess, float)
        cmap = ring_correlation(frame, self.radius_px, max(self.edge_px, 0.7))
        search = search or max(2, int(0.25 * self.radius_px))
        out = None
        for it in range(2):
            hk, p = self.lattice_points(c, G)
            q_pred = self.distortion.forward(p)
            pk = sample_max_near(cmap, q_pred, search)
            ok = np.isfinite(pk).all(1)
            q, val = subpixel(cmap, pk[ok])
            hk_ok = hk[ok]
            w = np.clip(val, 0, None)
            if w.sum() <= 0 or len(q) < 3:
                return dict(c=c, G=G, rms=np.nan, n=0, q=q, hk=hk_ok, w=w)
            u = self.distortion.inverse(q)
            c, G, res = fit_lattice(u, hk_ok, w)
            # reject outliers (more than 3 robust sigma)
            r = np.hypot(res[:, 0], res[:, 1])
            s = 1.4826 * np.median(r) + 1e-3
            good = r < max(3 * s, 0.5)
            if good.sum() >= 3 and good.sum() < len(r):
                c, G, res = fit_lattice(u[good], hk_ok[good], w[good])
                q, hk_ok, w = q[good], hk_ok[good], w[good]
            rms = float(np.sqrt(np.average(np.sum(res ** 2, 1), weights=w)))
            out = dict(c=c, G=G, rms=rms, n=int(len(q)), q=q, hk=hk_ok, w=w)
            search = max(2, int(0.15 * self.radius_px))
        return out

    def disc_mask(self, c, G, margin=None):
        U = self.ensure_U()
        return disc_mask(self.det_shape, c, G, self.radius_px, self.bg_margin if margin is None else margin, U)

    def background(self, frame, c, G):
        m = self.disc_mask(c, G)
        return background(frame, m, self.bg_method, self.bg_sigma, centre=self.distortion.forward(c[None])[0])


def lattice_from_autocorrelation(frames, radius):
    """Lattice basis (columns, pixels) from the summed autocorrelation of the edge images (independent of beam
    shifts between the frames)."""
    acc = None
    for f in frames:
        e = edge_image(f)
        e = e - e.mean()
        F = np.fft.rfft2(e)
        a = np.fft.irfft2(F * np.conj(F), s=e.shape)
        acc = a if acc is None else acc + a
    acc = np.fft.fftshift(acc)
    ny, nx = acc.shape
    pk = local_maxima(acc, min_distance=1.2 * radius, threshold_rel=0.05, n_max=200)
    pts, _ = subpixel(acc, pk)
    a, b = find_basis(pts, (ny // 2, nx // 2), min_len=1.5 * radius)
    G = np.stack([a, b], 1)
    if np.linalg.det(G) < 0:
        G = np.stack([b, a], 1)
    return G


def comb_centre(cmap, G, start, radius, hk_max=8):
    """Lattice origin maximizing the summed ring correlation over all lattice points (search +-1/2 cell)."""
    ny, nx = cmap.shape
    h, k = np.meshgrid(np.arange(-hk_max, hk_max + 1), np.arange(-hk_max, hk_max + 1), indexing="ij")
    hk = np.stack([h.ravel(), k.ravel()], 1).astype(float)
    off = hk @ G.T
    best = (-np.inf, np.asarray(start, float))
    span = 0.5 * max(np.linalg.norm(G[:, 0]), np.linalg.norm(G[:, 1]))
    for dy in np.arange(-span, span + 0.5, 1.0):
        for dx in np.arange(-span, span + 0.5, 1.0):
            c = np.asarray(start, float) + (dy, dx)
            p = c + off
            ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] <= ny - 1) & (p[:, 1] <= nx - 1)
            if ok.sum() < 3:
                continue
            v = ndimage.map_coordinates(cmap, p[ok].T, order=1).sum() / np.sqrt(ok.sum())
            if v > best[0]:
                best = (v, c)
    return best[1]


def initial_calibration(frames, r_guess=None, centre_guess=None, g_max_px=None):
    """First calibration from a few patterns (e.g. the coarse sample): radius, lattice basis (summed
    autocorrelation), lattice origin of the first pattern, edge width."""
    frames = [np.asarray(f, np.float32) for f in (frames if np.ndim(frames) == 3 else [frames])]
    f0 = frames[len(frames) // 2]
    cal = Calibration(det_shape=tuple(f0.shape))
    r = r_guess or float(np.median([estimate_radius(f) for f in frames[:5]]))
    G = lattice_from_autocorrelation(frames, r)
    cmap = ring_correlation(f0, r, 1.0)
    if centre_guess is None:
        ny, nx = f0.shape
        sm = ndimage.gaussian_filter(f0, r / 2)
        Y, X = np.mgrid[0:ny, 0:nx]
        score = sm / (1 + (np.hypot(Y - ny / 2, X - nx / 2) / (0.25 * min(ny, nx))) ** 2)
        start = np.unravel_index(np.argmax(score), score.shape)
    else:
        start = centre_guess
    hk_max = int(np.ceil(max(f0.shape) / min(np.linalg.norm(G[:, 0]), np.linalg.norm(G[:, 1])))) + 1
    c = comb_centre(cmap, G, start, r, hk_max)
    cal.radius_px, cal.centre0, cal.G0, cal.hk_max = float(r), c, G, hk_max
    cal.g_max_px = float(g_max_px or 0.0)
    cal.edge_px = 1.0
    m = cal.measure(f0)
    if m and m["n"] >= 4:
        cal.centre0, cal.G0 = m["c"], m["G"]
    # the radius from the ring correlation is used; the radial edge profile of the specimen discs is biased by the
    # intensity distribution inside the discs -- use the vacuum reference (uniform disc) or the model-based
    # refinement (tab 3) for the final radius and edge width
    return cal


# ====================================================================== zone assignment ==


def zone_candidates(G_px, g_cry, tol=0.04, max_index=3):
    """Pairs of crystal reciprocal vectors (1/A, projection frame) matching the detector basis (columns of G_px) in
    length ratio and angle.  Returns a list of dict(B (2x2: crystal vectors as columns), M (px per 1/A),
    mrad_scale, score), best first.  g_cry: (n, 2) candidate crystal vectors (e.g. low-order reflections)."""
    a_px, b_px = G_px[:, 0], G_px[:, 1]
    ratio = np.linalg.norm(b_px) / np.linalg.norm(a_px)
    cosang = np.dot(a_px, b_px) / (np.linalg.norm(a_px) * np.linalg.norm(b_px))
    det_sign = np.sign(np.linalg.det(G_px))
    cands = []
    g = np.asarray(g_cry, float)
    g = g[np.hypot(*g.T) > 1e-9]
    for i in range(len(g)):
        for j in range(len(g)):
            if i == j:
                continue
            A, B = g[i], g[j]
            r = np.linalg.norm(B) / np.linalg.norm(A)
            ca = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
            if abs(r / ratio - 1) < tol and abs(ca - cosang) < 2 * tol:
                Bc = np.stack([A, B], 1)
                M = G_px @ np.linalg.inv(Bc)
                sv = np.linalg.svd(M, compute_uv=False)
                score = abs(r / ratio - 1) + abs(ca - cosang) + 0.002 * (np.linalg.norm(A) + np.linalg.norm(B))
                cands.append(dict(B=Bc, M=M, px_per_invA=float(np.sqrt(sv[0] * sv[1])),
                                  anisotropy=float(sv[0] / sv[1] - 1), mirror=bool(np.linalg.det(M) < 0),
                                  score=float(score), det_sign=float(det_sign)))
    cands.sort(key=lambda d: (d["score"], -d["px_per_invA"]))
    # keep the largest pixel scale (smallest indices) for the best metric
    return cands


def growth_direction_on_detector(M):
    """Unit vector (detector y, x) of the crystal growth direction [001] (projection y axis) on the detector."""
    v = M @ np.array([0.0, 1.0])
    return v / np.linalg.norm(v)


# ====================================================================== scan maps ==


def interpolate_scan(positions, values, scan_shape, method="thin_plate_spline", smoothing=0.0, degree=None):
    """Interpolate values (n, ...) known at coarse scan positions (n, 2) onto the full scan grid."""
    from scipy.interpolate import RBFInterpolator
    pos = np.asarray(positions, float)
    v = np.asarray(values, float)
    flat = v.reshape(len(v), -1)
    ok = np.isfinite(flat).all(1)
    sy, sx = scan_shape
    Y, X = np.mgrid[0:sy, 0:sx]
    grid = np.stack([Y.ravel(), X.ravel()], 1).astype(float)
    if degree is not None:
        out = np.stack([poly_fit_eval(pos[ok], flat[ok, i], grid, degree) for i in range(flat.shape[1])], 1)
    else:
        rbf = RBFInterpolator(pos[ok], flat[ok], kernel=method, smoothing=smoothing,
                              degree=1 if method == "thin_plate_spline" else None)
        out = rbf(grid)
    return out.reshape((sy, sx) + v.shape[1:])


def poly_fit_eval(pos, val, grid, degree):
    """Least-squares 2D polynomial of the given degree (normalized coordinates)."""
    s = np.maximum(np.abs(grid).max(0), 1)
    def terms(p):
        p = p / s
        return np.stack([p[:, 0] ** i * p[:, 1] ** j for n in range(degree + 1) for i in range(n + 1) for j in
                         [n - i]], 1)
    coef, *_ = np.linalg.lstsq(terms(pos), val, rcond=None)
    return terms(grid) @ coef


def coarse_calibration(frames, positions, cal, distortion_order=3, n_iter=2, log=None):
    """Measure the coarse patterns, fit the common distortion, re-measure; fills cal.coarse_* and cal.distortion."""
    meas = []
    for it in range(n_iter):
        meas = [cal.measure(f) for f in frames]
        good = [m for m in meas if m["n"] >= 6 and np.isfinite(m["rms"])]
        if len(good) < 1:
            raise ValueError("lattice could not be measured in the coarse patterns")
        D, lat, rms = fit_distortion(good, order=distortion_order)
        cal.distortion = D
        cal.U = None
        cs = np.array([m["c"] for m in good])
        Gs = np.array([m["G"] for m in good])
        cal.centre0 = np.median(cs, 0)
        cal.G0 = np.median(Gs, 0)
        if log:
            log(f"distortion fit {it + 1}: order {distortion_order}, rms {rms:.3f} px, max "
                f"{D.magnitude(cal.det_shape):.2f} px")
    meas = [cal.measure(f) for f in frames]
    cal.coarse_positions = [tuple(p) for p in positions]
    cal.coarse_centres = np.array([m["c"] for m in meas])
    cal.coarse_G = np.array([m["G"] for m in meas])
    return meas


def interpolated_maps(cal, scan_shape, smoothing=0.0):
    """Centre and lattice maps over the whole scan interpolated from the coarse measurements."""
    pos = np.array(cal.coarse_positions, float)
    ok = np.isfinite(cal.coarse_centres).all(1)
    if ok.sum() < 3:
        c = np.broadcast_to(cal.centre0, tuple(scan_shape) + (2,)).copy()
        G = np.broadcast_to(cal.G0, tuple(scan_shape) + (2, 2)).copy()
        return c, G
    c = interpolate_scan(pos[ok], cal.coarse_centres[ok], scan_shape, smoothing=smoothing)
    G = interpolate_scan(pos[ok], cal.coarse_G[ok].reshape(-1, 4), scan_shape, smoothing=smoothing)
    return c, G.reshape(tuple(scan_shape) + (2, 2))


def full_pass(ds, cal, progress=None, stop=None, block=4, log=None):
    """Measure the lattice (beam position, lattice vectors), the background fraction and the counts of every
    pattern (starting from the interpolated coarse maps).  Fills cal.centre_map, G_map, quality_map,
    bg_fraction_map, counts_map."""
    import time as _time
    sy, sx = ds.scan_shape
    c_int, G_int = interpolated_maps(cal, (sy, sx))
    C = np.full((sy, sx, 2), np.nan)
    Gm = np.full((sy, sx, 2, 2), np.nan)
    Q = np.full((sy, sx), np.nan)
    BF = np.full((sy, sx), np.nan)
    N = np.full((sy, sx), np.nan)
    t0 = _time.time()
    for y0, y1, blk in ds.iter_rows(block):
        for a in range(y1 - y0):
            iy = y0 + a
            for ix in range(sx):
                f = blk[a, ix]
                m = cal.measure(f, c_int[iy, ix], G_int[iy, ix])
                if m is None or m["n"] < 4 or not np.isfinite(m["rms"]):
                    C[iy, ix], Gm[iy, ix] = c_int[iy, ix], G_int[iy, ix]
                    continue
                C[iy, ix], Gm[iy, ix], Q[iy, ix] = m["c"], m["G"], m["rms"]
                B = cal.background(f, m["c"], m["G"])
                tot = float(f.sum())
                N[iy, ix] = tot
                BF[iy, ix] = float(B.sum()) / tot if tot > 0 else np.nan
        if progress:
            el = _time.time() - t0
            progress(y1 / sy, f"full pass: row {y1}/{sy}, about {el / y1 * (sy - y1):.0f} s left")
        if stop and stop():
            raise InterruptedError("stopped")
    cal.centre_map, cal.G_map, cal.quality_map, cal.bg_fraction_map, cal.counts_map = C, Gm, Q, BF, N
    cal.centre_map_source = "full pass"
    if log:
        log(f"full pass: {sy * sx} patterns in {_time.time() - t0:.0f} s, median lattice rms "
            f"{np.nanmedian(Q):.3f} px")
    return cal


def apply_interpolated(cal, scan_shape):
    """Use the interpolated coarse maps (before / instead of the full pass)."""
    cal.centre_map, cal.G_map = interpolated_maps(cal, scan_shape)
    cal.centre_map_source = "interpolated from the coarse sample"
    return cal


def refine_aperture(cal, model_factory, frames, lattices, params_list, fitter_factory, radii=None, edges=None):
    """Refine the disc radius and edge width by minimizing the summed fit cost of some patterns at fixed fit
    parameters (one dimension at a time on a small grid)."""
    def total(r, e):
        cal.radius_px, cal.edge_px = r, e
        mdl = model_factory()
        ft = fitter_factory(mdl)
        s = 0.0
        n = 0
        from .fitting import reduce_frame
        for f, (c, G), p in zip(frames, lattices, params_list):
            red = reduce_frame(f, c, G, mdl)
            s += ft.cost(red, p)
            n += len(red.y)
        return s / max(n, 1)
    r0, e0 = cal.radius_px, cal.edge_px
    radii = radii if radii is not None else r0 * np.linspace(0.92, 1.06, 8)
    rc = [total(r, e0) for r in radii]
    r_best = float(radii[int(np.argmin(rc))])
    edges = edges if edges is not None else np.clip(e0 * np.array([0.5, 0.75, 1.0, 1.4, 2.0]), 0.3, None)
    ec = [total(r_best, e) for e in edges]
    e_best = float(edges[int(np.argmin(ec))])
    cal.radius_px, cal.edge_px = r_best, e_best
    return r_best, e_best, list(zip(radii, rc)), list(zip(edges, ec))
