# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Plane-wave multislice for non-overlapping CBED and the rocking-curve library.

For non-overlapping discs every detector pixel inside disc g is reached by exactly one incident direction k (its
offset from the disc centre), so the CBED pattern is the incoherent set of rocking curves I_g(k) of plane waves.
The engine propagates a batch of plane waves with incident directions k (relative to the zone axis) through the
periodic zone cell; the incident tilt enters the Fresnel propagator (P = exp(-i pi lambda dz (|g|^2 + 2 g.k)),
equivalent to tilting the crystal).  The patterns are therefore independent of the probe position, focus and
aberrations (shown in the 4D-STEM notebook), and a crystal tilt tau shifts the rocking curves inside the fixed discs.

The projected potential of every slice is computed analytically in Fourier space from the 5-Gaussian electron
scattering factors of Peng (Acta Cryst. A52, 257 (1996); "high-angle" set as in abTEM): for a Gaussian term
a exp(-beta g^2) the slab integral between z0 and z1 is a exp(-beta g_perp^2) * [erf(pi (z1 - z_a)/sqrt(beta)) -
erf(pi (z0 - z_a)/sqrt(beta))] / 2.  This is exact for any (sub-pixel) atom position and any slicing, band limited,
and includes Debye-Waller smearing (beta += B/4) and optionally a phenomenological absorptive potential
V' = absorption * V.

The rocking-curve library stores I_g(k; n, u, w) for thicknesses n = 1..N unit cells (along the beam), incident
directions k on a square grid and Ti displacements (u, w) on a grid; fitting interpolates it (Catmull-Rom).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy.special import erf

from . import crystal as cr

# Peng et al. electron scattering factors f(s) = sum a_i exp(-b_i s^2), s = g/2 (0 < s < 6 A^-1)
PENG = {
    "O": ((0.0365, 0.1729, 0.5805, 0.8814, 0.3121), (0.0652, 0.6184, 2.9449, 9.6298, 28.2194)),
    "Ti": ((0.2123, 0.896, 2.1765, 3.0436, 2.4439), (0.1399, 1.4568, 6.7534, 33.1168, 101.8238)),
    "Sr": ((0.3723, 1.4598, 3.5124, 4.4612, 3.3031), (0.148, 1.4643, 9.232, 49.8807, 148.0937)),
    "Ba": ((0.5229, 2.2874, 4.7243, 5.0807, 5.6389), (0.1434, 1.6019, 9.4511, 42.7685, 148.4969)),
    "Pb": ((0.354, 1.5453, 3.5975, 4.3152, 2.7743), (0.0668, 0.6465, 3.6968, 16.2056, 61.4909)),
    "Zr": ((0.2997, 1.1879, 3.1075, 3.974, 3.5769), (0.1121, 1.0638, 6.3891, 28.7081, 97.4289)),
    "Ca": ((0.2355, 0.9916, 2.3959, 3.7252, 2.5647), (0.1742, 1.8329, 8.8407, 47.4583, 134.9613)),
    "La": ((0.5461, 2.3856, 5.0653, 5.7601, 4.0463), (0.1479, 1.6552, 10.0059, 47.3245, 145.8464)),
}
# Debye-Waller B (A^2) at room temperature (typical values; adjustable)
B_DEFAULT = {"Ba": 0.35, "Sr": 0.45, "Ti": 0.40, "O": 0.60, "Pb": 0.8, "Zr": 0.4, "Ca": 0.45, "La": 0.4}
KAPPA_V = 47.87801   # h^2 / (2 pi m0 e) in V A^2


def _fft():
    import scipy.fft as sf
    return sf


class Engine:
    """Plane-wave multislice of one perovskite in a zone-axis orientation.

    Parameters
    ----------
    material : crystal.Material
    zone : "[110]" or "[100]"
    energy_eV : acceleration voltage
    sampling : real-space sampling of the cell (Angstrom); the antialiasing aperture is at 2/3 of Nyquist
    slices_per_cell : slices per zone cell along the beam (default: about 0.7 Angstrom per slice)
    B : Debye-Waller factors per element (A^2), None = B_DEFAULT
    absorption : phenomenological absorptive potential as a fraction of the elastic one
    device : "cpu" or "gpu" (CuPy)
    """

    def __init__(self, material, zone="[110]", energy_eV=60e3, sampling=0.08, slices_per_cell=None, B=None,
                 absorption=0.0, device="cpu"):
        self.mat, self.zone, self.E = material, zone, float(energy_eV)
        self.lam = cr.wavelength_A(self.E)
        self.sigma = cr.interaction_sigma(self.E)
        self.Lx, self.Ly, self.Lz = cr.cell_lengths(material, zone)
        self.nx = int(2 * math.ceil(self.Lx / sampling / 2))
        self.ny = int(2 * math.ceil(self.Ly / sampling / 2))
        self.ns = int(slices_per_cell or max(2, round(self.Lz / 0.7)))
        self.dz = self.Lz / self.ns
        self.B = dict(B_DEFAULT if B is None else B)
        self.absorption = float(absorption)
        self.device = device
        self.xp = np
        if device == "gpu":
            import cupy
            self.xp = cupy
        fx = np.fft.fftfreq(self.nx, self.Lx / self.nx)
        fy = np.fft.fftfreq(self.ny, self.Ly / self.ny)
        self.gx, self.gy = np.meshgrid(fx, fy, indexing="ij")
        self.g2 = self.gx ** 2 + self.gy ** 2
        kmax = min(np.abs(fx).max(), np.abs(fy).max()) * 2 / 3
        self.aperture = (np.sqrt(self.g2) <= kmax)
        self.g_limit = kmax
        self._tcache = {}

    # ------------------------------------------------------------------ potential
    def describe(self):
        return dict(material=self.mat.to_dict(), zone=self.zone, energy_eV=self.E, wavelength_A=self.lam,
                    cell_A=[self.Lx, self.Ly, self.Lz], gpts=[self.nx, self.ny], slices_per_cell=self.ns,
                    debye_waller_B=self.B, absorption=self.absorption, g_limit_inv_A=self.g_limit,
                    scattering_factors="Peng et al. 1996 (5 Gaussians, 0<s<6)")

    def atoms(self, u_pm=0.0, w_pm=0.0):
        return cr.zone_cell(self.mat, self.zone, u_pm, w_pm)

    def slice_potentials(self, u_pm=0.0, w_pm=0.0):
        """Projected potentials of the slices (ns, nx, ny) in V*A (complex if absorption > 0)."""
        at = self.atoms(u_pm, w_pm)
        pos = at.get_positions()
        sym = at.get_chemical_symbols()
        area = self.Lx * self.Ly
        V = np.zeros((self.ns, self.nx, self.ny), complex)
        edges = np.arange(self.ns + 1) * self.dz
        for el in sorted(set(sym)):
            a, b = (np.array(v) for v in PENG[el])
            beta = b / 4.0 + self.B.get(el, 0.0) / 4.0          # exp(-b s^2) = exp(-(b/4) g^2); DW exp(-B g^2/4)
            idx = [i for i, s in enumerate(sym) if s == el]
            for i in idx:
                x, y, z = pos[i]
                phase = np.exp(-2j * np.pi * (self.gx * x + self.gy * y))
                for k in range(len(a)):
                    sb = math.sqrt(beta[k])
                    fz = np.zeros(self.ns)
                    for m in (-1, 0, 1):                       # periodic images along the beam
                        zz = z + m * self.Lz
                        fz += 0.5 * (erf(np.pi * (edges[1:] - zz) / sb) - erf(np.pi * (edges[:-1] - zz) / sb))
                    G = a[k] * np.exp(-beta[k] * self.g2) * phase
                    V += fz[:, None, None] * G[None]
        V *= KAPPA_V
        v = np.fft.ifft2(V, axes=(1, 2)) * (self.nx * self.ny) / area
        v = v.real
        if self.absorption:
            return v * (1 + 1j * self.absorption)
        return v

    def transmission(self, u_pm=0.0, w_pm=0.0):
        key = (round(float(u_pm), 6), round(float(w_pm), 6))
        t = self._tcache.get(key)
        if t is None:
            v = self.slice_potentials(u_pm, w_pm)
            t = np.exp(1j * self.sigma * v)
            # band limit the transmission function (2/3 rule: wave and transmission function both band limited)
            t = np.fft.ifft2(np.fft.fft2(t, axes=(1, 2)) * self.aperture[None], axes=(1, 2)).astype(np.complex64)
            if len(self._tcache) > 64:
                self._tcache.clear()
            t = self.xp.asarray(t)
            self._tcache[key] = t
        return t

    # ------------------------------------------------------------------ propagation
    def _propagators(self, directions_mrad):
        xp = self.xp
        k = np.asarray(directions_mrad, float) * 1e-3 / self.lam          # 1/A
        ph = -np.pi * self.lam * self.dz * (self.g2[None] + 2 * (k[:, 0, None, None] * self.gx[None]
                                                                 + k[:, 1, None, None] * self.gy[None]))
        P = np.exp(1j * ph) * self.aperture[None]
        return xp.asarray(P.astype(np.complex64))

    def reflection_index(self, hk):
        hk = np.asarray(hk, int)
        return (hk[:, 0] % self.nx, hk[:, 1] % self.ny)

    def run(self, segments, directions_mrad, hk, record=None, batch=None, progress=None, stop=None):
        """Rocking curves of a stack of segments.

        segments : list of (n_cells, u_pm, w_pm) from the entrance surface down
        directions_mrad : (n_dir, 2) incident directions relative to the zone axis (mrad, projection axes x, y)
        hk : (n_refl, 2) reflection indices of the zone cell
        record : cell numbers (1 = after the first cell) at which to record; default: only the exit surface
        Returns I[len(record), n_dir, n_refl] (fraction of the incident intensity)."""
        xp = self.xp
        sf = None
        if xp is np:
            sf = _fft()
        n_total = int(sum(s[0] for s in segments))
        record = [n_total] if record is None else sorted(int(r) for r in record)
        rset = set(record)
        dirs = np.asarray(directions_mrad, float).reshape(-1, 2)
        ix, iy = self.reflection_index(hk)
        out = np.zeros((len(record), len(dirs), len(ix)), np.float32)
        batch = batch or len(dirs)
        layer_struct = []
        for n, u, w in segments:
            layer_struct += [(u, w)] * int(n)
        done = 0
        for b0 in range(0, len(dirs), batch):
            d = dirs[b0:b0 + batch]
            P = self._propagators(d)
            psi = xp.ones((len(d), self.nx, self.ny), np.complex64)
            ri = 0
            for c, (u, w) in enumerate(layer_struct):
                T = self.transmission(u, w)
                for s in range(self.ns):
                    psi = psi * T[s][None]
                    if sf is not None:
                        psi = sf.ifft2(sf.fft2(psi, axes=(1, 2), workers=-1) * P, axes=(1, 2), workers=-1)
                    else:
                        psi = xp.fft.ifft2(xp.fft.fft2(psi, axes=(1, 2)) * P, axes=(1, 2))
                if (c + 1) in rset:
                    F = (sf.fft2(psi, axes=(1, 2), workers=-1) if sf is not None else xp.fft.fft2(psi, axes=(1, 2)))
                    amp = F[:, ix, iy] / (self.nx * self.ny)
                    I = (amp.real ** 2 + amp.imag ** 2)
                    out[ri, b0:b0 + len(d)] = I.get() if hasattr(I, "get") else np.asarray(I)
                    ri += 1
                done += 1
                if progress is not None and (c % 4 == 0):
                    progress((b0 + len(d) * (c + 1) / len(layer_struct)) / len(dirs))
                if stop is not None and stop():
                    raise InterruptedError("stopped")
        return out


# ====================================================================== library ==
def direction_grid(radius_mrad, step_mrad):
    """Square grid of incident directions covering a disc of the given radius: (kx axis, ky axis, mask)."""
    n = int(math.ceil(radius_mrad / step_mrad)) + 1
    ax = np.arange(-n, n + 1) * step_mrad
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    mask = KX ** 2 + KY ** 2 <= (radius_mrad + 1.5 * step_mrad) ** 2
    return ax, mask


class Library:
    """Rocking-curve library I[u, w, n, g, kx, ky] (float16 on disk / in memory) with Catmull-Rom interpolation in
    (u, w, n) and bilinear interpolation in the incident direction."""

    def __init__(self, data, u_pm, w_pm, cells, hk, g_inv_A, k_axis, meta):
        self.data = data                    # [nu, nw, nn, ng, nk, nk]
        self.u = np.asarray(u_pm, float)
        self.w = np.asarray(w_pm, float)
        self.cells = np.asarray(cells, int)
        self.hk = np.asarray(hk, int)
        self.g = np.asarray(g_inv_A, float)
        self.k = np.asarray(k_axis, float)
        self.meta = dict(meta)
        self.Lz = float(self.meta["cell_A"][2])

    # ------------------------------------------------------------------ io
    def save(self, path):
        """Write to ``path`` (via ``path.part``, renamed when complete: an interrupted write, e.g. a full disk,
        leaves no broken library) or to a writable binary file object."""
        import h5py
        if hasattr(path, "write"):
            with h5py.File(path, "w") as f:
                self._write(f)
            return path
        tmp = str(path) + ".part"
        try:
            with h5py.File(tmp, "w") as f:
                self._write(f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path

    def _write(self, f):
        f.create_dataset("rocking_curves", data=self.data, compression="gzip", compression_opts=1,
                         chunks=(1, 1, 1) + self.data.shape[3:])
        f["u_pm"], f["w_pm"], f["cells"] = self.u, self.w, self.cells
        f["hk"], f["g_inv_A"], f["k_mrad"] = self.hk, self.g, self.k
        f.attrs["meta"] = json.dumps(self.meta)

    def to_bytes(self):
        """The library file (HDF5) as bytes, without writing to disk."""
        import io
        buf = io.BytesIO()
        self.save(buf)
        return buf.getvalue()

    @property
    def file_size_estimate(self):
        """Upper estimate of the file size in bytes (the data compress little)."""
        return int(self.data.nbytes * 1.02) + (1 << 20)

    @classmethod
    def load(cls, path):
        """From a file path or a readable, seekable binary file object (e.g. bytes in a BytesIO, or a file of a
        NOMAD upload read on demand by :class:`fit4dstem.nomad.RemoteFile`)."""
        import h5py
        with h5py.File(path, "r") as f:
            return cls(f["rocking_curves"][()], f["u_pm"][()], f["w_pm"][()], f["cells"][()], f["hk"][()],
                       f["g_inv_A"][()], f["k_mrad"][()], json.loads(f.attrs["meta"]))

    @property
    def thickness_nm(self):
        return self.cells * self.Lz / 10

    @property
    def material(self):
        return self.meta.get("material", {}).get("name", "?")

    # ------------------------------------------------------------------ interpolation
    @staticmethod
    def _cr_weights(x, axis):
        """Catmull-Rom indices (4) and weights for x on a (possibly non-uniform) axis (clamped)."""
        n = len(axis)
        x = float(np.clip(x, axis[0], axis[-1]))
        i = int(np.clip(np.searchsorted(axis, x) - 1, 0, n - 2))
        t = (x - axis[i]) / (axis[i + 1] - axis[i])
        if n < 4:
            return np.array([i, i + 1]), np.array([1 - t, t])
        t2, t3 = t * t, t * t * t
        w = np.array([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t,
                      0.5 * t3 - 0.5 * t2])
        idx = np.array([i - 1, i, i + 1, i + 2])
        # clamp at the ends (duplicate end points)
        idx = np.clip(idx, 0, n - 1)
        return idx, w

    def curves(self, n_cells, u_pm, w_pm):
        """Rocking curves [ng, nk, nk] (float32) at a continuous thickness (cells) and displacement."""
        iu, wu = self._cr_weights(u_pm, self.u)
        iw, ww = self._cr_weights(w_pm, self.w)
        inn, wn = self._cr_weights(n_cells, self.cells.astype(float))
        out = None
        for a, wa in zip(iu, wu):
            if wa == 0:
                continue
            for b, wb in zip(iw, ww):
                if wb == 0:
                    continue
                blk = self.data[a, b][inn].astype(np.float32)          # [4, ng, nk, nk]
                s = np.tensordot(wn.astype(np.float32), blk, axes=(0, 0)) * np.float32(wa * wb)
                out = s if out is None else out + s
        return np.maximum(out, 0.0)

    def sample(self, curves, gi, kx, ky):
        """Bilinear sampling of curves[gi] at incident directions (kx, ky) mrad (arrays of equal length)."""
        return sample_curves(curves, self.k, gi, kx, ky)


def sample_curves(curves, k_axis, gi, kx, ky):
    """Bilinear sampling of rocking curves [ng, nk, nk] on the square direction grid k_axis (mrad)."""
    k0, dk = k_axis[0], k_axis[1] - k_axis[0]
    fx = (kx - k0) / dk
    fy = (ky - k0) / dk
    n = len(k_axis)
    x0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, n - 2)
    tx = np.clip(fx - x0, 0, 1)
    ty = np.clip(fy - y0, 0, 1)
    c = curves
    return ((1 - tx) * (1 - ty) * c[gi, x0, y0] + tx * (1 - ty) * c[gi, x0 + 1, y0]
            + (1 - tx) * ty * c[gi, x0, y0 + 1] + tx * ty * c[gi, x0 + 1, y0 + 1])


def build_library(material, zone="[110]", energy_eV=60e3, radius_mrad=8.0, step_mrad=0.5, g_max_inv_A=1.3,
                  max_thickness_nm=50.0, u_pm=(-40, -25, -12, 0, 12, 25, 40), w_pm=(-40, -25, -12, 0, 12, 25, 40),
                  sampling=0.08, slices_per_cell=None, B=None, absorption=0.03, device="cpu", batch=None,
                  progress=None, stop=None, path=None, log=print):
    """Compute a rocking-curve library (see :class:`Library`).  Returns the Library (and saves it to ``path``).

    radius_mrad must cover the convergence semi-angle + the largest tilt to be fitted (+ the disc edge)."""
    eng = Engine(material, zone, energy_eV, sampling, slices_per_cell, B, absorption, device)
    hk, gv = cr.reflections(material, zone, g_max_inv_A)
    if g_max_inv_A > eng.g_limit:
        raise ValueError(f"g_max {g_max_inv_A} exceeds the band limit {eng.g_limit:.2f} 1/A of the sampling; "
                         f"use a finer sampling")
    k_axis, mask = direction_grid(radius_mrad, step_mrad)
    KX, KY = np.meshgrid(k_axis, k_axis, indexing="ij")
    dirs = np.stack([KX[mask], KY[mask]], 1)
    n_max = int(math.ceil(max_thickness_nm * 10 / eng.Lz))
    cells = np.arange(1, n_max + 1)
    nk = len(k_axis)
    data = np.zeros((len(u_pm), len(w_pm), n_max, len(hk), nk, nk), np.float16)
    n_struct = len(u_pm) * len(w_pm)
    t0 = time.time()
    k = 0
    for a, u in enumerate(u_pm):
        for b, w in enumerate(w_pm):
            sub = None if progress is None else (lambda f, k=k: progress((k + f) / n_struct))
            I = eng.run([(n_max, u, w)], dirs, hk, record=cells, batch=batch, progress=sub, stop=stop)
            full = np.zeros((n_max, len(hk), nk, nk), np.float32)
            full[:, :, mask] = np.transpose(I, (0, 2, 1))
            data[a, b] = full.astype(np.float16)
            k += 1
            if log:
                el = time.time() - t0
                log(f"library {material.name} {zone}: structure {k}/{n_struct} (u={u}, w={w} pm), "
                    f"{el:.0f} s, about {el / k * (n_struct - k):.0f} s left")
    meta = eng.describe()
    meta.update(radius_mrad=radius_mrad, step_mrad=step_mrad, g_max_inv_A=g_max_inv_A,
                max_thickness_nm=max_thickness_nm, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    lib = Library(data, u_pm, w_pm, cells, hk, gv, k_axis, meta)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        lib.save(path)
    return lib
