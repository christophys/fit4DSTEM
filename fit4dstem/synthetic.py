# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Synthetic 4D-STEM data of a BaTiO3/SrTiO3 superlattice in cross-section (for tests and the demo).

Ground truth: layers along the scan y axis (growth direction), vortex-like Ti displacement field in the BaTiO3 layers
(divergence-free flux closure, period and amplitude similar to Hevelke et al. 2026), smaller induced displacement in
SrTiO3, a thickness wedge along x, bending (tilt), descan of the beam (linear + quadratic in the scan position), a
deflection of the beam by the projected field at the interfaces, polynomial (energy-filter-like) distortions, a
smooth incoherent background and Poisson noise.  A vacuum reference scan (smaller detector frames, same sampling)
records only the descan.
"""
from __future__ import annotations

import numpy as np

from . import calibration as calm
from . import fitting as fm


def truth_fields(scan_shape, step_nm=0.25, period_bto_uc=12, period_sto_uc=5, a_A=3.99, amp_pm=20.0,
                 vortex_period_nm=7.0, t_range_nm=(24.0, 34.0), tilt_amp_mrad=0.8, sto_amp=0.35, phase=0.0):
    """Ground-truth maps (sy, sx): material (0 BTO, 1 STO), thickness (nm), tilt x/y (mrad), u, w (pm)."""
    sy, sx = scan_shape
    y = (np.arange(sy) + 0.5) * step_nm
    x = (np.arange(sx) + 0.5) * step_nm
    Y, X = np.meshgrid(y, x, indexing="ij")
    hb, hs = period_bto_uc * a_A / 10, period_sto_uc * a_A / 10
    per = hb + hs
    zl = np.mod(Y + phase, per)
    sto = (zl >= hb).astype(int)
    # stream function psi = A sin(k x) sin(pi z / h) inside BTO; u = d psi / dz, w = -d psi / dx
    k = 2 * np.pi / vortex_period_nm
    zz = np.clip(zl, 0, hb)
    u = amp_pm * np.sin(k * X) * np.cos(np.pi * zz / hb)
    w = -amp_pm * (hb / np.pi) * k * np.cos(k * X) * np.sin(np.pi * zz / hb)
    w *= 1.0 / max(1.0, (hb / np.pi) * k)
    # SrTiO3 follows the adjacent BaTiO3 (weaker)
    u = np.where(sto == 1, sto_amp * amp_pm * np.sin(k * X) * np.cos(np.pi * (zl - hb) / hs + np.pi), u)
    w = np.where(sto == 1, sto_amp * w.max() * np.cos(k * X) * 0.5, w)
    t = t_range_nm[0] + (t_range_nm[1] - t_range_nm[0]) * X / X.max()
    tx = tilt_amp_mrad * (X / X.max() - 0.5) * 2
    ty = 0.5 * tilt_amp_mrad * np.sin(2 * np.pi * Y / (Y.max() * 2))
    return dict(material=sto, thickness_nm=t, tilt_x_mrad=tx, tilt_y_mrad=ty, u_pm=u, w_pm=w)


def make_dataset(libs, scan_shape=(24, 32), det_shape=(128, 128), mrad_per_px=0.6, alpha_mrad=4.0,
                 edge_px=0.8, rotation_deg=12.0, mirror=False, distortion=(0.004, -0.003, 0.002), dose=2e5,
                 background_frac=0.15, descan_px=(3.0, -2.5), descan_quad_px=0.8, field_mrad=0.06, seed=0,
                 step_nm=0.25, ref_det_shape=(48, 48), truth=None):
    """Synthetic data set and vacuum reference.  libs: {"BaTiO3": Library, "SrTiO3": Library} (same zone).

    Returns dict(data (sy, sx, ky, kx) float32 counts, ref (sy, sx, rky, rkx), truth, cal_truth, meta)."""
    rng = np.random.default_rng(seed)
    sy, sx = scan_shape
    ny, nx = det_shape
    names = list(libs)
    lib0 = libs[names[0]]
    lam = fm.cr.wavelength_A(lib0.meta["energy_eV"])
    truth = truth or truth_fields(scan_shape, step_nm=step_nm,
                                  a_A=lib0.meta["material"]["a"])
    # detector mapping: crystal (1/A, projection frame x, y) -> detector px (row, col)
    th = np.deg2rad(rotation_deg)
    px_per_invA = lam * 1e3 / mrad_per_px
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # crystal x -> detector column-ish, crystal y (growth) -> detector row (upwards = -row)
    base = np.array([[0.0, -1.0], [1.0, 0.0]])        # (row, col) from (x, y): row = -y, col = x
    if mirror:
        base = base @ np.diag([-1, 1])
    M0 = px_per_invA * Rm @ base
    # distortion (q = u + D(u))
    c_mid = np.array([ny / 2, nx / 2])
    D = calm.Distortion(3, tuple(c_mid), float(max(ny, nx) / 2))
    nterm = len(D.terms())
    D.cy = np.zeros(nterm)
    D.cx = np.zeros(nterm)
    terms = D.terms()
    # a few physically plausible terms: quadratic + cubic radial
    D.cy[terms.index((2, 0))] = distortion[0] * D.scale
    D.cx[terms.index((1, 1))] = distortion[1] * D.scale
    D.cy[terms.index((3, 0))] = distortion[2] * D.scale
    D.cx[terms.index((2, 1))] = distortion[2] * D.scale * 0.5
    D.cy[terms.index((1, 2))] = distortion[2] * D.scale * 0.5
    D.cx[terms.index((0, 3))] = distortion[2] * D.scale
    # a calibration object holding the true geometry (for rendering)
    cal = calm.Calibration(det_shape=det_shape, radius_px=alpha_mrad / mrad_per_px, edge_px=edge_px, distortion=D)
    # detector lattice: use the two shortest crystal reflections of the library as basis
    g = lib0.g
    order = np.argsort(np.hypot(*g.T) + 1e-6 * np.arange(len(g)))
    gnz = g[order][1:]
    a = gnz[0]
    b = next(v for v in gnz[1:] if abs(a[0] * v[1] - a[1] * v[0]) > 1e-6 * np.linalg.norm(a) * np.linalg.norm(v))
    Bc = np.stack([a, b], 1)
    cal.B_cry = Bc
    cal.mrad_per_px = mrad_per_px
    model = fm.DetectorModel(libs, cal)
    Yi, Xi = np.mgrid[0:sy, 0:sx]
    sn = np.stack([(Yi - sy / 2) / sy, (Xi - sx / 2) / sx], -1)
    descan = sn * np.array(descan_px) * 2 + descan_quad_px * (sn ** 2).sum(-1, keepdims=True)
    # projected field deflection: opposite at the two interfaces of each layer (bound charge), along growth
    mat = truth["material"]
    dm = np.gradient(mat.astype(float), axis=0)
    defl_crys = np.stack([np.zeros_like(dm), field_mrad * np.clip(dm * 2, -1, 1)], -1)     # mrad, (x, y) crystal
    defl_px = defl_crys @ (M0 / (lam * 1e3)).T                                              # px
    centre = c_mid + descan + defl_px
    strain_yy = np.where(mat == 1, -0.02, 0.0)       # SrTiO3 shorter along growth (c_STO / c_BTO - 1)
    data = np.zeros((sy, sx, ny, nx), np.float32)
    ref = np.zeros((sy, sx) + tuple(ref_det_shape), np.float32)
    Uyx = np.stack(np.mgrid[0:ny, 0:nx], -1).astype(float)
    r_ref0 = np.array([(ny - ref_det_shape[0]) // 2, (nx - ref_det_shape[1]) // 2])
    G_true = np.zeros((sy, sx, 2, 2))
    for iy in range(sy):
        for ix in range(sx):
            F = np.diag([1.0, 1.0 + strain_yy[iy, ix]])            # real-space deformation (crystal frame)
            Mf = M0 @ np.linalg.inv(F).T
            G = Mf @ Bc
            G_true[iy, ix] = G
            c = centre[iy, ix]
            geo = model.geometry(c, G)
            p = [truth["thickness_nm"][iy, ix], truth["tilt_x_mrad"][iy, ix], truth["tilt_y_mrad"][iy, ix],
                 truth["u_pm"][iy, ix], truth["w_pm"][iy, ix]]
            if len(names) > 1:
                f = float(mat[iy, ix])
                cv = model.curves(p[0], p[3], p[4])
                I = model.render(cv, geo, (p[1], p[2]), f)
            else:
                cv = model.curves(p[0], p[3], p[4])
                I = model.render(cv, geo, (p[1], p[2]))
            img = np.zeros(ny * nx)
            area = (mrad_per_px ** 2) / (np.pi * alpha_mrad ** 2)
            img[geo.pix] = I * dose * area
            # background: smooth, centred on the beam, grows with thickness
            # (distorted coordinates: evaluate at the undistorted position of each pixel)
            Ud = model.Uflat
            rr = np.hypot(*(Ud - c).T) * mrad_per_px
            bg = background_frac * dose * area * (p[0] / 30.0) * 0.05 * np.exp(-rr / 25.0)
            img = img + bg
            data[iy, ix] = rng.poisson(np.clip(img, 0, None)).reshape(ny, nx)
            # vacuum reference: only the central disc, same descan (no field deflection)
            cr_ = c_mid + descan[iy, ix]
            qd = D.forward(cr_[None])[0] - r_ref0
            Yr, Xr = np.mgrid[0:ref_det_shape[0], 0:ref_det_shape[1]]
            Ur = D.inverse(np.stack([Yr.ravel() + r_ref0[0], Xr.ravel() + r_ref0[1]], 1)) - cr_
            rq = np.hypot(*Ur.T)
            from scipy.special import erfc
            disc = 0.5 * erfc((rq - cal.radius_px) / (np.sqrt(2) * edge_px)) * dose * area
            ref[iy, ix] = rng.poisson(disc).reshape(ref_det_shape)
    meta = dict(energy_eV=lib0.meta["energy_eV"], convergence_mrad=alpha_mrad, scan_step_nm=(step_nm, step_nm),
                detector_sampling=(mrad_per_px, mrad_per_px), detector_units="mrad", zone=lib0.meta["zone"],
                synthetic=True)
    return dict(data=data, ref=ref, truth=truth, centre=centre, descan=descan, deflection_mrad=defl_crys,
                G=G_true, M0=M0, B_cry=Bc, distortion=D.to_dict(), ref_offset=r_ref0.tolist(), meta=meta,
                strain_yy=strain_yy)
