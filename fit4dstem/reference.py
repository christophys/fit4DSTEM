# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Vacuum reference scan (descan of the beam) and projected electric field from the shift of the undiffracted beam.

The position of the undiffracted beam in a scan with a specimen is (descan) + (deflection by long-range fields).  A
scan recorded without specimen (same scan and projector settings) gives the descan alone; its frames may be smaller
(only the central disc is tracked) but must have the same pixel size.  ``offset`` is the position of the reference
frame's pixel (0, 0) on the full detector (default: centred crop).

Deflection angle theta and projected field:  theta = -(1 / (p v / e)) * integral of E dz, so
integral E dz = -theta * p v / e  and  E = integral / t  with the local thickness t from tab 3.
"""
from __future__ import annotations

import numpy as np

from . import calibration as calm
from . import crystal as cr


def centre_of_disc(frame, radius_px, edge_px=1.0):
    """Sub-pixel centre of the (single) disc of a frame from the ring correlation of its edge."""
    cmap = calm.ring_correlation(frame, radius_px, max(edge_px, 0.7))
    iy, ix = np.unravel_index(np.argmax(cmap), cmap.shape)
    p, _ = calm.subpixel(cmap, np.array([[iy, ix]]))
    return p[0]


def measure_reference(ds_ref, radius_px, edge_px=1.0, offset=None, distortion=None, positions=None, full_shape=None,
                      progress=None, stop=None):
    """Undistorted beam positions (full-detector pixels) of a vacuum scan.

    positions: list of (iy, ix) (coarse) or None for all.  Returns an array (sy, sx, 2) with NaN where not measured.
    """
    sy, sx = ds_ref.scan_shape
    ky, kx = ds_ref.det_shape
    if offset is None:
        if full_shape is None:
            offset = (0.0, 0.0)
        else:
            offset = ((full_shape[0] - ky) // 2, (full_shape[1] - kx) // 2)
    off = np.asarray(offset, float)
    out = np.full((sy, sx, 2), np.nan)
    if positions is None:
        k = 0
        for y0, y1, block in ds_ref.iter_rows(4):
            for a in range(y1 - y0):
                for b in range(sx):
                    out[y0 + a, b] = centre_of_disc(block[a, b], radius_px, edge_px) + off
            k += y1 - y0
            if progress:
                progress(k / sy, f"vacuum reference row {k}/{sy}")
            if stop and stop():
                raise InterruptedError("stopped")
    else:
        fr = ds_ref.frames(positions)
        for (iy, ix), f in zip(positions, fr):
            out[iy, ix] = centre_of_disc(f, radius_px, edge_px) + off
    if distortion is not None:
        ok = np.isfinite(out[..., 0])
        out[ok] = distortion.inverse(out[ok])
    return out


def descan_map(ref_centres, target_shape, mode="poly", degree=2):
    """Descan (undistorted px) on the scan grid of the specimen data.

    mode 'poly': 2D polynomial fit in normalized scan coordinates (same field of view assumed) -- smooth, robust;
    'interp': linear interpolation of the reference map (normalized coordinates); 'same': identical grids."""
    sy, sx = ref_centres.shape[:2]
    ty, tx = target_shape
    ok = np.isfinite(ref_centres[..., 0])
    if mode == "same":
        if (sy, sx) != (ty, tx):
            raise ValueError("the reference scan has a different size; use 'poly' or 'interp'")
        if ok.all():
            return ref_centres.copy()
        mode = "interp"
    Yr, Xr = np.mgrid[0:sy, 0:sx]
    pr = np.stack([(Yr[ok] + 0.5) / sy, (Xr[ok] + 0.5) / sx], 1)
    Yt, Xt = np.mgrid[0:ty, 0:tx]
    pt = np.stack([(Yt.ravel() + 0.5) / ty, (Xt.ravel() + 0.5) / tx], 1)
    if mode == "poly":
        out = np.stack([calm.poly_fit_eval(pr - 0.5, ref_centres[ok][:, i], pt - 0.5, degree) for i in range(2)], 1)
    else:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        out = np.empty((len(pt), 2))
        for i in range(2):
            li = LinearNDInterpolator(pr, ref_centres[ok][:, i])
            v = li(pt)
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = NearestNDInterpolator(pr, ref_centres[ok][:, i])(pt[bad])
            out[:, i] = v
    return out.reshape(ty, tx, 2)


def deflection_maps(centre_map, descan, M, energy_eV):
    """Deflection of the undiffracted beam (mrad, crystal projection frame x = in-plane, y = growth) from the
    undistorted beam positions with and without specimen.  M: px per 1/A (crystal -> detector)."""
    lam = cr.wavelength_A(energy_eV)
    dq = np.asarray(centre_map, float) - np.asarray(descan, float)          # px
    k = dq.reshape(-1, 2) @ np.linalg.inv(M).T                              # 1/A
    return (k * lam * 1e3).reshape(dq.shape)


def field_maps(deflection_mrad, thickness_nm, energy_eV):
    """Projected field integral (V) and mean field (MV/cm) in the crystal frame."""
    pv = cr.pv_over_e(energy_eV)
    integral_V = -np.asarray(deflection_mrad) * 1e-3 * pv                  # V (= integral of E dz)
    t = np.asarray(thickness_nm, float)[..., None]
    E_V_per_nm = integral_V / np.where(t > 0, t, np.nan)
    return integral_V, E_V_per_nm * 10.0                                    # 1 V/nm = 10 MV/cm
