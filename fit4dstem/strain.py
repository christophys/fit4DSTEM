# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Strain and lattice rotation from the lattice of the discs (positions, not intensities).

Per pattern the detector lattice G (columns: basis vectors, undistorted px) and the crystal assignment B (crystal
reciprocal basis vectors, 1/A) give M = G B^-1 (px per 1/A).  For a real-space deformation F of the crystal the
reciprocal vectors transform with F^-T, so relative to a reference (mean BaTiO3 lattice)

    F = (M_ref^-1 M)^-T,   strain = (F + F^T)/2 - 1,   rotation = (F_yx - F_xy)/2

in the crystal frame (x = in-plane, y = growth direction).  A common (linear) magnification or distortion of the
detector cancels; beam shifts (descan, fields) do not enter.
"""
from __future__ import annotations

import numpy as np


def mapping_maps(G_map, B_cry):
    return np.einsum("...ij,jk->...ik", np.asarray(G_map, float), np.linalg.inv(B_cry))


def strain_maps(G_map, B_cry, ref_mask=None):
    """Strain components (exx, eyy, exy) and rotation (rad) relative to the mean lattice in ref_mask."""
    M = mapping_maps(G_map, B_cry)
    ok = np.isfinite(M).all(axis=(-1, -2))
    m = ok if ref_mask is None else (ok & np.asarray(ref_mask, bool))
    if not m.any():
        raise ValueError("empty reference region")
    M_ref = np.mean(M[m], axis=0)
    Minv = np.linalg.inv(M_ref)
    A = np.einsum("ij,...jk->...ik", Minv, M)                  # = F^-T
    F = np.full_like(A, np.nan)
    F[ok] = np.linalg.inv(np.swapaxes(A[ok], -1, -2))
    exx = F[..., 0, 0] - 1
    eyy = F[..., 1, 1] - 1
    exy = 0.5 * (F[..., 0, 1] + F[..., 1, 0])
    rot = 0.5 * (F[..., 1, 0] - F[..., 0, 1])
    return dict(exx=exx, eyy=eyy, exy=exy, rotation_rad=rot, M_ref=M_ref)


def lattice_parameters(strain, a_ref_x, a_ref_y):
    """Local lattice parameters (Angstrom) along x and y from the strain and the reference lattice parameters."""
    return a_ref_x * (1 + strain["exx"]), a_ref_y * (1 + strain["eyy"])
