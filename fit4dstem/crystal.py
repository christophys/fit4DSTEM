# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Perovskite structures (BaTiO3, SrTiO3) in cross-section zone-axis orientation, polar displacement pattern,
reciprocal lattice of the zone and Born effective charges.

Frames
------
* cubic frame: the pseudo-cubic perovskite axes [100], [010], [001]; [001] is the growth direction.
* projection (zone) frame: x = in-plane direction perpendicular to the beam, y = [001] (growth),
  z = beam direction.  Zone [110]: x = [1-10], z = [110].  Zone [100]: x = [010], z = [100].

Displacements are given as the Ti displacement vector relative to the A-site (Ba/Sr) sublattice, in pm, with
components (u, w) along the projection axes x (in-plane) and y (growth).  The O ions follow the soft-mode pattern of
tetragonal P4mm BaTiO3 (Kwei et al. 1993): an O ion on the cube face normal to axis i ("apical" for a polarization
along i) moves by r_eq * d + (r_ap - r_eq) * d_i e_i, where d is the Ti displacement.  For d along an axis this is
exactly the measured P4mm pattern (Ti +0.0224 c, O_eq -0.0105 c, O_ap -0.0244 c).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# P4mm BaTiO3 (Kwei et al., J. Phys. Chem. 97, 2368 (1993)): shifts along the polar axis in units of c
KWEI_SHIFTS = {"Ti": 0.0224, "O_eq": -0.0105, "O_ap": -0.0244}
R_EQ = KWEI_SHIFTS["O_eq"] / KWEI_SHIFTS["Ti"]
R_AP = KWEI_SHIFTS["O_ap"] / KWEI_SHIFTS["Ti"]

# Born effective charges (e) along / perpendicular to the Ti-O bond of the O ion (Zhong, King-Smith, Vanderbilt,
# PRL 72, 3618 (1994)); used to convert displacements into a polarization estimate.
BORN = {
    "BaTiO3": {"A": 2.77, "Ti": 7.25, "O_par": -5.71, "O_perp": -2.15},
    "SrTiO3": {"A": 2.56, "Ti": 7.26, "O_par": -5.73, "O_perp": -2.05},
}

ZONES = {
    # name: (x, y, z) as cubic direction vectors (not normalized)
    "[110]": ((1, -1, 0), (0, 0, 1), (1, 1, 0)),
    "[100]": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
}


@dataclass
class Material:
    """A perovskite ABO3 with tetragonal lattice (a in-plane, c along the growth direction [001])."""
    name: str = "BaTiO3"
    A: str = "Ba"
    a: float = 3.990          # Angstrom
    c: float = 4.030          # Angstrom
    r_eq: float = R_EQ        # O displacement ratios (soft-mode pattern); 0, 0 = Ti only
    r_ap: float = R_AP

    def to_dict(self):
        return asdict(self)


MATERIALS = {
    "BaTiO3": Material("BaTiO3", "Ba", 3.990, 4.030),
    "SrTiO3": Material("SrTiO3", "Sr", 3.982, 3.905),
}


def zone_axes(zone):
    """Orthonormal projection axes (3x3, rows x, y, z) in the cubic frame and the integer direction vectors."""
    vx, vy, vz = (np.array(v, float) for v in ZONES[zone])
    R = np.array([vx / np.linalg.norm(vx), vy / np.linalg.norm(vy), vz / np.linalg.norm(vz)])
    return R, (vx, vy, vz)


def cell_lengths(mat: Material, zone):
    """Lengths (Lx, Ly, Lz) of the orthorhombic zone cell (Angstrom)."""
    _, (vx, vy, vz) = zone_axes(zone)
    L = np.array([mat.a, mat.a, mat.c])
    return tuple(float(np.linalg.norm(v * L)) for v in (vx, vy, vz))


def cubic_displacements(mat: Material, d_cubic_A):
    """Displacements (Angstrom, cubic frame) of the 5 atoms A, Ti, O_x, O_y, O_z of the perovskite cell for a Ti
    displacement d (Angstrom, cubic frame).  O_i is the O on the face normal to i (at coordinate i = 0)."""
    d = np.asarray(d_cubic_A, float)
    out = [np.zeros(3), d.copy()]
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        out.append(mat.r_eq * d + (mat.r_ap - mat.r_eq) * d[i] * e)
    return np.array(out)


def proj_to_cubic(zone, u_pm, w_pm):
    """Displacement (u, w) in pm along the projection axes x, y -> vector in Angstrom in the cubic frame."""
    R, _ = zone_axes(zone)
    return (u_pm * R[0] + w_pm * R[1]) / 100.0


def zone_cell(mat: Material, zone, u_pm=0.0, w_pm=0.0, sublattice_tags=True):
    """ase.Atoms of the orthorhombic zone cell (x, y lateral, z along the beam) with the Ti displacement (u, w) pm.

    Tags: 0 = A, 1 = Ti, 2 = O_x, 3 = O_y, 4 = O_z (O_i on the cube face normal to cubic axis i)."""
    from ase import Atoms

    _, (vx, vy, vz) = zone_axes(zone)
    Lc = np.array([mat.a, mat.a, mat.c])
    frac = np.array([[0, 0, 0], [.5, .5, .5], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    disp = cubic_displacements(mat, proj_to_cubic(zone, u_pm, w_pm))
    sym = [mat.A, "Ti", "O", "O", "O"]
    M = np.array([vx, vy, vz])                  # zone cell vectors in units of the cubic cell
    n = int(round(abs(np.linalg.det(M))))       # formula units per zone cell
    # all lattice translations of the cubic cell inside the zone cell
    rng = range(-3, 4)
    Minv = np.linalg.inv(M.T)
    pos, syms, tags = [], [], []
    for i in rng:
        for j in rng:
            for k in rng:
                for s, f, dd, t in zip(sym, frac, disp, range(5)):
                    fc = np.array([i, j, k], float) + f
                    fz = Minv @ fc                     # fractional coordinates in the zone cell
                    if np.all(fz >= -1e-9) and np.all(fz < 1 - 1e-9):
                        pos.append(fz)
                        syms.append(s)
                        tags.append(t)
    pos = np.array(pos)
    if len(pos) != 5 * n:
        raise RuntimeError(f"zone cell construction failed ({len(pos)} atoms for {n} formula units)")
    R, _ = zone_axes(zone)
    lengths = np.array(cell_lengths(mat, zone))
    cart = pos * lengths
    # displacements: cubic frame -> projection frame
    dproj = np.array([R @ disp[t] for t in tags])
    atoms = Atoms(syms, positions=cart + dproj, cell=np.diag(lengths), pbc=True)
    if sublattice_tags:
        atoms.set_tags(tags)
    atoms.wrap()
    return atoms


def reflections(mat: Material, zone, g_max):
    """Zero-order-Laue-zone reflections of the zone cell up to g_max (1/Angstrom): (h, k) integer indices along the
    projection axes and their vectors (1/Angstrom).  The cell is centred for [110] (perovskite in the orthorhombic
    setting): only reflections allowed by the lattice centring are returned."""
    Lx, Ly, _ = cell_lengths(mat, zone)
    nh, nk = int(g_max * Lx) + 1, int(g_max * Ly) + 1
    hk, gv = [], []
    for h in range(-nh, nh + 1):
        for k in range(-nk, nk + 1):
            g = np.array([h / Lx, k / Ly])
            if np.hypot(*g) <= g_max + 1e-9:
                hk.append((h, k))
                gv.append(g)
    return np.array(hk, int), np.array(gv)


def polarization(mat: Material, zone, u_pm, w_pm):
    """Polarization estimate (uC/cm^2) along the projection axes x, y from the Ti displacement (u, w) in pm and the
    displacement pattern, with Born effective charges: P = e / Omega * sum_i Z*_i d_i.  Arrays are broadcast; the
    result has a trailing axis of length 2 (P_x, P_y)."""
    Z = BORN.get(mat.name, BORN["BaTiO3"])
    R, _ = zone_axes(zone)
    # linear in d: P = K d_Ti with a 3x3 matrix K (cubic frame)
    K = Z["Ti"] * np.eye(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        # O_i displacement = r_eq d + (r_ap - r_eq) d_i e_i ; parallel part (along i) and perpendicular part
        Mi = mat.r_eq * np.eye(3) + (mat.r_ap - mat.r_eq) * np.outer(e, e)
        Pi = np.outer(e, e)
        K = K + Z["O_par"] * Pi @ Mi + Z["O_perp"] * (np.eye(3) - Pi) @ Mi
    omega = mat.a * mat.a * mat.c                       # Angstrom^3
    conv = 1.602176634e-19 * 1e-10 / (omega * 1e-30) * 100.0   # e*Angstrom/Angstrom^3 -> C/m^2 -> uC/cm^2
    u, w = np.broadcast_arrays(np.asarray(u_pm, float), np.asarray(w_pm, float))
    d = (u[..., None] * R[0] + w[..., None] * R[1]) / 100.0     # Angstrom, cubic frame
    P = np.einsum("ij,...j->...i", K, d) * conv
    return np.stack([P @ R[0], P @ R[1]], axis=-1)


@dataclass
class ZoneGeometry:
    """Geometry of the zone axis pattern: material, zone, energy."""
    material: Material = field(default_factory=lambda: MATERIALS["BaTiO3"])
    zone: str = "[110]"
    energy_eV: float = 60e3

    @property
    def wavelength_A(self):
        return wavelength_A(self.energy_eV)


def wavelength_A(E):
    """Relativistic electron wavelength (Angstrom) for the acceleration voltage E (V)."""
    h, m0, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    return h / np.sqrt(2 * m0 * e * E * (1 + e * E / (2 * m0 * c * c))) * 1e10


def interaction_sigma(E):
    """Interaction constant sigma (rad / (V Angstrom))."""
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0
    lam = wavelength_A(E) * 1e-10
    gamma = 1 + e * E / (m0 * c * c)
    return 2 * np.pi * gamma * m0 * e * lam / (6.62607015e-34 ** 2) * 1e-10


def pv_over_e(E):
    """p*v/e (V) of the electrons: deflection angle theta = (integral of E dz) / (p v / e)."""
    m0c2 = 510998.95
    gamma = 1 + E / m0c2
    beta2 = 1 - 1 / gamma ** 2
    return gamma * m0c2 * beta2
