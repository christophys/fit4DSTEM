# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Headless analysis of a 4D-STEM data set with the parameters of a previous analysis.

The app needs a person for the calibration (tab 2: zone assignment, aperture, distortion) and the libraries
(tab 3).  Once a data set has been analysed, the same parameters apply to further scans recorded under the same
conditions (same microscope settings, crystal and zone).  This module reads these parameters from

* a **previous result**: the NXem file written by fit4dstem (tab 4) -- calibration (disc radius and edge,
  polynomial distortion, crystal assignment of the disc lattice, mrad/px, background), fit options, the crystal
  structures and build parameters of the rocking-curve libraries, high tension, convergence angle; or
* an **ELN entry** (``<name>.archive.yaml`` / ``.json`` of ``ELNFourDSTEMMeasurement``, based on the
  ``MappingMeasurement`` of nomad-measurements): settings (high tension, convergence, scan step, zone axis), the
  data sets with their roles (specimen, vacuum reference, *library*) and the NXem results file, whose calibration
  is then used,

and runs the analysis of tabs 2-4 without the GUI: coarse calibration of the new data starting from the previous
one, full pass, vacuum reference (descan, deflection), map fit with neighbour seeding, strain, polarization and field
maps, NXem + ELN + workflow.  It is the core of the NOMAD action ``fit4dstem_nomad.actions:fit4dstem_map_fit``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import calibration as calm
from . import crystal as cr
from . import fitting as fm
from . import io as fio
from . import multislice as ms
from . import reference as ref
from . import strain as st

LIB_BUILD_KEYS = ("radius_mrad", "step_mrad", "g_max_inv_A", "max_thickness_nm", "absorption")


@dataclass
class Prior:
    """Parameters of a previous analysis."""
    source: str = ""
    energy_eV: float | None = None
    convergence_mrad: float | None = None
    scan_step_nm: float | None = None
    zone: str | None = None
    materials: dict = field(default_factory=dict)       # name -> Material fields (crystal structure)
    calibration: dict | None = None                     # Calibration.to_dict()
    options: dict = field(default_factory=dict)         # FitOptions fields
    library_meta: dict = field(default_factory=dict)    # name -> library meta (structure, zone, build parameters)
    library_files: list = field(default_factory=list)   # files of the libraries (paths in the upload / folder)
    data_file: str | None = None
    reference_file: str | None = None
    results_file: str | None = None

    def summary(self):
        mats = ", ".join(f"{k} (a = {v.get('a')} Å, c = {v.get('c')} Å)" for k, v in self.materials.items())
        return (f"parameters from {os.path.basename(self.source)}: "
                + ", ".join(x for x in [
                    f"{self.energy_eV / 1e3:g} kV" if self.energy_eV else "",
                    f"α = {self.convergence_mrad:g} mrad" if self.convergence_mrad else "",
                    f"zone {self.zone}" if self.zone else "", mats,
                    "calibration" if self.calibration else "no calibration",
                    f"{len(self.library_files)} library file(s)" if self.library_files else ""] if x))


# ---------------------------------------------------------------------------------------------- reading priors
def _find_collection(f):
    """The ``fit4dstem`` NXcollection of a fit4dstem NXem file (anywhere in the file)."""
    found = []

    def visit(name, obj):
        if name.split("/")[-1] == "fit4dstem" and hasattr(obj, "keys"):
            found.append(obj)
    f.visititems(visit)
    return found[0] if found else None


def _first(f, suffix):
    hits = []

    def visit(name, obj):
        if name.endswith(suffix) and not hasattr(obj, "keys"):
            hits.append(obj)
    f.visititems(visit)
    return hits[0] if hits else None


def _str(v):
    v = v[()] if hasattr(v, "shape") else v
    return v.decode() if isinstance(v, bytes) else str(v)


def read_prior_nxem(path):
    """Prior from a fit4dstem NXem results file."""
    import h5py
    p = Prior(source=str(path), results_file=os.path.basename(path))
    with h5py.File(path, "r") as f:
        coll = _find_collection(f)
        if coll is None:
            raise ValueError(f"{os.path.basename(path)} is not a fit4dstem results file (no fit4dstem collection)")
        if "calibration" in coll:
            p.calibration = json.loads(_str(coll["calibration"]))
        if "options" in coll:
            p.options = json.loads(_str(coll["options"]))
        if "libraries" in coll:
            p.library_meta = json.loads(_str(coll["libraries"]))
        v = _first(f, "electron_source/voltage")
        if v is not None:
            p.energy_eV = float(v[()])
        a = _first(f, "semi_convergence_angle")
        if a is not None:
            p.convergence_mrad = float(a[()]) * 1e3
        inputs = []

        def visit(name, obj):
            if name.split("/")[-1].startswith("input") and hasattr(obj, "keys") and "file_name" in obj:
                inputs.append((_str(obj["file_name"]), _str(obj["description"]) if "description" in obj else ""))
        f.visititems(visit)
    for fn, desc in inputs:
        if "reference" in desc.lower():
            p.reference_file = os.path.basename(fn)
        elif p.data_file is None:
            p.data_file = os.path.basename(fn)
    for name, meta in p.library_meta.items():
        if meta.get("material"):
            p.materials[name] = dict(meta["material"])
        p.zone = p.zone or meta.get("zone")
        p.energy_eV = p.energy_eV or meta.get("energy_eV")
    if p.calibration and p.calibration.get("zone_assignment", {}).get("zone"):
        p.zone = p.zone or p.calibration["zone_assignment"]["zone"]
    return p


def read_prior_eln(path, resolve=None):
    """Prior from an ELN archive (``.archive.yaml`` / ``.archive.json``) of ``ELNFourDSTEMMeasurement``.
    ``resolve(file)`` returns a local path for a file named in the ELN (paths relative to the upload root), used
    for the NXem results file (calibration) -- without it, only the settings and file names are taken."""
    if str(path).lower().endswith((".yaml", ".yml")):
        import yaml
        with open(path, encoding="utf-8") as fh:
            arch = yaml.safe_load(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            arch = json.load(fh)
    data = arch.get("data", arch)
    s = data.get("settings") or {}
    p = Prior(source=str(path))
    nx = data.get("nxem_file") or next((d.get("file") for d in data.get("datasets", [])
                                        if str(d.get("file", "")).lower().endswith(".nxs")
                                        and d.get("role") == "results"), None)
    if nx and resolve is not None:
        local = resolve(nx)
        if local and os.path.isfile(local):
            p = read_prior_nxem(local)
            p.source = str(path)
            p.results_file = nx
    p.energy_eV = float(s["acceleration_voltage"]) if s.get("acceleration_voltage") else p.energy_eV
    p.convergence_mrad = float(s["convergence_semi_angle"]) if s.get("convergence_semi_angle") else p.convergence_mrad
    p.scan_step_nm = float(s["scan_step"]) if s.get("scan_step") else p.scan_step_nm
    p.zone = s.get("zone_axis") or p.zone
    libs = []
    for d in data.get("datasets", []) or []:
        role, fn = str(d.get("role", "")).lower(), d.get("file")
        if not fn:
            continue
        if role.startswith("library"):
            libs.append(fn)
        elif role.startswith("vacuum") or "reference" in role:
            p.reference_file = p.reference_file or fn
        elif role.startswith("specimen"):
            p.data_file = p.data_file or fn
    p.library_files = libs or p.library_files
    return p


def read_prior(path, resolve=None):
    """Prior from a fit4dstem NXem file or an ELN archive (by the file name)."""
    low = str(path).lower()
    if low.endswith((".yaml", ".yml", ".json")):
        return read_prior_eln(path, resolve)
    return read_prior_nxem(path)


# ---------------------------------------------------------------------------------------------- stages
def _say(cb, stage, frac, msg):
    if cb:
        cb(stage, float(frac), msg)


def load_libraries(prior: Prior, library_paths=(), build_missing=False, work_dir=".", callback=None, stop=None):
    """{name: Library}: the given library files; missing ones (materials of the prior without a file) are built
    from the crystal structure and the build parameters of the prior if ``build_missing``."""
    libs = {}
    for pth in library_paths or ():
        lib = ms.Library.load(pth)
        lib.path = pth
        name = lib.material
        libs[name] = lib
        _say(callback, "libraries", 0.0, f"library {name} {lib.meta.get('zone')} from {os.path.basename(pth)}")
    missing = [n for n in (prior.library_meta or prior.materials) if n not in libs]
    if missing and build_missing:
        for k, name in enumerate(missing):
            meta = prior.library_meta.get(name, {})
            mat = cr.Material(**{**cr.MATERIALS.get(name, cr.Material(name)).to_dict(),
                                 **(prior.materials.get(name) or meta.get("material") or {})})
            zone = meta.get("zone") or prior.zone or "[110]"
            kw = {k2: meta[k2] for k2 in LIB_BUILD_KEYS if k2 in meta}
            for k2 in ("u_pm", "w_pm"):
                if meta.get(k2):
                    kw[k2] = tuple(float(v) for v in meta[k2])
            if meta.get("cell_A") and meta.get("gpts"):
                kw["sampling"] = float(meta["cell_A"][0]) / float(meta["gpts"][0])
            path = os.path.join(work_dir, f"lib_{name}_{zone.strip('[]')}.h5")
            _say(callback, "libraries", k / len(missing), f"building library {name} {zone} (this can take long)")
            ms.build_library(mat, zone, float(prior.energy_eV or meta.get("energy_eV") or 60e3), path=path, log=None,
                             stop=stop, **kw)
            libs[name] = ms.Library.load(path)
            libs[name].path = path
    elif missing and not libs:
        raise ValueError("no rocking-curve libraries: give the library files, or allow building them")
    return libs


def calibrate(ds, prior: Prior, ds_ref=None, recalibrate=True, full=True, coarse_n=8, callback=None, stop=None):
    """Calibration of ``ds`` from the prior: same aperture, distortion, crystal assignment and mrad/px; the beam
    position and lattice maps are measured on the new data (coarse sample + full pass); with a vacuum reference,
    the descan.  Returns (cal, descan or None)."""
    if not prior.calibration:
        raise ValueError(f"{os.path.basename(prior.source)} contains no calibration (analyse one data set in the app "
                         f"first and export its NXem file)")
    cal = calm.Calibration.from_dict(prior.calibration)
    if tuple(cal.det_shape) != tuple(ds.det_shape):
        raise ValueError(f"detector size {tuple(ds.det_shape)} differs from the calibration {tuple(cal.det_shape)} "
                         f"(use the same detector binning)")
    if recalibrate:
        pos = fio.coarse_positions(ds.scan_shape, coarse_n)
        frames = ds.frames(pos)
        order = max(int(cal.distortion.order), 2) if cal.distortion.order else 3
        _say(callback, "calibrate", 0.1, f"coarse calibration of {len(pos)} patterns (distortion order {order})")
        calm.coarse_calibration(frames, pos, cal, distortion_order=order, log=None)
    else:
        cal.coarse_positions, cal.coarse_centres, cal.coarse_G = [], np.zeros((0, 2)), np.zeros((0, 2, 2))
    if full:
        _say(callback, "calibrate", 0.3, "full pass over all patterns")
        calm.full_pass(ds, cal, progress=lambda f, m="": _say(callback, "calibrate", 0.3 + 0.6 * f, m),
                       stop=stop)
    else:
        calm.apply_interpolated(cal, ds.scan_shape)
    descan = None
    if ds_ref is not None:
        _say(callback, "calibrate", 0.92, "vacuum reference: descan")
        rc = ref.measure_reference(ds_ref, cal.radius_px, cal.edge_px, distortion=cal.distortion,
                                   full_shape=ds.det_shape, stop=stop)
        descan = ref.descan_map(rc, ds.scan_shape, mode="poly", degree=2)
    _say(callback, "calibrate", 1.0, cal.summary())
    return cal, descan


def fit_map(ds, cal, libs, options=None, scan_bin=1, seeds=4, refine_passes=1, roi=None, callback=None, stop=None,
            n_threads=1):
    """Map fit with neighbour seeding (tab 3).  Returns the MapResult."""
    names = set(fm.FitOptions.__dataclass_fields__)
    opt = fm.FitOptions(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in (options or {}).items()
                           if k in names})
    opt.mixture = bool(opt.mixture and len(libs) > 1)
    model = fm.DetectorModel(libs if opt.mixture else {k: v for k, v in libs.items()
                                                       if k == "BaTiO3" or len(libs) == 1}, cal)
    mf = fm.MapFitter(ds, cal, model, opt, scan_bin=scan_bin, roi=roi,
                      progress=lambda f, m="": _say(callback, "fit", f, m), stop=stop, n_threads=n_threads)
    mf.load()
    res = mf.run(fm.seed_grid(mf.shape, seeds), refine_passes=refine_passes)
    res.info.update(roi=roi, scan_bin=scan_bin, options=opt.__dict__)
    return res, opt


def library_meta(lib):
    """Metadata of a library incl. its (u, w) grid, so that it can be rebuilt from a results file."""
    m = dict(lib.meta)
    m.update(u_pm=[float(v) for v in np.ravel(lib.u)], w_pm=[float(v) for v in np.ravel(lib.w)])
    return m


def maps_from_result(res, cal, descan=None, energy_eV=60e3, materials=None, zone="[110]"):
    """{name: (array, units, long name, method)} as in tab 4 (without smoothing)."""
    labels = {"thickness_nm": ("thickness", "nm"), "tilt_x_mrad": ("tilt_x", "mrad"),
              "tilt_y_mrad": ("tilt_y", "mrad"), "u_pm": ("Ti_displacement_u", "pm"),
              "w_pm": ("Ti_displacement_w", "pm"), "f_STO": ("SrTiO3_fraction", "")}
    out = {}
    for k, n in enumerate(res.names):
        lab, u = labels[n]
        out[lab] = (res.params[..., k], u, lab.replace("_", " "), "CBED fit (fit4dstem, NOMAD action)")
        out[lab + "_sigma"] = (res.sigma[..., k], u, lab.replace("_", " ") + " uncertainty", "Cramer-Rao (fit)")
    out["reduced_chi2"] = (res.reduced_chi2(), "", "reduced chi2 (Anscombe)", "fit quality")
    mat = cr.Material(**(materials or {}).get("BaTiO3", cr.MATERIALS["BaTiO3"].to_dict()))
    P = cr.polarization(mat, zone, out["Ti_displacement_u"][0], out["Ti_displacement_w"][0])
    out["polarization_x"] = (P[..., 0], "uC/cm^2", "polarization estimate in-plane", "Born effective charges")
    out["polarization_y"] = (P[..., 1], "uC/cm^2", "polarization estimate growth", "Born effective charges")
    lat = res.lattice if res.lattice is not None and np.isfinite(res.lattice).any() else None
    seg = fm.segment_layers(res.params[..., 5]) if res.params.shape[-1] > 5 else None
    if lat is not None:
        try:
            S = st.strain_maps(lat, cal.B_cry, ref_mask=None if seg is None else seg == 0)
            for key, lab in (("exx", "strain_xx"), ("eyy", "strain_yy"), ("exy", "strain_xy"),
                             ("rotation_rad", "lattice_rotation")):
                out[lab] = (S[key], "rad" if "rot" in key else "", lab.replace("_", " "),
                            "disc lattice, relative to mean BaTiO3")
        except ValueError:
            pass
    if descan is not None and res.shape == tuple(descan.shape[:2]):
        centre = cal.centre_map
        if res.centre is not None and np.isfinite(res.centre).any() and res.bin == 1:
            centre = np.where(np.isfinite(res.centre), res.centre, centre)
        M = st.mapping_maps(cal.G_map, cal.B_cry)
        d = ref.deflection_maps(centre, descan, np.nanmean(M.reshape(-1, 2, 2), 0), energy_eV)
        _, E = ref.field_maps(d, out["thickness"][0], energy_eV)
        out["deflection_x"] = (d[..., 0], "mrad", "beam deflection in-plane", "beam - vacuum ref.")
        out["deflection_y"] = (d[..., 1], "mrad", "beam deflection growth", "beam - vacuum ref.")
        out["field_x"] = (E[..., 0], "MV/cm", "projected field / thickness, in-plane", "deflection")
        out["field_y"] = (E[..., 1], "MV/cm", "projected field / thickness, growth", "deflection")
    return out


def run(data_path, prior, out_dir, reference_path=None, library_paths=(), build_missing=False, dataset=None,
        det_bin=1, recalibrate=True, full=True, scan_bin=1, seeds=4, refine_passes=1, overrides=None, name=None,
        callback=None, stop=None, metadata=None, write_eln=True, file_names=None):
    """Whole analysis with the parameters of ``prior`` (a :class:`Prior` or the path of a NXem / ELN file).
    ``file_names``: names of the input files as they should appear in the ELN/workflow (paths in the upload).
    Returns dict(files, maps, summary, ...)."""
    from . import eln, nxem
    t0 = time.time()
    if not isinstance(prior, Prior):
        prior = read_prior(prior)
    o = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    if "energy_kV" in o:
        prior.energy_eV = float(o["energy_kV"]) * 1e3
    if "convergence_mrad" in o:
        prior.convergence_mrad = float(o["convergence_mrad"])
    if "scan_step_nm" in o:
        prior.scan_step_nm = float(o["scan_step_nm"])
    if "zone" in o:
        prior.zone = o["zone"]
    _say(callback, "load", 0.0, prior.summary())
    os.makedirs(out_dir, exist_ok=True)
    libs = load_libraries(prior, library_paths, build_missing, out_dir, callback, stop)
    ds = fio.open_dataset(data_path, dataset, det_bin=det_bin)
    ds_ref = fio.open_dataset(reference_path, det_bin=det_bin) if reference_path else None
    energy = float(prior.energy_eV or ds.meta.get("energy_eV") or 60e3)
    step = float(prior.scan_step_nm or (ds.meta.get("scan_step_nm") or [0.2])[0])
    _say(callback, "load", 1.0, f"{os.path.basename(data_path)}: " + ds.describe())
    try:
        cal, descan = calibrate(ds, prior, ds_ref, recalibrate, full, callback=callback, stop=stop)
        res, opt = fit_map(ds, cal, libs, prior.options, scan_bin, seeds, refine_passes, callback=callback, stop=stop)
        maps = maps_from_result(res, cal, descan, energy, prior.materials, prior.zone or "[110]")
        nm = name or os.path.splitext(os.path.basename(data_path))[0] + "_fit4dstem"
        out = os.path.join(out_dir, nm + ".nxs")
        meta = metadata or nxem.default_metadata()
        fnames = dict(file_names or {})
        inputs = [dict(file=fnames.get("data", os.path.basename(data_path)), description="4D-STEM data set (specimen)")]
        if reference_path:
            inputs.append(dict(file=fnames.get("reference", os.path.basename(reference_path)),
                               description="vacuum reference scan"))
        extra = {"libraries": {k: library_meta(v) for k, v in libs.items()},
                 "parameters_from": fnames.get("prior", os.path.basename(prior.source))}
        _say(callback, "write", 0.0, f"writing {nm}.nxs")
        nxem.write_results(out, maps, step_nm=(step * res.bin, step * res.bin), metadata=meta, energy_eV=energy,
                           convergence_mrad=prior.convergence_mrad, calibration=cal.to_dict(), options=opt.__dict__,
                           inputs=inputs, mrad_per_px=cal.mrad_per_px, extra=extra,
                           description=f"fit4dstem maps of {os.path.basename(data_path)} with the parameters of "
                                       f"{os.path.basename(prior.source)} (NOMAD action)")
        files = [out]
        summary = (f"thickness {np.nanmedian(res.params[..., 0]):.1f} nm (median), |Ti displacement| "
                   f"{np.nanmedian(np.hypot(res.params[..., 3], res.params[..., 4])):.1f} pm (median), "
                   f"{int(res.done.sum())} patterns fitted")
        if write_eln:
            pre = fnames.get("prefix", "").strip("/")

            def _libname(k, v):
                if f"library:{k}" in fnames:
                    return fnames[f"library:{k}"]
                b = os.path.basename(getattr(v, "path", "") or f"lib_{k}.h5")
                return f"{pre}/{b}" if pre else b
            lib_names = [_libname(k, v) for k, v in libs.items()]
            paths = eln.write_eln(
                out_dir, nm, data_file=inputs[0]["file"], reference_file=inputs[1]["file"] if reference_path else None,
                results_file=fnames.get("results", os.path.basename(out)), library_files=lib_names,
                settings=dict(energy_eV=energy, convergence_mrad=prior.convergence_mrad, scan_step_nm=step,
                              scan_shape=list(ds.scan_shape), det_shape=list(ds.det_shape),
                              mrad_per_px=cal.mrad_per_px, zone=prior.zone),
                metadata=meta, calibration=cal.summary(), options=opt.__dict__, results_summary=summary,
                prefix=fnames.get("prefix", ""),
                step_details=[{"parameters_from": extra["parameters_from"], "data": inputs[0]["file"]},
                              {"recalibrated": recalibrate, "full_pass": full}, {"scan_bin": scan_bin, "seeds": seeds,
                                                                                "refine_passes": refine_passes}, {}])
            files += [paths[k] for k in paths]
        _say(callback, "write", 1.0, f"done in {time.time() - t0:.0f} s")
        built = [getattr(v, "path") for v in libs.values() if getattr(v, "path", None)
                 and os.path.dirname(os.path.abspath(v.path)) == os.path.abspath(out_dir)]
        return dict(files=files, built_libraries=built, maps=sorted(maps), summary=summary,
                    scan_shape=list(ds.scan_shape), runtime_s=round(time.time() - t0, 1), prior=prior.summary())
    finally:
        ds.close()
        if ds_ref is not None:
            ds_ref.close()
