# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Export of the reconstructed maps as NeXus / NXem (HDF5), validated with pynxtools.

Layout (patterned on inlineHolo's NXem writer, which passes ``pynx validate``)::

  /entry1                                  NXentry  definition = "NXem", start_time, end_time
    profiling/program1                     NXcs_profiling / NXprogram  (fit4dstem + version)
    user1                                  NXuser
    sample1                                NXsample  is_simulation, atom_types, preparation_date, ...
    measurement                            NXem_measurement
      instrument                           NXem_instrument  type = stem
        fabrication                        NXfabrication  vendor, model
        ebeam_column/electron_source       NXsource  voltage [V], emitter_type, probe
        optics                             NXem_optical_system  semi_convergence_angle [rad]
        detector1                          NXdetector  name
        fit4dstem                          NXcollection  calibration (tab 2), library, fit options (JSON)
      event1                               NXem_event_data
        image1                             NXimage  mean diffraction pattern of the coarse sample (NXdata)
      input1, input2                       NXnote  the 4D-STEM data set and the vacuum reference (file, size)
    roi1                                   NXroi_process
      img                                  NXem_img
        image1 ... imageK                  NXimage  imaging_mode = 4dstem_cbed_fit (custom)
          reconstruction                   NXprocess  program, version, date, sequence_index
            parameters                     NXparameters  quantity, units, method, options (JSON)
          image_2d                         NXdata  real [units], axis_i / axis_j (nm, scan coordinates)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os

import h5py
import numpy as np

from . import __version__

IMAGING_MODE = "4dstem_cbed_fit"
IDENTIFIER_TYPES = ("ARK", "DOI", "Hdl", "IGSN", "ISNI", "ISSN", "ISSN-L", "ORCID", "PURL", "ROR", "URL", "URN")

METADATA_FIELDS = {
    "experiment": {
        "identifier": ("Experiment identifier", "", False, "lab-book number / proposal ID"),
        "description": ("Experiment description", "", False, "free text"),
        "start_time": ("Acquisition start (ISO 8601)", "", True, "e.g. 2026-11-20T14:20:00+01:00"),
        "end_time": ("Acquisition end (ISO 8601)", "", False, ""),
        "doi": ("DOI of related publication", "", False, ""),
    },
    "user": {
        "name": ("Name", "", False, ""), "affiliation": ("Affiliation", "", False, ""),
        "email": ("E-mail", "", False, ""), "orcid": ("ORCID", "", False, ""),
        "role": ("Role", "", False, "e.g. operator"),
    },
    "sample": {
        "name": ("Sample name", "", False, "e.g. (BaTiO3)12/(SrTiO3)5 superlattice on Si(001)"),
        "atom_types": ("Elements (atom_types)", "Ba, Sr, Ti, O", True, "comma-separated"),
        "preparation_date": ("Preparation date (ISO 8601)", "", True, ""),
        "physical_form": ("Physical form", "thin_film", False, ""),
        "identifier": ("Sample identifier", "", False, ""),
        "identifier_type": ("Identifier type", "local", False, ""),
        "thickness_nm": ("Lamella thickness (nm)", "", False, ""),
        "description": ("Sample description", "", False, "e.g. FIB cross-section along [110]"),
        "is_simulation": ("Simulated data", False, True, ""),
    },
    "instrument": {
        "name": ("Microscope name", "Nion HERMES", False, ""), "location": ("Location", "", False, ""),
        "vendor": ("Vendor", "Nion", True, ""), "model": ("Model", "HERMES", True, ""),
        "serial_number": ("Serial number", "", False, ""),
        "emitter_type": ("Emitter type", "cold_field_emission", True, ""),
        "detector": ("Detector", "", False, "e.g. Dectris ELA (counting)"),
        "software": ("Acquisition software", "Nion Swift", False, ""),
    },
}


def default_metadata():
    return {s: {k: v[1] for k, v in fs.items()} for s, fs in METADATA_FIELDS.items()}


def merge_meta(meta):
    m = default_metadata()
    for s, d in (meta or {}).items():
        if s in m and isinstance(d, dict):
            m[s].update({k: v for k, v in d.items() if v is not None})
    return m


def _grp(parent, name, nxclass):
    g = parent.require_group(name)
    g.attrs["NX_class"] = nxclass
    return g


def _ds(parent, name, data, units=None, **attrs):
    if name in parent:
        del parent[name]
    if isinstance(data, np.ndarray) and data.size > 4096 and data.dtype.kind in "fciub":
        d = parent.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)
    else:
        d = parent.create_dataset(name, data=data)
    if units:
        d.attrs["units"] = units
    for k, v in attrs.items():
        d.attrs[k] = v
    return d


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _iso(s, fallback=None):
    s = str(s or "").strip()
    if not s:
        return fallback
    try:
        t = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            t = _dt.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            return s
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.isoformat()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 10000 else f"array{o.shape}"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _image_nxdata(g, arr, step_nm, title, units, long_name, name="image_2d"):
    d = _grp(g, name, "NXdata")
    ny, nx = arr.shape
    # 'real' is unitless in NXem (NX_UNITLESS): the unit is given in long_name and in the process parameters
    ln = f"{long_name} ({units})" if units else long_name
    _ds(d, "real", np.asarray(arr, np.float32), None, long_name=ln)
    d.attrs["signal"] = "real"
    d.attrs["axes"] = ["axis_j", "axis_i"]
    d.attrs["axis_j_indices"] = np.uint64(0)
    d.attrs["axis_i_indices"] = np.uint64(1)
    _ds(d, "title", title)
    _ds(d, "axis_i", np.arange(nx) * step_nm[1], "nm", long_name="scan x (nm)")
    _ds(d, "axis_j", np.arange(ny) * step_nm[0], "nm", long_name="scan y (nm)")
    return d


def _pattern_nxdata(g, arr, mrad_per_px, title):
    d = _grp(g, "image_2d", "NXdata")
    ny, nx = arr.shape
    _ds(d, "real", np.asarray(arr, np.float32), None, long_name="mean diffraction pattern (counts)")
    d.attrs["signal"] = "real"
    d.attrs["axes"] = ["axis_j", "axis_i"]
    d.attrs["axis_j_indices"] = np.uint64(0)
    d.attrs["axis_i_indices"] = np.uint64(1)
    _ds(d, "title", title)
    s = mrad_per_px or 1.0
    u = "mrad" if mrad_per_px else None
    _ds(d, "axis_i", (np.arange(nx) - nx / 2) * s, u, long_name="detector x")
    _ds(d, "axis_j", (np.arange(ny) - ny / 2) * s, u, long_name="detector y")
    return d


def write_results(path, maps, step_nm, metadata=None, energy_eV=60e3, convergence_mrad=None, calibration=None,
                  options=None, inputs=(), mean_pattern=None, mrad_per_px=None, description=None, extra=None):
    """Write maps {name: (array, units, long_name, method)} into a new NXem file."""
    meta = merge_meta(metadata)
    if meta["sample"].get("is_simulation"):
        for k, v in (("vendor", meta["instrument"].get("vendor") or "simulation"),
                     ("model", meta["instrument"].get("model") or "fit4dstem.synthetic")):
            meta["instrument"][k] = v
    with h5py.File(path, "w") as f:
        f.attrs["default"] = "entry1"
        f.attrs["creator"] = f"fit4dstem {__version__}"
        f.attrs["NX_class"] = "NXroot"
        e = _grp(f, "entry1", "NXentry")
        _ds(e, "definition", "NXem")
        ex = meta["experiment"]
        _ds(e, "start_time", _iso(ex.get("start_time"), _now()))
        _ds(e, "end_time", _iso(ex.get("end_time"), _iso(ex.get("start_time"), _now())))
        if ex.get("identifier"):
            _ds(e, "identifier_experiment", str(ex["identifier"]))
        _ds(e, "experiment_description", str(ex.get("description") or description or
                                             "4D-STEM (CBED) maps of thickness, tilt, Ti displacement and strain"))
        if ex.get("doi"):
            c = _grp(e, "cite1", "NXcite")
            _ds(c, "doi", str(ex["doi"]))
        prof = _grp(e, "profiling", "NXcs_profiling")
        prog = _grp(prof, "program1", "NXprogram")
        _ds(prog, "program", "fit4dstem", version=__version__)
        u = meta["user"]
        if any(str(u.get(k, "")).strip() for k in ("name", "affiliation", "email", "orcid")):
            ug = _grp(e, "user1", "NXuser")
            for k in ("name", "affiliation", "email", "role"):
                if str(u.get(k, "")).strip():
                    _ds(ug, k, str(u[k]))
            if str(u.get("orcid", "")).strip():
                _ds(ug, "identifier_orcid", str(u["orcid"]).strip(), type="ORCID")
        s = meta["sample"]
        sg = _grp(e, "sample1", "NXsample")
        _ds(sg, "is_simulation", bool(s.get("is_simulation", False)))
        _ds(sg, "atom_types", ", ".join(a.strip() for a in str(s.get("atom_types") or "").replace(";", ",").split(",")
                                        if a.strip()) or "unknown")
        _ds(sg, "preparation_date", _iso(s.get("preparation_date"), _iso(ex.get("start_time"), _now())))
        for k in ("name", "description", "physical_form"):
            if str(s.get(k, "")).strip():
                _ds(sg, k, str(s[k]))
        if str(s.get("identifier", "")).strip():
            t = str(s.get("identifier_type") or "local").strip()
            known = {x.lower(): x for x in IDENTIFIER_TYPES}
            d = _ds(sg, "identifier_sample", str(s["identifier"]), type=known.get(t.lower(), t))
            if t.lower() not in known:
                d.attrs["type_custom"] = True
        if str(s.get("thickness_nm", "")).strip():
            try:
                _ds(sg, "thickness", float(s["thickness_nm"]), "nm")
            except ValueError:
                pass
        im = meta["instrument"]
        m = _grp(e, "measurement", "NXem_measurement")
        ins = _grp(m, "instrument", "NXem_instrument")
        _ds(ins, "type", "tem")
        for k in ("name", "location"):
            if str(im.get(k, "")).strip():
                _ds(ins, k, str(im[k]))
        fab = _grp(ins, "fabrication", "NXfabrication")
        _ds(fab, "vendor", str(im.get("vendor") or "unknown"))
        _ds(fab, "model", str(im.get("model") or "unknown"))
        if str(im.get("serial_number", "")).strip():
            _ds(fab, "serial_number", str(im["serial_number"]))
        if str(im.get("software", "")).strip():
            pg = _grp(ins, "program1", "NXprogram")
            sw = str(im["software"]).strip()
            name, _, ver = sw.partition(" ")
            _ds(pg, "program", name, version=ver or "unknown")
        col = _grp(ins, "ebeam_column", "NXebeam_column")
        src = _grp(col, "electron_source", "NXsource")
        _ds(src, "voltage", float(energy_eV), "V")
        _ds(src, "emitter_type", str(im.get("emitter_type") or "unknown"))
        _ds(src, "probe", "electron")
        if convergence_mrad:
            opt = _grp(ins, "optics", "NXem_optical_system")
            _ds(opt, "semi_convergence_angle", float(convergence_mrad) * 1e-3, "rad")
        if str(im.get("detector", "")).strip():
            det = _grp(ins, "detector1", "NXdetector")
            _ds(det, "name", str(im["detector"]))
        coll = _grp(ins, "fit4dstem", "NXcollection")
        if calibration is not None:
            _ds(coll, "calibration", json.dumps(calibration, default=_json_default))
        if options is not None:
            _ds(coll, "options", json.dumps(options, default=_json_default))
        for k, v in (extra or {}).items():
            _ds(coll, k, json.dumps(v, default=_json_default) if not isinstance(v, str) else v)
        for i, inp in enumerate(inputs):
            n = _grp(m, f"input{i + 1}", "NXnote")
            _ds(n, "type", "application/x-hdf5")
            _ds(n, "file_name", str(inp.get("file", "")))
            _ds(n, "description", str(inp.get("description", "")))
        if mean_pattern is not None:
            ev = _grp(m, "event1", "NXem_event_data")
            img = _grp(ev, "image1", "NXimage")
            _pattern_nxdata(img, mean_pattern, mrad_per_px, "mean diffraction pattern (coarse scan sample)")
        roi = _grp(e, "roi1", "NXroi_process")
        imgs = _grp(roi, "img", "NXem_img")
        for k, (name, item) in enumerate(maps.items(), start=1):
            arr, units, long_name, method = (list(item) + [None, None, None])[:4]
            g = _grp(imgs, f"image{k}", "NXimage")
            _ds(g, "imaging_mode", IMAGING_MODE, custom=True)
            pr = _grp(g, "reconstruction", "NXprocess")
            _ds(pr, "program", "fit4dstem")
            _ds(pr, "version", __version__)
            _ds(pr, "date", _now())
            _ds(pr, "sequence_index", np.uint64(k))
            pa = _grp(pr, "parameters", "NXparameters")
            _ds(pa, "quantity", str(name))
            _ds(pa, "units", str(units or ""))
            _ds(pa, "method", str(method or ""))
            _image_nxdata(g, np.asarray(arr, float), step_nm, f"{name}", units, long_name or name)
    return path


def validate(path, ignore_undocumented=False):
    """Validate with pynxtools.  Returns (is_valid, problems, undocumented) or (None, [message], [])."""
    try:
        from pynxtools.dataconverter.validation import validate_hdf_group_against
    except ImportError:
        return None, ["pynxtools is not installed (pip install pynxtools pynxtools-em)"], []
    msgs = []

    class H(logging.Handler):
        def emit(self, record):
            msgs.append((record.levelname, record.getMessage()))

    h = H()
    lg = logging.getLogger("pynxtools")
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    ok = True
    try:
        with h5py.File(path, "r") as f:
            for name, g in f.items():
                if isinstance(g, h5py.Group) and g.attrs.get("NX_class") in ("NXentry", b"NXentry"):
                    d = g.get("definition")
                    v = d[()] if d is not None else None
                    nxdl = v.decode() if isinstance(v, bytes) else v
                    if nxdl:
                        ok &= bool(validate_hdf_group_against(nxdl, g, path, ignore_undocumented))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    undoc = [m for lv, m in msgs if "has no documentation" in m]
    problems = [f"{lv}: {m}" for lv, m in msgs if "has no documentation" not in m and "is valid according" not in m
                and lv in ("WARNING", "ERROR", "CRITICAL")]
    return ok, problems, undoc


def file_info(path):
    try:
        st = os.stat(path)
        return dict(file=os.path.abspath(path), size_bytes=int(st.st_size),
                    modified=_dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat())
    except OSError:
        return dict(file=str(path))
