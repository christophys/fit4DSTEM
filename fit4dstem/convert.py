# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Convert Nion Swift 4D-STEM exports (``<name>.npy`` + ``<name>.json``, or both in ``<name>.zip``) to NeXus/NXem.

    python -m fit4dstem.convert data_folder [--out nxs] [--atom-types "Ba, Sr, Ti, O"] [--sample-name ...]
    python -m fit4dstem.convert file.npy|file.zip ...

The data are streamed row by row (never fully in memory; a zip is decompressed on the fly), stored as unsigned
integers when all values are integer counts (lossless, uint16 if the maximum fits, else uint32; float32 otherwise),
chunked per scan position and gzip-compressed.  Layout (pynxtools-valid NXem)::

  /entry1  NXentry  definition = NXem, start_time, end_time
    profiling/program1                           NXprogram  fit4dstem convert
    sample1                                      NXsample
    measurement                                  NXem_measurement
      instrument                                 NXem_instrument  type, fabrication, ebeam_column/electron_source
                                                                  (voltage), optics (semi_convergence_angle), detector1
      instrument/nion_metadata                   NXcollection  original Nion JSON (complete)
      event1                                     NXem_event_data  start_time, end_time
        image1                                   NXimage
          image_4d                               NXdata  real (counts: scan y, scan x, detector y, detector x),
                                                         axis_m, axis_k (scan, nm), axis_j, axis_i (detector, px)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import zipfile

import numpy as np

try:
    from . import __version__
except ImportError:  # run as a stand-alone script
    __version__ = "standalone"

EXTS = (".npy", ".zip")


def read_npy_header(fh):
    """(shape, fortran_order, dtype) of a .npy stream positioned at its start (public numpy API only)."""
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fh)
    return np.lib.format.read_array_header_2_0(fh)


# ====================================================================== sources ==
class NpySource:
    """Row-wise reader of a .npy file (memory map) or of a .npy inside a zip (streamed decompression)."""

    def __init__(self, path):
        self.path = path
        self.zip = None
        self.meta = {}
        base = os.path.splitext(path)[0]
        if path.lower().endswith(".zip"):
            self.zip = zipfile.ZipFile(path)
            names = self.zip.namelist()
            npy = [n for n in names if n.lower().endswith(".npy")]
            js = [n for n in names if n.lower().endswith(".json")]
            if len(npy) != 1:
                raise ValueError(f"{path}: expected one .npy in the zip, found {npy}")
            self.member = npy[0]
            if js:
                self.meta = json.loads(self.zip.read(js[0]))
            with self.zip.open(self.member) as fh:
                self.shape, self.fortran, self.dtype = read_npy_header(fh)
        else:
            if os.path.exists(base + ".json"):
                with open(base + ".json") as f:
                    self.meta = json.load(f)
            self.arr = np.load(path, mmap_mode="r")
            self.shape, self.fortran, self.dtype = self.arr.shape, not self.arr.flags["C_CONTIGUOUS"], self.arr.dtype
        if self.fortran:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        if len(self.shape) != 4:
            raise ValueError(f"{path}: not a 4D data set ({self.shape})")

    def rows(self, start=0):
        """Yield (iy, row array (sx, ky, kx)) from row ``start`` on (a zip is decompressed from the beginning)."""
        sy = self.shape[0]
        if self.zip is None:
            for iy in range(start, sy):
                yield iy, np.asarray(self.arr[iy])
            return
        row_items = int(np.prod(self.shape[1:]))
        nbytes = row_items * self.dtype.itemsize
        with self.zip.open(self.member) as fh:
            read_npy_header(fh)
            for iy in range(sy):
                buf = bytearray(nbytes)
                mv = memoryview(buf)
                got = 0
                while got < nbytes:
                    n = fh.readinto(mv[got:])
                    if not n:
                        raise IOError(f"{self.path}: unexpected end of data in row {iy}")
                    got += n
                if iy >= start:
                    yield iy, np.frombuffer(buf, self.dtype).reshape(self.shape[1:])

    def close(self):
        if self.zip is not None:
            self.zip.close()


def find_datasets(folder):
    """Distinct data sets in a folder: <name>.npy (+ .json) preferred over <name>.zip of the same name."""
    files = sorted(os.listdir(folder))
    names = {}
    for f in files:
        b, e = os.path.splitext(f)
        if e.lower() == ".npy":
            names[b] = os.path.join(folder, f)
    for f in files:
        b, e = os.path.splitext(f)
        if e.lower() == ".zip" and b not in names:
            try:
                with zipfile.ZipFile(os.path.join(folder, f)) as z:
                    if any(n.lower().endswith(".npy") for n in z.namelist()):
                        names[b] = os.path.join(folder, f)
            except zipfile.BadZipFile:
                pass
    return [names[k] for k in sorted(names)]


# ====================================================================== metadata ==
def _find(d, keys):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in keys and not isinstance(v, (dict, list)):
                return v
            r = _find(v, keys)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def _iso(s):
    if not s:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.isoformat()


def _to_nm(scale, units):
    u = (units or "").strip()
    return {"nm": 1.0, "m": 1e9, "um": 1e3, "µm": 1e3, "A": 0.1, "Å": 0.1}.get(u, None), scale


def nion_summary(meta, shape):
    md = meta.get("metadata", {})
    cal = meta.get("spatial_calibrations") or meta.get("dimensional_calibrations") or []
    s = {}
    s["voltage_V"] = _find(md, ("high_tension", "EHT"))
    ha = _find(md, ("probe_ha",))
    s["semi_convergence_angle_rad"] = float(ha) if ha is not None else None
    s["detector"] = _find(md, ("description",)) or _find(md, ("hardware_source_name",))
    s["start_time"] = _iso(_find(md, ("data_collection_date",)))
    s["exposure_s"] = _find(md.get("hardware_source", {}), ("exposure",))
    s["counts_per_electron"] = _find(md, ("counts_per_electron",))
    s["fov_nm"] = _find(md.get("scan", {}), ("fov_nm",))
    s["scan_rotation_deg"] = _find(md.get("scan", {}), ("rotation_deg",))
    s["defocus_m"] = _find(md.get("instrument", {}), ("defocus",))
    axes = []
    for i in range(4):
        c = cal[i] if i < len(cal) else {}
        axes.append((float(c.get("offset", 0.0)), float(c.get("scale", 1.0)), str(c.get("units", ""))))
    s["axes"] = axes
    return s


# ====================================================================== writer ==
def _grp(parent, name, nxclass):
    g = parent.require_group(name)
    g.attrs["NX_class"] = nxclass
    return g


def _ds(parent, name, data, units=None, **attrs):
    d = parent.create_dataset(name, data=data)
    if units:
        d.attrs["units"] = units
    for k, v in attrs.items():
        d.attrs[k] = v
    return d


def convert(src_path, out_path, atom_types="unknown", sample_name=None, instrument_model="HERMES",
            instrument_vendor="Nion", emitter="cold_field_emission", compression=4, dtype="auto", log=print,
            max_seconds=None):
    """Convert one data set.  dtype: 'auto' (integer counts -> uint16/uint32, else float32), 'uint16', 'float32'."""
    import h5py
    t0 = time.time()
    src = NpySource(src_path)
    sy, sx, ky, kx = (int(v) for v in src.shape)
    meta = src.meta
    S = nion_summary(meta, src.shape)
    # 'auto': optimistic single pass as uint16 (lossless for integer counts <= 65535); if a value is not an
    # integer count or too large, the conversion restarts with uint32 or float32
    if dtype == "auto":
        out_dtype = np.dtype(np.uint16) if np.dtype(src.dtype).kind in "fiu" else np.dtype(src.dtype)
    else:
        out_dtype = np.dtype(dtype)
    tmp = out_path + ".part"
    if dtype == "auto" and os.path.exists(tmp):        # resume with the type chosen in the earlier call
        try:
            import h5py
            with h5py.File(tmp, "r") as f:
                if "fit4dstem_rows_done" in f.attrs:
                    out_dtype = f["entry1/measurement/event1/image1/image_4d/real"].dtype
        except Exception:  # noqa: BLE001
            pass
    while True:
        try:
            return _write(src, src_path, out_path, out_dtype, dtype == "auto", S, meta, atom_types, sample_name,
                          instrument_model, instrument_vendor, emitter, compression, log, t0, max_seconds)
        except _Retry as r:
            log(f"  {os.path.basename(src_path)}: {r.reason} -> restarting as {np.dtype(r.dtype).name}")
            out_dtype = np.dtype(r.dtype)
            src.close()
            src = NpySource(src_path)


class Paused(Exception):
    """Raised when the time budget is used up; the conversion resumes from the .part file on the next call."""


class _Retry(Exception):
    def __init__(self, dtype, reason):
        super().__init__(reason)
        self.dtype, self.reason = dtype, reason


def _write(src, src_path, out_path, out_dtype, auto, S, meta, atom_types, sample_name, instrument_model,
           instrument_vendor, emitter, compression, log, t0, max_seconds=None):
    import h5py
    sy, sx, ky, kx = (int(v) for v in src.shape)
    tmp = out_path + ".part"
    start_row, total, vmax = 0, 0.0, 0.0
    f = None
    if os.path.exists(tmp):
        try:
            f = h5py.File(tmp, "r+")
            ds = f["entry1/measurement/event1/image1/image_4d/real"]
            if ds.shape == (sy, sx, ky, kx) and ds.dtype == out_dtype and "fit4dstem_rows_done" in f.attrs:
                start_row = int(f.attrs["fit4dstem_rows_done"])
                total = float(f.attrs["fit4dstem_total"])
                vmax = float(f.attrs["fit4dstem_vmax"])
                log(f"  {os.path.basename(src_path)}: resuming at row {start_row + 1}/{sy}")
            else:
                f.close()
                f = None
        except Exception:  # noqa: BLE001
            if f is not None:
                f.close()
            f = None
    if f is None:
        f = h5py.File(tmp, "w")
        _create(f, sy, sx, ky, kx, out_dtype, src_path, S, meta, atom_types, sample_name, instrument_model,
                instrument_vendor, emitter, compression)
        f.attrs["fit4dstem_rows_done"] = 0
        f.attrs["fit4dstem_total"] = 0.0
        f.attrs["fit4dstem_vmax"] = 0.0
    ds = f["entry1/measurement/event1/image1/image_4d/real"]
    t1 = time.time()
    done_now = 0
    try:
        for iy, row in src.rows(start_row):
            if out_dtype.kind == "u" and np.dtype(src.dtype).kind == "f":
                if row.min() < 0 or not np.array_equal(row, np.floor(row)):
                    if not auto:
                        raise ValueError("values are not non-negative integers")
                    raise _Retry(np.float32, "values are not integer counts")
            if out_dtype.kind == "u":
                vmax = max(vmax, float(row.max()))
                if vmax > np.iinfo(out_dtype).max:
                    if auto and out_dtype == np.uint16:
                        raise _Retry(np.uint32, f"maximum {vmax:.0f} > 65535")
                    raise ValueError(f"value {vmax} exceeds {out_dtype}")
            ds[iy] = row.astype(out_dtype, copy=False)
            total += float(row.sum(dtype=np.float64))
            done_now += 1
            f.attrs["fit4dstem_rows_done"] = iy + 1
            f.attrs["fit4dstem_total"] = total
            f.attrs["fit4dstem_vmax"] = vmax
            if iy % 20 == 0 or iy == sy - 1:
                el = time.time() - t1
                log(f"  {os.path.basename(src_path)}: row {iy + 1}/{sy}, {el:.0f} s, about "
                    f"{el / done_now * (sy - iy - 1):.0f} s left")
            if max_seconds and time.time() - t0 > max_seconds and iy + 1 < sy:
                f.flush()
                raise Paused(f"{os.path.basename(src_path)}: paused after row {iy + 1}/{sy}")
        coll = f["entry1/measurement/instrument/nion_metadata"]
        coll.create_dataset("total_counts", data=total)
        coll.create_dataset("source", data=os.path.basename(src_path))
        coll.create_dataset("source_dtype", data=str(src.dtype))
        for k in ("fit4dstem_rows_done", "fit4dstem_total", "fit4dstem_vmax"):
            del f.attrs[k]
    finally:
        f.close()
    src.close()
    os.replace(tmp, out_path)
    log(f"  -> {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB, {time.time() - t0:.0f} s)")
    return out_path


def _create(f, sy, sx, ky, kx, out_dtype, src_path, S, meta, atom_types, sample_name, instrument_model,
            instrument_vendor, emitter, compression):
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    f.attrs["NX_class"] = "NXroot"
    f.attrs["default"] = "entry1"
    f.attrs["creator"] = f"fit4dstem.convert {__version__}"
    e = _grp(f, "entry1", "NXentry")
    e.attrs["default"] = "measurement"
    _ds(e, "definition", "NXem")
    start = S["start_time"] or now
    _ds(e, "start_time", start)
    _ds(e, "end_time", start)
    _ds(e, "experiment_description", f"4D-STEM scan {os.path.basename(os.path.splitext(src_path)[0])} "
                                     f"(Nion Swift export converted with fit4dstem)")
    prog = _grp(_grp(e, "profiling", "NXcs_profiling"), "program1", "NXprogram")
    _ds(prog, "program", "fit4dstem.convert", version=str(__version__))
    smp = _grp(e, "sample1", "NXsample")
    _ds(smp, "is_simulation", False)
    _ds(smp, "atom_types", atom_types)
    _ds(smp, "preparation_date", start)
    if sample_name:
        _ds(smp, "name", sample_name)
    m = _grp(e, "measurement", "NXem_measurement")
    m.attrs["default"] = "event1"
    ins = _grp(m, "instrument", "NXem_instrument")
    _ds(ins, "type", "tem")
    fab = _grp(ins, "fabrication", "NXfabrication")
    _ds(fab, "vendor", instrument_vendor)
    _ds(fab, "model", instrument_model)
    col = _grp(ins, "ebeam_column", "NXebeam_column")
    es = _grp(col, "electron_source", "NXsource")
    if S["voltage_V"] is not None:
        _ds(es, "voltage", float(S["voltage_V"]), "V")
    _ds(es, "emitter_type", emitter)
    _ds(es, "probe", "electron")
    if S["semi_convergence_angle_rad"] is not None:
        opt = _grp(ins, "optics", "NXem_optical_system")
        _ds(opt, "semi_convergence_angle", S["semi_convergence_angle_rad"], "rad")
    if S["detector"]:
        det = _grp(ins, "detector1", "NXdetector")
        _ds(det, "name", str(S["detector"]))
    coll = _grp(ins, "nion_metadata", "NXcollection")
    _ds(coll, "json", json.dumps(meta))
    _ds(coll, "summary", json.dumps({k: v for k, v in S.items()}))
    ev = _grp(m, "event1", "NXem_event_data")
    ev.attrs["default"] = "image1"
    _ds(ev, "start_time", start)
    _ds(ev, "end_time", start)
    img = _grp(ev, "image1", "NXimage")
    img.attrs["default"] = "image_4d"
    d = _grp(img, "image_4d", "NXdata")
    _ds(d, "title", "4D-STEM: scan y, scan x, detector y, detector x")
    d.attrs["signal"] = "real"
    d.attrs["axes"] = ["axis_m", "axis_k", "axis_j", "axis_i"]
    for i, a in enumerate(("axis_m", "axis_k", "axis_j", "axis_i")):
        d.attrs[f"{a}_indices"] = np.uint64(i)
    ds = d.create_dataset("real", shape=(sy, sx, ky, kx), dtype=out_dtype,
                          chunks=(1, min(sx, 16), ky, kx), compression="gzip", compression_opts=compression,
                          shuffle=True)
    ds.attrs["long_name"] = "counts per pixel"
    # axes: scan in nm (from the calibration), detector in pixels (the Nion export calibrates it as EELS)
    n_scan = []
    for i, n in ((0, sy), (1, sx)):
        off, sc, u = S["axes"][i]
        f_nm, _ = _to_nm(sc, u)
        if f_nm is not None:
            n_scan.append((np.arange(n) * sc + off) * f_nm)
        else:
            n_scan.append(np.arange(n, dtype=float))
    unit_scan = "nm" if _to_nm(1, S["axes"][0][2])[0] is not None else None
    _ds(d, "axis_m", n_scan[0], unit_scan, long_name="scan y (nm)")
    _ds(d, "axis_k", n_scan[1], unit_scan, long_name="scan x (nm)")
    _ds(d, "axis_j", np.arange(ky, dtype=float), None, long_name="detector y (pixel)")
    _ds(d, "axis_i", np.arange(kx, dtype=float), None, long_name="detector x (pixel)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", help="folders or .npy / .zip files")
    p.add_argument("--out", default=None, help="output folder (default: <input folder>/nxs)")
    p.add_argument("--atom-types", default="unknown")
    p.add_argument("--sample-name", default=None)
    p.add_argument("--model", default="HERMES")
    p.add_argument("--vendor", default="Nion")
    p.add_argument("--compression", type=int, default=4)
    p.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "float32"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop after this time and resume from the .part file on the next call")
    a = p.parse_args(argv)
    todo = []
    for inp in a.inputs:
        if os.path.isdir(inp):
            todo += find_datasets(inp)
        elif inp.lower().endswith(EXTS):
            todo.append(inp)
    t_start = time.time()
    for src in todo:
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(src)), "nxs")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".nxs")
        if os.path.exists(out) and not a.overwrite:
            print(f"skip (exists): {out}", flush=True)
            continue
        left = None
        if a.max_seconds:
            left = a.max_seconds - (time.time() - t_start)
            if left < 20:
                print("PAUSED (time budget used up)", flush=True)
                return 2
        print(f"converting {src}", flush=True)
        try:
            convert(src, out, a.atom_types, a.sample_name, a.model, a.vendor, compression=a.compression,
                    dtype=a.dtype, log=lambda s: print(s, flush=True), max_seconds=left)
        except Paused as p_:
            print(f"PAUSED {p_}", flush=True)
            return 2
        except Exception as ex:  # noqa: BLE001
            print(f"FAILED {src}: {ex}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
