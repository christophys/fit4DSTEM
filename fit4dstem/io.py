# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Lazy access to 4D-STEM data sets: HDF5 (any layout), NeXus/NXem, Nion Swift (.h5 / .ndata), NumPy (.npy),
other formats via RosettaSciIO (lazy), and files of a NOMAD upload read with HTTP range requests.

A :class:`Dataset4D` never loads the whole data set: frames, rows or sparse sets of scan positions are read on
demand (HDF5 chunks / memory maps / remote blocks).  Axis order is always (scan y, scan x, detector y, detector x)
("ky, kx" below: detector rows, columns).
"""
from __future__ import annotations

import json
import os
import re
import zipfile

import numpy as np

EXTENSIONS = (".h5", ".hdf5", ".hdf", ".nxs", ".nx", ".emd", ".ndata", ".npy", ".mib", ".blo", ".dm4", ".zspy",
              ".hspy")


class Dataset4D:
    """Lazy 4D-STEM data set.

    Attributes: ``shape`` (sy, sx, ky, kx), ``meta`` (energy_eV, convergence_mrad, scan_step_nm (y, x),
    detector_sampling (y, x) and detector_units, source, layout notes), ``name``.
    ``det_bin``: detector binning applied when reading (sums b x b pixels).
    """

    def __init__(self, array, meta=None, name="", closer=None, det_bin=1):
        self._a = array
        self.meta = dict(meta or {})
        self.name = name
        self._closer = closer
        self.det_bin = int(det_bin)
        if len(array.shape) != 4:
            raise ValueError(f"expected a 4D data set (scan y, scan x, detector y, detector x), got {array.shape}")

    # ------------------------------------------------------------------ shape
    @property
    def raw_shape(self):
        return tuple(int(s) for s in self._a.shape)

    @property
    def shape(self):
        sy, sx, ky, kx = self.raw_shape
        b = self.det_bin
        return (sy, sx, ky // b, kx // b)

    @property
    def scan_shape(self):
        return self.shape[:2]

    @property
    def det_shape(self):
        return self.shape[2:]

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def nbytes(self):
        return int(np.prod(self.raw_shape)) * np.dtype(self.dtype).itemsize

    def _bin(self, a):
        b = self.det_bin
        if b == 1:
            return np.asarray(a, np.float32)
        ky, kx = a.shape[-2:]
        a = np.asarray(a, np.float32)[..., :ky // b * b, :kx // b * b]
        return a.reshape(a.shape[:-2] + (ky // b, b, kx // b, b)).sum(axis=(-3, -1))

    # ------------------------------------------------------------------ reading
    def frame(self, iy, ix):
        return self._bin(self._a[int(iy), int(ix)])

    def frames(self, positions):
        """Frames at a list of (iy, ix) scan positions: array [n, ky, kx] (read row by row)."""
        positions = [(int(a), int(b)) for a, b in positions]
        out = np.empty((len(positions),) + self.det_shape, np.float32)
        by_row = {}
        for k, (iy, ix) in enumerate(positions):
            by_row.setdefault(iy, []).append((ix, k))
        for iy, lst in by_row.items():
            xs = sorted(set(ix for ix, _ in lst))
            if len(xs) > 0.3 * self.shape[1] and len(xs) > 4:
                row = self._bin(self._a[iy, xs[0]:xs[-1] + 1])
                for ix, k in lst:
                    out[k] = row[ix - xs[0]]
            else:
                for ix, k in lst:
                    out[k] = self._bin(self._a[iy, ix])
        return out

    def rows(self, y0, y1, x0=0, x1=None):
        """Block of frames [y1 - y0, x1 - x0, ky, kx] (float32)."""
        x1 = self.shape[1] if x1 is None else x1
        return self._bin(self._a[int(y0):int(y1), int(x0):int(x1)])

    def iter_rows(self, block=4):
        for y0 in range(0, self.shape[0], block):
            y1 = min(self.shape[0], y0 + block)
            yield y0, y1, self.rows(y0, y1)

    def mean_frame(self, positions):
        return self.frames(positions).mean(0)

    def close(self):
        if self._closer is not None:
            try:
                self._closer()
            except Exception:  # noqa: BLE001
                pass
            self._closer = None

    def describe(self):
        m = self.meta
        gb = self.nbytes / 1e9
        s = (f"{self.name}: scan {self.shape[0]} x {self.shape[1]}, detector {self.raw_shape[2]} x "
             f"{self.raw_shape[3]}" + (f" (binned {self.det_bin}: {self.shape[2]} x {self.shape[3]})"
                                       if self.det_bin > 1 else "") + f", {self.dtype}, {gb:.2f} GB")
        if m.get("energy_eV"):
            s += f", {m['energy_eV'] / 1e3:.0f} kV"
        if m.get("convergence_mrad"):
            s += f", alpha {m['convergence_mrad']:.2f} mrad"
        if m.get("scan_step_nm"):
            s += f", scan step {m['scan_step_nm'][0]:.3g} nm"
        if m.get("detector_sampling"):
            s += f", detector {m['detector_sampling'][0]:.4g} {m.get('detector_units', '')}/px"
        return s


# ====================================================================== scan sampling ==
def coarse_positions(scan_shape, n=12, include_edges=True):
    """Regular coarse grid of about n x n scan positions covering the whole map: list of (iy, ix)."""
    sy, sx = scan_shape
    ny, nx = min(n, sy), min(n, sx)
    if include_edges:
        ys = np.unique(np.round(np.linspace(0, sy - 1, ny)).astype(int))
        xs = np.unique(np.round(np.linspace(0, sx - 1, nx)).astype(int))
    else:
        ys = np.unique(((np.arange(ny) + 0.5) * sy / ny).astype(int))
        xs = np.unique(((np.arange(nx) + 0.5) * sx / nx).astype(int))
    return [(int(y), int(x)) for y in ys for x in xs]


# ====================================================================== metadata helpers ==
def _walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            yield from _walk(v, f"{path}/{i}")
    else:
        yield path, d


def _find_number(meta, patterns):
    for p, v in _walk(meta):
        low = p.lower()
        if any(re.search(pt, low) for pt in patterns):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _units_scale(units, value):
    """Convert a calibration to nm (scan) or mrad / (1/nm) (detector)."""
    u = (units or "").strip()
    if u in ("m",):
        return value * 1e9, "nm"
    if u in ("um", "µm"):
        return value * 1e3, "nm"
    if u in ("A", "Å", "angstrom"):
        return value * 0.1, "nm"
    if u == "nm":
        return value, "nm"
    if u in ("rad",):
        return value * 1e3, "mrad"
    if u in ("mrad",):
        return value, "mrad"
    if u in ("1/nm", "nm^-1", "1 / nm"):
        return value, "1/nm"
    if u in ("1/A", "1/Å", "Å^-1"):
        return value * 10, "1/nm"
    if u in ("1/m", "m^-1"):
        return value * 1e-9, "1/nm"
    return value, u


def nion_metadata(props):
    """Metadata of a Nion Swift data item (properties dict)."""
    meta = {"source_format": "Nion Swift"}
    cal = props.get("dimensional_calibrations") or []
    if len(cal) >= 4:
        sy, unit_s = _units_scale(cal[0].get("units"), float(cal[0].get("scale", 1)))
        sx, _ = _units_scale(cal[1].get("units"), float(cal[1].get("scale", 1)))
        dy, unit_d = _units_scale(cal[2].get("units"), float(cal[2].get("scale", 1)))
        dx, _ = _units_scale(cal[3].get("units"), float(cal[3].get("scale", 1)))
        if unit_s == "nm":
            meta["scan_step_nm"] = (sy, sx)
        if unit_d:
            meta["detector_sampling"] = (dy, dx)
            meta["detector_units"] = unit_d
    md = props.get("metadata") or {}
    ht = _find_number(md, [r"high_tension", r"/eht$", r"/ht$", r"voltage"])
    if ht:
        meta["energy_eV"] = ht if ht > 1000 else ht * 1e3
    ca = _find_number(md, [r"convergence", r"probe_ha", r"/ca$", r"semi.?angle"])
    if ca:
        meta["convergence_mrad"] = ca * 1e3 if ca < 0.5 else ca
    meta["nion_properties"] = props
    return meta


# ====================================================================== openers ==
def _h5_4d_datasets(f):
    """All 4D datasets of an h5py file: list of (path, shape)."""
    import h5py
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
            out.append(("/" + name, obj.shape))
    f.visititems(visit)
    return out


def _nxdata_4d(f):
    """NXdata groups with a 4D signal: list of (signal path, group path)."""
    import h5py
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            cls = obj.attrs.get("NX_class")
            cls = cls.decode() if isinstance(cls, bytes) else cls
            if cls == "NXdata":
                sig = obj.attrs.get("signal")
                sig = sig.decode() if isinstance(sig, bytes) else sig
                if sig and sig in obj and isinstance(obj[sig], h5py.Dataset) and obj[sig].ndim == 4:
                    out.append(("/" + name + "/" + sig, "/" + name))
    f.visititems(visit)
    return out


def _nx_axes_meta(f, group):
    """Calibrations of an NXdata group (axes with units)."""
    g = f[group]
    axes = g.attrs.get("axes")
    if axes is None:
        return {}
    axes = [a.decode() if isinstance(a, bytes) else str(a) for a in np.atleast_1d(axes)]
    meta = {}
    vals = []
    for a in axes:
        if a in g and g[a].ndim == 1 and g[a].shape[0] > 1:
            arr = g[a][:2]
            u = g[a].attrs.get("units", "")
            u = u.decode() if isinstance(u, bytes) else u
            vals.append(_units_scale(u, float(arr[1] - arr[0])))
        else:
            vals.append((None, None))
    if len(vals) == 4:
        if vals[0][1] == "nm":
            meta["scan_step_nm"] = (vals[0][0], vals[1][0])
        if vals[2][1] in ("mrad", "1/nm"):
            meta["detector_sampling"] = (vals[2][0], vals[3][0])
            meta["detector_units"] = vals[2][1]
    return meta


def _nx_instrument_meta(f):
    meta = {}
    found = {}

    def visit(name, obj):
        low = name.lower()
        try:
            if hasattr(obj, "shape") and obj.shape in ((), (1,)):
                v = obj[()]
                v = float(np.asarray(v).ravel()[0])
                u = obj.attrs.get("units", "")
                u = u.decode() if isinstance(u, bytes) else u
                if low.endswith("voltage") and "energy_eV" not in found:
                    found["energy_eV"] = v * (1e3 if u.lower() == "kv" else 1)
                if ("semi_convergence_angle" in low or low.endswith("convergence_angle")) and \
                        "convergence_mrad" not in found:
                    found["convergence_mrad"] = v * (1e3 if u in ("rad",) else (1.0 if u in ("mrad", "") else 1))
        except Exception:  # noqa: BLE001
            pass
    f.visititems(visit)
    meta.update(found)
    return meta


def open_hdf5(fileobj_or_path, dataset=None, name="", det_bin=1):
    """Open a 4D data set of an HDF5 / NeXus / Nion .h5 file lazily.  ``dataset``: path of the 4D dataset
    (default: the NXdata signal, else the Nion 'data' dataset, else the largest 4D dataset)."""
    import h5py
    f = h5py.File(fileobj_or_path, "r")
    meta = {"source_file": name or str(fileobj_or_path)}
    path = dataset
    if path is None:
        nx = _nxdata_4d(f)
        if nx:
            path = nx[0][0]
            meta["source_format"] = "NeXus NXdata"
            meta.update(_nx_axes_meta(f, nx[0][1]))
            meta.update(_nx_instrument_meta(f))
        elif "data" in f and isinstance(f["data"], h5py.Dataset) and f["data"].ndim == 4 \
                and "properties" in f["data"].attrs:
            path = "/data"
            props = json.loads(f["data"].attrs["properties"])
            meta.update(nion_metadata(props))
        else:
            cands = _h5_4d_datasets(f)
            if not cands:
                f.close()
                raise ValueError("no 4D dataset in this file")
            path = max(cands, key=lambda c: int(np.prod(c[1])))[0]
            meta["source_format"] = "HDF5"
            meta.update(_nx_instrument_meta(f))
    meta["dataset_path"] = path
    ds = f[path]
    if ds.ndim != 4:
        f.close()
        raise ValueError(f"{path} is not 4D")
    return Dataset4D(ds, meta, name or os.path.basename(str(fileobj_or_path)), closer=f.close, det_bin=det_bin)


def list_4d_datasets(path):
    import h5py
    with h5py.File(path, "r") as f:
        return _h5_4d_datasets(f)


def open_ndata(path, det_bin=1):
    """Nion Swift .ndata (zip with data.npy + metadata.json); memory mapped when stored uncompressed."""
    z = zipfile.ZipFile(path)
    props = json.loads(z.read("metadata.json")) if "metadata.json" in z.namelist() else {}
    info = z.getinfo("data.npy")
    meta = nion_metadata(props)
    meta["source_file"] = path
    if info.compress_type == zipfile.ZIP_STORED:
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            hdr = fh.read(30)
            n_name = int.from_bytes(hdr[26:28], "little")
            n_extra = int.from_bytes(hdr[28:30], "little")
            data_off = info.header_offset + 30 + n_name + n_extra
            fh.seek(data_off)
            from .convert import read_npy_header
            shape, fortran, dtype = read_npy_header(fh)
            arr_off = fh.tell()
        arr = np.memmap(path, dtype=dtype, mode="r", offset=arr_off, shape=shape, order="F" if fortran else "C")
        z.close()
        return Dataset4D(arr, meta, os.path.basename(path), det_bin=det_bin)
    with z.open("data.npy") as fh:
        arr = np.load(io_bytes(fh.read()))
    z.close()
    return Dataset4D(arr, meta, os.path.basename(path), det_bin=det_bin)


def io_bytes(b):
    import io
    return io.BytesIO(b)


def open_npy(path, meta=None, det_bin=1):
    arr = np.load(path, mmap_mode="r")
    m = {"source_file": path, "source_format": "NumPy"}
    side = os.path.splitext(path)[0] + ".json"
    if os.path.exists(side):
        m.update(json.load(open(side)))
    m.update(meta or {})
    return Dataset4D(arr, m, os.path.basename(path), det_bin=det_bin)


def open_rsciio(path, det_bin=1):
    """Other formats (Merlin .mib, .blo, .dm4, .zspy, ...) via RosettaSciIO, lazily (dask)."""
    from rsciio import IO_PLUGINS  # noqa: F401
    import rsciio.utils.tools  # noqa: F401
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    import importlib
    from rsciio import IO_PLUGINS as plugins
    for p in plugins:
        if ext in [e.lower() for e in p.get("file_extensions", [])]:
            mod = importlib.import_module(p["api"])
            res = mod.file_reader(path, lazy=True)
            d = res[0]
            arr = d["data"]
            if arr.ndim != 4:
                raise ValueError(f"{path}: not 4D ({arr.shape})")
            axes = d.get("axes", [])
            meta = {"source_file": path, "source_format": "RosettaSciIO " + p["name"]}
            if len(axes) == 4:
                s0, u0 = _units_scale(axes[0].get("units"), float(axes[0].get("scale", 1)))
                s1, _ = _units_scale(axes[1].get("units"), float(axes[1].get("scale", 1)))
                s2, u2 = _units_scale(axes[2].get("units"), float(axes[2].get("scale", 1)))
                s3, _ = _units_scale(axes[3].get("units"), float(axes[3].get("scale", 1)))
                if u0 == "nm":
                    meta["scan_step_nm"] = (s0, s1)
                if u2 in ("mrad", "1/nm"):
                    meta["detector_sampling"] = (s2, s3)
                    meta["detector_units"] = u2
            return Dataset4D(_DaskWrap(arr), meta, os.path.basename(path), det_bin=det_bin)
    raise ValueError(f"no RosettaSciIO reader for .{ext}")


class _DaskWrap:
    """numpy-like indexing of a dask array (computes the requested block)."""

    def __init__(self, a):
        self.a = a
        self.shape = a.shape
        self.dtype = a.dtype

    def __getitem__(self, idx):
        return np.asarray(self.a[idx].compute())


def open_dataset(path, dataset=None, det_bin=1):
    """Open a local 4D-STEM file lazily (format from the extension / content)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return open_npy(path, det_bin=det_bin)
    if ext == ".ndata":
        return open_ndata(path, det_bin=det_bin)
    if ext in (".h5", ".hdf5", ".hdf", ".nxs", ".nx", ".emd"):
        return open_hdf5(path, dataset, os.path.basename(path), det_bin=det_bin)
    return open_rsciio(path, det_bin=det_bin)


def open_remote(client, upload_id, path, dataset=None, det_bin=1, block_size=8 << 20, max_blocks=64):
    """Open a 4D-STEM HDF5 / NeXus / Nion .h5 file of a NOMAD upload lazily (HTTP range requests)."""
    from .nomad import RemoteFile
    rf = RemoteFile(client, upload_id, path, block_size=block_size, max_blocks=max_blocks)
    ds = open_hdf5(rf, dataset, os.path.basename(path), det_bin=det_bin)
    ds.meta["source_file"] = f"{client.url} upload {upload_id}: {path}"
    ds.meta["nomad"] = {"api": client.url, "upload_id": upload_id, "path": path}
    ds.remote = rf
    return ds


# ====================================================================== writing (tests, examples) ==
def write_nion_h5(path, data, scan_step_nm=0.2, det_sampling_mrad=0.5, energy_eV=60e3, convergence_mrad=4.0):
    """Write a 4D array in the Nion Swift HDF5 layout (dataset 'data' with JSON 'properties')."""
    import h5py
    sy, sx, ky, kx = data.shape
    props = {
        "dimensional_calibrations": [
            {"offset": 0.0, "scale": scan_step_nm, "units": "nm"},
            {"offset": 0.0, "scale": scan_step_nm, "units": "nm"},
            {"offset": -ky / 2 * det_sampling_mrad / 1e3, "scale": det_sampling_mrad / 1e3, "units": "rad"},
            {"offset": -kx / 2 * det_sampling_mrad / 1e3, "scale": det_sampling_mrad / 1e3, "units": "rad"}],
        "intensity_calibration": {"offset": 0.0, "scale": 1.0, "units": "counts"},
        "metadata": {"hardware_source": {"autostem": {"high_tension": energy_eV},
                                         "probe_ha_rad": convergence_mrad / 1e3}},
        "collection_dimension_count": 2, "datum_dimension_count": 2}
    with h5py.File(path, "w") as f:
        d = f.create_dataset("data", data=data, chunks=(1, min(sx, 16), ky, kx))
        d.attrs["properties"] = json.dumps(props)
    return path
