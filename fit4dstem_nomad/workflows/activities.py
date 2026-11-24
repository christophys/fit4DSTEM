# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Activities of the fit4dstem map-fit action (all on the CPU task queue).

* ``stage_map_fit``    reads the previous analysis (NXem or ELN) from the upload and finds the files: data set,
  vacuum reference and libraries are used in place where the worker can read the upload's files, otherwise copied
* ``run_map_fit``      calibration, full pass, map fit, maps, NXem + ELN + workflow (:mod:`fit4dstem.pipeline`)
* ``publish_map_fit``  adds the results to the upload (NOMAD processes them) and removes the working folder

Long activities heartbeat every few seconds (also during long single steps) and stop when the action is cancelled.
"""
import contextvars
import os
import posixpath
import shutil
import tempfile
import threading
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .backend import get_backend
from .models import MapFitInput, MapFitResult, StagedInputs


class Heartbeat:
    """Heartbeats from a helper thread + progress messages + cancellation (raises InterruptedError)."""

    def __init__(self, every_s=5.0):
        self.every = every_s
        self.msg = "started"
        self._stop = threading.Event()
        self._ctx = contextvars.copy_context()
        self._t = threading.Thread(target=lambda: self._ctx.run(self._loop), daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                activity.heartbeat(self.msg)
            except Exception:  # noqa: BLE001
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def progress(self, stage, frac, msg=""):
        self.msg = f"{stage} {frac:.0%}: {msg}"
        if activity.is_cancelled():
            raise InterruptedError("cancelled")

    def stopped(self):
        return activity.is_cancelled()


def _safe(path):
    p = "/".join(s for s in str(path or "").replace("\\", "/").split("/") if s not in ("", "."))
    if any(s == ".." for s in p.split("/")):
        raise ApplicationError(f"invalid path {path!r}", non_retryable=True)
    return p


def _resolve_in_upload(be, uid, ref, base):
    """A file named in the ELN: path relative to the upload root, else relative to the ELN's folder."""
    for cand in (_safe(ref), _safe(posixpath.join(base, ref))):
        try:
            if be.is_file(uid, cand):
                return cand
        except Exception:  # noqa: BLE001
            pass
    return None


@activity.defn
def stage_map_fit(data: MapFitInput) -> StagedInputs:
    from fit4dstem.pipeline import read_prior

    be = get_backend()
    be.check_access(data.upload_id, data.user_id)
    uid = data.upload_id
    work = tempfile.mkdtemp(prefix="fit4dstem_action_", dir=be.tmp_root())
    names = {}
    try:
        dp = _safe(data.data_path)
        if not be.is_file(uid, dp):
            raise ApplicationError(f"data set {dp!r} not found in the upload", non_retryable=True)
        prior_path = _safe(data.parameters_from)
        if not be.is_file(uid, prior_path):
            raise ApplicationError(f"{prior_path!r} not found in the upload", non_retryable=True)
        base = posixpath.dirname(prior_path)
        local_prior = be.local_path(uid, prior_path, os.path.join(work, "prior"))
        resolved = {}

        def resolve(ref):
            p = _resolve_in_upload(be, uid, ref, base)
            if p is None:
                return None
            resolved[ref] = p
            return be.local_path(uid, p, os.path.join(work, "prior"))

        try:
            prior = read_prior(local_prior, resolve=resolve)
        except (ValueError, KeyError, OSError) as e:
            raise ApplicationError(f"cannot read the previous analysis {prior_path}: {e}", non_retryable=True) from e
        libs = [_safe(x) for x in data.library_paths] if data.library_paths else []
        if not libs:
            libs = [p for p in (_resolve_in_upload(be, uid, f, base) for f in prior.library_files) if p]
        if not libs:                                    # NXem prior: lib_*.h5 next to it
            libs = [posixpath.join(base, n) if base else n for n, f in be.listdir(uid, base)
                    if f and n.lower().startswith("lib") and n.lower().endswith((".h5", ".hdf5"))]
        local_libs = []
        for i, lp in enumerate(libs):
            local_libs.append(be.local_path(uid, lp, os.path.join(work, "libs")))
            names[f"library{i}"] = lp
        names.update(data=dp, prior=prior_path)
        rp = ""
        if data.reference_path:
            rpath = _safe(data.reference_path)
            rp = be.local_path(uid, rpath, os.path.join(work, "data"))
            names["reference"] = rpath
        return StagedInputs(work_dir=work, data=be.local_path(uid, dp, os.path.join(work, "data")), reference=rp,
                            prior=local_prior, libraries=local_libs, names=names, prior_summary=prior.summary())
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


@activity.defn
def run_map_fit(args: list) -> list[str]:
    """args: [MapFitInput, StagedInputs] (as dicts).  Returns the written files (local paths)."""
    from fit4dstem import pipeline
    data, st = MapFitInput.model_validate(args[0]), StagedInputs.model_validate(args[1])
    folder = _safe(data.output_folder)
    pre = (folder + "/") if folder else ""
    name = (os.path.basename(data.output_name).removesuffix(".nxs") if data.output_name else
            os.path.splitext(os.path.basename(st.names["data"]))[0] + "_fit4dstem")
    fnames = {"data": st.names["data"], "prior": st.names["prior"], "results": f"{pre}{name}.nxs", "prefix": folder}
    if "reference" in st.names:
        fnames["reference"] = st.names["reference"]
    with Heartbeat() as hb:
        try:
            prior = pipeline.read_prior(st.prior, resolve=lambda f: next(
                (os.path.join(os.path.dirname(st.prior), *p.split("/")) for p in (f, os.path.basename(f))
                 if os.path.isfile(os.path.join(os.path.dirname(st.prior), *p.split("/")))), None))
            libs = list(st.libraries)
            from fit4dstem import multislice as ms
            for i, lp in enumerate(libs):              # file names of the libraries in the upload, by material
                fnames[f"library:{ms.Library.load(lp).material}"] = st.names.get(f"library{i}", os.path.basename(lp))
            out = os.path.join(st.work_dir, "output")
            r = pipeline.run(st.data, prior, out, reference_path=st.reference or None, library_paths=libs,
                             build_missing=data.build_missing_libraries, dataset=data.dataset or None,
                             det_bin=data.det_bin, recalibrate=data.recalibrate, full=data.full_pass,
                             scan_bin=data.scan_bin, seeds=data.seeds, refine_passes=data.refine_passes,
                             overrides=data.overrides(), name=name, callback=hb.progress, stop=hb.stopped,
                             file_names=fnames)
        except InterruptedError:
            raise
        except (ValueError, KeyError, OSError) as e:
            raise ApplicationError(f"fit4dstem: {e}", non_retryable=True) from e
    return list(r["files"]) + list(r.get("built_libraries", []))


@activity.defn
def publish_map_fit(args: list) -> MapFitResult:
    """args: [MapFitInput, StagedInputs, files]."""
    data, st, files = MapFitInput.model_validate(args[0]), StagedInputs.model_validate(args[1]), list(args[2])
    be = get_backend()
    folder = _safe(data.output_folder)
    t0 = time.time()
    try:
        be.add_files(data.upload_id, files, folder)
    finally:
        shutil.rmtree(st.work_dir, ignore_errors=True)
    rel = [f"{folder}/{os.path.basename(f)}" if folder else os.path.basename(f) for f in files]
    return MapFitResult(upload_id=data.upload_id, files=rel, parameters_from=st.names.get("prior", ""),
                        runtime_s=round(time.time() - t0, 1),
                        message=f"maps of {st.names.get('data')} with the parameters of {st.names.get('prior')}: "
                                + ", ".join(rel))


@activity.defn
def cleanup_map_fit(work_dir: str) -> None:
    if work_dir and os.path.basename(work_dir).startswith("fit4dstem_action_"):
        shutil.rmtree(work_dir, ignore_errors=True)
