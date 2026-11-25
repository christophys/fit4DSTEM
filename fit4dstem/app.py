# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""The fit4dstem app: four tabs (load, calibrate, fit, export) as an ipywidgets GUI.

Runs in Jupyter, Voila (``voila notebooks/fit4dstem_app.ipynb``), Solara (``solara run webapp/solara_app.py``) and as
a NOMAD dashboard (``fit4dstem_nomad``).  Long computations run in a background thread with progress, remaining time
and a stop button.  Figures are rendered to PNG (identical everywhere).
"""
from __future__ import annotations

import html
import json
import os
import threading
import time
import traceback

import ipywidgets as W
import numpy as np

from . import __version__
from . import calibration as calm
from . import crystal as cr
from . import depth as dp
from . import eln
from . import fitting as fm
from . import io as fio
from . import multislice as ms
from . import nxem
from . import plots
from . import reference as ref
from . import strain as st
from .nomad import NOMAD_URLS, NomadClient, NomadError, default_results_folder

_S = {"description_width": "150px"}
# layouts are widgets too: one new instance per use (a shared module-level instance belongs to no
# session in a multi-user server such as Solara and shows "widget model not found")


def _L():
    return W.Layout(width="420px")


def _LS():
    return W.Layout(width="260px")


def _html_ok(msg, ok=True):
    col = "green" if ok else "#b00020"
    return f"<span style='color:{col}'>{msg}</span>"


class Job:
    """Runs a function in a background thread with a progress bar, remaining time and a stop button."""

    def __init__(self):
        self.bar = W.FloatProgress(min=0, max=1, layout=W.Layout(width="420px"))
        self.msg = W.HTML()
        self.stop_btn = W.Button(description="stop", icon="stop", button_style="danger", disabled=True,
                                 layout=W.Layout(width="90px"))
        self.stop_btn.on_click(lambda _: self._stop())
        self._stop_flag = False
        self.thread = None
        self.box = W.HBox([self.bar, self.stop_btn, self.msg])
        self.t0 = 0

    def _stop(self):
        self._stop_flag = True
        self.msg.value = "stopping ..."

    def stopped(self):
        return self._stop_flag

    def progress(self, f, text=""):
        self.bar.value = float(np.clip(f, 0, 1))
        el = time.time() - self.t0
        self.msg.value = f"<small>{text} ({el:.0f} s)</small>"

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def run(self, fn, done=None, err=None, block=False):
        if self.busy():
            self.msg.value = _html_ok("another computation is running", False)
            return
        self._stop_flag = False
        self.stop_btn.disabled = False
        self.bar.value = 0
        self.t0 = time.time()

        def target():
            try:
                out = fn()
                if done:
                    done(out)
                self.msg.value = f"<small>done ({time.time() - self.t0:.0f} s)</small>"
                self.bar.value = 1
            except InterruptedError:
                self.msg.value = _html_ok("stopped", False)
            except Exception as ex:  # noqa: BLE001
                self.msg.value = _html_ok(f"error: {ex}", False)
                if err:
                    err(ex)
                traceback.print_exc()
            finally:
                self.stop_btn.disabled = True
        if block:
            target()
        else:
            self.thread = threading.Thread(target=target, daemon=True)
            self.thread.start()


class App:
    """The GUI.  ``data_dir``: folder shown in the file browser (and searched for libraries); ``work_dir``: where
    results are written; ``server_mode``: no access to other folders of the server (NOMAD dashboard)."""

    def __init__(self, data_dir=None, work_dir=None, server_mode=False):
        self.data_dir = os.path.abspath(data_dir or os.getcwd())
        self.work_dir = os.path.abspath(work_dir or os.path.join(os.getcwd(), "fit4dstem_results"))
        os.makedirs(self.work_dir, exist_ok=True)
        self.server_mode = server_mode
        self.ds = None
        self.ds_ref = None
        self.cal = None
        self.libs = {}
        self.model = None
        self.result = None
        self.mapfitter = None
        self.coarse_frames = None
        self.coarse_meas = None
        self.zone_cands = []
        self.ref_centres = None
        self.descan = None
        self.deflection = None
        self.strain = None
        self.depth_results = []
        self.generated_files = []
        self.nomad_ctx = {}
        self.job = Job()
        self.log_box = W.Textarea(layout=W.Layout(width="98%", height="110px"))
        self._build()

    # ================================================================== helpers
    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + str(msg)
        self.log_box.value = (line + "\n" + self.log_box.value)[:20000]

    def _status(self, w, msg, ok=True):
        w.value = _html_ok(msg, ok)

    def _img(self, width="480px"):
        return W.Image(format="png", layout=W.Layout(width=width))

    # ================================================================== layout
    def _build(self):
        self.tabs = W.Tab(children=[self._tab_load(), self._tab_calibrate(), self._tab_fit(), self._tab_export()])
        for i, t in enumerate(("1 Load", "2 Calibrate", "3 Fit maps", "4 Export")):
            self.tabs.set_title(i, t)
        head = W.HTML(f"<h3 style='margin:4px 0'>fit4dstem {__version__} – 4D-STEM CBED maps of ferroelectric "
                      f"superlattices</h3><small>thickness · tilt · Ti displacement · strain · field · NXem/NOMAD"
                      f"</small>")
        self.ui = W.VBox([head, self.tabs, self.job.box, W.HTML("<b>log</b>"), self.log_box])

    # ================================================================== tab 1
    def _tab_load(self):
        self.w_dir = W.Text(value=self.data_dir, description="folder", style=_S, layout=W.Layout(width="560px"),
                            disabled=self.server_mode)
        self.w_files = W.Select(options=[], rows=8, layout=W.Layout(width="560px"))
        self.w_up = W.Button(description="up", icon="level-up", layout=W.Layout(width="70px"),
                             disabled=self.server_mode)
        self.w_refresh = W.Button(description="list", icon="refresh", layout=W.Layout(width="80px"))
        self.w_up.on_click(lambda _: self._browse(os.path.dirname(self.w_dir.value)))
        self.w_refresh.on_click(lambda _: self._browse(self.w_dir.value))
        self.w_files.observe(self._file_select, names="value")
        self.w_dsp = W.Dropdown(options=[("(auto)", "")], description="4D dataset", style=_S, layout=_L())
        self.w_bin = W.BoundedIntText(value=1, min=1, max=16, description="detector binning", style=_S, layout=_LS())
        self.w_open = W.Button(description="open as specimen data", icon="folder-open", button_style="primary",
                               layout=W.Layout(width="220px"))
        self.w_open_ref = W.Button(description="open as vacuum reference", icon="folder-open",
                                   layout=W.Layout(width="220px"))
        self.w_open.on_click(lambda _: self._open_local(False))
        self.w_open_ref.on_click(lambda _: self._open_local(True))
        self.w_info = W.HTML()
        self.w_info_ref = W.HTML()
        # metadata (from the file, editable)
        self.w_E = W.FloatText(value=60.0, description="high tension (kV)", style=_S, layout=_LS())
        self.w_alpha = W.FloatText(value=4.0, description="conv. semi-angle (mrad)", style=_S, layout=_LS())
        self.w_step = W.FloatText(value=0.2, description="scan step (nm)", style=_S, layout=_LS())
        self.w_gain = W.FloatText(value=1.0, description="counts per electron", style=_S, layout=_LS())
        self.w_zone = W.Dropdown(options=list(cr.ZONES), value="[110]", description="zone axis", style=_S,
                                 layout=_LS())
        mats = []
        self.w_mat = {}
        for name, m in cr.MATERIALS.items():
            a = W.FloatText(value=m.a, description=f"{name} a (Å)", style=_S, layout=_LS())
            c = W.FloatText(value=m.c, description=f"{name} c (Å)", style=_S, layout=_LS())
            self.w_mat[name] = (a, c)
            mats.append(W.HBox([a, c]))
        # preview
        self.w_prev_n = W.BoundedIntText(value=6, min=1, max=64, description="preview: n x n frames", style=_S,
                                         layout=_LS())
        self.w_prev = W.Button(description="preview", icon="eye")
        self.w_prev.on_click(lambda _: self._preview())
        self.w_prev_img = self._img("420px")
        self.w_vimg = self._img("420px")
        # NOMAD
        nomad = self._nomad_box()
        upload = self._upload_box()
        return W.VBox([
            W.HTML("<b>Local files</b> (HDF5, NeXus/NXem, Nion Swift .h5/.ndata, .npy, other formats via "
                   "RosettaSciIO). Data are read lazily: only the frames needed are loaded."),
            W.HBox([self.w_dir, self.w_up, self.w_refresh]), self.w_files, upload,
            W.HBox([self.w_dsp, self.w_bin]), W.HBox([self.w_open, self.w_open_ref]), self.w_info, self.w_info_ref,
            nomad,
            W.HTML("<b>Acquisition parameters</b> (read from the file where available; please check)"),
            W.HBox([self.w_E, self.w_alpha]), W.HBox([self.w_step, self.w_gain]), self.w_zone, *mats,
            W.HBox([self.w_prev_n, self.w_prev]), W.HBox([self.w_prev_img, self.w_vimg]),
        ])

    def _upload_box(self):
        """Upload from the user's computer (server mode: Solara / NOMAD dashboard): chunked transfer with progress
        into the session folder; the file then appears in the list above (open it as specimen or reference)."""
        from .nomad_project import ProjectPicker, upload_button
        self.w_upload_progress = W.HTML()
        self.w_upload = None
        self._stored = {}
        self.project = ProjectPicker(folder="4dstem_data", on_change=self._project_changed)
        if not self.server_mode:
            return W.VBox([])
        self.w_upload = upload_button(accept=",".join(fio.EXTENSIONS), label="Upload 4D-STEM data file(s)")
        if self.w_upload is None:                          # ipyvuetify not available
            return W.VBox([])
        self.w_upload.observe(lambda ch: ch["new"] and threading.Thread(
            target=self._stream_job, args=(list(ch["new"]),), daemon=True).start(), names="file_info")
        return W.VBox([self.project.box,
                       W.HBox([W.HTML("<b>From your computer:</b>", layout=W.Layout(width="150px")),
                               W.Box([self.w_upload], layout=W.Layout(width="400px"))]),
                       W.HTML("<small>transferred in chunks (no size limit); for data sets of many GB, loading "
                              "lazily from a NOMAD upload (below) is faster</small>"), self.w_upload_progress])

    def _project_changed(self, upload_id):
        """The chosen NOMAD project is also where the results go (tab 4)."""
        if upload_id:
            self.w_nm_upload.value = upload_id

    def _store_in_project(self, paths):
        """Put the uploaded files into the chosen NOMAD project (a new one is created if chosen).
        Returns (result of ProjectPicker.store or None, error message)."""
        if not self.project.connected or not paths:
            return None, ""
        t0 = time.time()

        def prog(sent, size):
            self.w_upload_progress.value = (
                f"<span style='color:#1565c0'>storing in the NOMAD project: {100 * sent / max(size, 1):.0f} % "
                f"({sent / 1e6:.1f} of {size / 1e6:.1f} MB, {sent / max(time.time() - t0, 1e-3) / 1e6:.1f} "
                f"MB/s)</span>")
        try:
            r = self.project.store(paths, progress=prog, tmp_dir=self.work_dir,
                                   default_name=os.path.splitext(os.path.basename(paths[0]))[0])
        except Exception as e:  # noqa: BLE001
            return None, str(e)
        if r:
            self._project_changed(r["upload_id"])
            for p, up in zip(paths, r["paths"]):
                self._stored[os.path.abspath(p)] = {"url": r["url"], "upload_id": r["upload_id"], "path": up}
        return r, ""

    def _stream_job(self, info, chunk=8 << 20):
        files = self.w_upload.get_files(timeout=300)
        total = sum(int(f["size"]) for f in info) or 1
        done, t0 = 0, time.time()
        dest = os.path.join(self.work_dir, "uploads")
        os.makedirs(dest, exist_ok=True)
        paths = []
        try:
            for k, f in enumerate(files):
                name = os.path.basename(f["name"]) or f"upload_{k}"
                size, got = int(f["size"]), 0
                path = os.path.join(dest, name)
                with open(path, "wb") as fo:
                    while got < size:
                        b = f["file_obj"].read(min(chunk, size - got))
                        if not b:
                            break
                        fo.write(b)
                        got += len(b)
                        done += len(b)
                        rate = done / max(time.time() - t0, 1e-3) / 1e6
                        self.w_upload_progress.value = (
                            f"<span style='color:#1565c0'>⬆ uploading <b>{html.escape(name)}</b>"
                            f"{f' ({k + 1} of {len(files)})' if len(files) > 1 else ''}: {100 * got / max(size, 1):.0f} % "
                            f"({got / 1e6:.1f} of {size / 1e6:.1f} MB, {rate:.1f} MB/s)</span>"
                            f"<progress value='{done}' max='{total}' style='width:300px;margin-left:8px'></progress>")
                if got != size:
                    raise IOError(f"transfer of {name} incomplete ({got} of {size} bytes)")
                paths.append(path)
        except Exception as e:  # noqa: BLE001
            self.w_upload_progress.value = f"<span style='color:crimson'>upload failed: {html.escape(repr(e))}</span>"
            return
        stored, err = self._store_in_project(paths)
        self._browse(dest)
        if paths:
            self.w_files.value = ("file", paths[-1])
        where = ""
        if err:
            where = f"<br><span style='color:crimson'>not stored in NOMAD: {html.escape(err)}</span>"
        elif stored:
            where = (f"<br><span style='color:green'>stored in NOMAD project <a href='{stored['gui_url']}' "
                     f"target='_blank' rel='noopener'>{html.escape(stored['upload_id'])}</a>/"
                     f"{html.escape(stored['folder'])}; the results go there too (tab 4)</span>")
        self.w_upload_progress.value = (
            f"<span style='color:green'>✔ uploaded {html.escape(', '.join(os.path.basename(p) for p in paths))} "
            f"({total / 1e6:.1f} MB in {time.time() - t0:.0f} s) — selected above: press <i>open as specimen data</i> "
            f"or <i>open as vacuum reference</i></span>" + where)
        self.log("uploaded " + ", ".join(paths))

    def _browse(self, path):
        if self.server_mode and not os.path.abspath(path).startswith(self.data_dir) and \
                not os.path.abspath(path).startswith(self.work_dir):
            path = self.data_dir
        self.w_dir.value = path
        try:
            items = sorted(os.listdir(path))
        except OSError as e:
            self.log(str(e))
            return
        opts = [(f"📁 {n}", ("dir", os.path.join(path, n))) for n in items if os.path.isdir(os.path.join(path, n))]
        for n in items:
            p = os.path.join(path, n)
            if os.path.isfile(p) and n.lower().endswith(fio.EXTENSIONS):
                opts.append((f"📄 {n}  ({os.path.getsize(p) / 1e9:.2f} GB)", ("file", p)))
        self._browsing = True
        self.w_files.options = opts
        self.w_files.value = None
        self._browsing = False

    def _file_select(self, ch):
        v = ch["new"]
        if not v or getattr(self, "_browsing", False):
            return
        if v[0] == "dir":
            self._browse(v[1])
            return
        p = v[1]
        opts = [("(auto)", "")]
        if p.lower().endswith((".h5", ".hdf5", ".nxs", ".nx", ".emd", ".hdf")):
            try:
                opts += [(f"{a}  {tuple(s)}", a) for a, s in fio.list_4d_datasets(p)]
            except Exception as e:  # noqa: BLE001
                self.log(f"cannot list {p}: {e}")
        self.w_dsp.options = opts

    def _apply_meta(self, ds):
        m = ds.meta
        if m.get("energy_eV"):
            self.w_E.value = round(m["energy_eV"] / 1e3, 3)
        if m.get("convergence_mrad"):
            self.w_alpha.value = round(m["convergence_mrad"], 3)
        if m.get("scan_step_nm"):
            self.w_step.value = float(m["scan_step_nm"][0])

    def _open_local(self, is_ref):
        v = self.w_files.value
        if not v or v[0] != "file":
            self.log("select a file first")
            return
        try:
            ds = fio.open_dataset(v[1], self.w_dsp.value or None, det_bin=self.w_bin.value)
        except Exception as e:  # noqa: BLE001
            self.log(f"cannot open {v[1]}: {e}")
            return
        if not is_ref and os.path.abspath(v[1]) in getattr(self, "_stored", {}):
            self.nomad_ctx = dict(self._stored[os.path.abspath(v[1])])     # the file is also in a NOMAD project
        self._set_dataset(ds, is_ref)

    def _set_dataset(self, ds, is_ref):
        if is_ref:
            if self.ds_ref is not None:
                self.ds_ref.close()
            self.ds_ref = ds
            self.w_info_ref.value = "<b>reference:</b> " + ds.describe()
            self.log("vacuum reference: " + ds.describe())
        else:
            if self.ds is not None:
                self.ds.close()
            self.ds = ds
            self._apply_meta(ds)
            self.w_info.value = "<b>specimen:</b> " + ds.describe()
            self.log("specimen data: " + ds.describe())
            self.cal = None
            self.result = None

    def _preview(self):
        if self.ds is None:
            self.log("open a data set first")
            return
        ds = self.ds

        def work():
            pos = fio.coarse_positions(ds.scan_shape, self.w_prev_n.value)
            fr = ds.frames(pos)
            mean = fr.mean(0)
            self.w_prev_img.value = plots.pattern(mean, title=f"mean of {len(pos)} frames")
            # coarse virtual image: total counts
            sy, sx = ds.scan_shape
            vi = np.full((sy, sx), np.nan)
            for (iy, ix), f in zip(pos, fr):
                vi[iy, ix] = f.sum()
            ys = sorted({p[0] for p in pos})
            xs = sorted({p[1] for p in pos})
            small = np.array([[vi[y, x] for x in xs] for y in ys])
            self.w_vimg.value = plots.maps([("total counts (coarse sample)", small, "gray", None)], ncols=1)
            return pos
        self.job.run(work)

    # ------------------------------------------------------------------ NOMAD (tab 1 and 4)
    def _nomad_box(self):
        lay = W.Layout(width="440px")
        self.w_nm_url = W.Dropdown(options=[("nomad-lab.eu (production)", "production"),
                                            ("nomad-lab.eu (test)", "test"), ("Oasis: API URL →", "custom")],
                                   value="production", description="NOMAD", style=_S, layout=lay)
        self.w_nm_custom = W.Text(placeholder="https://your-oasis/nomad-oasis/api/v1", description="API URL",
                                  style=_S, layout=lay)
        self.w_nm_token = W.Password(placeholder="kept for this session only", description="API token", style=_S,
                                     layout=lay)
        self.w_nm_upload = W.Text(placeholder="upload ID", description="upload ID", style=_S, layout=lay)
        self.w_nm_my = W.Button(description="my uploads", icon="list")
        self.w_nm_uploads = W.Dropdown(options=[], layout=W.Layout(width="330px"))
        self.w_nm_uploads.observe(lambda ch: ch["new"] and setattr(self.w_nm_upload, "value", ch["new"]),
                                  names="value")
        self.w_nm_my.on_click(lambda _: self._nm_list_uploads())
        self.w_nm_path = W.Text(description="folder", style=_S, layout=lay)
        self.w_nm_browse = W.Button(description="list files", icon="folder-open")
        self.w_nm_up = W.Button(description="up", icon="level-up", layout=W.Layout(width="70px"))
        self.w_nm_list = W.Select(options=[], rows=7, layout=W.Layout(width="640px"))
        self.w_nm_browse.on_click(lambda _: self._nm_browse())
        self.w_nm_up.on_click(lambda _: (setattr(self.w_nm_path, "value", "/".join(
            self.w_nm_path.value.strip("/").split("/")[:-1])), self._nm_browse()))
        self.w_nm_list.observe(self._nm_select, names="value")
        self.w_nm_lazy = W.Button(description="open lazily (specimen)", icon="cloud", button_style="primary",
                                  layout=W.Layout(width="210px"),
                                  tooltip="read only the frames needed with HTTP range requests (HDF5/NeXus/Nion .h5)")
        self.w_nm_lazy_ref = W.Button(description="open lazily (reference)", icon="cloud",
                                      layout=W.Layout(width="210px"))
        self.w_nm_dl = W.Button(description="download file", icon="download", layout=W.Layout(width="150px"),
                                tooltip="download (e.g. libraries, small files) into the work folder")
        self.w_nm_lazy.on_click(lambda _: self._nm_open(False))
        self.w_nm_lazy_ref.on_click(lambda _: self._nm_open(True))
        self.w_nm_dl.on_click(lambda _: self._nm_download())
        self.w_nm_status = W.HTML()
        box = W.VBox([W.HTML("Data on NOMAD (production, test server or another Oasis): an <b>API token</b> (NOMAD "
                             "user menu → API tokens, read/write uploads) and the <b>upload ID</b>. Large files are "
                             "read lazily with HTTP range requests (only the frames needed are transferred)."),
                      W.HBox([self.w_nm_url, self.w_nm_custom]), W.HBox([self.w_nm_token, self.w_nm_upload]),
                      W.HBox([self.w_nm_my, self.w_nm_uploads]), W.HBox([self.w_nm_path, self.w_nm_up,
                                                                         self.w_nm_browse]),
                      self.w_nm_list, W.HBox([self.w_nm_lazy, self.w_nm_lazy_ref, self.w_nm_dl]), self.w_nm_status])
        acc = W.Accordion(children=[box])
        acc.set_title(0, "load from NOMAD (Oasis)")
        acc.selected_index = None
        return acc

    def _nm_client(self):
        url = self.w_nm_custom.value.strip() if self.w_nm_url.value == "custom" else self.w_nm_url.value
        return NomadClient(url, self.w_nm_token.value, opener=getattr(self, "_nm_opener", None))

    def _nm_list_uploads(self):
        try:
            ups = self._nm_client().uploads()
        except NomadError as e:
            self._status(self.w_nm_status, str(e), False)
            return
        self.w_nm_uploads.options = [(f"{n or '(no name)'} · {u}" + (" · published" if p else ""), u)
                                     for u, n, p in ups]
        self._status(self.w_nm_status, f"{len(ups)} upload(s)")

    def _nm_browse(self):
        uid = self.w_nm_upload.value.strip()
        if not uid:
            self._status(self.w_nm_status, "enter an upload ID", False)
            return
        try:
            items = self._nm_client().listdir(uid, self.w_nm_path.value.strip("/"))
        except NomadError as e:
            self._status(self.w_nm_status, str(e), False)
            return
        base = self.w_nm_path.value.strip("/")
        opts = []
        for name, is_file, size in sorted(items, key=lambda t: (t[1], t[0].lower())):
            p = f"{base}/{name}" if base else name
            if not is_file:
                opts.append((f"📁 {name}", ("dir", p)))
            else:
                opts.append((f"📄 {name}   ({(size or 0) / 1e9:.2f} GB)", ("file", p)))
        self._nm_browsing = True
        self.w_nm_list.options = opts
        self.w_nm_list.value = None
        self._nm_browsing = False
        self._status(self.w_nm_status, f"{len(opts)} item(s) in /{base}")

    def _nm_select(self, ch):
        v = ch["new"]
        if v and v[0] == "dir" and not getattr(self, "_nm_browsing", False):
            self.w_nm_path.value = v[1]
            self._nm_browse()

    def _nm_open(self, is_ref):
        v = self.w_nm_list.value
        uid = self.w_nm_upload.value.strip()
        if not v or v[0] != "file":
            self._status(self.w_nm_status, "select a file", False)
            return

        def work():
            cl = self._nm_client()
            ds = fio.open_remote(cl, uid, v[1], det_bin=self.w_bin.value)
            self.nomad_ctx = {"url": cl.url, "upload_id": uid, "path": v[1]}
            self._set_dataset(ds, is_ref)
            self._status(self.w_nm_status, f"opened lazily from NOMAD upload {uid}: {v[1]}")
        self.job.run(work, err=lambda e: self._status(self.w_nm_status, str(e), False))

    def _nm_download(self):
        v = self.w_nm_list.value
        uid = self.w_nm_upload.value.strip()
        if not v or v[0] != "file":
            self._status(self.w_nm_status, "select a file", False)
            return

        def work():
            local = self._nm_client().download(uid, v[1], os.path.join(self.work_dir, "nomad_downloads"))
            self._status(self.w_nm_status, f"downloaded to {local}")
            self.log(f"downloaded {v[1]} -> {local}")
            self._lib_refresh()
        self.job.run(work, err=lambda e: self._status(self.w_nm_status, str(e), False))

    # ================================================================== tab 2
    def _tab_calibrate(self):
        self.w_coarse_n = W.BoundedIntText(value=12, min=2, max=64, description="coarse sample n x n", style=_S,
                                           layout=_LS())
        self.w_r = W.FloatText(value=0.0, description="disc radius (px, 0 = auto)", style=_S, layout=_LS())
        self.w_find = W.Button(description="1 read coarse sample + find lattice", icon="search",
                               button_style="primary", layout=W.Layout(width="300px"))
        self.w_find.on_click(lambda _: self._find_lattice())
        self.w_dist_order = W.Dropdown(options=[("none", 0), ("2", 2), ("3", 3), ("4", 4), ("5", 5)], value=3,
                                       description="distortion order", style=_S, layout=_LS())
        self.w_gmax = W.FloatText(value=0.0, description="discs used up to (mrad, 0 = all)", style=_S, layout=_LS())
        self.w_dist = W.Button(description="2 fit distortions", icon="th", layout=W.Layout(width="300px"))
        self.w_dist.on_click(lambda _: self._fit_distortion())
        self.w_zone_c = W.Dropdown(options=[], description="crystal axes", style=_S, layout=W.Layout(width="620px"))
        self.w_zone_c.observe(lambda ch: self._zone_changed(), names="value")
        self.w_bg_method = W.Dropdown(options=[("smooth (normalized convolution)", "smooth"),
                                               ("radial spline x harmonics", "radial"), ("none", "none")],
                                      value="smooth", description="background", style=_S, layout=_L())
        self.w_bg_sigma = W.FloatText(value=6.0, description="smoothing sigma (px)", style=_S, layout=_LS())
        self.w_bg_margin = W.FloatText(value=1.5, description="mask margin (px)", style=_S, layout=_LS())
        self.w_bg_auto = W.Button(description="3 choose sigma (cross-validation)", icon="magic",
                                  layout=W.Layout(width="300px"))
        self.w_bg_auto.on_click(lambda _: self._bg_auto())
        self.w_bg_show = W.BoundedIntText(value=0, min=0, max=10000, description="show coarse pattern #",
                                          style=_S, layout=_LS())
        self.w_bg_prev = W.Button(description="show background", icon="eye")
        self.w_bg_prev.on_click(lambda _: self._bg_preview())
        self.w_maps_coarse = W.Button(description="4 interpolated beam maps (coarse)", icon="map",
                                      layout=W.Layout(width="300px"))
        self.w_maps_coarse.on_click(lambda _: self._coarse_maps())
        self.w_full = W.Button(description="5 full pass (all patterns)", icon="play", button_style="warning",
                               layout=W.Layout(width="300px"))
        self.w_full.on_click(lambda _: self._full_pass())
        # vacuum reference
        self.w_ref_use_ap = W.Checkbox(value=True, description="disc radius and edge from the reference",
                                       indent=False)
        self.w_ref_off = W.Text(value="auto", description="ref. frame offset (y, x px)", style=_S, layout=_LS())
        self.w_ref_mode = W.Dropdown(options=[("polynomial (degree 1)", ("poly", 1)), ("polynomial (2)", ("poly", 2)),
                                              ("polynomial (3)", ("poly", 3)), ("interpolate", ("interp", 0)),
                                              ("same grid", ("same", 0))],
                                     value=("poly", 2), description="descan model", style=_S, layout=_L())
        self.w_ref_all = W.Checkbox(value=False, description="all reference positions (else coarse)", indent=False)
        self.w_ref_btn = W.Button(description="6 vacuum reference -> descan + deflection", icon="crosshairs",
                                  layout=W.Layout(width="320px"))
        self.w_ref_btn.on_click(lambda _: self._reference())
        self.w_cal_save = W.Button(description="save calibration", icon="save")
        self.w_cal_save.on_click(lambda _: self._save_cal())
        self.w_cal_info = W.HTML()
        self.w_cal_img1 = self._img("420px")
        self.w_cal_img2 = self._img("420px")
        self.w_cal_maps = self._img("900px")
        return W.VBox([
            W.HTML("Calibration from a <b>coarse sample</b> of patterns that covers the whole scan; the maps of the "
                   "beam position and lattice are interpolated from it and refined by the full pass."),
            W.HBox([self.w_coarse_n, self.w_r]), self.w_find,
            W.HBox([self.w_dist_order, self.w_gmax]), self.w_dist, self.w_zone_c,
            W.HBox([self.w_bg_method]), W.HBox([self.w_bg_sigma, self.w_bg_margin]),
            W.HBox([self.w_bg_auto, self.w_bg_show, self.w_bg_prev]),
            W.HBox([self.w_maps_coarse, self.w_full]),
            W.HTML("<b>Vacuum reference</b> (descan; frames may be smaller, same pixel size)"),
            W.HBox([self.w_ref_use_ap, self.w_ref_all]), W.HBox([self.w_ref_off, self.w_ref_mode]), self.w_ref_btn,
            W.HBox([self.w_cal_save]), self.w_cal_info, W.HBox([self.w_cal_img1, self.w_cal_img2]),
            self.w_cal_maps])

    def _need(self, what):
        if what == "ds" and self.ds is None:
            self.log("open a data set in tab 1 first")
            return False
        if what == "cal" and (self.cal is None or self.coarse_frames is None):
            self.log("find the lattice first (tab 2, step 1)")
            return False
        return True

    def _find_lattice(self):
        if not self._need("ds"):
            return
        ds = self.ds

        def work():
            pos = fio.coarse_positions(ds.scan_shape, self.w_coarse_n.value)
            self.job.progress(0.05, f"reading {len(pos)} frames")
            fr = ds.frames(pos)
            self.coarse_frames, self.coarse_pos = fr, pos
            self.job.progress(0.5, "lattice")
            cal = calm.initial_calibration(fr, r_guess=self.w_r.value or None)
            cal.coarse_positions = pos
            self.cal = cal
            self.w_bg_show.max = len(pos) - 1
            m = cal.measure(fr[len(fr) // 2])
            pts = cal.distortion.forward(m["c"] + m["hk"] @ m["G"].T) if m["n"] else None
            self.w_cal_img1.value = plots.pattern(fr[len(fr) // 2], points=pts, radius=cal.radius_px,
                                                  lattice=(m["c"], m["G"]), title="discs and lattice")
            self._update_zone_candidates()
            self.w_cal_info.value = "<small>" + cal.summary() + "</small>"
            self.log("lattice: " + cal.summary())
        self.job.run(work)

    def _materials(self):
        out = {}
        for name, (a, c) in self.w_mat.items():
            m0 = cr.MATERIALS[name]
            out[name] = cr.Material(name, m0.A, float(a.value), float(c.value), m0.r_eq, m0.r_ap)
        return out

    def _update_zone_candidates(self):
        cal = self.cal
        mat = self._materials()["BaTiO3"]
        hk, g = cr.reflections(mat, self.w_zone.value, 0.8)
        g = g[np.hypot(*g.T) > 0]
        cands = calm.zone_candidates(cal.G0, g)
        lam = cr.wavelength_A(self.w_E.value * 1e3)
        self.zone_cands = cands[:16]
        opts = []
        for i, c in enumerate(self.zone_cands):
            M = c["M"]
            gy = M @ np.array([0, 1.0])
            ang = np.degrees(np.arctan2(-gy[0], gy[1]))
            opts.append((f"#{i}: {lam * 1e3 / c['px_per_invA']:.4f} mrad/px, growth [001] at {ang:.0f}° "
                         f"{'(mirrored)' if np.linalg.det(M) < 0 else ''}, misfit {c['score']:.3f}", i))
        self.w_zone_c.options = opts
        if opts:
            self.w_zone_c.value = 0
            self._zone_changed()

    def _zone_changed(self):
        if self.cal is None or not self.zone_cands or self.w_zone_c.value is None:
            return
        c = self.zone_cands[self.w_zone_c.value]
        lam = cr.wavelength_A(self.w_E.value * 1e3)
        self.cal.B_cry = c["B"]
        self.cal.mrad_per_px = lam * 1e3 / c["px_per_invA"]
        self.cal.zone_assignment = dict(zone=self.w_zone.value, B=np.asarray(c["B"]).tolist(),
                                        mrad_per_px=self.cal.mrad_per_px)
        if self.coarse_frames is not None:
            M = c["M"]
            L = 0.8 * np.linalg.norm(self.cal.G0[:, 0])
            ax = dict(origin=self.cal.centre0, x=M @ np.array([1.0, 0]) / np.linalg.norm(M @ np.array([1.0, 0])) * L,
                      y=M @ np.array([0, 1.0]) / np.linalg.norm(M @ np.array([0, 1.0])) * L)
            self.w_cal_img2.value = plots.pattern(self.coarse_frames[len(self.coarse_frames) // 2], axes=ax,
                                                  title="crystal axes (choose the orientation above)")
        self.w_cal_info.value = "<small>" + self.cal.summary() + "</small>"

    def _fit_distortion(self):
        if not self._need("cal"):
            return

        def work():
            cal = self.cal
            if self.w_gmax.value and cal.mrad_per_px:
                cal.g_max_px = self.w_gmax.value / cal.mrad_per_px
            self.coarse_meas = calm.coarse_calibration(self.coarse_frames, self.coarse_pos, cal,
                                                       distortion_order=self.w_dist_order.value, log=self.log)
            self.w_cal_img2.value = plots.distortion_field(cal.distortion, cal.det_shape)
            self._update_zone_candidates()
            self.w_cal_info.value = "<small>" + cal.summary() + "</small>"
        self.job.run(work)

    def _bg_masks(self):
        cal = self.cal
        meas = self.coarse_meas or [cal.measure(f) for f in self.coarse_frames]
        self.coarse_meas = meas
        cal.bg_margin = self.w_bg_margin.value
        masks, dists = [], []
        for m in meas:
            masks.append(cal.disc_mask(m["c"], m["G"]))
            dists.append(calm.distance_outside_discs(cal.det_shape, m["c"], m["G"], cal.radius_px, cal.ensure_U()))
        return meas, masks, dists

    def _bg_auto(self):
        if not self._need("cal"):
            return

        def work():
            meas, masks, dists = self._bg_masks()
            k = np.linspace(0, len(masks) - 1, min(12, len(masks))).astype(int)
            s, scores = calm.choose_background_sigma([self.coarse_frames[i] for i in k], [masks[i] for i in k],
                                                     dist_maps=[dists[i] for i in k])
            self.w_bg_sigma.value = s
            self.cal.bg_sigma = s
            self.log(f"background sigma {s} px (cross-validation: " +
                     ", ".join(f"{a}: {b:.0f}" for a, b in scores) + ")")
            self._bg_preview()
        self.job.run(work)

    def _bg_preview(self):
        if not self._need("cal"):
            return
        cal = self.cal
        cal.bg_method, cal.bg_sigma, cal.bg_margin = self.w_bg_method.value, self.w_bg_sigma.value, \
            self.w_bg_margin.value
        i = min(self.w_bg_show.value, len(self.coarse_frames) - 1)
        f = self.coarse_frames[i]
        m = (self.coarse_meas[i] if self.coarse_meas else cal.measure(f))
        B = cal.background(f, m["c"], m["G"])
        self.w_cal_img1.value = plots.pattern(B, title=f"background of coarse pattern #{i} "
                                                         f"({B.sum() / max(f.sum(), 1):.1%} of the counts)")
        self.w_cal_img2.value = plots.pattern(np.clip(f - B, 0, None), title="pattern - background")

    def _coarse_maps(self):
        if not self._need("cal"):
            return
        cal = self.cal
        if not len(cal.coarse_centres):
            cal.coarse_positions = self.coarse_pos
            meas = [cal.measure(f) for f in self.coarse_frames]
            cal.coarse_centres = np.array([m["c"] for m in meas])
            cal.coarse_G = np.array([m["G"] for m in meas])
        calm.apply_interpolated(cal, self.ds.scan_shape)
        self._show_beam_maps("interpolated from the coarse sample")

    def _show_beam_maps(self, src):
        cal = self.cal
        c = cal.centre_map
        items = [("beam y (px)", c[..., 0] - np.nanmean(c[..., 0]), "RdBu_r", None),
                 ("beam x (px)", c[..., 1] - np.nanmean(c[..., 1]), "RdBu_r", None)]
        if cal.quality_map is not None:
            items.append(("lattice fit rms (px)", cal.quality_map, "viridis", None))
        if cal.bg_fraction_map is not None:
            items.append(("background fraction", cal.bg_fraction_map, "viridis", None))
        if self.deflection is not None:
            items += [("deflection x (mrad)", self.deflection[..., 0], "RdBu_r", None),
                      ("deflection growth (mrad)", self.deflection[..., 1], "RdBu_r", None)]
        step = self.w_step.value
        self.w_cal_maps.value = plots.maps(items, step_nm=(step, step), ncols=3, suptitle=f"beam position: {src}")

    def _full_pass(self):
        if not self._need("cal"):
            return
        cal = self.cal
        cal.bg_method, cal.bg_sigma, cal.bg_margin = self.w_bg_method.value, self.w_bg_sigma.value, \
            self.w_bg_margin.value
        if not len(cal.coarse_centres):
            cal.coarse_positions = self.coarse_pos
            meas = [cal.measure(f) for f in self.coarse_frames]
            cal.coarse_centres = np.array([m["c"] for m in meas])
            cal.coarse_G = np.array([m["G"] for m in meas])

        def work():
            calm.full_pass(self.ds, cal, progress=self.job.progress, stop=self.job.stopped, log=self.log)
            if self.descan is not None:
                self._update_deflection()
            self._show_beam_maps("full pass")
            self.w_cal_info.value = "<small>" + cal.summary() + "</small>"
        self.job.run(work)

    def _reference(self):
        if self.ds_ref is None:
            self.log("open a vacuum reference scan in tab 1")
            return
        if not self._need("cal"):
            return
        cal = self.cal

        def work():
            dr = self.ds_ref
            off = None
            if self.w_ref_off.value.strip() not in ("", "auto"):
                off = tuple(float(v) for v in self.w_ref_off.value.replace("(", "").replace(")", "").split(","))
            if self.w_ref_use_ap.value:
                f = dr.frame(dr.scan_shape[0] // 2, dr.scan_shape[1] // 2)
                c0 = ref.centre_of_disc(f, cal.radius_px, cal.edge_px)
                R, w, _, _ = calm.edge_profile(f, [c0], cal.radius_px)
                if 0.7 * cal.radius_px < R < 1.3 * cal.radius_px:
                    cal.radius_px, cal.edge_px = R, max(0.3, w)
                    cal.U = None
                    self.log(f"disc radius {R:.2f} px, edge {w:.2f} px from the vacuum reference")
            pos = None if self.w_ref_all.value else fio.coarse_positions(dr.scan_shape, self.w_coarse_n.value)
            self.ref_centres = ref.measure_reference(dr, cal.radius_px, cal.edge_px, offset=off,
                                                     distortion=cal.distortion, positions=pos,
                                                     full_shape=self.ds.det_shape, progress=self.job.progress,
                                                     stop=self.job.stopped)
            mode, deg = self.w_ref_mode.value
            self.descan = ref.descan_map(self.ref_centres, self.ds.scan_shape, mode=mode, degree=deg)
            if cal.centre_map is None:
                calm.apply_interpolated(cal, self.ds.scan_shape)
            self._update_deflection()
            self._show_beam_maps(cal.centre_map_source)
            self.w_cal_info.value = "<small>" + cal.summary() + "</small>"
        self.job.run(work)

    def _update_deflection(self):
        cal = self.cal
        centre = cal.centre_map
        G = cal.G_map
        if self.result is not None and self.result.bin == 1 and self.result.centre is not None and \
                np.isfinite(self.result.centre).any() and self.result.shape == tuple(self.ds.scan_shape):
            centre = np.where(np.isfinite(self.result.centre), self.result.centre, centre)
        M = st.mapping_maps(G, cal.B_cry)
        Mm = np.nanmean(M.reshape(-1, 2, 2), 0)
        self.deflection = ref.deflection_maps(centre, self.descan, Mm, self.w_E.value * 1e3)

    def _save_cal(self):
        if self.cal is None:
            return
        p = os.path.join(self.work_dir, "calibration.json")
        with open(p, "w") as f:
            json.dump(self.cal.to_dict(), f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                      else str(o))
        self.log(f"calibration saved: {p}")
        self.generated_files.append(p)

    # ================================================================== tab 3
    def _tab_fit(self):
        self.w_lib_dir = W.Text(value=self.data_dir, description="library folder", style=_S,
                                layout=W.Layout(width="560px"), disabled=self.server_mode)
        self.w_lib_refresh = W.Button(description="list", icon="refresh", layout=W.Layout(width="80px"))
        self.w_lib_refresh.on_click(lambda _: self._lib_refresh())
        self.w_lib = {n: W.Dropdown(options=[("(none)", "")], description=f"{n} library", style=_S,
                                    layout=W.Layout(width="560px")) for n in ("BaTiO3", "SrTiO3")}
        self.w_lib_load = W.Button(description="load libraries", icon="upload", button_style="primary")
        self.w_lib_load.on_click(lambda _: self._lib_load())
        self.w_lib_info = W.HTML()
        # library builder
        self.w_b_mat = W.Dropdown(options=list(cr.MATERIALS), description="material", style=_S, layout=_LS())
        self.w_b_tilt = W.FloatText(value=2.0, description="max. tilt (mrad)", style=_S, layout=_LS())
        self.w_b_step = W.FloatText(value=0.5, description="direction step (mrad)", style=_S, layout=_LS())
        self.w_b_gmax = W.FloatText(value=1.2, description="g max (1/Å)", style=_S, layout=_LS())
        self.w_b_tmax = W.FloatText(value=50.0, description="max. thickness (nm)", style=_S, layout=_LS())
        self.w_b_uw = W.Text(value="-40, -25, -12, 0, 12, 25, 40", description="u, w grid (pm)", style=_S,
                             layout=_L())
        self.w_b_samp = W.FloatText(value=0.08, description="sampling (Å)", style=_S, layout=_LS())
        self.w_b_abs = W.FloatText(value=0.03, description="absorption (V'/V)", style=_S, layout=_LS())
        self.w_b_dev = W.Dropdown(options=["cpu", "gpu"], value="cpu", description="device", style=_S, layout=_LS())
        self.w_b_est = W.Button(description="estimate time", icon="clock-o")
        self.w_b_est.on_click(lambda _: self._lib_estimate())
        self.w_b_go = W.Button(description="build library", icon="cogs", button_style="warning")
        self.w_b_go.on_click(lambda _: self._lib_build())
        builder = W.VBox([W.HTML("Rocking-curve library by plane-wave multislice (analytic Peng potentials with "
                                 "Debye–Waller factors; independent of the probe, focus and position). The "
                                 "direction radius is the convergence semi-angle + max. tilt + 1 mrad."),
                          W.HBox([self.w_b_mat, self.w_b_tilt]), W.HBox([self.w_b_step, self.w_b_gmax]),
                          W.HBox([self.w_b_tmax, self.w_b_samp]), self.w_b_uw, W.HBox([self.w_b_abs, self.w_b_dev]),
                          W.HBox([self.w_b_est, self.w_b_go])])
        acc_b = W.Accordion(children=[builder])
        acc_b.set_title(0, "build a library")
        acc_b.selected_index = None
        # fit options
        self.w_mix = W.Checkbox(value=True, description="fit SrTiO3 fraction (two libraries)", indent=False)
        self.w_refl = W.Checkbox(value=True, description="model-based lattice refinement (strain, field)",
                                 indent=False)
        self.w_sbin = W.BoundedIntText(value=1, min=1, max=16, description="scan binning", style=_S, layout=_LS())
        self.w_roi = W.Text(value="", description="ROI y0,y1,x0,x1", placeholder="(whole map)", style=_S,
                            layout=_LS())
        self.w_seeds = W.BoundedIntText(value=5, min=1, max=30, description="seeds per axis", style=_S, layout=_LS())
        self.w_passes = W.BoundedIntText(value=2, min=0, max=6, description="refinement passes", style=_S,
                                         layout=_LS())
        self.w_threads = W.BoundedIntText(value=1, min=1, max=64, description="threads", style=_S, layout=_LS())
        self.w_tstep = W.FloatText(value=2.0, description="seed thickness step (nm)", style=_S, layout=_LS())
        self.w_test_y = W.BoundedIntText(value=0, min=0, max=100000, description="test pattern y", style=_S,
                                         layout=_LS())
        self.w_test_x = W.BoundedIntText(value=0, min=0, max=100000, description="x", style=_S, layout=_LS())
        self.w_test = W.Button(description="test fit", icon="flask")
        self.w_test.on_click(lambda _: self._test_fit())
        self.w_ap = W.Button(description="refine disc radius + edge (model)", icon="circle-o",
                             layout=W.Layout(width="260px"),
                             tooltip="fit a few coarse patterns, then optimize the disc radius and edge width "
                                     "(when no vacuum reference is available)")
        self.w_ap.on_click(lambda _: self._refine_aperture())
        self.w_dm = W.Button(description="refine distortion (model)", icon="th", layout=W.Layout(width="220px"),
                             tooltip="disc positions from the fitted model of the coarse patterns (unbiased by the "
                                     "intensity inside the discs) -> polynomial distortion")
        self.w_dm.on_click(lambda _: self._refine_distortion())
        self.w_fit = W.Button(description="fit map", icon="play", button_style="warning")
        self.w_fit.on_click(lambda _: self._fit_map())
        self.w_fit_info = W.HTML()
        self.w_fit_img = self._img("900px")
        # display
        self.w_sm = W.FloatSlider(value=0.0, min=0, max=5, step=0.25, description="smoothing sigma (px)",
                                  style=_S, layout=W.Layout(width="420px"), continuous_update=False)
        self.w_sm_mode = W.Dropdown(options=[("layer-aware (separately in BaTiO3 / SrTiO3)", "layers"),
                                             ("Gaussian", "gauss")], value="layers", description="smoothing",
                                    style=_S, layout=_L())
        self.w_sm_w = W.Checkbox(value=True, description="weight by 1/sigma^2", indent=False)
        self.w_show = W.Dropdown(options=[("structure (t, tilt, u, w, f)", "structure"),
                                          ("polarization + arrows", "polar"), ("strain", "strain"),
                                          ("field", "field"), ("uncertainties / quality", "quality")],
                                 value="structure", description="show", style=_S, layout=_L())
        self.w_arrows = W.BoundedIntText(value=2, min=1, max=20, description="arrow every n px", style=_S,
                                         layout=_LS())
        for w in (self.w_sm, self.w_sm_mode, self.w_sm_w, self.w_show, self.w_arrows):
            w.observe(lambda ch: self._show_maps(), names="value")
        self.w_maps = self._img("1000px")
        # depth
        self.w_d_n = W.BoundedIntText(value=2, min=2, max=4, description="slabs along the beam", style=_S,
                                      layout=_LS())
        self.w_d_free = W.Checkbox(value=False, description="free boundary (2 slabs)", indent=False)
        self.w_d_reg = W.FloatText(value=0.02, description="regularization", style=_S, layout=_LS())
        self.w_d_roi = W.Text(value="", description="region y0,y1,x0,x1", placeholder="binned-map pixels",
                              style=_S, layout=_LS())
        self.w_d_block = W.BoundedIntText(value=0, min=0, max=64, description="or: blocks of n x n (map)",
                                          style=_S, layout=_LS())
        self.w_d_go = W.Button(description="depth fit", icon="align-justify", button_style="warning")
        self.w_d_go.on_click(lambda _: self._depth())
        self.w_d_img = self._img("700px")
        self.w_d_info = W.HTML()
        depth_box = W.VBox([W.HTML("The Ti displacement may change along the beam (e.g. vortex tubes inclined to "
                                   "the beam in a [100] lamella, or different polarization near the surfaces). "
                                   "The crystal is split into slabs, each with its own displacement; all "
                                   "patterns of the region are fitted jointly with plane-wave multislice (slow: "
                                   "minutes per region; GPU recommended). Correlations show what is determined."),
                            W.HBox([self.w_d_n, self.w_d_free, self.w_d_reg]),
                            W.HBox([self.w_d_roi, self.w_d_block]), self.w_d_go, self.w_d_info, self.w_d_img])
        acc_d = W.Accordion(children=[depth_box])
        acc_d.set_title(0, "depth-resolved Ti displacement (slab model)")
        acc_d.selected_index = None
        return W.VBox([
            W.HBox([self.w_lib_dir, self.w_lib_refresh]), self.w_lib["BaTiO3"], self.w_lib["SrTiO3"],
            W.HBox([self.w_lib_load]), self.w_lib_info, acc_b,
            W.HTML("<b>Fit</b> (thickness, tilt, Ti displacement u (in-plane) and w (growth), SrTiO3 fraction)"),
            W.HBox([self.w_mix, self.w_refl]), W.HBox([self.w_sbin, self.w_roi]),
            W.HBox([self.w_seeds, self.w_passes]), W.HBox([self.w_threads, self.w_tstep]),
            W.HTML("<small>Recommended order: test fit → refine disc radius/edge (if no vacuum reference) → "
                   "refine distortion → full pass (tab 2) → fit map.</small>"),
            W.HBox([self.w_test_y, self.w_test_x, self.w_test]), W.HBox([self.w_ap, self.w_dm, self.w_fit]),
            self.w_fit_info,
            self.w_fit_img,
            W.HTML("<b>Maps</b>"), W.HBox([self.w_show, self.w_arrows]), W.HBox([self.w_sm, self.w_sm_mode,
                                                                                self.w_sm_w]),
            self.w_maps, acc_d])

    def _lib_refresh(self):
        dirs = [self.w_lib_dir.value, self.work_dir, os.path.join(self.work_dir, "nomad_downloads"), self.data_dir]
        files = []
        for d in dirs:
            if os.path.isdir(d):
                for root, _, fs in os.walk(d):
                    for f in fs:
                        if f.endswith(".h5") and "lib" in f.lower():
                            files.append(os.path.join(root, f))
        files = sorted(set(files))
        for n, w in self.w_lib.items():
            w.options = [("(none)", "")] + [(os.path.relpath(f, self.work_dir) if f.startswith(self.work_dir)
                                             else f, f) for f in files]
            for f in files:
                if n.lower() in os.path.basename(f).lower():
                    w.value = f
                    break

    def _lib_load(self):
        def work():
            libs = {}
            for n, w in self.w_lib.items():
                if w.value:
                    libs[n] = ms.Library.load(w.value)
                    self.log(f"library {n}: {w.value} ({libs[n].data.nbytes / 1e6:.0f} MB, thickness "
                             f"{libs[n].thickness_nm[0]:.1f}-{libs[n].thickness_nm[-1]:.1f} nm, u {libs[n].u[0]}.."
                             f"{libs[n].u[-1]} pm, directions up to {libs[n].k[-1]:.1f} mrad)")
            if not libs:
                raise ValueError("choose at least one library")
            self.libs = libs
            z = {lb.meta.get("zone") for lb in libs.values()}
            E = {round(lb.meta.get("energy_eV", 0)) for lb in libs.values()}
            self.w_lib_info.value = (f"<small>{', '.join(libs)} loaded: zone {', '.join(z)}, "
                                     f"{', '.join(str(e / 1e3) for e in E)} kV</small>")
            if self.w_zone.value not in z:
                self.log(f"note: the libraries are for zone {z}, tab 1 says {self.w_zone.value}")
        self.job.run(work)

    def _lib_params(self):
        uw = [float(v) for v in self.w_b_uw.value.replace(";", ",").split(",") if v.strip()]
        radius = self.w_alpha.value + self.w_b_tilt.value + 1.0
        return dict(material=self._materials()[self.w_b_mat.value], zone=self.w_zone.value,
                    energy_eV=self.w_E.value * 1e3, radius_mrad=radius, step_mrad=self.w_b_step.value,
                    g_max_inv_A=self.w_b_gmax.value, max_thickness_nm=self.w_b_tmax.value, u_pm=tuple(uw),
                    w_pm=tuple(uw), sampling=self.w_b_samp.value, absorption=self.w_b_abs.value,
                    device=self.w_b_dev.value)

    def _lib_estimate(self):
        p = self._lib_params()
        eng = ms.Engine(p["material"], p["zone"], p["energy_eV"], p["sampling"], device=p["device"])
        ax, mask = ms.direction_grid(p["radius_mrad"], p["step_mrad"])
        KX, KY = np.meshgrid(ax, ax, indexing="ij")
        dirs = np.stack([KX[mask], KY[mask]], 1)
        hk, _ = cr.reflections(p["material"], p["zone"], p["g_max_inv_A"])
        t0 = time.time()
        eng.run([(2, 0, 0)], dirs, hk)
        dt = (time.time() - t0) / 2
        n = int(np.ceil(p["max_thickness_nm"] * 10 / eng.Lz))
        tot = dt * n * len(p["u_pm"]) * len(p["w_pm"])
        size = len(p["u_pm"]) ** 2 * n * len(hk) * len(ax) ** 2 * 2 / 1e9
        self.log(f"library estimate: {len(dirs)} directions, {len(hk)} reflections, {n} cells, "
                 f"{len(p['u_pm']) ** 2} structures: about {tot / 60:.0f} min, {size:.2f} GB")

    def _lib_build(self):
        p = self._lib_params()
        name = (f"lib_{p['material'].name}_{p['zone'].strip('[]')}_{p['energy_eV'] / 1e3:.0f}kV_"
                f"{p['radius_mrad']:.1f}mrad.h5")
        path = os.path.join(self.work_dir, name)

        def work():
            ms.build_library(path=path, progress=self.job.progress, stop=self.job.stopped, log=self.log, **p)
            self.generated_files.append(path)
            self._lib_refresh()
            self.log(f"library written: {path}")
        self.job.run(work)

    def _options(self):
        cal = self.cal
        lim_t = None
        return fm.FitOptions(mixture=self.w_mix.value and len(self.libs) > 1, gain=self.w_gain.value,
                             refine_lattice=self.w_refl.value, t_grid_step_nm=self.w_tstep.value)

    def _make_model(self):
        if not self.libs:
            raise ValueError("load the libraries first")
        if self.cal is None or not self.cal.mrad_per_px:
            raise ValueError("calibrate first (tab 2)")
        self.cal.bg_method, self.cal.bg_sigma, self.cal.bg_margin = self.w_bg_method.value, \
            self.w_bg_sigma.value, self.w_bg_margin.value
        if self.cal.centre_map is None:
            calm.apply_interpolated(self.cal, self.ds.scan_shape)
        libs = {k: v for k, v in self.libs.items() if self.w_mix.value or k == "BaTiO3" or len(self.libs) == 1}
        self.model = fm.DetectorModel(libs, self.cal)
        return self.model

    def _roi(self, text, shape):
        t = text.strip()
        if not t:
            return None
        v = [int(float(a)) for a in t.split(",")]
        return (max(0, v[0]), min(shape[0], v[1]), max(0, v[2]), min(shape[1], v[3]))

    def _test_fit(self):
        if not self._need("ds"):
            return

        def work():
            model = self._make_model()
            iy, ix = self.w_test_y.value, self.w_test_x.value
            f = self.ds.frame(iy, ix)
            c, G = self.cal.centre_map[iy, ix], self.cal.G_map[iy, ix]
            fit = fm.PatternFitter(model, self._options())
            red = fm.reduce_frame(f, c, G, model, self.w_gain.value)
            p, cost, s = fit.fit(red, global_search=True, n_local=3)
            if fit.opt.refine_lattice:
                c, G, _ = fit.refine_lattice(red, p, c, G)
                p, cost, s = fit.local_fit(red, p)
            M, sc, b = fit.predict(red, p)
            img = np.zeros(f.size)
            img[red.geo.pix] = M
            dat = np.zeros(f.size)
            dat[red.geo.pix] = red.y
            names = fm.PARAMS[:fit.nparam]
            txt = ", ".join(f"{n} = {v:.3f} ± {e:.3f}" for n, v, e in zip(names, p, s if s is not None else
                                                                           [np.nan] * len(p)))
            self.w_fit_info.value = (f"<small>pattern ({iy}, {ix}): {txt}; reduced chi2 "
                                     f"{cost / max(len(red.y) - len(p), 1):.2f}</small>")
            self.w_fit_img.value = plots.compare(dat.reshape(f.shape), img.reshape(f.shape),
                                                 title=f"pattern ({iy}, {ix}), disc pixels")
            self.log("test fit: " + txt)
        self.job.run(work)

    def _refine_aperture(self):
        if not self._need("cal"):
            return

        def work():
            model = self._make_model()
            opt = self._options()
            opt.refine_lattice = False
            k = np.linspace(0, len(self.coarse_frames) - 1, min(4, len(self.coarse_frames))).astype(int)
            frames, lats, pars = [], [], []
            for j, i in enumerate(k):
                f = self.coarse_frames[i]
                m = self.cal.measure(f)
                fit = fm.PatternFitter(model, opt)
                red = fm.reduce_frame(f, m["c"], m["G"], model, self.w_gain.value)
                p, _, _ = fit.fit(red, global_search=True, n_local=2)
                frames.append(f)
                lats.append((m["c"], m["G"]))
                pars.append(p)
                self.job.progress((j + 1) / (len(k) + 2), f"fit of coarse pattern {j + 1}/{len(k)}")
            libs = dict(model.libs)
            r, e, rc, ec = calm.refine_aperture(self.cal, lambda: fm.DetectorModel(libs, self.cal), frames, lats,
                                                pars, lambda mdl: fm.PatternFitter(mdl, opt))
            self.model = None
            self.log(f"disc radius {r:.2f} px, edge width {e:.2f} px (model); cost per pixel vs radius: " +
                     ", ".join(f"{a:.2f}: {b:.3f}" for a, b in rc))
            self.w_cal_info.value = "<small>" + self.cal.summary() + "</small>"
        self.job.run(work)

    def _refine_distortion(self):
        if not self._need("cal"):
            return

        def work():
            model = self._make_model()
            opt = self._options()
            opt.refine_lattice = False
            k = np.linspace(0, len(self.coarse_frames) - 1, min(9, len(self.coarse_frames))).astype(int)
            frames, lats, pars = [], [], []
            for j, i in enumerate(k):
                f = self.coarse_frames[i]
                m = self.cal.measure(f)
                fit = fm.PatternFitter(model, opt)
                red = fm.reduce_frame(f, m["c"], m["G"], model, self.w_gain.value)
                p, _, _ = fit.fit(red, global_search=True, n_local=2)
                frames.append(f)
                lats.append((m["c"], m["G"]))
                pars.append(p)
                self.job.progress((j + 1) / (len(k) + 3), f"fit of coarse pattern {j + 1}/{len(k)}")
            libs = dict(model.libs)
            order = self.w_dist_order.value or 3
            fm.refine_distortion(lambda: fm.PatternFitter(fm.DetectorModel(libs, self.cal), opt), self.cal, frames,
                                 lats, pars, order=order, n_iter=2, log=self.log)
            # coarse maps with the new distortion (the full pass of tab 2 should be repeated)
            meas = [self.cal.measure(f) for f in self.coarse_frames]
            self.coarse_meas = meas
            self.cal.coarse_centres = np.array([m["c"] for m in meas])
            self.cal.coarse_G = np.array([m["G"] for m in meas])
            calm.apply_interpolated(self.cal, self.ds.scan_shape)
            self.model = None
            self.w_cal_img2.value = plots.distortion_field(self.cal.distortion, self.cal.det_shape)
            self.w_cal_info.value = "<small>" + self.cal.summary() + " — repeat the full pass (tab 2)</small>"
        self.job.run(work)

    def _fit_map(self):
        if not self._need("ds"):
            return

        def work():
            model = self._make_model()
            roi = self._roi(self.w_roi.value, self.ds.scan_shape)
            mf = fm.MapFitter(self.ds, self.cal, model, self._options(), scan_bin=self.w_sbin.value, roi=roi,
                              progress=self.job.progress, log=self.log, stop=self.job.stopped,
                              n_threads=self.w_threads.value)
            self.mapfitter = mf
            mf.load()
            seeds = fm.seed_grid(mf.shape, self.w_seeds.value)
            self.result = mf.run(seeds, refine_passes=self.w_passes.value)
            self.result.info.update(roi=roi, scan_bin=self.w_sbin.value)
            if self.descan is not None:
                self._update_deflection()
            self._compute_strain()
            self._show_maps()
        self.job.run(work)

    # ------------------------------------------------------------------ derived maps
    def _smoothed(self, a, k=None):
        R = self.result
        s = self.w_sm.value
        if s <= 0:
            return a
        seg = fm.segment_layers(R.params[..., 5]) if (self.w_sm_mode.value == "layers" and
                                                         R.params.shape[-1] > 5) else None
        wts = None
        if self.w_sm_w.value and k is not None:
            sg = R.sigma[..., k]
            wts = 1 / np.where(np.isfinite(sg) & (sg > 0), sg, np.inf) ** 2
        return fm.smooth_map(a, s, weights=wts, segments=seg)

    def _compute_strain(self):
        R = self.result
        if R is None:
            return
        lat = R.lattice if R.lattice is not None and np.isfinite(R.lattice).any() else None
        if lat is None:
            return
        seg = fm.segment_layers(R.params[..., 5]) if R.params.shape[-1] > 5 else None
        try:
            self.strain = st.strain_maps(lat, self.cal.B_cry, ref_mask=None if seg is None else seg == 0)
        except ValueError as e:
            self.log(f"strain: {e}")

    def _step_nm(self):
        b = self.result.bin if self.result is not None else 1
        return (self.w_step.value * b, self.w_step.value * b)

    def current_maps(self):
        """All maps (smoothed as displayed): {name: (array, units, long name, method)}."""
        R = self.result
        if R is None:
            return {}
        out = {}
        labels = {"thickness_nm": ("thickness", "nm"), "tilt_x_mrad": ("tilt_x", "mrad"),
                  "tilt_y_mrad": ("tilt_y", "mrad"), "u_pm": ("Ti_displacement_u", "pm"),
                  "w_pm": ("Ti_displacement_w", "pm"), "f_STO": ("SrTiO3_fraction", "")}
        sm = self.w_sm.value
        meth = "CBED fit (fit4dstem)" + (f", smoothed sigma {sm} px ({self.w_sm_mode.value})" if sm > 0 else "")
        for k, n in enumerate(R.names):
            lab, u = labels[n]
            out[lab] = (self._smoothed(R.params[..., k], k), u, lab.replace("_", " "), meth)
            out[lab + "_sigma"] = (R.sigma[..., k], u, lab.replace("_", " ") + " uncertainty", "Cramer-Rao (fit)")
        out["reduced_chi2"] = (R.reduced_chi2(), "", "reduced chi2 (Anscombe)", "fit quality")
        mat = self._materials()["BaTiO3"]
        P = cr.polarization(mat, self.w_zone.value, out["Ti_displacement_u"][0], out["Ti_displacement_w"][0])
        out["polarization_x"] = (P[..., 0], "uC/cm^2", "polarization estimate in-plane", "Born effective charges")
        out["polarization_y"] = (P[..., 1], "uC/cm^2", "polarization estimate growth", "Born effective charges")
        if self.strain is not None:
            for key, lab in (("exx", "strain_xx"), ("eyy", "strain_yy"), ("exy", "strain_xy"),
                             ("rotation_rad", "lattice_rotation")):
                out[lab] = (self._smoothed(self.strain[key]), "rad" if "rot" in key else "", lab.replace("_", " "),
                            "disc lattice, relative to mean BaTiO3")
        if self.deflection is not None and self.deflection.shape[:2] == R.shape:
            d = self.deflection
            t = out["thickness"][0]
            integ, E = ref.field_maps(d, t, self.w_E.value * 1e3)
            out["deflection_x"] = (self._smoothed(d[..., 0]), "mrad", "beam deflection in-plane", "beam - vacuum ref.")
            out["deflection_y"] = (self._smoothed(d[..., 1]), "mrad", "beam deflection growth", "beam - vacuum ref.")
            out["field_x"] = (self._smoothed(E[..., 0]), "MV/cm", "projected field / thickness, in-plane",
                              "deflection")
            out["field_y"] = (self._smoothed(E[..., 1]), "MV/cm", "projected field / thickness, growth",
                              "deflection")
        return out

    def _show_maps(self):
        if self.result is None:
            return
        try:
            m = self.current_maps()
        except Exception as e:  # noqa: BLE001
            self.log(f"maps: {e}")
            return
        mode = self.w_show.value
        quiv = dict(u=m["Ti_displacement_u"][0], w=m["Ti_displacement_w"][0], step=self.w_arrows.value)
        if mode == "structure":
            keys = [("thickness", "viridis"), ("tilt_x", "RdBu_r"), ("tilt_y", "RdBu_r"),
                    ("Ti_displacement_u", "RdBu_r"), ("Ti_displacement_w", "RdBu_r"), ("SrTiO3_fraction", "gray")]
        elif mode == "polar":
            amp = np.hypot(m["Ti_displacement_u"][0], m["Ti_displacement_w"][0])
            m["|Ti displacement|"] = (amp, "pm", "", "")
            keys = [("|Ti displacement|", "magma"), ("polarization_x", "RdBu_r"), ("polarization_y", "RdBu_r")]
            quiv["on"] = "|Ti displacement| (pm)"
        elif mode == "strain":
            keys = [("strain_xx", "RdBu_r"), ("strain_yy", "RdBu_r"), ("strain_xy", "RdBu_r"),
                    ("lattice_rotation", "RdBu_r")]
        elif mode == "field":
            keys = [("deflection_x", "RdBu_r"), ("deflection_y", "RdBu_r"), ("field_x", "RdBu_r"),
                    ("field_y", "RdBu_r")]
        else:
            keys = [("thickness_sigma", "viridis"), ("Ti_displacement_u_sigma", "viridis"),
                    ("Ti_displacement_w_sigma", "viridis"), ("reduced_chi2", "viridis")]
        items = []
        for k, cmap in keys:
            if k in m:
                a, u = m[k][0], m[k][1]
                items.append((f"{k} ({u})" if u else k, a, cmap, None))
        if not items:
            self.w_maps.value = b""
            return
        self.w_maps.value = plots.maps(items, step_nm=self._step_nm(), ncols=3,
                                       quiver=quiv if mode == "polar" else None)

    # ------------------------------------------------------------------ depth
    def _depth(self):
        if self.mapfitter is None or self.result is None:
            self.log("fit a map first")
            return
        mf, R = self.mapfitter, self.result

        def regions():
            if self.w_d_block.value:
                b = self.w_d_block.value
                Y, X = R.shape
                return [(y, min(Y, y + b), x, min(X, x + b)) for y in range(0, Y, b) for x in range(0, X, b)]
            roi = self._roi(self.w_d_roi.value, R.shape)
            if roi is None:
                raise ValueError("give a region (y0,y1,x0,x1 of the fitted map) or a block size")
            return [roi]

        def work():
            regs = regions()
            out = []
            for k, (y0, y1, x0, x1) in enumerate(regs):
                pats, tilts, ts, mats = [], [], [], []
                for iy in range(y0, y1):
                    for ix in range(x0, x1):
                        if (iy, ix) in mf.reduced and R.done[iy, ix]:
                            pats.append(mf.reduced[(iy, ix)])
                            tilts.append((R.params[iy, ix, 1], R.params[iy, ix, 2]))
                            ts.append(R.params[iy, ix, 0])
                            mats.append(R.params[iy, ix, 5] if R.params.shape[-1] > 5 else 0)
                if not pats:
                    continue
                mat = "SrTiO3" if np.median(mats) > 0.5 and "SrTiO3" in self.model.libs else "BaTiO3"
                u0 = float(np.nanmedian(R.params[y0:y1, x0:x1, 3]))
                w0 = float(np.nanmedian(R.params[y0:y1, x0:x1, 4]))
                self.job.progress(k / len(regs), f"depth fit region {k + 1}/{len(regs)}")
                res = dp.fit_depth(pats, self.model, mat, self.w_d_n.value, float(np.median(ts)), tilts,
                                   start=(u0, w0), regularization=self.w_d_reg.value, free_boundary=self.w_d_free.value,
                                   device=self.w_b_dev.value, log=self.log, stop=self.job.stopped)
                res.info.update(region=(y0, y1, x0, x1), material=mat)
                out.append(res)
            self.depth_results = out
            if len(out) == 1:
                r = out[0]
                z = np.arange(r.n_slabs) + 0.5
                self.w_d_img.value = plots.profile(z, [r.params[:, 0], r.params[:, 1]], ["u (in-plane)",
                                                                                        "w (growth)"],
                                                   xlabel="slab (0 = beam entrance)", ylabel="Ti displacement (pm)",
                                                   title=f"region {r.info['region']}, t = "
                                                         f"{r.info['thickness_nm']:.1f} nm",
                                                   errors=[r.sigma[:, 0], r.sigma[:, 1]])
                cc = np.round(r.corr, 2)
                self.w_d_info.value = (f"<small>cost {r.cost:.1f} vs one slab {r.cost_single:.1f} "
                                       f"(improvement ~{r.significance():.1f} sigma); polarization (uC/cm²): "
                                       f"{np.round(r.polarization, 1).tolist()}; correlations: {cc.tolist()}"
                                       f"</small>")
            else:
                Y, X = R.shape
                maps_ = []
                for s in range(self.w_d_n.value):
                    for comp, lab in ((0, "u"), (1, "w")):
                        a = np.full((Y, X), np.nan)
                        for r in out:
                            y0, y1, x0, x1 = r.info["region"]
                            a[y0:y1, x0:x1] = r.params[s, comp]
                        maps_.append((f"slab {s} {lab} (pm)", a, "RdBu_r", None))
                self.w_d_img.value = plots.maps(maps_, step_nm=self._step_nm(), ncols=2)
        self.job.run(work)

    # ================================================================== tab 4
    def _tab_export(self):
        self.w_meta = {}
        rows = []
        for sec, fields in nxem.METADATA_FIELDS.items():
            rows.append(W.HTML(f"<b>{sec}</b>"))
            for k, (lab, default, req, help_) in fields.items():
                if isinstance(default, bool):
                    w = W.Checkbox(value=default, description=lab + (" *" if req else ""), indent=False)
                else:
                    w = W.Text(value=str(default), description=lab + (" *" if req else ""), placeholder=help_,
                               style=_S, layout=W.Layout(width="520px"))
                self.w_meta[(sec, k)] = w
                rows.append(w)
        meta_acc = W.Accordion(children=[W.VBox(rows)])
        meta_acc.set_title(0, "metadata (NXem; * = required)")
        meta_acc.selected_index = None
        self.w_out_name = W.Text(value="fit4dstem_results", description="file name", style=_S, layout=_L())
        self.w_nx = W.Button(description="write NXem + validate", icon="save", button_style="primary",
                             layout=W.Layout(width="220px"))
        self.w_nx.on_click(lambda _: self._write_nxem())
        self.w_nx_status = W.HTML()
        self.w_eln_schema = W.Dropdown(options=[("fit4dstem plugin (based on nomad-measurements MappingMeasurement)",
                                                 "plugin"), ("core NOMAD ELNMeasurement", "core")],
                                       value="plugin", description="ELN schema", style=_S, layout=W.Layout(
                                           width="620px"))
        self.w_eln = W.Button(description="write ELN + workflow", icon="book", layout=W.Layout(width="220px"))
        self.w_eln.on_click(lambda _: self._write_eln())
        self.w_eln_status = W.HTML()
        self.w_wf = W.HTML()
        self.w_nm_folder = W.Text(value=default_results_folder(), description="NOMAD subfolder", style=_S, layout=_L())
        self.w_nm_newname = W.Text(description="new upload name", style=_S, layout=_L())
        self.w_nm_push = W.Button(description="upload results to NOMAD", icon="cloud-upload", button_style="warning",
                                  layout=W.Layout(width="230px"))
        self.w_nm_push.on_click(lambda _: self._nm_push())
        self.w_nm_status4 = W.HTML()
        self.w_files_html = W.HTML()
        return W.VBox([meta_acc, W.HBox([self.w_out_name, self.w_nx]), self.w_nx_status,
                       W.HBox([self.w_eln_schema, self.w_eln]), self.w_eln_status,
                       W.HTML("<b>Workflow preview</b> (as NOMAD will show it after the upload)"), self.w_wf,
                       W.HTML("<b>NOMAD upload</b> (settings of tab 1: URL, token, upload ID)"),
                       W.HBox([self.w_nm_folder, self.w_nm_newname]), self.w_nm_push, self.w_nm_status4,
                       self.w_files_html])

    def metadata(self):
        m = nxem.default_metadata()
        for (sec, k), w in self.w_meta.items():
            m[sec][k] = w.value
        return m

    def _results_name(self):
        return os.path.join(self.work_dir, self.w_out_name.value.strip() or "fit4dstem_results")

    def _write_nxem(self):
        if self.result is None:
            self._status(self.w_nx_status, "nothing to export yet (fit a map in tab 3)", False)
            return

        def work():
            path = self._results_name() + ".nxs"
            maps = self.current_maps()
            inputs = []
            for d, lab in ((self.ds, "4D-STEM data set (specimen)"), (self.ds_ref, "vacuum reference scan")):
                if d is not None:
                    inf = dict(file=d.meta.get("source_file", d.name), description=lab)
                    inputs.append(inf)
            extra = {}
            if self.depth_results:
                extra["depth_results"] = [dict(region=r.info.get("region"), material=r.info.get("material"),
                                               n_slabs=r.n_slabs, u_w_pm=r.params.tolist(),
                                               sigma_pm=r.sigma.tolist(), polarization_uC_cm2=r.polarization.tolist(),
                                               cost=r.cost, cost_single=r.cost_single) for r in self.depth_results]
            if self.libs:
                from .pipeline import library_meta
                extra["libraries"] = {k: library_meta(v) for k, v in self.libs.items()}
            mean = self.coarse_frames.mean(0) if self.coarse_frames is not None else None
            nxem.write_results(path, maps, self._step_nm(), metadata=self.metadata(), energy_eV=self.w_E.value * 1e3,
                               convergence_mrad=self.w_alpha.value, calibration=self.cal.to_dict(),
                               options=self._options().__dict__, inputs=inputs, mean_pattern=mean,
                               mrad_per_px=self.cal.mrad_per_px, extra=extra)
            ok, problems, _ = nxem.validate(path)
            self.generated_files.append(path)
            self.nxs_path = path
            msg = f"written {path} ({os.path.getsize(path) / 1e6:.1f} MB); pynxtools: " + \
                ("valid" if ok else ("not installed" if ok is None else "problems: " + "; ".join(problems[:5])))
            self._status(self.w_nx_status, msg, bool(ok))
            self.log(msg)
            self._files_html()
        self.job.run(work)

    def _write_eln(self):
        def work():
            base = self._results_name()
            name = os.path.basename(base)
            res = getattr(self, "nxs_path", None)
            libs = [os.path.basename(w.value) for w in self.w_lib.values() if w.value]
            settings = dict(energy_eV=self.w_E.value * 1e3, convergence_mrad=self.w_alpha.value,
                            scan_step_nm=self.w_step.value,
                            scan_shape=list(self.ds.scan_shape) if self.ds else None,
                            det_shape=list(self.ds.det_shape) if self.ds else None,
                            mrad_per_px=self.cal.mrad_per_px if self.cal else None, zone=self.w_zone.value)
            summary = None
            if self.result is not None:
                R = self.result
                summary = (f"thickness {np.nanmedian(R.params[..., 0]):.1f} nm (median), |Ti displacement| "
                           f"{np.nanmedian(np.hypot(R.params[..., 3], R.params[..., 4])):.1f} pm (median), "
                           f"{int(R.done.sum())} patterns fitted")
            files = eln.write_eln(self.work_dir, name, data_file=os.path.basename(
                self.ds.meta.get("source_file", "data.h5")) if self.ds else "data.h5",
                reference_file=os.path.basename(self.ds_ref.meta.get("source_file", "")) if self.ds_ref else None,
                results_file=os.path.basename(res) if res else None, library_files=libs, settings=settings,
                metadata=self.metadata(), calibration=self.cal.summary() if self.cal else None,
                options=self._options().__dict__, results_summary=summary, schema=self.w_eln_schema.value)
            for p in files.values():
                self.generated_files.append(p)
            self.w_wf.value = open(files["svg"]).read()
            self._status(self.w_eln_status, "written: " + ", ".join(os.path.basename(p) for p in files.values()))
            self._files_html()
        self.job.run(work)

    def _files_html(self):
        fs = [f for f in dict.fromkeys(self.generated_files) if os.path.isfile(f)]
        self.w_files_html.value = "<small>files of this session: " + (", ".join(
            f"<code>{os.path.basename(f)}</code> ({os.path.getsize(f) / 1e6:.1f} MB)" for f in fs) or "none") + \
            "</small>"

    def _nm_push(self):
        files = [f for f in dict.fromkeys(self.generated_files) if os.path.isfile(f)]
        if not files:
            self._status(self.w_nm_status4, "no files written yet", False)
            return

        def work():
            r = self._nm_client().upload(files, upload_id=self.w_nm_upload.value.strip(),
                                         folder=self.w_nm_folder.value.strip(), upload_name=self.w_nm_newname.value)
            link = f" · <a href='{r['gui_url']}' target='_blank'>open in NOMAD</a>" if r.get("gui_url") else ""
            self.w_nm_status4.value = _html_ok(f"{r['files']} file(s) uploaded to {r['upload_id']}/{r['folder']}") \
                + link
        self.job.run(work, err=lambda e: self._status(self.w_nm_status4, str(e), False))


def main(data_dir=None, work_dir=None):
    """Display the app in a notebook."""
    from IPython.display import display
    app = App(data_dir, work_dir)
    app._browse(app.data_dir)
    app._lib_refresh()
    display(app.ui)
    return app
