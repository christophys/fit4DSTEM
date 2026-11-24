# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD ELN entry, workflow (``workflow2``) and a preview of the workflow graph (tab 4).

Three archive files are written next to the results (upload them together with the data, the reference scan, the
libraries and the NXem file):

* ``<name>_datasets.archive.yaml``   ``nomad.datamodel.metainfo.eln.ElnFileManager`` listing the input files
  (4D-STEM data set, vacuum reference scan, rocking-curve libraries) -- core NOMAD, no plugin needed.
* ``<name>.archive.yaml``            the measurement ELN: ``fit4dstem_nomad.schema_packages.fourdstem.
  ELNFourDSTEMMeasurement`` (based on ``nomad_measurements.mapping.schema.MappingMeasurement`` of the
  nomad-measurements plugin; needs the fit4dstem NOMAD plugin), or -- ``schema="core"`` -- the core
  ``nomad.datamodel.metainfo.eln.ELNMeasurement``.  The processing steps (load, calibration, fit, export) are
  ``steps`` with their parameters; the description explains the workflow.
* ``<name>.workflow.archive.yaml``   ``workflow2``: inputs (data, reference, libraries) -> tasks (the four tabs) ->
  outputs (NXem results, measurement ELN), linked by references to sections of the entries above
  (``../upload/archive/mainfile/<file>#/<path>``), see https://docs.nomad-lab.eu/howto/manage/gui/workflows.html.

:func:`workflow_svg` draws the graph like NOMAD's workflow card (inputs left, tasks, outputs right).
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import os

import yaml

from . import __version__

PLUGIN_SECTION = "fit4dstem_nomad.schema_packages.fourdstem.ELNFourDSTEMMeasurement"
CORE_SECTION = "nomad.datamodel.metainfo.eln.ELNMeasurement"

STEPS = [
    ("Load 4D-STEM data", "Tab 1: lazy loading of the 4D-STEM data set and the vacuum reference scan "
                          "(HDF5 / NXem / Nion Swift, local or from a NOMAD upload with HTTP range requests)."),
    ("Calibrate patterns", "Tab 2: disc radius and edge, lattice of the discs, polynomial distortions (orders 2-N), "
                           "background between the discs, beam position maps (coarse sample -> interpolated -> "
                           "full pass), descan from the vacuum reference, crystal zone assignment."),
    ("Fit maps", "Tab 3: pattern-by-pattern maximum-likelihood fit of thickness, tilt, Ti displacement (and "
                 "SrTiO3 fraction) with plane-wave multislice rocking-curve libraries, neighbour seeding and "
                 "refinement passes; Gaussian / layer-aware smoothing; strain from the disc lattice; projected "
                 "field from the beam deflection; optional depth-resolved (slab) refinement."),
    ("Export NXem", "Tab 4: maps and calibration written as NeXus/NXem (validated with pynxtools); ELN and "
                    "workflow archives for NOMAD."),
]


def _iso_now():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def workflow_description(settings=None, calibration=None, options=None, results_summary=None):
    """Markdown description of the applied workflow (for the ELN)."""
    s = settings or {}
    lines = ["## 4D-STEM CBED mapping (fit4dstem %s)" % __version__, ""]
    if s:
        lines.append("**Acquisition:** " + ", ".join(f"{k} = {v}" for k, v in s.items()))
        lines.append("")
    lines.append("**Workflow**")
    lines.append("")
    for i, (n, d) in enumerate(STEPS, 1):
        lines.append(f"{i}. **{n}** – {d}")
    if calibration:
        lines += ["", "**Calibration:** " + html.escape(str(calibration))]
    if options:
        lines += ["", "**Fit options:** " + html.escape(json.dumps(options, default=str))]
    if results_summary:
        lines += ["", "**Results:** " + html.escape(str(results_summary))]
    return "\n".join(lines)


def _plain(o):
    """Convert numpy scalars / arrays (recursively) to plain Python for YAML."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if np is not None:
        if isinstance(o, np.ndarray):
            return _plain(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
    return o


def write_eln(folder, name, data_file, reference_file=None, results_file=None, library_files=(), settings=None,
              metadata=None, calibration=None, options=None, results_summary=None, schema="plugin", step_details=None,
              sample_name=None, prefix=""):
    """Write the datasets, measurement and workflow archives (see module docstring).  Paths of the files are given
    relative to the upload root; ``prefix`` is the folder of the upload the archives go to (default: the root).
    Returns dict of written paths."""
    pre = (prefix.strip("/") + "/") if prefix and prefix.strip("/") else ""
    os.makedirs(folder, exist_ok=True)
    settings = _plain(settings or {})
    now = _iso_now()
    meta = metadata or {}
    settings = dict(settings or {})
    # ---------------------------------------------------------------- datasets (core ELN file manager)
    files = [dict(file=data_file, description="4D-STEM data set (specimen)")]
    if reference_file:
        files.append(dict(file=reference_file, description="4D-STEM vacuum reference scan (descan)"))
    for lf in library_files or ():
        files.append(dict(file=lf, description="rocking-curve library (plane-wave multislice, fit4dstem)"))
    ds_name = f"{name}_datasets.archive.yaml"
    ds_arch = {"data": {"m_def": "nomad.datamodel.metainfo.eln.ElnFileManager",
                        "name": f"{name}: 4D-STEM input data",
                        "description": "Input files of the fit4dstem analysis.",
                        "Files": files}}
    # ---------------------------------------------------------------- measurement
    sec = PLUGIN_SECTION if schema == "plugin" else CORE_SECTION
    desc = workflow_description(settings, calibration, options, results_summary)
    smp = sample_name or (meta.get("sample") or {}).get("name") or "specimen"
    steps = []
    for i, (n, d) in enumerate(STEPS):
        st = {"name": n, "start_time": now, "comment": d}
        if step_details and i < len(step_details) and step_details[i]:
            st["comment"] += "\n\nParameters: " + json.dumps(step_details[i], default=str)
        steps.append(st)
    data = {"m_def": sec, "name": f"{name}: 4D-STEM CBED mapping", "datetime": now,
            "description": desc, "method": "4D-STEM (CBED mapping, fit4dstem)",
            "samples": [{"name": smp, "lab_id": str((meta.get("sample") or {}).get("identifier") or "")}],
            "instruments": [{"name": str((meta.get("instrument") or {}).get("name") or "Nion HERMES"),
                             "lab_id": str((meta.get("instrument") or {}).get("serial_number") or "")}],
            "steps": steps}
    if schema == "plugin":
        data["settings"] = {k: v for k, v in {
            "acceleration_voltage": settings.get("energy_eV"),
            "convergence_semi_angle": settings.get("convergence_mrad"),
            "detector_sampling": settings.get("mrad_per_px"),
            "scan_step": settings.get("scan_step_nm"),
            "scan_shape": settings.get("scan_shape"),
            "detector_shape": settings.get("det_shape"),
            "zone_axis": settings.get("zone"),
        }.items() if v is not None}
        data["datasets"] = [{"name": os.path.basename(f["file"]), "file": f["file"], "role": r}
                            for f, r in zip(files, ["specimen"] + (["vacuum reference"] if reference_file else [])
                                            + ["library"] * len(library_files or ()))]
        if results_file:
            data["nxem_file"] = results_file
        if results_summary:
            data["results_summary"] = str(results_summary)
    else:
        data["tags"] = ["4D-STEM", "CBED", "fit4dstem"]
    for s_ in data["samples"]:
        if not s_["lab_id"]:
            del s_["lab_id"]
    for s_ in data["instruments"]:
        if not s_["lab_id"]:
            del s_["lab_id"]
    me_name = f"{name}.archive.yaml"
    me_arch = {"data": data}
    # ---------------------------------------------------------------- workflow
    up = "../upload/archive/mainfile/"
    dsr = lambda i: f"{up}{pre}{ds_name}#/data/Files/{i}"           # noqa: E731
    stp = lambda i: f"{up}{pre}{me_name}#/data/steps/{i}"           # noqa: E731
    inputs = [{"name": "4D-STEM data", "section": dsr(0)}]
    if reference_file:
        inputs.append({"name": "vacuum reference", "section": dsr(1)})
    lib_links = [{"name": f"library {os.path.basename(lf)}", "section": dsr(len(inputs) + k)}
                 for k, lf in enumerate(library_files or ())]
    inputs += lib_links
    outputs = []
    if results_file:
        outputs.append({"name": "NXem results", "section": f"{up}{results_file}#/data"})
    outputs.append({"name": "measurement ELN", "section": f"{up}{pre}{me_name}#/data"})
    tasks = [
        {"m_def": "nomad.datamodel.metainfo.workflow.Task", "name": STEPS[0][0],
         "inputs": [dict(x) for x in inputs[:2 if reference_file else 1]],
         "outputs": [{"name": "loaded data", "section": stp(0)}]},
        {"m_def": "nomad.datamodel.metainfo.workflow.Task", "name": STEPS[1][0],
         "inputs": [{"name": "loaded data", "section": stp(0)}],
         "outputs": [{"name": "calibration", "section": stp(1)}]},
        {"m_def": "nomad.datamodel.metainfo.workflow.Task", "name": STEPS[2][0],
         "inputs": [{"name": "calibration", "section": stp(1)}] + [dict(x) for x in lib_links],
         "outputs": [{"name": "maps", "section": stp(2)}]},
        {"m_def": "nomad.datamodel.metainfo.workflow.Task", "name": STEPS[3][0],
         "inputs": [{"name": "maps", "section": stp(2)}],
         "outputs": [dict(x) for x in outputs] + [{"name": "export", "section": stp(3)}]},
    ]
    wf_name = f"{name}.workflow.archive.yaml"
    wf_arch = {"workflow2": {"name": f"{name}: fit4dstem 4D-STEM analysis", "inputs": inputs, "outputs": outputs,
                             "tasks": tasks}}
    paths = {}
    for fn, arch in ((ds_name, ds_arch), (me_name, me_arch), (wf_name, wf_arch)):
        p = os.path.join(folder, fn)
        with open(p, "w") as f:
            yaml.safe_dump(_plain(arch), f, sort_keys=False, allow_unicode=True, width=120)
        paths[fn] = p
    svg = workflow_svg(wf_arch["workflow2"])
    p = os.path.join(folder, f"{name}_workflow.svg")
    with open(p, "w") as f:
        f.write(svg)
    paths["svg"] = p
    return paths


# ====================================================================== preview ==


def _label(link):
    return link.get("name") or link.get("section", "").split("#")[-1]


def workflow_svg(wf, width=980):
    """SVG of the workflow graph: inputs (left) -> tasks (middle, in order) -> outputs (right), with the edges
    NOMAD derives from shared section references."""
    inputs, outputs, tasks = wf.get("inputs", []), wf.get("outputs", []), wf.get("tasks", [])
    col_x = [20, 330, width - 250]
    box_w = [230, 260, 230]
    row_h = 58
    n_rows = max(len(inputs), len(tasks), len(outputs), 1)
    H = 70 + n_rows * row_h + 20
    pos = {}

    def ypos(i, n):
        top = 70 + (n_rows - n) * row_h / 2
        return top + i * row_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" font-family="Helvetica, Arial, '
             f'sans-serif" font-size="12">',
             '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
             '<path d="M0,0 L10,4 L0,8 z" fill="#5a6b7b"/></marker></defs>',
             f'<text x="{col_x[0]}" y="30" font-size="14" font-weight="bold" fill="#2a3a4a">'
             f'{html.escape(wf.get("name", "workflow"))}</text>',
             f'<text x="{col_x[0]}" y="55" fill="#5a6b7b">inputs</text>',
             f'<text x="{col_x[1]}" y="55" fill="#5a6b7b">tasks</text>',
             f'<text x="{col_x[2]}" y="55" fill="#5a6b7b">outputs</text>']
    edges = []
    # sections -> node
    sec_owner = {}
    for i, l in enumerate(inputs):
        y = ypos(i, len(inputs))
        pos[("in", i)] = (col_x[0], y)
        sec_owner.setdefault(l.get("section"), []).append(("in", i))
        parts.append(f'<rect x="{col_x[0]}" y="{y}" rx="18" width="{box_w[0]}" height="40" fill="#e3f0fb" '
                     f'stroke="#2f80c3"/><text x="{col_x[0] + 14}" y="{y + 25}" fill="#123">'
                     f'{html.escape(_label(l))}</text>')
    for i, t in enumerate(tasks):
        y = ypos(i, len(tasks))
        pos[("task", i)] = (col_x[1], y)
        parts.append(f'<rect x="{col_x[1]}" y="{y}" rx="4" width="{box_w[1]}" height="40" fill="#fff1df" '
                     f'stroke="#e08a1e"/><text x="{col_x[1] + 14}" y="{y + 25}" fill="#321">'
                     f'{i + 1}. {html.escape(t.get("name", "task"))}</text>')
    for i, l in enumerate(outputs):
        y = ypos(i, len(outputs))
        pos[("out", i)] = (col_x[2], y)
        parts.append(f'<rect x="{col_x[2]}" y="{y}" rx="18" width="{box_w[2]}" height="40" fill="#e5f6e8" '
                     f'stroke="#2e9e4f"/><text x="{col_x[2] + 14}" y="{y + 25}" fill="#132">'
                     f'{html.escape(_label(l))}</text>')
    # edges: input -> task when a task input shares the section of a workflow input; task -> task when an output
    # section of one task is an input section of a later task; task -> output likewise.
    task_out = {}
    for i, t in enumerate(tasks):
        for l in t.get("outputs", []):
            task_out.setdefault(l.get("section"), []).append(i)
    for i, t in enumerate(tasks):
        for l in t.get("inputs", []):
            s = l.get("section")
            for j, w in enumerate(inputs):
                if w.get("section") == s:
                    edges.append((("in", j), ("task", i)))
            for k in task_out.get(s, []):
                if k != i:
                    edges.append((("task", k), ("task", i)))
    for j, w in enumerate(outputs):
        for k in task_out.get(w.get("section"), []):
            edges.append((("task", k), ("out", j)))
    for a, b in dict.fromkeys(edges):
        xa, ya = pos[a]
        xb, yb = pos[b]
        wa = box_w[{"in": 0, "task": 1, "out": 2}[a[0]]]
        if a[0] == "task" and b[0] == "task":
            # vertical edge on the right side of the task column
            x = xa + wa
            parts.append(f'<path d="M{x},{ya + 20} C{x + 30},{ya + 20} {x + 30},{yb + 20} {x},{yb + 20}" '
                         f'fill="none" stroke="#5a6b7b" marker-end="url(#arr)"/>')
        else:
            parts.append(f'<path d="M{xa + wa},{ya + 20} C{(xa + wa + xb) / 2},{ya + 20} {(xa + wa + xb) / 2},'
                         f'{yb + 20} {xb},{yb + 20}" fill="none" stroke="#5a6b7b" marker-end="url(#arr)"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def preview_html(wf_path_or_dict):
    """HTML (inline SVG) preview of a workflow archive (yaml path or workflow2 dict)."""
    if isinstance(wf_path_or_dict, str):
        with open(wf_path_or_dict) as f:
            wf = yaml.safe_load(f)["workflow2"]
    else:
        wf = wf_path_or_dict
    return workflow_svg(wf)
