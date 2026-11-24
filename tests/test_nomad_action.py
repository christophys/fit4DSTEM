# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Headless analysis with the parameters of a previous analysis (fit4dstem.pipeline) and the NOMAD action on a real
Temporal server (Temporal CLI dev server; skipped if it is not available), with a local folder as the upload."""
import asyncio
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fit4dstem import calibration as calm  # noqa: E402
from fit4dstem import crystal as cr  # noqa: E402
from fit4dstem import eln, nxem, pipeline  # noqa: E402
from fit4dstem import fitting as fm  # noqa: E402
from fit4dstem import io as fio  # noqa: E402
from fit4dstem import multislice as ms  # noqa: E402
from fit4dstem import synthetic as syn  # noqa: E402

SCAN = (6, 7)


@pytest.fixture(scope="module")
def upload(tmp_path_factory):
    """Upload U1 with a previous analysis of scan A (NXem with calibration + ELN + libraries) and a new scan B."""
    root = tmp_path_factory.mktemp("uploads")
    up = root / "U1"
    (up / "previous").mkdir(parents=True)
    (up / "raw").mkdir()
    libs = {}
    for name in ("BaTiO3", "SrTiO3"):
        p = str(up / "previous" / f"lib_{name}_100.h5")
        ms.build_library(cr.MATERIALS[name], "[100]", 60e3, radius_mrad=5.0, step_mrad=0.7, g_max_inv_A=0.6,
                         max_thickness_nm=14, u_pm=(-20, 0, 20), w_pm=(-20, 0, 20), sampling=0.12, path=p, log=None)
        libs[name] = ms.Library.load(p)

    def make(seed):
        np.random.seed(seed)
        return syn.make_dataset(libs, scan_shape=SCAN, det_shape=(96, 96), mrad_per_px=0.6, alpha_mrad=3.0, dose=4e5,
                                truth=syn.truth_fields(SCAN, t_range_nm=(9.0, 12.0), amp_pm=12.0))
    A, B = make(1), make(2)
    pa = fio.write_nion_h5(str(up / "raw" / "scan_A.h5"), A["data"], 0.25, 0.6, 60e3, 3.0)
    fio.write_nion_h5(str(up / "raw" / "scan_B.h5"), B["data"], 0.25, 0.6, 60e3, 3.0)
    # the interactive analysis of scan A (calibration and zone assignment as in the app)
    ds = fio.open_dataset(pa)
    pos = fio.coarse_positions(ds.scan_shape, 3)
    fr = ds.frames(pos)
    cal = calm.initial_calibration(fr)
    calm.coarse_calibration(fr, pos, cal, distortion_order=3)
    best = min(calm.zone_candidates(cal.G0, libs["BaTiO3"].g[np.hypot(*libs["BaTiO3"].g.T) < 0.4]),
               key=lambda c: np.abs(c["M"] - A["M0"]).sum())
    cal.B_cry = best["B"]
    cal.mrad_per_px = cr.wavelength_A(60e3) * 1e3 / best["px_per_invA"]
    cal.radius_px, cal.edge_px = 3.0 / 0.6, 0.8
    ds.close()
    opt = fm.FitOptions(mixture=True, t_grid_step_nm=1.0, seed_uw=(-10, 0, 10))
    nxem.write_results(str(up / "previous" / "scan_A_fit4dstem.nxs"),
                       {"thickness": (np.zeros(SCAN), "nm", "thickness", "CBED fit")}, (0.25, 0.25), energy_eV=60e3,
                       convergence_mrad=3.0, calibration=cal.to_dict(), options=opt.__dict__,
                       inputs=[dict(file="raw/scan_A.h5", description="4D-STEM data set (specimen)")],
                       extra={"libraries": {k: pipeline.library_meta(v) for k, v in libs.items()}})
    eln.write_eln(str(up / "previous"), "scan_A_fit4dstem", data_file="raw/scan_A.h5",
                  results_file="previous/scan_A_fit4dstem.nxs",
                  library_files=[f"previous/lib_{n}_100.h5" for n in libs], prefix="previous",
                  settings=dict(energy_eV=60e3, convergence_mrad=3.0, scan_step_nm=0.25, zone="[100]"))
    return dict(root=root, up=up, truth=B["truth"])


def _check_maps(path, truth):
    import h5py
    with h5py.File(path) as f:
        found = {}

        def visit(name, obj):
            if name.endswith("reconstruction/parameters/quantity"):
                found[obj[()].decode()] = name.rsplit("/reconstruction", 1)[0]
        f.visititems(visit)
        t = f[found["thickness"] + "/image_2d/real"][()]
    assert t.shape == SCAN and np.sqrt(np.nanmean((t - truth["thickness_nm"]) ** 2)) < 1.0
    return found


@pytest.mark.parametrize("prior", ["previous/scan_A_fit4dstem.nxs", "previous/scan_A_fit4dstem.archive.yaml"])
def test_pipeline_with_previous_parameters(upload, tmp_path, prior):
    up = upload["up"]
    pr = pipeline.read_prior(str(up / prior), resolve=lambda f: str(up / f))
    assert pr.calibration and pr.zone == "[100]" and pr.energy_eV == 60e3
    assert set(pr.materials) == {"BaTiO3", "SrTiO3"} and pr.materials["SrTiO3"]["c"] == cr.MATERIALS["SrTiO3"].c
    assert pr.library_meta["BaTiO3"]["u_pm"] == [-20.0, 0.0, 20.0]
    if prior.endswith(".yaml"):
        assert pr.library_files == ["previous/lib_BaTiO3_100.h5", "previous/lib_SrTiO3_100.h5"]
    r = pipeline.run(str(up / "raw" / "scan_B.h5"), pr, str(tmp_path),
                     library_paths=[str(up / "previous" / f"lib_{n}_100.h5") for n in ("BaTiO3", "SrTiO3")],
                     seeds=2, refine_passes=1)
    maps = _check_maps(r["files"][0], upload["truth"])
    assert {"thickness", "Ti_displacement_u", "strain_yy", "polarization_x"} <= set(maps)
    names = sorted(os.path.basename(f) for f in r["files"])
    assert "scan_B_fit4dstem.workflow.archive.yaml" in names and "scan_B_fit4dstem.archive.yaml" in names


def test_prior_without_calibration_is_refused(upload, tmp_path):
    pr = pipeline.Prior(source="x.nxs", energy_eV=60e3)
    with pytest.raises(ValueError, match="no calibration"):
        pipeline.run(str(upload["up"] / "raw" / "scan_B.h5"), pr, str(tmp_path),
                     library_paths=[str(upload["up"] / "previous" / "lib_BaTiO3_100.h5")])


# ------------------------------------------------------------------------------------------------ Temporal
def _cli():
    for c in (os.environ.get("TEMPORAL_CLI"), shutil.which("temporal"), "/home/claude/temporal/temporal"):
        if c and os.path.isfile(c):
            return c
    return None


@pytest.fixture(scope="module")
def temporal():
    pytest.importorskip("temporalio")
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment
    try:
        env = asyncio.new_event_loop().run_until_complete(
            WorkflowEnvironment.start_local(data_converter=pydantic_data_converter, dev_server_existing_path=_cli()))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Temporal dev server: {e}")
    yield env
    asyncio.new_event_loop().run_until_complete(env.shutdown())


@pytest.mark.parametrize("prior", ["previous/scan_A_fit4dstem.archive.yaml", "previous/scan_A_fit4dstem.nxs"])
def test_action_workflow(upload, temporal, prior):
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Worker

    from fit4dstem_nomad.workflows import backend
    from fit4dstem_nomad.workflows.activities import cleanup_map_fit, publish_map_fit, run_map_fit, stage_map_fit
    from fit4dstem_nomad.workflows.map_fit_workflow import MapFitWorkflow
    from fit4dstem_nomad.workflows.models import MapFitInput
    backend.set_backend(backend.LocalBackend(str(upload["root"])))
    folder = "results_" + prior.rsplit(".", 1)[-1]

    async def go():
        client = await Client.connect(temporal.client.service_client.config.target_host,
                                      data_converter=pydantic_data_converter)
        async with Worker(client, task_queue="cpu-task-queue", workflows=[MapFitWorkflow],
                          activities=[stage_map_fit, run_map_fit, publish_map_fit, cleanup_map_fit],
                          activity_executor=ThreadPoolExecutor(4)):
            return await client.execute_workflow(
                MapFitWorkflow.run, MapFitInput(upload_id="U1", user_id="me", data_path="raw/scan_B.h5",
                                                parameters_from=prior, seeds=2, output_folder=folder),
                id=f"fit4dstem-{uuid.uuid4()}", task_queue="cpu-task-queue")

    res = asyncio.new_event_loop().run_until_complete(go())
    up = upload["up"]
    assert f"{folder}/scan_B_fit4dstem.nxs" in res.files and res.parameters_from == prior
    _check_maps(str(up / folder / "scan_B_fit4dstem.nxs"), upload["truth"])
    import yaml
    me = yaml.safe_load((up / folder / "scan_B_fit4dstem.archive.yaml").read_text())["data"]
    roles = {d["role"]: d["file"] for d in me["datasets"]}
    assert roles["specimen"] == "raw/scan_B.h5" and me["nxem_file"] == f"{folder}/scan_B_fit4dstem.nxs"
    assert sorted(d["file"] for d in me["datasets"] if d["role"] == "library") == [
        "previous/lib_BaTiO3_100.h5", "previous/lib_SrTiO3_100.h5"]
    wf = yaml.safe_load((up / folder / "scan_B_fit4dstem.workflow.archive.yaml").read_text())["workflow2"]
    assert wf["outputs"][0]["section"] == f"../upload/archive/mainfile/{folder}/scan_B_fit4dstem.nxs#/data"
    assert not [d for d in os.listdir(upload["root"] / "_work") if d] if (upload["root"] / "_work").exists() else True


def test_missing_libraries_are_rebuilt_from_the_previous_result(upload, tmp_path):
    """Crystal structures, zone and build parameters (incl. the u/w grid) stored in the NXem file suffice."""
    pr = pipeline.read_prior(str(upload["up"] / "previous" / "scan_A_fit4dstem.nxs"))
    libs = pipeline.load_libraries(pr, (), build_missing=True, work_dir=str(tmp_path))
    old = ms.Library.load(str(upload["up"] / "previous" / "lib_BaTiO3_100.h5"))
    new = libs["BaTiO3"]
    assert new.meta["zone"] == "[100]" and np.allclose(new.u, old.u) and new.data.shape == old.data.shape
    assert np.allclose(new.data.astype(float), old.data.astype(float), atol=2e-3)
    with pytest.raises(ValueError, match="libraries"):
        pipeline.load_libraries(pr, (), build_missing=False, work_dir=str(tmp_path))
