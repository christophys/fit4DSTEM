import io
import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fit4dstem import calibration as calm  # noqa: E402
from fit4dstem import crystal as cr  # noqa: E402
from fit4dstem import eln  # noqa: E402
from fit4dstem import fitting as fm  # noqa: E402
from fit4dstem import io as fio  # noqa: E402
from fit4dstem import multislice as ms  # noqa: E402
from fit4dstem import nxem  # noqa: E402
from fit4dstem import strain as st  # noqa: E402
from fit4dstem import synthetic as syn  # noqa: E402
from fit4dstem.nomad import RemoteFile  # noqa: E402


@pytest.fixture(scope="session")
def libs(tmp_path_factory):
    d = tmp_path_factory.mktemp("libs")
    out = {}
    for name in ("BaTiO3", "SrTiO3"):
        p = str(d / f"lib_{name}.h5")
        ms.build_library(cr.MATERIALS[name], "[100]", 60e3, radius_mrad=5.0, step_mrad=0.7, g_max_inv_A=0.6,
                         max_thickness_nm=14, u_pm=(-20, 0, 20), w_pm=(-20, 0, 20), sampling=0.12, path=p, log=None)
        out[name] = ms.Library.load(p)
    return out


def test_zone_cells():
    for zone, n in (("[100]", 5), ("[110]", 10)):
        at = cr.zone_cell(cr.MATERIALS["BaTiO3"], zone, 10, -5)
        assert len(at) == n
    P = cr.polarization(cr.MATERIALS["BaTiO3"], "[110]", 0.0, 9.0)
    assert P[1] > 20 and abs(P[0]) < 1e-9


def test_engine_reciprocity_and_tilt():
    eng = ms.Engine(cr.MATERIALS["BaTiO3"], "[100]", 60e3, sampling=0.12)
    hk, g = cr.reflections(cr.MATERIALS["BaTiO3"], "[100]", 0.6)
    I = eng.run([(5, 0, 0)], [[0, 0], [1.0, 0.5]], hk)
    assert I.shape == (1, 2, len(hk))
    assert np.all(I >= 0) and I[0, 0].sum() <= 1.0 + 1e-4
    # centrosymmetric projection at zero tilt: I(g) = I(-g)
    i = {tuple(h): k for k, h in enumerate(hk)}
    for h in hk:
        assert abs(I[0, 0, i[tuple(h)]] - I[0, 0, i[(-h[0], -h[1])]]) < 1e-5


def test_library_interpolation(libs):
    lib = libs["BaTiO3"]
    c = lib.curves(lib.cells[3], lib.u[1], lib.w[2])
    assert np.allclose(c, lib.data[1, 2, 3].astype(np.float32), atol=1e-6)


def test_remote_file_h5(tmp_path):
    p = tmp_path / "d.h5"
    data = np.random.default_rng(0).random((4, 5, 16, 16)).astype(np.float32)
    fio.write_nion_h5(str(p), data)
    raw = p.read_bytes()
    rf = RemoteFile(reader=lambda off, n: raw[off:off + n], size=len(raw), block_size=4096)
    ds = fio.open_hdf5(rf, name="remote")
    assert ds.shape == (4, 5, 16, 16)
    assert np.allclose(ds.frame(2, 3), data[2, 3])
    assert ds.meta["energy_eV"] == 60e3 and abs(ds.meta["convergence_mrad"] - 4.0) < 1e-9
    assert rf.bytes_fetched < len(raw) * 1.5


def test_calibration_and_fit(libs):
    S = syn.make_dataset(libs, scan_shape=(4, 5), det_shape=(96, 96), mrad_per_px=0.6, alpha_mrad=3.0, dose=4e5,
                         truth=syn.truth_fields((4, 5), t_range_nm=(9.0, 12.0), amp_pm=12.0))
    d = S["data"]
    pos = fio.coarse_positions((4, 5), 3)
    fr = np.array([d[p] for p in pos])
    cal = calm.initial_calibration(fr)
    calm.coarse_calibration(fr, pos, cal, distortion_order=3)
    m = cal.measure(d[1, 2])
    assert np.hypot(*(m["c"] - S["centre"][1, 2])) < 0.3
    best = min(calm.zone_candidates(cal.G0, libs["BaTiO3"].g[np.hypot(*libs["BaTiO3"].g.T) < 0.4]),
               key=lambda c: np.abs(c["M"] - S["M0"]).sum())
    cal.B_cry = best["B"]
    cal.mrad_per_px = cr.wavelength_A(60e3) * 1e3 / best["px_per_invA"]
    cal.radius_px, cal.edge_px = 3.0 / 0.6, 0.8
    model = fm.DetectorModel(libs, cal)
    fit = fm.PatternFitter(model, fm.FitOptions(mixture=True, t_grid_step_nm=1.0, seed_uw=(-10, 0, 10)))
    red = fm.reduce_frame(d[1, 2], m["c"], m["G"], model)
    p, cost, s = fit.fit(red, global_search=True, n_local=2)
    T = S["truth"]
    assert abs(p[0] - T["thickness_nm"][1, 2]) < 0.5
    assert abs(p[3] - T["u_pm"][1, 2]) < 3 and abs(p[4] - T["w_pm"][1, 2]) < 3


def test_strain_identity():
    G = np.tile(np.array([[10.0, 1.0], [-1.0, 10.0]]), (3, 4, 1, 1))
    G[1, 1] = G[1, 1] @ np.diag([1.0, 1 / 1.02])        # 2 % larger lattice along y (reciprocal shorter)
    s = st.strain_maps(G, np.eye(2))
    assert abs(s["eyy"][1, 1] - 0.02 * 11 / 12) < 5e-3  # relative to the mean of all 12 positions
    assert abs(s["exx"][0, 0]) < 1e-9


def test_nxem_and_eln(tmp_path):
    maps = {"thickness": (np.ones((5, 6)), "nm", "thickness", "test"),
            "Ti_displacement_u": (np.zeros((5, 6)), "pm", "u", "test")}
    p = str(tmp_path / "r.nxs")
    meta = nxem.default_metadata()
    meta["sample"]["is_simulation"] = True
    nxem.write_results(p, maps, (0.2, 0.2), metadata=meta, convergence_mrad=4.0, mean_pattern=np.ones((8, 8)),
                       mrad_per_px=0.5)
    with h5py.File(p) as f:
        assert f["entry1/definition"][()] in (b"NXem", "NXem")
    ok, problems, _ = nxem.validate(p)
    if ok is not None:
        assert ok, problems
    files = eln.write_eln(str(tmp_path), "t", "data.h5", "ref.h5", "r.nxs", ["lib.h5"],
                          settings=dict(energy_eV=np.float64(60e3)))
    import yaml
    wf = yaml.safe_load(open(files["t.workflow.archive.yaml"]))["workflow2"]
    assert len(wf["tasks"]) == 4 and wf["inputs"][0]["section"].endswith("#/data/Files/0")
    svg = open(files["svg"]).read()
    assert svg.count("<rect") == len(wf["inputs"]) + len(wf["tasks"]) + len(wf["outputs"])


def test_app_builds(tmp_path):
    from fit4dstem.app import App
    app = App(data_dir=str(tmp_path), work_dir=str(tmp_path / "w"))
    assert len(app.tabs.children) == 4
