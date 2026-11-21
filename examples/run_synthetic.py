"""End-to-end example on synthetic data (also used as an integration test).

    python examples/run_synthetic.py [workdir]

1. builds small [100] rocking-curve libraries for BaTiO3 and SrTiO3 (if not present; a few minutes),
2. writes a synthetic superlattice 4D-STEM scan and a vacuum reference scan in the Nion Swift HDF5 layout,
3. runs the calibration (tab 2), the map fit with neighbour seeding (tab 3), strain and field maps,
4. writes the NXem file (validated with pynxtools) and the NOMAD ELN / workflow archives (tab 4).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fit4dstem import calibration as calm  # noqa: E402
from fit4dstem import crystal as cr  # noqa: E402
from fit4dstem import eln  # noqa: E402
from fit4dstem import fitting as fm  # noqa: E402
from fit4dstem import io as fio  # noqa: E402
from fit4dstem import multislice as ms  # noqa: E402
from fit4dstem import nxem  # noqa: E402
from fit4dstem import reference as ref  # noqa: E402
from fit4dstem import strain as st  # noqa: E402
from fit4dstem import synthetic as syn  # noqa: E402


def main(work="synthetic_work", scan=(28, 20), quick=True):
    os.makedirs(work, exist_ok=True)
    libs = {}
    for name in ("BaTiO3", "SrTiO3"):
        p = os.path.join(work, f"lib_{name}_100.h5")
        if not os.path.exists(p):
            print("building library", name, flush=True)
            ms.build_library(cr.MATERIALS[name], "[100]", 60e3, radius_mrad=6.5, step_mrad=0.5, g_max_inv_A=0.8,
                             max_thickness_nm=36, u_pm=(-30, -15, 0, 15, 30), w_pm=(-30, -15, 0, 15, 30),
                             sampling=0.1, absorption=0.03, path=p, log=None)
        libs[name] = ms.Library.load(p)
    S = syn.make_dataset(libs, scan_shape=scan, det_shape=(128, 128), dose=2e5)
    data_path = fio.write_nion_h5(os.path.join(work, "synthetic_superlattice.h5"), S["data"], 0.25, 0.6, 60e3, 4.0)
    ref_path = fio.write_nion_h5(os.path.join(work, "synthetic_vacuum.h5"), S["ref"], 0.25, 0.6, 60e3, 4.0)
    np.savez(os.path.join(work, "truth.npz"), **{k: v for k, v in S["truth"].items()}, centre=S["centre"],
             descan=S["descan"], strain_yy=S["strain_yy"], deflection_mrad=S["deflection_mrad"])

    # ---------------------------------------------------------------- tab 1
    ds = fio.open_dataset(data_path)
    dr = fio.open_dataset(ref_path)
    print(ds.describe())
    # ---------------------------------------------------------------- tab 2
    pos = fio.coarse_positions(ds.scan_shape, 5)
    frames = ds.frames(pos)
    cal = calm.initial_calibration(frames)
    meas = calm.coarse_calibration(frames, pos, cal, distortion_order=3, log=print)
    g = libs["BaTiO3"].g[np.hypot(*libs["BaTiO3"].g.T) < 0.4]
    cands = calm.zone_candidates(cal.G0, g)
    # choose the candidate consistent with the synthetic geometry (the app lets the user choose)
    M_true = S["M0"]
    best = min(cands, key=lambda c: np.abs(c["M"] - M_true).sum())
    cal.B_cry = best["B"]
    cal.mrad_per_px = cr.wavelength_A(60e3) * 1e3 / best["px_per_invA"]
    # radius / edge from the vacuum reference (uniform disc)
    rf = dr.frame(0, 0)
    c0 = ref.centre_of_disc(rf, cal.radius_px, cal.edge_px)
    R, w, _, _ = calm.edge_profile(rf, [c0], cal.radius_px)
    cal.radius_px, cal.edge_px = R, w
    print("aperture from the vacuum reference:", R, w)
    # background parameters (cross-validated)
    masks, dists = [], []
    for f, m in zip(frames, meas):
        masks.append(cal.disc_mask(m["c"], m["G"], margin=cal.bg_margin))
        dists.append(calm.distance_outside_discs(cal.det_shape, m["c"], m["G"], cal.radius_px, cal.ensure_U()))
    cal.bg_sigma, scores = calm.choose_background_sigma(list(frames), masks, dist_maps=dists)
    print(cal.summary())
    # model-based refinement of the distortion (disc positions from the fitted model of the coarse patterns)
    opt0 = fm.FitOptions(mixture=True, refine_lattice=False, t_grid_step_nm=2.0, seed_uw=(-15.0, 0.0, 15.0))
    k = np.linspace(0, len(frames) - 1, 9).astype(int)
    fit0 = fm.PatternFitter(fm.DetectorModel(libs, cal), opt0)
    lats, pars = [], []
    for i in k:
        m = cal.measure(frames[i])
        p, _, _ = fit0.fit(fm.reduce_frame(frames[i], m["c"], m["G"], fit0.model), global_search=True, n_local=2)
        lats.append((m["c"], m["G"]))
        pars.append(p)
    fm.refine_distortion(lambda: fm.PatternFitter(fm.DetectorModel(libs, cal), opt0), cal, [frames[i] for i in k],
                         lats, pars, order=3, n_iter=2, log=print)
    meas = [cal.measure(f) for f in frames]
    cal.coarse_centres = np.array([m["c"] for m in meas])
    cal.coarse_G = np.array([m["G"] for m in meas])
    calm.full_pass(ds, cal, log=print)
    # vacuum reference -> descan, deflection
    ref_c = ref.measure_reference(dr, cal.radius_px, cal.edge_px, full_shape=ds.det_shape,
                                  distortion=cal.distortion)
    desc = ref.descan_map(ref_c, ds.scan_shape, mode="same")
    M = st.mapping_maps(cal.G_map, cal.B_cry)
    Mmean = np.nanmean(M.reshape(-1, 2, 2), 0)
    defl = ref.deflection_maps(cal.centre_map, desc, Mmean, 60e3)
    # ---------------------------------------------------------------- tab 3
    model = fm.DetectorModel(libs, cal)
    opt = fm.FitOptions(mixture=True, seed_uw=(-15.0, 0.0, 15.0), seed_tilt=(-0.8, 0.0, 0.8),
                        t_grid_step_nm=2.0)
    mf = fm.MapFitter(ds, cal, model, opt, scan_bin=1, log=print)
    t0 = time.time()
    mf.load()
    seeds = fm.seed_grid(mf.shape, 3)
    res = mf.run(seeds, refine_passes=1)
    print(f"map fit {time.time() - t0:.0f} s")
    T = S["truth"]
    for k, name in enumerate(res.names[:5]):
        err = res.params[..., k] - T[name]
        print(f"{name:14s} rms error {np.sqrt(np.nanmean(err ** 2)):.3f}  median sigma "
              f"{np.nanmedian(res.sigma[..., k]):.3f}")
    f_err = np.mean((res.params[..., 5] > 0.5) != (T["material"] == 1))
    print("material misclassified:", f_err)
    seg = fm.segment_layers(res.params[..., 5])
    S0 = st.strain_maps(cal.G_map, cal.B_cry, ref_mask=seg == 0)
    S_ = st.strain_maps(res.lattice, cal.B_cry, ref_mask=seg == 0)
    for lab, SS in (("edge correlation (tab 2)", S0), ("model-based (tab 3)", S_)):
        e_sto, e_bto = SS["eyy"][seg == 1], SS["eyy"][seg == 0]
        print(f"strain from the {lab}: eyy SrTiO3 {np.nanmean(e_sto):+.4f} (true -0.02) +- {np.nanstd(e_sto):.4f}, "
              f"BaTiO3 +- {np.nanstd(e_bto):.4f}; exx SrTiO3 {np.nanmean(SS['exx'][seg == 1]):+.4f} (true 0)")
    t_map = res.params[..., 0]
    integ, E = ref.field_maps(defl, t_map, 60e3)
    defl2 = ref.deflection_maps(res.centre, desc, Mmean, 60e3)
    for lab, dd in (("edge correlation", defl), ("model-based", defl2)):
        print(f"deflection rms error ({lab}): {np.sqrt(np.nanmean((dd - S['deflection_mrad']) ** 2)):.4f} mrad "
              f"(true field step {np.abs(S['deflection_mrad']).max():.3f} mrad)")
    defl = defl2
    # ---------------------------------------------------------------- tab 4
    P = cr.polarization(cr.MATERIALS["BaTiO3"], "[100]", res.params[..., 3], res.params[..., 4])
    maps = {"thickness": (res.params[..., 0], "nm", "thickness", "CBED fit"),
            "tilt_x": (res.params[..., 1], "mrad", "crystal tilt about y", "CBED fit"),
            "tilt_y": (res.params[..., 2], "mrad", "crystal tilt about x", "CBED fit"),
            "Ti_displacement_u": (res.params[..., 3], "pm", "Ti displacement in-plane", "CBED fit"),
            "Ti_displacement_w": (res.params[..., 4], "pm", "Ti displacement growth direction", "CBED fit"),
            "SrTiO3_fraction": (res.params[..., 5], "", "SrTiO3 fraction", "CBED fit"),
            "strain_xx": (S_["exx"], "", "strain in-plane", "disc lattice"),
            "strain_yy": (S_["eyy"], "", "strain growth direction", "disc lattice"),
            "deflection_y": (defl[..., 1], "mrad", "beam deflection (growth)", "beam position - vacuum reference"),
            "polarization_x": (P[..., 0], "uC/cm^2", "polarization estimate in-plane", "Born effective charges"),
            "polarization_y": (P[..., 1], "uC/cm^2", "polarization estimate growth", "Born effective charges")}
    out = os.path.join(work, "synthetic_results.nxs")
    meta = nxem.default_metadata()
    meta["sample"]["is_simulation"] = True
    meta["sample"]["name"] = "synthetic BaTiO3/SrTiO3 superlattice"
    nxem.write_results(out, maps, step_nm=(0.25, 0.25), metadata=meta, energy_eV=60e3, convergence_mrad=4.0,
                       calibration=cal.to_dict(), options=opt.__dict__,
                       inputs=[nxem.file_info(data_path), nxem.file_info(ref_path)],
                       mean_pattern=frames.mean(0), mrad_per_px=cal.mrad_per_px)
    ok, problems, _ = nxem.validate(out)
    print("NXem valid:", ok, problems[:5])
    files = eln.write_eln(work, name="synthetic", data_file=os.path.basename(data_path),
                          reference_file=os.path.basename(ref_path), results_file=os.path.basename(out),
                          library_files=[f"lib_{n}_100.h5" for n in libs], settings=dict(
                              energy_eV=60e3, convergence_mrad=4.0, scan_shape=list(ds.scan_shape),
                              det_shape=list(ds.det_shape), scan_step_nm=0.25, mrad_per_px=cal.mrad_per_px,
                              zone="[100]"), metadata=meta)
    print("ELN / workflow:", files)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_work")
