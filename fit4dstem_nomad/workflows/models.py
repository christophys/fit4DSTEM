# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Input (= the form NOMAD shows for the action) and result of the fit4dstem map-fit action."""
from pydantic import BaseModel, Field


class MapFitInput(BaseModel):
    """Fit the maps (thickness, tilt, Ti displacement, strain, field) of a 4D-STEM scan in a NOMAD upload with the
    parameters (calibration, crystal structures, libraries, fit options) of a previous fit4dstem analysis."""

    upload_id: str = Field(..., description="ID of the NOMAD upload with the 4D-STEM data (not published); the "
                                            "results are written into it")
    user_id: str = Field("", description="set by NOMAD to the user who starts the action")
    data_path: str = Field(..., description="4D-STEM data set in the upload (HDF5 / NXem / Nion Swift / .npy ...)")
    parameters_from: str = Field(..., description="previous analysis in the same upload: its NXem results file "
                                                  "(*.nxs written by fit4dstem) or its ELN entry "
                                                  "(*.archive.yaml / *.archive.json of the 4D-STEM CBED mapping ELN)")
    reference_path: str = Field("", description="vacuum reference scan in the upload (descan, beam deflection, "
                                                "field); default: none")
    library_paths: list[str] = Field(default_factory=list,
                                     description="rocking-curve library files in the upload; default: those of the "
                                                 "ELN entry, or the lib_*.h5 files next to the previous results")
    build_missing_libraries: bool = Field(False, description="build libraries that are not found from the crystal "
                                                             "structures and build parameters of the previous "
                                                             "analysis (can take hours)")
    dataset: str = Field("", description="HDF5 path of the 4D data set inside the file (default: automatic)")
    det_bin: int = Field(1, ge=1, le=16, description="detector binning (must match the previous calibration)")
    recalibrate: bool = Field(True, description="coarse calibration of the new scan, starting from the previous one")
    full_pass: bool = Field(True, description="measure beam position and lattice in every pattern before the fit")
    scan_bin: int = Field(1, ge=1, le=16, description="binning of the scan for the fit")
    seeds: int = Field(4, ge=1, le=32, description="seed patterns per scan direction (global search)")
    refine_passes: int = Field(1, ge=0, le=5, description="refinement passes with neighbour seeding")
    energy_kV: float | None = Field(None, description="high tension, if it differs from the previous analysis")
    convergence_mrad: float | None = Field(None, description="convergence semi-angle, if different")
    scan_step_nm: float | None = Field(None, description="scan step, if not in the data file")
    zone_axis: str | None = Field(None, description="zone axis, e.g. [110] (default: as before)")
    output_folder: str = Field("fit4dstem_results", description="folder in the upload for the results")
    output_name: str = Field("", description="base name of the result files (default: <data>_fit4dstem)")
    timeout_hours: float = Field(24.0, gt=0, le=168, description="maximum run time")

    def overrides(self):
        return {k: v for k, v in dict(energy_kV=self.energy_kV, convergence_mrad=self.convergence_mrad,
                                      scan_step_nm=self.scan_step_nm, zone=self.zone_axis).items() if v}


class StagedInputs(BaseModel):
    """Local paths of the inputs (on the worker) and their paths in the upload."""
    work_dir: str
    data: str
    reference: str = ""
    prior: str
    libraries: list[str] = Field(default_factory=list)
    names: dict = Field(default_factory=dict)       # role -> path in the upload
    prior_summary: str = ""


class MapFitResult(BaseModel):
    upload_id: str
    files: list[str] = Field(default_factory=list, description="paths of the written files in the upload")
    summary: str = ""
    parameters_from: str = ""
    runtime_s: float = 0.0
    message: str = ""
