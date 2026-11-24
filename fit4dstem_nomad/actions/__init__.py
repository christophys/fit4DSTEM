# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD action ``fit4dstem_nomad.actions:fit4dstem_map_fit``: maps of a 4D-STEM scan of an upload with the
parameters of a previous fit4dstem analysis (its NXem results or its ELN entry), on a NOMAD CPU worker.

Started from NOMAD's Actions page (form generated from :class:`~fit4dstem_nomad.workflows.models.MapFitInput`), from
the fit4dstem dashboard (tab 4) or via the API::

    POST {api}/v1/actions/fit4dstem_nomad.actions:fit4dstem_map_fit/start
    {"data": {"upload_id": "...", "data_path": "raw/scan_02.h5", "parameters_from": "results/scan_01.archive.yaml"}}
"""
from nomad.actions import TaskQueue
from nomad.config.models.plugins import ActionEntryPoint
from pydantic import Field

ACTION_ID = "fit4dstem_nomad.actions:fit4dstem_map_fit"


class Fit4DSTEMActionEntryPoint(ActionEntryPoint):
    task_queue: str = Field(default=TaskQueue.CPU, description="task queue of the worker")

    def load(self):
        from nomad.actions import Action

        from ..workflows.activities import cleanup_map_fit, publish_map_fit, run_map_fit, stage_map_fit
        from ..workflows.map_fit_workflow import MapFitWorkflow

        return Action(task_queue=self.task_queue, workflow=MapFitWorkflow,
                      activities=[stage_map_fit, run_map_fit, publish_map_fit, cleanup_map_fit])


fit4dstem_map_fit = Fit4DSTEMActionEntryPoint(
    name="fit4dstem: fit 4D-STEM maps with the parameters of a previous analysis",
    description="Thickness, tilt, Ti displacement, strain, polarization and field maps of a 4D-STEM scan in an upload, "
                "with calibration, crystal structures, libraries and fit options taken from a previous fit4dstem "
                "result (NXem) or its ELN entry; NXem + ELN + workflow are added to the upload.",
)
