# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Temporal workflow of the fit4dstem map-fit action: stage -> run -> publish (CPU workers)."""
import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import cleanup_map_fit, publish_map_fit, run_map_fit, stage_map_fit
    from .models import MapFitInput, MapFitResult

HEARTBEAT = timedelta(seconds=60)


@workflow.defn
class MapFitWorkflow:
    def __init__(self):
        self.stage = "starting"

    @workflow.query
    def progress(self) -> str:
        return self.stage

    @workflow.run
    async def run(self, data: MapFitInput) -> MapFitResult:
        self.stage = "staging inputs"
        st = await workflow.execute_activity(stage_map_fit, data, start_to_close_timeout=timedelta(hours=2),
                                             heartbeat_timeout=None, retry_policy=RetryPolicy(maximum_attempts=2))
        try:
            self.stage = "calibration and map fit"
            files = await workflow.execute_activity(
                run_map_fit, [data.model_dump(), st.model_dump()], heartbeat_timeout=HEARTBEAT,
                start_to_close_timeout=timedelta(hours=data.timeout_hours),
                retry_policy=RetryPolicy(maximum_attempts=1))
            self.stage = "adding the results to the upload"
            res = await workflow.execute_activity(
                publish_map_fit, [data.model_dump(), st.model_dump(), files],
                start_to_close_timeout=timedelta(hours=2), retry_policy=RetryPolicy(maximum_attempts=3))
            self.stage = "done"
            return res
        except (asyncio.CancelledError, Exception):
            self.stage = "cleaning up"
            await workflow.execute_activity(cleanup_map_fit, st.work_dir, start_to_close_timeout=timedelta(minutes=10))
            raise
