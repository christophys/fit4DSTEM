# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD dashboard: the fit4dstem app (same GUI as in Jupyter / Voila / Solara) inside NOMAD.

Served at ``{api_base_path}/dashboards/fit4dstem/``.  The user's NOMAD login is passed on: data sets can be read
lazily from the user's uploads (tab 1) and results written back (tab 4) without an API token.

Environment variables of the NOMAD app process:

* ``FIT4DSTEM_DASHBOARD_URL`` -- serve the app elsewhere and only link it from NOMAD (``external_url``)
* ``FIT4DSTEM_DATA``          -- folder with example data / libraries shown in tab 1 (read only)
* ``FIT4DSTEM_NOMAD_API``     -- API URL the app uses for NOMAD (default: this NOMAD)
* ``FIT4DSTEM_LIBRARIES``     -- persistent folder(s) for rocking-curve libraries, shared by the users (several
  separated by ``:``); without it, the users store libraries in their NOMAD projects or download them
"""
import os

from nomad.config.models.plugins import DashboardEntryPoint


class Fit4DSTEMDashboardEntryPoint(DashboardEntryPoint):
    def load(self):
        from .server import create_app

        return create_app()


fit4dstem_dashboard = Fit4DSTEMDashboardEntryPoint(
    name="4D-STEM CBED fitting",
    description="Maps of thickness, tilt, Ti displacement, strain and field from 4D-STEM data of ferroelectric "
                "superlattices; lazy loading from NOMAD uploads, NXem export, ELN and workflow.",
    id_url_safe="fit4dstem",
    launch_modes=["embedded", "tab"],
    external_url=os.environ.get("FIT4DSTEM_DASHBOARD_URL") or None,
)
