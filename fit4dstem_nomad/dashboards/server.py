# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""FastAPI app of the dashboard that NOMAD mounts at ``{api_base_path}/dashboards/fit4dstem``.

The Solara server of the fit4dstem app (:mod:`.page`) runs as a child process of the NOMAD app
(:mod:`.solara_proxy`); this app forwards the pages, assets and websockets to it.  Several Solara
dashboards (e.g. fit4DSTEM and inlineHolo) can so be installed in the same NOMAD.

The sessions live in the child process of the NOMAD app process that started it: run the NOMAD app with one
worker process or sticky sessions, or serve the app separately (FIT4DSTEM_DASHBOARD_URL).
"""
PAGE = "fit4dstem_nomad.dashboards.page:Page"


def create_app():
    from fastapi.responses import JSONResponse

    from .solara_proxy import SolaraProcess, proxy_app

    child = SolaraProcess(PAGE, name="fit4dstem").start()

    def healthz():
        import fit4dstem

        return JSONResponse({"status": "ok" if child.ready(0) else "starting",
                             "fit4dstem": getattr(fit4dstem, "__version__", "")})

    return proxy_app(child, title="fit4dstem dashboard", extra_routes={"/healthz": healthz})
