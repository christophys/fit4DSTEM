# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""FastAPI app of the dashboard: the Solara server of the fit4dstem app, mounted by NOMAD (as in inlineHolo).

Solara keeps the state of each browser session in the process that served it: run the NOMAD app with one worker
process or sticky sessions, or serve the app separately (FIT4DSTEM_DASHBOARD_URL).
"""
import os
import sys

PAGE = "fit4dstem_nomad.dashboards.page:Page"


def _solara_environment():
    os.environ.setdefault("SOLARA_TELEMETRY_MIXPANEL_ENABLE", "False")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "SOLARA_ASSETS_PROXY_CACHE_DIR" not in os.environ:
        for root in (sys.prefix, sys.base_prefix, "/usr/local", "/usr"):
            cdn = os.path.join(root, "share", "solara", "cdn")
            if os.path.isdir(cdn):
                os.environ["SOLARA_ASSETS_PROXY_CACHE_DIR"] = cdn
                break
    os.environ["SOLARA_APP"] = PAGE


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _solara_environment()
    import solara.server.app as solara_app
    import solara.server.starlette as solara_starlette

    if solara_app.apps.get("__default__") is None or getattr(solara_app.apps["__default__"], "fullname", "") != PAGE:
        solara_app.apps["__default__"] = solara_app.AppScript(PAGE)
    solara_starlette.on_startup()          # NOMAD does not run the lifespan of mounted apps
    app = FastAPI(title="fit4dstem dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        import fit4dstem

        return JSONResponse({"status": "ok", "fit4dstem": getattr(fit4dstem, "__version__", "")})

    app.mount("/", solara_starlette.app)
    return app
