# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Solara page of the NOMAD dashboard: the fit4dstem app plus the user's NOMAD session.

The NOMAD login of the user (``Authorization`` cookie of the NOMAD GUI) is passed to the NOMAD import (tab 1) and
upload (tab 4): "this NOMAD" is preselected and no API token is needed.  The app runs in server mode: every session
has its own working folder (removed at the end of the session) and cannot browse other folders of the server.
"""
import os
import shutil
import tempfile
import urllib.parse

import ipywidgets as W
import solara

from fit4dstem.app import App
from fit4dstem.nomad import NomadClient, NomadError


def nomad_token(cookies):
    raw = (cookies or {}).get("Authorization") or ""
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    tok = urllib.parse.unquote(str(raw)).strip()
    return tok[7:].strip() if tok.lower().startswith("bearer ") else tok


def _h(headers, key):
    v = (headers or {}).get(key) or (headers or {}).get(key.lower())
    return (v[0] if isinstance(v, (list, tuple)) and v else v) or ""


def api_candidates(headers, root_path):
    out = []
    env = os.environ.get("FIT4DSTEM_NOMAD_API")
    if env:
        out.append(env.rstrip("/"))
    base = (root_path or "").split("/dashboards/")[0].rstrip("/")
    host = _h(headers, "x-forwarded-host") or _h(headers, "host")
    proto = _h(headers, "x-forwarded-proto") or "http"
    if host:
        out.append(f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}{base}/api/v1")
    try:
        from nomad.config import config
        out.append(config.api_url(ssl=False, api_host="localhost").rstrip("/") + "/v1")
        out.append(config.api_url().rstrip("/") + "/v1")
    except Exception:  # noqa: BLE001
        pass
    return list(dict.fromkeys(out))


def connect(app, token, candidates, banner):
    if not token:
        banner.value = "<small>Not signed in to NOMAD: open the dashboard from the NOMAD GUI, or use an API token " \
                       "in tab 1.</small>"
        return
    err = ""
    for url in candidates:
        try:
            NomadClient(url, token, timeout=10).me()
        except NomadError as e:
            err = str(e)
            continue
        app.w_nm_url.value = "custom"
        app.w_nm_custom.value = url
        app.w_nm_token.value = token
        banner.value = ""                 # signed in: nothing to show above the title
        try:
            app.project.connect(url, token)
        except Exception:  # noqa: BLE001
            pass
        try:
            app._nm_list_uploads()
        except Exception:  # noqa: BLE001
            pass
        return
    banner.value = f"<small style='color:#b26a00'>NOMAD API not reachable from the app: {err}</small>"


def make_app(cookies=None, headers=None, root_path=None, work_dir=None):
    data = os.environ.get("FIT4DSTEM_DATA") or work_dir
    app = App(data_dir=data, work_dir=work_dir, server_mode=True)
    banner = W.HTML()
    connect(app, nomad_token(cookies), api_candidates(headers, root_path), banner)
    app.ui.children = (banner,) + tuple(app.ui.children)
    app._browse(app.data_dir)
    app._lib_refresh()
    return app


@solara.component
def Layout(children=[]):  # noqa: B006
    solara.Column(children=children, style={"padding": "4px 8px", "background": "transparent"})


@solara.component
def Page():
    cookies = solara.lab.cookies.value
    headers = solara.lab.headers.value
    from solara.server import settings

    def build():
        wd = tempfile.mkdtemp(prefix="fit4dstem_")
        return make_app(cookies, headers, settings.main.root_path, work_dir=wd), wd

    app, wd = solara.use_memo(build, [])

    def cleanup():
        return lambda: shutil.rmtree(wd, ignore_errors=True)
    solara.use_effect(cleanup, [])
    solara.Title("fit4dstem")
    solara.display(app.ui)
