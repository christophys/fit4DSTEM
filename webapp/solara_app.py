# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""The fit4dstem app as a stand-alone web application (no Jupyter / Voila).

    solara run webapp/solara_app.py --host 0.0.0.0 --port 8765

Every browser session gets its own App.  Environment: FIT4DSTEM_DATA (folder shown in tab 1 and searched for
libraries), FIT4DSTEM_WORK (results folder).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import solara  # noqa: E402

from fit4dstem.app import App  # noqa: E402

DATA = os.environ.get("FIT4DSTEM_DATA", os.getcwd())
WORK = os.environ.get("FIT4DSTEM_WORK", os.path.join(os.getcwd(), "fit4dstem_results"))


def _make():
    app = App(data_dir=DATA, work_dir=WORK)
    app._browse(app.data_dir)
    app._lib_refresh()
    return app


@solara.component
def Page():
    app = solara.use_memo(_make, [])
    solara.Title("fit4dstem – 4D-STEM CBED maps")
    solara.display(app.ui)
