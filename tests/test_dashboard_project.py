# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Dashboard session: no sign-in line above the title; uploaded files go into a NOMAD project (new or existing)
that also receives the results."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomad_mock import TOKEN, MockNomad  # noqa: E402


@pytest.fixture
def mock(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    m = MockNomad()
    yield m
    m.server.shutdown()


def test_dashboard_project_for_uploads(mock, tmp_path, monkeypatch):
    pytest.importorskip("solara")
    pytest.importorskip("ipyvuetify")
    from fit4dstem.nomad_project import NEW
    from fit4dstem_nomad.dashboards.page import make_app
    monkeypatch.setenv("FIT4DSTEM_NOMAD_API", mock.url)
    a = make_app({"Authorization": "Bearer " + TOKEN}, {}, "/nomad-oasis/dashboards/fit4dstem",
                 work_dir=str(tmp_path / "w"))
    assert a.ui.children[0].value == ""                       # no "Signed in to NOMAD as ..." line
    assert 'label="Upload 4D-STEM data file(s)"' in a.w_upload.template
    p = a.project
    assert p.box.layout.display == "flex" and p.w_sel.value == NEW and "U1" in [v for _l, v in p.w_sel.options]
    f = tmp_path / "scan.h5"
    f.write_bytes(b"\x89HDF not really")
    p.w_name.value = "STO 4D-STEM"
    r, err = a._store_in_project([str(f)])
    assert not err and r["upload_id"] != "U1" and mock.names[r["upload_id"]] == "STO 4D-STEM"
    assert "4dstem_data/scan.h5" in mock.uploads[r["upload_id"]]
    assert a.w_nm_upload.value == r["upload_id"]               # results go into the same project (tab 4)
    assert a._stored[str(f)]["path"] == "4dstem_data/scan.h5"
    p.limit = p.n_open                                         # maximum number of open projects reached
    p.refresh()
    assert NEW not in [v for _l, v in p.w_sel.options] and "maximum is reached" in p.w_info.value
    p.w_sel.value = "U1"
    assert a.w_nm_upload.value == "U1"
