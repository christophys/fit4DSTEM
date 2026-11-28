# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Libraries: stored in the library folder, a NOMAD project or downloaded (from memory); a full disk loses nothing;
the library folder can be changed (on a server only below the allowed folders); libraries are read from NOMAD."""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomad_mock import TOKEN, MockNomad  # noqa: E402

from fit4dstem import multislice as ms  # noqa: E402


def small_lib(name="SrTiO3"):
    rng = np.random.default_rng(1)
    data = rng.random((2, 2, 3, 4, 5, 5)).astype(np.float16)
    meta = {"cell_A": [3.9, 3.9, 3.9], "material": {"name": name}, "zone": "[100]", "energy_eV": 60e3}
    return ms.Library(data, [-10, 10], [-10, 10], [1, 2, 3], np.zeros((4, 2), int), np.arange(4.0),
                      np.linspace(-2, 2, 5), meta)


@pytest.fixture
def mock(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    m = MockNomad()
    yield m
    m.server.shutdown()


def test_library_bytes_and_safe_save(tmp_path, monkeypatch):
    lib = small_lib()
    b = lib.to_bytes()
    back = ms.Library.load(io.BytesIO(b))
    assert np.array_equal(back.data, lib.data) and back.meta == lib.meta
    p = tmp_path / "lib_SrTiO3.h5"
    lib.save(str(p))
    assert ms.Library.load(str(p)).data.shape == lib.data.shape and not (tmp_path / "lib_SrTiO3.h5.part").exists()

    def full(self, f):
        f.create_dataset("x", data=np.zeros(10))
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(ms.Library, "_write", full)
    with pytest.raises(OSError):
        lib.save(str(tmp_path / "lib_b.h5"))
    assert not (tmp_path / "lib_b.h5").exists() and not (tmp_path / "lib_b.h5.part").exists()


def test_library_folder_server_mode(tmp_path, monkeypatch):
    from fit4dstem.app import App
    shared = tmp_path / "shared_libs"
    shared.mkdir()
    monkeypatch.setenv("FIT4DSTEM_LIBRARIES", str(shared))
    a = App(data_dir=str(tmp_path / "w"), work_dir=str(tmp_path / "w"), server_mode=True)
    assert a.lib_dir == str(shared) and not a.w_lib_dir.disabled and a.w_st_dir.value
    a.w_lib_dir.value = "/etc"                                   # not allowed on a server
    assert a.lib_dir == str(shared) and a.w_lib_dir.value == str(shared) and "must be below" in \
        a.w_lib_dir_info.value
    a.w_lib_dir.value = str(shared / "BTO")                      # below an allowed folder
    assert a.lib_dir == str(shared / "BTO")
    monkeypatch.delenv("FIT4DSTEM_LIBRARIES")
    b = App(data_dir=str(tmp_path / "w2"), work_dir=str(tmp_path / "w2"), server_mode=True)
    assert b.lib_dir == str(tmp_path / "w2") and "temporary folder" in b.w_lib_dir_info.value
    c = App(data_dir=str(tmp_path), work_dir=str(tmp_path / "w3"))           # local: any folder
    c.w_lib_dir.value = str(tmp_path / "anywhere")
    assert c.lib_dir == str(tmp_path / "anywhere")


def test_store_disk_full_nomad_and_reload(tmp_path, monkeypatch, mock):
    pytest.importorskip("solara")
    pytest.importorskip("ipyvuetify")
    import shutil as _sh

    from fit4dstem_nomad.dashboards.page import make_app
    monkeypatch.setenv("FIT4DSTEM_NOMAD_API", mock.url)
    a = make_app({"Authorization": "Bearer " + TOKEN}, {}, "/nomad-oasis/dashboards/fit4dstem",
                 work_dir=str(tmp_path / "w"))
    a.project.w_sel.value = "U1"
    lib = small_lib()
    name = "lib_SrTiO3_100_60kV_8.0mrad.h5"
    a.session_libs[name] = lib
    a.w_st_dir.value, a.w_st_nomad.value = True, True
    real = _sh.disk_usage
    monkeypatch.setattr(_sh, "disk_usage", lambda p: real(p)._replace(free=1000))     # "disk full"
    done = a._lib_store(name, lib)
    assert len(done) == 1 and "not enough space" in a.w_st_status.value
    assert not os.path.exists(os.path.join(a.lib_dir, name))
    assert mock.uploads["U1"]["libraries/" + name] == lib.to_bytes()       # stored in the project, as is
    a._lib_refresh()
    ref = "nomad:U1:libraries/" + name
    assert ref in [v for _l, v in a.w_lib["SrTiO3"].options] and a.w_lib["SrTiO3"].value in (ref, "session:" + name)
    a.session_libs.clear()
    back = a._lib_open(ref)                                                # read from NOMAD, nothing on disk
    assert np.array_equal(back.data, lib.data)
    a.w_st_nomad.value = False
    with pytest.raises(RuntimeError, match="kept in this session"):
        a._lib_store(name, lib, {"dir": True})


def test_download_widget_protocol():
    pytest.importorskip("ipyvuetify")
    from fit4dstem.nomad_project import download_widget
    w = download_widget()
    got = []

    def browser(ch):                     # what the page does: collect the chunk, acknowledge
        got.append(bytes(w.chunk))
        w.ack = ch["new"]
    w.observe(browser, names="seq")
    data = os.urandom(3 * 1024 + 17)
    prog = []
    assert w.send(data, "x/lib.h5", chunk=1024, progress=lambda s, t: prog.append(s)) == len(data)
    assert b"".join(got) == data and w.n_chunks == 4 and w.filename == "lib.h5" and prog[-1] == len(data)
    assert w.chunk is None
    w.unobserve(browser, names="seq")
    from fit4dstem.nomad_project import ProjectError
    with pytest.raises(ProjectError, match="does not answer"):
        w.send(b"abc", "y.h5", timeout=0.2)
