# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD (production, test or an Oasis) API v1 client with lazy, range-based reading of large raw files.

Adapted from inlineHolo (``inlineholo/nomad.py``): an API token (NOMAD user menu -> API tokens / app token, with
permission to read and write uploads) is sent as ``Authorization: Bearer <token>``; the token is never written to disk.

Endpoints used (nomad-lab v1 API):

* ``GET  /users/me``                             who is logged in
* ``GET  /uploads``                              uploads of the user
* ``GET  /uploads/{id}/rawdir/{path}``           folder listing (paginated)
* ``GET  /uploads/{id}/raw/{path}?offset=&length=``  (part of) a raw file -- used for lazy reading
* ``PUT  /uploads/{id}/raw/{folder}?file_name=`` add files (a .zip is extracted into the folder)
* ``POST /uploads?file_name=&upload_name=``      new upload from a .zip

:class:`RemoteFile` is a read-only, seekable file object that fetches blocks of a raw file on demand (LRU cache),
so that ``h5py.File(RemoteFile(...))`` reads only the frames that are needed from a 4D-STEM file of tens of GB.
"""
from __future__ import annotations

import collections
import datetime as dt
import io
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile

NOMAD_URLS = {"production": "https://nomad-lab.eu/prod/v1/api/v1",
              "test": "https://nomad-lab.eu/prod/v1/test/api/v1"}


class NomadError(RuntimeError):
    pass


class NomadClient:
    """Minimal NOMAD API v1 client (urllib only).  ``opener`` can replace urllib.request.urlopen (tests)."""

    def __init__(self, url="production", token="", opener=None, timeout=300):
        url = str(url or "production").strip()
        self.url = NOMAD_URLS.get(url, url).rstrip("/")
        self.token = str(token or "").strip()
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _req(self, method, path, params=None, data=None, ctype=None, raw=False):
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        hdr = {"Accept": "application/json" if not raw else "*/*"}
        if self.token:
            hdr["Authorization"] = "Bearer " + self.token
        if ctype:
            hdr["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path + q, data=data, headers=hdr, method=method)
        try:
            with self.opener(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as ex:
            detail = ""
            try:
                detail = ex.read().decode(errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if ex.code in (401, 403):
                raise NomadError(f"NOMAD refused the request ({ex.code}): check the API token and its permissions. "
                                 f"{detail}")
            if ex.code == 404:
                raise NomadError(f"not found on NOMAD (404): check the upload ID and the path. {detail}")
            raise NomadError(f"NOMAD request failed: {ex} {detail}")
        except urllib.error.URLError as ex:
            raise NomadError(f"NOMAD is not reachable ({ex.reason}); internet access to {self.url} is needed")
        if raw:
            return body
        return json.loads(body.decode() or "{}")

    @staticmethod
    def _qp(path):
        return urllib.parse.quote(str(path or "").strip("/"))

    # ------------------------------------------------------------------ reading
    def me(self):
        return self._req("GET", "/users/me")

    def uploads(self, page_size=100):
        """[(upload_id, upload_name, published)] of the user (needs a token)."""
        if not self.token:
            raise NomadError("an API token is needed to list your uploads")
        r = self._req("GET", "/uploads", {"page_size": page_size, "order_by": "upload_create_time", "order": "desc"})
        return [(d.get("upload_id"), d.get("upload_name") or "", bool(d.get("published"))) for d in r.get("data", [])]

    def listdir(self, upload_id, path=""):
        """[(name, is_file, size)] of a folder of the upload (all pages)."""
        out, page = [], 1
        while True:
            r = self._req("GET", f"/uploads/{self._qp(upload_id)}/rawdir/{self._qp(path)}",
                          {"page_size": 500, "page": page})
            dm = r.get("directory_metadata")
            if dm is None:
                if r.get("file_metadata") is not None:
                    raise NomadError(f"{path} is a file, not a folder")
                break
            items = dm.get("content", [])
            out += [(e.get("name"), bool(e.get("is_file")), e.get("size")) for e in items]
            pag = r.get("pagination") or {}
            if len(items) < 500 or not pag or page * 500 >= int(pag.get("total", 0) or 0):
                break
            page += 1
        return out

    def file_size(self, upload_id, path):
        r = self._req("GET", f"/uploads/{self._qp(upload_id)}/rawdir/{self._qp(path)}")
        fm = r.get("file_metadata") or {}
        if "size" not in fm:
            raise NomadError(f"{path} is not a file of upload {upload_id}")
        return int(fm["size"])

    def read_range(self, upload_id, path, offset, length):
        """Bytes [offset, offset + length) of a raw file of the upload."""
        return self._req("GET", f"/uploads/{self._qp(upload_id)}/raw/{self._qp(path)}",
                         {"offset": int(offset), "length": int(length), "decompress": "false",
                          "ignore_mime_type": "true"}, raw=True)

    def download(self, upload_id, path, dest_dir):
        """Download one raw file of the upload into dest_dir (keeps its relative path); returns the local path."""
        rel = str(path).strip("/")
        if ".." in rel.split("/"):
            raise NomadError("invalid path")
        body = self._req("GET", f"/uploads/{self._qp(upload_id)}/raw/{self._qp(rel)}", raw=True)
        local = os.path.join(dest_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        with open(local, "wb") as f:
            f.write(body)
        return local

    # ------------------------------------------------------------------ writing
    def upload(self, paths, upload_id="", folder="", upload_name="", base_dir=None):
        """Put local files into an upload (zip, extracted by NOMAD into ``folder``); without ``upload_id`` a new
        upload is created.  Returns dict(upload_id, folder, files, gui_url)."""
        if not self.token:
            raise NomadError("a NOMAD API token is needed for uploading")
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            raise NomadError("no files to upload")
        folder = str(folder or "").strip().strip("/")
        if ".." in folder.split("/"):
            raise NomadError("invalid subfolder")
        upload_id = str(upload_id or "").strip()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                arc = os.path.relpath(p, base_dir).replace(os.sep, "/") if base_dir else os.path.basename(p)
                if not upload_id and folder:
                    arc = folder + "/" + arc
                z.write(p, arc)
        zipname = "fit4dstem_%s.zip" % dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ctype = "application/octet-stream"
        if upload_id:
            r = self._req("PUT", f"/uploads/{self._qp(upload_id)}/raw/{self._qp(folder)}", {"file_name": zipname},
                          data=buf.getvalue(), ctype=ctype)
        else:
            params = {"file_name": zipname}
            if upload_name:
                params["upload_name"] = str(upload_name)
            r = self._req("POST", "/uploads", params, data=buf.getvalue(), ctype=ctype)
        uid = r.get("upload_id") or (r.get("data") or {}).get("upload_id") or upload_id
        return {"upload_id": uid, "folder": folder, "files": len(paths), "gui_url": self.gui_url(uid)}

    def gui_url(self, upload_id):
        if not upload_id:
            return None
        return self.url.replace("/api/v1", "/gui/user/uploads/upload/id/" + upload_id)


class RemoteFile(io.RawIOBase):
    """Read-only, seekable file object over a raw file of a NOMAD upload (blocks fetched on demand, LRU cache).

    ``reader(offset, length) -> bytes`` can replace the NOMAD request (tests, other servers)."""

    def __init__(self, client=None, upload_id="", path="", size=None, block_size=4 << 20, max_blocks=64,
                 reader=None):
        super().__init__()
        self.client, self.upload_id, self.path = client, upload_id, path
        self._reader = reader or (lambda off, n: client.read_range(upload_id, path, off, n))
        self.size = int(size if size is not None else client.file_size(upload_id, path))
        self.block = int(block_size)
        self.max_blocks = int(max_blocks)
        self._cache = collections.OrderedDict()
        self._pos = 0
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        self.name = f"nomad:{upload_id}/{path}"

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        elif whence == io.SEEK_END:
            self._pos = self.size + int(offset)
        return self._pos

    def _get_block(self, b):
        with self._lock:
            data = self._cache.get(b)
            if data is not None:
                self._cache.move_to_end(b)
                return data
        off = b * self.block
        n = min(self.block, self.size - off)
        data = self._reader(off, n)
        if len(data) != n:
            raise NomadError(f"short read from {self.name} at {off}: {len(data)} of {n} bytes")
        with self._lock:
            self.bytes_fetched += n
            self._cache[b] = data
            while len(self._cache) > self.max_blocks:
                self._cache.popitem(last=False)
        return data

    def readinto(self, buf):
        mv = memoryview(buf).cast("B")
        want = min(len(mv), max(0, self.size - self._pos))
        got = 0
        while got < want:
            p = self._pos + got
            b = p // self.block
            data = self._get_block(b)
            o = p - b * self.block
            n = min(len(data) - o, want - got)
            mv[got:got + n] = data[o:o + n]
            got += n
        self._pos += got
        return got

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        buf = bytearray(max(0, min(n, self.size - self._pos)))
        k = self.readinto(buf)
        return bytes(buf[:k])


def default_results_folder():
    return "fit4dstem_results_" + dt.date.today().isoformat()
