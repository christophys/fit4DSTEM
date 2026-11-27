# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Mock servers of the tests: NOMAD API v1 (uploads, raw files, processing state; further endpoints via ``extra``)
and elabFTW API v2 (experiments, items, uploads, users)."""
import io
import json
import threading
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = "secret-token"
ELAB_KEY = "3-elabkey"


class MockNomad:
    """uploads -> {path: bytes}; implements the endpoints used by inlineholo.nomad."""

    def __init__(self):
        self.uploads = {"U1": {}}
        self.names = {"U1": "focal series project"}
        self.published = set()
        self.requests = []
        self.bodies = []
        self.extra = {}      # (method, path) -> f(query, body) -> (code, obj): further endpoints for other tests
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None):
                body = raw if raw is not None else json.dumps(obj or {}).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                return self.headers.get("Authorization") == "Bearer " + TOKEN

            def _route(self, method):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                parts = urllib.parse.unquote(u.path).split("/")[3:]          # after /api/v1
                mock.requests.append((method, "/".join(parts), q))
                if not self._auth():
                    return self._send(401, {"detail": "not authorized"})
                key = (method, "/".join(parts))
                if key in mock.extra:
                    n = int(self.headers.get("Content-Length", 0) or 0)
                    code, obj = mock.extra[key](q, self.rfile.read(n) if n else b"")
                    return self._send(code, obj)
                if parts == ["users", "me"] and method == "GET":
                    return self._send(200, {"user_id": "u0", "name": "Test User"})
                if parts == ["uploads"] and method == "GET":
                    return self._send(200, {"data": [{"upload_id": k, "upload_name": mock.names.get(k, ""),
                                                      "published": k in mock.published} for k in mock.uploads]})
                if parts == ["uploads"] and method == "POST":
                    uid = "U%d" % (len(mock.uploads) + 1)
                    mock.uploads[uid] = {}
                    mock.names[uid] = q.get("upload_name", "")
                    self._extract(uid, "")
                    return self._send(200, {"upload_id": uid, "data": {"upload_id": uid}})
                if len(parts) == 2 and parts[0] == "uploads" and method == "GET":
                    return self._send(200, {"upload_id": parts[1], "data": {
                        "upload_id": parts[1], "process_running": False, "process_status": "SUCCESS", "errors": [],
                        "upload_name": mock.names.get(parts[1], "")}})
                if len(parts) >= 3 and parts[0] == "uploads":
                    uid, kind, path = parts[1], parts[2], "/".join(parts[3:]).strip("/")
                    if uid not in mock.uploads:
                        return self._send(404, {"detail": "no upload"})
                    files = mock.uploads[uid]
                    if kind == "rawdir" and method == "GET":
                        if path in files:
                            return self._send(200, {"path": path, "access": "unpublished",
                                                    "file_metadata": {"name": path.split("/")[-1],
                                                                      "size": len(files[path])}})
                        pre = path + "/" if path else ""
                        content = {}
                        for f, b in files.items():
                            if f.startswith(pre):
                                rest = f[len(pre):].split("/")
                                if len(rest) == 1:
                                    content[rest[0]] = {"name": rest[0], "is_file": True, "size": len(b)}
                                else:
                                    content.setdefault(rest[0], {"name": rest[0], "is_file": False, "size": 0})
                        items = sorted(content.values(), key=lambda e: e["name"])
                        ps, pg = int(q.get("page_size", 10)), int(q.get("page", 1))
                        page = items[(pg - 1) * ps: pg * ps]
                        return self._send(200, {"path": path, "access": "unpublished",
                                                "directory_metadata": {"name": path, "content": page},
                                                "pagination": {"page_size": ps, "page": pg, "total": len(items)}})
                    if kind == "raw" and method == "GET":
                        if path not in files:
                            return self._send(404, {"detail": "no file"})
                        b = files[path]
                        if "offset" in q:                   # range read (RemoteFile)
                            o = int(q["offset"])
                            b = b[o:o + int(q.get("length", len(b)))]
                        return self._send(200, raw=b)
                    if kind == "raw" and method == "PUT":
                        if uid in mock.published:
                            return self._send(403, {"detail": "published"})
                        self._extract(uid, path, q.get("file_name"))
                        return self._send(200, {"upload_id": uid, "data": {"upload_id": uid}})
                return self._send(400, {"detail": "unknown"})

            def _extract(self, uid, folder, file_name=None):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n) if n else b""
                if not body:                        # empty upload (POST /uploads without a file)
                    return
                mock.bodies.append(len(body))
                if not zipfile.is_zipfile(io.BytesIO(body)):        # a single file is stored as it is
                    name = (folder + "/" + (file_name or "file")) if folder else (file_name or "file")
                    mock.uploads[uid][name.strip("/")] = body
                    return
                z = zipfile.ZipFile(io.BytesIO(body))
                for it in z.infolist():
                    name = (folder + "/" + it.filename) if folder else it.filename
                    mock.uploads[uid][name.strip("/")] = z.read(it)

            def do_GET(self):
                self._route("GET")

            def do_PUT(self):
                self._route("PUT")

            def do_POST(self):
                self._route("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/api/v1" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class MockElab:
    """entities: {("experiments", id): dict}; files: {("experiments", id): [(upload dict, bytes)]}."""

    def __init__(self):
        self.entities, self.files, self.users, self.requests = {}, {}, {}, []
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None):
                body = raw if raw is not None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                mock.requests.append((u.path, q, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != ELAB_KEY:
                    return self._send(401, {"code": 401, "message": "No access token provided!"})
                parts = u.path.split("/api/v2/", 1)[-1].strip("/").split("/")
                if parts[0] == "users" and len(parts) == 2:
                    return self._send(200, mock.users.get(int(parts[1]), {}))
                key = (parts[0], int(parts[1])) if len(parts) >= 2 and parts[1].isdigit() else None
                if key not in mock.entities:
                    return self._send(404, {"code": 404, "message": "Nothing to show with this id"})
                if len(parts) == 2:
                    ent = dict(mock.entities[key])
                    ent["uploads"] = [u for u, _b in mock.files.get(key, [])]
                    return self._send(200, ent)
                if parts[2] == "uploads" and len(parts) == 3:
                    return self._send(200, [u for u, _b in mock.files.get(key, [])])
                if parts[2] == "uploads" and len(parts) == 4:
                    for up, b in mock.files.get(key, []):
                        if up["id"] == int(parts[3]):
                            return self._send(200, raw=b) if q.get("format") == "binary" else self._send(200, up)
                return self._send(404, {"message": "no"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
