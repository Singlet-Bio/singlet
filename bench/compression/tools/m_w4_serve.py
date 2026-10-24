"""m_w4: serve a v2 root (the directory that contains v2/) over loopback HTTP with Range / ETag support; one JSON line per request.

    python -P tools/m_w4_serve.py <root> <port_file> <log_file>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "python"))
import threading  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

import _v2_range_server as rs  # noqa: E402

root, port_file, log_file = sys.argv[1:4]
rec = rs.Recorder()
srv = ThreadingHTTPServer(("127.0.0.1", 0), rs._handler(os.path.abspath(root), rec, False, None))
lock = threading.Lock()
orig_add = rec.add


def add(method, path, rng, status, nbytes):
    orig_add(method, path, rng, status, nbytes)
    with lock, open(log_file, "a") as fh:
        fh.write(json.dumps({"path": path, "range": rng, "status": status, "bytes": nbytes}) + "\n")


rec.add = add
open(log_file, "w").close()
with open(port_file, "w") as fh:
    fh.write(str(srv.server_address[1]))
srv.serve_forever()
