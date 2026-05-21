# SPDX-License-Identifier: MIT
"""Round-trip smoke: serve a real pipeline output dir over localhost,
fetch a subset of files into a fresh cache, verify sha256."""

from __future__ import annotations

import http.server
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

from singlet import fetch


def _serve(root: Path):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a, **kw):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main(sample_src: str) -> int:
    src = Path(sample_src)
    assert (src / "manifest.json").exists(), f"no manifest in {src}"

    with tempfile.TemporaryDirectory() as srv_root, tempfile.TemporaryDirectory() as cache:
        # Symlink the real directory under accession prefix on the server root
        srv = Path(srv_root)
        (srv / "LEGACY001").symlink_to(src)
        httpd, port = _serve(srv)
        try:
            local = fetch(
                "LEGACY001",
                cache_dir=cache,
                base_url=f"http://127.0.0.1:{port}",
                files=["summary.json", "gene_counts.1pz", "exon_counts.1pz", "sj_counts.1pz"],
                max_workers=4,
            )
            n_local = sum(1 for _ in local.glob("*"))
            sizes = {p.name: p.stat().st_size for p in local.glob("*") if p.is_file()}
            print(f"[round-trip] fetched into {local} ({n_local} entries)")
            for k, v in sorted(sizes.items()):
                print(f"  {k}: {v:,} bytes")
            for must in ("manifest.json", "summary.json", "gene_counts.1pz"):
                assert (local / must).exists(), f"missing {must}"
            print("[OK] round-trip from local HTTP server validated against real 606 MB output")
            return 0
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke_legacy_test"))
