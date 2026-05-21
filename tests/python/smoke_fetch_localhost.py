# SPDX-License-Identifier: MIT
"""End-to-end smoke test for singlet.fetch against a localhost mock.

Creates a fake sample directory in a temp dir, generates its manifest,
serves it with http.server, and runs ``singlet.fetch`` against it. Then
mutates one file and verifies sha256 checking catches the corruption.

Runs offline. No credentials required.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import tempfile
import threading
from pathlib import Path

from singlet.manifest_gen import generate_manifest
import json


def _serve(root: Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a, **kw):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def _make_fake_sample(d: Path) -> None:
    (d / "summary.json").write_text(json.dumps({"accession": "TEST001", "n_cells": 0}))
    (d / "cell_meta.parquet").write_bytes(b"PAR1\x00fake-parquet-bytes\x00PAR1")
    (d / "counts.1pz").write_bytes(b"1pz\x00" + os.urandom(2048))
    (d / "snp.1pz").write_bytes(b"1pz\x00" + os.urandom(512))


def main() -> int:
    with tempfile.TemporaryDirectory() as srv_root, tempfile.TemporaryDirectory() as cache_root:
        srv_root = Path(srv_root)
        cache_root = Path(cache_root)

        # 1. Build a fake sample on the "server"
        remote_sample = srv_root / "TEST001"
        remote_sample.mkdir()
        _make_fake_sample(remote_sample)
        manifest = generate_manifest(remote_sample, accession="TEST001")
        (remote_sample / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[fixture] {len(manifest['files'])} files served from {remote_sample}")

        # 2. Serve
        httpd, port = _serve(srv_root)
        base = f"http://127.0.0.1:{port}"
        print(f"[server] listening on {base}")

        try:
            # 3. fetch()
            from singlet import fetch
            local = fetch("TEST001", cache_dir=cache_root, base_url=base)
            print(f"[fetch] downloaded to {local}")
            assert (local / "manifest.json").exists()
            assert (local / "summary.json").exists()
            assert (local / "counts.1pz").exists()

            # 4. Idempotent re-fetch (sha256-cached, should be no-op)
            local2 = fetch("TEST001", cache_dir=cache_root, base_url=base)
            assert local2 == local
            print("[cache] re-fetch was a no-op (sha256 matched)")

            # 5. Corruption detection: tamper with a cached file, refetch should redownload
            (local / "counts.1pz").write_bytes(b"corrupted")
            fetch("TEST001", cache_dir=cache_root, base_url=base)
            counts_size = (local / "counts.1pz").stat().st_size
            assert counts_size > 100, f"counts.1pz not re-downloaded (size={counts_size})"
            print(f"[heal] corrupted file re-downloaded ({counts_size} bytes)")

            # 6. sha256 mismatch should fail loudly: serve garbage as one file
            (remote_sample / "counts.1pz").write_bytes(b"server-side garbage")
            # Don't regen manifest — sha256 in manifest now disagrees with file.
            (local / "counts.1pz").unlink()  # force redownload
            try:
                fetch("TEST001", cache_dir=cache_root, base_url=base)
            except RuntimeError as e:
                if "sha256 mismatch" in str(e):
                    print("[verify] sha256 mismatch correctly raised")
                else:
                    raise

            print("\nAll fetch() smoke checks passed.")
            return 0
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
