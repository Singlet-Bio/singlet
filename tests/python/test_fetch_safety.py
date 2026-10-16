# SPDX-License-Identifier: MIT
"""fetch() takes file names from a remote manifest, so it must not let them
choose where files are written or skip verification."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from singlet.fetch import fetch

FILE = b"hello"
SHA = hashlib.sha256(FILE).hexdigest()


class Mirror:
    def __init__(self, files: list[dict]):
        self.requests: list[str] = []
        outer = self
        manifest = json.dumps({"files": files}).encode()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                outer.requests.append(self.path)
                body = manifest if self.path.endswith("/manifest.json") else FILE
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def mirror_for():
    made: list[Mirror] = []

    def make(files: list[dict]) -> Mirror:
        m = Mirror(files)
        made.append(m)
        return m

    yield make
    for m in made:
        m.close()


def test_a_normal_sample_downloads_and_verifies(tmp_path, mirror_for):
    m = mirror_for([{"path": "counts.1pz", "sha256": SHA}])
    out = fetch("GSM1", cache_dir=tmp_path, base_url=m.url)
    assert (out / "counts.1pz").read_bytes() == FILE


@pytest.mark.parametrize(
    "rel", ["../escaped.txt", "/abs/escaped.txt", "a/../../escaped.txt", "a\\b", "x\x00y", ""]
)
def test_manifest_paths_cannot_leave_the_cache_directory(tmp_path, mirror_for, rel):
    m = mirror_for([{"path": rel, "sha256": SHA}])
    cache = tmp_path / "cache"
    with pytest.raises((ValueError, RuntimeError)):
        fetch("GSM1", cache_dir=cache, base_url=m.url)
    assert not (tmp_path / "escaped.txt").exists()
    assert not any(p.name == "escaped.txt" for p in tmp_path.rglob("*"))
    assert not any("escaped" in r for r in m.requests[1:])  # only the manifest was requested


def test_a_file_without_a_checksum_is_refused(tmp_path, mirror_for):
    m = mirror_for([{"path": "counts.1pz"}])
    with pytest.raises((ValueError, RuntimeError), match="sha256"):
        fetch("GSM1", cache_dir=tmp_path, base_url=m.url)
    assert not (tmp_path / "GSM1" / "counts.1pz").exists()


def test_a_malformed_checksum_is_refused(tmp_path, mirror_for):
    m = mirror_for([{"path": "counts.1pz", "sha256": "not-a-digest"}])
    with pytest.raises((ValueError, RuntimeError), match="sha256"):
        fetch("GSM1", cache_dir=tmp_path, base_url=m.url)


def test_a_wrong_checksum_deletes_the_file(tmp_path, mirror_for):
    m = mirror_for([{"path": "counts.1pz", "sha256": "0" * 64}])
    with pytest.raises(RuntimeError, match="mismatch"):
        fetch("GSM1", cache_dir=tmp_path, base_url=m.url)
    assert not (tmp_path / "GSM1" / "counts.1pz").exists()


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".hidden/..", "GSM1/../.."])
def test_sample_names_cannot_contain_path_separators(tmp_path, mirror_for, name):
    m = mirror_for([{"path": "counts.1pz", "sha256": SHA}])
    with pytest.raises(ValueError):
        fetch(name, cache_dir=tmp_path, base_url=m.url)
    assert m.requests == []


def test_file_names_with_spaces_are_url_encoded(tmp_path, mirror_for):
    m = mirror_for([{"path": "my file.txt", "sha256": SHA}])
    fetch("GSM1", cache_dir=tmp_path, base_url=m.url)
    assert any("my%20file.txt" in r for r in m.requests)
    assert Path(tmp_path / "GSM1" / "my file.txt").exists()
