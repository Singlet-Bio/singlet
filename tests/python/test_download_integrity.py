# SPDX-License-Identifier: MIT
"""Downloads are verified before they reach the cache.

A local HTTP server stands in for data.singlet.bio. Each test serves a small
synthetic bundle (or a damaged one) and checks what ends up in the cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from singlet import _loader
from singlet._integrity import BundleIntegrityError, normalize_accession, verify_bundle

MEMBERS = {
    "feature_vocab.json": b'{"genes": []}',
    "samples/GSM1/exon_counts.1pz": bytes(range(256)) * 40,
    "samples/GSM1/summary.json": b'{"n_cells": 1}',
}


def make_bundle(
    *,
    schema: str = "1.0",
    tamper_checksum: str | None = None,
    drop_member: str | None = None,
    extra_manifest: dict | None = None,
) -> bytes:
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in MEMBERS.items()}
    if tamper_checksum:
        checksums[tamper_checksum] = "0" * 64
    manifest = {"schema_version": schema, "gse_id": "GSE1", "gsm_ids": ["GSM1"], "checksums": checksums}
    manifest.update(extra_manifest or {})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", allowZip64=True) as zf:
        for name, data in MEMBERS.items():
            if name == drop_member:
                continue
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


class Server:
    """Serves one GSE at /data/GSE1/GSE1.singlet; `mode` decides how."""

    def __init__(self) -> None:
        self.body = make_bundle()
        self.mode = "ok"
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401 - silence
                pass

            def do_GET(self):  # noqa: N802
                outer.hits += 1
                if self.path != "/data/GSE1/GSE1.singlet":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = outer.body
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.mode == "truncate":
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server(tmp_path, monkeypatch):
    srv = Server()
    monkeypatch.setenv("SINGLET_DATA_BASE", srv.url)
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    yield srv
    srv.close()


def cached_files(tmp_path: Path) -> list[str]:
    cache = tmp_path / "cache"
    return sorted(p.name for p in cache.iterdir()) if cache.exists() else []


def test_good_download_is_verified_and_cached(server, tmp_path):
    path = _loader.download("GSE1")
    assert path.name == "GSE1.singlet"
    assert cached_files(tmp_path) == ["GSE1.singlet"]
    manifest = verify_bundle(path)
    assert manifest["gse_id"] == "GSE1"


def test_lowercase_accession_is_accepted(server, tmp_path):
    assert _loader.download("gse1").name == "GSE1.singlet"


def test_truncated_download_is_rejected_and_not_cached(server, tmp_path):
    server.mode = "truncate"
    with pytest.raises(BundleIntegrityError, match="ended after"):
        _loader.download("GSE1")
    assert cached_files(tmp_path) == []  # no bundle and no .part file left behind


def test_a_member_that_does_not_match_its_checksum_is_rejected(server, tmp_path):
    server.body = make_bundle(tamper_checksum="samples/GSM1/summary.json")
    with pytest.raises(BundleIntegrityError, match="does not match its checksum"):
        _loader.download("GSE1")
    assert cached_files(tmp_path) == []


def test_verification_can_be_skipped_but_not_the_archive_structure(server, tmp_path):
    server.body = make_bundle(tamper_checksum="samples/GSM1/summary.json")
    assert _loader.download("GSE1", verify=False).exists()  # hashes skipped on request
    (tmp_path / "cache" / "GSE1.singlet").unlink()
    server.body = server.body[: len(server.body) // 2]
    with pytest.raises(BundleIntegrityError):
        _loader.download("GSE1", verify=False)


def test_a_newer_schema_is_refused_with_an_upgrade_hint(server, tmp_path):
    server.body = make_bundle(schema="2.0")
    with pytest.raises(BundleIntegrityError, match="Upgrade singlet"):
        _loader.download("GSE1")
    assert cached_files(tmp_path) == []


def test_a_listed_member_missing_from_the_archive_is_rejected(server, tmp_path):
    server.body = make_bundle(drop_member="samples/GSM1/summary.json")
    with pytest.raises(BundleIntegrityError, match="missing from the archive"):
        _loader.download("GSE1")


@pytest.mark.parametrize("name", ["../evil.json", "/etc/passwd", "samples/../../x", "a\\b"])
def test_member_names_that_could_escape_a_directory_are_rejected(tmp_path, name):
    checksums = {name: "0" * 64}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": "1.0", "checksums": checksums}))
    path = tmp_path / "x.singlet"
    path.write_bytes(buf.getvalue())
    with pytest.raises(BundleIntegrityError, match="unsafe member name"):
        verify_bundle(path)


def test_a_damaged_cached_file_is_replaced_not_trusted(server, tmp_path):
    path = _loader.download("GSE1")
    path.write_bytes(path.read_bytes()[:100])  # a cache entry cut short by an old bug or a full disk
    hits = server.hits
    again = _loader.download("GSE1")
    assert server.hits == hits + 1  # fetched again
    verify_bundle(again)


def test_an_intact_cache_entry_is_reused_without_a_request(server, tmp_path):
    _loader.download("GSE1")
    hits = server.hits
    _loader.download("GSE1")
    assert server.hits == hits


def test_force_refetches(server, tmp_path):
    _loader.download("GSE1")
    hits = server.hits
    _loader.download("GSE1", force=True)
    assert server.hits == hits + 1


def test_unknown_study_is_a_not_found_error(server, tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader.download("GSE2")
    assert cached_files(tmp_path) == []


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "GSE1/../../x", "GSE", "GSE12a", "GSE1 ", "GSE1;rm", "", "http://evil/GSE1", "GSE1\n", "GSE1234567890"],
)
def test_accessions_that_are_not_accessions_never_reach_a_path_or_url(bad, server, tmp_path):
    if bad.strip() == "GSE1":
        pytest.skip("whitespace is trimmed")
    with pytest.raises(ValueError):
        _loader.download(bad)
    assert server.hits == 0
    assert cached_files(tmp_path) == []


def test_normalize_accession():
    assert normalize_accession(" gse138867 ") == "GSE138867"
    assert normalize_accession("GSM4120733") == "GSM4120733"
    with pytest.raises(ValueError):
        normalize_accession("GPL570")


def test_gsm_resolution_needs_an_exact_match(monkeypatch):
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        # the API returns a different sample for the query: it must not be trusted
        return Resp(json.dumps({"data": [{"gsm_id": "GSM999", "gse_id": "GSE5"}]}).encode())

    monkeypatch.setattr(_loader.urllib.request, "urlopen", fake_urlopen)
    assert _loader._resolve_gsm_to_gse("GSM1") is None
    assert all("GSM1" in u and " " not in u for u in calls)

    def exact(req, timeout=0):
        return Resp(json.dumps({"data": [{"gsm_id": "GSM1", "gse_id": "gse7"}]}).encode())

    monkeypatch.setattr(_loader.urllib.request, "urlopen", exact)
    assert _loader._resolve_gsm_to_gse("gsm1") == "GSE7"


def test_gsm_lookup_rejects_url_syntax_before_any_request(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no request may be made")

    monkeypatch.setattr(_loader.urllib.request, "urlopen", boom)
    with pytest.raises(ValueError):
        _loader._resolve_gsm_to_gse("GSM1&limit=999999#x")


def _bundle_with(tmp_path, *, extra=None, checksums_as_list=False, duplicate=False):
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in MEMBERS.items()}
    manifest = {"schema_version": "1.0", "checksums": sorted(checksums) if checksums_as_list else checksums}
    path = tmp_path / "x.singlet"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in MEMBERS.items():
            zf.writestr(name, data)
        if extra:
            zf.writestr(extra, b"not vouched for")
        if duplicate:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zf.writestr("feature_vocab.json", b"{}")
        zf.writestr("manifest.json", json.dumps(manifest))
    return path


def test_a_member_the_manifest_does_not_list_is_rejected(tmp_path):
    with pytest.raises(BundleIntegrityError, match="not listed"):
        verify_bundle(_bundle_with(tmp_path, extra="samples/GSM1/evil.1pz"))


def test_two_members_with_one_name_are_rejected(tmp_path):
    with pytest.raises(BundleIntegrityError, match="same name"):
        verify_bundle(_bundle_with(tmp_path, duplicate=True))


def test_checksums_must_be_a_name_to_hash_map(tmp_path):
    with pytest.raises(BundleIntegrityError, match="checksums"):
        verify_bundle(_bundle_with(tmp_path, checksums_as_list=True))


def test_a_huge_manifest_is_refused_before_it_is_parsed(tmp_path):
    path = tmp_path / "big.singlet"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", '{"schema_version": "1.0", "pad": "' + "x" * (17 << 20) + '"}')
    with pytest.raises(BundleIntegrityError, match="implausibly large"):
        verify_bundle(path)


def test_a_symlinked_cache_entry_still_works(server, tmp_path, monkeypatch):
    real = _loader.download("GSE1")
    cache = tmp_path / "cache"
    other = tmp_path / "elsewhere" / "GSE1.singlet"
    other.parent.mkdir()
    real.replace(other)
    try:
        (cache / "GSE1.singlet").symlink_to(other)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    hits = server.hits
    assert _loader.download("GSE1").exists()
    assert server.hits == hits


def test_gsm_ids_with_a_suffix_resolve_but_cannot_be_downloaded_as_bundles():
    assert normalize_accession("gsm7152960_val") == "GSM7152960_VAL"
    with pytest.raises(ValueError):
        _loader.download("GSM7152960")
    with pytest.raises(ValueError):
        _loader.download("GSE1_val")
