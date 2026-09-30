# SPDX-License-Identifier: MIT
"""Tests for the retired dead-host entry points: login, singlet.atlas, fetch().

Each targeted a host that was never in service (api.singlet.bio/v1,
r2.singlet.bio, data.singlet.bio/v1). They must raise NotImplementedError
with a pointer to singlet.load / singlet.find — and fetch() must keep working
against an explicitly given mirror.
"""

import hashlib
import importlib
import json

import pytest
import singlet
import singlet._auth as auth

# The package re-exports the function under the module's name, so
# `import singlet.fetch` would not give the module.
fetch_mod = importlib.import_module("singlet.fetch")


class TestLogin:
    def test_login_raises_with_guidance(self):
        with pytest.raises(NotImplementedError, match="set_api_key"):
            auth.login("sk-test-key-123")

    def test_login_without_key_raises_the_same(self, monkeypatch):
        monkeypatch.setenv("SINGLET_API_KEY", "sk-env")
        with pytest.raises(NotImplementedError, match="api.singlet.bio/v1"):
            singlet.login()

    def test_retired_message_names_the_function(self):
        assert auth.retired("singlet.query()").startswith("singlet.query() targeted")


class TestAtlas:
    @pytest.mark.parametrize(
        ("name", "args", "kwargs", "hint"),
        [
            ("index", (), {}, "singlet.catalog"),
            ("sample", ("GSM4120733",), {}, "singlet.info('GSM4120733')"),
            ("search", (), {"tissue": "lung"}, "singlet.find"),
        ],
    )
    def test_atlas_functions_raise(self, name, args, kwargs, hint):
        import singlet.atlas

        with pytest.raises(NotImplementedError) as exc:
            getattr(singlet.atlas, name)(*args, **kwargs)
        assert "not in service" in str(exc.value)
        assert hint in str(exc.value)


class TestFetch:
    def test_default_host_is_retired(self, tmp_path):
        with pytest.raises(NotImplementedError, match="singlet.load"):
            fetch_mod.fetch("GSM3308814", cache_dir=tmp_path)

    def test_default_base_url_is_the_bundle_host(self, monkeypatch):
        monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
        assert fetch_mod.default_base_url() == "https://data.singlet.bio"
        monkeypatch.setenv("SINGLET_DATA_BASE", "https://mirror.test/data/")
        assert fetch_mod.default_base_url() == "https://mirror.test"

    def test_explicit_mirror_still_works(self, tmp_path):
        """fetch() against a file:// mirror: manifest, sha256 check, caching."""
        mirror = tmp_path / "mirror" / "GSM1"
        mirror.mkdir(parents=True)
        payload = b"summary contents"
        (mirror / "summary.json").write_bytes(payload)
        (mirror / "manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "path": "summary.json",
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ]
                }
            )
        )
        base = (tmp_path / "mirror").as_uri()
        out = fetch_mod.fetch("GSM1", cache_dir=tmp_path / "cache", base_url=base)
        assert (out / "summary.json").read_bytes() == payload
        # Second call finds the file already cached with the right digest.
        again = fetch_mod.fetch("GSM1", cache_dir=tmp_path / "cache", base_url=base)
        assert again == out

    def test_checksum_mismatch_is_reported(self, tmp_path):
        mirror = tmp_path / "mirror" / "GSM2"
        mirror.mkdir(parents=True)
        (mirror / "a.txt").write_bytes(b"real")
        (mirror / "manifest.json").write_text(
            json.dumps({"files": [{"path": "a.txt", "sha256": "0" * 64}]})
        )
        base = (tmp_path / "mirror").as_uri()
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            fetch_mod.fetch("GSM2", cache_dir=tmp_path / "cache", base_url=base)

    def test_open_sample_accession_without_mirror_raises(self, tmp_path):
        with pytest.raises(NotImplementedError):
            singlet.open("GSM_NOT_LOCAL", cache_dir=tmp_path)
