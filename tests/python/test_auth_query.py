# SPDX-License-Identifier: MIT
"""Tests for the retired dead-host entry points: login, singlet.atlas, fetch().

Each targeted a host that was never in service (api.singlet.bio/v1,
r2.singlet.bio, data.singlet.bio/v1). They must raise NotImplementedError
with a pointer to singlet.load / singlet.find — and fetch() must keep working
against a mirror given as base_url= or $SINGLET_SAMPLE_MIRROR.
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

    def test_login_message_says_what_to_do_and_hides_the_key(self):
        with pytest.raises(NotImplementedError) as exc:
            auth.login("sk-secret-123")
        msg = str(exc.value)
        assert "delete the singlet.login(...) line" in msg
        assert "need no key" in msg
        assert "$SINGLET_API_KEY" in msg
        assert "sk-secret-123" not in msg

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


def _mirror(tmp_path, gsm: str, payload: bytes = b"summary contents") -> str:
    """A file:// mirror holding one sample directory; returns its base URL."""
    sample = tmp_path / "mirror" / gsm
    sample.mkdir(parents=True)
    (sample / "summary.json").write_bytes(payload)
    (sample / "manifest.json").write_text(
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
    return (tmp_path / "mirror").as_uri()


@pytest.fixture
def no_mirror_env(monkeypatch):
    monkeypatch.delenv("SINGLET_SAMPLE_MIRROR", raising=False)
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    return monkeypatch


class TestFetch:
    def test_default_host_is_retired(self, tmp_path, no_mirror_env):
        with pytest.raises(NotImplementedError, match="singlet.load"):
            fetch_mod.fetch("GSM3308814", cache_dir=tmp_path)

    def test_default_base_url_is_the_retired_sample_host(self, tmp_path, no_mirror_env):
        assert fetch_mod.default_base_url() == "https://data.singlet.bio/v1"
        # Passing it back in explicitly is refused the same way, not a 404.
        with pytest.raises(NotImplementedError, match="SINGLET_SAMPLE_MIRROR"):
            fetch_mod.fetch("GSM1", cache_dir=tmp_path, base_url=fetch_mod.default_base_url())

    def test_sample_mirror_env_is_used(self, tmp_path, no_mirror_env):
        base = _mirror(tmp_path, "GSM5")
        no_mirror_env.setenv("SINGLET_SAMPLE_MIRROR", base + "/")
        assert fetch_mod.default_base_url() == base
        out = fetch_mod.fetch("GSM5", cache_dir=tmp_path / "cache")
        assert (out / "summary.json").read_bytes() == b"summary contents"
        assert singlet.open("GSM5", cache_dir=tmp_path / "cache").path == out

    def test_legacy_data_base_still_names_a_mirror_with_a_warning(self, tmp_path, no_mirror_env):
        base = _mirror(tmp_path, "GSM6")
        no_mirror_env.setenv("SINGLET_DATA_BASE", base)
        with pytest.warns(FutureWarning, match="SINGLET_SAMPLE_MIRROR"):
            out = fetch_mod.fetch("GSM6", cache_dir=tmp_path / "cache")
        assert (out / "summary.json").exists()

    def test_legacy_data_base_404_explains_the_rename(self, tmp_path, no_mirror_env, monkeypatch):
        no_mirror_env.setenv("SINGLET_DATA_BASE", "https://data.singlet.bio")

        def not_found(url, dest):
            raise FileNotFoundError(f"{url} → HTTP 404")

        monkeypatch.setattr(fetch_mod, "_http_get", not_found)
        with pytest.warns(FutureWarning):
            with pytest.raises(FileNotFoundError, match="SINGLET_SAMPLE_MIRROR"):
                fetch_mod.fetch("GSM7", cache_dir=tmp_path / "cache")

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

    def test_open_sample_accession_without_mirror_raises(self, tmp_path, no_mirror_env):
        with pytest.raises(NotImplementedError):
            singlet.open("GSM_NOT_LOCAL", cache_dir=tmp_path)
