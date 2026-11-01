"""common.require_pinned_binary: the campaign never runs an engine from a shared mutable directory."""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "atlas_rerun"))
import common as C  # noqa: E402


@pytest.fixture(autouse=True)
def _checks_on(monkeypatch):
    monkeypatch.delenv("S9_NO_WL_CHECK", raising=False)


def build(root, name, commit, layout="bin"):
    d = root / name
    (d / "bin" if layout == "bin" else d).mkdir(parents=True)
    f = (d / "bin" / "singlet-gpu") if layout == "bin" else (d / "singlet-gpu")
    f.write_text("engine")
    if commit is not None:
        (d / "commit").write_text(commit + "\n")
    return str(f)


def test_commit_named_build_is_accepted(tmp_path):
    assert C.require_pinned_binary(build(tmp_path, "bbf6497d", "bbf6497d5f22438203b2")) == "bbf6497d5f22438203b2"
    assert C.require_pinned_binary(build(tmp_path, "bin_9a1b2c3d", "9a1b2c3d99", layout="flat")) == "9a1b2c3d99"


def test_shared_directories_are_refused(tmp_path):
    for shared in ("bin", "s8", "engine"):
        b = build(tmp_path, shared, "bbf6497d5f", layout="flat")
        with pytest.raises(SystemExit) as e:
            C.require_pinned_binary(b)
        assert "not a pinned build" in str(e.value)


def test_commit_file_must_match_directory_and_config(tmp_path):
    with pytest.raises(SystemExit):
        C.require_pinned_binary(build(tmp_path, "aaaaaaaa", None))                    # no commit file
    with pytest.raises(SystemExit):
        C.require_pinned_binary(build(tmp_path, "bbbbbbbb", "cccccccc11"))            # another commit than the name says
    ok = build(tmp_path, "dddddddd", "dddddddd22")
    assert C.require_pinned_binary(ok, "dddddddd22")
    with pytest.raises(SystemExit) as e:
        C.require_pinned_binary(ok, "eeeeeee")                                         # the config pins another commit
    assert "config pins" in str(e.value)


def test_missing_binary_and_content_addressed_copy(tmp_path):
    with pytest.raises(SystemExit):
        C.require_pinned_binary(str(tmp_path / "nope" / "singlet"))
    data = b"engine bytes"
    f = tmp_path / ("singlet-" + hashlib.sha256(data).hexdigest()[:12])
    f.write_bytes(data)
    assert C.require_pinned_binary(str(f)) == f.name[len("singlet-"):]
    g = tmp_path / "singlet-000000000000"
    g.write_bytes(data)
    with pytest.raises(SystemExit):
        C.require_pinned_binary(str(g))


def test_bypass_for_fake_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("S9_NO_WL_CHECK", "1")
    assert C.require_pinned_binary(str(tmp_path / "anything")) is None
