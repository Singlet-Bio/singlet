"""scripts/store_rekey.py on a local stand-in for R2: copies, rewritten manifest and pack index, idempotent apply, verify."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
_spec = importlib.util.spec_from_file_location("store_rekey", REPO / "scripts" / "store_rekey.py")
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)

from singlet.store import manifest as M  # noqa: E402

GSE, GSMS = "GSE000001", ("GSM000001", "GSM000002")


def _seed(root: Path) -> dict:
    b = root / rk.BUCKET
    samples = []
    for i, gsm in enumerate(GSMS):
        body = bytes([i]) * 1000
        key = f"v2/s/{gsm}/{hashlib.sha256(body).hexdigest()[:32]}.s2"
        (b / key).parent.mkdir(parents=True, exist_ok=True)
        (b / key).write_bytes(body)
        samples.append(
            {"gsm": gsm, "key": key, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        )
    pack = b"SPK2" + b"\0" * 60
    pid = hashlib.sha256(pack).hexdigest()[:32]
    (b / "v2/ref").mkdir(parents=True)
    (b / f"v2/ref/{pid}.pack").write_bytes(pack)
    (b / "v2/ref/index.json").write_text(
        json.dumps({"format": "singlet-v2-packs", "version": 1, "packs": [{"id": pid}]})
    )
    (b / "v2/ledger").mkdir(parents=True)
    (b / f"v2/ledger/{GSE}.jsonl").write_text('{"type": "study"}\n')
    m = {
        "format": "singlet-v2",
        "manifest_version": 1,
        "gse": GSE,
        "rev": M.compute_rev([s["key"] for s in samples], {}),
        "study_files": {},
        "samples": samples,
        "legacy": {"key": f"data/{GSE}/{GSE}.singlet", "ledger": f"v2/ledger/{GSE}.jsonl"},
    }
    (b / "v2/m").mkdir(parents=True)
    (b / f"v2/m/{GSE}.json").write_bytes(M.dumps_manifest(m))
    return m


def test_plan_apply_verify(tmp_path, capsys):
    root, out = tmp_path / "r2", tmp_path / "plan"
    old = _seed(root)
    before = {p: p.read_bytes() for p in (root / rk.BUCKET / "v2").rglob("*") if p.is_file()}
    args = ["--out", str(out), "--local-root", str(root)]
    assert rk.main(["plan", *args]) == 0
    assert rk.main(["apply", *args]) == 0
    assert rk.main(["apply", *args]) == 0  # idempotent
    assert rk.main(["verify", *args]) == 0

    b = root / rk.BUCKET
    new = json.loads((b / f"store/m/{GSE}.json").read_bytes())
    assert new["format"] == "singlet-store" and new["rev"] != old["rev"]
    assert [s["key"] for s in new["samples"]] == [rk.new_key(s["key"]) for s in old["samples"]]
    assert all(
        k.startswith("store/s/") and k.endswith(".s1") for k in (s["key"] for s in new["samples"])
    )
    assert (
        new["legacy"]["ledger"] == f"store/ledger/{GSE}.jsonl"
        and new["legacy"]["key"] == old["legacy"]["key"]
    )
    assert new["rev"] == M.compute_rev([s["key"] for s in new["samples"]], {})
    for s_old, s_new in zip(old["samples"], new["samples"]):
        assert (b / s_new["key"]).read_bytes() == (b / s_old["key"]).read_bytes()
    assert json.loads((b / "store/ref/index.json").read_bytes())["format"] == "singlet-store-packs"
    assert (b / f"store/ledger/{GSE}.jsonl").read_bytes() == (
        b / f"v2/ledger/{GSE}.jsonl"
    ).read_bytes()
    # the old tree is untouched
    assert before == {p: p.read_bytes() for p in (b / "v2").rglob("*") if p.is_file()}
    sql = (out / "catalog.sql").read_text()
    assert f"rev = '{new['rev']}'" in sql and f"AND rev = '{old['rev']}'" in sql


def test_apply_refuses_to_overwrite_different_object(tmp_path):
    root, out = tmp_path / "r2", tmp_path / "plan"
    old = _seed(root)
    args = ["--out", str(out), "--local-root", str(root)]
    rk.main(["plan", *args])
    dst = root / rk.BUCKET / rk.new_key(old["samples"][0]["key"])
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"something else")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rk.main(["apply", *args])
