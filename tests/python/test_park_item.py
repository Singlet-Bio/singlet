"""A parked (no GEX to stage) sample keeps its companion hand-over in failed/<GSM>.fetch/item.json and leaves nothing in incoming/."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "atlas_rerun"))
import fetch_worker as F  # noqa: E402
import stagequeue as Q  # noqa: E402


def test_parked_sample_records_companion_candidates_and_frees_incoming(tmp_path):
    stage = str(tmp_path / "stage")
    Q.init(stage)
    item = {"gsm": "GSM9", "gse": "GSE9", "organism": "Homo sapiens", "runs": ["R1", "R2"], "run_reads": [123, 205]}
    claim = os.path.join(stage, "fetching", "GSM9")
    os.makedirs(claim)
    json.dump(item, open(os.path.join(claim, "item.json"), "w"))
    inc = Q.new_incoming(stage, "GSM9", "w1")
    open(os.path.join(inc, "R1.1fq"), "w").write("reads")
    cands = [{"lib_gsm": None, "lib_type": None, "status": "unclassified", "protocol_family": "agnostic-bc6+umi7", "protocol_tags": {"R2": "agnostic-bc6+umi7"}, "runs": ["R2"], "reads": 205}]
    vote = {"status": "agree", "excluded_runs": []}
    assert F.park_item(stage, "GSM9", claim, inc, item, "parked_no_gex", vote, cands, {"R1": ("10x-3p-v2", 3, 0.97)}) == "parked_no_gex"
    assert not os.path.exists(inc) and os.listdir(os.path.join(stage, "incoming")) == []
    rec = json.load(open(os.path.join(stage, "failed", "GSM9.fetch", "item.json")))
    assert rec["runs"] == ["R1", "R2"] and rec["parked"]["reason"] == "parked_no_gex"
    assert rec["parked"]["companion_candidates"] == cands and rec["parked"]["protocol_vote"] == vote and rec["parked"]["calls"] == {"R1": ["10x-3p-v2", 3, 0.97]}
    assert not os.path.exists(claim)


def test_parked_sample_without_vote_has_only_the_reason(tmp_path):
    stage = str(tmp_path / "stage")
    Q.init(stage)
    claim = os.path.join(stage, "fetching", "GSM8")
    os.makedirs(claim)
    inc = Q.new_incoming(stage, "GSM8", "w1")
    F.park_item(stage, "GSM8", claim, inc, {"gsm": "GSM8", "runs": ["R"]}, "parked_non10x")
    assert json.load(open(os.path.join(stage, "failed", "GSM8.fetch", "item.json")))["parked"] == {"reason": "parked_non10x"}
