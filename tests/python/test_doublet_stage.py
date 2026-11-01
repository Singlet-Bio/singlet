# SPDX-License-Identifier: MIT
"""Doublet stage: the engine's score always becomes the private doublet_scores_engine.tsv; the public name is the stacker's score or absent."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

import doublet_stage as DS  # noqa: E402


def _out(tmp_path, species="human", cells=5):
    d = tmp_path / "out"
    d.mkdir()
    (d / "doublet_scores.tsv").write_text("barcode\ttotal_umis\tdoublet_score\tis_doublet\n" + "".join(
        "BC%d\t100\t0.1\tFalse\n" % i for i in range(cells)))
    (d / "summary.json").write_text(json.dumps({"species": species}))
    return str(d)


def test_failure_keeps_engine_score_private_and_publishes_none(tmp_path):
    out = _out(tmp_path)                                  # no cell_calls.tsv / count matrices: the stage fails
    rec = DS.run(out, log=lambda *a: None)
    assert rec["status"] == "failed" and rec["wall_s"] >= 0 and "peak_rss_gb" in rec
    assert not os.path.exists(os.path.join(out, "doublet_scores.tsv"))      # no public score of another method
    assert os.path.exists(os.path.join(out, "doublet_scores_engine.tsv"))
    assert json.load(open(os.path.join(out, "doublet_stage.json")))["status"] == "failed"


def test_missing_engine_score_is_recorded(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    rec = DS.run(str(out), log=lambda *a: None)
    assert rec["status"] == "failed" and "no engine doublet_scores.tsv" in rec["reason"]


def test_engine_score_is_a_withheld_layer():
    from singlet._restricted import is_restricted_member
    from singlet.s1 import policy

    assert is_restricted_member("doublet_scores_engine.tsv")
    assert policy.omitted_layer("doublet_scores_engine.tsv") == "doublet_scores_engine"
    assert policy.omitted_layer("doublet_scores_engine.tsv", ("doublet_scores",)) == "doublet_scores_engine"   # never liftable
    assert policy.omitted_layer("doublet_scores.tsv") == "doublet_scores"                                       # Alpha: withheld


def test_beta_policy_publishes_only_the_stackers_doublet_scores(tmp_path):
    from singlet.s1 import policy

    here = os.path.join(HERE, "..", "..", "scripts", "atlas_rerun", "beta-policy.json")
    beta = policy.Policy.load(here)
    assert beta.published == frozenset({"doublet_scores"})
    assert policy.omitted_layer("doublet_scores.tsv", beta.published) is None
    assert policy.omitted_layer("doublet_scores_engine.tsv", beta.published) == "doublet_scores_engine"
    assert "doublet_scores" not in beta.manifest_block()["omitted_layers"]
    assert "doublet_scores_engine" in beta.manifest_block()["omitted_layers"]
    assert "ambient" in beta.manifest_block()["omitted_layers"]                                              # everything else stays withheld
    bad = tmp_path / "p.json"
    bad.write_text('{"version": "x", "withheld_studies": [], "excluded_samples": [], "published_layers": ["ambient"]}')
    try:
        policy.Policy.load(str(bad))
        raise AssertionError("ambient must not be liftable")
    except ValueError:
        pass
