# SPDX-License-Identifier: MIT
"""The deferred HLA route: the batch script reports every run that did not land, and pending/ (reads) can never reach a public member."""

import importlib.util
import json
import os
import shlex
import shutil
from pathlib import Path

import pytest
from singlet._restricted import is_restricted_member, is_restricted_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location(
    "hla_pending_batch", ROOT / "docs" / "gpu-engine" / "hla_pending_batch.py"
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

PENDING_FILES = ["private/hla/pending/candidate_reads.fq.gz", "private/hla/pending/manifest.json"]


@pytest.mark.parametrize("rel", PENDING_FILES)
def test_pending_is_never_a_public_member(rel):
    assert is_restricted_member(rel)
    assert is_restricted_path(f"samples/GSM0000881/{rel}")


def test_private_sidecar_rule_matches():
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "atlas_rerun"))
    import private_sidecar as PS

    for rel in PENDING_FILES:
        assert PS.is_withheld(rel)


def _run(tmp, name, genome="/ref/GRCh38-2024-A/star", state=B.PENDING):
    run = tmp / name
    hla = run / "private" / "hla"
    hla.mkdir(parents=True)
    (run / "pileup_stats.json").write_text(json.dumps({"provenance": {"genome_dir": genome}}))
    if state == B.PENDING:
        (hla / "pending").mkdir()
        (hla / "pending" / "manifest.json").write_text("{}")
    elif state == B.LANDED:
        (hla / "hla_summary.json").write_text("{}")
    return str(run)


def _list(tmp, runs):
    p = tmp / "runs.txt"
    p.write_text("# batch\n" + "\n".join(runs) + "\n\n")
    return str(p)


def _land(run):
    hla = Path(run) / "private" / "hla"
    (hla / "hla_summary.json").write_text("{}")
    shutil.rmtree(hla / "pending")


def test_states(tmp_path):
    assert B.run_state(_run(tmp_path, "a")) == B.PENDING
    assert B.run_state(_run(tmp_path, "b", state=B.LANDED)) == B.LANDED
    assert B.run_state(_run(tmp_path, "c", state=B.MISSING)) == B.MISSING
    assert (
        B.run_state(_run(tmp_path, "m", genome="/ref/GRCm39-2024-A/star", state=B.MISSING))
        == B.NOT_APPLICABLE
    )
    assert (
        B.run_state(
            _run(tmp_path, "bn", genome="/ref/GRCh38-GRCm39-barnyard/star", state=B.MISSING)
        )
        == B.NOT_APPLICABLE
    )


def test_a_summary_beside_a_pending_manifest_has_not_landed(tmp_path):
    run = _run(tmp_path, "a")
    (Path(run) / "private" / "hla" / "hla_summary.json").write_text("{}")
    assert B.run_state(run) == B.PENDING


def test_a_run_without_a_genome_record_is_reported(tmp_path):
    run = tmp_path / "x"
    run.mkdir()
    assert B.run_state(str(run)) == B.MISSING


def test_check_only_fails_loudly(tmp_path, capsys):
    runs = [_run(tmp_path, "a", state=B.LANDED), _run(tmp_path, "b", state=B.MISSING)]
    assert B.main([_list(tmp_path, runs), "--check-only"]) == 1
    assert f"HLA NOT LANDED (missing): {runs[1]}" in capsys.readouterr().err


def test_all_landed_exits_zero(tmp_path):
    runs = [
        _run(tmp_path, "a", state=B.LANDED),
        _run(tmp_path, "m", genome="/ref/GRCm39/star", state=B.MISSING),
    ]
    assert B.main([_list(tmp_path, runs), "--check-only"]) == 0


def test_empty_list_fails(tmp_path):
    assert B.main([_list(tmp_path, []), "--check-only"]) == 1


def _args(tmp, runs):
    return [_list(tmp, runs), "--singlet", "/bin/singlet", "--log-dir", str(tmp / "logs")]


def test_submits_pending_runs_and_retries_only_the_failures(tmp_path, monkeypatch):
    runs = [_run(tmp_path, "a"), _run(tmp_path, "b"), _run(tmp_path, "c", state=B.LANDED)]
    seen = []

    def fake(cmd, check):
        list_file = shlex.split(cmd[cmd.index("--wrap") + 1].split('p" ')[1].split(")")[0])[0]
        todo = open(list_file).read().split()
        seen.append(todo)
        if len(seen) == 1:
            _land(todo[0])  # the first attempt types one run; the other task fails
        else:
            for r in todo:
                _land(r)

    monkeypatch.setattr(B.subprocess, "run", fake)
    assert B.main(_args(tmp_path, runs)) == 0
    assert seen == [[runs[0], runs[1]], [runs[1]]]


def test_a_run_that_never_lands_exits_nonzero(tmp_path, monkeypatch, capsys):
    runs = [_run(tmp_path, "a")]
    calls = []
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, check: calls.append(cmd))
    assert B.main(_args(tmp_path, runs)) == 1
    assert len(calls) == 2  # the first attempt and one retry
    assert f"HLA NOT LANDED (pending): {runs[0]}" in capsys.readouterr().err


def test_the_array_runs_singlet_hla_pending_on_cpu_nodes(tmp_path):
    a = B.argparse.Namespace(
        singlet="/b/singlet gpu",
        partition="short-cpu",
        threads=8,
        mem="32G",
        time="03:59:00",
        nice=0,
        array_limit=4,
        log_dir=str(tmp_path),
        account="debruinz",
    )
    cmd = B.sbatch_command(a, os.path.join(str(tmp_path), "l st"), 3)
    assert (
        "--wait" in cmd
        and "--array=0-2%4" in cmd
        and "--partition=short-cpu" in cmd
        and "--account=debruinz" in cmd
    )
    assert "'/b/singlet gpu' hla --pending \"$RUN/private/hla/pending\"" in cmd[-1]
