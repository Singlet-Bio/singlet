import json
import os

import pytest

import ab
from helpers import leg, result, t1, t2

BIN_A = "bin=/mnt/x/bin/s3"
BIN_B = "bin=/mnt/x/bin/e20w,env=SINGLET_X=1"
GOOD = ("a" * 40, "", "a" * 40)


class Fake:
    """A scripted stand-in for Remote. `states` entries are strings (the job state) or the BridgeDown class."""

    def __init__(self, result_json=None, states=("PENDING", "RUNNING", "COMPLETED"), up=True):
        self.result_json, self.states, self.up = result_json, list(states), up
        self.calls, self.puts = [], []

    def status(self):
        return self.up

    def put(self, local, remote):
        self.puts.append(remote)

    def run(self, cmd, timeout=300):
        self.calls.append(cmd)
        if "sbatch --parsable" in cmd:
            return 0, "4242\n", ""
        if "squeue -j" in cmd:
            s = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if s is ab.BridgeDown:
                raise ab.BridgeDown("down")
            return 0, f"{s}\n", ""
        if "result.json" in cmd:
            return (0, json.dumps(self.result_json), "") if self.result_json else (1, "", "")
        return 0, "", ""


@pytest.fixture(autouse=True)
def clean_git(monkeypatch):
    monkeypatch.setattr(ab, "local_state", lambda: GOOD)


def argv(tmp_path, tier=1, extra=()):
    rule = ["--rule", "gain"] if tier == 1 and "--rule" not in extra else []   # the old tier 1 rule, which these synthetic results were written for
    return ["--tier", str(tier), "--a", BIN_A, "--b", BIN_B, "--name", "idea", "--out", str(tmp_path / "o"), "--poll", "0", *rule, *extra]


def test_dry_run_prints_spec_and_sbatch_and_touches_nothing(tmp_path, capsys):
    rc = ab.main(argv(tmp_path, 1, ["--dry-run"]), remote=Fake(up=False))
    out = capsys.readouterr().out
    assert rc == 0 and not (tmp_path / "o").exists()
    assert "sbatch --parsable --job-name=gf1-idea" in out and "ENV_B=(SINGLET_X=1)" in out and "t1.sbatch" in out
    assert "EXPECT_SCRIPTS=aaaaaaaaaaaa" in out and "SBATCH_SHA=" in out


def test_dry_run_tier2_has_load_and_rule(tmp_path, capsys):
    rc = ab.main(argv(tmp_path, 2, ["--dry-run"]), remote=Fake(up=False))
    cap = capsys.readouterr()
    assert rc == 0 and "K=8" in cap.out and "NRUNS=24" in cap.out and "--exclude" not in cap.out and "SEP_RATE=0.09" in cap.out
    assert "provisional rule" in cap.err


def test_dry_run_reports_unpushed_head(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ab, "local_state", lambda: ("a" * 40, " M x", "b" * 40))
    ab.main(argv(tmp_path, 1, ["--dry-run"]), remote=Fake())
    out = capsys.readouterr().out
    assert "A real run would refuse" in out and "uncommitted" in out and "not the tip" in out


def test_real_run_refuses_unpushed_head(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ab, "local_state", lambda: ("a" * 40, "", "b" * 40))
    assert ab.main(argv(tmp_path, 1), remote=Fake()) == 2
    assert "Push it first" in capsys.readouterr().err


def test_real_run_never_starts_a_bridge(tmp_path, capsys):
    assert ab.main(argv(tmp_path, 1), remote=Fake(up=False)) == 2
    assert "does not start it" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [["--name", "bad name"], ["--order", "AAAA"], ["--sample", "x/y"], ["--ref", "relative"]])
def test_bad_arguments(tmp_path, capsys, bad):
    assert ab.main(argv(tmp_path, 1, bad + ["--dry-run"]), remote=Fake()) == 2


def test_ref_needed_for_other_sample(tmp_path):
    assert ab.main(argv(tmp_path, 1, ["--sample", "pbmc5p_v1", "--dry-run"])) == 2
    assert ab.main(argv(tmp_path, 1, ["--sample", "pbmc5p_v1", "--ref", "/mnt/r/x", "--dry-run"])) == 0


def test_full_flow_tier1_promote(tmp_path, capsys):
    res = t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5))
    f = Fake(res)
    out = []
    rc = ab.main(argv(tmp_path, 1), remote=f, out_fn=out.append)
    assert rc == 0
    o = tmp_path / "o"
    v = json.load(open(o / "verdict.json"))
    assert v["verdict"] == "PROMOTE" and v["rule"] == "T1-5" and v["job"] == "4242" and v["job_state"] == "COMPLETED"
    assert (o / "table.md").exists() and (o / "result.json").exists() and (o / "state.json").exists()
    assert any("VERDICT: PROMOTE" in x for x in out)
    assert len(f.puts) == 2 and f.puts[0].endswith("t1.sbatch") and f.puts[1].endswith("spec.sh")
    assert "# written by ab.py" in open(o / "spec.sh").read()
    assert b"" + bytes([13]) not in open(o / "spec.sh", "rb").read()


def test_tier2_rule_parameters_reach_the_verdict(tmp_path):
    res = t2([("A", 2.0), ("B", 2.1), ("B", 2.1), ("A", 2.0)])
    ab.main(argv(tmp_path, 2, ["--rate-gain", "0.04"]), remote=Fake(res), out_fn=lambda x: None)
    v = json.load(open(tmp_path / "o" / "verdict.json"))
    assert v["verdict"] == "PROMOTE" and v["params"]["rate_gain"] == 0.04 and v["params"]["core_gain"] == 0.02


def test_tier2_default_rule(tmp_path):
    res = t2([("A", 2.0), ("B", 2.1), ("B", 2.1), ("A", 2.0)])
    ab.main(argv(tmp_path, 2), remote=Fake(res), out_fn=lambda x: None)
    v = json.load(open(tmp_path / "o" / "verdict.json"))
    assert v["verdict"] == "INCONCLUSIVE" and v["rule"] == "T2-7"


def test_job_without_result_is_inconclusive(tmp_path, capsys):
    rc = ab.main(argv(tmp_path, 1), remote=Fake(None, states=("RUNNING", "FAILED")), out_fn=lambda x: None)
    v = json.load(open(tmp_path / "o" / "verdict.json"))
    assert rc == 0 and v["verdict"] == "INCONCLUSIVE" and v["rule"] == "J1"


def test_bridge_drop_during_wait_recovers(tmp_path):
    res = t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5))
    f = Fake(res, states=("RUNNING", ab.BridgeDown, ab.BridgeDown, "COMPLETED"))
    rc = ab.main(argv(tmp_path, 1), remote=f, out_fn=lambda x: None)
    assert rc == 0 and json.load(open(tmp_path / "o" / "verdict.json"))["verdict"] == "PROMOTE"


def test_bridge_down_too_long_gives_unknown_state_and_resume(tmp_path, capsys):
    f = Fake(None, states=("RUNNING", ab.BridgeDown))
    rc = ab.main(argv(tmp_path, 1, ["--bridge-grace", "0"]), remote=f, out_fn=lambda x: None)
    err = capsys.readouterr().err
    assert rc == 3 and "UNKNOWN" in err and "--resume" in err
    assert not (tmp_path / "o" / "verdict.json").exists()
    state = json.load(open(tmp_path / "o" / "state.json"))
    assert state["job"] == "4242"
    # the bridge is back: resume finishes the job
    res = t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5))
    rc = ab.main(["--resume", str(tmp_path / "o"), "--poll", "0"], remote=Fake(res), out_fn=lambda x: None)
    assert rc == 0 and json.load(open(tmp_path / "o" / "verdict.json"))["verdict"] == "PROMOTE"


def test_bridge_drop_during_submit_is_resumable_by_job_name(tmp_path, capsys):
    class DropAtSubmit(Fake):
        def run(self, cmd, timeout=300):
            if "sbatch --parsable" in cmd:
                raise ab.BridgeDown("down")
            if "squeue -u $USER -n" in cmd:
                return 0, "4243\n", ""
            return super().run(cmd, timeout)
    rc = ab.main(argv(tmp_path, 1), remote=DropAtSubmit(), out_fn=lambda x: None)
    assert rc == 3 and "state unknown" in capsys.readouterr().err
    res = t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5))
    f = DropAtSubmit(res)
    f.run = lambda cmd, timeout=300, _f=Fake(res): _f.run(cmd, timeout) if "squeue -u $USER -n" not in cmd else (0, "4243\n", "")
    rc = ab.main(["--resume", str(tmp_path / "o"), "--poll", "0"], remote=f, out_fn=lambda x: None)
    assert rc == 0
    assert json.load(open(tmp_path / "o" / "state.json"))["job"] == "4243"


def test_parse_state():
    assert ab.parse_state("RUNNING\nRUNNING\n") == "RUNNING"
    assert ab.parse_state("") == ""
    assert ab.parse_state("  COMPLETED  \n") == "COMPLETED"
    assert ab.parse_state("CANCELLED+\n") == "CANCELLED"


def test_tier1_default_rule_is_the_demand_rule(tmp_path):
    from test_demand import dleg
    from helpers import result
    res = result([dleg(1, "A", 8.94, 39.1), dleg(2, "B", 7.79, 27.6), dleg(3, "B", 7.79, 27.6), dleg(4, "A", 8.94, 39.1)], 1)
    ab.main(["--tier", "1", "--a", BIN_A, "--b", BIN_B, "--name", "idea", "--out", str(tmp_path / "o"), "--poll", "0"], remote=Fake(res), out_fn=lambda x: None)
    v = json.load(open(tmp_path / "o" / "verdict.json"))
    assert v["verdict"] == "PROMOTE" and v["rule"] == "T1D-3" and v["params"]["node_gain"] == 0.05


def test_after_adds_a_dependency(tmp_path, capsys):
    assert ab.main(argv(tmp_path, 2, ["--dry-run", "--after", "800408"])) == 0
    assert "--dependency=afterany:800408" in capsys.readouterr().out
    assert ab.main(argv(tmp_path, 2, ["--dry-run", "--after", "80x"])) == 2


def test_after_rejects_non_numbers(tmp_path):
    f = Fake()
    assert ab.main(argv(tmp_path, 2, ["--after", "abc"]), remote=f, out_fn=lambda x: None) == 2
