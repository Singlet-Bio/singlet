import json
import os

import pytest

import ab
import campaign
from helpers import leg, result
from test_demand import dleg

GOOD = ("a" * 40, "", "a" * 40)
S3 = "bin=/mnt/x/bin/s3,commit=e42c0636"


@pytest.fixture(autouse=True)
def clean_git(monkeypatch):
    monkeypatch.setattr(ab, "local_state", lambda: GOOD)


def cfg_file(tmp_path, **kw):
    cfg = {"name": "c1", "defaults": {"node": "g051", "slots": 2},
           "steps": [{"id": "t1x", "tier": 1, "a": S3, "b": "bin=/mnt/x/bin/e20w,commit=aeef7a5e"},
                     {"id": "t1y", "tier": 1, "a": S3, "b": S3 + ",env=SINGLET_Y=1"},
                     {"id": "batch", "tier": 2, "order": "ABCCBA", "arms": {"A": S3, "B": S3 + ",env=SINGLET_X=1", "C": S3 + ",env=SINGLET_Y=1"},
                      "predict": {"B": ["t1x"], "C": ["t1y"]}, "opts": {"runs": 24}}]}
    cfg.update(kw)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    return str(p)


class Chain:
    """A fake Remote. sbatch gives job ids 100, 101, ... Results are looked up by the run directory in the command."""

    def __init__(self, results=None, state="COMPLETED"):
        self.results, self.state, self.cmds, self.puts, self.n = results or {}, state, [], [], 99

    def status(self):
        return True

    def put(self, local, remote):
        self.puts.append(remote)

    def run(self, cmd, timeout=300):
        self.cmds.append(cmd)
        if "sbatch --parsable" in cmd:
            self.n += 1
            return 0, f"{self.n}\n", ""
        if "squeue -j" in cmd:
            return 0, f"{self.state}\n", ""
        if "result.json" in cmd:
            for key, r in self.results.items():
                if f"/{key}-t" in cmd:
                    return 0, json.dumps(r), ""
            return 1, "", ""
        return 0, "", ""


def test_step_argv_and_config_checks(tmp_path):
    cfg = campaign.load_config(cfg_file(tmp_path))
    argv = campaign.step_argv("c1", cfg["defaults"], cfg["steps"][2])
    assert argv[:4] == ["--tier", "2", "--name", "c1-batch"] and "--arm" in argv and "C=" + S3 + ",env=SINGLET_Y=1" in argv
    assert ["--node", "g051"] == argv[argv.index("--node"):argv.index("--node") + 2] and "--order" in argv
    bad = cfg_file(tmp_path, steps=[{"id": "x", "tier": 1, "a": S3, "b": S3, "after": ["later"]}, {"id": "later", "tier": 1, "a": S3, "b": S3}])
    with pytest.raises(ValueError):
        campaign.load_config(bad)


def test_dry_run_prints_the_chain(tmp_path, capsys):
    rc = ab.main(["--campaign", cfg_file(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("######## step") == 3
    assert "after ['<job of t1x>']" in out and "--dependency=afterany:<job of t1y>" in out and "ARMS=ABC" in out and "PIN_SCRIPTS=" + "a" * 40 in out
    assert "ORDER=ABCCBA" in out


def test_campaign_submits_one_chain_and_collect_reads_it_later(tmp_path):
    out_dir = tmp_path / "camp"
    f = Chain()
    rc = ab.main(["--campaign", cfg_file(tmp_path), "--out", str(out_dir)], remote=f, out_fn=lambda x: None)
    assert rc == 0
    subs = [c for c in f.cmds if "sbatch --parsable" in c]
    assert len(subs) == 3 and "--dependency" not in subs[0]
    assert "--dependency=afterany:100" in subs[1] and "--dependency=afterany:101" in subs[2]
    m = json.load(open(out_dir / "campaign.json"))
    assert [s["job"] for s in m["steps"]] == ["100", "101", "102"]
    assert any(p.endswith("campaign.json") for p in f.puts)
    # later: the bridge is up again and every step has written its result
    r1 = result([dleg(1, "A", 8.94, 39.1), dleg(2, "B", 7.79, 27.6), dleg(3, "B", 7.79, 27.6), dleg(4, "A", 8.94, 39.1)], 1)
    r2 = result([dleg(1, "A", 8.94, 39.1), dleg(2, "B", 8.9, 39.0), dleg(3, "B", 8.9, 39.0), dleg(4, "A", 8.94, 39.1)], 1)
    legs = []
    for i, arm in enumerate("ABCCBA"):
        l = leg(i + 1, arm, core={"A": 9.8, "B": 8.7, "C": 9.8}[arm], held=40.0, span=1.7)
        l["metrics"]["progress_rate"] = {"A": 1.9, "B": 2.2, "C": 1.9}[arm]
        legs.append(l)
    r3 = result(legs, 2)
    lines = []
    rc = ab.main(["--collect", str(out_dir)], remote=Chain({"c1-t1x": r1, "c1-t1y": r2, "c1-batch": r3}), out_fn=lines.append)
    text = "\n".join(lines)
    assert rc == 0 and "| t1x | 1 | PROMOTE (T1D-3)" in text and "| t1y | 1 | INCONCLUSIVE (T1D-4)" in text
    assert "B: PROMOTE" in text and "C: INCONCLUSIVE" in text
    v = json.load(open(out_dir / "steps" / "batch" / "verdict.json"))
    assert v["batch"] and v["predicted"]["B"] == pytest.approx(0.147, abs=0.01) and v["predicted"]["C"] == pytest.approx(0.0, abs=0.01)
    assert "Measured against predicted" in open(out_dir / "steps" / "batch" / "table.md").read()


def test_collect_reports_unfinished_steps_without_failing(tmp_path):
    out_dir = tmp_path / "camp"
    ab.main(["--campaign", cfg_file(tmp_path), "--out", str(out_dir)], remote=Chain(), out_fn=lambda x: None)
    lines = []
    assert ab.main(["--collect", str(out_dir)], remote=Chain(state="PENDING"), out_fn=lines.append) == 0
    assert "not finished" in "\n".join(lines)


def test_collect_with_a_missing_result_is_inconclusive(tmp_path):
    out_dir = tmp_path / "camp"
    ab.main(["--campaign", cfg_file(tmp_path), "--out", str(out_dir)], remote=Chain(), out_fn=lambda x: None)
    lines = []
    ab.main(["--collect", str(out_dir)], remote=Chain(state="FAILED"), out_fn=lines.append)
    assert "INCONCLUSIVE (J1)" in "\n".join(lines)


def test_bridge_drop_while_submitting_leaves_the_queued_steps_listed(tmp_path, capsys):
    class Drop(Chain):
        def run(self, cmd, timeout=300):
            if "sbatch --parsable" in cmd and self.n >= 100:
                raise ab.BridgeDown("down")
            return super().run(cmd, timeout)
    out_dir = tmp_path / "camp"
    rc = ab.main(["--campaign", cfg_file(tmp_path), "--out", str(out_dir)], remote=Drop(), out_fn=lambda x: None)
    assert rc == 3 and "Steps before it are queued" in capsys.readouterr().err
    assert json.load(open(out_dir / "campaign.json"))["steps"][0]["job"] == "100"


def test_batch_check_with_several_arms_through_ab(tmp_path):
    legs = []
    for i, arm in enumerate("ABCCBA"):
        l = leg(i + 1, arm, core=9.8, held=40.0, span=1.7)
        l["metrics"]["progress_rate"] = {"A": 1.9, "B": 2.2, "C": 1.9}[arm]
        legs.append(l)
    argv = ["--tier", "2", "--a", S3, "--b", S3 + ",env=X=1", "--arm", "C=" + S3 + ",env=Y=1", "--name", "bt", "--out", str(tmp_path / "o"), "--poll", "0",
            "--predict", "B=" + str(tmp_path / "none")]
    lines = []
    assert ab.main(argv, remote=Chain({"bt": result(legs, 2)}), out_fn=lines.append) == 0
    v = json.load(open(tmp_path / "o" / "verdict.json"))
    assert v["batch"] and v["arms"]["B"]["verdict"] == "PROMOTE" and v["arms"]["C"]["verdict"] == "INCONCLUSIVE"
    assert "ORDER=ABCCBA" in open(tmp_path / "o" / "spec.sh").read()


def test_arm_option_is_for_tier2_only(tmp_path):
    assert ab.main(["--tier", "1", "--a", S3, "--b", S3, "--arm", "C=" + S3, "--name", "x", "--dry-run"]) == 2


def test_the_example_campaign_loads_and_dry_runs(capsys):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "campaigns", "example-e20w.json")
    assert ab.main(["--campaign", p, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("######## step") == 3 and "ARMS=ABCD" in out and "ORDER=ABCDDCBA" in out


def test_external_after_and_a_final_summary_job(tmp_path):
    cfg = {"name": "on", "defaults": {"node": "g050"},
           "steps": [{"id": "s1", "tier": 2, "a": S3, "b": S3 + ",env=X=1", "opts": {"after": "777"}},
                     {"id": "s2", "tier": 2, "a": S3, "b": S3 + ",env=Y=1"}],
           "summary": {"outdir": "/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-fast/results/overnight", "alone": [], "loaded": []}}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    f = Chain()
    assert ab.main(["--campaign", str(p), "--out", str(tmp_path / "o")], remote=f, out_fn=lambda x: None) == 0
    subs = [c for c in f.cmds if "sbatch --parsable" in c]
    assert len(subs) == 3
    assert "--dependency=afterany:777" in subs[0] and "-w g050" in subs[0]
    assert "--dependency=afterany:100" in subs[1]
    assert "--dependency=afterany:101" in subs[2] and "gf-summary" in subs[2] and "PIN_SCRIPTS=" + "a" * 40 in subs[2] and "OUTDIR=/mnt/projects" in subs[2]
    m = json.load(open(tmp_path / "o" / "campaign.json"))
    assert m["steps"][0]["run"].endswith("/runs/on-s1-t2-" + m["created"]) or "/runs/on-s1-t2-" in m["steps"][0]["run"]
    assert m["summary_job"] == "102" and m["steps"][0]["rule_params"]["rate_gain"] == 0.09
