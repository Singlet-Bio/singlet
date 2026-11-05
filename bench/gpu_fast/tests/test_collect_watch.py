import json
import os
import time

import collect
import watch

COLS = ("job node bin bin_sha256 build_commit scripts_commit T point k slots runs failed_runs span_s span_rate steady_rate win_rate win_n win_s core_s_per_M "
        "cores_span cores_win lock_wait_s lock_wait_frac lock_held_s gpu_util host_peak_gb dev_peak_gb fallback_runs ident_alone ident_cpu other_jobs_max mean_wall_s").split()


def mkleg(res, d, runs=1, ident="1/1", fb=0, core="9.5"):
    D = res / d
    D.mkdir()
    row = dict.fromkeys(COLS, "NA")
    row.update(runs=str(runs), failed_runs="0", core_s_per_M=core, lock_held_s="34.5", lock_wait_s="0.0", mean_wall_s="150.0", dev_peak_gb="44.0",
               host_peak_gb="53", fallback_runs=str(fb), ident_cpu=ident, node="g051", span_rate="0.4")
    (D / "results.tsv").write_text("\t".join(COLS) + "\n" + "\t".join(row[c] for c in COLS) + "\n")
    (D / "fallback.txt").write_text("c0\t" + ("FALLBACK\tlevel 5 NO" if fb else "ok\t") + "\n")
    runs_ = [{"rc": 0, "worker": w, "start": w, "end": 100 + w, "pairs": 66e6, "stage_core_s": {"STAR": 100.0, "demux": 10.0}} for w in range(1)]
    (D / "point.json").write_text(json.dumps({"runs": runs_}))
    return D


def test_collect_builds_result(tmp_path):
    res = tmp_path
    (res / "params.json").write_text(json.dumps({"tier": 1, "name": "x", "job": "9", "scripts": "abc"}))
    (res / "arms.tsv").write_text("arm\tbin\tsha256\tcommit\tembedded\tenv\targs\nA\t/b/a\tsha\te42\tyes\tX=1\t\nB\t/b/b\tsha2\te42\tyes\t\t--no-snps\n")
    mkleg(res, "leg1A")
    mkleg(res, "leg2B", ident="0/1", fb=1)
    mkleg(res, "leg10A")
    out = collect.collect(str(res))
    assert [l["leg"] for l in out["legs"]] == [1, 2, 10]
    a, b = out["legs"][0], out["legs"][1]
    assert a["prov"]["env"] == "X=1" and b["prov"]["args"] == "--no-snps" and a["metrics"]["core_s_per_M"] == 9.5
    assert b["ident"] == {"checked": 1, "identical": 0, "detail": "leg2B/"} and b["fallback"]["runs"] == 1 and "level 5 NO" in b["fallback"]["lines"][0]
    assert a["stage_core_s"]["STAR"] == 100.0 and a["completed"]
    assert json.load(open(res / "result.json"))["tier"] == 1


def mkrun(leg, i, rc, lines, star="E11: Solo on the device\n", end=None):
    d = leg / f"c{i}"
    d.mkdir(exist_ok=True)
    (d / "run.log.ts").write_text("".join(f"1 {l}\n" for l in lines))
    (d / "star_Log.out").write_text(star)
    with open(leg / "runs.tsv", "a") as f:
        f.write(f"{i}\t0\ts\t0\t{end or time.time() - 60}\t{rc}\n")


def test_watch_stops_on_fallback_only(tmp_path):
    leg, scr = tmp_path / "res", tmp_path / "scr"
    leg.mkdir()
    mkrun(leg, 0, 0, ["[gpu_pileup] on: GPU 1 ms"])
    assert watch.main(str(leg), str(scr)) is None and not (scr / "STOP").exists()
    mkrun(leg, 1, 0, ["[gpu_pileup] on: x", "[engine_gpu] fallback: out of memory"])
    r = watch.main(str(leg), str(scr))
    assert r and "run c1" in r and (scr / "STOP").exists() and (leg / "STOPPED").read_text().startswith("fallback in run c1")


def test_watch_stops_on_failed_run_and_missing_level5(tmp_path):
    leg, scr = tmp_path / "res", tmp_path / "scr"
    leg.mkdir()
    mkrun(leg, 0, 1, ["[gpu_pileup] on: x"])
    assert "exit 1" in watch.main(str(leg), str(scr))
    leg2, scr2 = tmp_path / "res2", tmp_path / "scr2"
    leg2.mkdir()
    mkrun(leg2, 0, 0, ["[gpu_pileup] on: x"], star="nothing\n")
    assert "level 5 NO" in watch.main(str(leg2), str(scr2))


def test_watch_waits_for_the_grace_period(tmp_path):
    leg, scr = tmp_path / "res", tmp_path / "scr"
    leg.mkdir()
    mkrun(leg, 0, 1, [], end=time.time())
    assert watch.main(str(leg), str(scr)) is None
