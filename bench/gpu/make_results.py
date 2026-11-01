"""Assemble bench/gpu/results/{m1,m2,m10}.json from the raw files pulled from Clipper, and print the summary numbers.

    python bench/gpu/make_results.py          # reads bench/gpu/results/{m1_raw.jsonl,m1_knee.jsonl,m2_*.json,m10_runs.json}
"""

import glob
import json
import os
import statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(p):
    return json.load(open(os.path.join(R, p)))


def m1():
    rows = [json.loads(l) for f in ("m1_raw.jsonl", "m1_knee.jsonl") for l in open(os.path.join(R, f))
            if l.startswith("{") and os.path.exists(os.path.join(R, f))]
    dev = next(r for r in rows if r["kind"] == "device")
    out = {"device": dev, "gather": [r for r in rows if r["kind"] == "gather"], "chase": [r for r in rows if r["kind"] == "chase"],
           "gather_hot": [r for r in rows if r["kind"] == "gather_hot"]}
    g32 = {(r["alloc"], r["gb"]): r["median_gacc_s"] for r in out["gather"] if r["bytes"] == 32}
    out["summary"] = {
        "gather32_median_gsectors_s": {f"{k[0]}_{k[1]:g}GB": v for k, v in sorted(g32.items())},
        "rs_clean_substitute_g_s": min(g32[("malloc", 32.0)], g32[("malloc", 48.0)]),
        "pass_rule": "gather >= 15 G sectors/s at 32 and 48 GB, 32 B; kill the base scenario if < 8 G/s at 48 GB",
        "pass": g32[("malloc", 32.0)] >= 15 and g32[("malloc", 48.0)] >= 15,
        "killed": g32[("malloc", 48.0)] < 8,
    }
    return out


def m2():
    out = {"samples": {}, "batches": {}}
    for f in sorted(glob.glob(os.path.join(R, "m2_*.json"))):
        name = os.path.basename(f)[3:-5]
        d = json.load(open(f))
        if "_cross" in d:
            paths = [k for k in d if k != "_cross"]
            out["batches"][name] = {"members": {p: d[p] for p in paths}, "cross": d["_cross"]}
        else:
            out["samples"][name] = next(iter(d.values()))
    return out


def m10():
    runs = load("m10_runs.json")
    sel = [x for x in runs if x.get("core_s") and x["cell_caller"] == "called_v2" and x["stages"]["export_total"]]
    rows = []
    for x in sel:
        c, w, t, s = x["core_s"], x["window_wall_s"], x["timers"], x["stages"]
        U = x["reads"] / 1e6
        W_up = c["export"] + c["nonhost_residual"]
        rows.append({
            "run": x["dir"].split("singlet/")[1], "reads_M": round(U, 1), "threads": x["threads"],
            "wall_s": {"export_total": s["export_total"], "nonhost_residual_window": w["nonhost_residual"],
                       "cell_calling": t.get("cell_calling"), "ambient": t.get("ambient"), "doublet": t.get("doublet"),
                       "sex_calling": t.get("sex_calling"), "ancestry": t.get("ancestry"), "saturation_curve": t.get("saturation_curve"),
                       "demux_autoK_selected": x["demux"]["auto_k_selected_s"], "polya": t.get("polya")},
            "core_s_measured": {"export": c["export"], "nonhost_residual": c["nonhost_residual"],
                                "W_upper = export + nonhost_residual": round(W_up, 1),
                                "removed_by_gpu (index+cb_refresh+pileup)": round(c["index"] + c["cb_refresh"] + c["pileup"], 1)},
            "W_core_s_per_M_pairs_upper": round(W_up / U, 2), "W_core_s_per_M_pairs_export_only": round(c["export"] / U, 2),
        })
    # one row per sample (r1 only), for the summary
    uniq = {}
    for r in rows:
        key = r["run"].split("/")[2] if r["run"].startswith("bam-stream") else r["run"]
        if "deep-perf" in r["run"] or key in uniq:
            continue
        uniq[key] = r
    ps = list(uniq.values())
    ratio_up = [r["W_core_s_per_M_pairs_upper"] for r in ps]
    ratio_ex = [r["W_core_s_per_M_pairs_export_only"] for r in ps]
    return {
        "method": "stage timers from run.log + cgroup CPU seconds (monitor.tsv) in post-mapping windows; runs at --threads 4 on Clipper CPU nodes",
        "samples": ps, "all_monitored_runs": rows,
        "summary": {
            "W_upper_core_s_66_70M": [r["W_core_s_per_M_pairs_upper"] * r["reads_M"] for r in ps if r["reads_M"] < 72],
            "W_per_M_pairs_upper_median": st.median(ratio_up), "W_per_M_pairs_export_only_median": st.median(ratio_ex),
            "host_rate_M_pairs_s_per_gpu_26_cores_upper": round(26 / st.median(ratio_up), 1),
            "host_rate_M_pairs_s_per_gpu_26_cores_export_only": round(26 / st.median(ratio_ex), 1),
            "pass_rule": "W <= 120 core-s on a 66 M sample",
        },
        "callers_without_timers": ["non-host EM (no timer line at all)", "called_ed (no run log contains it; cell_caller=called_v2 has a timer)",
                                   "demux outside the auto-K fits (VB on called cells, back-assignment, VCF writes)",
                                   "export writers: pz_writer, splice_psi, sj_features, mt/mt_event_caller (G6 only), ase, vdj, cell_qc, metrics_summary, rrna"],
        "n_runs_parsed": len(runs),
    }


if __name__ == "__main__":
    for name, f in (("m1", m1), ("m2", m2), ("m10", m10)):
        d = f()
        json.dump(d, open(os.path.join(R, f"{name}.json"), "w"), indent=1)
        print(name, "->", f"{R}/{name}.json")
    print(json.dumps(m1()["summary"], indent=1))
    print(json.dumps(m10()["summary"], indent=1))
    for r in m10()["samples"]:
        print(r["run"], r["reads_M"], r["core_s_measured"], r["wall_s"]["export_total"], r["wall_s"]["demux_autoK_selected"])
