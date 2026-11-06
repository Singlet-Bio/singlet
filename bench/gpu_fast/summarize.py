#!/usr/bin/env python3
"""Final summary of a campaign, run on Clipper by jobs/summarize.sbatch inside in_runtime. Needs nothing from the laptop.

Usage: summarize.py CAMPAIGN_MANIFEST OUTDIR
Reads the manifest of a campaign (written by `ab.py --campaign` and copied to Clipper) and the result.json of each step in its run directory.
Writes OUTDIR/summary.md, OUTDIR/load_fit.json and OUTDIR/steps/<id>/{table.md,verdict.json}.

summary.md has
  1. the steps with their jobs and whether they left a result
  2. for every tier 2 step the verdicts with all legs and without the legs that shared the node, then the table
  3. co-tenants per leg (cotenants.log of each leg)
  4. demand vector and node rate limits of the tier 1 runs named in the manifest (summary.limits_for)
  5. the refit of the load factors and the efficiency on clean legs (summary.alone, summary.loaded, and the A legs of the steps)
Tier 0 numbers are a ranking screen only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abcore  # noqa: E402
import cotenants  # noqa: E402
import demand  # noqa: E402
import model  # noqa: E402


def step_result(step):
    p = os.path.join(step["run"], "res", "result.json")
    return demand.load_result(step["run"]) if os.path.exists(p) else None


def verdict_of(step, res, legs=None):
    rp, intended = step.get("rule_params") or {}, bool(step.get("intended"))
    r = res if legs is None else abcore.clean_result(res)
    if step["tier"] == 1:
        return abcore.tier1_verdict(r, rp, intended, step.get("rule_kind", "gain"))
    return abcore.tier2_verdict(r, rp, intended)


def main(manifest_path, outdir):
    M = json.load(open(manifest_path))
    cfg = M.get("summary") or {}
    os.makedirs(os.path.join(outdir, "steps"), exist_ok=True)
    L = [f"# Overnight summary of campaign {M.get('camp')}", "", "Tier 0 numbers are a ranking screen only. Read ratios between arms.", "", "## Steps", "",
         "| step | tier | job | result |", "|---|---|---|---|"]
    results = {}
    for st in M["steps"]:
        res = step_result(st)
        results[st["id"]] = res
        L.append(f"| {st['id']} | {st['tier']} | {st.get('job')} | {'result.json found' if res else 'no result.json (job did not run or failed)'} |")
    L += ["", "## Verdicts", ""]
    for st in M["steps"]:
        res = results[st["id"]]
        if not res:
            L += [f"### {st['id']}", "", "No result. INCONCLUSIVE (J1).", ""]
            continue
        sdir = os.path.join(outdir, "steps", st["id"])
        os.makedirs(sdir, exist_ok=True)
        v = verdict_of(st, res)
        cont = abcore.contaminated_legs(res)
        vc = verdict_of(st, res, legs="clean") if cont else None
        table = abcore.render(res, v, st.get("notes", []))
        open(os.path.join(sdir, "table.md"), "w").write(table + "\n")
        vj = {"verdict": v["verdict"], "rule": v["rule"], "why": v["why"], "gain": v.get("gain"), "job": st.get("job")}
        if vc:
            vj["without_contaminated"] = {"verdict": vc["verdict"], "rule": vc["rule"], "why": vc["why"]}
        json.dump(vj, open(os.path.join(sdir, "verdict.json"), "w"), indent=1)
        L += [f"### {st['id']} (job {st.get('job')})", "", f"- All legs: **{v['verdict']}** (rule {v['rule']}) {v['why']}"]
        if vc:
            L.append(f"- Without the co-tenant legs ({', '.join('leg %s (%s)' % (l['leg'], l['arm']) for l in cont)}): **{vc['verdict']}** (rule {vc['rule']}) {vc['why']}")
        else:
            L.append("- No leg shared the node with another job.")
        L += ["", table, ""]
    L += ["## Co-tenants per leg", ""]
    for st in M["steps"]:
        if not results[st["id"]]:
            continue
        L.append(f"### {st['id']}")
        L.append("")
        for name, win, oth in cotenants.report(st["run"], own=str(st.get("job")), legs_root=os.path.join(st["run"], "res")):
            if oth is None:
                L.append(f"- {name} {win}: no cotenants.log")
            elif not oth:
                L.append(f"- {name} {win}: clean")
            else:
                L.append(f"- {name} {win}: CO-TENANT " + "; ".join(f"job {j} {e['name']} {e['cpus']} cpus seen {e['first'][11:19]} to {e['last'][11:19]}" for j, e in oth.items()))
        L.append("")
    L += ["## Demand vector and node rate limits", ""]
    for d in cfg.get("limits_for", []):
        try:
            res = demand.load_result(d)
            v = demand.tier1_demand_verdict(res)
            L += [f"### {os.path.basename(d.rstrip('/'))}", "", demand.render_demand(res, v), "", f"Demand rule: **{v['verdict']}** ({v['rule']}) {v['why']}", ""]
        except Exception as e:   # a missing directory must not stop the rest of the summary
            L += [f"### {d}", "", f"not read: {e}", ""]
    L += ["## Load factors and efficiency, refit on clean legs", ""]
    loaded = list(cfg.get("loaded", []))
    for st in M["steps"]:
        if st["tier"] == 2 and results[st["id"]] and cfg.get("loaded_from_steps_arm"):
            loaded.append(f"{st['run']}:{cfg['loaded_from_steps_arm']}")
    try:
        f = model.fit_load(cfg.get("alone", []), loaded, 28.0, 2, 93.6, cfg.get("exclude_legs", []))
        json.dump(f, open(os.path.join(outdir, "load_fit.json"), "w"), indent=1)
        old = {}
        try:
            old = json.load(open(demand.LOAD_FIT_FILE))
        except (OSError, ValueError):
            pass
        L += [f"- alone: {f['alone']['legs']} legs, core-s/M {f['alone']['core_s_per_M']:.3f}, hold {f['alone']['hold_s']:.2f} s",
              f"- loaded: {f['loaded']['legs']} legs, core-s/M {f['loaded']['core_s_per_M']:.3f}, hold {f['loaded']['lock_held_s']:.2f} s",
              f"- load factor core {f['load_core']:.3f}, hold {f['load_hold']:.3f}"
              + (f" (committed fit: core {old['load_core']:.3f}, hold {old['load_hold']:.3f})" if old.get("load_core") else ""),
              f"- binding limit with load factors {f['binding_limit']:.3f} ({f['binding']}). Efficiency {f['efficiency']:.3f}"
              + (f" (committed fit {old['efficiency']:.3f})" if old.get("efficiency") else "")
              + (f", leg to leg CV {100 * f['efficiency_cv']:.1f}%" if f.get("efficiency_cv") else ""),
              "- efficiency per loaded leg: " + " ".join(f"{x:.3f}" for x in f["efficiency_per_leg"]),
              "- Legs with a fallback, a failed run or another job on the node are left out.", ""]
    except SystemExit as e:
        L += [f"not fitted: {e}", ""]
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(outdir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
