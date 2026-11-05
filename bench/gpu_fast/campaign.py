#!/usr/bin/env python3
"""Self-running campaigns of the fast lane. Used through ab.py.

  python bench/gpu_fast/ab.py --campaign FILE.json [--dry-run]
  python bench/gpu_fast/ab.py --collect bench/gpu_fast/out/campaign-<name>-<time>

The campaign file lists steps. Each step is an A/B pair (tier 1) or a batch check (tier 2). All steps are submitted in one session as one
Slurm chain. A step starts after the previous step ends, in any end state (afterany), or after the steps named in "after". Each job writes its
own result.json on Clipper. Nothing needs the laptop or the bridge between the steps. Pushing to bench/gpu-fast later does not matter, because each job
uses the scripts commit it was submitted with (PIN_SCRIPTS). --collect reads the results afterwards and prints the verdicts.

File format.
{
  "name": "e20-stack",
  "defaults": {"node": "g051", "T": 7, "slots": 2},               # ab.py options without the leading dashes. true means the flag alone.
  "steps": [
    {"id": "t1-e20w", "tier": 1, "a": "bin=/.../s3,commit=e42c0636", "b": "bin=/.../e20w,commit=aeef7a5e"},
    {"id": "batch", "tier": 2, "order": "ABCCBA",
     "arms": {"A": "bin=...", "B": "bin=...,env=X=1", "C": "bin=...,env=Y=1"},
     "predict": {"B": ["t1-x"], "C": ["t1-y"]},                    # tier 1 step ids that give the predicted change of that arm. Several ids stack.
     "after": ["t1-e20w"],
     "opts": {"runs": 24, "intended-output-change": true}}
  ]
}
A step that names no "after" follows the step before it. "after": [] starts it at once (the H100 cap of the account still applies).
"""
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abcore  # noqa: E402


def flags_from(d):
    out = []
    for k, v in (d or {}).items():
        if v is None or v is False:
            continue
        flag = "--" + k.lstrip("-")
        if v is True:
            out.append(flag)
        elif isinstance(v, list):
            for x in v:
                out += [flag, str(x)]
        else:
            out += [flag, str(v)]
    return out


def step_argv(cfg_name, defaults, step):
    """The ab.py command line of one step."""
    tier = int(step["tier"])
    arms = dict(step.get("arms") or {})
    if "a" in step:
        arms["A"] = step["a"]
    if "b" in step:
        arms["B"] = step["b"]
    if "A" not in arms or "B" not in arms:
        raise ValueError(f"step {step.get('id')}: arms A and B are needed")
    argv = ["--tier", str(tier), "--name", f"{cfg_name}-{step['id']}", "--a", arms["A"], "--b", arms["B"]]
    for x in sorted(k for k in arms if k not in ("A", "B")):
        argv += ["--arm", f"{x}={arms[x]}"]
    if step.get("order"):
        argv += ["--order", step["order"]]
    opts = dict(defaults or {})
    opts.update(step.get("opts") or {})
    return argv + flags_from(opts)


def load_config(path):
    cfg = json.load(open(path))
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,30}", cfg.get("name", "")):
        raise ValueError("campaign name: letters digits . _ - and at most 30 characters")
    ids = [s.get("id") for s in cfg.get("steps", [])]
    if not ids or len(set(ids)) != len(ids) or not all(re.fullmatch(r"[A-Za-z0-9._-]{1,20}", str(i)) for i in ids):
        raise ValueError("steps need distinct ids of letters digits . _ - (at most 20 characters)")
    for s in cfg["steps"]:
        for dep in s.get("after", []):
            if dep not in ids or ids.index(dep) >= ids.index(s["id"]):
                raise ValueError(f"step {s['id']}: 'after' names {dep}, which is not an earlier step")
        for arm, lst in (s.get("predict") or {}).items():
            for i in lst:
                if i not in ids or ids.index(i) >= ids.index(s["id"]):
                    raise ValueError(f"step {s['id']}: 'predict' names {i}, which is not an earlier step")
    return cfg


def run(ab, a, remote, out_fn=print):
    """Submit every step as one Slurm chain."""
    try:
        cfg = load_config(a.campaign)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    camp = f"{cfg['name']}-{stamp}"
    camp_dir = a.out or os.path.join(HERE, "out", f"campaign-{camp}")
    head, dirty, rem = ab.local_state()
    plans = []
    try:
        for st in cfg["steps"]:
            sa = ab.parser().parse_args(step_argv(cfg["name"], cfg.get("defaults"), st))
            plan = ab.make_plan(sa, head, stamp)
            plan["state"].update(id=st["id"], predict=st.get("predict") or {}, after=st.get("after"))
            plans.append((st, plan))
    except (ab.Fatal, abcore.SpecError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    problems = ab.head_problems(head, dirty, rem)
    if a.dry_run:
        prev = None
        for st, plan in plans:
            after = st.get("after")
            deps = [f"<job of {i}>" for i in (after if after is not None else ([prev] if prev else []))]
            print(f"\n######## step {st['id']} (tier {plan['a'].tier}) after {deps or 'nothing'}")
            ab.print_plan(plan, head, problems, deps or None)
            prev = st["id"]
        return 0
    if problems:
        print("error: " + "; ".join(problems), file=sys.stderr)
        return 2
    remote = remote or ab.Remote()
    if not remote.status():
        print("error: the Clipper bridge is not connected. This tool does not start it.", file=sys.stderr)
        return 2
    os.makedirs(camp_dir, exist_ok=True)
    manifest = {"name": cfg["name"], "camp": camp, "created": stamp, "scripts": head, "config": cfg, "steps": []}
    jobs = {}
    prev = None
    for st, plan in plans:
        after = st.get("after")
        dep_ids = after if after is not None else ([prev] if prev else [])
        deps = [jobs[i] for i in dep_ids if jobs.get(i)]
        out = os.path.join(camp_dir, "steps", st["id"])
        state = dict(plan["state"], dir=os.path.relpath(out, camp_dir).replace(os.sep, "/"))
        try:
            os.makedirs(out, exist_ok=True)
            ab.save_state(out, state)
            jobs[st["id"]] = state["job"] = ab.submit_plan(remote, plan, out, deps or None)
        except (ab.BridgeDown, ab.Fatal) as e:
            manifest["steps"].append({"id": st["id"], "job": None, "dir": state["dir"]})
            json.dump(manifest, open(os.path.join(camp_dir, "campaign.json"), "w"), indent=1)
            print(f"stopped at step {st['id']}: {e}. Steps before it are queued. Directory {camp_dir}", file=sys.stderr)
            return 3
        state["submitted"] = True
        ab.save_state(out, state)
        manifest["steps"].append({"id": st["id"], "job": state["job"], "dir": state["dir"], "tier": plan["a"].tier, "after": deps})
        out_fn(f"step {st['id']}: job {state['job']} after {deps or 'nothing'}")
        prev = st["id"]
    mpath = os.path.join(camp_dir, "campaign.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    try:
        rdir = f"{ab.W}/jobs/gpu-fast/campaigns/{camp}"
        remote.run(f"mkdir -p {rdir}")
        remote.put(mpath, f"{rdir}/campaign.json")
    except (ab.BridgeDown, ab.Fatal):
        pass   # the chain is queued. The manifest is on the laptop
    out_fn(f"\nCampaign {camp} queued. Nothing more is needed from the laptop. Read it later with:\n  python bench/gpu_fast/ab.py --collect {camp_dir}")
    return 0


def summary_row(step_dir, state):
    """One line per step for the collect table, from verdict.json."""
    vp = os.path.join(step_dir, "verdict.json")
    if not os.path.exists(vp):
        return None
    v = json.load(open(vp))
    pct = abcore.pct
    if v.get("batch"):
        return {"verdict": " / ".join(f"{x}: {r['verdict']}" for x, r in v["arms"].items()),
                "numbers": "; ".join(f"{x} predicted {pct(r.get('predicted_change'))} measured progress {pct(r['gain']['progress_rate'])} core-s/M {pct(r['gain']['core_s_per_M'])}"
                                     for x, r in v["arms"].items())}
    g = v.get("gain") or {}
    if v.get("demand"):
        d = v["demand"]
        nums = f"predicted node rate {pct(d['predicted_node_change'])} (A binds on {d['limits_A']['binding']}), core-s/M {pct(g.get('core_s_per_M'))}, lock held {pct(g.get('lock_held_s'))}"
    elif v.get("tier") == 2:
        nums = f"progress {pct(g.get('progress_rate'))}, core-s/M {pct(g.get('core_s_per_M'))}, lock held {pct(g.get('lock_held_s'))}"
    else:
        nums = f"core-s/M {pct(g.get('core_s_per_M'))}, lock held {pct(g.get('lock_held_s'))}"
    return {"verdict": f"{v['verdict']} ({v['rule']})", "numbers": nums}


def collect(ab, a, remote, out_fn=print):
    """Read the results of a campaign from Clipper, write each step's table and verdict.json, print one summary."""
    path = a.collect
    camp_dir = os.path.dirname(path) if path.endswith(".json") else path
    mpath = path if path.endswith(".json") else os.path.join(path, "campaign.json")
    try:
        manifest = json.load(open(mpath))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not remote.status():
        print("error: the Clipper bridge is not connected. This tool does not start it. Steps already collected are in their directories.", file=sys.stderr)
        return 2
    rows = []
    unknown = False
    for st in manifest["steps"]:
        sdir = os.path.join(camp_dir, st["dir"])
        try:
            state = json.load(open(os.path.join(sdir, "state.json")))
        except OSError:
            rows.append((st["id"], "-", "no state file", "", "", ""))
            continue
        job = state.get("job")
        if not job:
            rows.append((st["id"], state.get("tier"), "NOT SUBMITTED", "", "", ""))
            continue
        try:
            rc, so, _ = remote.run(f"squeue -j {job} -h -o '%T' 2>/dev/null; sacct -j {job} -X -n -o State 2>/dev/null")
            js = ab.parse_state(so)
            if js in ab.TERMINAL:
                res, tail = ab.fetch_result(remote, state["run"], state["jobname"], job)
                if res is None:
                    v = {"verdict": "INCONCLUSIVE", "rule": "J1", "why": f"job {job} ended as {js} and left no result.json", "tier": state["tier"], "params": {}, "log_tail": tail}
                    json.dump(v, open(os.path.join(sdir, "verdict.json"), "w"), indent=1)
                else:
                    ab.report(sdir, state, res, js, lambda x: None, camp_dir)
        except ab.BridgeDown:
            js, unknown = "UNKNOWN", True
        sr = summary_row(sdir, state)
        rows.append((st["id"], state.get("tier"), sr["verdict"] if sr else "not finished", job, js or "?", sr["numbers"] if sr else ""))
    out_fn(f"Campaign {manifest.get('camp')}")
    out_fn("")
    out_fn("| step | tier | verdict | job | job state | key numbers |")
    out_fn("|---|---|---|---|---|---|")
    for r in rows:
        out_fn("| " + " | ".join(str(x) for x in r) + " |")
    out_fn("\nEach step directory has table.md, verdict.json and result.json.")
    if unknown:
        print("the bridge dropped while collecting. Run --collect again later.", file=sys.stderr)
        return 3
    return 0
