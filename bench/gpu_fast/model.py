#!/usr/bin/env python3
"""Tier 0: a queue model of one node. No GPU and no cluster needed.

What it models. K runs in flight on a node with a number of device slots and a number of cores.
A run is a list of segments taken from one run alone (a k=1 point of the throughput harness):
  host segment   needs the cores it used alone (core-s over wall) and no slot
  hold segment   needs a slot for a device phase (prefill, handoff, solo, pileup, nonhost)
Rules.
  Slots   S servers. A waiting run takes a slot when one frees, first come first served (or random order, as flock).
  Cores   all running segments share the cores. When the demand is above the cores every segment slows by the same factor.
  Sharing A hold progresses at 1 / (1 + gamma * (n - 1)) with n holders at that moment. gamma is the one fitted number.
          It stands for two holders slowing each other on the device. gamma 0 means no slowing.
The queue of runs is the one of tp.sbatch: K workers take the next run when they finish.

Commands.
  model.py extract POINT_JSON [--samples SAMPLES_JSONL] [--results-tsv FILE] --out OUT.json
      Compacts a point of the throughput harness. With --samples of a k=1 point it also builds the run template.
  model.py predict PROFILE.json --k 8 --slots 2 --runs 40 [--cpus 28] [--gamma 0.37|auto] [--scale prefill_hold=0.5] [--scale demux_core_s=0]
      Rate of the node at K and slots, with what-if edits.
  model.py validate [--data model_data] [--markdown]
      Predicts every point of model_data/manifest.json and prints the error against the measured windowed rate and span rate.

What-if names for --scale (a factor, 0 removes the part).
  <phase>_hold    wall time of that phase's device holds. Phases: prefill handoff solo pileup nonhost (those the template has).
                  The CPU work inside the hold stays.
  <stage>_core_s  CPU work of that stage. Host segments scale in wall too (same parallelism). Hold segments keep their wall.
                  Stages: start scan refs STAR stream handoff pileup pileupfin nonhost export demux.
  <stage>_wall    wall time of that stage at the same CPU work (more cores per second).
  hold            all holds.  host_core_s  all host segments.  host_wall  wall of all host segments.
A segment whose wall becomes 0 drops its CPU work.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abcore  # noqa: E402

PHASES = ("prefill", "handoff", "solo", "pileup", "nonhost")
STAGES = ("start", "scan", "refs", "STAR", "stream", "handoff", "pileup", "pileupfin", "nonhost", "export", "demux")
EPS = 1e-9
GAMMA_E18 = 0.374   # E18 report: a sample holds a slot 47.4 s at k = 8 with two slots and 34.5 s with one (binary 44a6e9fa)


# --- reading a point ------------------------------------------------------------------------------------------------------------------------
def measured(runs, k):
    """Span rate, windowed rate and steady rate with the formulas of bench/gpu_batch1/tp_point.py. Rates in M pairs per second."""
    pairs = {r["item"]: (r.get("pairs") or 0) / 1e6 for r in runs}
    first, last = min(r["start"] for r in runs), max(r["end"] for r in runs)
    span = last - first
    out = {"span_s": span, "span_rate": sum(pairs.values()) / span if span > 0 else None, "win_rate": None, "win_n": None, "win_s": None, "steady_rate": None}
    ok = sorted(r["end"] for r in runs if r["rc"] == 0)
    if len(ok) >= k:
        ws, we = ok[k - 1], max(r["start"] for r in runs)
        out["win_s"] = we - ws
        if we > ws:
            inside = [r for r in runs if r["rc"] == 0 and ws < r["end"] <= we + 1e-6]
            out["win_n"] = len(inside)
            out["win_rate"] = sum(pairs[r["item"]] for r in inside) / (we - ws) if inside else None
        else:
            out["win_n"] = 0
    out["progress_rate"] = abcore.progress_rate(runs, k)
    ends = sorted(r["end"] for r in runs)
    later = [r for r in runs if r["start"] >= ends[0] - 1.0]
    if later and len(runs) > k:
        mw = sum(r["end"] - r["start"] for r in later) / len(later)
        out["steady_rate"] = k * sum(pairs[r["item"]] for r in later) / len(later) / mw
    return out


def build_template(run, samples, copy="c0"):
    """Segments of one run alone, from the sampler of the k=1 point.
    Hold intervals come from the lock holder per tick. Phases (prefill, handoff, solo, pileup, nonhost) take their logged hold times in
    order along the held time. The run is cut at the stage boundaries as well. CPU per segment is the run's own CPU time from the sampler."""
    t0, t1 = run["start"], run["end"]
    pts = [(t0, 0.0)] + [(s["t"], s["copies"][copy]["cpu"]) for s in samples if copy in s.get("copies", {})] + [(t1, None)]
    pts[-1] = (t1, max(c for _, c in pts[:-1]))
    xs = [p[0] for p in pts]

    def cpu(t):
        t = min(max(t, xs[0]), xs[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if xs[m] <= t:
                lo = m
            else:
                hi = m
        a, b = pts[lo], pts[hi]
        return a[1] if b[0] == a[0] else a[1] + (b[1] - a[1]) * (t - a[0]) / (b[0] - a[0])

    ticks = [(s["t"], any(e.get("h") == copy for e in s.get("locks", {}).values())) for s in samples if t0 - 1 <= s["t"] <= t1 + 1]
    iv = []
    for i, (t, h) in enumerate(ticks):
        if not h:
            continue
        a = (ticks[i - 1][0] + t) / 2 if i > 0 else t
        b = (t + ticks[i + 1][0]) / 2 if i + 1 < len(ticks) else t
        if iv and a <= iv[-1][1] + 1e-6:
            iv[-1][1] = b
        else:
            iv.append([a, b])
    H = sum(b - a for a, b in iv)
    L = run.get("locks", {})
    ph = [(p, L[p]["held"]) for p in PHASES if p in L and L[p].get("held") is not None]
    cuts = {t0, t1}
    for a, b in iv:
        cuts.update((a, b))
    phase_at = []   # (a, b, phase) along time
    if iv and ph:
        tot = sum(h for _, h in ph)
        pos = 0.0
        axis = []   # (axis_start, axis_end, time_start)
        for a, b in iv:
            axis.append((pos, pos + (b - a), a))
            pos += b - a
        cum = 0.0
        for p, h in ph:
            lo, hi = cum * H / tot, (cum + h) * H / tot
            cum += h
            for s0, s1, ta in axis:
                x0, x1 = max(lo, s0), min(hi, s1)
                if x1 > x0 + EPS:
                    phase_at.append((ta + (x0 - s0), ta + (x1 - s0), p))
        for a, b, _ in phase_at:
            cuts.update((a, b))
    sb, acc = [], t0
    for name, d in (run.get("stages") or {}).items():
        sb.append((acc, acc + d, name))
        acc += d
    for a, b, _ in sb:
        cuts.update((a, b))
    cuts = sorted(c for c in cuts if t0 - 1e-9 <= c <= t1 + 1e-9)
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-6:
            continue
        mid = (a + b) / 2
        phase = next((p for x, y, p in phase_at if x <= mid < y), None)
        stage = next((n for x, y, n in sb if x <= mid < y), sb[-1][2] if sb else "")
        segs.append({"kind": "hold" if phase else "host", "phase": phase or "", "stage": stage, "wall": b - a, "core_s": max(cpu(b) - cpu(a), 0.0)})
    return segs


def extract(point_path, samples_path=None, results_tsv=None):
    P = json.load(open(point_path))
    runs = []
    for r in P["runs"]:
        runs.append({k: r.get(k) for k in ("item", "worker", "start", "end", "rc", "pairs", "wall", "core_s", "locks", "stages", "stage_core_s",
                                           "sampler_hold_s", "sampler_wait_s", "log_wait_s", "log_held_s")})
    k = len({r["worker"] for r in runs})
    out = {"name": os.path.basename(point_path).replace(".point.json", ""), "tag": P.get("tag"), "node": P.get("node"), "env": P.get("env"), "k": k,
           "T": P.get("T"), "slots": P.get("slots"), "pairs_M_per_run": (sum((r["pairs"] or 0) for r in runs) / len(runs)) / 1e6,
           "cgroup_core_s_per_M": P.get("cgroup_core_s_per_M"), "lock_held_mean": P.get("lock_held_mean"), "lock_wait_mean": P.get("lock_wait_mean"),
           "runs": runs, "measured": measured(runs, k), "results_tsv": None}
    if results_tsv and os.path.exists(results_tsv):
        rows = [l.rstrip("\n").split("\t") for l in open(results_tsv)]
        out["results_tsv"] = dict(zip(rows[0], rows[1]))
    if samples_path and k == 1:
        samples = [json.loads(l) for l in open(samples_path) if l.strip()]
        out["template"] = build_template(runs[0], samples)
    return out


# --- what-if edits -------------------------------------------------------------------------------------------------------------------------
def valid_scale_names(template):
    phases = sorted({s["phase"] for s in template if s["phase"]})
    stages = sorted({s["stage"] for s in template if s["stage"]})
    names = ["hold", "host_core_s", "host_wall"] + [f"{p}_hold" for p in phases] + [f"{s}_core_s" for s in stages] + [f"{s}_wall" for s in stages]
    return names


def parse_scales(items, template):
    out = []
    names = valid_scale_names(template)
    for it in items or []:
        m = re.fullmatch(r"([A-Za-z_]+)=([0-9.]+)", it)
        if not m or m.group(1) not in names:
            raise SystemExit(f"bad --scale '{it}'. Names: {' '.join(names)}")
        out.append((m.group(1), float(m.group(2))))
    return out


def apply_scales(template, scales):
    out = []
    for s in template:
        fw = fc = 1.0
        for name, f in scales:
            host = s["kind"] == "host"
            if name == "hold" and not host:
                fw *= f
            elif name == "host_core_s" and host:
                fc *= f
                fw *= f
            elif name == "host_wall" and host:
                fw *= f
            elif name.endswith("_hold") and not host and s["phase"] == name[:-5]:
                fw *= f
            elif name.endswith("_core_s") and s["stage"] == name[:-7]:
                fc *= f
                fw *= f if host else 1.0
            elif name.endswith("_wall") and s["stage"] == name[:-5]:
                fw *= f
        wall, core = s["wall"] * fw, s["core_s"] * fc
        if wall <= EPS:
            continue
        out.append({**s, "wall": wall, "core_s": core})
    return out


# --- the simulation ---------------------------------------------------------------------------------------------------------------------------
def simulate(template, k, slots, nruns, cpus=28.0, gamma=0.0, order="fifo", seed=1):
    rnd = random.Random(seed)
    segs = [s for s in template if s["wall"] > EPS]
    dem = [s["core_s"] / s["wall"] for s in segs]
    W = [{"item": None, "seg": 0, "rem": 0.0, "state": "idle", "start": 0.0, "wait": 0.0, "hold": 0.0, "tq": 0.0} for _ in range(k)]
    t = 0.0
    nxt = 0
    done = []
    queue = []
    holders = set()
    cpu_s = 0.0
    waits = []

    def begin(w):
        nonlocal nxt
        if nxt >= nruns:
            W[w]["state"] = "idle"
            return
        W[w].update(item=nxt, seg=-1, start=t, wait=0.0, hold=0.0)
        nxt += 1
        advance(w)

    def advance(w):
        x = W[w]
        x["seg"] += 1
        if x["seg"] >= len(segs):
            done.append((x["item"], x["start"], t, x["wait"], x["hold"]))
            begin(w)
            return
        s = segs[x["seg"]]
        x["rem"] = s["wall"]
        if s["kind"] == "hold":
            if len(holders) < slots:
                holders.add(w)
                x["state"] = "hold"
            else:
                x["state"] = "wait"
                x["tq"] = t
                queue.append(w)
        else:
            x["state"] = "host"

    for w in range(k):
        begin(w)
    guard = 0
    while any(x["state"] != "idle" for x in W):
        guard += 1
        if guard > 5_000_000:
            raise RuntimeError("simulation did not end")
        run = [w for w in range(k) if W[w]["state"] in ("host", "hold")]
        demand = sum(dem[W[w]["seg"]] for w in run)
        cs = min(1.0, cpus / demand) if demand > EPS else 1.0
        n = len(holders)
        dv = 1.0 / (1.0 + gamma * (n - 1)) if n >= 1 else 1.0
        rate = {w: cs * (dv if W[w]["state"] == "hold" else 1.0) for w in run}
        dt = min(W[w]["rem"] / rate[w] for w in run)
        for w in run:
            W[w]["rem"] -= dt * rate[w]
            if W[w]["state"] == "hold":
                W[w]["hold"] += dt
        cpu_s += dt * min(demand, cpus)
        t += dt
        for w in [w for w in run if W[w]["rem"] <= 1e-7]:
            if W[w]["state"] == "hold":
                holders.discard(w)
                if queue:
                    j = 0 if order == "fifo" else rnd.randrange(len(queue))
                    w2 = queue.pop(j)
                    W[w2]["wait"] += t - W[w2]["tq"]
                    waits.append(t - W[w2]["tq"])
                    holders.add(w2)
                    W[w2]["state"] = "hold"
            advance(w)
    pairs = None
    return {"done": done, "span": t, "cpu_s": cpu_s}


def rates(done, k, pairs_M):
    ends = sorted(d[2] for d in done)
    starts = [d[1] for d in done]
    span = max(ends) - min(starts)
    out = {"span_s": span, "span_rate": len(done) * pairs_M / span, "win_rate": None, "win_n": None, "win_s": None}
    if len(ends) >= k:
        ws, we = ends[k - 1], max(starts)
        out["win_s"] = we - ws
        if we > ws:
            inside = [d for d in done if ws < d[2] <= we + 1e-6]
            out["win_n"] = len(inside)
            out["win_rate"] = len(inside) * pairs_M / (we - ws)
    out["progress_rate"] = abcore.progress_rate([{"start": d[1], "end": d[2], "pairs": pairs_M * 1e6, "rc": 0} for d in done], k)
    out["mean_wall_s"] = sum(d[2] - d[1] for d in done) / len(done)
    out["mean_wait_s"] = sum(d[3] for d in done) / len(done)
    out["mean_hold_s"] = sum(d[4] for d in done) / len(done)
    return out


def predict(profile, k, slots, nruns, cpus=28.0, gamma=0.0, scales=(), order="fifo", seeds=1):
    tpl = apply_scales(profile["template"], scales)
    res = [rates(simulate(tpl, k, slots, nruns, cpus, gamma, order, s)["done"], k, profile["pairs_M_per_run"]) for s in range(1, seeds + 1)]
    out = {}
    for key in res[0]:
        v = [r[key] for r in res if r[key] is not None]
        out[key] = sum(v) / len(v) if v else None
    return out


def caps(core_s_per_M, lock_held_s, cpus, slots, pairs_M):
    """Sustained rate limits in M pairs per second. Core cap: allocated cores over core-s per M. Lock cap: slots times pairs per run over lock held
    seconds per run. The smaller one binds."""
    cc = cpus / core_s_per_M if core_s_per_M else None
    lc = slots * pairs_M / lock_held_s if lock_held_s else None
    vals = [(v, n) for v, n in ((cc, "cores"), (lc, "lock")) if v]
    return {"core_cap": cc, "lock_cap": lc, "binds": min(vals)[1] if vals else None}


def auto_gamma(profile, point):
    """gamma from the point itself: mean hold of its runs against the hold of the run alone. Fitted on the point it predicts."""
    alone = profile["runs"][0].get("sampler_hold_s") or 0
    held = [r["sampler_hold_s"] for r in point["runs"] if r.get("sampler_hold_s") is not None]
    if not alone or not held:
        return 0.0
    ratio = (sum(held) / len(held)) / alone
    return max(ratio - 1.0, 0.0) / max(point["slots"] - 1, 1)


# --- validation -------------------------------------------------------------------------------------------------------------------------------
def err(pred, meas):
    return None if pred is None or meas in (None, 0) else (pred - meas) / meas


def pc(x):
    return "NA" if x is None else f"{100 * x:+.1f}%"


def validate(data, cpus=28.0):
    man = json.load(open(os.path.join(data, "manifest.json")))
    rows = []
    for m in man["points"]:
        pt = json.load(open(os.path.join(data, m["point"])))
        pr = json.load(open(os.path.join(data, m["profile"])))
        k, S, N = pt["k"], pt["slots"], len(pt["runs"])
        ms = pt["measured"]
        row = {"point": m["name"], "profile": m["profile"].replace(".compact.json", ""), "k": k, "slots": S, "runs": N, "T": pt["T"],
               "meas_win": ms["win_rate"], "meas_span": ms["span_rate"], "meas_prog": ms["progress_rate"],
               "caps": caps(pt.get("cgroup_core_s_per_M"), pt.get("lock_held_mean"), cpus, S, pt["pairs_M_per_run"]),
               "meas_wait": pt.get("lock_wait_mean"), "meas_hold": pt.get("lock_held_mean"), "variants": {}}
        g_auto = auto_gamma(pr, pt)
        for label, g, c in (("gamma 0", 0.0, cpus), ("gamma E18", GAMMA_E18, cpus), ("gamma auto", g_auto, cpus), ("gamma E18, no cpu limit", GAMMA_E18, 1e9)):
            p = predict(pr, k, S, N, c, g)
            row["variants"][label] = {"gamma": g, **p, "err_win": err(p["win_rate"], ms["win_rate"]), "err_span": err(p["span_rate"], ms["span_rate"]),
                                      "err_prog": err(p["progress_rate"], ms["progress_rate"])}
        rows.append(row)
    return rows


def markdown(rows):
    f = lambda x, d=3: "NA" if x is None else f"{x:.{d}f}"  # noqa: E731
    L = ["| point | profile run | slots | k | runs | T | measured progress | measured win | measured span | core cap | lock cap | binds | variant | gamma | pred progress | err progress | pred win | err win | pred span | err span | pred wait s | pred hold s |",
         "|" + "---|" * 22]
    for r in rows:
        first = True
        for lab, v in r["variants"].items():
            c = r["caps"]
            head = [r["point"], r["profile"], r["slots"], r["k"], r["runs"], r["T"], f(r["meas_prog"]), f(r["meas_win"]), f(r["meas_span"]), f(c["core_cap"]), f(c["lock_cap"]), c["binds"]]
            head = head if first else [""] * len(head)
            L.append("| " + " | ".join(str(x) for x in head) + f" | {lab} | {v['gamma']:.2f} | {f(v['progress_rate'])} | {pc(v['err_prog'])} | {f(v['win_rate'])} | {pc(v['err_win'])} | "
                     f"{f(v['span_rate'])} | {pc(v['err_span'])} | {f(v['mean_wait_s'], 0)} | {f(v['mean_hold_s'], 0)} |")
            first = False
    return "\n".join(L) + "\n\n" + RANKING_ONLY


# --- command line -----------------------------------------------------------------------------------------------------------------------------
RANKING_ONLY = "Tier 0 is a ranking screen only. Use the change between base and edited runs. The absolute rate is not a prediction."


def split_dir_arms(item):
    """'DIR' or 'DIR:AB' -> (dir, set of arm letters or None for all). A Windows drive letter before the colon is not an arm list."""
    m = re.fullmatch(r"(.+):([A-H]+)", item)
    return (m.group(1), set(m.group(2))) if m else (item, None)


def fit_load(alone, loaded, cores=28.0, slots=2, device_gb=93.6, exclude=()):
    """Fit the two load factors and the efficiency ratio on legs of the same binary.
    alone:  tier 1 out directories (single run demand). loaded: tier 2 out directories (k=8 load). Each item is DIR or DIR:ARMS.
    load_core = mean core-s/M loaded over alone. load_hold = mean lock held loaded over alone.
    efficiency = measured progress rate over the binding limit of the alone demand with the load factors. Legs with a fallback or a failed run are left out.
    exclude: 'DIR:LEGDIR' strings of legs to leave out, for example legs that shared the node with another job."""
    import demand
    bad = set(exclude)

    def legs_of(items):
        out = []
        for it in items:
            d, arms = split_dir_arms(it)
            res = demand.load_result(d)
            for l in res.get("legs", []):
                if arms is not None and l["arm"] not in arms:
                    continue
                if f"{os.path.basename(d.rstrip('/'))}:{l.get('dir')}" in bad or f"{d}:{l.get('dir')}" in bad:
                    continue
                if (l.get("fallback") or {}).get("runs") or (l.get("metrics") or {}).get("failed_runs"):
                    continue
                if abcore.contaminated_legs({"legs": [l]}):
                    continue   # the leg shared the node with another job
                out.append((d, l))
        return out

    A, L = legs_of(alone), legs_of(loaded)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    a_core = mean([l["demand"]["core_s_per_M"] for _, l in A if (l.get("demand") or {}).get("core_s_per_M")])
    a_hold = mean([l["demand"]["hold_s"] for _, l in A if (l.get("demand") or {}).get("hold_s")])
    l_core = mean([l["metrics"]["core_s_per_M"] for _, l in L if l["metrics"].get("core_s_per_M")])
    l_hold = mean([l["metrics"]["lock_held_s"] for _, l in L if l["metrics"].get("lock_held_s")])
    if not (a_core and a_hold and l_core and l_hold):
        raise SystemExit("not enough legs with core-s/M and lock held in both sets")
    lc, lh = l_core / a_core, l_hold / a_hold
    base = {"pairs_M": mean([l["demand"]["pairs_M"] for _, l in A]), "hold_s": a_hold, "core_s_per_M": a_core, "dev_peak_gb": max(l["demand"]["dev_peak_gb"] for _, l in A)}
    lim = demand.limits(base, cores, slots, device_gb, lc, lh)
    effs = [l["metrics"]["progress_rate"] / lim["node"] for _, l in L if l["metrics"].get("progress_rate")]
    eff = mean(effs)
    sd = (sum((x - eff) ** 2 for x in effs) / (len(effs) - 1)) ** 0.5 if len(effs) > 1 else None
    return {"load_core": lc, "load_hold": lh, "efficiency": eff, "efficiency_cv": (sd / eff) if sd else None, "binding": lim["binding"], "binding_limit": lim["node"],
            "alone": {"core_s_per_M": a_core, "hold_s": a_hold, "legs": len(A)}, "loaded": {"core_s_per_M": l_core, "lock_held_s": l_hold, "legs": len(L)},
            "efficiency_per_leg": effs, "cores": cores, "slots": slots, "device_gb": device_gb,
            "source": {"alone": [os.path.basename(x.rstrip("/")) for x in alone], "loaded": [os.path.basename(x.rstrip("/")) for x in loaded], "excluded": sorted(bad)}}


def run_loadfit(a):
    f = fit_load(a.alone, a.loaded, a.cores, a.slots, a.device_gb, a.exclude_leg)
    print(RANKING_ONLY)
    print(f"alone  : {f['alone']['legs']} legs, core-s/M {f['alone']['core_s_per_M']:.3f}, hold {f['alone']['hold_s']:.2f} s")
    print(f"loaded : {f['loaded']['legs']} legs, core-s/M {f['loaded']['core_s_per_M']:.3f}, hold {f['loaded']['lock_held_s']:.2f} s")
    print(f"load factor core {f['load_core']:.3f}, hold {f['load_hold']:.3f}")
    cv = f["efficiency_cv"]
    print(f"binding limit with load factors: {f['binding_limit']:.3f} ({f['binding']}). Efficiency {f['efficiency']:.3f}" + (f", leg to leg CV {100 * cv:.1f}%" if cv else ""))
    print("efficiency per loaded leg: " + " ".join(f"{x:.3f}" for x in f["efficiency_per_leg"]))
    if a.write:
        import demand
        path = a.out or demand.LOAD_FIT_FILE
        json.dump(f, open(path, "w"), indent=1)
        print(f"wrote {path}")
    return 0


def run_limits(a):
    """Demand vector and node rate limits of fast lane out directories. A ranking screen, never an absolute prediction."""
    import demand
    rows = []
    for d in a.out_dirs:
        res = demand.load_result(d)
        v = demand.tier1_demand_verdict(res, {"cores": a.cores, "device_gb": a.device_gb})
        g = abcore.tier1_verdict(res, None, False, "gain")
        name = os.path.basename(d.rstrip("/"))
        if a.rule_table:
            pc = v["demand"]
            pr = lambda x: "NA" if x is None else f"{100 * x:+.1f}%"  # noqa: E731
            rows.append(f"| {name} | {pr(pc['change_by_limit']['core'])} | {pr(pc['change_by_limit']['lock'])} | {pc['limits_A']['binding']} | "
                        f"{pr(pc['change'])} | {v['verdict']} ({v['rule']}) | {g['verdict']} ({g['rule']}) |")
        else:
            print(f"== {name}")
            print(RANKING_ONLY)
            print(demand.render_demand(res, v))
            print(f"\nDemand rule: {v['verdict']} ({v['rule']}) {v['why']}")
            print(f"Gain rule:   {g['verdict']} ({g['rule']}) {g['why']}")
    if a.rule_table:
        print("| run | core limit change | lock limit change | A binds on | predicted node change | demand rule | gain rule |")
        print("|---|---|---|---|---|---|---|")
        print("\n".join(rows))
        print("\n" + RANKING_ONLY)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("point")
    e.add_argument("--samples")
    e.add_argument("--results-tsv")
    e.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("profile")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--cpus", type=float, default=28.0)
    p.add_argument("--gamma", default="0", help="a number or 'auto' (from the profile's own k=1 and the point given with --point)")
    p.add_argument("--point", help="a loaded point for --gamma auto")
    p.add_argument("--scale", action="append", default=[])
    p.add_argument("--order", choices=("fifo", "random"), default="fifo")
    p.add_argument("--seeds", type=int, default=1)
    lf = sub.add_parser("loadfit", help="fit the load factors and the efficiency ratio on legs of the same binary")
    lf.add_argument("--alone", nargs="+", required=True, help="tier 1 out directories, DIR or DIR:ARMS")
    lf.add_argument("--loaded", nargs="+", required=True, help="tier 2 out directories, DIR or DIR:ARMS")
    lf.add_argument("--exclude-leg", action="append", default=[], help="DIRNAME:LEGDIR of a leg to leave out, for example one that shared the node")
    lf.add_argument("--cores", type=float, default=28.0)
    lf.add_argument("--slots", type=int, default=2)
    lf.add_argument("--device-gb", type=float, default=93.6)
    lf.add_argument("--write", action="store_true", help="write model_data/load_fit.json, which demand.py reads")
    lf.add_argument("--out", help="with --write: write this file instead")
    lm = sub.add_parser("limits", help="demand vector and node rate limits of a fast lane out directory, for both arms")
    lm.add_argument("out_dirs", nargs="+")
    lm.add_argument("--cores", type=float, default=28.0)
    lm.add_argument("--slots", type=int)
    lm.add_argument("--device-gb", type=float, default=93.6)
    lm.add_argument("--rule-table", action="store_true", help="one row per directory: both rules side by side")
    v = sub.add_parser("validate")
    v.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data"))
    v.add_argument("--cpus", type=float, default=28.0)
    v.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        json.dump(extract(a.point, a.samples, a.results_tsv), open(a.out, "w"), separators=(",", ":"))
        print(f"wrote {a.out}")
        return 0
    if a.cmd == "loadfit":
        return run_loadfit(a)
    if a.cmd == "limits":
        return run_limits(a)
    if a.cmd == "predict":
        pr = json.load(open(a.profile))
        if "template" not in pr:
            raise SystemExit("this profile has no template. Extract a k=1 point with --samples.")
        g = auto_gamma(pr, json.load(open(a.point))) if a.gamma == "auto" else float(a.gamma)
        sc = parse_scales(a.scale, pr["template"])
        base = predict(pr, a.k, a.slots, a.runs, a.cpus, g, (), a.order, a.seeds)
        new = predict(pr, a.k, a.slots, a.runs, a.cpus, g, sc, a.order, a.seeds) if sc else base
        print(RANKING_ONLY)
        print(f"profile {pr['name']} (T={pr['T']}, slots {pr['slots']}) k={a.k} slots={a.slots} runs={a.runs} cpus={a.cpus:g} gamma={g:.3f} edits={a.scale or 'none'}")
        run0 = pr["runs"][0]
        c = caps(pr.get("cgroup_core_s_per_M"), run0.get("sampler_hold_s"), a.cpus, a.slots, pr["pairs_M_per_run"])
        print(f"  caps (from the profile run alone): cores {c['core_cap']:.3f}, lock {c['lock_cap']:.3f} M pairs/s. Binding: {c['binds']}")
        for key in ("progress_rate", "win_rate", "span_rate", "mean_wall_s", "mean_wait_s", "mean_hold_s"):
            b, n = base[key], new[key]
            ch = "" if b in (None, 0) or n is None else f"  ({100 * (n - b) / b:+.1f}%)"
            print(f"  {key:12s} base {'NA' if b is None else f'{b:.3f}'}  edited {'NA' if n is None else f'{n:.3f}'}{ch}")
        return 0
    rows = validate(a.data, a.cpus)
    if not a.markdown:
        print(RANKING_ONLY)
    if a.markdown:
        print(markdown(rows))
    else:
        for r in rows:
            print(f"{r['point']}: measured progress {r['meas_prog']} win {r['meas_win']} span {r['meas_span']:.3f}. caps cores {r['caps']['core_cap']:.3f} lock {r['caps']['lock_cap']:.3f}, binds {r['caps']['binds']}")
            for lab, v in r["variants"].items():
                print(f"   {lab:26s} gamma {v['gamma']:.2f} progress {v['progress_rate']} ({pc(v['err_prog'])}) win {v['win_rate']} ({pc(v['err_win'])}) span {v['span_rate']:.3f} ({pc(v['err_span'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
