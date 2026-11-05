#!/usr/bin/env python3
"""Pure logic of the fast lane: arm specs, verdict rules and the markdown table.

Nothing here talks to the cluster. ab.py (laptop) and the jobs (Clipper) both import it.
The jobs call it through `python3 abcore.py decide ...` to stop a tier 2 run early.

Result schema (written by collect.py as result.json, read by ab.py):
  {"schema": 1, "tier": 1|2, "name": str, "job": str, "node": str, "scripts": str, "ref": str,
   "params": {"T": 7, "slots": 2, "sample": "pbmc1k_v3", "k": 8, "runs": 24, ...},
   "legs": [{"leg": 1, "arm": "A",
             "prov": {"bin", "sha256", "commit", "embedded", "env", "args"},
             "metrics": {"wall_s", "core_s_per_M", "lock_held_s", "lock_wait_s", "dev_peak_gb", "host_peak_gb",
                         "win_rate", "progress_rate", "span_rate", "win_n", "runs", "failed_runs", "pairs_M"},
             "stage_core_s": {stage: core-s},
             "ident": {"checked": n, "identical": n, "detail": path},
             "fallback": {"runs": n, "lines": [str]},
             "stopped": str or ""}],
   "notes": [str]}
"""
import json
import re
import shlex
import sys

ARM_KEYS = ("bin", "env", "commit", "args", "label")
HARNESS_VARS = ("SINGLET_ENGINE_GPU_SLOTS", "SINGLET_ENGINE_GPU_E2E", "SINGLET_ENGINE_GPU_INGEST", "SINGLET_GPU_PILEUP",
                "SINGLET_ENGINE_GPU")
FORBIDDEN_VARS = ("E18_COPY", "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "HOME", "SLURM_JOB_ID")
STAGES = ("start", "scan", "refs", "STAR", "stream", "handoff", "pileup", "pileupfin", "nonhost", "export", "demux")

# Provisional rules. Every number here can be changed from the command line of ab.py.
DEFAULT_TIER1 = {"promote_gain": 0.05, "respect_spread": True, "regress_guard": None}


class SpecError(ValueError):
    pass


_MSYS = re.compile(r"^[A-Za-z]:/(Program Files/)?Git(?=/)", re.I)


def unmsys(p):
    """Git Bash rewrites an argument that starts with /mnt/... into C:/Program Files/Git/mnt/... Undo it (as clipper.py does for remote paths)."""
    p = p.replace("\\", "/")
    return _MSYS.sub("", p) if _MSYS.match(p) else p


def parse_arm(text):
    """'bin=/d,env=X=1 Y=2,commit=abc12345,args=--no-snps' -> dict. Keys are split where ',key=' starts."""
    parts = re.split(r",(?=(?:%s)=)" % "|".join(ARM_KEYS), text)
    d = {}
    for p in parts:
        k, _, v = p.partition("=")
        if k not in ARM_KEYS or not _:
            raise SpecError(f"arm spec: bad part '{p}' (keys: {', '.join(ARM_KEYS)})")
        if k in d:
            raise SpecError(f"arm spec: key '{k}' given twice")
        d[k] = v
    if "bin" not in d:
        raise SpecError("arm spec: bin=<dir> is required")
    d["bin"] = unmsys(d["bin"])
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", d["bin"]):
        raise SpecError(f"arm spec: bin must be an absolute path of plain characters, got '{d['bin']}'")
    d["bin"] = d["bin"].rstrip("/")
    d["commit"] = d.get("commit", "")
    if d["commit"] and not re.fullmatch(r"[0-9a-f]{8,40}", d["commit"]):
        raise SpecError("arm spec: commit must be 8 to 40 lowercase hex digits")
    env = []
    for tok in shlex.split(d.get("env", "")):
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", tok, re.S)
        if not m:
            raise SpecError(f"arm spec: env item '{tok}' is not NAME=value")
        if m.group(1) in FORBIDDEN_VARS:
            raise SpecError(f"arm spec: env {m.group(1)} is controlled by the harness")
        env.append((m.group(1), m.group(2)))
    d["env"] = env
    d["args"] = shlex.split(d.get("args", ""))
    d["label"] = d.get("label", "")
    return d


def arm_notes(arm, name):
    """Warnings about an arm that change what the comparison means."""
    notes = []
    for k, _ in arm["env"]:
        if k in HARNESS_VARS:
            notes.append(f"arm {name}: env sets {k}, which the harness also sets. The arm value wins.")
    if arm["args"]:
        notes.append(f"arm {name}: extra command line arguments {arm['args']}. They can change the output.")
    return notes


def fmt_env(env):
    return " ".join(f"{k}={v}" for k, v in env)


# --- numbers ------------------------------------------------------------------------------------------------------------------------
def num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    """(max - min) / mean over the values, or None with fewer than two."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (max(xs) - min(xs)) / m if m else None


def arm_legs(res, arm):
    return [l for l in res.get("legs", []) if l.get("arm") == arm]


def arm_values(res, arm, metric):
    return [num(l.get("metrics", {}).get(metric)) for l in arm_legs(res, arm)]


def arm_summary(res, arm):
    legs = arm_legs(res, arm)
    s = {"legs": len(legs)}
    for m in ("wall_s", "core_s_per_M", "lock_held_s", "lock_wait_s", "win_rate", "span_rate", "progress_rate"):
        v = arm_values(res, arm, m)
        s[m] = mean(v)
        s[m + "_spread"] = spread(v)
    for m in ("dev_peak_gb", "host_peak_gb"):
        v = [x for x in arm_values(res, arm, m) if x is not None]
        s[m] = max(v) if v else None
    s["ident_checked"] = sum(int(l.get("ident", {}).get("checked", 0)) for l in legs)
    s["ident_identical"] = sum(int(l.get("ident", {}).get("identical", 0)) for l in legs)
    s["fallback_runs"] = sum(int(l.get("fallback", {}).get("runs", 0)) for l in legs)
    s["fallback_lines"] = [x for l in legs for x in l.get("fallback", {}).get("lines", [])]
    s["failed_runs"] = sum(int(num(l.get("metrics", {}).get("failed_runs")) or 0) for l in legs)
    stages = {}
    for l in legs:
        for k, v in (l.get("stage_core_s") or {}).items():
            stages.setdefault(k, []).append(num(v))
    s["stage_core_s"] = {k: mean(v) for k, v in stages.items()}
    return s


def gain_lower_better(a, b):
    """Fraction by which B improves on A for a metric where less is better. Positive means B is better."""
    if a is None or b is None or a == 0:
        return None
    return (a - b) / a


def gain_higher_better(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a


def pct(x, d=1):
    return "NA" if x is None else f"{100 * x:+.{d}f}%"


# --- validity rules shared by both tiers ---------------------------------------------------------------------------------------------------
def validity(res, A, B, intended_change):
    """Returns (verdict, rule, why) when the data forces a verdict, else None.
    Rules, in order:
      V1 an arm has no completed leg or no metrics          -> INCONCLUSIVE
      V2 B fell back and A did not                          -> REJECT
      V3 A fell back                                        -> INCONCLUSIVE (the baseline is invalid)
      V4 B lost identity and A did not (unless declared)    -> REJECT
      V5 A lost identity                                    -> INCONCLUSIVE
      V6 identity was not checked at all                    -> INCONCLUSIVE
    """
    if not A["legs"] or not B["legs"]:
        return "INCONCLUSIVE", "V1", "an arm has no leg in the result"
    for nm, s in (("A", A), ("B", B)):
        if s["wall_s"] is None and s["progress_rate"] is None:
            return "INCONCLUSIVE", "V1", f"arm {nm} has no measured run"
    if B["fallback_runs"] and not A["fallback_runs"]:
        return "REJECT", "V2", f"B fell back in {B['fallback_runs']} run(s). First: {B['fallback_lines'][0] if B['fallback_lines'] else 'see the log'}"
    if A["fallback_runs"]:
        return "INCONCLUSIVE", "V3", f"the baseline A fell back in {A['fallback_runs']} run(s). B had {B['fallback_runs']}. Tier results are not valid"
    a_lost = A["ident_checked"] and A["ident_identical"] < A["ident_checked"]
    b_lost = B["ident_checked"] and B["ident_identical"] < B["ident_checked"]
    if b_lost and not a_lost and not intended_change:
        return "REJECT", "V4", f"B output differs from the reference in {B['ident_checked'] - B['ident_identical']} of {B['ident_checked']} checks"
    if a_lost:
        return "INCONCLUSIVE", "V5", "the baseline A output differs from the reference. Check the reference and the commit"
    if not A["ident_checked"] or not B["ident_checked"]:
        return "INCONCLUSIVE", "V6", "identity was not checked for both arms"
    return None


# --- tier 1 ---------------------------------------------------------------------------------------------------------------------------
def tier1_verdict(res, p=None, intended_change=False, kind="gain"):
    """kind gain: the first provisional rule (core-s/M or lock held gain). kind demand: the demand rule of demand.py (predicted node rate)."""
    if kind == "demand":
        import demand
        return demand.tier1_demand_verdict(res, p, intended_change)
    return tier1_gain_verdict(res, p, intended_change)


def tier1_gain_verdict(res, p=None, intended_change=False):
    p = dict(DEFAULT_TIER1, **(p or {}))
    A, B = arm_summary(res, "A"), arm_summary(res, "B")
    out = {"tier": 1, "params": {**p, "intended_change": intended_change}, "A": A, "B": B}
    g = {"core_s_per_M": gain_lower_better(A["core_s_per_M"], B["core_s_per_M"]),
         "lock_held_s": gain_lower_better(A["lock_held_s"], B["lock_held_s"])}
    sp = {m: max([x for x in (A[m + "_spread"], B[m + "_spread"]) if x is not None] or [0.0]) for m in g}
    out["gain"], out["spread"] = g, sp
    v = validity(res, A, B, intended_change)
    if v:
        out["verdict"], out["rule"], out["why"] = v
        return out
    best = max((k for k in g if g[k] is not None), key=lambda k: g[k], default=None)
    if best is None:
        out.update(verdict="INCONCLUSIVE", rule="T1-1", why="neither core-s/M nor lock held could be compared")
        return out
    other = [k for k in g if k != best and g[k] is not None]
    guard = p["regress_guard"]
    if g[best] < p["promote_gain"]:
        out.update(verdict="INCONCLUSIVE", rule="T1-2",
                   why=f"the best gain is {best} {pct(g[best])}, below the promotion gain {100 * p['promote_gain']:.1f}%")
    elif p["respect_spread"] and g[best] <= sp[best]:
        out.update(verdict="INCONCLUSIVE", rule="T1-3",
                   why=f"the gain in {best} {pct(g[best])} is inside the run-to-run spread {100 * sp[best]:.1f}% of the two runs per arm")
    elif guard is not None and any(g[o] < -guard for o in other):
        o = next(o for o in other if g[o] < -guard)
        out.update(verdict="INCONCLUSIVE", rule="T1-4", why=f"{best} improves {pct(g[best])} but {o} worsens {pct(g[o])}, past the guard {100 * guard:.1f}%")
    else:
        out.update(verdict="PROMOTE", rule="T1-5",
                   why=f"{best} improves {pct(g[best])} (promotion gain {100 * p['promote_gain']:.1f}%), identity kept, no fallback")
    return out


# --- tier 2 ---------------------------------------------------------------------------------------------------------------------------
# The provisional tier 2 rule lives here and nowhere else. ab.py can override each value from the command line.
# Noise behind the values (A/A runs): core-s/M CV 0.4%, progress rate CV about 4.5%, lock held CV 4.8%. Lock wait CV is 14%: report it, never decide on it.
# The values are recalibrated on the s3 against s3 pair in phase 3.
TIER2_RULE = {
    "core_gain": 0.02,       # PROMOTE when core-s/M falls by this fraction. REJECT when it rises by this fraction with no rate gain.
    "rate_gain": 0.09,       # PROMOTE when the progress rate rises by this fraction. REJECT when it falls by this fraction.
    "lock_held_gain": 0.10,  # PROMOTE when lock held seconds fall by this fraction.
}


def progress_rate(runs, k):
    """Progress-weighted rate in M pairs per second. Each run adds pairs * overlap(run, window) / wall. The window runs from the k-th completion
    to the last run start. Edge shifts move it 0.4 to 2.6%. Windowed completions moved 9 to 30%. runs: dicts with start, end, pairs, rc."""
    ends = sorted(r["end"] for r in runs if r.get("rc", 0) == 0)
    if len(ends) < k or not runs:
        return None
    ws, we = ends[k - 1], max(r["start"] for r in runs)
    if we <= ws:
        return None
    tot = 0.0
    for r in runs:
        ov = min(r["end"], we) - max(r["start"], ws)
        if ov > 0 and r["end"] > r["start"]:
            tot += (r["pairs"] or 0) / 1e6 * ov / (r["end"] - r["start"])
    return tot / (we - ws)


def separation(res, rule=None):
    """Is B clearly better or clearly worse than A on the progress rate, by the rule's threshold, with the legs present so far?
    Better: the lowest B leg beats the highest A leg by more than rate_gain. Worse is the mirror. Needs three legs, so one arm has been seen twice."""
    rule = rule or TIER2_RULE
    if len(arms_of(res)) > 2:
        return None   # a batch check never stops early on the separation of two arms
    if len([l for l in res.get("legs", []) if l.get("completed", True)]) < 3:
        return None
    ra = [x for x in (num(l.get("metrics", {}).get("progress_rate")) for l in arm_legs(res, "A")) if x is not None]
    rb = [x for x in (num(l.get("metrics", {}).get("progress_rate")) for l in arm_legs(res, "B")) if x is not None]
    if not ra or not rb:
        return None
    m = rule["rate_gain"]
    if min(rb) > max(ra) * (1 + m):
        return "better"
    if max(rb) < min(ra) * (1 - m):
        return "worse"
    return None


def tier2_verdict(res, rule=None, intended_change=False):
    """Rules after the validity rules V1 to V6:
      T2-1 progress rate or core-s/M missing                                   -> INCONCLUSIVE
      T2-2 progress rate falls by rate_gain or more                            -> REJECT
      T2-3 core-s/M rises by core_gain or more and the progress rate gains nothing -> REJECT
      T2-4 core-s/M falls by core_gain or more                                 -> PROMOTE
      T2-5 progress rate rises by rate_gain or more                            -> PROMOTE
      T2-6 lock held falls by lock_held_gain or more                           -> PROMOTE
      T2-7 anything else                                                       -> INCONCLUSIVE
    REJECT rules come before PROMOTE rules."""
    rule = dict(TIER2_RULE, **(rule or {}))
    A, B = arm_summary(res, "A"), arm_summary(res, "B")
    out = {"tier": 2, "params": {**rule, "intended_change": intended_change}, "A": A, "B": B}
    va, vb = arm_values(res, "A", "progress_rate"), arm_values(res, "B", "progress_rate")
    A["progress_rate"], B["progress_rate"] = mean(va), mean(vb)
    A["progress_rate_spread"], B["progress_rate_spread"] = spread(va), spread(vb)
    g = {"progress_rate": gain_higher_better(A["progress_rate"], B["progress_rate"]),
         "core_s_per_M": gain_lower_better(A["core_s_per_M"], B["core_s_per_M"]),
         "lock_held_s": gain_lower_better(A["lock_held_s"], B["lock_held_s"]),
         "lock_wait_s": gain_lower_better(A["lock_wait_s"], B["lock_wait_s"])}
    out["gain"] = g
    out["spread"] = {"progress_rate_A": A["progress_rate_spread"], "progress_rate_B": B["progress_rate_spread"]}
    out["separation"] = separation(res, rule)
    v = validity(res, A, B, intended_change)
    if v:
        out["verdict"], out["rule"], out["why"] = v
        return out
    pr, co, lk = g["progress_rate"], g["core_s_per_M"], g["lock_held_s"]
    if pr is None or co is None:
        out.update(verdict="INCONCLUSIVE", rule="T2-1", why="the progress rate or core-s/M could not be compared. The window may be empty. Use more than 2k runs")
    elif pr <= -rule["rate_gain"]:
        out.update(verdict="REJECT", rule="T2-2", why=f"the progress rate falls {pct(pr)}, past {100 * rule['rate_gain']:.0f}%")
    elif co <= -rule["core_gain"] and pr <= 0:
        out.update(verdict="REJECT", rule="T2-3", why=f"core-s/M rises {pct(-co)} with no rate gain ({pct(pr)})")
    elif co >= rule["core_gain"]:
        out.update(verdict="PROMOTE", rule="T2-4", why=f"core-s/M falls {pct(-co)}, past {100 * rule['core_gain']:.0f}%. Identity kept, no fallback")
    elif pr >= rule["rate_gain"]:
        out.update(verdict="PROMOTE", rule="T2-5", why=f"the progress rate rises {pct(pr)}, past {100 * rule['rate_gain']:.0f}%. Identity kept, no fallback")
    elif lk is not None and lk >= rule["lock_held_gain"]:
        out.update(verdict="PROMOTE", rule="T2-6", why=f"lock held falls {pct(-lk)}, past {100 * rule['lock_held_gain']:.0f}%. Identity kept, no fallback")
    else:
        out.update(verdict="INCONCLUSIVE", rule="T2-7", why=f"progress rate {pct(pr)}, core-s/M {pct(-co)}, lock held {pct(-lk) if lk is not None else 'NA'}: none passes a threshold")
    return out


# --- table ------------------------------------------------------------------------------------------------------------------------------
def f(x, d=2):
    return "NA" if x is None else f"{x:.{d}f}"


def render(res, verdict, warnings=()):
    L = []
    for arm in ("A", "B"):
        legs = arm_legs(res, arm)
        if not legs:
            L.append(f"**{arm}** no leg in the result")
            continue
        pv = legs[0].get("prov", {})
        L.append(f"**{arm}** bin `{pv.get('bin', '?')}` commit `{pv.get('commit', '?')}` sha256 `{(pv.get('sha256') or '?')[:16]}` "
                 f"embedded {pv.get('embedded', '?')} env `{pv.get('env') or '-'}` args `{pv.get('args') or '-'}`")
    P = res.get("params", {})
    L.append(f"tier {res.get('tier')} job {res.get('job', '?')} node {res.get('node', '?')} scripts `{res.get('scripts', '?')}` ref `{res.get('ref', '?')}` "
             f"sample {P.get('sample', '?')} T={P.get('T', '?')} slots={P.get('slots', '?')}"
             + (f" k={P['k']} runs per leg={P['runs']}" if res.get("tier") == 2 else ""))
    L.append(f"order {' '.join(l['arm'] for l in res.get('legs', []))}")
    cn = contaminated_legs(res)
    if cn:
        L.append("note: CO-TENANT. These legs shared the node with another job: " + ", ".join(f"leg {l['leg']} ({l['arm']})" for l in cn) + ". Read their numbers with care.")
    for n in list(res.get("notes", [])) + list(warnings):
        L.append(f"note: {n}")
    L.append("")
    A, B = verdict["A"], verdict["B"]
    rows = [("wall s", "wall_s", 1, "lower"), ("core-s/M", "core_s_per_M", 2, "lower"), ("lock held s", "lock_held_s", 1, "lower"),
            ("lock wait s", "lock_wait_s", 1, "lower")]
    if res.get("tier") == 2:
        rows = [("progress rate M pairs/s", "progress_rate", 3, "higher"), ("span rate M pairs/s (same run count only)", "span_rate", 3, "higher")] + rows
    L.append("| measure | A | B | B vs A | A legs | B legs |")
    L.append("|---|---|---|---|---|---|")
    for lab, key, d, better in rows:
        a, b = A.get(key), B.get(key)
        g = gain_lower_better(a, b) if better == "lower" else gain_higher_better(a, b)
        la = ", ".join(f(x, d) for x in arm_values(res, "A", key))
        lb = ", ".join(f(x, d) for x in arm_values(res, "B", key))
        L.append(f"| {lab} | {f(a, d)} | {f(b, d)} | {pct(g)} | {la} | {lb} |")
    L.append(f"| device peak GB | {f(A['dev_peak_gb'], 1)} | {f(B['dev_peak_gb'], 1)} | | | |")
    L.append(f"| host peak GB | {f(A['host_peak_gb'], 1)} | {f(B['host_peak_gb'], 1)} | | | |")
    L.append(f"| identity vs reference | {A['ident_identical']}/{A['ident_checked']} | {B['ident_identical']}/{B['ident_checked']} | | | |")
    L.append(f"| fallback runs | {A['fallback_runs']} | {B['fallback_runs']} | | | |")
    L.append(f"| failed runs | {A['failed_runs']} | {B['failed_runs']} | | | |")
    stages = [s for s in STAGES if s in A["stage_core_s"] or s in B["stage_core_s"]]
    if stages:
        L.append("")
        L.append("Stage core-s (mean per run)")
        L.append("")
        L.append("| stage | " + " | ".join(stages) + " |")
        L.append("|---|" + "---|" * len(stages))
        for nm, s in (("A", A), ("B", B)):
            L.append(f"| {nm} | " + " | ".join(f(s["stage_core_s"].get(k), 0) for k in stages) + " |")
    for nm, s in (("A", A), ("B", B)):
        if s["fallback_lines"]:
            L.append("")
            L.append(f"Fallback lines of {nm}: " + " / ".join(x[:160] for x in s["fallback_lines"][:3]))
    stopped = [l.get("stopped") for l in res.get("legs", []) if l.get("stopped")]
    if stopped:
        L.append("")
        L.append("Stopped early: " + "; ".join(stopped))
    if res.get("tier") == 1:
        import demand
        L.append(demand.render_demand(res, verdict))
    L.append("")
    L.append(f"**VERDICT: {verdict['verdict']}** (rule {verdict['rule']}) {verdict['why']}")
    pr = verdict["params"]
    L.append("Parameters: " + ", ".join(f"{k}={'unset' if v is None else v}" for k, v in pr.items()))
    return "\n".join(L)


def contaminated_legs(res):
    """Legs that shared the node with another job. tp_point.py counts the other jobs of the node in the squeue snapshots (other_jobs_max)."""
    return [l for l in res.get("legs", []) if (num(l.get("metrics", {}).get("other_jobs_max")) or 0) > 0]


def clean_result(res):
    """The result without the legs that shared the node. An arm left with no leg gives V1 in the verdict."""
    bad = {l["leg"] for l in contaminated_legs(res)}
    return dict(res, legs=[l for l in res.get("legs", []) if l["leg"] not in bad])


def arms_of(res):
    return sorted({l.get("arm") for l in res.get("legs", [])})


def pair_result(res, arm):
    """A result that holds only the legs of arm A and the given arm, the latter relabelled B. The two arm rules then apply to a batch check."""
    legs = []
    for l in res.get("legs", []):
        if l.get("arm") == "A":
            legs.append(l)
        elif l.get("arm") == arm:
            legs.append(dict(l, arm="B"))
    return dict(res, legs=legs)


def batch_verdicts(res, rule=None, intended_change=False, predicted=None):
    """Tier 2 as a batch check: every arm other than A against A, with the tier 2 rule. predicted: arm -> predicted node rate change (or None)."""
    out = {}
    for x in [a for a in arms_of(res) if a != "A"]:
        v = tier2_verdict(pair_result(res, x), rule, intended_change)
        v["predicted_change"] = (predicted or {}).get(x)
        out[x] = v
    return out


def render_batch(res, verdicts, warnings=()):
    """Markdown for a batch check: provenance of every arm, then measured against predicted per arm."""
    L = []
    for arm in arms_of(res):
        legs = arm_legs(res, arm)
        pv = legs[0].get("prov", {})
        L.append(f"**{arm}** bin `{pv.get('bin', '?')}` commit `{pv.get('commit', '?')}` sha256 `{(pv.get('sha256') or '?')[:16]}` embedded {pv.get('embedded', '?')} "
                 f"env `{pv.get('env') or '-'}` args `{pv.get('args') or '-'}`")
    P = res.get("params", {})
    L.append(f"tier 2 batch check job {res.get('job', '?')} node {res.get('node', '?')} scripts `{res.get('scripts', '?')}` ref `{res.get('ref', '?')}` "
             f"sample {P.get('sample', '?')} T={P.get('T', '?')} slots={P.get('slots', '?')} k={P.get('k', '?')} runs per leg={P.get('runs', '?')}")
    L.append(f"order {' '.join(l['arm'] for l in res.get('legs', []))}")
    cn = contaminated_legs(res)
    if cn:
        L.append("note: CO-TENANT. These legs shared the node with another job: " + ", ".join(f"leg {l['leg']} ({l['arm']})" for l in cn) + ". Read their numbers with care.")
    for n in list(res.get("notes", [])) + list(warnings):
        L.append(f"note: {n}")
    L += ["", "Measured against predicted. Predicted is the node rate change from tier 1 demand vectors. Measured is the change of the progress rate against arm A.", "",
          "| arm | predicted node rate | measured progress rate | measured core-s/M | measured lock held | measured span rate | fallback runs | identity | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    A = arm_summary(res, "A")
    for x, v in verdicts.items():
        B = v["B"]
        sg = gain_higher_better(A["span_rate"], B["span_rate"])
        L.append(f"| {x} | {pct(v.get('predicted_change'))} | {pct(v['gain']['progress_rate'])} | {pct(v['gain']['core_s_per_M'])} | {pct(v['gain']['lock_held_s'])} | "
                 f"{pct(sg)} | {B['fallback_runs']} | {B['ident_identical']}/{B['ident_checked']} | {v['verdict']} ({v['rule']}) |")
    L += ["", "Per leg, progress rate and span rate (M pairs/s), core-s/M, lock held s", "", "| leg | arm | progress | span | core-s/M | lock held | lock wait |", "|---|---|---|---|---|---|---|"]
    for l in res.get("legs", []):
        m = l.get("metrics", {})
        L.append(f"| {l['leg']} | {l['arm']} | {f(num(m.get('progress_rate')), 3)} | {f(num(m.get('span_rate')), 3)} | {f(num(m.get('core_s_per_M')))} | "
                 f"{f(num(m.get('lock_held_s')), 1)} | {f(num(m.get('lock_wait_s')), 1)} |")
    L += ["", "Tier 2 rule values: " + json.dumps({k: v for k, v in next(iter(verdicts.values()))["params"].items()}) if verdicts else ""]
    return "\n".join(L)


def decide(res, rule=None):
    """For the tier 2 job after each leg. Prints CONTINUE or STOP <reason>. The caller stops on any STOP."""
    for l in res.get("legs", []):
        if l.get("fallback", {}).get("runs", 0) or l.get("stopped", "").startswith("fallback"):
            return f"STOP fallback in leg {l['leg']} ({l['arm']})"
    s = separation(res, rule)
    if s:
        return f"STOP arms clearly separated (B {s})"
    return "CONTINUE"


def main(argv):
    if len(argv) >= 2 and argv[0] == "decide":
        res = json.load(open(argv[1]))
        rule = dict(TIER2_RULE, rate_gain=float(argv[2])) if len(argv) > 2 and argv[2] not in ("", "unset") else None
        print(decide(res, rule))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
