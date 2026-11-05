#!/usr/bin/env python3
"""Fast lane A/B driver. Run from the laptop. Reads bench/gpu_fast/README.md for what each tier can and cannot show.

  python bench/gpu_fast/ab.py --tier 1 --a bin=<dir>,env="X=1" --b bin=<dir>,env="Y=2" --name <idea>

It writes the spec, submits one job through the Clipper bridge, waits, reads result.json, prints one markdown table and a verdict
(REJECT, PROMOTE or INCONCLUSIVE, with the rule that fired) and writes verdict.json. Output goes to bench/gpu_fast/out/<name>-t<tier>-<time>/.

Arm spec: bin=<dir>[,env=NAME=value NAME2=value][,commit=<hex8+>][,args=<extra singlet arguments>][,label=<text>]
  bin     a directory with singlet-gpu and commit-gpu (as the batch 1 harness). The job checks the recorded commit, records the sha256 and checks
          that the commit string is inside the binary.
  commit  optional. When given, the recorded commit must start with it.
  env     added to the environment of every run of that arm.
  args    added to the singlet command line of that arm. Output can change. Use --intended-output-change then.

The bridge. This tool never starts it. When the bridge drops while waiting, the job state is treated as unknown. The tool retries for
--bridge-grace minutes, then exits with code 3 and prints the command that resumes the wait (--resume).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abcore  # noqa: E402

W = "/mnt/projects/debruinz_project/debruinz/singlet"
CLIPPER = os.path.join(os.path.expanduser("~"), ".claude", "skills", "clipper", "clipper.py")
DEFAULT_REF = f"{W}/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3"
BRANCH = "bench/gpu-fast"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
DOWN_PATTERNS = ("Clipper bridge is not running", "Clipper bridge is not reachable", "not connected: restart bridge")


class BridgeDown(Exception):
    pass


class Fatal(Exception):
    pass


# --- cluster access ------------------------------------------------------------------------------------------------------------------------
class Remote:
    def __init__(self, python=sys.executable, clipper=CLIPPER):
        self.cmd = [python, clipper]

    def _call(self, args, timeout=300):
        try:
            p = subprocess.run(self.cmd + args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise BridgeDown("call timed out")
        # clipper.py reports a dead bridge on stderr. Remote output on stdout is never searched, so a log line cannot look like a dead bridge.
        if any(x in (p.stderr or "") for x in DOWN_PATTERNS):
            raise BridgeDown((p.stderr or "").strip()[:200])
        return p.returncode, p.stdout or "", p.stderr or ""

    def run(self, command, timeout=300):
        return self._call(["run", command, "--timeout", str(timeout)], timeout + 60)

    def put(self, local, remote):
        rc, out, err = self._call(["put", local, remote])
        if '"ok": true' not in out and '"ok":true' not in out:
            raise Fatal(f"put {remote} failed: {out.strip()[:200]} {err.strip()[:200]}")

    def status(self):
        try:
            rc, out, err = self._call(["status"], 30)
        except BridgeDown:
            return False
        return rc == 0 and "error" not in out.lower()


# --- spec -----------------------------------------------------------------------------------------------------------------------------------
def q(x):
    return shlex.quote(str(x))


def arr(items):
    return "(" + " ".join(q(i) for i in items) + ")"


def build_spec(a, armA, armB, run, scripts, sbatch_sha):
    L = ["# written by ab.py. Do not edit.",
         f"NAME={q(a.name)}", f"RUN={q(run)}", f"SAMPLE={q(a.sample)}", f"T={a.T}", f"SLOTS={a.slots}", f"ORDER={q(a.order)}",
         f"REFDIR={q(a.ref)}", f"INTENDED={1 if a.intended_output_change else 0}", f"SBATCH_SHA={q(sbatch_sha)}",
         f"SCRIPTS_BRANCH={q(BRANCH)}", f"EXPECT_SCRIPTS={q(scripts[:12])}"]
    if a.tier == 2:
        L += [f"K={a.k}", f"NRUNS={a.runs}", f"IDENT_N={a.ident_n}", f"STOP_ON_FALLBACK={0 if a.no_stop_on_fallback else 1}",
              f"SEP_RATE={rule_params(a)['rate_gain']}"]
    for nm, arm in (("A", armA), ("B", armB)):
        L += [f"BIN_{nm}={q(arm['bin'])}", f"COMMIT_{nm}={q(arm['commit'])}", f"ENV_{nm}={arr([f'{k}={v}' for k, v in arm['env']])}", f"ARGS_{nm}={arr(arm['args'])}"]
    return "\n".join(L) + "\n"


def sbatch_command(a, run, jobname):
    parts = ["sbatch", "--parsable", f"--job-name={jobname}", f"--export=ALL,SPEC={run}/spec.sh"]
    if a.node:
        parts.append(f"-w {a.node}")
    if a.exclude:
        parts.append(f"--exclude={a.exclude}")
    parts.append(f"{run}/t{a.tier}.sbatch")
    return f"cd {run} && " + " ".join(parts)


# --- local checks ---------------------------------------------------------------------------------------------------------------------------
def git(*args):
    return subprocess.run(["git", "-C", HERE] + list(args), capture_output=True, text=True).stdout.strip()


def local_state():
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", ".")
    remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}").split("\t")[0]
    return head, dirty, remote


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- waiting --------------------------------------------------------------------------------------------------------------------------------
def parse_state(text):
    for l in text.splitlines():
        t = l.split()
        if t and t[0].rstrip("+") in TERMINAL | {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}:
            return t[0].rstrip("+")
    return ""


def find_job(remote, jobname):
    rc, out, _ = remote.run(f"squeue -u $USER -n {jobname} -h -o %i; sacct -n -X -S now-2days --name={jobname} -o JobID | head -3")
    ids = re.findall(r"^\s*(\d+)\s*$", out, re.M)
    return ids[-1] if ids else None


def wait_for(remote, job, args, log=print):
    unknown_since = None
    last = None
    start = time.time()
    while True:
        try:
            rc, out, _ = remote.run(f"squeue -j {job} -h -o '%T' 2>/dev/null; sacct -j {job} -X -n -o State 2>/dev/null")
            state = parse_state(out)
            unknown_since = None
        except BridgeDown as e:
            state = "UNKNOWN"
            unknown_since = unknown_since or time.time()
            if time.time() - unknown_since >= args.bridge_grace * 60:
                raise
        if state != last:
            log(f"[{time.strftime('%H:%M:%S')}] job {job}: {state or 'not listed yet'}")
            last = state
        if state in TERMINAL:
            return state
        if time.time() - start > args.max_wait * 60:
            raise Fatal(f"waited {args.max_wait} min. The job {job} may still run. Resume with --resume.")
        time.sleep(args.poll)


# --- main -----------------------------------------------------------------------------------------------------------------------------------
def parser():
    p = argparse.ArgumentParser(description="Fast lane A/B driver", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--tier", type=int, choices=(1, 2))
    p.add_argument("--a", help="arm A (the baseline)")
    p.add_argument("--b", help="arm B (the idea)")
    p.add_argument("--name", help="idea name, letters digits . _ -")
    p.add_argument("--sample", default="pbmc1k_v3")
    p.add_argument("--T", type=int, default=7, help="threads per run")
    p.add_argument("--slots", type=int, default=2, help="device slots")
    p.add_argument("--order", default="ABBA", help="arm order of the legs")
    p.add_argument("--ref", help=f"reference sample directory with out/, CellReads.stats, readhash.txt. Default {DEFAULT_REF} (only for pbmc1k_v3 at T=7)")
    p.add_argument("--intended-output-change", action="store_true", help="the idea changes output on purpose. Identity loss is reported and does not reject")
    p.add_argument("--node", help="sbatch -w")
    p.add_argument("--exclude", help="sbatch --exclude (tier 2 default g050, as tp.sbatch)")
    g = p.add_argument_group("tier 1 rule")
    g.add_argument("--promote-gain", type=float, default=0.05, help="promote when core-s/M or lock held improves by at least this fraction")
    g.add_argument("--no-spread-guard", action="store_true", help="do not require the gain to exceed the run-to-run spread of the two runs per arm")
    g.add_argument("--regress-guard", type=float, help="block a promotion when the other measure worsens by more than this fraction. Off by default")
    g = p.add_argument_group("tier 2 rule and load. The rule values live in abcore.TIER2_RULE")
    g.add_argument("--k", type=int, default=8, help="runs in flight")
    g.add_argument("--runs", type=int, default=24, help="runs per leg")
    g.add_argument("--ident-n", type=int, default=4, help="runs per arm checked for identity")
    g.add_argument("--core-gain", type=float, help=f"core-s/M change that promotes or rejects. Default {abcore.TIER2_RULE['core_gain']}")
    g.add_argument("--rate-gain", type=float, help=f"progress rate change that promotes or rejects, and that stops a run early. Default {abcore.TIER2_RULE['rate_gain']}")
    g.add_argument("--lock-held-gain", type=float, help=f"lock held fall that promotes. Default {abcore.TIER2_RULE['lock_held_gain']}")
    g.add_argument("--no-stop-on-fallback", action="store_true")
    g = p.add_argument_group("running")
    g.add_argument("--dry-run", action="store_true", help="print the spec and the sbatch lines. Touch nothing")
    g.add_argument("--out", help="output directory. Default bench/gpu_fast/out/<name>-t<tier>-<time>")
    g.add_argument("--poll", type=int, default=45, help="seconds between state checks")
    g.add_argument("--max-wait", type=int, default=240, help="minutes")
    g.add_argument("--bridge-grace", type=int, default=20, help="minutes of unknown state before the tool gives up and exits with code 3")
    g.add_argument("--resume", help="out directory of an earlier run. Waits for its job and prints the verdict")
    return p


def validate(a):
    if a.tier is None or not a.a or not a.b or not a.name:
        raise Fatal("--tier, --a, --b and --name are required")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", a.name):
        raise Fatal("--name: letters, digits, . _ - and at most 40 characters")
    if not re.fullmatch(r"[AB]+", a.order) or set(a.order) != {"A", "B"}:
        raise Fatal("--order: letters A and B, both present, for example ABBA")
    if not re.fullmatch(r"[A-Za-z0-9_]+", a.sample):
        raise Fatal("--sample: plain name")
    if a.ref is not None:
        a.ref = abcore.unmsys(a.ref)
    if a.ref is None:
        if (a.sample, a.T) != ("pbmc1k_v3", 7):
            raise Fatal("--ref is needed unless the sample is pbmc1k_v3 at T=7")
        a.ref = DEFAULT_REF
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", a.ref):
        raise Fatal("--ref: absolute path of plain characters")
    if a.node and a.exclude:
        raise Fatal("--node and --exclude together")
    if a.tier == 2 and a.exclude is None and not a.node:
        a.exclude = "g050"
    if a.k < 1 or a.runs <= 2 * a.k:
        print(f"note: runs per leg ({a.runs}) is not above 2k ({2 * a.k}). The windowed rate will be NA.", file=sys.stderr)


def main(argv=None, remote=None, out_fn=print):
    a = parser().parse_args(argv)
    if a.resume:
        return resume(a, remote or Remote(), out_fn)
    try:
        validate(a)
        armA, armB = abcore.parse_arm(a.a), abcore.parse_arm(a.b)
    except (Fatal, abcore.SpecError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    notes = abcore.arm_notes(armA, "A") + abcore.arm_notes(armB, "B")
    if a.intended_output_change:
        notes.append("an output change was declared. Identity loss is shown and does not reject.")
    if a.tier == 2:
        notes.append("tier 2 uses the provisional rule " + json.dumps(rule_params(a)) + ". Span rate is shown but never decides.")
    for n in notes:
        print(n, file=sys.stderr)
    head, dirty, rem = local_state()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = f"{W}/jobs/gpu-fast/runs/{a.name}-t{a.tier}-{stamp}"
    jobname = f"gf{a.tier}-{a.name}"[:40]
    sbatch_local = os.path.join(HERE, "jobs", f"t{a.tier}.sbatch")
    spec = build_spec(a, armA, armB, run, head, sha256_file(sbatch_local))
    cmd = sbatch_command(a, run, jobname)
    problems = []
    if not head or len(head) != 40:
        problems.append("no git head")
    if dirty:
        problems.append("uncommitted changes under bench/gpu_fast")
    if rem != head:
        problems.append(f"HEAD {head[:8]} is not the tip of origin/{BRANCH} ({rem[:8] or 'missing'}). Push it first")
    if a.dry_run:
        print(f"# dry run. tier {a.tier}. Nothing is sent.")
        print(f"# scripts commit {head[:12]}" + ("" if not problems else f". A real run would refuse: {'; '.join(problems)}"))
        print(f"# remote run directory {run}")
        print("# --- spec.sh ---")
        print(spec, end="")
        print("# --- commands ---")
        print(f"clipper put {sbatch_local} {run}/t{a.tier}.sbatch")
        print(f"clipper put <spec.sh> {run}/spec.sh")
        print(f"clipper run {shlex.quote(cmd)}")
        print(f"# log {W}/logs/{jobname}-<jobid>.out   result {run}/res/result.json")
        return 0
    if problems:
        print("error: " + "; ".join(problems), file=sys.stderr)
        return 2
    remote = remote or Remote()
    if not remote.status():
        print("error: the Clipper bridge is not connected. This tool does not start it. Start it yourself and run again.", file=sys.stderr)
        return 2
    out = a.out or os.path.join(HERE, "out", f"{a.name}-t{a.tier}-{stamp}")
    os.makedirs(out, exist_ok=True)
    state = {"name": a.name, "tier": a.tier, "run": run, "jobname": jobname, "job": None, "scripts": head, "args": argv if argv is not None else sys.argv[1:],
             "notes": notes, "submitted": None, "ref": a.ref, "intended": a.intended_output_change,
             "rule_params": rule_params(a)}
    save_state(out, state)
    open(os.path.join(out, "spec.sh"), "w").write(spec)
    try:
        remote.run(f"mkdir -p {run}")
        remote.put(sbatch_local, f"{run}/t{a.tier}.sbatch")
        remote.put(os.path.join(out, "spec.sh"), f"{run}/spec.sh")
        rc, so, se = remote.run(cmd)
    except BridgeDown as e:
        print(f"the bridge dropped during the submission ({e}). Job state unknown. Resume with: python {sys.argv[0]} --resume {out}", file=sys.stderr)
        return 3
    m = re.search(r"^(\d+)", so.strip())
    if not m:
        print(f"error: sbatch did not return a job id: {so.strip()[:300]} {se.strip()[:300]}", file=sys.stderr)
        return 2
    state["job"], state["submitted"] = m.group(1), time.time()
    save_state(out, state)
    out_fn(f"submitted job {state['job']} ({jobname}). Run directory {run}")
    return finish(out, state, a, remote, out_fn)


def rule_params(a):
    if a.tier == 1:
        return {"promote_gain": a.promote_gain, "respect_spread": not a.no_spread_guard, "regress_guard": a.regress_guard}
    r = dict(abcore.TIER2_RULE)
    for k, v in (("core_gain", a.core_gain), ("rate_gain", a.rate_gain), ("lock_held_gain", a.lock_held_gain)):
        if v is not None:
            r[k] = v
    return r


def save_state(out, state):
    json.dump(state, open(os.path.join(out, "state.json"), "w"), indent=1)


def resume(a, remote, out_fn):
    out = a.resume
    state = json.load(open(os.path.join(out, "state.json")))
    if not remote.status():
        print("error: the Clipper bridge is not connected. This tool does not start it.", file=sys.stderr)
        return 2
    try:
        if not state.get("job"):
            state["job"] = find_job(remote, state["jobname"])
            if not state["job"]:
                print(f"no job named {state['jobname']} found on the cluster. The submission did not happen.", file=sys.stderr)
                return 2
            save_state(out, state)
    except BridgeDown:
        print("the bridge is down again. Job state unknown.", file=sys.stderr)
        return 3
    a.tier, a.name = state["tier"], state["name"]
    return finish(out, state, a, remote, out_fn)


def finish(out, state, a, remote, out_fn):
    job, run = state["job"], state["run"]
    try:
        final = wait_for(remote, job, a, out_fn)
        res, tail = fetch_result(remote, run, state["jobname"], job)
    except BridgeDown as e:
        print(f"the bridge has been down for more than {a.bridge_grace} min ({e}). Job {job} state is UNKNOWN. "
              f"Resume with: python {os.path.relpath(__file__)} --resume {out}", file=sys.stderr)
        return 3
    except Fatal as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if res is None:
        v = {"verdict": "INCONCLUSIVE", "rule": "J1", "why": f"job {job} ended as {final} and left no result.json", "tier": state["tier"], "params": {}, "log_tail": tail}
        json.dump(v, open(os.path.join(out, "verdict.json"), "w"), indent=1)
        print(f"VERDICT: INCONCLUSIVE (rule J1) job {job} ended as {final} and left no result.json\n{tail}")
        return 0
    return report(out, state, res, final, out_fn)


def fetch_result(remote, run, jobname, job):
    rc, so, se = remote.run(f"cat {run}/res/result.json 2>/dev/null")
    if rc == 0 and so.strip().startswith("{"):
        return json.loads(so), ""
    rc, so, se = remote.run(f"cat {run}/res/REFUSED 2>/dev/null; tail -n 25 {W}/logs/{jobname}-{job}.out 2>/dev/null")
    return None, so.strip()


def report(out, state, res, final, out_fn=print):
    rp = state["rule_params"]
    intended = bool(state.get("intended"))
    if state["tier"] == 1:
        v = abcore.tier1_verdict(res, rp, intended)
    else:
        v = abcore.tier2_verdict(res, rp, intended)
    if final != "COMPLETED":
        v.setdefault("notes", []).append(f"job ended as {final}")
    table = abcore.render(res, v, state.get("notes", []) + ([f"job ended as {final}"] if final != "COMPLETED" else []))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    open(os.path.join(out, "table.md"), "w").write(table + "\n")
    vj = {"verdict": v["verdict"], "rule": v["rule"], "why": v["why"], "tier": v["tier"], "params": v["params"], "gain": v.get("gain"), "spread": v.get("spread"),
          "name": state["name"], "job": state["job"], "run_dir": state["run"], "scripts": state["scripts"], "job_state": final,
          "A": strip(v["A"]), "B": strip(v["B"])}
    json.dump(vj, open(os.path.join(out, "verdict.json"), "w"), indent=1)
    out_fn(table)
    out_fn(f"\nfiles in {out}: table.md verdict.json result.json")
    return 0


def strip(s):
    return {k: val for k, val in s.items() if k != "fallback_lines"}


if __name__ == "__main__":
    sys.exit(main())
