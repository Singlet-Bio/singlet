"""Run a dependency graph of tasks inside one whole-node Slurm job (Clipper rule: one node at a time, every core used).

Task file, tab-separated: name, group, cpus, mem_gb, after (comma list or -), command. Tasks start in file order whenever their
dependencies have succeeded and the node has the cores and memory; --limit GROUP=N caps concurrent tasks of a group (downloads).
A task that succeeded writes STATE/<name>.ok and is skipped on the next run, so the same job can be resubmitted (afterany) to
continue past the partition's time limit. A failed task blocks only its dependents. Logs: LOGS/<name>.log.
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--state", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--cpus", type=int, default=os.cpu_count())
    ap.add_argument("--mem", type=int, required=True, help="GB available to tasks")
    ap.add_argument("--limit", action="append", default=[])
    ap.add_argument("--stop-after", type=float, default=0, help="seconds; start no new task after this")
    a = ap.parse_args()
    os.makedirs(a.state, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    limits = {k: int(v) for k, v in (x.split("=") for x in a.limit)}
    tasks = []
    for line in open(a.tasks):
        if not line.strip() or line.startswith("#"):
            continue
        name, group, cpus, mem, after, cmd = line.rstrip("\n").split("\t", 5)
        tasks.append(dict(name=name, group=group, cpus=min(int(cpus), a.cpus), mem=int(mem),
                          after=[] if after == "-" else after.split(","), cmd=cmd))
    done = {t["name"] for t in tasks if os.path.exists(os.path.join(a.state, t["name"] + ".ok"))}
    failed, running = set(), {}
    pending = [t for t in tasks if t["name"] not in done]
    t0 = time.time()
    print(f"[node_run] {len(tasks)} tasks, {len(done)} done already, {a.cpus} cpus, {a.mem} GB", flush=True)
    while pending or running:
        for n, (p, t) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            if rc == 0:
                done.add(n)
                open(os.path.join(a.state, n + ".ok"), "w").write(time.strftime("%FT%T\n"))
            else:
                failed.add(n)
            print(f"[node_run] {time.strftime('%T')} {n} exit {rc} ({len(done)} ok, {len(failed)} failed, {len(running)} running)", flush=True)
        blocked = [t for t in pending if any(d in failed for d in t["after"])]
        for t in blocked:
            failed.add(t["name"])
            pending.remove(t)
            print(f"[node_run] {t['name']} skipped: a dependency failed", flush=True)
        late = a.stop_after and time.time() - t0 > a.stop_after
        used_c = sum(t["cpus"] for _, t in running.values())
        used_m = sum(t["mem"] for _, t in running.values())
        for t in list(pending):
            if late:
                break
            if not all(d in done for d in t["after"]):
                continue
            g = t["group"]
            if g in limits and sum(1 for _, r in running.values() if r["group"] == g) >= limits[g]:
                continue
            if used_c + t["cpus"] > a.cpus or used_m + t["mem"] > a.mem:
                continue
            log = open(os.path.join(a.logs, t["name"] + ".log"), "a")
            env = dict(os.environ, TASK_CPUS=str(t["cpus"]), TASK_NAME=t["name"])
            running[t["name"]] = (subprocess.Popen(["bash", "-c", t["cmd"]], stdout=log, stderr=subprocess.STDOUT, env=env), t)
            pending.remove(t)
            used_c += t["cpus"]
            used_m += t["mem"]
            print(f"[node_run] {time.strftime('%T')} start {t['name']} ({t['cpus']} cpus)", flush=True)
        if late and not running:
            break
        time.sleep(5)
    print(f"[node_run] finished: {len(done)} ok, {len(failed)} failed, {len(pending)} not started: {sorted(failed)}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
