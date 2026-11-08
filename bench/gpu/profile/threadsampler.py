#!/usr/bin/env python3
"""Per-thread CPU sampler for nodes where perf is unavailable (Clipper: perf_event_paranoid 4, so nsys cannot sample CPUs).

sample: every INTERVAL s, read /proc/<pid>/task/<tid>/stat for every process in a cgroup (default: this job step's own cgroup) and append
        the CPU time each thread used since the last sample: one TSV row per (sample, thread) that used any.
          threadsampler.py sample --out FILE.tsv [--cgroup DIR] [--interval 0.5] [--stop-file PATH]
        Stops on SIGTERM/SIGINT or when the stop file appears. Overhead: one pass over /proc per interval (about 1 ms per 100 threads).
summary: core-seconds by process name and thread name, overall and inside time windows given as start:end in seconds since the first
        sample (or as NVTX phase rows from nvtx_phases.py with --phases TSV --epoch the run's start in epoch seconds).
          threadsampler.py summary FILE.tsv [--window NAME=START:END ...] [--top 25]

Row format: t (epoch s), pid, tid, process name (the thread group leader's comm), thread name (comm), ppid, user_s, sys_s.
Thread names are what the program sets with pthread_setname_np; unnamed threads carry the process name, and the tid tells them apart.
"""
import argparse, collections, os, signal, sys, time

HZ = os.sysconf("SC_CLK_TCK")


def own_cgroup():
    for line in open("/proc/self/cgroup"):
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return "/sys/fs/cgroup" + parts[2]
    raise SystemExit("no cgroup v2 path in /proc/self/cgroup; pass --cgroup")


def read_stat(path):
    s = open(path).read()
    # comm may hold spaces and parentheses: it is the text between the first "(" and the last ")"
    l, r = s.index("("), s.rindex(")")
    comm, rest = s[l + 1:r], s[r + 2:].split()
    return comm, int(rest[1]), int(rest[11]), int(rest[12])   # comm, ppid, utime, stime (clock ticks)


def pids_of(cg):
    pids = set()
    for root, _dirs, files in os.walk(cg):
        if "cgroup.procs" in files:
            try:
                pids.update(int(x) for x in open(os.path.join(root, "cgroup.procs")).read().split())
            except OSError:
                pass
    return pids


def sample(a):
    cg = a.cgroup or own_cgroup()
    me = os.getpid()
    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))
    last = {}
    with open(a.out, "w", buffering=1) as f:
        f.write(f"# cgroup {cg} interval {a.interval} hz {HZ}\n")
        f.write("t\tpid\ttid\tprocess\tthread\tppid\tuser_s\tsys_s\n")
        while not stop["now"] and not (a.stop_file and os.path.exists(a.stop_file)):
            t = time.time()
            for pid in pids_of(cg):
                if pid == me:
                    continue
                try:
                    pname = read_stat(f"/proc/{pid}/stat")[0]
                    tids = os.listdir(f"/proc/{pid}/task")
                except (OSError, ValueError):
                    continue
                for tid in tids:
                    try:
                        comm, ppid, ut, st = read_stat(f"/proc/{pid}/task/{tid}/stat")
                    except (OSError, ValueError):
                        continue
                    key = (pid, int(tid))
                    u0, s0 = last.get(key, (0, 0))
                    last[key] = (ut, st)
                    du, ds = ut - u0, st - s0
                    if du or ds:
                        f.write(f"{t:.3f}\t{pid}\t{tid}\t{pname}\t{comm}\t{ppid}\t{du / HZ:.2f}\t{ds / HZ:.2f}\n")
            time.sleep(max(0.0, a.interval - (time.time() - t)))


def summary(a):
    rows = []
    for line in open(a.tsv):
        if line.startswith("#") or line.startswith("t\t"):
            continue
        t, pid, tid, pname, comm, ppid, us, ss = line.rstrip("\n").split("\t")
        rows.append((float(t), int(pid), int(tid), pname, comm, float(us) + float(ss), float(ss)))
    if not rows:
        raise SystemExit("no samples")
    t0 = min(r[0] for r in rows)
    wins = [("all", -1e18, 1e18)]
    for w in a.window:
        name, span = w.split("=", 1)
        s, e = span.split(":")
        wins.append((name, float(s), float(e)))
    if a.phases:
        import csv
        for r in csv.DictReader(open(a.phases), delimiter="\t"):
            s = float(r["first_start_s"]) + a.epoch - t0
            wins.append((f'{r["process"]}: {r["range"]}', s, float(r["last_end_s"]) + a.epoch - t0))
    for name, s, e in wins:
        sel = [r for r in rows if s <= r[0] - t0 <= e]
        tot = sum(r[5] for r in sel)
        if not sel:
            continue
        span = (max(r[0] for r in sel) - min(r[0] for r in sel)) or 1.0
        print(f"## {name}: {tot:.1f} core-s over {span:.1f} s ({tot / span:.1f} cores), system {sum(r[6] for r in sel):.1f}")
        by = collections.defaultdict(lambda: [0.0, set()])
        for r in sel:
            k = (r[3], r[4], r[1])
            by[k][0] += r[5]; by[k][1].add(r[2])
        print("| process | pid | thread name | threads | core-s | share % |")
        print("|---|---|---|---|---|---|")
        for (pname, comm, pid), (cs, tids) in sorted(by.items(), key=lambda kv: -kv[1][0])[:a.top]:
            print(f"| {pname} | {pid} | {comm} | {len(tids)} | {cs:.1f} | {100 * cs / tot:.1f} |")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("sample")
p.add_argument("--out", required=True)
p.add_argument("--cgroup")
p.add_argument("--interval", type=float, default=0.5)
p.add_argument("--stop-file")
p = sub.add_parser("summary")
p.add_argument("tsv")
p.add_argument("--window", action="append", default=[])
p.add_argument("--phases")
p.add_argument("--epoch", type=float, default=0.0)
p.add_argument("--top", type=int, default=25)
a = ap.parse_args()
sample(a) if a.cmd == "sample" else summary(a)
