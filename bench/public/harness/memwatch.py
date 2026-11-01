"""Sample the resident set of a process tree once per second until the root process exits.

    python memwatch.py ROOT_PID OUT.json

Writes {"peak_rss_gb": <max over time of the RSS summed over the root and all its descendants>,
        "peak_proc_rss_gb": <largest single process>, "samples": n}.
RSS is read from /proc/<pid>/status (VmRSS). Pages shared between processes are counted once per process; that
overestimates only when a genome is mapped by several processes, which is not the case for the runs here (one STAR per
job). Standard library only.
"""
import json
import os
import sys
import time


def children_map():
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as fh:
                data = fh.read().decode("utf-8", "replace")
            ppid = int(data[data.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    return kids


def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return 0


def main():
    root, out = int(sys.argv[1]), sys.argv[2]
    peak = peak_one = n = 0
    while os.path.exists(f"/proc/{root}"):
        kids = children_map()
        stack, seen, tot, one = [root], set(), 0, 0
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            r = rss_kb(p)
            tot += r
            one = max(one, r)
            stack.extend(kids.get(p, []))
        peak, peak_one, n = max(peak, tot), max(peak_one, one), n + 1
        time.sleep(1.0)
    with open(out, "w") as fh:
        json.dump({"peak_rss_gb": peak / 1024 ** 2, "peak_proc_rss_gb": peak_one / 1024 ** 2, "samples": n}, fh)


if __name__ == "__main__":
    main()
