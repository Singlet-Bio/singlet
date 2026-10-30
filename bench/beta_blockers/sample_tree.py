"""Sample the memory of a process tree: sample_tree.py <pid> <out.tsv>  (until <pid> exits).

Per sample, over <pid> and all its descendants, from /proc/<p>/smaps_rollup: Rss, Pss, Private (clean + dirty), Shared
(clean + dirty), in MB. Private is what one more job of the same kind costs; Shared (shared genome segment, memory-mapped
index, libraries) is paid once per node. Prints the peaks on exit.
"""
import os
import sys
import time


def kids(pid, tab):
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(tab.get(p, ()))
    return out


def children_table():
    tab = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % d) as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        tab.setdefault(ppid, []).append(int(d))
    return tab


def rollup(p):
    r = {"Rss": 0, "Pss": 0, "Private_Clean": 0, "Private_Dirty": 0, "Shared_Clean": 0, "Shared_Dirty": 0}
    try:
        with open("/proc/%d/smaps_rollup" % p) as fh:
            for ln in fh:
                k = ln.split(":")[0]
                if k in r:
                    r[k] = int(ln.split()[1])
    except OSError:
        pass
    return r


def main():
    pid, out = int(sys.argv[1]), sys.argv[2]
    peak = {"rss": 0, "pss": 0, "private": 0, "shared": 0}
    t0 = time.time()
    with open(out, "w") as fh:
        fh.write("t_s\trss_mb\tpss_mb\tprivate_mb\tshared_mb\tnproc\n")
        while os.path.exists("/proc/%d" % pid):
            ps = kids(pid, children_table())
            tot = {"Rss": 0, "Pss": 0, "Private_Clean": 0, "Private_Dirty": 0, "Shared_Clean": 0, "Shared_Dirty": 0}
            for p in ps:
                for k, v in rollup(p).items():
                    tot[k] += v
            row = {"rss": tot["Rss"] / 1024, "pss": tot["Pss"] / 1024,
                   "private": (tot["Private_Clean"] + tot["Private_Dirty"]) / 1024,
                   "shared": (tot["Shared_Clean"] + tot["Shared_Dirty"]) / 1024}
            for k in peak:
                peak[k] = max(peak[k], row[k])
            fh.write("%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%d\n" % (time.time() - t0, row["rss"], row["pss"], row["private"],
                                                              row["shared"], len(ps)))
            fh.flush()
            time.sleep(2)
    print("PEAK_MB rss=%.0f pss=%.0f private=%.0f shared=%.0f" % (peak["rss"], peak["pss"], peak["private"], peak["shared"]))


main()
