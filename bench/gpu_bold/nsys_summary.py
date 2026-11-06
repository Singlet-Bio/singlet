#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export of one singlet run (gpu/bold step 3).

Usage: nsys_summary.py REPORT.sqlite [--top N] [--csv-prefix PATH]

Prints:
  1. a kernel table: launches, device seconds, mean duration, mean blocks per launch, threads per block, registers, shared memory,
     theoretical occupancy (H100 limits), share of kernel time
  2. device activity: first and last kernel, kernel busy time as the union of all kernel intervals, memcpy busy time, their overlap,
     idle time inside the active window, and the idle gaps by size
  3. copies by kind (bytes, seconds, GB/s)
  4. GPU metrics from the sampler (device wide, so other processes count): mean SM active and so on over the whole trace and over the
     time a kernel of this process was running
Only reads the file. Writes CSV copies of the tables when --csv-prefix is given.
"""
import argparse
import bisect
import collections
import csv
import sqlite3
import sys

SM_COUNT = 132           # H100 NVL
MAX_THREADS_SM = 2048
MAX_WARPS_SM = 64
MAX_BLOCKS_SM = 32
REGS_SM = 65536
SMEM_SM = 228 * 1024     # bytes configurable per SM (227 KB per block)
REG_ALLOC_UNIT = 256     # registers per warp allocation granularity
SMEM_ALLOC_UNIT = 128


def occupancy(threads, regs, smem):
    """theoretical active warps / max warps per SM for one block shape (CUDA occupancy rules, simplified)"""
    if threads <= 0:
        return float("nan")
    warps = (threads + 31) // 32
    lim = [MAX_BLOCKS_SM, MAX_WARPS_SM // warps]
    if regs > 0:
        rw = ((regs * 32 + REG_ALLOC_UNIT - 1) // REG_ALLOC_UNIT) * REG_ALLOC_UNIT   # regs per warp, rounded
        lim.append(REGS_SM // (rw * warps))
    if smem > 0:
        sm = ((smem + SMEM_ALLOC_UNIT - 1) // SMEM_ALLOC_UNIT) * SMEM_ALLOC_UNIT + 1024   # 1 KB reserved per block
        lim.append(SMEM_SM // sm)
    blocks = max(0, min(lim))
    return 100.0 * blocks * warps / MAX_WARPS_SM


def union(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def total(u):
    return sum(b - a for a, b in u)


def inter(u, v):
    i = j = 0
    s = 0
    while i < len(u) and j < len(v):
        a = max(u[i][0], v[j][0])
        b = min(u[i][1], v[j][1])
        if b > a:
            s += b - a
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return s


def cols(con, t):
    return [r[1] for r in con.execute(f"PRAGMA table_info({t})")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--csv-prefix", default=None)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    print("tables:", ", ".join(sorted(t for t in tabs if "CUPTI" in t or "GPU" in t or "STRING" in t.upper() or "NVTX" in t)))
    strings = dict(con.execute("select id, value from StringIds"))
    kc = cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    need = ["start", "end", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory", "dynamicSharedMemory"]
    namecol = "demangledName" if "demangledName" in kc else "shortName"
    sel = ", ".join(need + [namecol, "globalPid" if "globalPid" in kc else "0", "streamId", "deviceId"])
    rows = con.execute(f"select {sel} from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    print(f"kernel launches: {len(rows)}")
    if not rows:
        return
    t0 = min(r[0] for r in rows)
    t1 = max(r[1] for r in rows)
    agg = collections.OrderedDict()
    kiv = []
    for r in rows:
        s, e, gx, gy, gz, bx, by, bz, regs, ss, ds, nm, pid, strm, dev = r
        nm = strings.get(nm, str(nm))
        short = nm.split("(")[0]
        short = short.replace("void ", "")[:70]
        key = (short, gx * gy * gz, bx * by * bz, regs, ss + ds)
        g = agg.setdefault(short, {"n": 0, "t": 0, "blocks": 0, "threads": collections.Counter(), "regs": collections.Counter(), "smem": collections.Counter(), "streams": set()})
        g["n"] += 1
        g["t"] += e - s
        g["blocks"] += gx * gy * gz
        g["threads"][bx * by * bz] += 1
        g["regs"][regs] += 1
        g["smem"][ss + ds] += 1
        g["streams"].add(strm)
        kiv.append([s, e])
    ku = union(kiv)
    kbusy = total(ku) / 1e9
    ktime = sum(g["t"] for g in agg.values()) / 1e9
    span = (t1 - t0) / 1e9
    print(f"\nkernels: first {0:.2f} s, last {span:.2f} s after the first kernel")
    print(f"kernel time summed over launches {ktime:.2f} s, union (device busy with kernels) {kbusy:.2f} s, "
          f"idle inside the window {span - kbusy:.2f} s ({100 * (span - kbusy) / span:.0f}%), streams used {len({r[-2] for r in rows})}")
    print("\n| kernel | launches | device s | share % | mean us | mean blocks | threads/block | regs | smem B | theor. occupancy % | waves of 132 SMs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    outrows = []
    for nm, g in sorted(agg.items(), key=lambda kv: -kv[1]["t"])[: a.top]:
        th = g["threads"].most_common(1)[0][0]
        rg = g["regs"].most_common(1)[0][0]
        sm = g["smem"].most_common(1)[0][0]
        occ = occupancy(th, rg, sm)
        mb = g["blocks"] / g["n"]
        blocks_per_sm = max(1, int(occ / 100 * MAX_WARPS_SM / max(1, (th + 31) // 32))) if occ == occ else 1
        waves = mb / (SM_COUNT * blocks_per_sm)
        row = (nm, g["n"], g["t"] / 1e9, 100 * g["t"] / 1e9 / ktime, g["t"] / g["n"] / 1e3, mb, th, rg, sm, occ, waves)
        outrows.append(row)
        print(f"| {nm} | {g['n']} | {row[2]:.3f} | {row[3]:.1f} | {row[4]:.1f} | {row[5]:.0f} | {th} | {rg} | {sm} | {occ:.0f} | {waves:.2f} |")
    # time-weighted occupancy over all kernels
    wo = 0.0
    for nm, g in agg.items():
        th = g["threads"].most_common(1)[0][0]
        rg = g["regs"].most_common(1)[0][0]
        sm = g["smem"].most_common(1)[0][0]
        o = occupancy(th, rg, sm)
        if o == o:
            wo += o * g["t"]
    print(f"\ntime-weighted theoretical occupancy over all kernels: {wo / sum(g['t'] for g in agg.values()):.0f}%")

    # memcpy
    mc = con.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall() if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else []
    kinds = {1: "H2D", 2: "D2H", 3: "H2A", 4: "A2H", 5: "A2A", 6: "A2D", 7: "D2A", 8: "D2D", 9: "H2H", 10: "P2P"}
    ms = con.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMSET").fetchall() if "CUPTI_ACTIVITY_KIND_MEMSET" in tabs else []
    civ = [[r[0], r[1]] for r in mc]
    cu = union(civ)
    mu = union([[r[0], r[1]] for r in ms])
    allu = union(ku + cu + mu)
    print(f"\ncopies: {len(mc)} memcpy, {len(ms)} memset; memcpy union {total(cu) / 1e9:.2f} s, memset union {total(mu) / 1e9:.2f} s; "
          f"copy busy while a kernel runs {inter(cu, ku) / 1e9:.2f} s; copy-only time {(total(cu) - inter(cu, ku)) / 1e9:.2f} s")
    wa0, wa1 = min(allu[0][0], t0), max(allu[-1][1], t1)
    act = total(allu) / 1e9
    print(f"all device activity (kernels, copies, sets) union {act:.2f} s of a {(wa1 - wa0) / 1e9:.2f} s window")
    print("\n| copy kind | count | GB | summed s | GB/s (bytes over summed time) |")
    print("|---|---|---|---|---|")
    byk = collections.defaultdict(lambda: [0, 0, 0])
    for s, e, b, k in mc:
        x = byk[kinds.get(k, str(k))]
        x[0] += 1
        x[1] += b
        x[2] += e - s
    for k, (n, b, t) in sorted(byk.items(), key=lambda kv: -kv[1][2]):
        print(f"| {k} | {n} | {b / 2**30:.2f} | {t / 1e9:.2f} | {b / 2**30 / (t / 1e9):.1f} |")

    # idle gaps between kernel/copy activity
    gaps = [(allu[i + 1][0] - allu[i][1]) / 1e9 for i in range(len(allu) - 1)]
    bins = [(0, 1e-4, "<0.1 ms"), (1e-4, 1e-3, "0.1-1 ms"), (1e-3, 1e-2, "1-10 ms"), (1e-2, 0.1, "10-100 ms"), (0.1, 1, "0.1-1 s"), (1, 1e9, ">1 s")]
    print("\n| idle gap between device activity | count | summed s |")
    print("|---|---|---|")
    for lo, hi, lab in bins:
        g = [x for x in gaps if lo <= x < hi]
        print(f"| {lab} | {len(g)} | {sum(g):.2f} |")
    big = sorted(((allu[i + 1][0] - allu[i][1]) / 1e9, (allu[i][1] - wa0) / 1e9) for i in range(len(allu) - 1))[-12:]
    print("largest idle gaps (length s, starts at s after window start):", ", ".join(f"{l:.2f}@{s:.1f}" for l, s in reversed(big)))

    # per second kernel busy fraction
    nb = int((wa1 - wa0) / 1e9) + 1
    busy = [0.0] * nb
    for s, e in ku:
        x = s
        while x < e:
            b = int((x - wa0) / 1e9)
            y = min(e, wa0 + (b + 1) * 1e9)
            busy[b] += (y - x) / 1e9
            x = y
    print("\nkernel busy fraction per second (percent):", " ".join(f"{100 * b:.0f}" for b in busy))

    # GPU metrics (device wide)
    if "GPU_METRICS" in tabs and "TARGET_INFO_GPU_METRICS" in tabs:
        names = dict(con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
        gm = collections.defaultdict(list)
        for ts, mid, val in con.execute("select timestamp, metricId, value from GPU_METRICS order by timestamp"):
            gm[mid].append((ts, val))
        print("\nGPU metrics sampler (device wide, includes other processes on the GPU):")
        print("| metric | mean whole trace | mean while this process runs a kernel | max |")
        print("|---|---|---|---|")
        kstarts = [u[0] for u in ku]
        for mid, ser in sorted(gm.items(), key=lambda kv: names.get(kv[0], "")):
            vals = [v for _, v in ser]
            inw = []
            for ts, v in ser:
                i = bisect.bisect_right(kstarts, ts) - 1
                if i >= 0 and ku[i][0] <= ts <= ku[i][1]:
                    inw.append(v)
            print(f"| {names.get(mid, mid)} | {sum(vals) / len(vals):.1f} | {(sum(inw) / len(inw)) if inw else float('nan'):.1f} | {max(vals):.1f} |")
    else:
        print("\nno GPU_METRICS table in this export")

    if a.csv_prefix:
        with open(a.csv_prefix + "_kernels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "launches", "device_s", "share_pct", "mean_us", "mean_blocks", "threads_per_block", "regs", "smem_B", "theor_occ_pct", "waves"])
            for r in outrows:
                w.writerow(r)
        with open(a.csv_prefix + "_busy_per_s.csv", "w") as f:
            f.write("second,kernel_busy_frac\n")
            for i, b in enumerate(busy):
                f.write(f"{i},{b:.4f}\n")


if __name__ == "__main__":
    main()
