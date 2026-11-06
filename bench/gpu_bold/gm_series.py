#!/usr/bin/env python3
"""Per-second device metrics from an nsys SQLite export (GPU_METRICS, sampled device wide, so it includes the STAR child that nsys did not trace).
Usage: gm_series.py REPORT.sqlite [OUT.csv]
Prints one line per second (SMs active, warps in flight, unallocated warps in active SMs, SM issue, GR active, DRAM read and write, PCIe rx and tx),
then the share of seconds by SM-active band, and the mean over the seconds where GR Active is above 50 percent."""
import collections
import sqlite3
import sys

con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
names = dict(con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
want = {"SMs Active [Throughput %]": "sm_act", "Compute Warps in Flight [Throughput %]": "warps", "Unallocated Warps in Active SMs [Throughput %]": "unalloc",
        "SM Issue [Throughput %]": "issue", "GR Active [Throughput %]": "gr", "DRAM Read Bandwidth [Throughput %]": "dram_r",
        "DRAM Write Bandwidth [Throughput %]": "dram_w", "PCIe RX Throughput [Throughput %]": "pcie_rx", "PCIe TX Throughput [Throughput %]": "pcie_tx"}
ids = {mid: want[n] for mid, n in names.items() if n in want}
rows = con.execute("select timestamp, metricId, value from GPU_METRICS order by timestamp").fetchall()
t0 = rows[0][0]
b = collections.defaultdict(lambda: collections.defaultdict(list))
for ts, mid, v in rows:
    if mid in ids:
        b[int((ts - t0) / 1e9)][ids[mid]].append(v)
cols = ["sm_act", "warps", "unalloc", "issue", "gr", "dram_r", "dram_w", "pcie_rx", "pcie_tx"]
mean = lambda x: sum(x) / len(x) if x else 0.0
print("second," + ",".join(cols))
series = []
for s in sorted(b):
    r = [mean(b[s][c]) for c in cols]
    series.append(r)
    print(f"{s}," + ",".join(f"{x:.1f}" for x in r))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        f.write("second," + ",".join(cols) + "\n")
        for i, r in enumerate(series):
            f.write(f"{i}," + ",".join(f"{x:.2f}" for x in r) + "\n")
n = len(series)
print(f"\nseconds {n}")
for lo, hi in ((0, 5), (5, 25), (25, 50), (50, 75), (75, 101)):
    k = sum(1 for r in series if lo <= r[0] < hi)
    print(f"SM active {lo}-{hi}%: {k} s ({100 * k / n:.0f}%)")
act = [r for r in series if r[4] > 50]
if act:
    print(f"mean over the {len(act)} s with GR active above 50%: " + ", ".join(f"{c} {mean([r[i] for r in act]):.1f}" for i, c in enumerate(cols)))
print("mean over all seconds: " + ", ".join(f"{c} {mean([r[i] for r in series]):.1f}" for i, c in enumerate(cols)))
