#!/usr/bin/env python3
"""Trim an ncu `--page source --print-source sass --csv` export to the hot SASS lines.
Usage: ncu_source_trim.py IN.csv > OUT.txt
Per profiled kernel block: totals (warp stall samples, warp instructions executed, thread instructions executed, mean active threads per warp instruction,
divergent branches, local-memory L2 sectors, global L1 tag requests), summed stall reasons, the 40 SASS lines with the most stall samples, the 15 with the
most divergent branches and the 15 with the most local-memory sectors."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1], encoding="utf-8", errors="replace")))
blocks = []
i = 0
while i < len(rows):
    r = rows[i]
    if r and r[0] == "Kernel Name":
        blocks.append([r[1], None, []])
        i += 1
        continue
    if r and r[0] == "Address":
        blocks[-1][1] = r
        i += 1
        continue
    if blocks and blocks[-1][1] is not None and r:
        blocks[-1][2].append(r)
    i += 1


def num(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


for name, hdr, data in blocks:
    if hdr is None:
        continue
    ix = {h: k for k, h in enumerate(hdr)}
    col = lambda n: ix.get(n)
    S, IE, TIE, AVG, DIV, LOC, G1 = (col("Warp Stall Sampling (All Samples)"), col("Instructions Executed"), col("Thread Instructions Executed"),
                                     col("Avg. Threads Executed"), col("Divergent Branches"), col("L2 Theoretical Sectors Local"), col("L1 Tag Requests Global"))
    stalls = [k for k, h in enumerate(hdr) if h.startswith("stall_")]
    g = lambda r, k: num(r[k]) if k is not None and k < len(r) else 0.0
    tot = lambda k: sum(g(r, k) for r in data)
    ts, ie, tie = tot(S), tot(IE), tot(TIE)
    print("=" * 120)
    print(name[:200])
    print(f"SASS lines {len(data)}; warp stall samples {ts:.0f}; warp instructions executed {ie:.0f}; thread instructions executed {tie:.0f}; "
          f"mean active threads per warp instruction {tie / ie if ie else 0:.2f} of 32; divergent branches {tot(DIV):.0f}; "
          f"local-memory L2 sectors {tot(LOC):.0f}; global L1 tag requests {tot(G1):.0f}")
    st = sorted(((sum(g(r, k) for r in data), hdr[k]) for k in stalls), reverse=True)
    print("stall reasons (samples): " + ", ".join(f"{n[6:]} {v:.0f} ({100 * v / ts:.0f}%)" for v, n in st[:8] if v > 0) if ts else "no stall samples")

    def show(title, key, n):
        print(f"\n{title}")
        print("| address | SASS | samples | % samples | warp inst | avg threads | divergent br | local sectors | top stall |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(data, key=lambda r: -g(r, key))[:n]:
            if g(r, key) <= 0:
                break
            top = max(stalls, key=lambda k: g(r, k)) if stalls else None
            ts_ = hdr[top][6:] if top is not None and g(r, top) > 0 else ""
            print(f"| {r[0][-6:]} | {r[1].strip()[:70]} | {g(r, S):.0f} | {100 * g(r, S) / ts if ts else 0:.1f} | {g(r, IE):.0f} | {g(r, AVG):.1f} | {g(r, DIV):.0f} | {g(r, LOC):.0f} | {ts_} |")

    show("top 40 lines by warp stall samples", S, 40)
    show("top 15 lines by divergent branches", DIV, 15)
    show("top 15 lines by local-memory L2 sectors (register spills and local arrays)", LOC, 15)
    print()
