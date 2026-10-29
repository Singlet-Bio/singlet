"""E4: render the result tables as markdown fragments for notes/e4.md (prints to stdout).

    python -P tools/e4_report.py members|recompute|tiers|latency|derivable
"""
import csv
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402


def rd(name):
    with open(os.path.join(L.RESULTS, name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fm(x, nd=0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:,.{nd}f}"


def members():
    rows = rd("e4_members.tsv")
    print("| member / component | files | bytes (bundle) | share | per sample (mean) | derivable / sample | independent / sample | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['member']} | {r['files']} | {fm(r['bytes_total'])} | {r['share_of_bundle_pct']}% | {fm(r['bytes_per_sample_mean'])} | "
              f"{fm(r['derivable_bytes_per_sample']) if r['derivable_bytes_per_sample'] != '' else ''} | {fm(r['independent_bytes_per_sample']) if r['independent_bytes_per_sample'] != '' else ''} | {r['verdict']} |")


def recompute():
    rows = rd("e4_recompute.tsv")
    g = collections.OrderedDict()
    for r in rows:
        key = (r["test"], r["item"].replace(r["sample"], "") if False else r["item"])
        if r["test"] in ("summary.json",):
            key = ("summary.json", "numeric fields")
        if r["test"] in ("pileup_stats.json", "provenance.json"):
            key = (r["test"], "fields that are matrix sums")
        if r["test"] == "block_recompute":
            it = r["item"]
            key = ("block_recompute", it.split(".", 1)[1] if "." in it else it)
        g.setdefault(key, []).append(r)
    print("| test | what is compared | comparisons (equal / total) | samples | exact |")
    print("|---|---|---|---|---|")
    for (t, it), rs in g.items():
        tot = sum(int(x["n_compared"]) for x in rs)
        eq = sum(int(x["n_equal"]) for x in rs)
        ex = sum(1 for x in rs if x["exact"] == "True")
        print(f"| {t} | {it} | {eq:,} / {tot:,} | {len(rs)} | {ex}/{len(rs)} |")


def tiers():
    rows = rd("e4_tiers.tsv")
    T = {(r["sample"], r["member"], r["subset"]): r for r in rows}
    samples = [s for s in L.CORPUS if (s, "exon", "orig") in T]

    def f(x):
        try:
            return float(x)
        except ValueError:
            return float("nan")

    def tot(s, subset, key, members=("exon", "intron")):
        return sum(f(T[(s, m, subset)][key]) for m in members)

    print("| sample | called / columns | today exon+intron | cells (i) | empties (i) | cells (ii) | empties (ii) | cells (iii) | empties (iii) | cells (iii) exon only |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in samples:
        n = int(T[(s, "exon", "cells")]["ncols"])
        N = int(T[(s, "exon", "orig")]["ncols"])
        print(f"| {s} | {n:,} / {N:,} | {fm(tot(s, 'orig', 'file_bytes_i'))} | {fm(tot(s, 'cells', 'file_bytes_i'))} | {fm(tot(s, 'empties', 'file_bytes_i'))} | "
              f"{fm(tot(s, 'cells', 'bytes_ii'))} | {fm(tot(s, 'empties', 'bytes_ii'))} | {fm(tot(s, 'cells', 'bytes_iii'))} | {fm(tot(s, 'empties', 'bytes_iii'))} | "
              f"{fm(f(T[(s, 'exon', 'cells')]['bytes_iii']))} |")


def latency():
    rows = rd("e4_latency.tsv")
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["sample"], r["scenario"], r["link"])][r["variant"]] = r
    order = ["today", "tiered_i", "untiered_ii", "tiered_ii", "tiered_iii"]
    for link in ("100Mbit", "1Gbit"):
        print(f"\n**{link}, RTT 50 ms** (cell = MB transferred / seconds / rounds)\n")
        print("| sample | scenario | " + " | ".join(order) + " |")
        print("|---|---|" + "---|" * len(order))
        for (s, sc, ln), d in by.items():
            if ln != link:
                continue
            cells = []
            for v in order:
                r = d.get(v)
                cells.append(f"{r['mb']} MB / {float(r['t_ms']) / 1000:.2f} s / {r['rounds']}r" if r else "")
            print(f"| {s} | {sc} | " + " | ".join(cells) + " |")


def derivable():
    rows = rd("e4_derivable.tsv")
    cols = list(rows[0].keys())
    print("| category | mean bytes / sample | share of sample |")
    print("|---|---|---|")
    mean_total = sum(float(r["sample_bytes"]) for r in rows) / len(rows)
    for c in cols[2:]:
        mu = sum(float(r[c]) for r in rows) / len(rows)
        print(f"| {c} | {mu:,.0f} | {100 * mu / mean_total:.1f}% |")
    print(f"| sample total | {mean_total:,.0f} | 100% |")




def latency_short():
    rows = rd("e4_latency.tsv")
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["sample"], r["scenario"], r["link"])][r["variant"]] = r
    samples = []
    for (s, sc, ln) in by:
        if s not in samples:
            samples.append(s)
    print("| sample | scenario | today MB | proposed (iii) MB | 100 Mbit/s today -> proposed (s) | 1 Gbit/s today -> proposed (s) | rounds today -> proposed |")
    print("|---|---|---|---|---|---|---|")
    for s in samples:
        for sc in ("first_chunk", "all_cells", "everything"):
            a = by.get((s, sc, "100Mbit"))
            b = by.get((s, sc, "1Gbit"))
            if not a or "today" not in a or "tiered_iii" not in a:
                continue
            t0, t1 = a["today"], a["tiered_iii"]
            u0, u1 = b["today"], b["tiered_iii"]
            print(f"| {s} | {sc} | {t0['mb']} | {t1['mb']} | {float(t0['t_ms']) / 1000:.2f} -> {float(t1['t_ms']) / 1000:.2f} | {float(u0['t_ms']) / 1000:.2f} -> {float(u1['t_ms']) / 1000:.2f} | {t0['rounds']} -> {t1['rounds']} |")


def today_layout():
    rows = rd("e4_today_layout.tsv")
    print("| sample | matrix | columns | called | last called column | chunks | chunks holding a called cell | chunk bytes of those / all |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['sample']} | {r['member']} | {int(r['n_cols']):,} | {int(r['n_called']):,} | {int(r['last_called_col']):,} | {r['n_chunks']} | {r['last_cell_chunk']} | "
              f"{int(r['selective_chunk_bytes']):,} / {int(r['total_chunk_bytes']):,} ({100 * int(r['selective_chunk_bytes']) / int(r['total_chunk_bytes']):.0f}%) |")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
