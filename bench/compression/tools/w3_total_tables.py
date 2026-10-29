#!/usr/bin/env python
"""Render the w3_total tables (markdown, to stdout) from results/w3_total_14*.tsv, results/w3_total_bundle*.tsv and
results/w3_total_second.tsv.      python -P tools/w3_total_tables.py > results/w3_total_tables.md
"""
import csv, collections, os, sys, zipfile

R = "results/"
ORDER = ["brain_sn_v3_deep", "brain_sn_v3_intronic", "pbmc_v2_shallow", "pbmc_v2_typical", "pbmc_v3_lowgenes", "tcell_5p", "tumor_3p"]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fmt(n):
    return f"{int(round(float(n))):,}"


def main():
    main_rows = rows(R + "w3_total_14.tsv")
    det_rows = rows(R + "w3_total_14_detail.tsv")
    by = collections.defaultdict(dict)       # (sample, kind) -> codec -> row
    for r in main_rows:
        kind = "exon" if r["member"].startswith("exon") else "intron"
        by[(r["sample"], kind)][r["codec"]] = r
    det = collections.defaultdict(dict)
    for r in det_rows:
        kind = "exon" if r["member"].startswith("exon") else "intron"
        det[(r["sample"], kind)][r["codec"]] = r
    keys = [(s, k) for s in ORDER for k in ("exon", "intron")]
    codecs = ["ref_1pz_v1_L3", "a_names_perm_model", "w3_total_v1", "w3_total_v1c", "w3_total_v2", "w3_total_v2r", "w3_total_v2c"]

    print("## Table 1. 14 matrices, bytes of the whole stream (model files not counted)\n")
    print("| file | nnz | today (original .1pz) | Track A a_names_perm_model | w3_total_v1 | w3_total_v1c (class order) | w3_total_v2 (ambient empties) | w3_total_v2r (empties via b_rans16) | w3_total_v2c | v1 bits/nnz | matrix_ok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    nnz_tot = 0
    allok = True
    for k in keys:
        d = by[k]
        r1 = d["w3_total_v1"]
        nnz = int(r1["nnz"])
        nnz_tot += nnz
        ok = all(d[c]["matrix_ok"] == "1" for c in d)
        allok &= ok
        vals = [int(r1["orig_bytes"]), int(d["a_names_perm_model"]["enc_bytes"])] + [int(d[c]["enc_bytes"]) for c in codecs[2:]]
        tot["orig"] += vals[0]; tot["reemit"] += int(d["ref_1pz_v1_L3"]["enc_bytes"])
        for name, v in zip(["A"] + codecs[2:], vals[1:]):
            tot[name] += v
        print(f"| {k[0]} {k[1]} | {nnz:,} | {vals[0]:,} | {vals[1]:,} | {vals[2]:,} | {vals[3]:,} | {vals[4]:,} | {vals[5]:,} | {vals[6]:,} | {vals[2]*8/nnz:.2f} | {'1' if ok else '0'} |")
    print(f"| **total (14)** | {nnz_tot:,} | **{tot['orig']:,}** | **{tot['A']:,}** | **{tot['w3_total_v1']:,}** | **{tot['w3_total_v1c']:,}** | **{tot['w3_total_v2']:,}** | **{tot['w3_total_v2r']:,}** | **{tot['w3_total_v2c']:,}** | {tot['w3_total_v1']*8/nnz_tot:.2f} | {'all 1' if allok else 'FAIL'} |")
    print(f"| vs today (original) | | 1.000 | {tot['A']/tot['orig']:.3f} | {tot['w3_total_v1']/tot['orig']:.3f} | {tot['w3_total_v1c']/tot['orig']:.3f} | {tot['w3_total_v2']/tot['orig']:.3f} | {tot['w3_total_v2r']/tot['orig']:.3f} | {tot['w3_total_v2c']/tot['orig']:.3f} | | |")
    print(f"| vs re-emitted today ref_1pz_v1_L3 ({tot['reemit']:,}) | | | {tot['A']/tot['reemit']:.3f} | {tot['w3_total_v1']/tot['reemit']:.3f} | {tot['w3_total_v1c']/tot['reemit']:.3f} | {tot['w3_total_v2']/tot['reemit']:.3f} | {tot['w3_total_v2r']/tot['reemit']:.3f} | {tot['w3_total_v2c']/tot['reemit']:.3f} | | |")
    print(f"\nbits/nnz overall: today {tot['orig']*8/nnz_tot:.2f}, Track A {tot['A']*8/nnz_tot:.2f}, w3_total_v1 {tot['w3_total_v1']*8/nnz_tot:.2f}, w3_total_v2 {tot['w3_total_v2']*8/nnz_tot:.2f}\n")

    print("## Table 2. Components of the final w3_total_v1 stream, bytes (hdr = magic, flags, m, vt, n, nnz, chunk_cols, order, length fields)\n")
    print("| file | row-name id | barcodes (whitelist index) | user kv | row counts | column-nnz stream (depth side info) | coder tables | chunk streams (the payload symbols) | header | total | b_rans16 payload bits/nnz | side info % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    sums = collections.Counter()
    for k in keys:
        r = det[k]["w3_total_v1"]
        nnz = int(r["nnz"])
        comp = dict(row=int(r["names_row"]), col=int(r["names_col"]), kv=int(r["names_kv"]), cnt=int(r["cells_counts"]),
                    colcnt=int(r["cells_colcnt"]), tab=int(r["cells_tables"]), ch=int(r["cells_chunks"]))
        total = int(r["enc_bytes"])
        hdr = total - sum(comp.values())
        for kk, v in comp.items():
            sums[kk] += v
        sums["hdr"] += hdr; sums["total"] += total; sums["nnz"] += nnz
        pay = comp["colcnt"] + comp["tab"] + comp["ch"]
        side = (comp["row"] + comp["col"] + comp["kv"] + comp["cnt"] + hdr + comp["colcnt"] + comp["tab"])
        print(f"| {k[0]} {k[1]} | {comp['row']:,} | {comp['col']:,} | {comp['kv']:,} | {comp['cnt']:,} | {comp['colcnt']:,} | {comp['tab']:,} | {comp['ch']:,} | {hdr:,} | {total:,} | {pay*8/nnz:.2f} | {100*side/total:.1f} |")
    t = sums
    print(f"| **total** | {t['row']:,} | {t['col']:,} | {t['kv']:,} | {t['cnt']:,} | {t['colcnt']:,} | {t['tab']:,} | {t['ch']:,} | {t['hdr']:,} | {t['total']:,} | | |")
    print("| share | " + " | ".join(f"{100*t[x]/t['total']:.1f}%" for x in ["row", "col", "kv", "cnt", "colcnt", "tab", "ch", "hdr"]) + " | 100% | | |")
    print("\nModel bytes (read-only side files, NOT counted above): GRCh38 vocabulary 3,569,124 (exon + intron lists), GRCm39 vocabulary 4,470,944, 3M-february-2018 whitelist 6,465, 737K-august-2016 whitelist 3,283.\n")

    print("## Table 3. w3_total_v2 components, bytes\n")
    print("| file | names (row id + barcodes + kv) | cells: row counts | cells: column-nnz | cells: tables | cells: chunk streams | empties: k per droplet | empties: ambient multiset | split bitmap | header | total | empties nnz | empties bits/nnz (k + multiset) | same empties via b_rans16 (v2r: counts + payload) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = det[k]["w3_total_v2"]; rr = det[k]["w3_total_v2r"]
        total = int(r["enc_bytes"])
        parts = [int(r["names"]), int(r["cells_counts"]), int(r["cells_colcnt"]), int(r["cells_tables"]), int(r["cells_chunks"]), int(r["empt_k"]), int(r["empt_ambient"]) - int(r["empt_k"]), int(r["bitmap"])]
        hdr = total - sum(parts)
        en = int(r["empt_nnz"])
        rrb = int(rr["empt_counts"]) + int(rr["empt_payload"])
        print(f"| {k[0]} {k[1]} | " + " | ".join(f"{x:,}" for x in parts) + f" | {hdr:,} | {total:,} | {en:,} | {int(r['empt_ambient'])*8/max(en,1):.2f} | {rrb:,} ({rrb*8/max(en,1):.2f} b/nnz) |")

    print("\n## Table 4. Decode time, single thread, min of 5, idle machine (ms); Mnnz/s in brackets\n")
    print("| file | nnz | today ref_1pz_v1_L3 | Track A a_names_perm_model | b_rans16 full (its own zstd names) | w3_total_v1 full, canonical | w3_total_v1 full, native | payload only canonical | payload only native | b_rans16 payload canonical | b_rans16 payload native | core GHz (calibrated) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    sp = collections.Counter()
    for k in keys:
        d = by[k]; dd = det[k]
        nnz = int(d["w3_total_v1"]["nnz"])
        g = lambda c: float(d[c]["dec_ms_min"])
        r1 = dd["w3_total_v1"]; rb = dd["b_rans16"]
        def cell(ms): return f"{ms:.1f} ({nnz/ms/1e3:.1f})"
        vals = dict(today=g("ref_1pz_v1_L3"), A=g("a_names_perm_model"), b=g("b_rans16"), v1=g("w3_total_v1"), fn=float(r1["full_ms_native"]),
                    pc=float(r1["payload_ms_canon"]), pn=float(r1["payload_ms_native"]), bc=float(rb["payload_ms_canon"]), bn=float(rb["payload_ms_native"]))
        for kk, v in vals.items():
            sp[kk] += v
        sp["nnz"] += nnz
        star = '*' if float(r1['core_ghz']) >= 2.5 else ''
        print(f"| {k[0]} {k[1]}{star} | {nnz:,} | {cell(vals['today'])} | {cell(vals['A'])} | {cell(vals['b'])} | {cell(vals['v1'])} | {cell(vals['fn'])} | {cell(vals['pc'])} | {cell(vals['pn'])} | {cell(vals['bc'])} | {cell(vals['bn'])} | {float(r1['core_ghz']):.2f} |")
    n = sp["nnz"]
    def tc(x): return f"{sp[x]:.0f} ({n/sp[x]/1e3:.1f})"
    print(f"| **sum of 14** | {n:,} | {tc('today')} | {tc('A')} | {tc('b')} | {tc('v1')} | {tc('fn')} | {tc('pc')} | {tc('pn')} | {tc('bc')} | {tc('bn')} | |")

    print("\n## Table 5. Second pass over the w3_total_v1 streams (bytes; ratio = recompressed / input)\n")
    sec = {r["file"].split("/")[-1].split("__w3_total_v1")[0]: r for r in rows(R + "w3_total_second.tsv")}
    print("| file | stream | zstd -19 | ratio | xz -9e | ratio |")
    print("|---|---|---|---|---|---|")
    lo = 9
    for k in keys:
        nm = f"{k[0]}__{k[1]}_counts.1pz"
        r = sec[nm]
        lo = min(lo, float(r["ratio_zstd19"]), float(r["ratio_xz9e"]))
        print(f"| {k[0]} {k[1]} | {int(r['in_bytes']):,} | {int(r['zstd19']):,} | {float(r['ratio_zstd19']):.4f} | {int(r['xz9e']):,} | {float(r['ratio_xz9e']):.4f} |")
    print(f"\nlowest ratio over all: {lo:.4f}")


if __name__ == "__main__":
    main()
