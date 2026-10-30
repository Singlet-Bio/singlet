#!/usr/bin/env python3
"""Markdown tables from ../results/m10b_timers.json (core-seconds per section, per sample and per M pairs, ranked)."""
import json, sys
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "../results/m10b_timers.json"))
S = [g for g in ["GSM4006843", "GSM5411066", "GSM7901264", "GSM7036403", "GSM6470208", "GSM7656149"] if r.get(g, {}).get("status") == "ok"]

def sec(g, n, k="cpu_s"):
    return r[g]["sections"].get(n, {}).get(k, 0.0)

def cat(g):
    L = r[g]["ledger"]; D = r[g]["demux"]; ss = r[g]["sections"]
    c = {}
    nh = sec(g, "stage.nonhost_em_total")
    c["nonhost: load 3 indices (viral+bacterial+fungal)"] = sec(g, "nonhost.load_indices")
    c["nonhost: k-mer classify (OpenMP)"] = sec(g, "nonhost.classify")
    c["nonhost: EM deconvolution"] = sec(g, "nonhost.em")
    c["nonhost: read FASTQ, cell matrix, writes, bloom"] = nh - c["nonhost: load 3 indices (viral+bacterial+fungal)"] - c["nonhost: k-mer classify (OpenMP)"] - c["nonhost: EM deconvolution"]
    c["gap: BAM unlink"] = sec(g, "stage.bam_unlink")
    c["gap: other (strand check, log parse, cleanup)"] = sec(g, "gap.pileup_end_to_export_start") - nh - c["gap: BAM unlink"]
    dthr = ss["demux.thread"]["thr_cpu_s"]
    ak = ss.get("autoK.total", {}).get("thr_cpu_s", 0.0)
    ff = sum(v["cpu_s"] for k, v in D["final_fit"].items() if k.startswith("demux.final_fit"))
    c["demux: auto-K selection (K fits, 2 folds)"] = ak
    c["demux: final fit (selected K)"] = ff
    c["demux: other (covered SNPs, doublet stage, back-assign)"] = dthr - ak - ff
    for n, lab in [("export.csc_conversion", "export: COO->CSC (8 threads)"), ("doublet", "export: doublet scoring (Scrublet-like)"),
                   ("cell_calling", "export: cell calling (called_v2)"), ("ambient", "export: ambient (DecontX-like)"),
                   ("saturation_curve", "export: saturation curve"), ("mt.call_and_write_variants", "export: mt heteroplasmy calls"),
                   ("mt_donor_outputs(G6)", "export: mt event caller (G6)")]:
        c[lab] = L[n]["core_s"] if n in L else 0.0
    small = ["cell_qc.compute", "sex_calling", "cell_cycle", "read_stats", "ancestry", "cell_qc.star_read_stats", "cell_qc.write_tsv",
             "mt.rev_csc_and_reference", "demux.write_assignments", "demux.aggregate_depths", "demux.write_vcfs_coverages"]
    c["export: small callers + cell_qc + donor writes"] = sum(L[n]["core_s"] for n in small if n in L)
    c["export: .1pz / tsv writers (own threads, sum of thread CPU)"] = sum(v["core_s"] for k, v in L.items() if k.startswith("writer:"))
    c["export: unattributed (writer codec helper threads, gaps)"] = L["UNATTRIBUTED (helper threads of writers/pz codec, section gaps)"]["core_s"]
    c["post-export (provenance, summary, stats json, validate)"] = r[g]["W"]["post_export_core_s"]
    return c

C = {g: cat(g) for g in S}
names = list(C[S[0]].keys())
tot = {n: sum(C[g][n] for g in S) for n in names}
rank = sorted(names, key=lambda n: -tot[n])
print("| rank | section | " + " | ".join("%s (%.0f M)" % (g, r[g]["reads_M"]) for g in S) + " | sum core-s | per M pairs (pooled) |")
print("|---|---|" + "---|" * (len(S) + 2))
M = sum(r[g]["reads_M"] for g in S)
for i, n in enumerate(rank, 1):
    print("| %d | %s | " % (i, n) + " | ".join("%.1f" % C[g][n] for g in S) + " | %.0f | %.3f |" % (tot[n], tot[n] / M))
print("| | **TOTAL W (post-pileup host)** | " + " | ".join("**%.1f**" % sum(C[g].values()) for g in S) + " | %.0f | %.3f |" % (sum(tot.values()), sum(tot.values()) / M))
print("| | W per M pairs | " + " | ".join("%.2f" % (sum(C[g].values()) / r[g]["reads_M"]) for g in S) + " | | |")
print("| | check: W from timers (gap+export+post) | " + " | ".join("%.1f" % r[g]["W"]["W_total_core_s"] for g in S) + " | | |")
print()
print("per M pairs:")
print("| section | " + " | ".join(S) + " |")
for n in rank:
    print("| %s | " % n + " | ".join("%.3f" % (C[g][n] / r[g]["reads_M"]) for g in S) + " |")
json.dump({"categories_core_s": C, "ranking": rank, "pooled_M_pairs": M}, open("../results/m10b_ranked.json", "w"), indent=1)
