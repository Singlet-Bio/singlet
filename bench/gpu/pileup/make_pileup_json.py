#!/usr/bin/env python3
"""Merge the per-sample pileup results (bench/gpu/results/pileup_<SET>.json, written by pileup.cu) with the host stage timers of the
campaign engine runs into bench/gpu/results/pileup.json.  Usage: python make_pileup_json.py   (from anywhere)"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# read pairs of each sample (.1fq) and the campaign-binary host stage timers (T=4 engine runs of bam-stream, 2 repeats; seconds).
# Source on Clipper: $W/bam-stream/runs/<SET>/t4_r{1,2}/stages.json (parse_run.py of run.log): pileup_total = '[singlet] Pileup:' (annotation +
# pileup + UMI collapse + merges, 4 workers), index = BAM index build, cb_refresh = CB-tag scan of the BAM for the barcode list, sort = STAR BAM sort.
SAMPLES = {
    "GSM4006843": dict(pairs=66.699625e6, desc="human 3' v2", job=789669,
                       host=dict(pileup_total=[65.319, 65.4508], index=[5.4914, 5.58481], cb_refresh=[24.326, 23.707], pileup_pass=[29.1686, 28.4757], sort=[378.62, 317.0])),
    "GSM5411066": dict(pairs=69.6e6, desc="mouse 3' v3 (GRCm39)", job=789670,
                       host=dict(pileup_total=[49.2586, 50.6617], index=[6.54549, 6.17273], cb_refresh=[19.8188, 20.0596], pileup_pass=[23.2892, 23.8639], sort=[286.788, 302.81])),
    "GSM6470208": dict(pairs=121.8e6, desc="human 5' v2", job=789671,
                       host=dict(pileup_total=[197.294, 169.082], index=[149.504, 118.362], cb_refresh=[59.4632, 66.7365], pileup_pass=[114.214, 93.6794], sort=[2273.54, 2289.85])),
}
DEV = {"pbmc2m": dict(pairs=2.0e6, desc="PBMC 2 M-pair subsample (development run)", job=789652)}
WORKERS = 4

def group(ph):
    g = {"annotate": ph["1.annotate"], "signature_lists": ph["2.siglists"], "junction_ids": ph["3.sjids"], "expand_to_reads": ph["4.expand"],
         "gene_collapse_M6_bcd": sum(v for k, v in ph.items() if k[0] in "bcd"), "junction_count": ph["5.sj"], "snp_count": ph["6.snp"], "chrM_count": ph["7.mt"]}
    return {k: round(v, 3) for k, v in g.items()}

out = {"task": "W11 GPU annotation and pileup", "engine_commit": "c4a1be3844ae5a43c35eb3124850f91d303ebbc8 (prod/campaign), dev build with bench/gpu/pileup/pileup_dump.patch",
       "gpu": "NVIDIA H100 NVL (sm_90, nvcc 12.4.131, -fmad=false)", "timing": "CUDA events, median of 3 after one warm-up; H2D of the records and D2H of the outputs excluded and reported separately",
       "samples": {}, "dev_samples": {},
       "gpu_hours_used": 1.10, "core_hours_used": 31.4,
       "notes": ["records come from the CPU engine's STAR BAM via extract.cpp (M5 stand-in); truth = the engine's accumulators after run_parallel (to_csc, clamps included)",
                 "junctions compared by (chromosome, donor, acceptor); mt indels compared as sorted multisets; all other matrices entry by entry (cell, row, value)"]}
for table, store in ((SAMPLES, "samples"), (DEV, "dev_samples")):
    for name, meta in table.items():
        j = json.load(open(os.path.join(RES, "pileup_%s.json" % name)))
        eq = j["equivalence"]
        ident = {k: bool(v["identical"]) for k, v in eq.items()}
        e = {"description": meta["desc"], "job": meta["job"], "read_pairs": meta["pairs"], "records_reads": j["reads"], "signatures": j["signatures"], "alignments": j["alignments"],
             "barcodes": j["barcodes"], "tuples": j["tuples"], "identical": ident, "all_identical": all(ident.values()), "equivalence_detail": eq,
             "deterministic_across_runs": j["deterministic_across_runs"], "output_sha256": j["output_sha256"],
             "gpu_ms_by_stage": group(j["phase_median_ms"]), "gpu_ms_total": j["total_median_ms"], "reads_per_s": j["reads_per_s"],
             "pairs_per_s": meta["pairs"] / (j["total_median_ms"] / 1e3), "peak_device_memory_GB": j["peak_device_memory_GB"], "resident_inputs_GB": j["resident_inputs_GB"],
             "h2d_records_ms": j["h2d_records_ms"], "h2d_records_GB": j["h2d_records_GB"]}
        if "host" in meta:
            h = {k: statistics.mean(v) for k, v in meta["host"].items()}
            Mp = meta["pairs"] / 1e6
            e["host_work_removed"] = {
                "stage_wall_s_mean_of_2": {k: round(v, 2) for k, v in h.items()},
                "pileup_total_core_s_per_Mpairs": round(h["pileup_total"] * WORKERS / Mp, 3),
                "index_plus_cb_refresh_core_s_per_Mpairs_at_1_core": round((h["index"] + h["cb_refresh"]) / Mp, 3),
                "removed_core_s_per_Mpairs": round((h["pileup_total"] * WORKERS + h["index"] + h["cb_refresh"]) / Mp, 3),
                "not_counted_BAM_sort_core_s_per_Mpairs_at_4_threads": round(h["sort"] * WORKERS / Mp, 3),
                "basis": "campaign binary, 4 threads: pileup_total x 4 workers (allocated cores); index and cb_refresh at 1 core (lower bound); source runs t4_r1,t4_r2 in $W/bam-stream/runs"}
        out[store][name] = e
json.dump(out, open(os.path.join(RES, "pileup.json"), "w"), indent=1)
print(json.dumps({k: {n: {"all_identical": v["all_identical"], "gpu_ms_total": v["gpu_ms_total"], "peak_GB": v["peak_device_memory_GB"], "pairs_per_s": v["pairs_per_s"],
                         "removed": v.get("host_work_removed", {}).get("removed_core_s_per_Mpairs")} for n, v in out[k].items()} for k in ("samples", "dev_samples")}, indent=1))
