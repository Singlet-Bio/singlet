"""Turn scores.json (score.py), breakdown_hgmm1k_v3.json and the locus json into the contract fragment
`bench/public/results/_exact_vs_pseudo_metrics.json` (contract 1.1.0 objects the coordinator merges into
alignment-performance.json), plus a self-check that wraps it in a draft document and runs contract.validate.

  python build_fragment.py --scores scores.json [--breakdown b.json] [--t4 scores_t4.json] --out FRAGMENT.json --check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "public"))
import contract  # noqa: E402

TOOLS_ALL = ["STARsolo", "alevin-fry", "alevin-fry-em"]
CR = "cellranger-published"
CR_VERSION = "3.0.0"
CR_URLS = {
    "hgmm1k_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v3/hgmm_1k_v3_filtered_feature_bc_matrix.h5",
    "pbmc1k_cite_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_filtered_feature_bc_matrix.h5",
}
FASTQ_URLS = {
    "hgmm1k_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v3/hgmm_1k_v3_fastqs.tar",
    "pbmc1k_cite_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_fastqs.tar",
    "pbmc1k_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar",
}
SINGLET_COMMIT = "02aa9b938b54e0f92bba00c44c07b883a8660aa2"
SINGLET_SHA256 = "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c"


def tools_section():
    return [
        {"name": "singlet", "version": "1.0.0", "role": "singlet",
         "params": "singlet-campaign <sample>.1fq --ref-base REF --genome-dir STAR_2.7.11b_index --exons genes.gtf --out-prefix OUT/ "
                   "--threads N --pipeline --output-format mtx --raw-matrix  (after `singlet encode`); counts = raw_feature_bc_matrix "
                   "(exon+intron, each molecule once); exact genome-wide alignment by the STAR fork, then pileup. Encode step reported separately."},
        {"name": "STARsolo", "version": "2.7.11b", "role": "competitor", "url": "https://github.com/alexdobin/STAR",
         "params": "STAR --soloType CB_UMI_Simple --soloCBlen 16 --soloUMIlen 12 --soloStrand Forward --soloFeatures Gene GeneFull Velocyto "
                   "--soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR "
                   "--soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None; GeneFull scored (exon+intron)"},
        {"name": "alevin-fry", "version": "0.18.3 (simpleaf; piscem 0.23.0 index, piscem-rs 0.10.0 mapper)", "role": "competitor",
         "url": "https://github.com/COMBINE-lab/alevin-fry",
         "params": "simpleaf quant --chemistry 10xv3 --expected-ori fw --resolution cr-like --unfiltered-pl 3M-february-2018.txt "
                   "(splici index of the 10x 2024-A GTF, USA mode); counts = spliced+unspliced+ambiguous"},
        {"name": "alevin-fry-em", "version": "0.18.3", "role": "competitor", "url": "https://github.com/COMBINE-lab/alevin-fry",
         "params": "same mapping and permit list as alevin-fry; alevin-fry quant -r cr-like-em (EM resolution of multi-gene UMIs); "
                   "quant step timed separately"},
        {"name": CR, "version": CR_VERSION, "role": "competitor", "url": CR_URLS["hgmm1k_v3"],
         "params": "10x Genomics PUBLISHED Cell Ranger 3.0.0 filtered_feature_bc_matrix.h5 (not re-run; no 10x software was installed or run). "
                   "Older references than the other tools: hg19+mm10 for hgmm1k_v3, GRCh38 (3.0.0) for pbmc1k_cite_v3: "
                   f"{CR_URLS['hgmm1k_v3']} ; {CR_URLS['pbmc1k_cite_v3']}. No runtime reported."},
    ]


def datasets_section():
    def ds(i, title, truth, notes):
        return {"id": i, "title": title, "data_kind": "real", "accessions": {"url": FASTQ_URLS[i]}, "truth": truth, "notes": notes}
    return [
        ds("hgmm1k_v3", "10x HEK293T + NIH3T3 1:1, 1k, 3' v3 (77M read pairs)",
           "Species of origin: a cell is human or mouse (pure cells: >=90% of UMIs one species in the published Cell Ranger matrix). "
           "UMIs of the other species inside a pure cell are mapping errors or ambient RNA (identical for every tool on the same reads). "
           "Second truth: HEK293T is female-derived, so UMIs on non-PAR chrY genes in human cells are false positives.",
           "1,046 Cell Ranger cells: 495 pure human, 544 pure mouse, 7 mixed."),
        ds("pbmc1k_cite_v3", "10x PBMC 1k + TotalSeq-B (pbmc_1k_protein_v3), 3' v3",
           "Orthogonal assay: antibody-derived tags (protein) measured on the same cells. RNA of the matching gene should rank cells like the protein.",
           "Cells = published Cell Ranger filtered barcodes; protein = Cell Ranger Antibody Capture counts, CLR-normalised."),
        ds("pbmc1k_v3", "10x PBMC 1k, 3' v3 (66.6M read pairs)",
           "No per-molecule truth. Used for runtime and memory (same reads, same node, same threads) and for read accounting.",
           "Runtime dataset: 16 threads and 4 threads on one node."),
    ]


def sh(x, lo=None):
    return {"value": x} if lo is None else {"value": x, "ci": lo}


def build(scores, t4, breakdown, locus):
    hg, ci_, pb = scores["hgmm1k_v3"], scores.get("pbmc1k_cite_v3"), scores["pbmc1k_v3"]
    M = []

    def comp(d, key, tools, cikey=None):
        out = []
        for t in tools:
            if t in d["tools"]:
                e = {"tool": t, "value": d["tools"][t][key]}
                if cikey and cikey in d["tools"][t]:
                    e["ci"] = d["tools"][t][cikey]
                out.append(e)
        return out

    # ---- barnyard
    b = hg["barnyard"]
    s = b["tools"]["singlet"]
    half = (s["ci"][1] - s["ci"][0]) / 2
    M.append({"id": "evp_wrong_species_fraction", "name": "UMIs of the wrong species in pure barnyard cells", "unit": "fraction",
              "category": "accuracy", "dataset": "hgmm1k_v3", "direction": "lower",
              "singlet": {"value": s["wrong_species_fraction"], "ci": s["ci"]},
              "competitors": comp(b, "wrong_species_fraction", TOOLS_ALL + [CR], "ci"),
              "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": half},
                       "rationale": "half-width of Singlet's bootstrap 95% CI over cells"},
              "notes": "Real truth: species of the cell. Includes ambient RNA (shared by all tools); GeneFull-type counts (exon+intron) for Singlet, STARsolo and alevin-fry USA."})
    M.append({"id": "evp_cells_over_1pct_wrong_species", "name": "Pure cells with >=1% wrong-species UMIs", "unit": "fraction",
              "category": "accuracy", "dataset": "hgmm1k_v3", "direction": "lower",
              "singlet": {"value": s["frac_cells_ge_1pct_wrong"]},
              "competitors": comp(b, "frac_cells_ge_1pct_wrong", TOOLS_ALL + [CR]),
              "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": 0.01},
                       "rationale": "1 percentage point of cells: about 10 of 1,039 pure cells, the cell-level sampling noise of a 1k-cell run"}})
    # ---- chrY
    c = hg["chrY"]
    sy = c["tools"]["singlet"]
    M.append({"id": "evp_chry_umi_per_million", "name": "chrY UMIs per million human UMIs in female-derived HEK293T cells", "unit": "UMI per million",
              "category": "accuracy", "dataset": "hgmm1k_v3", "direction": "lower",
              "singlet": {"value": sy["chrY_per_million"], "ci": sy["ci_per_million"]},
              "competitors": comp(c, "chrY_per_million", TOOLS_ALL, "ci_per_million"),
              "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": (sy["ci_per_million"][1] - sy["ci_per_million"][0]) / 2},
                       "rationale": "half-width of Singlet's bootstrap 95% CI over cells"},
              "notes": "Truth: no Y chromosome. Every UMI is a false positive (X/Y gametologs, Y-linked repeat loci). Published Cell Ranger is omitted: its hg19 gene ids do not map to the GRCh38 chrY list."})
    # ---- CITE
    if ci_ and "cite" in ci_:
        ct = ci_["cite"]
        sc = ct["tools"]["singlet"]
        others = [t for t in TOOLS_ALL + [CR] if t in ct["tools"]]
        d = ct["paired_diff_mean_ci"]["singlet"]
        best = max(others, key=lambda t: ct["tools"][t]["mean_spearman"])
        lo, hi = d[best][1], d[best][2]
        M.append({"id": "evp_rna_protein_spearman", "name": f"Mean RNA-protein Spearman over {len(ct['pairs'])} CITE-seq marker pairs", "unit": "correlation",
                  "category": "accuracy", "dataset": "pbmc1k_cite_v3", "direction": "higher",
                  "singlet": {"value": sc["mean_spearman"], "ci": sc["ci"]},
                  "competitors": [{"tool": t, "value": ct["tools"][t]["mean_spearman"], "ci": ct["tools"][t]["ci"]} for t in others],
                  "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": (hi - lo) / 2},
                           "rationale": f"half-width of the paired bootstrap 95% CI of (Singlet - {best}) on the same resampled cells"},
                  "notes": "Pairs: " + ", ".join(f"{p}~{g}" for p, g in ct["pairs"]) + ". CD16 protein vs FCGR3A is a paralog stress test (FCGR3A/FCGR3B)."})
    # ---- accuracy score (for the Pareto figure)
    comps = {"wrong": ("wrong_species_fraction", b, "lower"), "chrY": ("chrY_per_million", c, "lower")}
    names = ["singlet"] + TOOLS_ALL
    def rel(vals, direction):
        best = min(vals.values()) if direction == "lower" else max(vals.values())
        out = {}
        for t, v in vals.items():
            if direction == "lower":
                out[t] = 1.0 if v <= 0 else min(1.0, best / v) if best > 0 else 1.0
            else:
                out[t] = v / best if best > 0 else 1.0
        return out
    parts = []
    parts.append(rel({t: b["tools"][t]["wrong_species_fraction"] for t in names}, "lower"))
    parts.append(rel({t: c["tools"][t]["chrY_per_million"] for t in names}, "lower"))
    if ci_ and "cite" in ci_:
        parts.append(rel({t: ci_["cite"]["tools"][t]["mean_spearman"] for t in names}, "higher"))
    score = {t: sum(p[t] for p in parts) / len(parts) for t in names}
    M.append({"id": "evp_accuracy_score", "name": "Accuracy score: mean of each tool's value relative to the best tool", "unit": "score (1 = best on every component)",
              "category": "accuracy", "dataset": "pooled", "direction": "higher",
              "singlet": {"value": score["singlet"]},
              "competitors": [{"tool": t, "value": score[t]} for t in TOOLS_ALL],
              "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": 0.02},
                       "rationale": "2% of the score: smaller than the bootstrap half-widths of the components (they are 5-10% relative)"},
              "notes": "Components (real truth each): barnyard wrong-species fraction, HEK293T chrY UMIs per million, CITE-seq mean RNA-protein Spearman (the gex-counts adt_rna_spearman.total definition, same pairs; Singlet 0.4215, STARsolo 0.4276, alevin-fry 0.4149). "
                       "Each is scaled to the best of the four runnable tools (best/value for errors, value/best for correlation); equal weights."})

    # ---- resources
    R = []
    pairs = pb["timing"]["alevinfry"]  # placeholder to keep structure explicit
    n_pairs = 66601887
    per100 = 100e6 / n_pairs
    def resrow(tool, ds, d, threads, with_encode=False, notes=""):
        wall = d["wall_s_with_encode"] if with_encode and "wall_s_with_encode" in d else d["wall_s"]
        cpu = d["cpu_h_with_encode"] if with_encode and "cpu_h_with_encode" in d else d["cpu_h"]
        row = {"tool": tool, "dataset": ds, "wall_s": wall, "cpu_h": cpu, "max_rss_gb": d["max_rss_gb"], "threads": threads}
        if notes:
            row["notes"] = notes
        return row
    tmap = {"singlet": "singlet", "STARsolo": "starsolo", "alevin-fry": "alevinfry"}
    R16, R4 = [], []
    for ds, scr, lab in (("pbmc1k_v3", pb, 16),):
        for t, k in tmap.items():
            d = scr["timing"][k]
            R16.append(resrow(t, ds, d, 16, True, ("incl. .1fq encode " + f"{d.get('encode_wall_s', 0):.0f}s wall / {d.get('encode_cpu_h', 0):.3f} CPU-h (done by the 1-core download stage in the campaign)") if t == "singlet" else ""))
        d = pb["timing"]["alevinfry"]
        R16.append({"tool": "alevin-fry-em", "dataset": "pbmc1k_v3", "wall_s": d["wall_s"] + d.get("em_quant_wall_s", 0), "cpu_h": d["cpu_h"] + d.get("em_quant_cpu_h", 0),
                    "max_rss_gb": d["max_rss_gb"], "threads": 16, "notes": "alevin-fry run plus the cr-like-em quant step"})
    t4s = pb.get("timing_t4") or {}
    if t4s:
        for t, k in tmap.items():
            if k in t4s:
                d = t4s[k]
                R4.append(resrow(t, "pbmc1k_v3", d, 4, True, "4-thread run, same node"))
        if "alevinfry" in t4s:
            d = t4s["alevinfry"]
            R4.append({"tool": "alevin-fry-em", "dataset": "pbmc1k_v3", "wall_s": d["wall_s"] + d.get("em_quant_wall_s", 0), "cpu_h": d["cpu_h"] + d.get("em_quant_cpu_h", 0),
                       "max_rss_gb": d["max_rss_gb"], "threads": 4})
    resources = R16 + R4
    # pooled rows = per 100M pairs, 16 threads (wall in s per 100M pairs); used by the Pareto figure
    pooled16 = [{**r, "dataset": "pooled", "wall_s": r["wall_s"] * per100, "cpu_h": r["cpu_h"] * per100,
                 "notes": ((r.get("notes", "") + "; ") if r.get("notes") else "") + "normalised to 100M read pairs (pbmc1k_v3, 66.6M pairs, 16 threads)"} for r in R16]
    pooled4 = [{**r, "dataset": "pooled", "wall_s": r["wall_s"] * per100, "cpu_h": r["cpu_h"] * per100} for r in R4]

    def resmetric(mid, name, rows, vs, threads):
        by = {r["tool"]: r["cpu_h"] * per100 for r in rows}
        if "singlet" not in by:
            return None
        return {"id": mid, "name": name, "unit": "CPU-hours per 100M read pairs", "category": "resources", "dataset": "pbmc1k_v3", "direction": "lower",
                "singlet": {"value": by["singlet"]},
                "competitors": [{"tool": t, "value": by[t]} for t in vs if t in by],
                "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"rel": 0.10},
                         "rationale": "10%: run-to-run variation of CPU time on a shared node (the four runs of one tool differ by up to 10%)"},
                "notes": f"{threads} threads, same node, user+sys CPU. Singlet includes the .1fq encode step (labelled; in the campaign it runs on the 1-core download stage)."}
    for m in (resmetric("evp_cpu_h_per_100m_pairs_16t", "CPU-hours per 100M pairs, 16 threads", R16, ["STARsolo", "alevin-fry", "alevin-fry-em"], 16),
              resmetric("evp_cpu_h_per_100m_pairs_4t", "CPU-hours per 100M pairs, 4 threads (the campaign setting)", R4, ["STARsolo", "alevin-fry", "alevin-fry-em"], 4)):
        if m:
            M.append(m)
    # ---- read placement
    ra = pb.get("read_accounting", {})
    if "star" in ra and "piscem" in ra:
        placed_star = 1 - ra["star"]["unmapped_fraction"] - ra["star"]["too_many_loci_fraction"]
        M.append({"id": "evp_reads_placed_fraction", "name": "Reads placed (aligned to the genome / mapped to the splici index)", "unit": "fraction",
                  "category": "coverage", "dataset": "pbmc1k_v3", "direction": "higher",
                  "singlet": {"value": placed_star},
                  "competitors": [{"tool": "STARsolo", "value": placed_star}, {"tool": "alevin-fry", "value": ra["piscem"]["mapped_fraction"]}],
                  "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": 0.005}, "rationale": "0.5 percentage point: unique+multi-mapped vs 'too many loci' accounting differs by about this much between STAR logs"},
                  "notes": "Singlet uses the same STAR alignment (scoreMin 10 vs STARsolo 30, so slightly more reads are placed). Placed is not correct: see the accuracy metrics. No truth for the reads that only the genome aligner places."})
    return M, resources, pooled16, pooled4, score


REACH = [
    # (output, singlet, STARsolo, alevin-fry, note per tool)
    ("gene counts (exon+intron, each molecule once)", "yes", "yes", "yes", "all three; accuracy: evp_* metrics"),
    ("spliced / unspliced / ambiguous molecule layers", "yes", "yes", "yes",
     "Singlet: exon/intron/ambiguous (velocyto rules); STARsolo: Velocyto; alevin-fry: USA from the splici index. No real-truth benchmark in this fragment."),
    ("per-cell splice-junction counts", "yes", "partial", "no",
     "Singlet vs STARsolo SJ: r=0.994 for junctions with >=10 molecules (docs/modality-benchmarks/junctions-nonhost.md); alevin-fry maps to transcripts and introns, never to junction coordinates."),
    ("PSI / cassette-exon inclusion", "yes", "no", "no",
     "Cassette PSI from Singlet junction counts matches STARsolo-derived PSI: pseudobulk r=0.996, MAE 0.008 at coverage>=20. STARsolo gives junctions only."),
    ("per-cell SNP allele counts and genotype-free donor demultiplexing", "yes", "partial", "no",
     "Singlet ARI 0.986 vs vireo 0.923 (Kang 2018, 8 donors, K given; auto-K was broken in the benchmarked build). STARsolo has --varVCF genotypes per cell but no demultiplexing. alevin-fry has no alignments, so no base-level evidence."),
    ("genotype-based doublet calls", "partial", "no", "no",
     "Benchmarked build: prob_doublet hard-coded 0; filled in prod/modality-fixes, not re-benchmarked."),
    ("mtDNA per-cell heteroplasmy", "yes", "no", "no",
     "Singlet per-cell VAF r=0.85 vs read-level pileup; variant list was worse (12% supported) before modality-fixes."),
    ("non-host (viral, microbial) reads from unmapped reads", "yes", "partial", "no",
     "Singlet vs Kraken2 per-cell SARS-CoV-2 UMIs Spearman 0.985; STARsolo can write unmapped reads but classifies none; alevin-fry discards unmapped reads."),
    ("library strand detection", "partial", "no", "no",
     "Singlet probes strand; failed on 5' libraries in the atlas (17% of samples flagged, docs/pipeline-validation.md). STARsolo needs --soloStrand; simpleaf takes --expected-ori from the chemistry."),
    ("read accounting: intergenic / antisense / intronic per cell", "yes", "partial", "no",
     "STAR assigns reads to the genome, so reads outside every annotated gene are counted (STARsolo Features.stats noNoFeature). piscem reads that miss the splici index are simply unmapped."),
    ("reads from unannotated / novel loci", "partial", "partial", "no",
     "Aligned to the genome and kept in the BAM; counts are still per annotated gene."),
    ("poly(A) site counts with internal-priming flag", "yes", "no", "no", "polya_counts, polya_sites.tsv.gz (prod/modality-fixes); not benchmarked against an external truth yet."),
    ("ambient RNA estimate per cell", "yes", "no", "no", "ambient_contamination.tsv (Singlet); STARsolo and alevin-fry rely on external tools (SoupX, CellBender)."),
    ("cell calling", "yes", "yes", "partial", "Singlet called_v2 (EmptyDrops-like); STARsolo EmptyDrops_CR; alevin-fry permit list (knee or unfiltered whitelist)."),
    ("coordinate-sorted BAM with CB/UB tags", "yes", "yes", "no", "Optional (--tagged-bam / --outSAMtype); enables external SNP, mtDNA and isoform tools."),
]


def reach_section():
    out = []
    for o, a, b, c, note in REACH:
        for tool, sup in (("singlet", a), ("STARsolo", b), ("alevin-fry", c)):
            out.append({"tool": tool, "output": o, "supported": sup, "note": note if tool == "singlet" else ""})
    return out


def reach_metric():
    n = {"singlet": 0, "STARsolo": 0, "alevin-fry": 0}
    for o, a, b, c, note in REACH:
        for t, s in zip(("singlet", "STARsolo", "alevin-fry"), (a, b, c)):
            if s == "yes":
                n[t] += 1
    return {"id": "evp_reach_count", "name": "Documented outputs a tool produces (count of 'yes' in the reach table)", "unit": "outputs", "category": "coverage",
            "dataset": "pooled", "direction": "higher", "singlet": {"value": n["singlet"]},
            "competitors": [{"tool": "STARsolo", "value": n["STARsolo"]}, {"tool": "alevin-fry", "value": n["alevin-fry"]}],
            "rule": {"type": "match_or_beat", "against": "best"}, "notes": "A count of capabilities, not of benchmarked ones: see reach[] notes for which have a measured metric."}


def methods_text():
    return (
        "Exact versus pseudo-alignment. Singlet aligns every read to the genome with a STAR fork and counts from the alignments (pileup); "
        "alevin-fry (via simpleaf and piscem) maps reads by k-mer matching to a splici index of the annotated transcripts and introns. "
        "The genome-wide aligner costs more CPU and memory and places reads outside the annotation; the pseudoaligner is several times faster. "
        "This fragment measures both sides on real data. Truth: (1) species of origin in a human-mouse barnyard (hgmm1k_v3); (2) no chrY in "
        "female-derived HEK293T cells; (3) CITE-seq protein against RNA on the same PBMC cells. No simulated reads are used. "
        "All tools get the same FASTQs, the 10x 2024-A references, the same cells (published Cell Ranger barcodes) and, for timing, one node and 16 or 4 threads. "
        "10x software was not run: Cell Ranger is the published 3.0.0 matrix (older references). Runtime is reported as CPU-hours per 100M read pairs "
        "with wall time secondary; Singlet includes its .1fq encode step, labelled.")


def assemble(scores, t4, breakdown, locus):
    M, resources, pooled16, pooled4, score = build(scores, t4, breakdown, locus)
    M = [m for m in M if m["id"] != "evp_rna_protein_spearman"]  # one CITE definition on the site: gex-counts adt_rna_spearman.*
    M.append(reach_metric())
    return M, resources, pooled16, pooled4


def check(fragment, cohort_path):
    base = {"schema": "singlet.benchmark.capability", "schema_version": contract.SCHEMA_VERSION, "capability": "alignment-performance",
            "title": "draft", "conclusion": "draft", "methods": fragment["methods"], "status": "draft", "generated_at": "2026-10-25T00:00:00+00:00",
            "owner": "exact-vs-pseudo", "singlet": {"version": "1.0.0", "commit": SINGLET_COMMIT, "binary_sha256": SINGLET_SHA256},
            "tools": fragment["tools"], "datasets": fragment["datasets"], "metrics": copy.deepcopy(fragment["metrics"]),
            "resources": fragment["resources"], "figures": fragment["figures"], "reach": fragment["reach"], "artifacts": fragment.get("artifacts", []),
            "code": [{"path": "bench/exact_vs_pseudo/score.py", "url": f"https://github.com/Singlet-Bio/singlet/blob/{SINGLET_COMMIT}/bench/exact_vs_pseudo/score.py"}],
            "reproduce": {"module": "bench/exact_vs_pseudo/score.py", "function": "compute"}, "caveats": fragment["caveats"]}
    contract.finalize(base)
    return base, contract.validate(base, contract.load_cohort(cohort_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--breakdown"); ap.add_argument("--locus"); ap.add_argument("--out", required=True)
    ap.add_argument("--caveats", default=os.path.join(HERE, "caveats.json")); ap.add_argument("--artifacts", default=os.path.join(HERE, "artifacts.json"))
    ap.add_argument("--check", action="store_true"); ap.add_argument("--cohort", default=os.path.join(HERE, "..", "public", "cohort.tsv"))
    a = ap.parse_args()
    scores = json.load(open(a.scores))
    bd = json.load(open(a.breakdown)) if a.breakdown else None
    M, resources, pooled16, pooled4 = assemble(scores, None, bd, None)
    frag = {"_comment": "Draft fragment for alignment-performance.json (contract 1.1.0). Merge tools/datasets/metrics/resources/figures/reach/artifacts/caveats; "
                        "methods is a paragraph to add. Pareto figures are built with contract.pareto_figure on the 'pooled' resources (per 100M pairs).",
            "methods": methods_text(), "tools": tools_section(), "datasets": datasets_section(), "metrics": M,
            "resources": resources + pooled16 + pooled4, "reach": reach_section(), "figures": [],
            "artifacts": json.load(open(a.artifacts)) if os.path.exists(a.artifacts) else [],
            "caveats": json.load(open(a.caveats)) if os.path.exists(a.caveats) else []}
    # Pareto figures (x = CPU-h per 100M pairs): 16 threads and 4 threads, same metric
    for label, rows in (("16t", pooled16), ("4t", pooled4)):
        if not rows:
            continue
        doc = {"metrics": M, "resources": rows}
        frag["figures"].append(contract.pareto_figure(doc, "evp_accuracy_score", "pooled", "cpu_h", fig_id=f"pareto_accuracy_vs_cpu_h_{label}"))
        frag["figures"].append(contract.pareto_figure(doc, "evp_accuracy_score", "pooled", "wall_s", fig_id=f"pareto_accuracy_vs_wall_s_{label}"))
    # the 'pooled' resource list used by validators must be unique per tool: keep 16 threads pooled, drop 4-thread pooled rows from resources
    frag["resources"] = resources + pooled16
    if pooled4:
        frag["resources_4t_pooled"] = pooled4
    json.dump(frag, open(a.out, "w"), indent=1)
    if a.check:
        doc, errs = check(frag, a.cohort)
        print("validate:", "OK" if not errs else "\n".join(errs))
        for m in doc["metrics"]:
            print(m["id"], m["dataset"], m["singlet"]["value"], "->", m["pass"])


if __name__ == "__main__":
    main()
