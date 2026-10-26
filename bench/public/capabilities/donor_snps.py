"""donor-snps: accuracy of the donor genotypes a tool infers from a pool's reads, scored against truth genotypes.
METRICS ONLY: the genotypes, the SNPs and the per-site allele counts stay private on Clipper. The artifact is a table of
counts (a 3x3 genotype-class table per pool, tool and donor) written by bench/public/genotype/lib/snp_concordance.py.

    python bench/public/capabilities/donor_snps.py ARTIFACT_DIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402
from _gcommon import (SINGLET_BINARY, SINGLET_BINARY_SHA256, SINGLET_COMMIT, artifact_entry, boot, code_url, now,  # noqa: E402
                      paired_halfwidth, r6, wilson)

CAP = "donor-snps"
ART_BASE = "https://data.singlet.bio/benchmarking/1.0.0/donor-snps"
DEV = "kang2018_ctrl"
POOL_DATASET = {"kang2018_ctrl": "kang2018_ctrl", "kang2018_stim": "kang2018_stim", "kang2018_b1_a": "kang2018_b1_a",
                "kang2018_b1_b": "kang2018_b1_b", "kang2018_b1_c": "kang2018_b1_c", "lcl_hg00362": "lcl_gse230239",
                "lcl_hg01277": "lcl_gse230239", "osorio2019_lcl_mix": "osorio2019_lcl_mix"}
DATASETS = {
    "kang2018_ctrl": ("Kang 2018 batch 2 control lane (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560248"], "srr": ["SRR5398238"]},
                      "genotypes of each donor from the same donors' cells in the IFN-beta lane (kang2018_stim), labelled by the authors' demuxlet calls and genotyped from pooled reads"),
    "kang2018_stim": ("Kang 2018 batch 2 IFN-beta lane (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560249"], "srr": ["SRR5398239"]},
                      "genotypes of each donor from the same donors' cells in the control lane (kang2018_ctrl)"),
    "kang2018_b1_a": ("Kang 2018 batch 1 lane A (4 donors)", {"gse": "GSE96583", "gsm": ["GSM2560245"], "srr": ["SRR5398235"]},
                      "genotypes of each donor from the same donors' cells in lane C (kang2018_b1_c)"),
    "kang2018_b1_b": ("Kang 2018 batch 1 lane B (4 donors)", {"gse": "GSE96583", "gsm": ["GSM2560246"], "srr": ["SRR5398236"]},
                      "genotypes of each donor from the same donors' cells in lane C (kang2018_b1_c)"),
    "kang2018_b1_c": ("Kang 2018 batch 1 lane C (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560247"], "srr": ["SRR5398237"]},
                      "genotypes of each donor from the same donors' cells in lanes A and B"),
    "lcl_gse230239": ("1000 Genomes LCL libraries HG00362 and HG01277 (one donor each), 10x 3' v3, first run of each library",
                      {"gse": "GSE230239", "gsm": ["GSM7208744", "GSM7208745"], "srr": ["SRR24236333", "SRR24236331"]},
                      "1000 Genomes high-coverage GRCh38 callset genotypes of HG00362 and HG01277"),
    "osorio2019_lcl_mix": ("Osorio 2019 GM12878 + GM18502 1:1 mixture, 10x 3' v2", {"gse": "GSE126321", "gsm": ["GSM3596322"], "srr": ["SRR8551678"]},
                           "1000 Genomes high-coverage GRCh38 callset genotypes of NA12878 and NA18502"),
}
TOOLS = {"singlet": "singlet", "vireo": "vireo", "souporcell": "souporcell", "cellsnp-pileup": "cellsnp-pileup"}
COMPETITORS = ["vireo", "souporcell", "cellsnp-pileup"]
CM = [f"t{t}_c{u}" for t in range(3) for u in range(3)]


def _stat(c: np.ndarray, key: str):
    n = c.sum()
    if key == "conc":
        return float(np.trace(c) / n) if n else np.nan
    if key == "nonref_conc":
        d = c[1:].sum()
        return float((c[1, 1] + c[2, 2]) / d) if d else np.nan
    if key == "false_alt":
        d = c[0].sum()
        return float((c[0, 1] + c[0, 2]) / d) if d else np.nan
    raise KeyError(key)


def _agg(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    c = df[CM].sum().values.reshape(3, 3).astype(float)
    n = c.sum()
    nonref = c[1:].sum()
    hom = c[0].sum()
    return dict(n=int(n), conc=float(np.trace(c) / n) if n else None,
                nonref_conc=float((c[1, 1] + c[2, 2]) / nonref) if nonref else None,
                false_alt=float((c[0, 1] + c[0, 2]) / hom) if hom else None, n_donors=int(len(df)))


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    d = pd.read_csv(art / "donor_snps_counts.tsv", sep="\t")
    prov = json.loads((art / "provenance.json").read_text())
    d["dataset"] = d["pool"].map(POOL_DATASET)
    d = d.dropna(subset=["dataset"])
    datasets = [k for k in DATASETS if k in set(d["dataset"])]
    held = [k for k in datasets if k != DEV]
    metrics = []

    def S(ds, tool, stratum):
        x = d[(d["dataset"].isin(ds)) & (d["tool"] == tool) & (d["stratum"] == stratum)]
        return _agg(x), x

    def M(id_, name, unit, cat, ds, direction, sing, comps, rule, ci=None, notes=None):
        if sing is None:
            return
        m = {"id": id_, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
             "singlet": {"value": r6(sing)}, "rule": rule}
        if ci:
            m["singlet"]["ci"] = [r6(ci[0]), r6(ci[1])]
        cl = [{"tool": k, "value": r6(v)} for k, v in comps.items() if v is not None]
        if cl:
            m["competitors"] = cl
        if notes:
            m["notes"] = notes
        metrics.append(m)

    rule = {"type": "match_or_beat", "against": "best"}

    def trule(sel, stratum, key, comps, direction):
        """match-or-beat with the paired bootstrap noise (over donors) of Singlet minus the best competitor as tolerance."""
        vals = {t: v for t, v in comps.items() if v is not None}
        if not vals:
            return dict(rule)
        best = (max if direction == "higher" else min)(vals, key=vals.get)
        a = d[(d["dataset"].isin(sel)) & (d["tool"] == "singlet") & (d["stratum"] == stratum)].set_index(["pool", "donor"])
        b = d[(d["dataset"].isin(sel)) & (d["tool"] == best) & (d["stratum"] == stratum)].set_index(["pool", "donor"])
        idx = a.index.intersection(b.index)
        if len(idx) < 4:
            return dict(rule)
        A = a.loc[idx][CM].values.reshape(-1, 3, 3); B = b.loc[idx][CM].values.reshape(-1, 3, 3)
        hw = paired_halfwidth(lambda i: _stat(A[i].sum(0), key) - _stat(B[i].sum(0), key), len(idx))
        if hw is None or hw <= 0:
            return dict(rule)
        return {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(hw, 4)},
                "rationale": f"{round(hw, 4)} = half-width of the paired bootstrap 95% CI (over {len(idx)} donors) of the difference between Singlet and the best competitor, {best}"}
    for stratum, tag, nm in (("all", "", "every genotype the tool emits"), ("dp10", "_dp10", "sites where the donor has >= 10 reads")):
        for ds in datasets + (["pooled"] if len(held) > 1 else []):
            sel = held if ds == "pooled" else [ds]
            a, x = S(sel, "singlet", stratum)
            if a is None:
                continue
            comps = {t: (S(sel, t, stratum)[0] or {}).get("conc") for t in COMPETITORS}
            ci = wilson(int(round(a["conc"] * a["n"])), a["n"]) if a["n"] else None
            M(f"genotype_concordance{tag}", f"Genotype concordance with truth, {nm}", "fraction", "accuracy", ds, "higher",
              a["conc"], comps, trule(sel, stratum, "conc", comps, "higher"), ci=ci,
              notes=("Pooled over the held-out pools (development lane excluded)." if ds == "pooled" else
                     "Development lane: Singlet's demultiplexing was tuned here (donor genotypes are a by-product)." if ds == DEV else None))
    for ds in datasets + (["pooled"] if len(held) > 1 else []):
        sel = held if ds == "pooled" else [ds]
        a, _ = S(sel, "singlet", "dp10")
        if a is None:
            continue
        for key, id_, nm, direction in (("nonref_conc", "nonref_concordance_dp10", "Concordance at truth non-reference sites, >= 10 reads", "higher"),
                                        ("false_alt", "false_alt_rate_dp10", "Truth homozygous-reference sites called non-reference, >= 10 reads", "lower")):
            comps = {t: (S(sel, t, "dp10")[0] or {}).get(key) for t in COMPETITORS}
            M(id_, nm, "fraction", "accuracy", ds, direction, a[key], comps, trule(sel, "dp10", key, comps, direction))
        comps = {t: ((S(sel, t, "dp10")[0] or {}).get("n") or None) for t in COMPETITORS}
        M("sites_genotyped_dp10", "Truth sites genotyped at >= 10 reads", "sites", "coverage", ds, "higher", float(a["n"]),
          {t: (float(v) if v else None) for t, v in comps.items()}, rule,
          notes="Counts of site-donor pairs; a tool that genotypes fewer sites at the same confidence is less useful.")

    resources = []
    for row in prov.get("resources", []):
        resources.append(row)
    doc = {
        "schema": contract.SCHEMA_CAPABILITY, "schema_version": contract.SCHEMA_VERSION, "capability": CAP,
        "title": "Donor SNP genotypes (metrics only)", "conclusion": _conclusion(d, datasets, held, S), "methods": METHODS,
        "status": "final", "generated_at": now(), "owner": "pipeline-benchmark",
        "singlet": {"version": "1.0.0", "commit": SINGLET_COMMIT, "branch": "prod/campaign", "binary": SINGLET_BINARY,
                    "binary_sha256": SINGLET_BINARY_SHA256},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": prov["singlet_command"]},
            {"name": "vireo", "version": "0.5.9", "role": "competitor", "url": "https://github.com/single-cell-genetics/vireo",
             "citation": "Huang et al. 2019, Genome Biol 20:273", "params": prov["vireo_command"]},
            {"name": "souporcell", "version": "2.5", "role": "competitor", "url": "https://github.com/wheaton5/souporcell",
             "citation": "Heaton et al. 2020, Nat Methods 17:615", "params": prov["souporcell_command"]},
            {"name": "cellsnp-pileup", "version": "1.2.3", "role": "competitor", "url": "https://github.com/single-cell-genetics/cellsnp-lite",
             "params": prov["pileup_command"]},
            {"name": "1000genomes-highcov", "version": "20220422", "role": "truth",
             "url": "https://www.internationalgenome.org/data-portal/data-collection/30x-grch38",
             "params": "phased SNV/INDEL/SV callset of 3,202 samples, GRCh38"},
            {"name": "kang-replicate-lane", "version": "n/a", "role": "truth",
             "params": "pooled reads of the donor's cells in a second library of the same donors, binomial genotyping (error 0.01, posterior >= 0.99)"},
        ],
        "datasets": [{"id": k, "title": DATASETS[k][0], "accessions": DATASETS[k][1], "truth": DATASETS[k][2], "data_kind": "real",
                      **({"notes": "development data (PR #36 tuned Singlet's demultiplexing on this lane)"} if k == DEV else {})} for k in datasets],
        "metrics": metrics, "resources": resources,
        "figures": _figures(d, datasets, S),
        "artifacts": [artifact_entry(art / f, f, "derived" if f.endswith(".tsv") else "singlet_output", ART_BASE)
                      for f in ("donor_snps_counts.tsv", "provenance.json")],
        "reach": [
            {"tool": "singlet", "output": "donor genotypes without reference genotypes", "supported": "yes", "note": "private/donor/donor*.vcf, never published"},
            {"tool": "vireo", "output": "donor genotypes without reference genotypes", "supported": "yes", "note": "GT_donors.vireo.vcf.gz"},
            {"tool": "souporcell", "output": "donor genotypes without reference genotypes", "supported": "yes", "note": "cluster_genotypes.vcf"},
            {"tool": "cellsnp-pileup", "output": "donor genotypes without reference genotypes", "supported": "partial",
             "note": "genotypes pooled reads of clusters someone else defined; one donor, or vireo's labels"},
        ],
        "code": [
            {"path": "include/singlet/pileup/donor_demux.h", "role": "implementation", "url": code_url("include/singlet/pileup/donor_demux.h", SINGLET_COMMIT)},
            {"path": "bench/public/capabilities/donor_snps.py", "role": "metric", "url": code_url("bench/public/capabilities/donor_snps.py", prov["code_commit"])},
            {"path": "bench/public/genotype", "role": "harness", "url": code_url("bench/public/genotype", prov["code_commit"], tree=True)},
        ],
        "reproduce": {"module": "bench/public/capabilities/donor_snps.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/public/genotype/hpc/export.sbatch snps"},
        "caveats": [
            "The genotypes, SNPs and per-site allele counts are private; the only artifact is a table of genotype-class counts.",
            "Kang truth genotypes are called from RNA reads of a second library of the same donors (no public Kang genotypes exist): "
            "they are expression-limited and share RNA artefacts (editing, allele-specific expression) with the tested pool, so "
            "concordance is a lower bound for accuracy at sites both libraries cover and says nothing about unexpressed sites.",
            "The 1000 Genomes truth is whole-genome sequencing of the cell-line donor; LCL-specific somatic differences from the "
            "reference individual exist but are small.",
            "The panel is 1000 Genomes SNPs with allele frequency >= 5% (7.35 M sites); every tool sees the same panel.",
            "Singlet emits a genotype at every covered panel site, however shallow, so its 'all' concordance includes 1-read sites; "
            "the dp10 metrics are the fair view for a user who filters.",
            "The plain pileup baseline (cellsnp-pileup) is not a donor-demultiplexing method; it genotypes pooled reads of vireo's clusters "
            "(all cells for one-donor libraries) with the model used for the Kang truth.",
        ],
    }
    return contract.finalize(doc)


METHODS = (
    "Truth is real. For the Kang lanes (no public genotypes exist) it is the genotype of each donor called from RNA reads of a "
    "second library of the same donors (batch 2: control and IFN-beta lanes; batch 1: lanes A and B against lane C), pooling the "
    "cells that the authors' demuxlet labelled as that donor, with a fixed binomial model (error 0.01, flat prior, posterior >= 0.99). "
    "For two cell-line pools it is the 1000 Genomes high-coverage genotypes of the cell lines' donors (HG00362, HG01277 as one-donor "
    "libraries; GM12878 + GM18502 as a real 1:1 mixture). Singlet runs end to end with its automatic donor count and the default "
    "1000G panel (AF >= 5%); its per-donor genotypes are those of its own clusters. Vireo (K given) and souporcell (k given) are "
    "run on the same BAM, cells and panel; cellsnp-pileup genotypes the pooled reads of vireo's clusters (all cells for one-donor "
    "libraries). Clusters are matched to truth donors by cell overlap (Kang) or by genotype agreement (1000G). Concordance is the "
    "fraction of sites with a truth genotype and a tool call at which the genotype class (0/0, 0/1, 1/1) is identical; the dp10 "
    "stratum keeps sites where the tool's own donor-level depth is >= 10. Non-reference concordance and false-alt rate split the "
    "same table by truth class. The rule is match-or-beat against the best competitor, zero tolerance. No genotype, SNP or "
    "per-site count is published."
)


def _conclusion(d, datasets, held, S):
    a, _ = S(held, "singlet", "dp10")
    comps = {t: (S(held, t, "dp10")[0] or {}).get("conc") for t in COMPETITORS}
    s = f"At sites with >= 10 reads, Singlet's donor genotypes agree with truth at {a['conc']:.3f}" if a else "No result"
    parts = [f"{t} {v:.3f}" for t, v in comps.items() if v is not None]
    return s + " over the held-out pools (" + ", ".join(parts) + "); truth is replicate-lane RNA genotypes (Kang) and 1000 Genomes genotypes (cell lines). Genotypes themselves are not published."


def _figures(d, datasets, S):
    cats = datasets
    series = []
    for t in TOOLS:
        ys = []
        for ds in cats:
            a, _ = S([ds], t, "dp10")
            ys.append(r6(a["conc"]) if a else None)
        if any(y is not None for y in ys):
            series.append({"name": t, "tool": t, "x": cats, "y": ys})
    return [{"id": "concordance_dp10_by_dataset", "title": "Genotype concordance at >= 10 reads", "kind": "grouped_bar",
             "caption": "Fraction of truth genotypes reproduced, by dataset and tool.", "x": {"label": "dataset", "categories": cats},
             "y": {"label": "concordance", "unit": "fraction", "min": 0.5, "max": 1}, "series": series}]


def main(argv=None):
    a = argv or sys.argv[1:]
    doc = compute(Path(a[0]))
    Path(a[1]).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(a[1], contract.capability_status(doc))


if __name__ == "__main__":
    main()
