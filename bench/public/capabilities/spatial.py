"""Capability `spatial`: 10x Visium spot counts and tissue detection vs Space Ranger.

Artifacts (bench/atac_spatial/make_artifacts.py spatial):
- ``spatial_<DS>_spots.tsv.gz``: the 4,992 slide spots with array coordinates, Space Ranger's image-based in_tissue,
  Singlet's count-based in_tissue, and per tool (spaceranger, singlet, starsolo) UMIs, genes and the per-spot expression
  profile correlation with Space Ranger (derived: needs the matrices).
- ``spatial_<DS>_genes.tsv.gz``: per gene, molecules over Space Ranger's tissue spots for each tool.
- ``provenance.json``.

Competitor: STARsolo 2.7.11b with the visium-v1 spot list (Space Ranger 1.x-equivalent exonic counting) on the same
FASTQs and GRCm39-2024-A. Tissue detection competitor: an Otsu threshold on log10 UMIs per spot, the usual count-based
fallback when the image is missing (no tool for counts-only tissue detection is in common use). Truth: Space Ranger 1.1
published outputs (counts; tissue from the H&E image).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import artifact_entries, code_url, contract, load_provenance, now_iso  # noqa: E402
from _atacsp import bootstrap_metric, cohen_kappa, diff_tolerance, metric, otsu_threshold, read_tsv, spearman  # noqa: E402

CAP = "spatial"
MATCH = {"type": "match_or_beat", "against": "best"}


def dataset_metrics(art: Path, ds: str) -> tuple[list, dict]:
    s = read_tsv(art / f"spatial_{ds}_spots.tsv.gz")
    g = read_tsv(art / f"spatial_{ds}_genes.tsv.gz")
    tis = s[s.spaceranger_in_tissue == 1]
    has_solo = "starsolo_umis" in s
    out = []

    def cmp(id_, name, unit, col_s, col_c, ref, frame, stat, units, notes=None):
        a, c, r = (frame[x].to_numpy(float) if x in frame else None for x in (col_s, col_c, ref))
        f = (lambda v: lambda i: stat(v[i], r[i])) if r is not None else (lambda v: lambda i: stat(v[i]))
        sv = f(a)(np.arange(len(frame))); cv = f(c)(np.arange(len(frame))) if c is not None else None
        rule = (diff_tolerance(f(a), f(c), len(frame), unit=units) if c is not None else None) or MATCH
        return metric(id_, name, unit, "accuracy", ds, "higher", sv, [("starsolo", cv)], rule, notes=notes)

    out.append(cmp("spot_umis_spearman", "UMIs per tissue spot: Spearman with Space Ranger", "rho", "singlet_umis",
                   "starsolo_umis", "spaceranger_umis", tis, spearman, "tissue spots", f"{len(tis)} Space Ranger tissue spots"))
    out.append(cmp("spot_genes_spearman", "Genes per tissue spot: Spearman with Space Ranger", "rho", "singlet_genes",
                   "starsolo_genes", "spaceranger_genes", tis, spearman, "tissue spots"))
    out.append(cmp("spot_profile_pearson", "Per-spot expression profile vs Space Ranger: median Pearson (log CPM)", "r",
                   "singlet_profile_r_vs_spaceranger", "starsolo_profile_r_vs_spaceranger", "_none_", tis,
                   lambda v: float(np.nanmedian(v)), "tissue spots"))
    out.append(cmp("gene_totals_spearman", "Molecules per gene over tissue spots: Spearman with Space Ranger", "rho",
                   "singlet_tissue_total", "starsolo_tissue_total", "spaceranger_tissue_total", g, spearman, "genes",
                   f"{len(g):,} genes"))
    # tissue detection, all slide spots
    y = s.spaceranger_in_tissue.to_numpy(int)
    x = np.log10(s.singlet_umis.to_numpy(float) + 1)
    otsu = (x > otsu_threshold(x)).astype(int)
    yi = s.singlet_in_tissue.to_numpy(int)
    rule = diff_tolerance(lambda i: cohen_kappa(y[i], yi[i]), lambda i: cohen_kappa(y[i], otsu[i]), len(y),
                          unit="slide spots") or MATCH
    out.append(metric("tissue_kappa", "Spots under tissue: Cohen's kappa with Space Ranger (image)", "kappa",
                      "agreement", ds, "higher", cohen_kappa(y, yi), [("otsu-umi-threshold", cohen_kappa(y, otsu))],
                      rule, ci=bootstrap_metric(lambda i: cohen_kappa(y[i], yi[i]), len(y), 500),
                      notes=f"Space Ranger {int(y.sum())}, Singlet {int(yi.sum())}, Otsu {int(otsu.sum())} tissue spots "
                            f"of {len(y)}; agreement Singlet {np.mean(y == yi):.3f}, Otsu {np.mean(y == otsu):.3f}"))
    summ = {"ds": ds, "spots": s, "otsu": otsu}
    return out, summ


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    C = contract()
    prov = load_provenance(art)
    commit = prov["singlet"]["commit"]
    metrics, summ = [], []
    for ds in prov["test_datasets"]:
        m, s = dataset_metrics(art, ds); metrics += m; summ.append(s)
    s0 = summ[0]["spots"]
    figures = [{
        "id": "tissue_map", "title": "Spots under tissue: Space Ranger (image) vs Singlet (counts)", "kind": "scatter",
        "caption": "Each Visium spot at its array position. Agreement in grey; Singlet-only and Space Ranger-only spots "
                   "highlighted.",
        "x": {"label": "array column", "scale": "linear"}, "y": {"label": "array row", "scale": "linear"},
        "series": [{"name": n, "tool": "singlet" if "Singlet" in n else "spaceranger",
                    "x": s0.array_col[m].astype(int).tolist(), "y": (-s0.array_row[m]).astype(int).tolist()}
                   for n, m in (("both tissue", (s0.spaceranger_in_tissue == 1) & (s0.singlet_in_tissue == 1)),
                                ("Singlet only", (s0.spaceranger_in_tissue == 0) & (s0.singlet_in_tissue == 1)),
                                ("Space Ranger only", (s0.spaceranger_in_tissue == 1) & (s0.singlet_in_tissue == 0)))],
    }]
    def val(mid, ds):
        m = next(m for m in metrics if m["id"] == mid and m["dataset"] == ds)
        return m["singlet"]["value"], m["competitors"][0]["value"]
    parts = []
    for ds in prov["test_datasets"]:
        (us, uc), (ps, pc), (ks, kc) = val("spot_umis_spearman", ds), val("spot_profile_pearson", ds), val("tissue_kappa", ds)
        parts.append(f"{ds}: UMIs per spot rho {us} vs {uc}, profiles r {ps} vs {pc} (STARsolo), tissue kappa {ks} vs {kc} (Otsu)")
    conclusion = prov.get("conclusion") or (
        "Singlet's image-free tissue call beats a count threshold on both slides, but its spot counts trail STARsolo slightly "
        "(shared GEX counting; mouse mito genes in the chr1 NUMT are lost). " + "; ".join(parts) + ".")
    kinds = {}
    for ds in prov["test_datasets"]:
        kinds.update({f"spatial_{ds}_spots.tsv.gz": "derived", f"spatial_{ds}_genes.tsv.gz": "derived"})
    kinds["provenance.json"] = "derived"
    files = ["include/singlet/pileup/visium_tissue.h", "src/pipeline/singlet.cpp", "bench/atac_spatial/make_artifacts.py",
             "bench/atac_spatial/run_starsolo_visium.sbatch", "bench/public/capabilities/spatial.py",
             "bench/public/capabilities/_atacsp.py"]
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Spatial transcriptomics (Visium)", "conclusion": conclusion, "methods": prov["methods"],
        "status": prov.get("status", "draft"), "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "atac-multiome-spatial", "singlet": prov["singlet"], "tools": prov["tools"],
        "datasets": prov["datasets"], "metrics": metrics, "resources": prov["resources"], "figures": figures,
        "artifacts": artifact_entries(art, CAP, kinds),
        "code": [{"path": f, "url": code_url(f, commit),
                  "role": "metric" if "capabilities" in f else "harness" if "bench/" in f else "implementation"}
                 for f in files],
        "reproduce": {"module": "bench/public/capabilities/spatial.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "bash bench/atac_spatial/submit.sh"},
        "caveats": prov.get("caveats", []),
    }
    for m in metrics:
        if m["id"] in ("spot_umis_spearman", "spot_profile_pearson"):
            doc["figures"].append(C.pareto_figure(doc, m["id"], m["dataset"], "cpu_h"))
    doc["reach"] = [
        {"tool": t, "output": o, "supported": s, "note": n} for t, o, s, n in (
            ("singlet", "spot counts, spot coordinates", "yes", "visium-v1 spot list"),
            ("singlet", "tissue detection without the image", "yes", "count mixture + hexagonal smoothing"),
            ("singlet", "spliced / unspliced, junctions, mtDNA per spot", "yes", "same pipeline as single-cell"),
            ("starsolo", "spot counts", "yes", "needs the spot list"),
            ("starsolo", "tissue detection", "no", ""),
            ("spaceranger", "spot counts, tissue detection from the image", "yes", "10x EULA; published outputs used"))]
    return C.finalize(doc)
