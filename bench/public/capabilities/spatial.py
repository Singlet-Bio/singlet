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
from _atacsp import bootstrap_metric, cohen_kappa, metric, otsu_threshold, read_tsv, spearman  # noqa: E402

CAP = "spatial"
MATCH = {"type": "match_or_beat", "against": "best"}


def dataset_metrics(art: Path, ds: str) -> tuple[list, dict]:
    s = read_tsv(art / f"spatial_{ds}_spots.tsv.gz")
    g = read_tsv(art / f"spatial_{ds}_genes.tsv.gz")
    tis = s[s.spaceranger_in_tissue == 1]
    has_solo = "starsolo_umis" in s
    out = []
    out.append(metric("spot_umis_spearman", "UMIs per tissue spot: Spearman with Space Ranger", "rho", "accuracy", ds,
                      "higher", spearman(tis.singlet_umis, tis.spaceranger_umis),
                      [("starsolo", spearman(tis.starsolo_umis, tis.spaceranger_umis) if has_solo else None)], MATCH,
                      notes=f"{len(tis)} Space Ranger tissue spots"))
    out.append(metric("spot_genes_spearman", "Genes per tissue spot: Spearman with Space Ranger", "rho", "accuracy", ds,
                      "higher", spearman(tis.singlet_genes, tis.spaceranger_genes),
                      [("starsolo", spearman(tis.starsolo_genes, tis.spaceranger_genes) if has_solo else None)], MATCH))
    out.append(metric("spot_profile_pearson", "Per-spot expression profile vs Space Ranger: median Pearson (log CPM)",
                      "r", "accuracy", ds, "higher", float(np.nanmedian(tis.singlet_profile_r_vs_spaceranger)),
                      [("starsolo", float(np.nanmedian(tis.starsolo_profile_r_vs_spaceranger)) if has_solo else None)],
                      MATCH))
    out.append(metric("gene_totals_spearman", "Molecules per gene over tissue spots: Spearman with Space Ranger", "rho",
                      "accuracy", ds, "higher", spearman(g.singlet_tissue_total, g.spaceranger_tissue_total),
                      [("starsolo", spearman(g.starsolo_tissue_total, g.spaceranger_tissue_total) if has_solo else None)],
                      MATCH, notes=f"{len(g):,} genes"))
    # tissue detection, all slide spots
    y = s.spaceranger_in_tissue.to_numpy(int)
    x = np.log10(s.singlet_umis.to_numpy(float) + 1)
    otsu = (x > otsu_threshold(x)).astype(int)
    yi = s.singlet_in_tissue.to_numpy(int)
    out.append(metric("tissue_kappa", "Spots under tissue: Cohen's kappa with Space Ranger (image)", "kappa",
                      "agreement", ds, "higher", cohen_kappa(y, yi), [("otsu-umi-threshold", cohen_kappa(y, otsu))],
                      MATCH, ci=bootstrap_metric(lambda i: cohen_kappa(y[i], yi[i]), len(y), 500),
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
    km = next(m for m in metrics if m["id"] == "tissue_kappa")
    um = next(m for m in metrics if m["id"] == "spot_umis_spearman")
    conclusion = prov.get("conclusion") or (
        f"On the 10x Visium mouse brain section, Singlet's spot counts track Space Ranger (Spearman "
        f"{um['singlet']['value']} vs {um['competitors'][0]['value']} for STARsolo), and its image-free tissue call agrees "
        f"with Space Ranger's image-based one at kappa {km['singlet']['value']} (Otsu threshold "
        f"{km['competitors'][0]['value']}).")
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
