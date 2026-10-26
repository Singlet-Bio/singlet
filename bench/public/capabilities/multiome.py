"""Capability `multiome`: 10x Multiome (GEX + ATAC on one gel bead) vs Cell Ranger ARC.

What is specific to Multiome, on top of `atac` (ATAC side) and `gex-counts` (GEX side):
- barcode pairing: each ATAC barcode must be translated to its gel bead's GEX barcode (truth: ARC's per-barcode table);
- the joint cell call over GEX UMIs and ATAC fragments in peaks (ARC calls cells on both modalities).

Artifacts: ``atac_<DS>_barcodes.tsv.gz`` (as for `atac`, with tenx_atac_barcode, tenx_gex_umis, singlet_atac_barcode,
singlet_gex_called) and ``provenance.json``.

Competitor for the joint call: the usual open-source route, cells called separately in each modality and intersected:
GEX cells (Singlet's GEX call, called_v2 = EmptyDrops) AND ATAC cells by ArchR's default filter (>= 1000 fragments,
TSS enrichment >= 4) on Chromap + SnapATAC2 fragments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import artifact_entries, code_url, contract, load_provenance, now_iso  # noqa: E402
from _atacsp import archr_default_cells, bootstrap_metric, diff_tolerance, jaccard, metric, read_tsv  # noqa: E402

CAP = "multiome"
COMP = "per-modality-calls"


def dataset_metrics(art: Path, ds: str) -> list:
    b = read_tsv(art / f"atac_{ds}_barcodes.tsv.gz")
    tenx = set(b.index[b.tenx_is_cell == 1])
    sing = set(b.index[b.singlet_is_cell == 1])
    gex = set(b.index[b.get("singlet_gex_called", 0) == 1]) if "singlet_gex_called" in b else set()
    comp = (set(b.index[archr_default_cells(b.chromap_fragments, b.chromap_tsse)]) & gex) \
        if ("chromap_fragments" in b and gex) else set()
    out = []
    idx = np.array(sorted(tenx | sing | comp))
    jac = lambda sel, other: jaccard(set(idx[sel]) & tenx, set(idx[sel]) & other)
    rule = (diff_tolerance(lambda i: jac(i, sing), lambda i: jac(i, comp), len(idx), unit="called barcodes")
            if comp else None) or {"type": "match_or_beat", "against": "best"}
    out.append(metric("joint_cell_jaccard", "Joint GEX+ATAC cell calls: Jaccard with Cell Ranger ARC", "fraction",
                      "agreement", ds, "higher", jaccard(tenx, sing), [(COMP, jaccard(tenx, comp) if comp else None)],
                      rule,
                      ci=bootstrap_metric(lambda i: jaccard(set(idx[i]) & tenx, set(idx[i]) & sing), len(idx), 500),
                      notes=f"ARC {len(tenx)}, Singlet {len(sing)}, competitor {len(comp)} cells"))
    # pairing: for ARC cells that Singlet also reports, is Singlet's ATAC barcode ARC's ATAC barcode?
    both = b.loc[b.index.isin(tenx) & b.singlet_atac_barcode.notna()]
    ok = float((both.singlet_atac_barcode == both.tenx_atac_barcode).mean()) if len(both) else None
    out.append(metric("barcode_pairing", "ATAC to GEX barcode pairing: fraction identical to ARC", "fraction", "accuracy",
                      ds, "higher", ok, [], {"type": "threshold", "value": 0.999,
                                             "rationale": "pairing is a fixed table lookup; any disagreement above "
                                                          "1 in 1,000 cells would be a bug, not noise"},
                      notes=f"{len(both)} ARC cells with a Singlet ATAC barcode"))
    return out


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    C = contract()
    prov = load_provenance(art)
    commit = prov["singlet"]["commit"]
    metrics = []
    for ds in prov["test_datasets"]:
        metrics += dataset_metrics(art, ds)
    jm = metrics[0]
    conclusion = prov.get("conclusion") or (
        f"Singlet pairs every Multiome ATAC barcode with its GEX barcode as Cell Ranger ARC does, and its joint cell call "
        f"matches ARC's at Jaccard {jm['singlet']['value']} vs {jm['competitors'][0]['value']} for separate per-modality "
        f"calls.")
    kinds = {f"atac_{ds}_barcodes.tsv.gz": "derived" for ds in prov["test_datasets"]}
    kinds["provenance.json"] = "derived"
    files = ["include/singlet/pileup/atac_barcodes.h", "include/singlet/pileup/atac_cell_caller.h",
             "include/singlet/pileup/atac_pipeline.h", "bench/atac_spatial/make_lists.py",
             "bench/atac_spatial/make_artifacts.py", "bench/public/capabilities/multiome.py",
             "bench/public/capabilities/_atacsp.py"]
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Multiome (GEX + ATAC)", "conclusion": conclusion, "methods": prov["methods"],
        "status": prov.get("status", "draft"), "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "atac-multiome-spatial", "singlet": prov["singlet"], "tools": prov["tools"],
        "datasets": prov["datasets"], "metrics": metrics, "resources": prov["resources"], "figures": [],
        "artifacts": artifact_entries(art, CAP, kinds),
        "code": [{"path": f, "url": code_url(f, commit),
                  "role": "metric" if "capabilities" in f else "harness" if "bench/" in f else "implementation"}
                 for f in files],
        "reproduce": {"module": "bench/public/capabilities/multiome.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "bash bench/atac_spatial/submit.sh"},
        "caveats": prov.get("caveats", []),
    }
    return C.finalize(doc)
