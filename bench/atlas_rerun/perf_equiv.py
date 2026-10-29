"""Are optimized runs equivalent to the present one? Compare every matrix member, the barcodes
and the pileup stats of each perf config against the reference config.

    python perf_equiv.py <perf_dataset_dir> [reference=t4_r1]

For each member (exon, intron, ambiguous, splice junctions, mt heteroplasmy) the two matrices are
aligned on feature name and barcode, then: shared / only-in-one barcodes and features, total
counts, entries that differ and the sum of absolute differences. Identical = 0 differing entries
over identical barcode and feature sets.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

MEMBERS = ["exon_counts", "intron_counts", "ambiguous_counts", "sj_counts", "vdj_gene_usage"]


def lines(p: Path) -> list[str]:
    with gzip.open(p, "rt") as f:
        return [x.rstrip("\n") for x in f]


def load(run: Path, m: str):
    mtx = run / f"{m}.mtx.gz"
    if not mtx.exists():
        return None
    x = sp.csr_matrix(scipy.io.mmread(str(mtx)))  # features x barcodes
    return x, lines(run / f"{m}_features.tsv.gz"), lines(run / f"{m}_barcodes.tsv.gz")


def compare(a, b) -> dict:
    (xa, fa, ba), (xb, fb, bb) = a, b
    fs, bs = sorted(set(fa) & set(fb)), sorted(set(ba) & set(bb))
    ia_f = {f: i for i, f in enumerate(fa)}; ib_f = {f: i for i, f in enumerate(fb)}
    ia_b = {c: i for i, c in enumerate(ba)}; ib_b = {c: i for i, c in enumerate(bb)}
    A = xa[[ia_f[f] for f in fs]][:, [ia_b[c] for c in bs]]
    B = xb[[ib_f[f] for f in fs]][:, [ib_b[c] for c in bs]]
    d = abs(A - B)
    return {
        "barcodes_ref": len(ba), "barcodes_run": len(bb), "barcodes_shared": len(bs),
        "features_ref": len(fa), "features_run": len(fb), "features_shared": len(fs),
        "total_ref": float(xa.sum()), "total_run": float(xb.sum()),
        "total_ratio_on_shared": float(B.sum() / max(A.sum(), 1)),
        "entries_differing": int((d > 0).sum()), "abs_diff": float(d.sum()),
        "identical": bool(len(ba) == len(bb) == len(bs) and len(fa) == len(fb) == len(fs) and d.sum() == 0),
    }


def run_dirs(base: Path) -> dict[str, Path]:
    out = {}
    for d in sorted(base.iterdir()):
        if (d / "exon_counts.mtx.gz").exists():
            out[d.name] = d
        for c in sorted(d.glob("c[0-9]")):
            if (c / "exon_counts.mtx.gz").exists():
                out[f"{d.name}/{c.name}"] = c
    return out


def main():
    base = Path(sys.argv[1]); ref_name = sys.argv[2] if len(sys.argv) > 2 else "t4_r1"
    runs = run_dirs(base)
    ref = runs[ref_name]
    ref_m = {m: load(ref, m) for m in MEMBERS}
    ref_stats = json.loads((ref / "pileup_stats.json").read_text())
    res = {}
    for name, d in runs.items():
        if name == ref_name:
            continue
        r = {}
        for m in MEMBERS:
            b = load(d, m)
            if ref_m[m] is None or b is None:
                r[m] = "missing in ref" if ref_m[m] is None else "missing in run"
                continue
            r[m] = compare(ref_m[m], b)
        st = json.loads((d / "pileup_stats.json").read_text()) if (d / "pileup_stats.json").exists() else {}
        r["stats_diff"] = {k: (ref_stats.get(k), st.get(k)) for k in
                           ("total_reads", "barcoded_reads", "exon_hits", "intron_hits", "umi_unique",
                            "molecules_spliced", "molecules_unspliced", "molecules_ambiguous", "sj_hits",
                            "mt_pileup_bases", "n_barcodes")
                           if ref_stats.get(k) != st.get(k)}
        res[name] = r
        ok = all(isinstance(v, dict) and v.get("identical") for k, v in r.items() if k in MEMBERS and isinstance(v, dict))
        print(f"{name}: {'IDENTICAL' if ok else 'DIFFERS'}", flush=True)
        if not ok:
            for m in MEMBERS:
                v = r[m]
                if isinstance(v, dict) and not v["identical"]:
                    print(f"   {m}: barcodes {v['barcodes_ref']}/{v['barcodes_run']} shared {v['barcodes_shared']}, "
                          f"features {v['features_ref']}/{v['features_run']}, total ratio {v['total_ratio_on_shared']:.6f}, "
                          f"differing entries {v['entries_differing']}, |diff| {v['abs_diff']:.0f}")
            print("   stats:", r["stats_diff"])
    (base / "equivalence.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
