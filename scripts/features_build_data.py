#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the data files of ``singlet.features`` (python/singlet/features/data/).

    python scripts/features_build_data.py --msigdb-dir DIR --bundles human=A.singlet,B.singlet mouse=C.singlet [--max-cells 4000]

Writes, per organism:
  abundance_<organism>.parquet   gene_id, symbol, mean (mean log1p CP10k over the reference cells), bin (50 equal-count bins of
                                 detected genes); metadata names the reference samples. Null sets are drawn from it.
and the gene-set collections:
  hallmark.json                  MSigDB Hallmark (human H, mouse MH; release in --release), symbols mapped to the reference's
                                 Ensembl ids. Licence CC BY 4.0 (gsea-msigdb.org/gsea/msigdb_license_terms.jsp, checked 2026-10-23).
  cell_cycle.json                Tirosh et al. 2016 S and G2/M genes (Seurat cc.genes.updated.2019 names); mouse: the same
                                 symbols in mouse case, kept where the mouse reference has them.

MSigDB files (download once into --msigdb-dir):
  https://data.broadinstitute.org/gsea-msigdb/msigdb/release/<release>.Hs/h.all.v<release>.Hs.json
  https://data.broadinstitute.org/gsea-msigdb/msigdb/release/<release>.Mm/mh.all.v<release>.Mm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python"))
DATA = ROOT / "python" / "singlet" / "features" / "data"

S_GENES = (
    "MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 "
    "SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8"
).split()
G2M_GENES = (
    "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB "
    "BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 "
    "HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA"
).split()


def reference_cells(paths, max_cells, seed=0):
    """(gene_ids, symbols, mean log1p CP10k, samples) over up to max_cells called cells per sample of the given bundles."""
    from singlet.bundle import SingletBundle

    rng = np.random.default_rng(seed)
    ids = syms = None
    total = None
    n = 0
    used = []
    for p in paths:
        b = SingletBundle(p)
        try:
            for gsm in b.manifest["gsm_ids"]:
                try:
                    a = b.raw_counts(gsm)
                except Exception as e:  # a sample without matrices
                    print(f"  skip {gsm}: {e}", file=sys.stderr)
                    continue
                vid = np.array([str(v).split(".")[0] for v in a.var_names], dtype=object)
                if ids is None:
                    ids, syms = vid, np.array([str(s) for s in a.var["gene_name"]], dtype=object)
                    total = np.zeros(len(ids))
                elif len(vid) != len(ids) or (vid != ids).any():
                    raise SystemExit(f"{p} {gsm}: gene table differs from the first sample's")
                X = sps.csr_matrix(a.X, dtype=np.float32)
                det = np.diff(X.indptr)
                keep = np.flatnonzero(det >= 200)
                if len(keep) > max_cells:
                    keep = np.sort(rng.choice(keep, max_cells, replace=False))
                X = X[keep]
                tot = np.asarray(X.sum(1)).ravel()
                X = sps.diags(1e4 / np.maximum(tot, 1)) @ X
                X.data = np.log1p(X.data)
                total += np.asarray(X.sum(0)).ravel()
                n += X.shape[0]
                used.append(f"{b.manifest.get('gse_id')}/{gsm}:{X.shape[0]}")
                print(f"  {gsm}: {X.shape[0]} cells", file=sys.stderr)
        finally:
            b.close()
    return ids, syms, total / max(n, 1), used, n


def write_abundance(organism, ids, syms, mean, used, n, build):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from singlet.features.rank import abundance_bins

    det = mean > 0
    bins = abundance_bins(mean[det], 50)
    t = pa.table(
        {
            "gene_id": pa.array(list(ids[det])),
            "symbol": pa.array(list(syms[det])),
            "mean": pa.array(mean[det].astype(np.float32)),
            "bin": pa.array(bins.astype(np.int16)),
        }
    ).replace_schema_metadata(
        {
            b"organism": organism.encode(),
            b"reference_build": build.encode(),
            b"n_cells": str(n).encode(),
            b"samples": ";".join(used).encode(),
            b"statistic": b"mean log1p CP10k over called cells with >= 200 detected genes; 50 equal-count bins of detected genes",
        }
    )
    DATA.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, DATA / f"abundance_{organism}.parquet", compression="zstd")
    print(f"abundance_{organism}: {int(det.sum())} genes, {n} cells", file=sys.stderr)


def symbol_map(ids, syms):
    m = {}
    for g, s in zip(ids, syms):
        m.setdefault(str(s).upper(), str(g))
    return m


def name_of(msigdb_name):
    n = re.sub(r"^HALLMARK_", "", msigdb_name).lower()
    assert re.match(r"^[a-z][a-z0-9_]{0,47}$", n), n
    return n


def build_hallmark(msigdb_dir: Path, release: str, refs: dict):
    files = {"human": f"h.all.v{release}.Hs.json", "mouse": f"mh.all.v{release}.Mm.json"}
    docs = {o: json.loads((msigdb_dir / f).read_text(encoding="utf-8")) for o, f in files.items()}
    sets = {}
    unmapped = {}
    for msig in docs["human"]:
        name = name_of(msig)
        entry = {
            "source_name": msig,
            "source_id": {},
            "pmid": docs["human"][msig].get("pmid", ""),
            "genes": {},
        }
        for org, doc in docs.items():
            if org not in refs or msig not in doc:
                continue
            d = doc[msig]
            entry["source_id"][org] = d["systematicName"]
            smap = refs[org]
            pairs, miss = [], []
            for s in dict.fromkeys(d["geneSymbols"]):
                g = smap.get(s.upper())
                (pairs.append([g, s]) if g else miss.append(s))
            entry["genes"][org] = pairs
            if miss:
                unmapped.setdefault(org, {})[name] = miss
        sets[name] = entry
    sha = {o: hashlib.sha256((msigdb_dir / f).read_bytes()).hexdigest() for o, f in files.items()}
    out = {
        "collection": "hallmark",
        "release": release,
        "licence": "CC-BY-4.0",
        "attribution": "MSigDB Hallmark gene sets (Liberzon et al. 2015, Cell Systems 1:417); mouse MH collection (MSigDB, "
        "orthologs from the Alliance of Genome Resources). (c) Broad Institute, MIT and Regents of the University of California, "
        "CC BY 4.0.",
        "sources": {
            o: {
                "url": f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{release}.{'Hs' if o == 'human' else 'Mm'}/{f}",
                "sha256": sha[o],
            }
            for o, f in files.items()
        },
        "unmapped": unmapped,
        "sets": sets,
    }
    (DATA / "hallmark.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for o in refs:
        n_un = sum(len(v) for v in unmapped.get(o, {}).values())
        n_all = sum(len(s["genes"].get(o, [])) for s in sets.values()) + n_un
        print(f"hallmark {o}: {n_all - n_un}/{n_all} symbols mapped", file=sys.stderr)


def build_cell_cycle(refs: dict):
    sets = {}
    for name, genes in (("s_phase", S_GENES), ("g2m_phase", G2M_GENES)):
        e = {"source_name": name, "source_id": {}, "genes": {}}
        for org, smap in refs.items():
            pairs = []
            for s in genes:
                sym = s if org == "human" else s[0] + s[1:].lower()
                g = smap.get(sym.upper())
                if g:
                    pairs.append([g, sym])
            e["genes"][org] = pairs
        sets[name] = e
    out = {
        "collection": "cell_cycle",
        "release": "tirosh2016-seurat2019",
        "licence": "none: gene names listed in a publication, not a licensed collection",
        "attribution": "Tirosh et al. 2016, Science 352:189",
        "sources": {},
        "sets": sets,
    }
    (DATA / "cell_cycle.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for o in refs:
        print(
            f"cell_cycle {o}: S {len(sets['s_phase']['genes'][o])}/{len(S_GENES)}, G2M {len(sets['g2m_phase']['genes'][o])}/{len(G2M_GENES)}",
            file=sys.stderr,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--msigdb-dir", type=Path, required=True)
    ap.add_argument("--release", default="2026.2")
    ap.add_argument("--bundles", nargs="+", required=True, help="organism=path[,path...]")
    ap.add_argument("--max-cells", type=int, default=4000, help="reference cells per sample")
    a = ap.parse_args(argv)
    refs = {}
    for spec in a.bundles:
        org, _, paths = spec.partition("=")
        paths = [Path(p) for p in paths.split(",") if p]
        from singlet.bundle import SingletBundle

        b = SingletBundle(paths[0])
        build = str(b.manifest.get("reference_build") or "")
        b.close()
        ids, syms, mean, used, n = reference_cells(paths, a.max_cells)
        write_abundance(org, ids, syms, mean, used, n, build)
        refs[org] = symbol_map(ids, syms)
    build_hallmark(a.msigdb_dir, a.release, refs)
    build_cell_cycle(refs)


if __name__ == "__main__":
    main()
