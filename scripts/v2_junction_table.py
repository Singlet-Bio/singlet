#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the shared splice-junction table, version 1 (docs/format-v2.md section 12), from a census of the atlas.

Read-only against the public data host: legacy bundles are read by HTTP range (only each sample's ``sj_counts.1pz`` and
``cell_calls.tsv``), references come from GENCODE / Ensembl. Nothing is uploaded; the packs are written to ``--out``.

    v2_junction_table.py list    --out DIR                       # studies with a bundle (singlet.bio/api/gse), studies.tsv
    v2_junction_table.py census  --out DIR --shard I --shards N  # per-junction counts of one shard of studies -> shard_I.parquet
    v2_junction_table.py reduce  --out DIR                       # all shards -> census_<build>.parquet (one row per junction)
    v2_junction_table.py build   --out DIR --build GRCh38-2024-A # GTF + genome + census -> packs (core, rare, version) + report

Per junction the census keeps: studies, samples, UMIs and cells in called cells, UMIs in uncalled barcodes (the ambient noise model).
Tier rules (recorded in the version header): every annotated junction is core; an unannotated junction is core when it is seen in at
least ``--core-min-studies`` studies with ``--core-min-umi`` UMIs in called cells, a canonical motif and at most ``--max-ambient-frac``
of its UMIs in uncalled barcodes; other junctions seen in at least ``--rare-min-samples`` samples are rare; the rest stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from singlet.v2 import junctions as J  # noqa: E402
from singlet.v2 import packs as P  # noqa: E402

API = "https://singlet.bio/api/gse"
DATA = "https://data.singlet.bio"
UA = "singlet-junction-census/1.0 (+https://singlet.bio)"

# reference annotation and genome per build: the GENCODE releases behind 10x 2024-A, Ensembl 111 for the others (scgeo species.py)
REFS = {
    "GRCh38-2024-A": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.primary_assembly.annotation.gtf.gz",
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz",
    ),
    "GRCm39-2024-A": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.primary_assembly.annotation.gtf.gz",
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/GRCm39.primary_assembly.genome.fa.gz",
    ),
    "GRCz11": (
        "https://ftp.ensembl.org/pub/release-111/gtf/danio_rerio/Danio_rerio.GRCz11.111.gtf.gz",
        "https://ftp.ensembl.org/pub/release-111/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
    ),
    "Mmul_10": (
        "https://ftp.ensembl.org/pub/release-111/gtf/macaca_mulatta/Macaca_mulatta.Mmul_10.111.gtf.gz",
        "https://ftp.ensembl.org/pub/release-111/fasta/macaca_mulatta/dna/Macaca_mulatta.Mmul_10.dna.toplevel.fa.gz",
    ),
    "mRatBN7.2": (
        "https://ftp.ensembl.org/pub/release-111/gtf/rattus_norvegicus/Rattus_norvegicus.mRatBN7.2.111.gtf.gz",
        "https://ftp.ensembl.org/pub/release-111/fasta/rattus_norvegicus/dna/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz",
    ),
}
COLS = [
    "build",
    "chrom",
    "start",
    "end",
    "n_studies",
    "n_samples",
    "umi_called",
    "cells_called",
    "umi_uncalled",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


# ======================================================================================================== list
def cmd_list(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, page = [], 0  # the API's pages are 0-based
    while True:
        doc = get_json(f"{API}?has_bundle=1&page_size=500&page={page}&sort=id&asc=1")
        items = doc.get("data") or []
        rows += [
            (it["id"], int(it.get("n_gsm_done") or 0), it.get("r2_bundle_key") or "")
            for it in items
        ]
        if len(items) < 500:
            total = int(doc.get("total") or 0)
            break
        page += 1
    seen = set()
    rows = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]
    if total and len(rows) != total:
        log(
            f"WARNING: the API reports {total} studies, {len(rows)} listed (the catalog may be changing)"
        )
    (out / "studies.tsv").write_text("".join(f"{g}\t{n}\t{k}\n" for g, n, k in rows))
    log(f"{len(rows)} studies, {sum(n for _, n, _k in rows)} samples -> {out / 'studies.tsv'}")


# ======================================================================================================== census
def _called(cc: bytes):
    """Called barcodes from a legacy cell_calls.tsv (rows with is_cell true; a list of only called cells counts as all true)."""
    lines = cc.decode("utf-8", "replace").splitlines()
    if not lines:
        return None
    head = lines[0].split("\t")
    if "barcode" not in head:
        return None
    bi = head.index("barcode")
    ci = head.index("is_cell") if "is_cell" in head else None
    out = set()
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) <= bi:
            continue
        if ci is None or (len(f) > ci and f[ci].strip().lower() in ("true", "1", "yes", "t")):
            out.add(f[bi])
    return out


def census_study(gse: str, src: str | None = None):
    """-> DataFrame (one row per junction of the study) or None, and a small record."""
    import pandas as pd
    from singlet.v2.legacy import LegacyBundle, decode_1pz_bytes

    url = src or f"{DATA}/data/{gse}/{gse}.singlet"
    leg = LegacyBundle.open(url)
    man = json.loads(leg.read("manifest.json")) if leg.exists("manifest.json") else {}
    build = man.get("reference_build") or ""
    parts, rec = [], {"gse": gse, "build": build, "samples": 0, "with_sj": 0, "errors": []}
    t_fetch = t_work = 0.0
    for gsm in leg.samples():
        rec["samples"] += 1
        p = f"samples/{gsm}/sj_counts.1pz"
        if not leg.exists(p):
            continue
        try:
            t0 = time.time()
            raw = leg.read(p)
            t_fetch += time.time() - t0
            t0 = time.time()
            c = decode_1pz_bytes(raw)
            t_work += time.time() - t0
        except Exception as e:  # noqa: BLE001 - an unreadable member is recorded, not fatal
            rec["errors"].append(f"{gsm}: {type(e).__name__}: {e}"[:200])
            continue
        if not c.m or not c.n or not c.rownames:
            continue
        cpath = f"samples/{gsm}/cell_calls.tsv"
        called = _called(leg.read(cpath)) if leg.exists(cpath) else None
        cmask = (
            np.array([b in called for b in c.colnames], bool)
            if called is not None and c.colnames
            else np.zeros(c.n, bool)
        )
        ip = np.asarray(c.indptr, np.int64)
        col = np.repeat(np.arange(c.n), np.diff(ip))
        row = np.asarray(c.indices, np.int64)
        val = np.asarray(c.data, np.int64)
        inc = cmask[col]
        umi_c = np.bincount(row[inc], weights=val[inc], minlength=c.m)
        cells_c = np.bincount(row[inc], minlength=c.m)
        umi_u = np.bincount(row[~inc], weights=val[~inc], minlength=c.m)
        nm = pd.Series(c.rownames).str.extract(r"^(.*):(\d+)-(\d+)$")
        ok = nm[0].notna().to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "chrom": nm[0][ok].to_numpy(),
                    "start": nm[1][ok].astype(np.int64).to_numpy(),
                    "end": nm[2][ok].astype(np.int64).to_numpy(),
                    "umi_called": umi_c[ok].astype(np.int64),
                    "cells_called": cells_c[ok].astype(np.int64),
                    "umi_uncalled": umi_u[ok].astype(np.int64),
                    "n_samples": 1,
                }
            )
        )
        rec["with_sj"] += 1
        rec["sj_bytes"] = rec.get("sj_bytes", 0) + len(raw)
    rec["fetch_s"], rec["decode_s"] = round(t_fetch, 1), round(t_work, 1)
    if not parts:
        return None, rec
    df = pd.concat(parts).groupby(["chrom", "start", "end"], sort=False, as_index=False).sum()
    df["n_studies"] = 1
    df["build"] = build
    return df, rec


def cmd_census(a):
    import pandas as pd

    out = Path(a.out)
    studies = [
        ln.split("\t") for ln in (out / "studies.tsv").read_text().splitlines() if ln.strip()
    ]
    mine = studies[a.shard :: a.shards]
    log(f"shard {a.shard}/{a.shards}: {len(mine)} studies")
    recs, frames = [], []

    def one(row):
        g = row[0]
        key = row[2] if len(row) > 2 and row[2] else f"data/{g}/{g}.singlet"
        t0 = time.time()
        try:
            df, rec = census_study(g, f"{DATA}/{key}")
        except Exception as e:  # noqa: BLE001
            return None, {"gse": g, "error": f"{type(e).__name__}: {e}"[:300]}
        rec["seconds"] = round(time.time() - t0, 1)
        return df, rec

    with ThreadPoolExecutor(a.threads) as ex:
        for df, rec in ex.map(one, mine):
            recs.append(rec)
            if df is not None:
                frames.append(df)
            log(json.dumps(rec)[:300])
    if frames:
        df = (
            pd.concat(frames)
            .groupby(["build", "chrom", "start", "end"], sort=False, as_index=False)
            .sum()
        )
    else:
        df = pd.DataFrame(columns=COLS)
    df[COLS].to_parquet(out / f"shard_{a.shard:04d}.parquet", index=False)
    (out / f"shard_{a.shard:04d}.json").write_text(json.dumps(recs))
    log(f"shard {a.shard}: {len(df)} junction rows")


# ======================================================================================================== reduce
def cmd_reduce(a):
    import pandas as pd

    out = Path(a.out)
    shards = sorted(out.glob("shard_*.parquet"))
    log(f"{len(shards)} shards")
    df = pd.concat([pd.read_parquet(s) for s in shards])
    df = df.groupby(["build", "chrom", "start", "end"], sort=False, as_index=False).sum()
    for b, g in df.groupby("build"):
        name = b or "unknown"
        g.drop(columns=["build"]).to_parquet(out / f"census_{name}.parquet", index=False)
        log(f"{name}: {len(g)} junctions, {int(g['umi_called'].sum())} called UMIs")
    recs = []
    for s in sorted(out.glob("shard_*.json")):
        recs += json.loads(s.read_text())
    summary = {
        "studies": len(recs),
        "studies_failed": sum(1 for r in recs if "error" in r),
        "samples": sum(r.get("samples", 0) for r in recs),
        "samples_with_sj": sum(r.get("with_sj", 0) for r in recs),
        "member_errors": sum(len(r.get("errors", [])) for r in recs),
        "builds": {str(b or "unknown"): int(n) for b, n in df.groupby("build").size().items()},
    }
    (out / "census_summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))


# ======================================================================================================== build
def _download(url: str, dest: Path) -> str:
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 22)
        os.replace(tmp, dest)
    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_build(a):
    import pandas as pd

    out = Path(a.out)
    refdir = Path(a.refs) / a.build
    refdir.mkdir(parents=True, exist_ok=True)
    gtf_url, fa_url = REFS[a.build]
    if a.gtf:  # local files (tests, or references already on disk)
        gtf, fa = Path(a.gtf), Path(a.genome)
        gtf_url, fa_url = gtf.resolve().as_uri(), fa.resolve().as_uri()
    else:
        gtf, fa = refdir / Path(gtf_url).name, refdir / Path(fa_url).name
    sources = {
        "gtf": {"url": gtf_url, "sha256": _download(gtf_url, gtf)},
        "genome": {"url": fa_url, "sha256": _download(fa_url, fa)},
    }
    log("references", json.dumps(sources))

    pk = P.default_store().vocab_packs("exon", a.build)[0]
    gene_index = {g.split(".")[0]: i for i, (g, _n) in enumerate(pk.gene_table())}
    ann = J.read_gtf(str(gtf), gene_index)
    log(f"GTF: {len(ann.introns)} annotated introns")

    cen = pd.read_parquet(out / f"census_{a.build}.parquet")
    log(f"census: {len(cen)} junctions")
    annotated = pd.DataFrame(
        [(c, s, e) for (c, s, e) in ann.introns], columns=["chrom", "start", "end"]
    )
    allj = pd.concat([cen[["chrom", "start", "end"]], annotated]).drop_duplicates(ignore_index=True)
    allj = allj.merge(cen, on=["chrom", "start", "end"], how="left").fillna(0)
    ann_keys = pd.MultiIndex.from_frame(annotated)
    is_ann = pd.MultiIndex.from_frame(allj[["chrom", "start", "end"]]).isin(ann_keys)
    n_local = int(((allj["n_samples"].to_numpy() < a.rare_min_samples) & ~is_ann).sum())
    allj = allj[is_ann | (allj["n_samples"].to_numpy() >= a.rare_min_samples)].reset_index(
        drop=True
    )  # the rest stays local
    log(
        f"candidates: {len(allj)} (annotated or in >= {a.rare_min_samples} samples); {n_local} single-sample junctions stay local"
    )
    j = J.Junctions(
        sorted(set(allj["chrom"])),
        np.zeros(len(allj), np.uint16),
        allj["start"].to_numpy(np.uint32),
        allj["end"].to_numpy(np.uint32),
    )
    cidx = {c: i for i, c in enumerate(j.chroms)}
    j.chrom = allj["chrom"].map(cidx).to_numpy(np.uint16)
    genome = J.read_fasta(str(fa), set(j.chroms))
    log(f"genome: {len(genome)} of {len(j.chroms)} junction chromosomes")
    j = J.annotate(j, ann, genome)
    del genome

    umi = allj["umi_called"].to_numpy(np.float64)
    amb = allj["umi_uncalled"].to_numpy(np.float64)
    frac_amb = np.where(umi + amb > 0, amb / np.maximum(umi + amb, 1), 0.0)
    noise_ok = (
        (allj["n_studies"].to_numpy() >= a.core_min_studies)
        & (umi >= a.core_min_umi)
        & (j.motif > 0)
        & (frac_amb <= a.max_ambient_frac)
    )
    core = j.annotated | noise_ok
    rare = ~core & (allj["n_samples"].to_numpy() >= a.rare_min_samples)
    rules = {
        "core": "annotated, or n_studies >= %d and umi_called >= %d and canonical motif and ambient fraction <= %g"
        % (a.core_min_studies, a.core_min_umi, a.max_ambient_frac),
        "rare": "other junctions with n_samples >= %d" % a.rare_min_samples,
        "local": "everything else (stays in its sample object)",
    }
    pdir = out / "packs" / a.build
    pdir.mkdir(parents=True, exist_ok=True)
    seg_core = J.build_segment(j.take(np.flatnonzero(core)), build=a.build, id_lo=0, tier="core")
    seg_rare = J.build_segment(
        j.take(np.flatnonzero(rare)), build=a.build, id_lo=int(core.sum()), tier="rare"
    )
    ver = J.build_version(
        [seg_core, seg_rare],
        build=a.build,
        version=1,
        extra={"rules": rules, "sources": sources, "census": a.census_tag},
    )
    entries = []
    for data, kind in (
        (seg_core, "junction-segment"),
        (seg_rare, "junction-segment"),
        (ver, "junction-version"),
    ):
        (pdir / (P.pack_id(data).hex() + ".pack")).write_bytes(data)
        entries.append(P.index_entry(data, kind, {"build": a.build}))
    P.write_index(entries, str(pdir / "index.json"))
    t0 = time.perf_counter()
    J.Segment(seg_core).j  # noqa: B018 - decode once for the report
    report = {
        "build": a.build,
        "version_id": P.pack_id(ver).hex(),
        "junctions": {
            "census": int(len(cen)),
            "annotated_gtf": int(len(ann.introns)),
            "core": int(core.sum()),
            "core_annotated": int(j.annotated.sum()),
            "core_unannotated": int((core & ~j.annotated).sum()),
            "rare": int(rare.sum()),
            "local_only": int((~core & ~rare).sum()) + n_local,
            "annotated_seen": int((j.annotated & (allj["n_samples"].to_numpy() > 0)).sum()),
        },
        "called_umi_share": {
            "core": float(umi[core].sum() / max(umi.sum(), 1)),
            "rare": float(umi[rare].sum() / max(umi.sum(), 1)),
            "local": float(umi[~core & ~rare].sum() / max(umi.sum(), 1)),
        },
        "bytes": {"core": len(seg_core), "rare": len(seg_rare), "version": len(ver)},
        "decode_core_ms": round(1000 * (time.perf_counter() - t0), 1),
        "rules": rules,
        "sources": sources,
        "packs_dir": str(pdir),
    }
    (out / f"report_{a.build}.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("--out", required=True)
    s = sub.add_parser("census")
    s.add_argument("--out", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--threads", type=int, default=8)
    s = sub.add_parser("reduce")
    s.add_argument("--out", required=True)
    s = sub.add_parser("build")
    s.add_argument("--out", required=True)
    s.add_argument("--build", required=True, choices=sorted(REFS))
    s.add_argument("--refs", required=True, help="directory for downloaded references")
    s.add_argument("--core-min-studies", type=int, default=3)
    s.add_argument("--core-min-umi", type=int, default=10)
    s.add_argument("--max-ambient-frac", type=float, default=0.5)
    s.add_argument("--rare-min-samples", type=int, default=2)
    s.add_argument("--census-tag", default="")
    s.add_argument("--gtf", help="local GTF instead of the download (with --genome)")
    s.add_argument("--genome", help="local genome FASTA instead of the download")
    a = p.parse_args(argv)
    {"list": cmd_list, "census": cmd_census, "reduce": cmd_reduce, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
