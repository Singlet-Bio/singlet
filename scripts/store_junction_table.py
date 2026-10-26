#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the shared splice-junction table, version 1 (docs/format-store.md section 12), from a census of the atlas.

Read-only against the public data host: legacy bundles are read by HTTP range (only each sample's ``sj_counts.1pz`` and
``cell_calls.tsv``), references come from GENCODE / Ensembl. Nothing is uploaded; the packs are written to ``--out``.

    store_junction_table.py list    --out DIR                       # studies with a bundle (singlet.bio/api/gse), studies.tsv
    store_junction_table.py census  --out DIR --shard I --shards N  # per-junction counts of one shard of studies -> shard_I.parquet
    store_junction_table.py reduce  --out DIR                       # all shards -> census_<build>.parquet (one row per junction)
    store_junction_table.py build   --out DIR --build GRCh38-2024-A # GTF + genome + census -> packs (core, rare, version) + report
    store_junction_table.py publish --out DIR [--apply]             # upload packs/*/ to R2 store/ref/ and merge them into store/ref/index.json

Per junction the census keeps: studies, samples, UMIs and cells in called cells, UMIs in uncalled barcodes (the ambient noise model).
Tier rules (recorded in the version header; thresholds from census_stats.py on the v1 census): every annotated junction is core;
an unannotated junction is core with >= ``--min-studies`` studies, >= ``--core-min-umi`` UMIs in called cells, a canonical motif and
at most ``--max-ambient-frac`` of its UMIs in uncalled barcodes, and rare (one segment, or one per chromosome above 1 MiB) with >= ``--rare-min-umi`` UMIs
and the same other conditions; everything else stays local in its sample. Reference chromosome names are renamed to the pipeline's
(``chr1`` -> ``1``) when they differ only by the UCSC prefix.
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

from singlet.store import junctions as J  # noqa: E402
from singlet.store import packs as P  # noqa: E402

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
    from singlet.store.legacy import LegacyBundle, decode_1pz_bytes

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
        ln.split("\t")
        for ln in Path(a.studies or out / "studies.tsv").read_text().splitlines()
        if ln.strip()
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
    stem = (
        f"shard_{a.name}" if a.name else f"shard_{a.shard:04d}"
    )  # --name: a retry, merged like any shard
    df[COLS].to_parquet(out / f"{stem}.parquet", index=False)
    (out / f"{stem}.json").write_text(json.dumps(recs))
    log(f"shard {a.shard}: {len(df)} junction rows")


# ======================================================================================================== reduce
NBUCKET = 64  # the reduce splits every shard by (build, hash of chromosome) so each merge holds 1/64 of a build


def _split_shard(args):
    """Pass 1: one shard -> parts/<build>/<bucket>/<shard>.parquet."""
    import pandas as pd

    path, parts = args
    df = pd.read_parquet(path)
    if not len(df):
        return 0
    df["build"] = df["build"].fillna("").replace("", "unknown")
    bucket = pd.util.hash_array(df["chrom"].to_numpy(dtype=object)) % NBUCKET
    for (b, k), g in df.groupby([df["build"], bucket], sort=False):
        d = Path(parts) / str(b) / f"{int(k):03d}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["build"]).to_parquet(d / (Path(path).stem + ".parquet"), index=False)
    return len(df)


def _merge_bucket(args):
    """Pass 2: all shards' rows of one (build, bucket) -> census_<build>/part-<bucket>.parquet (one row per junction)."""
    import pandas as pd

    d, dest = args
    files = sorted(Path(d).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.groupby(["chrom", "start", "end"], sort=False, as_index=False).sum()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return len(df), int(df["umi_called"].sum()), int((df["n_samples"] >= 2).sum())


def cmd_reduce(a):
    from multiprocessing import get_context

    out = Path(a.out)
    shards = sorted(out.glob("shard_*.parquet"))
    parts = out / "parts"
    log(f"{len(shards)} shards, {a.workers} workers")
    with get_context("spawn").Pool(a.workers) as pool:
        rows = sum(pool.map(_split_shard, [(str(s), str(parts)) for s in shards], chunksize=1))
        log(f"pass 1: {rows} shard rows split by build and chromosome bucket")
        jobs = [
            (str(d), str(out / f"census_{d.parent.name}" / f"part-{d.name}.parquet"))
            for d in sorted(parts.glob("*/*"))
            if d.is_dir()
        ]
        res = pool.map(_merge_bucket, jobs, chunksize=1)
    builds: dict = {}
    for (d, _dest), (n, umi, rec) in zip(jobs, res):
        b = builds.setdefault(
            Path(d).parent.name, {"junctions": 0, "umi_called": 0, "in_2_or_more_samples": 0}
        )
        b["junctions"] += n
        b["umi_called"] += umi
        b["in_2_or_more_samples"] += rec
    recs = []
    for s in sorted(out.glob("shard_*.json")):
        recs += json.loads(s.read_text())
    last = {}
    for r in recs:  # a study rerun in a retry shard counts once, with its latest result
        last[r["gse"]] = r
    recs = list(last.values())
    summary = {
        "studies": len(recs),
        "studies_failed": sum(1 for r in recs if "error" in r),
        "samples": sum(r.get("samples", 0) for r in recs),
        "samples_with_sj": sum(r.get("with_sj", 0) for r in recs),
        "member_errors": sum(len(r.get("errors", [])) for r in recs),
        "sj_bytes_read": sum(r.get("sj_bytes", 0) for r in recs),
        "builds": builds,
    }
    (out / "census_summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    if not a.keep_parts:
        shutil.rmtree(parts, ignore_errors=True)


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

    import pyarrow.parquet as pq

    cdir = out / f"census_{a.build}"  # a directory of parts (reduce)
    n_census = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(cdir.glob("*.parquet")))
    # only recurrent, supported junctions can enter the table; everything else stays local in its sample
    cen = pd.read_parquet(
        cdir,
        filters=[("n_studies", ">=", a.min_studies), ("umi_called", ">=", a.rare_min_umi)],
    )
    log(
        f"census: {n_census} junctions, {len(cen)} in >= {a.min_studies} studies with >= {a.rare_min_umi} UMIs"
    )

    # chromosome names: the table keeps the pipeline's names (they are part of the junction names); the GTF and genome are
    # renamed to them when they differ only by the UCSC "chr" prefix (GENCODE mouse vs the pipeline's Ensembl-style names)
    rename = _chrom_rename(set(cen["chrom"].astype(str)), {c for c, _s, _e in ann.introns})
    if rename:
        log(
            f"renaming {len(rename)} reference chromosomes to the census names, e.g. {list(rename.items())[:3]}"
        )
        ann = J.Annotation(
            {(rename.get(c, c), s0, e0): v for (c, s0, e0), v in ann.introns.items()},
            {(rename.get(c, c), s0) for c, s0 in ann.donors},
            {(rename.get(c, c), e0) for c, e0 in ann.acceptors},
            {(rename.get(c, c), s0, e0): v for (c, s0, e0), v in ann.strand.items()},
        )
    annotated = pd.DataFrame(
        [(c, s0, e0) for (c, s0, e0) in ann.introns], columns=["chrom", "start", "end"]
    )
    allj = pd.concat([cen[["chrom", "start", "end"]], annotated]).drop_duplicates(ignore_index=True)
    allj = allj.merge(cen, on=["chrom", "start", "end"], how="left").fillna(0)
    j = J.Junctions(
        sorted(set(allj["chrom"].astype(str))),
        np.zeros(len(allj), np.uint16),
        allj["start"].to_numpy(np.uint32),
        allj["end"].to_numpy(np.uint32),
    )
    cidx = {c: i for i, c in enumerate(j.chroms)}
    j.chrom = allj["chrom"].astype(str).map(cidx).to_numpy(np.uint16)
    inv = {v: k for k, v in rename.items()}
    genome_ref = J.read_fasta(str(fa), {inv.get(c, c) for c in j.chroms})
    genome = {rename.get(c, c): sq for c, sq in genome_ref.items()}
    del genome_ref
    log(f"genome: {len(genome)} of {len(j.chroms)} junction chromosomes")
    j = J.annotate(j, ann, genome)
    del genome

    umi = allj["umi_called"].to_numpy(np.float64)
    amb = allj["umi_uncalled"].to_numpy(np.float64)
    frac_amb = np.where(umi + amb > 0, amb / np.maximum(umi + amb, 1), 0.0)
    supported = (
        (allj["n_studies"].to_numpy() >= a.min_studies)
        & (j.motif > 0)
        & (frac_amb <= a.max_ambient_frac)
    )
    core = j.annotated | (supported & (umi >= a.core_min_umi))
    rare = ~core & supported & (umi >= a.rare_min_umi)
    rules = {
        "core": "annotated (GTF), or unannotated with n_studies >= %d, umi_called >= %d, a canonical motif and ambient fraction <= %g"
        % (a.min_studies, a.core_min_umi, a.max_ambient_frac),
        "rare": "unannotated with n_studies >= %d, %d <= umi_called < %d, a canonical motif and ambient fraction <= %g; one segment, or one per chromosome above 1 MiB"
        % (a.min_studies, a.rare_min_umi, a.core_min_umi, a.max_ambient_frac),
        "local": "everything else: stays in its sample object as integer coordinates",
    }
    pdir = out / "packs" / a.build
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    segs = [J.build_segment(j.take(np.flatnonzero(core)), build=a.build, id_lo=0, tier="core")]
    nxt = int(core.sum())
    jr = j.take(np.flatnonzero(rare))
    jr = jr.take(jr.genomic_order())
    whole = J.build_segment(jr, build=a.build, id_lo=nxt, tier="rare") if len(jr) else None
    if whole is not None and len(whole) <= a.rare_split_bytes:  # small rare tier: one segment
        segs.append(whole)
    else:  # one rare segment per chromosome, so an on-demand fetch is one chromosome's
        for c in sorted(set(jr.chrom.tolist()), key=lambda i: jr.chroms[i]):
            rows = np.flatnonzero(jr.chrom == c)
            segs.append(
                J.build_segment(
                    jr.take(rows),
                    build=a.build,
                    id_lo=nxt,
                    tier="rare",
                    extra={"chrom": jr.chroms[c]},
                )
            )
            nxt += len(rows)
    ver = J.build_version(
        segs,
        build=a.build,
        version=1,
        extra={"rules": rules, "sources": sources, "census": a.census_tag},
    )
    entries = []
    for data in segs + [ver]:
        kind = "junction-version" if data is ver else "junction-segment"
        (pdir / (P.pack_id(data).hex() + ".pack")).write_bytes(data)
        entries.append(P.index_entry(data, kind, {"build": a.build}))
    P.write_index(entries, str(pdir / "index.json"))
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        J.Segment(segs[0]).j  # noqa: B018 - decode the whole core
        times.append(time.perf_counter() - t0)
    ref = np.arange(1_000_000, dtype=np.uint32)
    t0 = time.perf_counter()
    for _ in range(20):
        np.cumsum(ref, dtype=np.uint32)
    ref_ms = 1000 * (time.perf_counter() - t0) / 20
    summ = out / "census_summary.json"
    build_umi = (
        json.loads(summ.read_text())["builds"].get(a.build, {}).get("umi_called")
        if summ.exists()
        else None
    )
    tot = max(
        float(build_umi or cen["umi_called"].sum()), 1.0
    )  # all called junction UMIs of the build
    report = {
        "build": a.build,
        "version_id": P.pack_id(ver).hex(),
        "chrom_rename": dict(list(rename.items())[:5]) if rename else {},
        "junctions": {
            "census": int(n_census),
            "recurrent_supported_candidates": int(len(cen)),
            "annotated_gtf": int(len(ann.introns)),
            "annotated_seen": int((j.annotated & (allj["n_samples"].to_numpy() > 0)).sum()),
            "core": int(core.sum()),
            "core_annotated": int(j.annotated.sum()),
            "core_unannotated": int((core & ~j.annotated).sum()),
            "rare": int(rare.sum()),
            "rare_segments": len(segs) - 1,
            "local": int(n_census - (core & (allj["n_samples"].to_numpy() > 0)).sum() - rare.sum()),
        },
        "called_umi_share": {
            "core": float(umi[core].sum() / tot),
            "core_annotated": float(umi[j.annotated].sum() / tot),
            "rare": float(umi[rare].sum() / tot),
        },
        "bytes": {
            "core": len(segs[0]),
            "rare_total": int(sum(len(x) for x in segs[1:])),
            "rare_largest_segment": int(max((len(x) for x in segs[1:]), default=0)),
            "version": len(ver),
        },
        "decode_core_ms_median_of_7": round(1000 * float(np.median(times)), 1),
        "reference_cumsum_1M_u32_ms": round(ref_ms, 2),
        "rules": rules,
        "sources": sources,
        "packs_dir": str(pdir),
        "packs": [
            {"id": e["id"], "kind": e["kind"], "bytes": e["bytes"], "sha256": e["sha256"]}
            for e in entries
        ],
    }
    (out / f"report_{a.build}.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report)[:3000])


def _chrom_rename(census: set, reference: set) -> dict:
    """reference name -> census name for the "chr"-prefixed reference names whose unprefixed form (chrM -> MT) is what the census
    uses, when that matches more names than the reference names do as they are. Unplaced scaffolds (JH584295.1, ...) are named the
    same in both conventions, so only the prefixed names decide."""

    def strip(c: str) -> str:
        k = c[3:] if c.startswith("chr") else c
        return "MT" if k == "M" else k

    prefixed = {c for c in reference if c.startswith("chr")}
    direct = len(census & prefixed)
    stripped = {c: strip(c) for c in prefixed if strip(c) in census}
    return stripped if len(stripped) > direct else {}


# ======================================================================================================== publish
PACK_HEADERS = ("application/octet-stream", "public, max-age=31536000, immutable")
INDEX_HEADERS = ("application/json", "public, max-age=300")
INDEX_KEY = "store/ref/index.json"


def _s3():
    """boto3 client for R2 from R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY (as scripts/store_campaign.py does)."""
    import boto3
    from botocore.config import Config

    kw = dict(
        signature_version="s3v4",
        retries={"max_attempts": 10, "mode": "adaptive"},
        s3={"addressing_style": "path"},
    )
    for opt in ("request_checksum_calculation", "response_checksum_validation"):
        try:  # botocore >= 1.36 adds CRC32 trailers by default; R2 wants them only when required
            Config(**{opt: "when_required"})
            kw[opt] = "when_required"
        except TypeError:
            pass
    endpoint = (
        os.environ.get("R2_ENDPOINT")
        or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(**kw),
    )


def _index_text(doc: dict) -> str:
    """The exact serialisation of packs.write_index (so an index written by it round-trips byte for byte)."""
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def _public_get(key: str) -> bytes:
    req = urllib.request.Request(
        f"{DATA}/{key}", headers={"User-Agent": UA, "Cache-Control": "no-cache"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def cmd_publish(a):
    """Upload every pack of <out>/packs/<build>/ to <bucket>/store/ref/<id>.pack (skipping keys already there with the same sha256),
    read each one back, then merge the entries into store/ref/index.json (existing entries unchanged) and verify through the public
    host. Without --apply only the plan is printed."""
    out = Path(a.out)
    todo = []
    for idx in sorted((out / "packs").glob("*/index.json")):
        for e in json.loads(idx.read_text())["packs"]:
            f = idx.parent / f"{e['id']}.pack"
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if len(data) != e["bytes"] or sha != e["sha256"] or P.pack_id(data).hex() != e["id"]:
                raise SystemExit(f"{f}: size / sha256 / id do not match {idx}")
            todo.append((e, data))
    log(
        f"{len(todo)} packs, {sum(len(d) for _e, d in todo)} bytes, from {len(list((out / 'packs').glob('*/index.json')))} builds"
    )
    s3 = _s3()
    bucket = a.bucket
    live = s3.get_object(Bucket=bucket, Key=INDEX_KEY)["Body"].read()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (out / f"index.json.previous-{stamp}").write_bytes(live)
    old = json.loads(live)
    if old.get("format") not in ("singlet-store-packs", "singlet-v2-packs"):
        raise SystemExit("the live store/ref/index.json is not a singlet-store-packs index")
    if _index_text(old).encode() != live:
        raise SystemExit(
            "the live index.json does not round-trip through the writer: refusing to rewrite it"
        )
    have = {e["id"] for e in old["packs"]}
    new_entries = [e for e, _d in todo if e["id"] not in have]
    log(
        f"live index: {len(old['packs'])} entries (saved to index.json.previous-{stamp}); {len(new_entries)} new entries"
    )
    up = skip = 0
    for e, data in todo:
        key = f"store/ref/{e['id']}.pack"
        try:
            h = s3.head_object(Bucket=bucket, Key=key)
            same = (
                h["ContentLength"] == len(data)
                and (h.get("Metadata") or {}).get("sha256") == e["sha256"]
            )
        except Exception:  # noqa: BLE001 - not there
            same = False
        if same:
            skip += 1
            continue
        if not a.apply:
            up += 1
            continue
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=PACK_HEADERS[0],
            CacheControl=PACK_HEADERS[1],
            Metadata={"sha256": e["sha256"]},
        )
        back = s3.get_object(Bucket=bucket, Key=key)
        body = back["Body"].read()
        if (
            len(body) != len(data)
            or hashlib.sha256(body).hexdigest() != e["sha256"]
            or back.get("CacheControl") != PACK_HEADERS[1]
        ):
            raise SystemExit(f"{key}: read-back does not match (size, sha256 or Cache-Control)")
        up += 1
    log(
        f"packs: {up} {'uploaded and read back' if a.apply else 'to upload'}, {skip} already there with the same sha256"
    )
    doc = dict(old)
    doc["packs"] = list(old["packs"]) + new_entries
    text = _index_text(doc).encode()
    rec = {
        "packs_total": len(todo),
        "packs_uploaded": up,
        "packs_skipped": skip,
        "bytes": sum(len(d) for _e, d in todo),
        "index_entries_before": len(old["packs"]),
        "index_entries_after": len(doc["packs"]),
        "previous_index": f"index.json.previous-{stamp}",
        "apply": bool(a.apply),
    }
    if a.apply:
        s3.put_object(
            Bucket=bucket,
            Key=INDEX_KEY,
            Body=text,
            ContentType=INDEX_HEADERS[0],
            CacheControl=INDEX_HEADERS[1],
            Metadata={"sha256": hashlib.sha256(text).hexdigest()},
        )
        # verification through the public host
        pub = _public_get(INDEX_KEY)
        pdoc = json.loads(pub)
        # byte-identical old entries: everything before the closing of the old packs list is unchanged in the new file
        cut = live.rfind(b"\n ]")
        old_ok = (
            pdoc["packs"][: len(old["packs"])] == old["packs"]
            and cut > 0
            and pub[:cut] == live[:cut]
        )
        new_ok = {x["id"] for x in new_entries} <= {x["id"] for x in pdoc["packs"]}
        packs_ok = 0
        for e, _d in todo:
            body = _public_get(f"store/ref/{e['id']}.pack")
            packs_ok += hashlib.sha256(body).hexdigest() == e["sha256"]
        rec.update(
            {
                "public_index_sha256": hashlib.sha256(pub).hexdigest(),
                "public_index_matches_written": pub == text,
                "old_entries_unchanged": bool(old_ok),
                "new_entries_present": bool(new_ok),
                "packs_verified_via_public_host": packs_ok,
            }
        )
        if not (old_ok and new_ok and packs_ok == len(todo)):
            log("VERIFICATION FAILED: " + json.dumps(rec))
            (out / f"publish-{stamp}.json").write_text(json.dumps(rec, indent=1))
            raise SystemExit(1)
    (out / f"publish-{stamp}.json").write_text(json.dumps(rec, indent=1))
    log(json.dumps(rec))


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
    s.add_argument("--studies", help="study list to use instead of <out>/studies.tsv (a retry)")
    s.add_argument("--name", help="write shard_<name>.parquet instead of the shard number")
    s = sub.add_parser("reduce")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--keep-parts", action="store_true")
    s = sub.add_parser("build")
    s.add_argument("--out", required=True)
    s.add_argument("--build", required=True, choices=sorted(REFS))
    s.add_argument("--refs", required=True, help="directory for downloaded references")
    s.add_argument("--min-studies", type=int, default=3)
    s.add_argument("--core-min-umi", type=int, default=1000)
    s.add_argument("--rare-min-umi", type=int, default=100)
    s.add_argument("--max-ambient-frac", type=float, default=0.5)
    s.add_argument(
        "--rare-split-bytes",
        type=int,
        default=1 << 20,
        help="split the rare tier by chromosome above this size",
    )
    s.add_argument("--census-tag", default="")
    s.add_argument("--gtf", help="local GTF instead of the download (with --genome)")
    s.add_argument("--genome", help="local genome FASTA instead of the download")
    s = sub.add_parser("publish")
    s.add_argument("--out", required=True)
    s.add_argument("--bucket", default="singlet-data")
    s.add_argument("--apply", action="store_true", help="write to R2 (without it: plan only)")
    a = p.parse_args(argv)
    {
        "list": cmd_list,
        "census": cmd_census,
        "reduce": cmd_reduce,
        "build": cmd_build,
        "publish": cmd_publish,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
