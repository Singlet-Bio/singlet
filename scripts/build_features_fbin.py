#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a canonical ``features.fbin`` from a GTF file.

Implements Phase 2 of ``docs/V2_IMPLEMENTATION_PLAN.md``: scans
``genes.gtf`` for gene / exon records, derives intron intervals + exon-
exon junctions per gene, and writes the packed binary defined in §3.1 of
``docs/CANONICAL_OUTPUT_FORMAT.md``.

Usage::

    python -m singlet.scripts.build_features_fbin \\
        --gtf reference/GRCh38-2024-A/genes/genes.gtf \\
        --build-id GRCh38-2024-A \\
        --out  reference/GRCh38-2024-A/features.fbin

Or directly: ``python build_features_fbin.py --gtf ... --out ...``.

The builder is single-pass and streams the GTF — memory peaks at one
gene's worth of exon records.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import logging
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

# Allow running from a checkout without `pip install -e .`.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_PKG = _THIS_DIR.parent / "python"
if (_REPO_PKG / "singlet" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_PKG))

from singlet.refbundle._features import _GeneIn, biotype_code, write_features

logger = logging.getLogger("build_features_fbin")


# --------------------------------------------------------------------------
# GTF parsing
# --------------------------------------------------------------------------

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_attrs(field: str) -> dict:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(field)}


def _iter_gtf(path: Path) -> Iterator[Tuple[str, str, int, int, str, dict]]:
    """Yield ``(chrom, feature, start, end, strand, attrs)`` per GTF line."""
    with _open_text(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom = parts[0]
            feature = parts[2]
            start = int(parts[3])  # GTF is 1-based, inclusive
            end = int(parts[4])
            strand = parts[6]
            attrs = _parse_attrs(parts[8])
            yield chrom, feature, start, end, strand, attrs


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def _merge_intervals(ivs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [ivs[0]]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def build_genes_from_gtf(gtf_path: Path) -> List[_GeneIn]:
    """Stream a GTF and return a list of :class:`_GeneIn` records.

    Each gene's exons are merged to flat intervals; introns are the
    complement intervals between merged exons; junctions are the
    donor/acceptor pairs (one per intron).
    """
    # Pass 1 — collect per-gene metadata.
    gene_meta: dict = {}
    gene_exons: dict = {}

    for chrom, feat, start, end, strand, attrs in _iter_gtf(gtf_path):
        gid = attrs.get("gene_id")
        if gid is None:
            continue
        if feat == "gene":
            gene_meta[gid] = {
                "chrom": chrom,
                "strand": 1 if strand == "-" else 0,
                "tx_start": start,
                "tx_end": end,
                "symbol": attrs.get("gene_name", ""),
                "biotype": biotype_code(
                    attrs.get("gene_biotype", attrs.get("gene_type", ""))
                ),
            }
        elif feat == "exon":
            gene_exons.setdefault(gid, []).append((start, end))
            # Fill metadata if no `gene` row exists (some GTFs lack it).
            if gid not in gene_meta:
                gene_meta[gid] = {
                    "chrom": chrom,
                    "strand": 1 if strand == "-" else 0,
                    "tx_start": start,
                    "tx_end": end,
                    "symbol": attrs.get("gene_name", ""),
                    "biotype": biotype_code(
                        attrs.get("gene_biotype", attrs.get("gene_type", ""))
                    ),
                }
            else:
                m = gene_meta[gid]
                if start < m["tx_start"]:
                    m["tx_start"] = start
                if end > m["tx_end"]:
                    m["tx_end"] = end

    # Pass 2 — derive intron + junction tables and assemble _GeneIn.
    genes: List[_GeneIn] = []
    for gid, meta in gene_meta.items():
        exons = _merge_intervals(gene_exons.get(gid, []))
        introns: List[tuple] = []
        junctions: List[tuple] = []
        for i in range(len(exons) - 1):
            ex_end = exons[i][1]
            ex_next_start = exons[i + 1][0]
            intron_start = ex_end + 1
            intron_end = ex_next_start - 1
            # flank_exon_lo / hi: indices into this gene's exon list
            introns.append((intron_start, intron_end, i, i + 1))
            # junction: donor = ex_end, acceptor = ex_next_start
            junctions.append((ex_end, ex_next_start, i, i + 1, 0, 0))
        genes.append(
            _GeneIn(
                name=gid,
                symbol=meta["symbol"],
                chrom=meta["chrom"],
                strand=meta["strand"],
                biotype=meta["biotype"],
                tx_start=meta["tx_start"],
                tx_end=meta["tx_end"],
                exons=exons,
                introns=introns,
                junctions=junctions,
            )
        )

    # Stable sort by (chrom, tx_start) for deterministic output ordering.
    genes.sort(key=lambda g: (g.chrom, g.tx_start, g.name))
    return genes


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="build_features_fbin", description=__doc__)
    p.add_argument("--gtf", required=True, type=Path, help="Input genes.gtf[.gz]")
    p.add_argument(
        "--build-id",
        required=True,
        help="Build identifier, e.g. 'GRCh38-2024-A' (≤32 chars).",
    )
    p.add_argument("--out", required=True, type=Path, help="Output features.fbin path")
    p.add_argument(
        "--gtf-sha256",
        default=None,
        help="Pre-computed SHA-256 of the GTF (skip recomputation).",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.gtf.is_file():
        p.error(f"GTF not found: {args.gtf}")

    gtf_sha = args.gtf_sha256 or _sha256_file(args.gtf)
    logger.info("GTF sha256: %s", gtf_sha)

    genes = build_genes_from_gtf(args.gtf)
    n_exons = sum(len(g.exons) for g in genes)
    n_introns = sum(len(g.introns) for g in genes)
    n_junctions = sum(len(g.junctions) for g in genes)
    logger.info(
        "Parsed %d genes, %d exons, %d introns, %d junctions",
        len(genes),
        n_exons,
        n_introns,
        n_junctions,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_features(
        args.out,
        build_id=args.build_id,
        gtf_sha256=gtf_sha,
        genes=genes,
    )
    size_mb = args.out.stat().st_size / (1 << 20)
    logger.info("Wrote %s (%.1f MB)", args.out, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
