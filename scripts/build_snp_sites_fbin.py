#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a canonical ``snp_sites.fbin`` from a population SNP VCF.

Implements Phase 2 of ``docs/V2_IMPLEMENTATION_PLAN.md``: scans a sorted
VCF (typically 1000 Genomes phase-3 SNP panel at AF≥0.05), filters to
biallelic SNPs on autosomes + chrX, and writes the §3.2 binary.

Usage::

    python build_snp_sites_fbin.py \\
        --vcf reference/GRCh38-2024-A/snps/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz \\
        --build-id GRCh38-2024-A \\
        --panel-id 1KGP_AF5e2 \\
        --out reference/GRCh38-2024-A/snp_sites.fbin

The builder is single-pass; memory peaks at the chromosome-name table
(<256 entries) plus a small streaming buffer.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import sys
from pathlib import Path
from typing import Iterator

_THIS_DIR = Path(__file__).resolve().parent
_REPO_PKG = _THIS_DIR.parent / "python"
if (_REPO_PKG / "singlet" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_PKG))

from singlet.refbundle._snp import SnpSite, write_snp_panel

logger = logging.getLogger("build_snp_sites_fbin")


_VALID_BASES = set("ACGT")


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_info(field: str) -> dict:
    out: dict = {}
    for kv in field.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def iter_vcf_snps(vcf_path: Path, min_af: float = 0.0) -> Iterator[SnpSite]:
    """Yield biallelic SNPs from a VCF, sorted in input order.

    Filters:
    * REF and ALT are single A/C/G/T bases (no indels, no multi-allelic).
    * Variant ID parsed for ``rsNNNN``; otherwise ``rsid=0``.
    * AF taken from the INFO ``AF`` field; falls back to 0.0.
    """
    with _open_text(vcf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom = parts[0]
            pos = int(parts[1])
            rsid_str = parts[2]
            ref = parts[3]
            alt = parts[4]
            info = parts[7]

            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in _VALID_BASES or alt not in _VALID_BASES:
                continue

            af = 0.0
            if "AF=" in info:
                af_str = _parse_info(info).get("AF", "0")
                try:
                    # Multi-allelic AF is comma-separated; take first.
                    af = float(af_str.split(",", 1)[0])
                except ValueError:
                    af = 0.0
            if af < min_af:
                continue

            rsid = 0
            if rsid_str.startswith("rs"):
                try:
                    rsid = int(rsid_str[2:])
                except ValueError:
                    rsid = 0

            yield SnpSite(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                af_pop=af,
                rsid=rsid,
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="build_snp_sites_fbin", description=__doc__)
    p.add_argument("--vcf", required=True, type=Path, help="Input VCF[.gz]")
    p.add_argument(
        "--build-id", required=True, help="Reference build (e.g. GRCh38-2024-A)"
    )
    p.add_argument(
        "--panel-id", default="1KGP_AF5e2", help="Panel identifier (≤32 chars)"
    )
    p.add_argument("--out", required=True, type=Path, help="Output snp_sites.fbin")
    p.add_argument(
        "--min-af", type=float, default=0.0, help="Minimum population AF [0.0]"
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.vcf.is_file():
        p.error(f"VCF not found: {args.vcf}")

    sites = list(iter_vcf_snps(args.vcf, min_af=args.min_af))
    logger.info("Kept %d biallelic SNPs (min_af=%g)", len(sites), args.min_af)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_snp_panel(
        args.out,
        build_id=args.build_id,
        panel_id=args.panel_id,
        sites=sites,
    )
    size_mb = args.out.stat().st_size / (1 << 20)
    logger.info("Wrote %s (%.1f MB)", args.out, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
