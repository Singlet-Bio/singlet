# SPDX-License-Identifier: MIT
"""``singlet-process`` / ``python -m singlet.pipeline`` — CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from singlet.pipeline._errors import PipelineError
from singlet.pipeline._run import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="singlet-process",
        description=(
            "Process raw reads (an SRA/ENA/DDBJ accession, a download URL, or "
            "local FASTQ/.1fq files) into a canonical singlet sample directory."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="SRA/ENA/DDBJ accession, download URL, or local .1fq/.1pz archive.",
    )
    parser.add_argument(
        "--reads",
        nargs="+",
        default=None,
        metavar="FASTQ",
        help="Local raw FASTQ file(s) to encode (mutually exclusive with `source`).",
    )
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--organism", default="human")
    parser.add_argument("--binary", type=Path, default=None, help="Path to the singlet binary.")
    parser.add_argument("--ref-base", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-snps", action="store_true", help="Disable SNP calling.")
    parser.add_argument("--pipeline-extras", action="store_true")
    parser.add_argument("--cascade", default="off")
    parser.add_argument("--te-classify", default="off")
    parser.add_argument("--nonhost", action="store_true", help="Enable viral/microbial screens.")
    parser.add_argument("--raw-matrix", action="store_true")
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw: List[str] = list(sys.argv[1:] if argv is None else argv)

    extra_args: List[str] = []
    if "--" in raw:
        idx = raw.index("--")
        extra_args = raw[idx + 1 :]
        raw = raw[:idx]

    parser = _build_parser()
    args = parser.parse_args(raw)

    if args.source and args.reads:
        parser.error("Cannot specify both a positional source and --reads.")
    if not args.source and not args.reads:
        parser.error("Must specify either a source (accession/URL/local file) or --reads.")

    source = args.reads if args.reads else args.source

    try:
        result = run(
            source,
            output_dir=args.output_dir,
            binary=args.binary,
            ref_base=args.ref_base,
            organism=args.organism,
            threads=args.threads,
            enable_snps=not args.no_snps,
            enable_pipeline_extras=args.pipeline_extras,
            cascade=args.cascade,
            te_classify=args.te_classify,
            nonhost=args.nonhost,
            raw_matrix=args.raw_matrix,
            metadata_json=args.metadata_json,
            work_dir=args.work_dir,
            keep_intermediate=args.keep_intermediate,
            extra_args=extra_args,
            quiet=args.quiet,
        )
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"OK: {result.accession} -> {result.output_dir}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
