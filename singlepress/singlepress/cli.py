#!/usr/bin/env python3
"""singlepress CLI — command-line interface for .1pz files.

Usage:
    singlepress info FILE.1pz
    singlepress validate FILE.1pz
    singlepress colsums FILE.1pz
    singlepress inspect FILE.1pz
"""
from __future__ import annotations

import argparse
import sys
import json


def cmd_info(args):
    """Print .1pz file header info."""
    import singlepress
    info = singlepress.info_1pz(args.file)
    for k, v in info.items():
        print(f"  {k}: {v}")


def cmd_validate(args):
    """Validate .1pz file CRC32 integrity."""
    import singlepress
    result = singlepress.validate_1pz(args.file)
    if result["valid"]:
        print(f"VALID: {args.file}")
    else:
        print(f"INVALID: {args.file}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print(f"  File CRC OK: {result.get('file_crc_ok')}")
        print(f"  Footer OK: {result.get('footer_ok')}")
    sys.exit(0 if result["valid"] else 1)


def cmd_colsums(args):
    """Print column sums from a .1pz file."""
    import singlepress
    cs = singlepress.colsums_1pz(args.file)
    if args.json:
        print(json.dumps(cs.tolist()))
    else:
        print(f"Column sums ({len(cs)} columns):")
        print(f"  min: {cs.min()}, max: {cs.max()}, mean: {cs.mean():.1f}")
        print(f"  total: {cs.sum()}")


def cmd_inspect(args):
    """Detailed inspection of .1pz file."""
    import singlepress
    import os

    info = singlepress.info_1pz(args.file)
    file_sz = os.path.getsize(args.file)

    print(f"File: {args.file}")
    print(f"  Size: {file_sz:,} bytes ({file_sz/1024/1024:.2f} MB)")
    print(f"  Shape: {info['m']:,} x {info['n']:,}")
    print(f"  NNZ: {info['nnz']:,}")
    density = 100 * info["nnz"] / max(info["m"] * info["n"], 1)
    print(f"  Density: {density:.2f}%")
    print(f"  Codec: {info.get('codec', '?')}")
    print(f"  Chunks: {info['num_chunks']} x {info.get('chunk_cols', '?')} cols")
    print(f"  Gap width: {'16-bit' if info.get('gap16') else '32-bit'}")
    print(f"  Ptr width: {info.get('ptr_width', '?')} bytes")

    if info.get("has_metadata"):
        print(f"  Metadata: YES")
    if info.get("has_colsums"):
        print(f"  Column sums: YES")
        cs = singlepress.colsums_1pz(args.file)
        print(f"    min={cs.min()}, max={cs.max()}, mean={cs.mean():.1f}")
    if info.get("has_transpose"):
        print(f"  Transpose: YES (row-range reads available)")
    if info.get("zstd_checksums"):
        print(f"  ZSTD checksums: YES")

    # Validate
    val = singlepress.validate_1pz(args.file)
    print(f"  CRC valid: {val.get('file_crc_ok', 'N/A')}")

    # Raw CSC size estimate
    raw_csc = (info["n"] + 1) * 4 + info["nnz"] * 4 + info["nnz"] * 8
    ratio = raw_csc / max(file_sz, 1)
    print(f"  Compression ratio: {ratio:.1f}x ({raw_csc:,} -> {file_sz:,})")


def main():
    parser = argparse.ArgumentParser(
        prog="singlepress",
        description="SinglePress .1pz file management tool",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # info
    p = sub.add_parser("info", help="Show file header info")
    p.add_argument("file", help=".1pz file path")

    # validate
    p = sub.add_parser("validate", help="Validate file CRC integrity")
    p.add_argument("file", help=".1pz file path")

    # colsums
    p = sub.add_parser("colsums", help="Print column sum statistics")
    p.add_argument("file", help=".1pz file path")
    p.add_argument("--json", action="store_true", help="Output as JSON array")

    # inspect
    p = sub.add_parser("inspect", help="Detailed file inspection")
    p.add_argument("file", help=".1pz file path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "info": cmd_info,
        "validate": cmd_validate,
        "colsums": cmd_colsums,
        "inspect": cmd_inspect,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
