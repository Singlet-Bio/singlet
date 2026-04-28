"""
Command-line entry point for :mod:`singlify`.

Currently exposes:

    singlify convert DIR [-o OUTPUT.h5ad] [--primary-assay NAME]
        Read a pipeline output directory via read_anndata and write to
        a single .h5ad file.

    singlify info PATH
        Print a one-screen summary of a .1pz file or pipeline directory
        without decoding the full matrix.

    singlify verify PATH
        Open a .1pz file with CRC verification enabled and report
        pass/fail. Useful as a disk-corruption canary.

The CLI is wired via ``[project.scripts]`` in ``pyproject.toml`` so
``pip install singlify`` puts ``singlify`` on the user's PATH.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional


def _cmd_convert(args: argparse.Namespace) -> int:
    try:
        from .interop.anndata import read_anndata
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: `pip install anndata` to enable the convert command.",
            file=sys.stderr,
        )
        return 2

    input_dir = pathlib.Path(args.input)
    if not input_dir.is_dir():
        print(f"error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.output) if args.output else input_dir.with_suffix(".h5ad")
    print(f"[convert] reading {input_dir}", file=sys.stderr)
    adata = read_anndata(input_dir, primary_assay=args.primary_assay)
    print(
        f"[convert] loaded {adata.n_obs} cells × {adata.n_vars} genes "
        f"(layers={sorted(adata.layers.keys())})",
        file=sys.stderr,
    )
    print(f"[convert] writing {out_path}", file=sys.stderr)
    adata.write_h5ad(out_path)
    print(f"[convert] done ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from .io import open_pz, read_metadata

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    if path.is_dir():
        # Summarise every .1pz in the directory.
        files = sorted(path.glob("*.1pz"))
        if not files:
            print(f"error: no .1pz files in {path}", file=sys.stderr)
            return 1
        print(f"# singlify pipeline directory: {path}")
        total_bytes = 0
        for f in files:
            with open_pz(f) as pz:
                sz_mb = f.stat().st_size / (1 << 20)
                total_bytes += f.stat().st_size
                print(
                    f"  {f.name:<28} {pz.header.m:>7} × {pz.header.n:<7} "
                    f"nnz={pz.header.nnz:>10,}  "
                    f"vt={pz.header.vt_code.name:<7}  {sz_mb:>6.1f} MB"
                )
        print(f"  {'total':<28} {'':<17} {'':<16}  {total_bytes / (1 << 20):>6.1f} MB")

        # Print user_kv from the first file (all matrices carry the same)
        with open_pz(files[0]) as pz:
            md = read_metadata(pz)
            if md and md.user_kv:
                print("\n# Embedded metadata")
                for k in sorted(md.user_kv.keys()):
                    print(f"  {k}: {md.user_kv[k]}")
        return 0

    # Single file
    with open_pz(path) as pz:
        sz_mb = path.stat().st_size / (1 << 20)
        print(f"# {path}")
        print(f"  shape:   {pz.header.m} × {pz.header.n}")
        print(f"  nnz:     {pz.header.nnz:,}")
        print(f"  vt_code: {pz.header.vt_code.name}")
        print(f"  chunks:  {pz.header.num_chunks} (chunk_cols={pz.header.chunk_cols})")
        print(f"  flags:   {pz.header.flags!r}")
        print(f"  size:    {sz_mb:.1f} MB")
        md = read_metadata(pz)
        if md:
            if md.rownames:
                print(f"  rownames: {len(md.rownames)} features (e.g. {md.rownames[0]})")
            if md.colnames:
                print(f"  colnames: {len(md.colnames)} cells (e.g. {md.colnames[0]})")
            if md.user_kv:
                print("\n# Embedded metadata")
                for k in sorted(md.user_kv.keys()):
                    print(f"  {k}: {md.user_kv[k]}")
    return 0


def _cmd_cohort(args: argparse.Namespace) -> int:
    """Bulk read a cohort of pipeline output directories and merge to one h5ad.

    Walks ``cohort_root`` looking for subdirectories whose name starts with
    ``GSM`` (or are explicitly listed via ``--gsm``). For each, reads the
    matrix named by ``--matrix`` and concatenates them column-wise (cells)
    into a single AnnData. Each cell's source GSM is recorded in
    ``adata.obs["gsm_id"]``. Writes the result as a single ``.h5ad``.
    """
    try:
        from .interop.anndata import read_anndata
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import anndata as ad

    root = pathlib.Path(args.cohort_root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    if args.gsm:
        sample_dirs = [root / g for g in args.gsm]
        sample_dirs = [d for d in sample_dirs if d.is_dir()]
    else:
        sample_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and d.name.startswith("GSM")
        )
    if not sample_dirs:
        print(f"error: no GSM subdirectories under {root}", file=sys.stderr)
        return 1

    print(f"[cohort] {len(sample_dirs)} samples in {root}", file=sys.stderr)

    adatas = []
    for i, sd in enumerate(sample_dirs):
        print(f"  [{i+1}/{len(sample_dirs)}] {sd.name}", file=sys.stderr, flush=True)
        try:
            a = read_anndata(sd, primary_assay=args.primary_assay)
        except Exception as e:
            print(f"    skip — {e}", file=sys.stderr)
            continue
        gsm = a.uns.get("singlify", {}).get("gsm_id", sd.name)
        a.obs["gsm_id"] = gsm
        # Prefix obs_names with the directory name so cells from different
        # samples don't collide in the merged AnnData. We deliberately use
        # sd.name (directory), not the metadata gsm_id, because the metadata
        # can be identical across symlinks or re-runs — the directory path
        # is what's actually unique on disk, and collisions there indicate
        # a user setup error rather than a silent merge.
        a.obs_names = [f"{sd.name}_{bc}" for bc in a.obs_names]
        adatas.append(a)

    if not adatas:
        print("error: no readable samples", file=sys.stderr)
        return 1

    print(f"[cohort] merging {len(adatas)} AnnData objects", file=sys.stderr)
    merged = ad.concat(
        adatas,
        join="outer",
        label="cohort_index",
        index_unique=None,
        merge="same",
    )
    print(f"[cohort] merged: {merged.n_obs} cells × {merged.n_vars} genes", file=sys.stderr)

    out_path = pathlib.Path(args.output)
    print(f"[cohort] writing {out_path}", file=sys.stderr)
    merged.write_h5ad(out_path)
    print(f"[cohort] done ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Walk a GSE root and print one row per GSM sample.

    Reads each ``gene_counts.1pz`` (or ``--matrix``) header only — no
    matrix decode — so it's fast even on large cohorts. Useful as a
    smoke-test pass before running ``singlify cohort`` on a directory.

    Output format selected by ``--format``:

    - ``table`` (default): aligned columns for human reading
    - ``tsv``: tab-separated values with header row, suitable for
      pipelining into awk/sort/sqlite
    - ``json``: one JSON object per line (jsonl), suitable for ``jq``
    """
    from .io import open_pz, read_metadata

    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    sample_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("GSM")
    )
    if not sample_dirs:
        print(f"error: no GSM* subdirectories under {root}", file=sys.stderr)
        return 1

    rows = []
    for sd in sample_dirs:
        target = sd / f"{args.matrix}.1pz"
        if not target.is_file():
            rows.append({
                "gsm_id": sd.name,
                "status": "missing",
                "matrix": args.matrix,
            })
            continue
        try:
            with open_pz(target) as pz:
                md = read_metadata(pz)
                kv = md.user_kv if md else {}
                rows.append({
                    "gsm_id": sd.name,
                    "status": "ok",
                    "m": int(pz.header.m),
                    "n": int(pz.header.n),
                    "nnz": int(pz.header.nnz),
                    "vt_code": pz.header.vt_code.name,
                    "protocol": kv.get("protocol", ""),
                    "gse_id": kv.get("gse_id", ""),
                    "organism": kv.get("organism", ""),
                    "singlify_version": kv.get("singlify_version", ""),
                    "pipeline_date": kv.get("pipeline_date", ""),
                })
        except Exception as e:
            rows.append({
                "gsm_id": sd.name,
                "status": "error",
                "error": str(e),
            })

    fmt = args.format
    if fmt == "json":
        for r in rows:
            print(json.dumps(r))
    elif fmt == "tsv":
        cols = ["gsm_id", "status", "protocol", "m", "n", "nnz", "vt_code",
                "gse_id", "organism", "singlify_version", "pipeline_date"]
        print("\t".join(cols))
        for r in rows:
            print("\t".join(str(r.get(c, "")) for c in cols))
    else:  # table
        print(f"# {root}  ({len(rows)} samples)")
        print(f"# {'gsm_id':<14} {'protocol':<10} {'shape':<18} {'nnz':>10} "
              f"{'vt':<6} {'gse_id':<12}")
        for r in rows:
            if r["status"] == "missing":
                print(f"  {r['gsm_id']:<14} {'(no ' + r['matrix'] + ')':<48}")
            elif r["status"] == "error":
                print(f"  {r['gsm_id']:<14} ERROR: {r['error']}", file=sys.stderr)
            else:
                shape = f"{r['m']}x{r['n']}"
                print(
                    f"  {r['gsm_id']:<14} "
                    f"{r['protocol']:<10} "
                    f"{shape:<18} "
                    f"{r['nnz']:>10,} "
                    f"{r['vt_code']:<6} "
                    f"{r['gse_id']:<12}"
                )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .io import open_pz, PZError

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    try:
        with open_pz(path, verify_crc=True) as pz:
            print(f"OK {path} ({pz.header.m} × {pz.header.n}, nnz={pz.header.nnz:,})")
        return 0
    except PZError as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="singlify",
        description="Command-line interface to the singlify Python package.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    convert = sub.add_parser(
        "convert",
        help="Read a pipeline output directory and write it as a single .h5ad file.",
    )
    convert.add_argument("input", help="Pipeline output directory")
    convert.add_argument(
        "-o", "--output", default=None,
        help="Output .h5ad path (default: input dir name + .h5ad)",
    )
    convert.add_argument(
        "--primary-assay", default="spliced",
        help="Per-gene matrix to use for .X (default: spliced)",
    )
    convert.set_defaults(func=_cmd_convert)

    info = sub.add_parser(
        "info",
        help="Print a summary of a .1pz file or pipeline output directory.",
    )
    info.add_argument("path", help=".1pz file or pipeline directory")
    info.set_defaults(func=_cmd_info)

    verify = sub.add_parser(
        "verify",
        help="Verify the CRC32 of a .1pz file.",
    )
    verify.add_argument("path", help=".1pz file")
    verify.set_defaults(func=_cmd_verify)

    list_cmd = sub.add_parser(
        "list",
        help="Walk a GSE directory and print one row per GSM (header-only, fast).",
    )
    list_cmd.add_argument(
        "root",
        help="Directory containing GSM* subdirectories",
    )
    list_cmd.add_argument(
        "--matrix", default="gene_counts",
        help="Which .1pz file to inspect per GSM (default: gene_counts)",
    )
    list_cmd.add_argument(
        "--format", choices=["table", "tsv", "json"], default="table",
        help="Output format (default: table)",
    )
    list_cmd.set_defaults(func=_cmd_list)

    cohort = sub.add_parser(
        "cohort",
        help="Bulk-load a directory of GSM samples and merge to one .h5ad.",
    )
    cohort.add_argument(
        "cohort_root",
        help="Directory containing GSM* subdirectories (e.g. quant/scrna/GSE174/GSE174399)",
    )
    cohort.add_argument(
        "-o", "--output", required=True,
        help="Output .h5ad path",
    )
    cohort.add_argument(
        "--primary-assay", default="spliced",
        help="Per-gene matrix to use for .X (default: spliced)",
    )
    cohort.add_argument(
        "--gsm", nargs="*", default=None,
        help="Explicit GSM list (otherwise: all GSM* subdirs of cohort_root)",
    )
    cohort.set_defaults(func=_cmd_cohort)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
