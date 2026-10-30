#!/usr/bin/env python3
"""Publish a NEW study as .s1 objects: pipeline output -> verified `.s1` objects + manifest in R2 + catalog rows.

This replaces "pack a legacy bundle, upload it to data/<GSE>/<GSE>.singlet" in the Anvil publisher. Same inputs as
`singlet.bundle.pack_gse` (the pipeline results directory of the study's samples and the processing catalog). Steps, per study:

  1. pack_gse writes a legacy-format bundle into scratch. It is the definition of what a study contains and the oracle the
     verifier compares against; it is never uploaded and is deleted afterwards. With --private BUCKET the packer also writes the
     restricted sidecar (members with per-donor germline information) and it is uploaded to
     BUCKET/restricted/<GSE>/<GSE>.restricted.singlet; without it those members are not kept (as in the current publisher).
     A bundle that still holds restricted members is refused.
  2. `python -m singlet.s1.migrate`, manifest validation, `python -m singlet.s1.verify` (must exit 0), upload of every object
     (existing identical objects are skipped), ledger, manifest LAST, read-back: exactly the steps of scripts/s1_campaign.py
     (this script calls the same code).
  3. Catalog rows (s1_study, s1_sample, s1_migration; schema 032) are written to --rows-dir and, when $INGEST_TOKEN is set and
     --ingest-url is given, the manifest is POSTed to /api/ingest/s1-study. A new study never had a legacy file, so its migration row is
     `retired` (verify_ok 1) from the start and the site never offers a legacy URL for it. Without a token run `s1_campaign.py catalog-sql
     --rows-dir DIR` and apply the SQL (the primary route).

Dry run (no R2 writes) unless --apply. The last stdout line is `RESULT {json}`; the exit code is 0 when the study is live and
its catalog rows were accepted (or not requested), 1 when it was not published, 3 when it is live but the catalog POST failed
(rerun: `s1_campaign.py catalog-post --rows-dir DIR`). The gse/gsm catalog rows (titles, sources, ...) are not touched here.

  python scripts/s1_publish.py GSE123456 --results /scratch/results_v3 --catalog catalog.parquet --apply \\
      --private singlet-restricted --ingest-url https://singlet.bio

Credentials: R2_ACCOUNT_ID or R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (R2_BUCKET names the public bucket if
--public is not given); INGEST_TOKEN for the catalog POST.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_campaign as vc  # noqa: E402


def load_packer(module: str, path: Optional[str]):
    if path:
        sys.path.insert(0, path)
    mod = importlib.import_module(module)
    return mod.pack_gse


def make_provider(a: argparse.Namespace, packer):
    """The `provider` of process_study: pack from pipeline output instead of downloading a legacy bundle."""

    def provider(ctx: vc.Ctx, gse: str, work: Path, row: dict, events: List[dict]):
        bundle = work / f"{gse}.singlet"
        side = work / f"{gse}.restricted.singlet" if a.private else None
        params = inspect.signature(packer).parameters
        kw = dict(gse_id=gse, results_dir=a.results, catalog_path=a.catalog, out_path=str(bundle), verbose=False)
        for name, val in (("metadata_enriched_path", a.metadata_enriched), ("publications_enriched_path", a.publications_enriched),
                          ("series_meta_path", a.series_meta_out), ("private_out_path", str(side) if side else None)):
            if val:
                if name not in params:
                    raise vc.StudyFailure("failed", "pack", f"the packer has no {name} parameter (needs a singlet.bundle with the restricted sidecar)")
                kw[name] = val
        try:
            packer(**kw)
        except Exception as e:  # noqa: BLE001  e.g. "No done GSMs found", all samples hollow
            raise vc.StudyFailure("failed", "pack", f"{type(e).__name__}: {e}"[:500])
        if not bundle.is_file():
            raise vc.StudyFailure("failed", "pack", "the packer wrote no bundle")
        row["packed_bytes"] = bundle.stat().st_size
        row["packed_sha256"] = vc.sha256_file(bundle)
        events.append({"at": vc.iso(), "gse": gse, "event": "packed", "bytes": row["packed_bytes"], "sha256": row["packed_sha256"],
                       "packer": a.packer_module})
        summ = vc._restricted_module().restricted_summary(bundle)
        if summ["members"]:
            raise vc.StudyFailure("blocked_restricted", "pack", f"{summ['members']} restricted members in the public bundle")
        if side is not None and side.is_file() and ctx.apply:
            key = f"restricted/{gse}/{gse}.restricted.singlet"
            rec = vc.ensure_object(ctx, key, side, vc.hash_file(side, ctx.part_size), readback="sha256", bucket=a.private)
            events.append({"at": vc.iso(), "gse": gse, "event": "restricted-sidecar", **rec})
            row["restricted_sidecar"] = {"bucket": a.private, **rec}
        return bundle, None

    return provider


def run_args(a: argparse.Namespace, gse: str, scratch: Path) -> List[str]:
    args = ["run", "--mode", "full" if a.apply else "dry", "--gse", gse, "--workers", "1",
            "--io-threads", str(a.io_threads), "--blas-threads", str(a.blas_threads),
            "--migrate-timeout", str(a.migrate_timeout), "--verify-timeout", str(a.verify_timeout),
            "--scratch", str(scratch), "--rows-dir", str(a.rows_dir), "--ledger-dir", str(a.ledger_dir),
            "--readback", a.readback, "--public", a.public, "--commit", a.commit or os.environ.get("COMMIT", ""),
            "--members", "full"]
    for opt, val in (("--alpha-policy", a.policy), ("--migrate-cmd", a.migrate_cmd), ("--verify-cmd", a.verify_cmd), ("--pythonpath", a.pythonpath),
                     ("--local-root", a.local_root), ("--part-size-mib", a.part_size_mib), ("--retry-sleep", a.retry_sleep)):
        if val is not None:
            args += [opt, str(val)]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gse")
    ap.add_argument("--results", required=True, help="pipeline results directory (one child per GSM)")
    ap.add_argument("--catalog", required=True, help="processing catalog parquet (as for pack_gse)")
    ap.add_argument("--metadata-enriched")
    ap.add_argument("--publications-enriched")
    ap.add_argument("--series-meta-out", help="also write series_meta.json here (a local file, not uploaded)")
    ap.add_argument("--packer-module", default="singlet.bundle", help="module with pack_gse (Anvil: cf_pack_v3)")
    ap.add_argument("--packer-path", help="directory to put on sys.path for the packer module")
    ap.add_argument("--public", default=os.environ.get("R2_BUCKET", vc.PUBLIC_BUCKET))
    ap.add_argument("--private", help="private bucket for the restricted sidecar (restricted/<GSE>/<GSE>.restricted.singlet)")
    ap.add_argument("--policy", help="release policy JSON (version, withheld_studies, excluded_samples; scripts/s1_campaign.py "
                    "--alpha-policy): genotype, mtDNA allele, ancestry-call, ambient, doublet and legacy layers never reach an object")
    ap.add_argument("--local-root", help="directories under this root instead of R2 (tests)")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--rows-dir", help="catalog rows (default: <scratch>/rows)")
    ap.add_argument("--ledger-dir")
    ap.add_argument("--ingest-url", help="POST the catalog rows to <url>/api/ingest/<table> (token in $INGEST_TOKEN)")
    ap.add_argument("--token-env", default="INGEST_TOKEN")
    ap.add_argument("--readback", choices=["none", "sha256"], default="sha256")
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--migrate-timeout", type=float, default=7200)
    ap.add_argument("--verify-timeout", type=float, default=7200)
    ap.add_argument("--migrate-cmd")
    ap.add_argument("--verify-cmd")
    ap.add_argument("--pythonpath")
    ap.add_argument("--commit")
    ap.add_argument("--json-out", help="write the result row here")
    ap.add_argument("--apply", action="store_true", help="write to R2 (default: pack, migrate and verify only)")
    ap.add_argument("--part-size-mib", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--retry-sleep", type=float, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.private and a.private == a.public:
        ap.error("--private must differ from --public")

    import tempfile

    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="v2pub-"))
    scratch.mkdir(parents=True, exist_ok=True)
    a.rows_dir = a.rows_dir or str(scratch / "rows")
    a.ledger_dir = a.ledger_dir or str(scratch / "ledger")
    store = vc.LocalStore(Path(a.local_root)) if a.local_root else vc.S3Store()
    ns = vc.build_parser().parse_args(run_args(a, a.gse, scratch / "work"))
    ns.workers = 1
    ctx = vc.Ctx(ns, store)
    allow = {a.public: ("s1/",)}
    if a.private:
        allow[a.private] = ("restricted/",)
    ctx.w = vc.GuardedStore(store, allow)  # the only writes: store/ in the public bucket, restricted/ in the private one
    packer = load_packer(a.packer_module, a.packer_path)
    try:
        row = vc.process_study(ctx, a.gse, provider=make_provider(a, packer))
    finally:
        ctx.close()
    row.pop("report", None)
    ok = row["status"] in (("ok",) if a.apply else ("dry_ok",))
    rc = 0 if ok else 1
    rows_file = Path(a.rows_dir) / f"{a.gse}.json"
    if ok and a.apply and a.ingest_url:
        token = os.environ.get(a.token_env, "")
        try:
            if not token:
                raise RuntimeError(f"${a.token_env} is empty")
            manifest = json.loads(store.get_bytes(a.public, vc.manifest_key(a.gse)))
            vc.post_studies(a.ingest_url, token, {a.gse: (manifest, json.loads(rows_file.read_text("utf-8")))}, log=lambda *_: None)
            row["catalog"] = "posted"
        except Exception as e:  # noqa: BLE001  the study is live; the rows can be posted again
            row["catalog"] = f"not posted: {e}"[:300]
            rc = 3
    row["rows_file"] = str(rows_file) if rows_file.exists() else None
    if not a.keep_scratch:
        import shutil

        shutil.rmtree(scratch / "work", ignore_errors=True)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    print(vc.fmt_row(row))
    print("RESULT " + json.dumps(row, sort_keys=True), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
