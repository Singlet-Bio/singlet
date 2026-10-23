#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-cell feature sidecars for a list of studies (singlet.features; docs/cell_features.md).

  run       score samples and write OUT/<GSE>/<GSM>.features.<set>.parquet, <GSM>.features.rows.json (sample_cell_summary
            rows) and <GSM>.features.json (internal: coverage, dropped sets, timings; not for public serving)
  registry  write OUT/ingest/cell_feature.json, the cell_feature rows of every registered feature
  collect   gather every sample's summary rows into OUT/ingest/sample_cell_summary.<k>.json, <= 2000 rows per file

Studies: --gse GSE1 GSE2 ... or --list FILE (one study per line: "GSE" or "GSE GSM [GSM ...]" or "GSE path/to/bundle.singlet").
v2 studies are read from data.singlet.bio; legacy studies from a local bundle or one downloaded to --scratch (deleted after the
study unless --keep-bundles). --shard i --shards n takes every n-th study (Slurm arrays: scripts/features_run.sbatch). Samples
whose sidecars exist are skipped unless --force. Nothing is uploaded: the ingest and R2 commands are printed by `collect`.

  python scripts/features_run.py run --gse GSE138867 --out out/features
  python scripts/features_run.py registry --out out/features
  python scripts/features_run.py collect --out out/features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python"))

INGEST_MAX_ROWS = 2000


def read_list(a):
    items = []
    if a.list:
        for line in Path(a.list).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            gse, rest = parts[0], parts[1:]
            bundle = next((Path(p) for p in rest if p.endswith(".singlet")), None)
            gsms = [p for p in rest if p.startswith("GSM")] or None
            items.append((gse, gsms, bundle))
    for g in a.gse or []:
        items.append((g, a.gsm or None, Path(a.bundle) if a.bundle else None))
    return items[a.shard :: a.shards] if a.shards > 1 else items


def done(out: Path, gse: str, gsm: str, sets) -> bool:
    from singlet.features.output import sidecar_name

    d = out / gse
    return (d / f"{gsm}.features.rows.json").exists() and all(
        (d / sidecar_name(gsm, s)).exists() for s in sets
    )


def run_sample(s, out: Path, sets, read_s: float) -> dict:
    import numpy as np
    from singlet.features import pipeline as P
    from singlet.features.output import build_table, sidecar_name, write_features

    t0 = time.perf_counter()
    tim = {"read_s": round(read_s, 3)}
    res = P.score_sample(s, sets, timings=tim)
    d = out / s.gse
    order = np.argsort(s.col, kind="stable")
    col = np.asarray(s.col, np.int64)[order]
    info = {"gsm": s.gsm, "gse": s.gse, "species": s.species, "n_cells": int(len(col)), "sets": {}}
    for fs, r in res.items():
        tab = build_table(
            s.gsm,
            col,
            {k: np.asarray(v)[order] for k, v in r.values.items()},
            r.features,
            r.method_version,
        )
        write_features(tab, d / sidecar_name(s.gsm, fs), r.features, gene_symbols=s.gene_names)
        info["sets"][fs] = {
            "method_version": r.method_version,
            "dropped": r.dropped,
            "coverage": {k: round(v, 3) for k, v in r.coverage.items()},
        }
    rows = P.summaries(s, res)
    (d / f"{s.gsm}.features.rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    tim["total_score_s"] = round(time.perf_counter() - t0, 3)
    info["timings"] = {k: round(v, 3) for k, v in tim.items()}
    info["n_genes_detected_median"] = (
        float(np.median(np.diff(s.X.tocsr().indptr))) if len(col) else 0.0
    )
    (d / f"{s.gsm}.features.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return info


def cmd_run(a):
    from singlet.features import io as FIO
    from singlet.features.registry import FEATURE_SETS

    out = Path(a.out)
    sets = tuple(a.sets.split(",")) if a.sets else FEATURE_SETS
    ledger = Path(a.ledger) if a.ledger else out / "ledger" / f"shard-{a.shard}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch) if a.scratch else out / "scratch"

    failed = []

    def log(**k):
        if k.get("status") in ("read_error", "error", "study_error"):
            failed.append(k)
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **k}) + "\n"
            )

    for gse, gsms, bundle in read_list(a):
        fetched = None
        try:
            v2 = bundle is None and FIO.is_v2(gse)
            if not v2 and bundle is None:
                bundle = fetched = FIO.fetch_bundle(gse, scratch)
            todo = gsms or (FIO.v2_gsms(gse) if v2 else FIO.bundle_gsms(bundle))
            todo = [g for g in todo if a.force or not done(out, gse, g, sets)]
            if not todo:
                log(gse=gse, status="done")
                continue
            t = time.perf_counter()
            for gsm, s in FIO.iter_samples(gse, todo, bundle=bundle, v2=v2):
                read_s = time.perf_counter() - t
                if isinstance(s, Exception):
                    log(
                        gse=gse,
                        gsm=gsm,
                        status="read_error",
                        error=f"{type(s).__name__}: {s}"[:400],
                    )
                else:
                    try:
                        info = run_sample(s, out, sets, read_s)
                        log(
                            gse=gse,
                            gsm=gsm,
                            status="ok",
                            n_cells=info["n_cells"],
                            timings=info["timings"],
                        )
                        print(f"{gse} {gsm}: {info['n_cells']} cells {info['timings']}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        log(
                            gse=gse, gsm=gsm, status="error", error=f"{type(e).__name__}: {e}"[:400]
                        )
                        traceback.print_exc()
                t = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            log(gse=gse, status="study_error", error=f"{type(e).__name__}: {e}"[:400])
            traceback.print_exc()
        finally:
            if fetched is not None and not a.keep_bundles:
                fetched.unlink(missing_ok=True)
    if failed:
        print(f"{len(failed)} studies or samples failed; see {ledger}", file=sys.stderr)
        sys.exit(1)


def cmd_registry(a):
    from singlet.features.registry import FEATURE_SETS, registry_rows

    sets = tuple(a.sets.split(",")) if a.sets else FEATURE_SETS
    p = Path(a.out) / "ingest" / "cell_feature.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = registry_rows(sets)
    p.write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"{p}: {len(rows)} features")


def cmd_collect(a):
    out = Path(a.out)
    rows = []
    for f in sorted(out.glob("GSE*/GSM*.features.rows.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    d = out / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("sample_cell_summary.*.json"):
        old.unlink()
    files = []
    for k in range(0, len(rows), INGEST_MAX_ROWS):
        p = d / f"sample_cell_summary.{k // INGEST_MAX_ROWS:04d}.json"
        p.write_text(json.dumps({"rows": rows[k : k + INGEST_MAX_ROWS]}) + "\n", encoding="utf-8")
        files.append(p)
    print(f"{len(rows)} summary rows in {len(files)} files under {d}")
    print("Not run. To publish (after review):")
    print(
        '  curl -fsS -X POST https://singlet.bio/api/ingest/cell_feature -H "Authorization: Bearer $INGEST_TOKEN" '
        f'-H "Content-Type: application/json" --data-binary @{d / "cell_feature.json"}'
    )
    print(
        f"  for f in {d}/sample_cell_summary.*.json; do curl -fsS -X POST https://singlet.bio/api/ingest/sample_cell_summary "
        '-H "Authorization: Bearer $INGEST_TOKEN" -H "Content-Type: application/json" --data-binary @$f; done'
    )
    print(
        "  sidecars: the cell-index builder normalises them (singlet.v2.atlas.normalize_feature_sidecar -> "
        "v2/cf/<GSM>/<set>.<d16>.parquet) and attaches them to the study manifest; see docs/cell_features.md"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gse", nargs="*")
    r.add_argument("--gsm", nargs="*")
    r.add_argument("--bundle", help="local legacy bundle for a single --gse")
    r.add_argument("--list")
    r.add_argument("--out", required=True)
    r.add_argument("--sets", help="comma-separated feature sets (default: all)")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--scratch")
    r.add_argument("--ledger")
    r.add_argument("--keep-bundles", action="store_true")
    r.add_argument("--force", action="store_true")
    g = sub.add_parser("registry")
    g.add_argument("--out", required=True)
    g.add_argument("--sets")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"run": cmd_run, "registry": cmd_registry, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
