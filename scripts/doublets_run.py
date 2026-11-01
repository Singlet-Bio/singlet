#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Batch job: the registered per-cell feature ``qc/doublet_score`` for a list of studies (singlet.doublets; docs/doublet_score.md).

  run       score samples and write OUT/<GSE>/<GSM>.features.qc.parquet (the feature-sidecar layout of docs/cell_features.md),
            <GSM>.features.rows.json (its sample_cell_summary rows) and <GSM>.doublets.json (internal: status, timings)
  registry  write OUT/ingest/cell_feature.json, the cell_feature row of the feature
  collect   gather every sample's summary rows into OUT/ingest/sample_cell_summary.<k>.json, <= 2000 rows per file

Inputs besides the studies: --dbl-run (the engine's doublet harness, built by bench/doublets/build_dbl_run.sh), --model (a
celltype_nn model.pt) and --centroids (atlas cell-type centroids, atlas_centroids.npz with key M). Studies as in
scripts/features_run.py (--gse, or --list FILE of "GSE [GSM ...]"); only per-sample .s1 studies are read. Samples whose sidecar
exists are skipped unless --force. Nothing is uploaded: `collect` prints the ingest commands.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python"))

INGEST_MAX_ROWS = 2000


def read_list(a):
    items = []
    if a.list:
        for line in Path(a.list).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and not parts[0].startswith("#"):
                items.append((parts[0], [p for p in parts[1:] if p.startswith("GSM")] or None))
    items += [(g, a.gsm or None) for g in a.gse or []]
    return items[a.shard :: a.shards] if a.shards > 1 else items


def cmd_run(a):
    import numpy as np
    from singlet.celltype_nn import CellTypePredictor
    from singlet.doublets import score as S
    from singlet.doublets.combiner import Combiner
    from singlet.features import io as FIO
    from singlet.features.output import build_table, sidecar_name, write_features

    out = Path(a.out)
    comb = Combiner.load(a.combiner)
    pred = CellTypePredictor.load(a.model)
    cen = np.load(a.centroids)["M"]
    version = S.method_version(comb, a.celltype_version or pred.version, a.dbl_commit)
    feat = replace(S.FEATURE, method_version=version)
    ledger = Path(a.ledger) if a.ledger else out / "ledger" / f"shard-{a.shard}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    failed = []

    def log(**k):
        if k.get("status") in ("read_error", "error", "study_error"):
            failed.append(k)
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **k}) + "\n"
            )

    for gse, gsms in read_list(a):
        try:
            if not FIO.is_s1(gse):
                log(gse=gse, status="skipped_not_s1")
                continue
            d = out / gse
            todo = [
                g
                for g in (gsms or FIO.s1_gsms(gse))
                if a.force or not (d / f"{g}.features.rows.json").exists()
            ]
            t = time.perf_counter()
            for gsm, s in FIO.iter_samples(gse, todo, s1=True):
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
                        res = S.score_sample(
                            s,
                            comb,
                            dbl_run=a.dbl_run,
                            predictor=pred,
                            centroids=cen,
                            threads=a.threads,
                            score_pools=a.score_genotype_pools,
                        )
                        order = np.argsort(s.col, kind="stable")
                        tab = build_table(
                            s.gsm,
                            np.asarray(s.col, np.int64)[order],
                            {feat.name: res.score[order]},
                            [feat],
                            version,
                        )
                        write_features(
                            tab,
                            d / sidecar_name(s.gsm, feat.feature_set),
                            [feat],
                            gene_symbols=s.gene_names,
                        )
                        rows = S.summaries(s, res, version) if res.status == "ok" else []
                        (d / f"{s.gsm}.features.rows.json").write_text(
                            json.dumps(rows) + "\n", encoding="utf-8"
                        )
                        info = dict(
                            gsm=s.gsm,
                            gse=s.gse,
                            species=s.species,
                            n_cells=int(len(order)),
                            n_donors=S.n_donors(s),
                            status=res.status,
                            read_s=round(read_s, 3),
                            timings={
                                k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in res.timings.items()
                            },
                            max_rss_gb=round(
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2
                            ),
                            method_version=version,
                        )
                        (d / f"{s.gsm}.doublets.json").write_text(
                            json.dumps(info, indent=1) + "\n", encoding="utf-8"
                        )
                        log(
                            gse=gse,
                            gsm=gsm,
                            status=res.status,
                            n_cells=info["n_cells"],
                            timings=info["timings"],
                        )
                        print(
                            f"{gse} {gsm}: {info['n_cells']} cells {res.status} {info['timings']}",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001 - reported per sample
                        log(
                            gse=gse, gsm=gsm, status="error", error=f"{type(e).__name__}: {e}"[:400]
                        )
                        traceback.print_exc()
                t = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            log(gse=gse, status="study_error", error=f"{type(e).__name__}: {e}"[:400])
            traceback.print_exc()
    if failed:
        print(f"{len(failed)} studies or samples failed; see {ledger}", file=sys.stderr)
        sys.exit(1)


def cmd_registry(a):
    from singlet.celltype_nn import CellTypePredictor
    from singlet.doublets import score as S
    from singlet.doublets.combiner import Combiner

    version = S.method_version(
        Combiner.load(a.combiner),
        a.celltype_version or CellTypePredictor.load(a.model).version,
        a.dbl_commit,
    )
    p = Path(a.out) / "ingest" / "cell_feature.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rows": S.registry_rows(version)}, indent=1) + "\n", encoding="utf-8")
    print(f"{p}: 1 feature, {version}")


def cmd_collect(a):
    out = Path(a.out)
    rows = []
    for f in sorted(out.glob("GSE*/GSM*.features.rows.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    d = out / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("sample_cell_summary.*.json"):
        old.unlink()
    for k in range(0, len(rows), INGEST_MAX_ROWS):
        (d / f"sample_cell_summary.{k // INGEST_MAX_ROWS:04d}.json").write_text(
            json.dumps({"rows": rows[k : k + INGEST_MAX_ROWS]}) + "\n", encoding="utf-8"
        )
    print(
        f"{len(rows)} summary rows under {d}. Not published: POST cell_feature.json to /api/ingest/cell_feature, then the"
    )
    print(
        "  summary files to /api/ingest/sample_cell_summary (X-Ingest-Token), after review; sidecars go through"
    )
    print(
        "  singlet.s1.atlas.normalize_feature_sidecar as for the other feature sets (docs/cell_features.md)."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--combiner", help="combiner .npz (default: the packaged one)")
        p.add_argument("--model", required=True, help="celltype_nn model.pt")
        p.add_argument(
            "--celltype-version",
            help="recorded in method_version (default: the model file's own version)",
        )
        p.add_argument(
            "--dbl-commit",
            default="unpinned",
            help="commit of the dbl_run source, recorded in method_version",
        )

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--gse", nargs="*")
    r.add_argument("--gsm", nargs="*")
    r.add_argument("--list")
    r.add_argument("--dbl-run", required=True)
    r.add_argument("--centroids", required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--ledger")
    r.add_argument("--force", action="store_true")
    r.add_argument(
        "--score-genotype-pools",
        action="store_true",
        help="also score pools with >= 2 donors (benchmarks only; their doublet call is the genotype one)",
    )
    common(sub.add_parser("registry"))
    sub.add_parser("collect").add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"run": cmd_run, "registry": cmd_registry, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
