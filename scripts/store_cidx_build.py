# SPDX-License-Identifier: MIT
"""Build one version of the atlas-wide cell index (docs/format-store-cell-index.md) and write it in the server layout.

    # the pilot, from a local mirror of data.singlet.bio (store/m, store/s), labels from the annotation pipeline's output directory
    python -P scripts/store_cidx_build.py --base /data/mirror --gse GSE145370 --labels-dir out/labels \\
        --sample-meta sample_meta.parquet --ontology cl-basic.obo --ontology uberon-basic.obo --out /data/cidx

    # with per-cell features (gene-set scores, malignancy, ...) and the per-sample aggregates for the catalog (D1)
    python -P scripts/store_cidx_build.py ... --features features.json --features-dir out/features --d1-export out/d1

    # the atlas on Clipper/Anvil: every migrated study (one GSE per line), keep the ordinals of the previous version
    python -P scripts/store_cidx_build.py --base https://data.singlet.bio --gse-file migrated.txt --previous 20261101T0000Z-ab12cd34 \\
        --threads 32 --out $WORK/cidx

Writes ``<out>/store/cidx/obj/*`` (content-addressed shards, sample table, summary, plan blocks), ``<out>/store/cidx/v/<version>.json``
(the directory) and, unless ``--no-current``, ``<out>/store/cidx/current.json``; prints a JSON report. Uploading is a separate,
owner-run step (``rclone copy <out>/store/cidx r2:singlet-data/store/cidx``: objects first, ``current.json`` last). Nothing here writes to
R2 or D1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from singlet.store import atlas  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base",
        help="data host URL or a directory with the server layout (default SINGLET_DATA_BASE)",
    )
    ap.add_argument("--gse", action="append", default=[], help="study accession (repeatable)")
    ap.add_argument("--gse-file", help="file with one GSE per line")
    ap.add_argument(
        "--labels-dir",
        help="annotation output: <dir>/<GSM>.labels.parquet or <dir>/<GSE>/<GSM>.labels.parquet",
    )
    ap.add_argument(
        "--annot-version", help="manifest annotation version to use when --labels-dir is not given"
    )
    ap.add_argument(
        "--sample-meta",
        help="Parquet/CSV/JSON (the catalog's sample_meta.csv): gsm, organism, tissue, tissue_label, disease, development_stage, donor_status, perturbation, sample_kind",
    )
    ap.add_argument(
        "--ontology",
        action="append",
        default=[],
        help="OBO or JSON ontology (CL, UBERON, HANCESTRO); repeatable",
    )
    ap.add_argument("--previous", help="earlier index version whose sample ordinals are kept")
    ap.add_argument(
        "--features",
        help='feature registry JSON (cell_feature rows): {"features": [{feature_id, kind, method_version, thresholds, ...}]}',
    )
    ap.add_argument(
        "--features-dir",
        help="feature files: <dir>/<GSM>.features.<set>.parquet or <dir>/<GSE>/<GSM>.features.<set>.parquet",
    )
    ap.add_argument(
        "--d1-export", help="also write the per-sample aggregates as CSV (+ suggested schema) here"
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--parquet",
        action="store_true",
        help="also write one Parquet file per organism (DuckDB access)",
    )
    ap.add_argument("--out", required=True, help="output root (server layout)")
    ap.add_argument("--no-current", action="store_true", help="do not write store/cidx/current.json")
    a = ap.parse_args(argv)

    gses = list(a.gse)
    if a.gse_file:
        gses += [
            x.strip()
            for x in Path(a.gse_file).read_text().splitlines()
            if x.strip() and not x.startswith("#")
        ]
    if not gses:
        ap.error("give --gse or --gse-file")
    ont = atlas.Ontology()
    for p in a.ontology:
        ont = ont.merge(atlas.Ontology.load(p))
    res = atlas.build_index(
        gses,
        base=a.base,
        sample_meta=a.sample_meta,
        labels_dir=a.labels_dir,
        annot_version=a.annot_version,
        ontology=ont,
        previous=a.previous,
        features=a.features,
        features_dir=a.features_dir,
        threads=a.threads,
        parquet=a.parquet,
        log=lambda s: print(s, file=sys.stderr),
    )
    keys = res.write(a.out, set_current=not a.no_current)
    if a.d1_export:
        res.d1_export(a.d1_export)
    print(
        json.dumps(
            {
                "version": res.version,
                "stats": res.stats,
                "errors": res.errors[:50],
                "n_errors": len(res.errors),
                "objects": len(keys),
            },
            indent=1,
        )
    )
    return 0 if not res.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
