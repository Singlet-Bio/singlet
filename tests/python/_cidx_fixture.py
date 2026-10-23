# SPDX-License-Identifier: MIT
"""A small cell index over the synthetic v2 study of ``_v2_synth`` (real codec, real packs) with deterministic fake labels, plus the
expected answers of a fixed list of queries and gather plans.

Used by ``test_v2_atlas.py`` and, as golden files, by the data Worker's tests in the website repository::

    python -P tests/python/_cidx_fixture.py <website>/src/test/fixtures/cidx

writes the server layout (``v2/...``) and ``expected.json`` (query answers and gather plans computed by ``singlet.v2.atlas``)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

CT = [
    ("CL:0000084", "T cell"),
    ("CL:0000624", "CD4-positive, alpha-beta T cell"),
    ("CL:0000625", "CD8-positive, alpha-beta T cell"),
    ("CL:0000236", "B cell"),
    ("CL:0000576", "monocyte"),
]
ONTOLOGY = {
    "CL:0000000": {"label": "cell", "parents": []},
    "CL:0000542": {"label": "lymphocyte", "parents": ["CL:0000000"]},
    "CL:0000084": {"label": "T cell", "parents": ["CL:0000542"]},
    "CL:0000624": {"label": "CD4-positive, alpha-beta T cell", "parents": ["CL:0000084"]},
    "CL:0000625": {"label": "CD8-positive, alpha-beta T cell", "parents": ["CL:0000084"]},
    "CL:0000236": {"label": "B cell", "parents": ["CL:0000542"]},
    "CL:0000576": {"label": "monocyte", "parents": ["CL:0000000"]},
    "UBERON:0002048": {"label": "lung", "parents": ["UBERON:0000062"]},
    "UBERON:0002185": {"label": "bronchus", "parents": ["UBERON:0002048"]},
    "UBERON:0000062": {"label": "organ", "parents": []},
    "HANCESTRO:0005": {"label": "European", "parents": []},
    "HANCESTRO:0009": {"label": "East Asian", "parents": []},
}
TISSUE = {
    "GSM9000001": ("UBERON:0002048", "lung"),
    "GSM9000002": ("UBERON:0002048", "lung"),
    "GSM9000003": ("UBERON:0002185", "bronchus"),
}

QUERIES = [
    {"where": {}, "return": "count"},
    {"where": {"cell_type": "CL:0000084"}, "return": "count"},
    {
        "where": {"cell_type": "T cell", "tissue": "UBERON:0002048"},
        "return": "count",
        "by": ["cell_type"],
    },
    {"where": {"tissue": "UBERON:0002185"}, "return": "count", "by": ["sample", "sex"]},
    {"where": {"ranges": {"n_umi": [3000, None]}}, "return": "count"},
    {
        "where": {
            "cell_type": "CL:0000084",
            "ranges": {"n_umi": [2000, None], "cell_type_score": [0.5, None]},
        },
        "return": "count",
        "by": ["tissue"],
    },
    {"where": {"samples": ["GSM9000003"], "sex": ["female", "ambiguous"]}, "return": "count"},
    {
        "where": {"studies": ["GSE900001"], "ancestry": "HANCESTRO:0005"},
        "return": "count",
        "by": ["study"],
    },
    {"where": {"cell_type": "CL:0000625"}, "return": "cells"},
    {"where": {"ranges": {"n_genes": [150, 400], "ancestry_prob": [None, 0.9]}}, "return": "cells"},
    {"where": {"samples": ["GSM9000001"], "ranges": {"n_umi": [0, 2500]}}, "return": "cells"},
]


def fake_labels(root: str, synth) -> str:
    """Interim label files (annotation pipeline schema) for the synthetic samples with called cells."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(root, "labels")
    os.makedirs(out, exist_ok=True)
    for k, (gsm, t) in enumerate(sorted(synth.truth.items())):
        cols = np.flatnonzero(t["called"])
        if not len(cols):
            continue
        rng = np.random.default_rng(100 + k)
        n = len(cols)
        pick = rng.integers(0, len(CT), n)
        score = np.round(rng.uniform(0.2, 1.0, n), 3).astype(np.float32)
        sex = np.array(["female", "male", "ambiguous"])[np.where(rng.random(n) < 0.1, 2, k % 2)]
        anc = np.where(rng.random(n) < 0.7, "HANCESTRO:0005", "HANCESTRO:0009")
        tab = pa.table(
            {
                "gsm": [gsm] * n,
                "col": cols.astype(np.int64),
                "cell_type_cl": [CT[i][0] for i in pick],
                "cell_type_label": [CT[i][1] for i in pick],
                "cell_type_score": score,
                "sex_chrom": sex.tolist(),
                "sex_score": np.round(rng.uniform(0.5, 1, n), 3).astype(np.float32),
                "ancestry_hancestro": anc.tolist(),
                "ancestry_prob": np.round(rng.uniform(0.6, 1, n), 3).astype(np.float32),
                "model_version": ["test-1"] * n,
            }
        )
        pq.write_table(tab, os.path.join(out, f"{gsm}.labels.parquet"))
    return out


def build(root: str):
    """-> (synth, build result, index) with everything written under ``root`` (server layout)."""
    import _v2_synth as synth_mod
    from singlet.v2 import atlas

    s = synth_mod.Synth(root).build(n_cols=600, n_called=200, chunk_cols=64)
    labels = fake_labels(root, s)
    meta = os.path.join(root, "sample_meta.json")
    with open(meta, "w") as fh:
        json.dump(
            [
                {"gsm": g, "organism": "Homo sapiens", "tissue": t, "tissue_label": lab}
                for g, (t, lab) in TISSUE.items()
            ],
            fh,
        )
    res = atlas.build_index(
        [synth_mod.GSE],
        base=root,
        sample_meta=meta,
        labels_dir=labels,
        ontology=atlas.Ontology(ONTOLOGY),
        now="20261101T0000Z",
        group_rows=64,
        shard_rows=128,
    )
    res.write(root)
    idx = atlas.Index.open(base=root)
    return s, res, idx


def expected(idx) -> dict:
    """Answers of QUERIES (as the Worker returns them) and two gather plans."""
    from singlet.v2 import atlas

    out = {"version": idx.version, "queries": [], "gathers": []}
    for q in QUERIES:
        if q["return"] == "count":
            r = idx.count(q["where"], by=q.get("by", []))
            if q.get("by"):
                rows = r.to_dict("records")
                ans = {
                    "groups": sorted(rows, key=lambda x: json.dumps(x, sort_keys=True, default=str))
                }
            else:
                ans = {"count": int(r)}
        else:
            df = idx.find(q["where"], limit=None)
            ans = {"so": df["so"].tolist(), "col": df["col"].tolist(), "gsm": df["gsm"].tolist()}
        out["queries"].append(
            {
                "query": q,
                "answer": json.loads(
                    json.dumps(ans, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                ),
            }
        )
    df = idx.find({"cell_type": "CL:0000084"}, limit=None)
    for members, have in (
        (("exon_counts.1pz", "intron_counts.1pz"), []),
        (("exon_counts.1pz",), "exon"),
    ):
        want = atlas._cells_by_sample(idx, df)
        ent = [idx.sample(s) for s in sorted(want)]
        hv = [e["members"]["exon_counts.1pz"]["digest"] for e in ent] if have == "exon" else []
        p = atlas.plan_gather(ent, want, list(members), hv)
        out["gathers"].append(
            {
                "cells": [[int(s), want[s].tolist()] for s in sorted(want)],
                "members": list(members),
                "have_prefix": hv,
                "bytes": p["bytes"],
                "payload": p["payload"],
                "reads": [
                    {"sample": r["sample"], "key": r["key"], "start": r["start"], "end": r["end"]}
                    for r in p["reads"]
                ],
            }
        )
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1])
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    s, res, idx = build(tmp)
    exp = expected(idx)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "v2").mkdir(parents=True)
    # sample objects are left out: the Worker does not decode them, its tests fabricate bytes of the planned sizes
    keep = [k for k in res.objects] + ["v2/cidx/current.json"]
    exp["sample_objects"] = {t["key"]: len(t["object"]) for t in s.truth.values()}
    for k in keep:
        src = Path(tmp).joinpath(*k.split("/"))
        p = dest.joinpath(*k.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
    (dest / "expected.json").write_text(json.dumps(exp, indent=1))
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"{dest}: {len(keep)} objects, {total:,} bytes, index {idx.version}")
