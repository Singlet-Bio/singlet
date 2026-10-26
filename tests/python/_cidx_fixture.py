# SPDX-License-Identifier: MIT
"""A small cell index over two synthetic per-sample studies with deterministic fake labels and per-cell features, plus the expected answers
of a fixed list of queries, sampled cell lists and gather plans.

The first study is ``_store_synth``'s (real codec, real packs); the second (``GSE900002``) is a manifest that names the same sample
objects under other GSMs, with other labels and features, so study ranking has something to rank.

Used by ``test_store_atlas.py`` and, as golden files, by the data Worker's tests in the website repository::

    python -P tests/python/_cidx_fixture.py <website>/src/test/fixtures/cidx

writes the server layout (``store/...``) and ``expected.json`` (answers computed by ``singlet.store.atlas``)."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

GSE2 = "GSE900002"
COPIES = {"GSM9000011": "GSM9000001", "GSM9000012": "GSM9000002", "GSM9000013": "GSM9000003"}
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
    "MONDO:0004992": {"label": "cancer", "parents": []},
    "MONDO:0005061": {"label": "lung adenocarcinoma", "parents": ["MONDO:0004992"]},
    "PATO:0000461": {"label": "normal", "parents": []},
}
TISSUE = {
    "GSM9000001": ("UBERON:0002048", "lung"),
    "GSM9000002": ("UBERON:0002048", "lung"),
    "GSM9000003": ("UBERON:0002185", "bronchus"),
    "GSM9000011": ("UBERON:0002185", "bronchus"),
    "GSM9000012": ("UBERON:0002185", "bronchus"),
    "GSM9000013": ("UBERON:0002048", "lung"),
}
FEATURES = [
    {
        "feature_id": "test/hypoxia",
        "kind": "score",
        "lo": -1.0,
        "hi": 1.0,
        "thresholds": [0.0, 0.5],
        "method": "test-ucell",
        "method_version": "test-hallmark-1",
        "description": "test gene-set score",
        "licence": "CC0",
    },
    {
        "feature_id": "test/malignancy",
        "kind": "score",
        "dtype": "u16",
        "method_version": "test-mal-1",
        "description": "test malignancy probability",
    },
    {"feature_id": "test/cycling", "kind": "flag", "method_version": "test-1"},
    {
        "feature_id": "test/phase",
        "kind": "category",
        "categories": ["G1", "S", "G2M"],
        "method_version": "test-1",
    },
]
# sample-level properties (the ontology session's sample_meta.csv columns)
PROPS = {
    "GSM9000001": {
        "donor_status": "healthy",
        "perturbation": "control",
        "sample_kind": "primary",
        "disease": "PATO:0000461",
        "disease_label": "normal",
        "development_stage": "HsapDv:0000154",
        "development_stage_label": "60-year-old stage",
    },
    "GSM9000003": {
        "donor_status": "diseased",
        "perturbation": "treated",
        "sample_kind": "primary",
        "disease": "MONDO:0004992",
        "disease_label": "cancer",
    },
    "GSM9000011": {
        "donor_status": "diseased",
        "perturbation": "treated;genetic",
        "sample_kind": "organoid",
        "disease": "MONDO:0005061",
        "disease_label": "lung adenocarcinoma",
    },
    "GSM9000013": {"sample_kind": "cell_line"},
}
NO_FEATURES = {"GSM9000013"}  # a sample without a feature file: every value NA

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
    {"where": {"ancestry": "HANCESTRO:0005"}, "return": "count", "by": ["study"]},
    {"where": {"ranges": {"hypoxia": [0.0, None]}}, "return": "count", "by": ["study"]},
    {
        "where": {
            "cell_type": "B cell",
            "ranges": {"malignancy": [0.8, None], "test/hypoxia": [None, 0.5]},
        },
        "return": "count",
    },
    {
        "where": {"categories": {"test/phase": ["S", "G2M"]}, "ranges": {"cycling": [1, 1]}},
        "return": "count",
        "by": ["study"],
    },
    {"where": {"donor_status": "diseased"}, "return": "count", "by": ["sample"]},
    {"where": {"disease": "cancer", "perturbation": ["genetic"]}, "return": "count"},
    {
        "where": {"sample_kind": ["primary", "unknown"], "cell_type": "T cell"},
        "return": "count",
        "by": ["sample"],
    },
    {"where": {"cell_type": "CL:0000625"}, "return": "cells"},
    {"where": {"ranges": {"n_genes": [150, 400], "ancestry_prob": [None, 0.9]}}, "return": "cells"},
    {"where": {"samples": ["GSM9000001"], "ranges": {"n_umi": [0, 2500]}}, "return": "cells"},
    {"where": {"cell_type": "T cell"}, "return": "cells", "n": 50, "seed": 7, "max_studies": 1},
    {"where": {}, "return": "cells", "n": 40, "seed": 3, "max_studies": 2, "sampling": "chunks"},
    {
        "where": {"studies": [GSE2, "GSE900001"]},
        "return": "cells",
        "n": 30,
        "seed": 1,
        "max_studies": 1,
        "study_rank": "given",
    },
    {"where": {}, "return": "cells", "n": 25, "seed": 11, "max_studies": 1, "study_rank": "random"},
    {"where": {"ranges": {"malignancy": [0.8, None]}}, "return": "cells", "n": 10000},
    {
        "where": {"donor_status": "diseased", "categories": {"phase": ["G1"]}},
        "return": "cells",
        "n": 20,
        "seed": 4,
    },
]
PAGED = {"where": {"tissue": "lung"}, "return": "cells", "n": 60, "seed": 5, "max_studies": 2}


def _called(synth, gsm):
    return np.flatnonzero(synth.truth[COPIES.get(gsm, gsm)]["called"])


def fake_labels(root: str, synth, gsms) -> str:
    """Interim label files (annotation pipeline schema) for the synthetic samples with called cells."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(root, "labels")
    os.makedirs(out, exist_ok=True)
    for k, gsm in enumerate(gsms):
        cols = _called(synth, gsm)
        if not len(cols):
            continue
        rng = np.random.default_rng(100 + k)
        n = len(cols)
        p = [0.2, 0.2, 0.2, 0.2, 0.2] if gsm not in COPIES else [0.35, 0.25, 0.25, 0.1, 0.05]
        pick = rng.choice(len(CT), n, p=p)
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


def fake_features(root: str, synth, gsms) -> str:
    """``<GSM>.features.test.parquet`` in the agreed interchange (gsm, col, one column per feature; float32 scores, bool flags,
    uint16 categories; metadata singlet_features_schema, feature_set, method_version), 10 % of hypoxia missing, none for NO_FEATURES."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(root, "features")
    os.makedirs(out, exist_ok=True)
    for k, gsm in enumerate(gsms):
        cols = _called(synth, gsm)
        if not len(cols) or gsm in NO_FEATURES:
            continue
        rng = np.random.default_rng(500 + k)
        n = len(cols)
        hyp = np.round(rng.uniform(-1, 1, n), 4)
        hyp[rng.random(n) < 0.1] = np.nan
        tab = pa.table(
            {
                "gsm": [gsm] * n,
                "col": cols.astype(np.int64),
                "hypoxia": hyp.astype(np.float32),
                "malignancy": np.round(rng.beta(2, 2, n), 4).astype(np.float32),
                "cycling": pa.array((rng.random(n) < 0.3).tolist(), type=pa.bool_()),
                "phase": rng.integers(0, 3, n).astype(np.uint16),
            }
        ).replace_schema_metadata(
            {
                b"singlet_features_schema": b"1",
                b"feature_set": b"test",
                b"method_version": f"test-run-{k}".encode(),
            }
        )
        pq.write_table(tab, os.path.join(out, f"{gsm}.features.test.parquet"))
    return out


def second_study(root: str, synth) -> None:
    """``store/m/GSE900002.json``: the first study's manifest with other GSMs (same sample objects)."""
    from singlet.store import manifest as M

    m = copy.deepcopy(synth.manifest)
    back = {v: k for k, v in COPIES.items()}
    m["gse"] = GSE2
    for s in m["samples"]:
        s["gsm"] = back[s["gsm"]]
    meta = m["study_files"]["study_meta.json"]["json"]
    meta["gse_id"] = GSE2
    meta["gsm_meta"] = {back[g]: {**v, "gsm_id": back[g]} for g, v in meta["gsm_meta"].items()}
    p = Path(root) / "store" / "m" / f"{GSE2}.json"
    p.write_bytes(M.dumps_manifest(m))


def build(root: str):
    """-> (synth, build result, index) with everything written under ``root`` (server layout)."""
    import _store_synth as synth_mod
    from singlet.store import atlas

    s = synth_mod.Synth(root).build(n_cols=600, n_called=200, chunk_cols=64)
    second_study(root, s)
    gsms = sorted(TISSUE)
    labels = fake_labels(root, s, gsms)
    feats = fake_features(root, s, gsms)
    meta = os.path.join(root, "sample_meta.json")
    with open(meta, "w") as fh:
        json.dump(
            [
                {
                    "gsm": g,
                    "organism": "Homo sapiens",
                    "tissue": t,
                    "tissue_label": lab,
                    **PROPS.get(g, {}),
                }
                for g, (t, lab) in TISSUE.items()
            ],
            fh,
        )
    res = atlas.build_index(
        [synth_mod.GSE, GSE2],
        base=root,
        sample_meta=meta,
        labels_dir=labels,
        features=FEATURES,
        features_dir=feats,
        ontology=atlas.Ontology(ONTOLOGY),
        now="20261101T0000Z",
        group_rows=64,
        shard_rows=128,
    )
    res.write(root)
    idx = atlas.Index.open(base=root)
    return s, res, idx


def _cells_answer(idx, q, cursor=None):
    df, info = idx.sample_cells(
        q["where"],
        n=q.get("n", 10_000),
        seed=q.get("seed", 0),
        max_studies=q.get("max_studies", 10),
        study_rank=q.get("study_rank", "cells"),
        sampling=q.get("sampling", "uniform"),
        cursor=cursor,
    )
    return {
        "so": df["so"].tolist(),
        "col": df["col"].tolist(),
        "gsm": df["gsm"].tolist(),
        "eligible": info["eligible"],
        "eligible_max": info["eligible_max"],
        "remaining": info["remaining"],
        "selected": [r["gse"] for r in info["studies"] if r["selected"]],
        "cursor": info["cursor"],
    }


def expected(idx) -> dict:
    """Answers of QUERIES and PAGED (as the Worker returns them) and two gather plans."""
    from singlet.store import atlas

    out = {"version": idx.version, "queries": [], "gathers": [], "paged": None}
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
            ans = _cells_answer(idx, q)
        out["queries"].append(
            {
                "query": q,
                "answer": json.loads(
                    json.dumps(ans, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                ),
            }
        )
    pages = []
    cur = None
    while True:
        a = _cells_answer(idx, PAGED, cur)
        pages.append(a)
        cur = a["cursor"]
        if not cur:
            break
    out["paged"] = {"query": PAGED, "pages": pages}
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
    out["hash32"] = [
        [seed, a, b, int(atlas.hash32(seed, a, b))]
        for seed, a, b in ((0, 1, 0), (7, 1000, 123456), (2**32 - 1, 2**31, 2**32 - 1), (5, 3, 600))
    ]
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
    (dest / "store").mkdir(parents=True)
    # sample objects are left out: the Worker does not decode them, its tests fabricate bytes of the planned sizes
    keep = [k for k in res.objects] + ["store/cidx/current.json"]
    exp["sample_objects"] = {t["key"]: len(t["object"]) for t in s.truth.values()}
    for k in keep:
        src = Path(tmp).joinpath(*k.split("/"))
        p = dest.joinpath(*k.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
    (dest / ".gitattributes").write_text(
        "# content-addressed objects: keep bytes exactly as written\n* -text\n"
    )
    (dest / "expected.json").write_text(json.dumps(exp, indent=1))
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"{dest}: {len(keep)} objects, {total:,} bytes, index {idx.version}")
