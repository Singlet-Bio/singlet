#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training corpus for the cell type network: a stratified sample of CZ CELLxGENE Census cells with author labels.

Census (CC BY 4.0) holds author-annotated cells whose labels curators mapped to Cell Ontology ids. The sample is drawn from
primary data (``is_primary_data``) of the droplet assays Singlet processes, human and mouse, stratified by (cell type, tissue):
each stratum gets at most ``--per-stratum`` cells, taken round-robin across datasets so no single study dominates a type, and no
dataset contributes more than ``--per-dataset`` cells. Every collection linked to a benchmark study
(``--exclude-studies``, the JSON ``annotate_bench_prepare.py`` reads) is left out, so the benchmark stays held out. Whole
collections are assigned to the validation split (``--val-frac``), never single cells.

  python scripts/celltype_nn/corpus.py plan  --out $D --exclude-studies bench_studies.json [--census-version stable]
  python scripts/celltype_nn/corpus.py fetch --out $D --chunk $SLURM_ARRAY_TASK_ID     # one h5ad of raw counts per chunk
  python scripts/celltype_nn/corpus.py orthologs --out $D                               # Ensembl human-mouse 1:1 orthologs

Writes ``<out>/plan.parquet`` (one row per sampled cell: soma_joinid, organism, chunk, dataset/collection, donor, assay,
suspension, tissue, cell type, split), ``<out>/plan.json`` (Census release, filters, counts) and ``<out>/chunks/<k>.h5ad``
(all genes, raw counts, var = feature_id / feature_name).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ORGANISMS = {"homo_sapiens": "human", "mus_musculus": "mouse"}
# droplet and plate-free assays that Singlet's pipeline processes (10x 3'/5', multiome RNA, Drop-seq, Seq-Well, BD Rhapsody WTA)
ASSAY_OK = (
    "10x 3' v1",
    "10x 3' v2",
    "10x 3' v3",
    "10x 3' v4",
    "10x 3' transcription profiling",
    "10x 5' v1",
    "10x 5' v2",
    "10x 5' v3",
    "10x 5' transcription profiling",
    "10x multiome",
    "10x gene expression flex",
    "Drop-seq",
    "Seq-Well",
    "Seq-Well S3",
    "BD Rhapsody Whole Transcriptome Analysis",
)
# labels that carry no cell type
UNINFORMATIVE = {"unknown", "CL:0000000", "CL:0000003", "CL:0000548", "CL:0000255"}
OBS_COLS = [
    "soma_joinid",
    "dataset_id",
    "assay",
    "suspension_type",
    "donor_id",
    "tissue_general",
    "tissue_ontology_term_id",
    "cell_type_ontology_term_id",
    "cell_type",
    "disease_ontology_term_id",
    "development_stage_ontology_term_id",
    "raw_sum",
    "nnz",
]


def _excluded_collections(path: Path | None) -> set:
    if path is None:
        return set()
    return {s["collection_id"] for s in json.loads(path.read_text())}


def stratified_sample(obs, per_stratum: int, per_dataset: int, seed: int):
    """Indices into ``obs``: per (cell type, tissue) at most ``per_stratum`` cells, round-robin across datasets."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ct": obs["cell_type_ontology_term_id"].to_numpy(),
            "ti": obs["tissue_general"].to_numpy(),
            "ds": obs["dataset_id"].to_numpy(),
            "u": rng.random(len(obs)),
        }
    )
    # rank within (stratum, dataset) in random order; taking rank 0 of every dataset, then rank 1, ... is the round robin
    df["r"] = df.groupby(["ct", "ti", "ds"], sort=False)["u"].rank(method="first") - 1
    df = df.sort_values(["r", "u"])
    df["k"] = df.groupby(["ct", "ti"], sort=False).cumcount()
    keep = df[df["k"] < per_stratum]
    keep = keep.sort_values("u")
    keep = keep[keep.groupby("ds", sort=False).cumcount() < per_dataset]
    return np.sort(keep.index.to_numpy())


def cmd_plan(a) -> int:
    import cellxgene_census
    import pandas as pd

    t0 = time.time()
    excl = _excluded_collections(a.exclude_studies)
    d = cellxgene_census.get_census_version_directory()
    version = str(d[a.census_version].get("release_build", a.census_version)) if a.census_version in d else a.census_version
    with cellxgene_census.open_soma(census_version=version) as census:
        info = census["census_info"]["summary"].read().concat().to_pandas()
        release = dict(zip(info["label"], info["value"]))
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        ds_coll = dict(zip(ds["dataset_id"], ds["collection_id"]))
        ds_title = dict(zip(ds["dataset_id"], ds["dataset_title"]))
        parts, counts = [], {}
        for org, sp in ORGANISMS.items():
            if sp not in a.organisms.split(","):
                continue
            obs = (
                census["census_data"][org]
                .obs.read(value_filter="is_primary_data == True", column_names=OBS_COLS)
                .concat()
                .to_pandas()
            )
            n0 = len(obs)
            for c in obs.columns:
                if isinstance(obs[c].dtype, pd.CategoricalDtype):
                    obs[c] = obs[c].astype(str)
            obs = obs[obs["assay"].isin(ASSAY_OK)]
            n1 = len(obs)
            obs = obs[~obs["cell_type_ontology_term_id"].isin(UNINFORMATIVE)]
            obs = obs[obs["nnz"] >= a.min_genes]
            obs["collection_id"] = obs["dataset_id"].map(ds_coll)
            n_bench = int(obs["collection_id"].isin(excl).sum())
            obs = obs[~obs["collection_id"].isin(excl)].reset_index(drop=True)
            idx = stratified_sample(
                obs,
                a.per_stratum if sp == "human" else a.per_stratum_mouse,
                a.per_dataset,
                a.seed,
            )
            sub = obs.iloc[idx].copy()
            sub["organism"] = sp
            parts.append(sub)
            counts[sp] = dict(
                primary=n0,
                assay_ok=n1,
                bench_excluded=n_bench,
                eligible=len(obs),
                sampled=len(sub),
                datasets=int(sub["dataset_id"].nunique()),
                collections=int(sub["collection_id"].nunique()),
                cell_types=int(sub["cell_type_ontology_term_id"].nunique()),
                tissues=int(sub["tissue_general"].nunique()),
            )
            print(sp, counts[sp], f"{time.time() - t0:.0f}s", flush=True)
            del obs
    plan = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(a.seed + 1)
    colls = np.array(sorted(plan["collection_id"].unique()))
    val = set(colls[rng.random(len(colls)) < a.val_frac])
    plan["split"] = np.where(plan["collection_id"].isin(val), "val", "train")
    plan["dataset_title"] = plan["dataset_id"].map(ds_title)
    # chunks: one organism per chunk, cells in soma_joinid order (Census reads are faster on sorted coordinates)
    plan = plan.sort_values(["organism", "soma_joinid"]).reset_index(drop=True)
    plan["chunk"] = -1
    k = 0
    for sp, g in plan.groupby("organism", sort=True):
        n = int(np.ceil(len(g) / a.chunk_size))
        plan.loc[g.index, "chunk"] = k + np.arange(len(g)) * n // len(g)
        k += n
    a.out.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(a.out / "plan.parquet", index=False)
    summary = dict(
        census_version=version,
        census_alias=a.census_version,
        census_release=release,
        filters=dict(
            is_primary_data=True,
            assays=list(ASSAY_OK),
            min_genes=a.min_genes,
            uninformative_labels=sorted(UNINFORMATIVE),
            per_stratum=a.per_stratum,
            per_stratum_mouse=a.per_stratum_mouse,
            per_dataset=a.per_dataset,
            seed=a.seed,
        ),
        excluded_collections=sorted(excl),
        val_collections=sorted(val),
        counts=counts,
        n_cells=len(plan),
        n_chunks=int(plan["chunk"].max() + 1),
        split=plan["split"].value_counts().to_dict(),
        licence="CZ CELLxGENE Census data: CC BY 4.0 (per-dataset citations in census_info/datasets)",
    )
    (a.out / "plan.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["counts"]), summary["n_chunks"], "chunks", flush=True)
    return 0


def cmd_fetch(a) -> int:
    import cellxgene_census
    import pandas as pd

    plan = pd.read_parquet(a.out / "plan.parquet")
    meta = json.loads((a.out / "plan.json").read_text())
    sub = plan[plan["chunk"] == a.chunk]
    sp = sub["organism"].iloc[0]
    org = {v: k for k, v in ORGANISMS.items()}[sp]
    dst = a.out / "chunks" / f"{a.chunk:04d}.h5ad"
    if dst.exists():
        print(f"{dst} exists", flush=True)
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    version = meta["census_version"]
    with cellxgene_census.open_soma(census_version=version) as census:
        ad = cellxgene_census.get_anndata(
            census,
            organism=org.replace("_", " ").capitalize(),
            obs_coords=sub["soma_joinid"].to_numpy(),
            column_names={"obs": ["soma_joinid"], "var": ["feature_id", "feature_name"]},
        )
    ad.obs = ad.obs.reset_index(drop=True)
    pos = pd.Series(np.arange(len(ad)), index=ad.obs["soma_joinid"].to_numpy())
    ad = ad[pos.loc[sub["soma_joinid"].to_numpy()].to_numpy()].copy()
    ad.obs.index = ad.obs["soma_joinid"].astype(str).to_numpy()
    ad.var.index = ad.var["feature_id"].astype(str).to_numpy()
    ad.uns["census_version"] = str(version)
    tmp = dst.with_suffix(".part.h5ad")
    ad.write_h5ad(tmp, compression="gzip")
    tmp.replace(dst)
    print(f"chunk {a.chunk}: {ad.n_obs} cells x {ad.n_vars} genes, {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_leakcheck(a) -> int:
    """For every dataset of the excluded (benchmark) collections: how many of its cells are Census primary data. Census marks
    exactly one copy of a cell primary, so a dataset that is fully primary has no copy of its cells in the training corpus
    (which takes primary cells only); a non-primary dataset would mean its cells' primary copy lives in another collection."""
    import cellxgene_census

    excl = _excluded_collections(a.exclude_studies)
    meta = json.loads((a.out / "plan.json").read_text())
    rows = []
    with cellxgene_census.open_soma(census_version=meta["census_version"]) as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        ds = ds[ds["collection_id"].isin(excl)]
        for org in ORGANISMS:
            for d in ds["dataset_id"]:
                o = (
                    census["census_data"][org]
                    .obs.read(value_filter=f"dataset_id == '{d}'", column_names=["is_primary_data"])
                    .concat()
                    .to_pandas()
                )
                if len(o):
                    rows.append(dict(dataset_id=d, organism=ORGANISMS[org], cells=len(o), primary=int(o["is_primary_data"].sum())))
    (a.out / "leakcheck.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r, flush=True)
    print("datasets not fully primary:", [r["dataset_id"] for r in rows if r["primary"] < r["cells"]], flush=True)
    return 0


def cmd_dedup(a) -> int:
    """Find copies of benchmark cells in the corpus and list them for exclusion.

    Census has no barcodes, but a copy of a cell keeps its raw counts, so (raw_sum, nnz) identifies it. For each corpus dataset,
    the fraction of its sampled cells whose (organism, raw_sum, nnz) occurs among the cells of the benchmark collections is
    compared with chance: a dataset above ``--min-frac`` holds copies, and its matching cells are written to
    ``leak_exclude.parquet`` (prepare.py marks them ``split = 'excluded'``)."""
    import cellxgene_census
    import pandas as pd

    excl = _excluded_collections(a.exclude_studies)
    meta = json.loads((a.out / "plan.json").read_text())
    plan = pd.read_parquet(a.out / "plan.parquet", columns=["organism", "soma_joinid", "dataset_id", "collection_id", "raw_sum", "nnz"])
    keys = []
    with cellxgene_census.open_soma(census_version=meta["census_version"]) as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        bench_ds = ds.loc[ds["collection_id"].isin(excl), "dataset_id"].tolist()
        flt = "dataset_id in [" + ",".join(f"'{d}'" for d in bench_ds) + "]"
        for org, sp in ORGANISMS.items():
            o = census["census_data"][org].obs.read(value_filter=flt, column_names=["raw_sum", "nnz"]).concat().to_pandas()
            o["organism"] = sp
            keys.append(o)
    K = pd.concat(keys, ignore_index=True)
    kset = set(zip(K["organism"], K["raw_sum"].round(3), K["nnz"]))
    hit = np.array([(o, round(r, 3), n) in kset for o, r, n in zip(plan["organism"], plan["raw_sum"], plan["nnz"])])
    plan["hit"] = hit
    per = plan.groupby("dataset_id").agg(n=("hit", "size"), hits=("hit", "sum"))
    per["frac"] = per["hits"] / per["n"]
    base = float(np.median(per["frac"]))
    flagged = per[(per["frac"] >= a.min_frac) & (per["hits"] >= 20)]
    drop = plan[plan["dataset_id"].isin(flagged.index) & plan["hit"]]
    drop[["organism", "soma_joinid", "dataset_id"]].to_parquet(a.out / "leak_exclude.parquet", index=False)
    top = per.sort_values("frac", ascending=False).head(15).reset_index()
    titles = dict(zip(ds["dataset_id"], ds["dataset_title"]))
    top["title"] = top["dataset_id"].map(titles)
    rep = dict(bench_cells=len(K), median_dataset_match_frac=base, top=top.to_dict("records"), flagged=flagged.sort_values("frac", ascending=False).reset_index().to_dict("records"),
               excluded_cells=len(drop))
    (a.out / "leak_dedup.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)
    return 0


BIOMART = "https://www.ensembl.org/biomart/martservice"
_Q = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query virtualSchemaName="default" formatter="TSV" header="1" '
    'uniqueRows="1"><Dataset name="hsapiens_gene_ensembl" interface="default"><Attribute name="ensembl_gene_id"/>'
    '<Attribute name="external_gene_name"/><Attribute name="mmusculus_homolog_ensembl_gene"/>'
    '<Attribute name="mmusculus_homolog_associated_gene_name"/><Attribute name="mmusculus_homolog_orthology_type"/>'
    "</Dataset></Query>"
)


def cmd_orthologs(a) -> int:
    """Human-mouse one-to-one orthologs from Ensembl BioMart (Ensembl data: no restrictions on use)."""
    import io
    import urllib.parse
    import urllib.request

    import pandas as pd

    url = BIOMART + "?query=" + urllib.parse.quote(_Q)
    req = urllib.request.Request(url, headers={"User-Agent": "singlet-celltype-nn/0.1 (+https://singlet.bio)"})
    with urllib.request.urlopen(req, timeout=600) as r:
        txt = r.read().decode()
    df = pd.read_csv(io.StringIO(txt), sep="\t")
    df.columns = ["human_id", "human_symbol", "mouse_id", "mouse_symbol", "type"]
    one = df[df["type"] == "ortholog_one2one"].drop_duplicates(["human_id", "mouse_id"])
    one = one[~one["human_id"].duplicated(keep=False) & ~one["mouse_id"].duplicated(keep=False)]
    a.out.mkdir(parents=True, exist_ok=True)
    one[["human_id", "human_symbol", "mouse_id", "mouse_symbol"]].to_csv(
        a.out / "orthologs_human_mouse.tsv.gz", sep="\t", index=False
    )
    print(f"{len(one)} one-to-one orthologs (of {len(df)} rows)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-studies", type=Path)
    p.add_argument("--census-version", default="stable")
    p.add_argument("--organisms", default="human,mouse")
    p.add_argument("--per-stratum", type=int, default=1500)
    p.add_argument("--per-stratum-mouse", type=int, default=1500)
    p.add_argument("--per-dataset", type=int, default=20000)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--chunk-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--chunk", type=int, required=True)
    o = sub.add_parser("orthologs")
    o.add_argument("--out", type=Path, required=True)
    lc = sub.add_parser("leakcheck")
    lc.add_argument("--out", type=Path, required=True)
    lc.add_argument("--exclude-studies", type=Path, required=True)
    dd = sub.add_parser("dedup")
    dd.add_argument("--out", type=Path, required=True)
    dd.add_argument("--exclude-studies", type=Path, required=True)
    dd.add_argument("--min-frac", type=float, default=0.2)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "fetch": cmd_fetch, "orthologs": cmd_orthologs, "leakcheck": cmd_leakcheck, "dedup": cmd_dedup}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
