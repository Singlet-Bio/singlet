#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build per-cell "truth" cell-type labels for the samples of one Singlet study from CZ CELLxGENE.

CELLxGENE (CxG) datasets pool many samples and carry curated Cell Ontology ids (``cell_type_ontology_term_id``). This script
decides which CxG cells belong to which Singlet sample (GSM) by comparing barcode sets, then writes one truth table per GSM.

  # obs already saved as Parquet (several datasets of one collection may be given)
  python scripts/annotate_truth_cxg.py --gse GSE130973 --bundle bundles/GSE130973.singlet \
      --obs cxg/124744b8.obs.parquet --out truth/GSE130973
  # straight from the CxG h5ad (only obs is read; the h5ad can be deleted afterwards; --save-obs keeps the obs as Parquet)
  python scripts/annotate_truth_cxg.py --gse GSE130973 --bundle bundles/GSE130973.singlet \
      --h5ad dl/124744b8.h5ad --save-obs cxg/124744b8.obs.parquet --out truth/GSE130973

Matching rules (all thresholds are options):
  1. Barcodes on both sides are reduced to the core nucleotide sequence: the single run of ``--core-len`` (16) A/C/G/T
     (10x v2/v3). The remainder (sample prefix, ``-1`` suffix, ...) is kept as the "tag". Barcodes without exactly one such
     run are unusable. Singlet barcodes are the CALLED cells of ``SingletBundle.raw_counts(gsm).obs_names``.
  2. CxG cells are grouped by a sample-like key. Candidate keys are every low-cardinality obs column, the barcode tag, and
     each column crossed with the tag and/or the CxG dataset. For each candidate, each group (>= ``--min-group`` cells) is assigned to the GSM that
     contains >= ``--min-best`` (0.5) of the group's cores while the runner-up GSM contains < ``--max-second`` (0.1); a weaker
     owner (>= 20% overlap, runner-up < 5%) is accepted too, because Singlet may call fewer cells than the authors kept (chance
     overlap is under 1%). Columns whose names look like cell types/clusters are never used as keys (cell calling would bias
     the assignment). The candidate that places the most CxG cells inside their GSM's called cells wins (ties: the purest, i.e. finest grouping).
  3. Within an assigned group a CxG cell is matched when its core occurs in the GSM's called cells. Cores that are not unique
     on either side (within the GSM / within the group-to-GSM assignment) are dropped, as are cells with an unknown cell type.
     When several CxG datasets label the same cell, the first ``--obs``/``--h5ad`` given wins and conflicts are counted.
  4. GSMs with fewer than ``--min-matched`` (200) matched cells, or whose matched cells are < 90% from one ``donor_id``
     (mixed donors; the few cells of other donors are dropped) or with fewer than ``--min-types`` (3) distinct cell types
     (the CxG dataset is a one-lineage subset) are not written.

Outputs ``<out>/<GSM>.truth.parquet`` (gsm, barcode, cl_id, cl_label, author_cell_type, cxg_dataset_id) and
``<out>/<GSE>.match.json`` (chosen grouping, group->GSM table, per-GSM statistics incl. donor/sex composition of the assigned
CxG cells for sanity checks against GEO characteristics). Only reads data; nothing is annotated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def read_obs(h5ad: Path) -> pd.DataFrame:
    """Read only ``obs`` of an h5ad (the matrix is never loaded)."""
    import h5py

    with h5py.File(h5ad, "r") as f:
        try:
            from anndata.io import read_elem
        except ImportError:  # older anndata
            from anndata.experimental import read_elem
        obs = read_elem(f["obs"])
    for c in obs.columns:
        if isinstance(obs[c].dtype, pd.CategoricalDtype):
            obs[c] = obs[c].astype(object)
    obs.index = obs.index.astype(str)
    return obs


def split_barcode(bc: str, core_len: int):
    """(core, tag) of a barcode, or (None, None) when it does not hold exactly one run of ``core_len`` ACGT."""
    runs = list(re.finditer(rf"(?<![ACGTacgt])[ACGTacgt]{{{core_len}}}(?![ACGTacgt])", bc))
    if len(runs) != 1:
        return None, None
    m = runs[0]
    return m.group(0).upper(), (bc[: m.start()] + "|" + bc[m.end() :])


def singlet_cores(bundle_path: Path, core_len: int):
    """gsm -> DataFrame(core, barcode) over called cells; plus the sample metadata of the bundle."""
    from singlet.bundle import SingletBundle

    b = SingletBundle(bundle_path)
    out, meta = {}, b.study_meta.get("gsm_meta", {})
    for gsm in b.gsm_ids:
        try:
            names = [str(x) for x in b.raw_counts(gsm).obs_names]
        except Exception as e:  # sample without a readable matrix
            print(f"  {gsm}: no counts ({e})", file=sys.stderr)
            continue
        parts = [split_barcode(n, core_len) for n in names]
        df = pd.DataFrame({"barcode": names, "core": [p[0] for p in parts]})
        out[gsm] = df
    return out, meta


def assign_groups(
    cxg: pd.DataFrame,
    key: pd.Series,
    gsm_sets: dict,
    min_group,
    min_best,
    max_second,
    weak_best=0.2,
):
    """Group -> (gsm, best fraction, second fraction, size) for groups with a clear owner."""
    res = {}
    for g, idx in cxg.groupby(key, sort=False).groups.items():
        cores = set(cxg.loc[idx, "core"])
        if len(cores) < min_group:
            continue
        fr = sorted(
            ((len(cores & s) / len(cores), gsm) for gsm, s in gsm_sets.items()), reverse=True
        )
        best = fr[0]
        second = fr[1][0] if len(fr) > 1 else 0.0
        # strict rule, or (Singlet called far fewer cells than the authors kept) a weaker overlap that is still
        # far beyond chance (random 16-nt overlap is ~ n_cells / whitelist size, i.e. well under 1%)
        if second < max_second and (
            best[0] >= min_best or (best[0] >= weak_best and second < max_second / 2)
        ):
            res[g] = (best[1], best[0], second, len(cores))
    return res


def candidate_keys(cxg: pd.DataFrame, max_card: int):
    """Sample-like grouping keys: low-cardinality obs columns, the barcode tag, and column x tag."""
    skip = {"cell_type", "core", "tag", "dataset_id", "organism"}
    # never group by anything that can correlate with cell identity: cell calling would then bias the assignment
    celllike = re.compile(
        r"cell.?type|cluster|annot|lineage|compartment|leiden|louvain|phase|label|snn|score|subset|major|minor|class|identity|state|description|ontology",
        re.I,
    )
    ds = (
        cxg["dataset_id"].astype(str) + "§" if "dataset_id" in cxg else ""
    )  # datasets can differ in organism/chemistry
    keys = {"tag": cxg["tag"], "dataset+tag": ds + cxg["tag"]}
    for c in cxg.columns:
        if c in skip or celllike.search(c):
            continue
        s = cxg[c]
        if s.dtype == object or s.dtype.kind in "iub":
            if 1 < s.nunique() <= max_card:
                keys[c] = s.astype(str)
                keys[f"{c}+tag"] = s.astype(str) + "§" + cxg["tag"]
                keys[f"dataset+{c}+tag"] = ds + s.astype(str) + "§" + cxg["tag"]
    return keys


def build_truth(
    gse,
    bundle,
    cxg,
    min_group=50,
    min_best=0.5,
    max_second=0.1,
    min_matched=200,
    core_len=16,
    max_card=400,
    min_donor_purity=0.9,
    min_types=3,
):
    """Returns (truth tables by gsm, report dict). ``cxg`` has obs columns plus dataset_id and the original barcode as index."""
    sing, meta = singlet_cores(bundle, core_len)
    gsm_sets = {g: set(d["core"].dropna()) for g, d in sing.items()}
    parts = [split_barcode(n, core_len) for n in cxg.index]
    cxg = cxg.copy()
    cxg["core"] = [p[0] for p in parts]
    cxg["tag"] = [p[1] for p in parts]
    n_all = len(cxg)
    cxg = cxg[cxg["core"].notna()].copy()
    report = {"gse": gse, "n_cxg_cells": n_all, "n_cxg_parsable": len(cxg), "core_len": core_len}
    if cxg.empty:
        report["error"] = "no CxG barcode holds a single %d-nt A/C/G/T run" % core_len
        return {}, report
    best = None
    for name, key in candidate_keys(cxg, max_card).items():
        a = assign_groups(cxg, key, gsm_sets, min_group, min_best, max_second)
        owner = key.map({g: v[0] for g, v in a.items()})
        assigned = int(owner.notna().sum())
        hit = sum(
            int(cxg.loc[owner == gsm, "core"].isin(gsm_sets[gsm]).sum())
            for gsm in set(a_v[0] for a_v in a.values())
        )
        score = (
            hit,
            hit / max(assigned, 1),
        )  # most cells found in their GSM; then the purest grouping
        if best is None or score > best[0]:
            best = (score, name, assigned, a, key)
    _, name, size, groups, key = best
    report.update(grouping=name, n_cxg_assigned=size, n_groups=len(groups))
    cxg["group"] = key
    cxg["gsm"] = cxg["group"].map({g: v[0] for g, v in groups.items()})
    report["groups"] = {
        str(g): {"gsm": v[0], "best": round(v[1], 3), "second": round(v[2], 3), "n": v[3]}
        for g, v in groups.items()
    }
    truth, stats = {}, {}
    for gsm, sdf in sing.items():
        ass = cxg[cxg["gsm"] == gsm]
        st = {
            "n_singlet_called": len(sdf),
            "n_singlet_parsable": int(sdf["core"].notna().sum()),
            "n_cxg_assigned": len(ass),
        }
        sdf = sdf[sdf["core"].notna() & ~sdf["core"].duplicated(keep=False)]
        ass = ass[~ass["core"].duplicated(keep="first")]  # first dataset given wins
        st["n_cxg_assigned_unique"] = len(ass)
        m = ass.merge(sdf, on="core", how="inner", suffixes=("", "_s"))
        st["n_matched"] = len(m)
        bad = m["cell_type_ontology_term_id"].isna() | m["cell_type_ontology_term_id"].isin(
            ["unknown", ""]
        )
        st["n_unknown_type"] = int(bad.sum())
        m = m[~bad]
        st["n_truth"] = len(m)
        st["frac_singlet_with_truth"] = round(len(m) / max(len(sdf), 1), 4)
        st["frac_cxg_in_singlet"] = round(st["n_matched"] / max(len(ass), 1), 4)
        for col, k in (
            ("donor_id", "donors"),
            ("sex", "sex"),
        ):  # composition of the MATCHED cells (the ones that get labels)
            if col in m and len(m):
                st[k] = m[col].astype(str).value_counts().head(5).to_dict()
        if len(m) and "dataset_id" in m:
            st["datasets"] = sorted(m["dataset_id"].unique())
        # a GSM is one library from one donor: a second donor among its matched cells means the grouping mixed samples
        # (a few such cells are chance barcode collisions: drop them; many mean the sample assignment is not trustworthy)
        purity = 1.0
        if "donor_id" in m and len(m):
            donor = m["donor_id"].astype(str)
            purity = float(donor.value_counts(normalize=True).iloc[0])
            st["n_dropped_other_donor"] = int((donor != donor.value_counts().index[0]).sum())
            m = m[donor == donor.value_counts().index[0]]
        st["donor_purity"] = round(purity, 4)
        st["n_truth"] = len(m)
        st["n_cl_types"] = int(m["cell_type_ontology_term_id"].nunique())
        if "organism" in m and len(m):
            st["organism_cxg"] = sorted(m["organism"].astype(str).unique())
        st["organism_geo"] = str(meta.get(gsm, {}).get("organism", ""))
        mt = meta.get(gsm, {})
        st["geo"] = {
            "title": str(mt.get("title", ""))[:80],
            "characteristics": str(mt.get("sample_characteristics", ""))[:300],
        }
        st["kept"] = bool(
            len(m) >= min_matched and purity >= min_donor_purity and st["n_cl_types"] >= min_types
        )
        stats[gsm] = st
        if st["kept"]:
            truth[gsm] = pd.DataFrame(
                {
                    "gsm": gsm,
                    "barcode": m["barcode"].to_numpy(),
                    "cl_id": m["cell_type_ontology_term_id"].to_numpy(),
                    "cl_label": m["cell_type"].to_numpy(),
                    "author_cell_type": m["cell_type"].to_numpy(),
                    "cxg_dataset_id": m["dataset_id"].to_numpy(),
                }
            )
    report["gsm"] = stats
    return truth, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gse", required=True)
    ap.add_argument("--bundle", type=Path, required=True, help="legacy .singlet bundle on disk")
    ap.add_argument(
        "--obs",
        type=Path,
        action="append",
        default=[],
        help="CxG obs Parquet; file stem = dataset id (repeatable)",
    )
    ap.add_argument(
        "--h5ad",
        type=Path,
        action="append",
        default=[],
        help="CxG h5ad; only obs is read (repeatable)",
    )
    ap.add_argument("--save-obs", type=Path, help="with one --h5ad: write its obs to this Parquet")
    ap.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="dataset ids in the order of --obs then --h5ad (default: file stems)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--core-len", type=int, default=16)
    ap.add_argument("--min-group", type=int, default=50)
    ap.add_argument("--min-best", type=float, default=0.5)
    ap.add_argument("--max-second", type=float, default=0.1)
    ap.add_argument("--min-matched", type=int, default=200)
    ap.add_argument(
        "--min-types",
        type=int,
        default=3,
        help="drop a GSM whose truth has fewer cell types (a CxG lineage subset)",
    )
    ap.add_argument(
        "--min-donor-purity",
        type=float,
        default=0.9,
        help="drop a GSM whose matched cells come from several donors",
    )
    a = ap.parse_args(argv)

    frames = []
    srcs = [(p, False) for p in a.obs] + [(p, True) for p in a.h5ad]
    for i, (p, is_h5) in enumerate(srcs):
        obs = read_obs(p) if is_h5 else pd.read_parquet(p)
        if is_h5 and a.save_obs:
            obs.to_parquet(a.save_obs)
        obs.index = obs.index.astype(str)
        did = a.dataset_id[i] if i < len(a.dataset_id) else p.stem.split(".")[0]
        obs["dataset_id"] = did
        frames.append(obs)
    cxg = pd.concat(frames)
    truth, rep = build_truth(
        a.gse,
        a.bundle,
        cxg,
        a.min_group,
        a.min_best,
        a.max_second,
        a.min_matched,
        a.core_len,
        min_donor_purity=a.min_donor_purity,
        min_types=a.min_types,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob(
        "*.truth.parquet"
    ):  # no stale GSMs from an earlier run with other thresholds
        old.unlink()
    for gsm, df in truth.items():
        df.to_parquet(a.out / f"{gsm}.truth.parquet", index=False)
    (a.out / f"{a.gse}.match.json").write_text(json.dumps(rep, indent=1, default=str))
    print(
        f"{a.gse}: grouping={rep.get('grouping')} assigned={rep.get('n_cxg_assigned')}/{rep['n_cxg_cells']} kept GSMs={len(truth)}"
    )
    for gsm, st in rep.get("gsm", {}).items():
        print(
            f"  {gsm} called={st['n_singlet_called']} assigned={st['n_cxg_assigned']} matched={st['n_matched']} "
            f"truth={st['n_truth']} singlet_w_truth={st['frac_singlet_with_truth']} cxg_in_singlet={st['frac_cxg_in_singlet']} kept={st['kept']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
