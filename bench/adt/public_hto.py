# SPDX-License-Identifier: MIT
"""Capability ``hto`` for the public benchmark (results contract 1.1.0): ``compute(artifact_dir) -> capability document``.

Artifacts come from ``bench/adt/hto_public_run.py`` (Howitt et al. 2023, Zenodo 8304003: three BAL batches with 8 donors
each and a 3-line cell-line mix, vireo genotype truth). Metrics per dataset, rule ``match_or_beat`` against the best
competitor:

* ``f_score``: the paper's mean F-score over hashtags (``tp / (tp + (fp + fn) / 2)`` per hashtag, fp including genetic
  doublets and negatives called as that hashtag), exactly their ``calculate_HTO_fscore``;
* ``doublets_called_singlet``: share of genetic doublets a method calls a singlet (the error the paper calls most harmful).

Competitors: HTODemux, MULTIseqDemux (Seurat) and hashsolo (scanpy) run here on the same counts; demuxmix, GMM-Demux,
hashedDrops (default and the paper's optimised setting), deMULTIplex, BFFraw and BFFcluster from the paper's own published
calls (re-scoring them reproduces its Table 1). Tolerances: half-width of the bootstrap 95% CI (200 resamples of cells)
of Singlet minus the best competitor.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _contract():
    """bench/public/contract.py of this checkout, or the path in SINGLET_CONTRACT_PY."""
    import importlib.util

    p = os.environ.get("SINGLET_CONTRACT_PY") or str(Path(__file__).resolve().parents[1] / "public" / "contract.py")
    spec = importlib.util.spec_from_file_location("singlet_bench_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CAPABILITY = "hto"
RUN = ("htodemux", "multiseqdemux", "hashsolo")
PUBLISHED = {"demuxmix": "demuxmix", "GMMDemux": "gmm-demux", "hashedDrops": "hasheddrops-optimised",
             "hashedDrops_default": "hasheddrops", "deMULTIplex": "demultiplex", "BFF_raw": "bff-raw",
             "BFF_cluster": "bff-cluster"}
ART_URL = "https://data.singlet.bio/benchmarking/1.0.0/hto/"
REPO = "https://github.com/Singlet-Bio/singlet"
N_BOOT = 200
TITLES = {"howitt2023_bal1": "Howitt 2023 BAL batch 1 (8 donors, TotalSeq-A hashtags)",
          "howitt2023_bal2": "Howitt 2023 BAL batch 2 (8 donors, TotalSeq-A hashtags)",
          "howitt2023_bal3": "Howitt 2023 BAL batch 3 (8 donors, TotalSeq-A hashtags)",
          "howitt2023_cl": "Howitt 2023 lung cancer cell lines (3 lines, MULTI-seq lipid tags)"}


def _f(calls, truth_tag, tags):
    per = []
    for t in tags:
        tp = np.sum((calls == t) & (truth_tag == t))
        fp = np.sum((calls == t) & (truth_tag != t))
        fn = np.sum((calls != t) & (truth_tag == t))
        per.append(tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else np.nan)
    return float(np.nanmean(per))


def _dbl(calls, donor, tags):
    d = donor == "Doublet"
    return float(np.sum(d & np.isin(calls, tags)) / max(int(d.sum()), 1))


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dataset(adir: Path, ds: str):
    T = pd.read_parquet(adir / f"hto_{ds}_truth.parquet").set_index("barcode")
    hmap = json.load(open(adir / f"hto_{ds}_map.json"))["hashtag_to_donor"]
    tags = list(hmap)
    calls = {}
    for t in ("singlet",) + RUN:
        p = adir / f"hto_{ds}_{t}.parquet"
        if p.exists():
            calls[t] = pd.read_parquet(p).drop_duplicates("barcode").set_index("barcode")["call"].reindex(T.index)
    pub = adir / f"hto_{ds}_published.parquet"
    if pub.exists():
        P = pd.read_parquet(pub).drop_duplicates("barcode").set_index("barcode").reindex(T.index)
        for col, name in PUBLISHED.items():
            if col in P:
                calls[name] = P[col]
    truth_tag = T.truth_hashtag.astype(str).values
    donor = T.genetic_donor.astype(str).values
    arr = {k: v.astype(str).values for k, v in calls.items()}
    res = {k: {"f_score": _f(v, truth_tag, tags), "doublets_called_singlet": _dbl(v, donor, tags)} for k, v in arr.items()}
    tol = {}
    if "singlet" in arr:
        rng = np.random.default_rng(0)
        boots = [rng.integers(0, len(T), len(T)) for _ in range(N_BOOT)]
        for mid, fn, better in (("f_score", lambda c, i: _f(c[i], truth_tag[i], tags), max),
                                ("doublets_called_singlet", lambda c, i: _dbl(c[i], donor[i], tags), min)):
            comp = {k: v[mid] for k, v in res.items() if k != "singlet"}
            if not comp:
                continue
            best = better(comp, key=comp.get)
            d = [fn(arr["singlet"], b) - fn(arr[best], b) for b in boots]
            lo, hi = np.percentile(d, [2.5, 97.5])
            tol[mid] = (float((hi - lo) / 2), best)
    return {"n_cells": int(len(T)), "genetic_doublets": int((donor == "Doublet").sum()), "tags": len(tags),
            "res": res, "tol": tol}


def compute(artifact_dir) -> dict:
    adir = Path(artifact_dir)
    dss = sorted(Path(p).name[4:-len("_truth.parquet")] for p in glob.glob(str(adir / "hto_*_truth.parquet")))
    R = {ds: dataset(adir, ds) for ds in dss}
    resources, commit, cmds = [], None, {}
    for p in sorted(glob.glob(str(adir / "resources_hto_*.json"))):
        r = json.load(open(p))
        if r.get("returncode", 0):
            continue
        resources.append({"tool": r["tool"], "dataset": r["dataset"], "wall_s": r["wall_s"],
                          "max_rss_gb": r["max_rss_gb"], "threads": r["threads"], "cpu_h": r.get("cpu_h"),
                          "notes": "all captures of the dataset, one process; R methods include R and Seurat start-up"})
        commit = commit or r.get("singlet_commit")
        resources[-1] = {k: v for k, v in resources[-1].items() if v is not None}
        cmds.setdefault(r["tool"], r.get("command", ""))
    metrics = []
    for ds, r in R.items():
        if "singlet" not in r["res"]:
            continue
        for mid, name, direction in (("f_score", "Mean F-score of singlet assignments (Howitt et al.)", "higher"),
                                     ("doublets_called_singlet", "Genetic doublets called singlets", "lower")):
            comps = [{"tool": k, "value": round(v[mid], 6)} for k, v in r["res"].items() if k != "singlet"]
            rule = {"type": "match_or_beat", "against": "best"}
            if mid in r["tol"]:
                rule["tolerance"] = {"abs": round(r["tol"][mid][0], 4)}
                rule["rationale"] = ("half-width of the bootstrap 95% CI (200 resamples of cells) of Singlet minus the best "
                                     f"competitor ({r['tol'][mid][1]})")
            metrics.append({"id": mid, "name": name, "unit": "fraction", "category": "accuracy", "dataset": ds,
                            "direction": direction, "singlet": {"value": round(r["res"]["singlet"][mid], 6)},
                            "competitors": comps, "rule": rule})
    tools = [{"name": "singlet", "version": "1.0.0", "role": "singlet",
              "params": "include/singlet/pileup/hto_demux.h (HtoDemux): hashedDrops fold-change rule plus a per-tag mixture "
                        "doublet veto, cells only; the setting for each dataset was chosen on the other datasets "
                        "(bench/adt/hto_lodo.py), so every Singlet value here is held out"},
             {"name": "htodemux", "version": "Seurat 5.5.1", "role": "competitor",
              "params": "NormalizeData(CLR, margin 2); HTODemux(positive.quantile 0.99)"},
             {"name": "multiseqdemux", "version": "Seurat 5.5.1", "role": "competitor",
              "params": "MULTIseqDemux(autoThresh = TRUE)"},
             {"name": "hashsolo", "version": "scanpy 1.12.4", "role": "competitor",
              "params": "scanpy.external.pp.hashsolo defaults"},
             {"name": "vireo", "version": "vireoSNP 0.5.6", "role": "truth",
              "citation": "Howitt et al. 2023: genetic donors per cell from cellSNP-lite + vireo"}]
    for col, name in PUBLISHED.items():
        tools.append({"name": name, "version": "as published by Howitt et al. 2023", "role": "competitor",
                      "citation": "Howitt et al. 2023 NAR Genom Bioinform 5:lqad086 (calls in Zenodo 8304003)",
                      "params": "the paper's own run" + (" (confident.min 0.5)" if col == "hashedDrops" else "")})
    artifacts = []
    for p in sorted(glob.glob(str(adir / "hto_*"))):
        n = Path(p).name
        kind = ("truth" if n.endswith(("_truth.parquet", "_map.json")) else "singlet_output" if n.endswith("_singlet.parquet")
                else "competitor_output")
        artifacts.append({"name": n, "url": ART_URL + n, "sha256": _sha(p), "bytes": os.path.getsize(p), "kind": kind,
                          "dataset": next((d for d in dss if n.startswith(f"hto_{d}_")), None)})
    fs = {ds: R[ds]["res"]["singlet"]["f_score"] for ds in dss if "singlet" in R[ds]["res"]}
    conclusion = ("On the public Howitt et al. 2023 hashing benchmark with genotype truth, Singlet's hashtag calls reach mean "
                  "F-scores of " + ", ".join(f"{v:.3f}" for v in fs.values()) + " while calling few genetic doublets "
                  "singlets; it does not beat the best competitor on every batch (see failing metrics).") if fs else "Pending."
    methods = (
        "Data: Howitt et al. 2023 (Zenodo 8304003): paediatric bronchoalveolar lavage, 3 batches of 8 genetically distinct "
        "donors with 8 TotalSeq-A hashtags (2 captures each), and 3 lung cancer cell lines with MULTI-seq lipid tags (3 "
        "captures). Truth: vireo genetic donor per cell (cellSNP-lite on 1000 Genomes common SNPs), including genetic "
        "doublets; donor i is hashtag i. As in the paper, each method runs on each capture separately on the called cells "
        "(no empty droplets are available), and the F-score per hashtag is tp / (tp + (fp + fn) / 2) over all cells of "
        "the dataset, averaged over hashtags; the second metric is the share of genetic doublets called a singlet. "
        "HTODemux and MULTIseqDemux (Seurat 5.5.1) and hashsolo (scanpy 1.12.4) were run here at their defaults; the other "
        "competitors are the paper's own calls, and re-scoring them with this code reproduces the paper's Table 1. "
        "Singlet's caller is include/singlet/pileup/hto_demux.h, run through bench/adt/hto_demux_cli.cpp; its setting for each "
        "dataset is the one that leave-one-dataset-out selection chose on the other three (bench/adt/hto_lodo.py). Runtimes cover the four methods run here; they are drawn "
        "against the F-score (pareto figures) and are not a pass/fail gate.")
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.1.0", "capability": CAPABILITY,
        "title": "Hashtag demultiplexing (HTO)", "conclusion": conclusion, "methods": methods, "status": "draft",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "owner": "adt-modality",
        "singlet": {"version": "1.0.0", "commit": commit or "0" * 40, "branch": "claude/hto-demux"},
        "tools": tools,
        "datasets": [{"id": ds, "title": TITLES.get(ds, ds),
                      "accessions": {"url": "https://zenodo.org/records/8304003",
                                     "other": ["doi:10.5281/zenodo.8304002", "doi:10.1093/nargab/lqad086"]},
                      "truth": "vireo genetic donor per cell (Howitt et al. 2023)", "data_kind": "real", "n_cells": R[ds]["n_cells"],
                      "notes": f"{R[ds]['tags']} tags; {R[ds]['genetic_doublets']} genetic doublets"} for ds in dss],
        "metrics": metrics, "resources": resources, "artifacts": artifacts,
        "figures": [{"id": f"{mid}_by_dataset", "title": t, "kind": "grouped_bar",
                     "x": {"label": "dataset", "categories": dss}, "y": {"label": mid, "unit": "fraction"},
                     "series": [{"name": k, "tool": k, "x": dss, "y": [round(R[ds]["res"].get(k, {}).get(mid, np.nan), 6)
                                                                    for ds in dss]}
                                for k in ["singlet", *RUN, *PUBLISHED.values()]
                                if all(k in R[ds]["res"] for ds in dss)]}
                    for mid, t in (("f_score", "Mean F-score"), ("doublets_called_singlet", "Doublets called singlets"))],
        "code": [{"path": p, "role": r, "url": f"{REPO}/blob/{commit or 'main'}/{p}"} for p, r in (
            ("include/singlet/pileup/hto_demux.h", "implementation"), ("bench/adt/hto_demux_cli.cpp", "harness"),
            ("bench/adt/hto_public_run.py", "harness"), ("bench/adt/public_hto.py", "metric"),
            ("bench/adt/hto_lodo.py", "harness"))],
        "reproduce": {"module": "bench/adt/public_hto.py", "function": "compute", "laptop_minutes": 2,
                      "clipper_command": "python bench/adt/hto_public_run.py --zenodo DIR --exe hto_demux_cli --artifacts DIR"},
        "caveats": [
            "Selection: Singlet's setting for each dataset was chosen among 120 fixed settings of one caller family on the "
            "other three datasets (bench/adt/hto_lodo.py; criterion fixed in advance: mean of F minus genetic doublets called "
            "singlets), so no Singlet value is tuned on its own dataset. The paper's 'optimised' hashedDrops was tuned by its "
            "authors on these same datasets.",
            "Only these four datasets have an independent per-cell truth together with hashtag counts, so selection saw three "
            "datasets at a time. pbmc-ch, cline-ch and mkidney-ch (Xi & Li) label doublets from the hashtags themselves and are "
            "not independent truth for hashtag calls.",
            "F-score and genetic doublets called singlets trade off: the best competitor on one is rarely the best on the other "
            "(for example hashedDrops-optimised has the best F-score and calls 41-68% of genetic doublets singlets in BAL2, BAL3 "
            "and the cell lines). The Singlet setting favours few doublet calls.",
            "demuxEM is not compared: it needs RNA counts of empty droplets, which this benchmark does not include.",
            "Competitors from the paper's published calls were run by its authors with their versions and settings.",
        ],
    }
    for ds in dss:
        rs = [x for x in resources if x["dataset"] == ds]
        field = "cpu_h" if rs and all("cpu_h" in x for x in rs) else "wall_s"
        if any(m["id"] == "f_score" and m["dataset"] == ds for m in metrics) and len(rs) > 1:
            doc["figures"].append(_contract().pareto_figure(doc, "f_score", ds, field))
    complete = bool(metrics) and all(m["singlet"]["value"] is not None for m in metrics) and len(dss) == 4
    doc["status"] = "final" if complete else "draft"
    return _contract().finalize(doc)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    doc = compute(a.artifact_dir)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({m["id"] + "@" + m["dataset"]: m.get("pass") for m in doc["metrics"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
