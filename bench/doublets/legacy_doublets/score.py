# SPDX-License-Identifier: MIT
"""Score one sample: the registered feature ``qc/doublet_score`` as a sidecar table and its ``sample_cell_summary`` rows.

The production score comes from the engine's ``detect_doublets`` (``dbl_run`` harness, ``prod/modality-fixes`` doublet_detect.h:
the public store withholds the doublet columns, so the job recomputes it from the stored counts). Samples whose genotype
demultiplexing found two or more donors get no score: their doublet call is ``donor_assignments.tsv prob_doublet``.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps

from singlet.features import registry as REG

from . import __version__
from .combiner import Combiner, basic_features

MIN_CELLS = 100
FEATURE_SET = "qc"
FEATURE = REG.Feature(
    name="doublet_score",
    feature_set=FEATURE_SET,
    label="Expression doublet score",
    description=(
        "Probability-like score that the droplet holds two cells, from a cross-study gradient-boosted combiner over the engine's "
        "doublet score, library size and four features of the atlas cell-type network, each ranked within the sample. It orders "
        "the cells of one sample; it is not a calibrated doublet probability (the share of cells above 0.5 depends on the "
        "sample's doublet rate). Not computed for samples with two or more genetic donors, whose doublet call is the genotype "
        "one."
    ),
    method="gbm-prod-atlas",
    source_ref="Singlet doublet combiner (docs/doublet_score.md); atlas network singlet.celltype_nn",
    licence="CC0-1.0",
    organisms=("Homo sapiens", "Mus musculus"),
    thresholds=(0.5, 0.8),
)


def method_version(combiner: Combiner, celltype_version: str, dbl_commit: str) -> str:
    return (
        f"singlet-doublets={__version__};combiner={combiner.version};celltype-nn={celltype_version};"
        f"dbl-run={dbl_commit};min_cells={MIN_CELLS}"
    )


def registry_rows(version: str) -> list:
    f = replace(FEATURE, method_version=version)
    return [f.cell_feature_row()]


def prod_score(X: sps.csr_matrix, exe: str, threads: int = 4) -> np.ndarray:
    """The engine's doublet score for called cells x genes counts: ``dbl_run`` on the genes x cells CSC (uint16)."""
    G = sps.csc_matrix(X.T)
    with tempfile.TemporaryDirectory() as tmp:
        pre = f"{tmp}/m"
        np.array([G.shape[0], G.shape[1], G.nnz], np.uint32).tofile(pre + ".hdr")
        G.indptr.astype(np.int32).tofile(pre + ".indptr")
        G.indices.astype(np.int32).tofile(pre + ".indices")
        np.minimum(G.data, 65535).astype(np.uint16).tofile(pre + ".data")
        r = subprocess.run(
            [exe, pre, f"{tmp}/out.tsv", f"threads={threads}", "nomap=1"],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(f"dbl_run exit {r.returncode}: {r.stderr[-400:]}")
        import pandas as pd

        o = pd.read_csv(f"{tmp}/out.tsv", sep="\t")
    return o.score.to_numpy(np.float64)


@dataclass
class Result:
    score: np.ndarray
    status: str = "ok"
    timings: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, np.ndarray] = field(default_factory=dict)


def n_donors(sample) -> int:
    d = getattr(sample, "donor", None)
    return 0 if d is None else len({str(x) for x in d if str(x)})


def score_sample(
    sample,
    combiner: Combiner,
    *,
    dbl_run: str,
    predictor,
    centroids,
    threads: int = 4,
    device: Optional[str] = None,
    keep_features: bool = False,
    score_pools: bool = False,
) -> Result:
    from .atlas import atlas_features

    X = sps.csr_matrix(sample.X, dtype=np.float32)
    n = X.shape[0]
    nan = np.full(n, np.nan, np.float32)
    if sample.species not in ("human", "mouse"):
        return Result(nan, "other_organism")
    if n < MIN_CELLS:
        return Result(nan, "too_few_cells")
    if n_donors(sample) >= 2 and not score_pools:
        return Result(nan, "genotype_pool")
    tim: Dict[str, float] = {}
    t0 = time.perf_counter()
    F = basic_features(X)
    F["prod"] = prod_score(X, dbl_run, threads)
    tim["prod_s"] = time.perf_counter() - t0
    F.update(
        atlas_features(
            X,
            sample.gene_ids or sample.gene_names,
            sample.species,
            predictor,
            centroids,
            device=device,
            timings=tim,
        )
    )
    s = combiner.score(F)
    tim["total_s"] = time.perf_counter() - t0
    return Result(s, "ok", tim, F if keep_features else {})


def summaries(sample, res: Result, version: str) -> list:
    f = replace(FEATURE, method_version=version)
    return REG.summary_rows(sample.gsm, sample.gse, {f.name: res.score}, [f], version)
