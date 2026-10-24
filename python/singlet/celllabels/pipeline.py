# SPDX-License-Identifier: MIT
"""One sample in, one label table (and a small JSON summary) out."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from . import ancestry as ANC
from . import celltype as CT
from . import guide as G
from . import output as O
from . import sex as SX
from .io import SampleInput

ANNOTATE_VERSION = "0.1.0"


def model_version(method: str = "guide") -> str:
    return (
        f"singlet-annotate={ANNOTATE_VERSION};celltype={method}/cellguide-v2/{G.cl_version().split('/')[2]};"
        f"sex=xist-chrY-split/2;ancestry=1kgp-supervised-admixture/1/{ANC.HANCESTRO_VERSION.split('/')[-1]}"
    )


def annotate_sample(
    s: SampleInput,
    *,
    af_table=None,
    method: str = "guide",
    seed: int = 0,
) -> Tuple["object", dict]:
    """Label table (see :mod:`singlet.annotate.output`) and a summary dict for one sample. ``af_table``: the 1000 Genomes
    super-population allele frequencies (:class:`~singlet.annotate.ancestry.AFIndex`, or a DataFrame with chrom, pos, ref, alt,
    AFR, AMR, EAS, EUR, SAS); without it, or without
    donor SNP depths in ``s``, ancestry stays null."""
    import pandas as pd

    t0 = time.time()
    mv = model_version(method)
    df = O.empty_table(s.gsm, s.col, mv)
    n = len(s.col)
    summary: dict = dict(
        gsm=s.gsm,
        gse=s.gse,
        n_cells=n,
        species=s.species,
        source=s.source,
        model_version=mv,
        barcode_digest=s.barcode_digest,
        tissue=s.tissue,
        timings={},
    )
    if n == 0 or s.species is None:
        summary["skipped"] = "no cells" if n == 0 else "organism has no model"
        return df, summary

    # ---- cell type
    t = time.time()
    ct = CT.annotate_cell_types(
        s.X,
        s.gene_names,
        species=s.species,
        tissue=s.tissue,
        disease=s.disease,
        method=method,
        seed=seed,
    )
    df["cell_type_cl"] = pd.array(ct.cl, dtype="string")
    df["cell_type_label"] = pd.array(ct.label, dtype="string")
    # guide calls: product of the gating confidences along the path (better calibrated than the last step alone)
    score = ct.path_score if ct.method == "guide" and ct.path_score is not None else ct.score
    df["cell_type_score"] = np.where(ct.node > 0, score, np.nan).astype(np.float32)
    summary["timings"]["cell_type_s"] = time.time() - t
    summary["cell_type"] = {k: v for k, v in ct.notes.items() if isinstance(v, (int, float, str))}
    vals, cnt = np.unique(np.asarray(ct.cl, dtype=object).astype(str), return_counts=True)
    summary["cell_type"]["counts"] = {str(v): int(c) for v, c in zip(vals, cnt)}

    # ---- sex (pooled per donor when demultiplexed)
    t = time.time()
    sx = SX.call_sex(
        s.X, s.gene_names, species=s.species, groups=None if s.donor is None else list(s.donor)
    )
    df["sex_chrom"] = pd.array(sx.label, dtype="string")
    df["sex_score"] = sx.score.astype(np.float32)
    summary["sex"] = dict(groups=sx.groups)
    summary["timings"]["sex_s"] = time.time() - t

    # ---- ancestry (human only; needs restricted donor depths and the allele-frequency table)
    if s.species == "human" and af_table is not None and s.donor_sites:
        t = time.time()
        idx = af_table if isinstance(af_table, ANC.AFIndex) else ANC.AFIndex(af_table)
        calls = {}
        for d, sites in s.donor_sites.items():
            ad, dp, F = idx.align(sites)
            if len(ad) < ANC.MIN_MATCH_FRAC * len(
                sites
            ):  # another genome build or species: no call rather than a wrong one
                calls[d] = ANC.AncestryCall(None, None, {}, int(len(ad)))
                continue
            calls[d] = ANC.infer_ancestry(ad, dp, F, seed=seed)
        if s.donor is None:
            # one donor in the sample: its call covers every cell; several depth tables but no labels: no mapping
            target = {"": calls["donor0"]} if list(calls) == ["donor0"] else {}
        else:
            target = calls
        han = np.full(n, None, dtype=object)
        prob = np.full(n, np.nan, np.float32)
        for d, c in target.items():
            if c.group is None:
                continue
            sel = np.ones(n, bool) if d == "" else (s.donor == d)
            han[sel] = c.hancestro
            prob[sel] = c.prob
        df["ancestry_hancestro"] = pd.array(han, dtype="string")
        df["ancestry_prob"] = prob
        summary["ancestry"] = {
            d: dict(
                group=c.group,
                hancestro=c.hancestro,
                prob=c.prob,
                n_sites=c.n_sites,
                stability=c.stability,
                reason=c.reason,
                q={k: round(v, 4) for k, v in c.q.items()},
            )
            for d, c in calls.items()
        }
        summary["timings"]["ancestry_s"] = time.time() - t
    summary["timings"]["total_s"] = time.time() - t0
    O.assert_public_safe(df)
    return df, summary
