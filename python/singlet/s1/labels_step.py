# SPDX-License-Identifier: MIT
"""Per-cell labels computed inside the transcode, from the matrices the transcoder has already decoded (docs/cell_annotation.md).

One call per sample, after its .s1 object is final (the label sidecar records the digest of the ``cells`` member it was computed
on). The labeller is :func:`singlet.celllabels.pipeline.annotate_sample` with ``method="guide"`` (the atlas default):

* cell type: guide gating with tissue context from the GEO text and kNN smoothing; the full CL path with per-level confidences;
* chromosomal sex (human, mouse): XIST against non-PAR chrY; never the legacy ``sex_call.json``;
* genetic ancestry (human donors with per-donor depth tables only): the tables are read on the compute node, in memory, and only a
  HANCESTRO id and a proportion per cell leave this module.

Species without a model get no sidecar at all. A sample whose labelling would exceed its budget (time, memory estimate) or fails is
written without labels, and the reason is returned for the ledger; the transcode itself never fails because of the labeller.

The sidecar is the canonical form (:func:`singlet.s1.atlas.normalize_sidecar`, ``s1/a/<GSM>/<d16>.parquet``), referenced from
the study manifest (``samples[i].annotations[<annot_version>]``) and ingested by the atlas cell index.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

#: the annotation version name the manifest uses for these sidecars
ANNOT_VERSION = "a2026-10"
#: what the manifest says about the cell type levels (docs/cell_annotation.md 3: per-study accuracy by guide depth)
CELL_TYPE_DISPLAY = {
    "method": "guide",
    "default_levels": [1, 2],
    "preview_from_level": 3,
    "level_names": {"1": "compartment", "2": "lineage", "3": "cell type", "4": "subtype"},
    "per_study_accuracy": {"1": 0.842, "2": 0.747, "3": 0.428, "4": 0.320},
}


class LabelBudget(Exception):
    """The sample would exceed the labeller's budget."""


class LabelOptions:
    def __init__(
        self,
        af_path: Optional[str] = None,
        restricted_sidecar: Optional[str] = None,
        timeout_s: float = 1800.0,
        max_gb: float = 24.0,
        annot_version: str = ANNOT_VERSION,
        method: str = "guide",
        log=None,
    ):
        self.af_path = af_path
        self.restricted_sidecar = restricted_sidecar
        self.timeout_s = timeout_s
        self.max_gb = max_gb
        self.annot_version = annot_version
        self.method = method
        self.log = log or (lambda *a: None)
        self._af = None

    def af_index(self):
        """The 1000 Genomes super-population frequency table, loaded once per process (only when a sample needs it)."""
        if self._af is None and self.af_path:
            import pandas as pd

            from singlet.celllabels.ancestry import AFIndex

            t = time.time()
            self._af = AFIndex(pd.read_parquet(self.af_path))
            self.log("  allele-frequency table loaded in %.1fs" % (time.time() - t))
        return self._af


@contextlib.contextmanager
def _deadline(seconds: float):
    """SIGALRM-based time limit in the main thread on POSIX; elsewhere no limit (the campaign's own timeout still applies)."""
    use = (
        seconds
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and __import__("threading").current_thread() is __import__("threading").main_thread()
    )
    if not use:
        yield
        return

    def _raise(*_a):
        raise LabelBudget("labelling took longer than %.0f s" % seconds)

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def gene_level_called(exon, intron, basis_barcodes: List[str], called: np.ndarray, vocab: dict):
    """(X called cells x genes CSR float32, gene names, col) from the decoded legacy matrices, as ``SingletBundle.raw_counts`` builds
    it: feature rows summed onto the bundle's gene axis (``feature_vocab.json`` order), exon + intron, called cells only, in store
    column order (``col`` = the cell's column in the sample object)."""
    import scipy.sparse as sp

    genes = vocab.get("genes") or []
    gid = {str(g["gene_id"]): i for i, g in enumerate(genes)}
    names = [str(g.get("gene_name") or g["gene_id"]) for g in genes]
    col = np.flatnonzero(np.asarray(called, bool)).astype(np.int64)
    want = [basis_barcodes[i] for i in col]
    X = None
    for m in (exon, intron):
        if m is None or m.n == 0 or not m.rownames:
            continue
        rows = np.fromiter(
            (gid.get(str(r).split("_")[0], -1) for r in m.rownames), np.int64, count=m.m
        )
        M = sp.csc_matrix(
            (m.data.astype(np.float32), m.indices.astype(np.int64), m.indptr.astype(np.int64)),
            shape=(m.m, m.n),
        )
        if list(m.colnames) == basis_barcodes:
            M = M[:, col]
        else:  # align by barcode (missing = 0)
            pos = {b: i for i, b in enumerate(m.colnames)}
            src = np.array([pos.get(b, -1) for b in want], np.int64)
            ok = src >= 0
            S = sp.csc_matrix(
                (np.ones(int(ok.sum()), np.float32), (src[ok], np.flatnonzero(ok))),
                shape=(m.n, len(want)),
            )
            M = (M @ S).tocsc()
        ok = rows >= 0
        P = sp.csr_matrix(
            (np.ones(int(ok.sum()), np.float32), (np.flatnonzero(ok), rows[ok])),
            shape=(m.m, len(genes)),
        )
        part = (M.T.tocsr() @ P).tocsr()
        X = part if X is None else (X + part).tocsr()
    if X is None:
        X = sp.csr_matrix((len(want), len(genes)), dtype=np.float32)
    X.sort_indices()
    return X.astype(np.float32), names, col, want


def _donor_labels(tsv: Optional[bytes], barcodes: List[str]):
    if not tsv:
        return None
    import pandas as pd

    from singlet.celllabels.io import _donor_labels as dl

    try:
        df = pd.read_csv(io.BytesIO(tsv), sep="\t")
    except Exception:
        return None
    return dl(df, barcodes)


def donor_sites_from_members(members: Dict[str, bytes]):
    """{donor<k>: DataFrame(chrom, pos, ref, alt, dp, ad)} from ``donor/donor<k>_coverage.tsv`` bytes (in memory only)."""
    import re

    import pandas as pd

    out = {}
    for rel, data in members.items():
        m = re.search(r"(?:^|/)donor/(donor\d+)_coverage\.tsv$", rel)
        if not m:
            continue
        out[m.group(1)] = pd.read_csv(
            io.BytesIO(data),
            sep="\t",
            usecols=["chrom", "pos", "ref", "alt", "dp", "ad"],
            dtype={
                "chrom": str,
                "pos": np.int64,
                "ref": str,
                "alt": str,
                "dp": np.int64,
                "ad": np.int64,
            },
        )
    return out


def label_sample(
    *,
    gsm: str,
    gse: str,
    exon,
    intron,
    basis_barcodes: List[str],
    called: np.ndarray,
    cells_digest: str,
    vocab: dict,
    study_meta: dict,
    donor_tsv: Optional[bytes],
    depth_members: Dict[str, bytes],
    opts: LabelOptions,
) -> dict:
    """-> {"status": "ok"|"skipped"|"failed"|"over_budget", "reason", "sidecar": bytes, "entry": manifest entry, "coverage", "summary",
    "seconds"}. Never raises."""
    from singlet.celllabels import io as LIO
    from singlet.celllabels import output as O
    from singlet.celllabels.pipeline import annotate_sample

    from . import atlas

    t0 = time.time()
    res: dict = {"status": "skipped", "reason": "", "seconds": 0.0}
    try:
        meta = dict((study_meta.get("gsm_meta") or {}).get(gsm, {}))
        species = LIO.species_of(meta.get("organism", ""))
        if species is None:
            res["reason"] = "organism has no label model (%s)" % (meta.get("organism") or "unknown")
            return res
        n_called = int(np.asarray(called, bool).sum())
        if n_called == 0:
            res["reason"] = "no called cells"
            return res
        X, genes, col, bcs = gene_level_called(exon, intron, basis_barcodes, called, vocab)
        # peak memory of the guide: a few copies of X (float32 + int32 indices) plus ~8 kB per cell (node scores, SVD, kNN)
        est_gb = (X.nnz * 8 * 3 + n_called * 8000) / 1e9
        if opts.max_gb and est_gb > opts.max_gb:
            raise LabelBudget("estimated %.1f GB > budget %.1f GB" % (est_gb, opts.max_gb))
        donor = _donor_labels(donor_tsv, bcs)
        s = LIO.SampleInput(
            gsm=gsm,
            gse=gse,
            X=X,
            gene_names=genes,
            col=col,
            barcodes=bcs,
            species=species,
            tissue=LIO.tissue_text(meta, str(study_meta.get("series_title") or "")),
            disease=LIO.disease_text(meta),
            meta=meta,
            donor=donor,
            source="store-transcode",
        )
        af = None
        if species == "human":
            sites = donor_sites_from_members(depth_members) if depth_members else {}
            if not sites and opts.restricted_sidecar and os.path.exists(opts.restricted_sidecar):
                sites = LIO.read_donor_sites(opts.restricted_sidecar, gsm)
            if sites:
                s.donor_sites = sites
                af = opts.af_index()
        with _deadline(opts.timeout_s):
            df, summary = annotate_sample(s, af_table=af, method=opts.method)
        s.donor_sites = {}  # the depth tables go no further
        O.assert_public_safe(df)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, gsm + ".labels.parquet")
            O.write_labels(df, p)
            data, entry = atlas.normalize_sidecar(
                p,
                gsm=gsm,
                n_columns=len(basis_barcodes),
                called=np.asarray(called, bool),
                cells_digest=cells_digest,
                annot_version=opts.annot_version,
            )
            cp = os.path.join(td, "canonical.parquet")
            with open(cp, "wb") as fh:
                fh.write(data)
            hits = O.scan_labels_file(cp)
        if hits:
            raise RuntimeError("genotype-like content in the label sidecar: %r" % hits[:3])
        entry = dict(entry, cell_type=CELL_TYPE_DISPLAY)
        res.update(
            status="ok",
            sidecar=data,
            entry=entry,
            coverage=coverage(df),
            summary=_summary_public(summary),
        )
    except LabelBudget as e:
        res.update(status="over_budget", reason=str(e))
    except MemoryError:
        res.update(status="over_budget", reason="out of memory")
    except Exception as e:  # noqa: BLE001  the labeller must never fail the transcode
        res.update(status="failed", reason="%s: %s" % (type(e).__name__, str(e)[:300]))
    finally:
        res["seconds"] = round(time.time() - t0, 3)
    return res


def coverage(df) -> dict:
    """Cells labelled per label type (for the pilot report and the ledger)."""
    n = int(len(df))
    return {
        "cells": n,
        "cell_type": int(df["cell_type_cl"].notna().sum()),
        "cell_type_level2": int(df["cell_type_p2"].notna().sum()) if "cell_type_p2" in df else 0,
        "sex": int(df["sex_chrom"].notna().sum()),
        "sex_male": int((df["sex_chrom"] == "male").sum()),
        "sex_female": int((df["sex_chrom"] == "female").sum()),
        "ancestry": int(df["ancestry_hancestro"].notna().sum()),
    }


def _summary_public(summary: dict) -> dict:
    """The label summary without per-site or per-population detail (only group, prob, n_sites, stability, reason per donor)."""
    out = {
        k: summary.get(k) for k in ("species", "model_version", "tissue", "timings") if k in summary
    }
    if "sex" in summary:
        out["sex"] = summary["sex"]
    if "ancestry" in summary:
        out["ancestry"] = {
            d: {
                k: v.get(k)
                for k in ("group", "hancestro", "prob", "n_sites", "stability", "reason")
            }
            for d, v in summary["ancestry"].items()
        }
    if "skipped" in summary:
        out["skipped"] = summary["skipped"]
    return json.loads(json.dumps(out, default=float))
