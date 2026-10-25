# SPDX-License-Identifier: MIT
"""Run a trained cell type network on one sample and summarise its probabilities.

::

    from singlet.celllabels import read_v2
    from singlet.celltype_nn import CellTypePredictor
    s = read_v2("GSE145370", "GSM4317426")
    pred = CellTypePredictor.load("model.pt")
    P = pred.predict(s.X, gene_ids, s.species)          # cells x output terms, rows sum to 1
    table = pred.table(P, gsm=s.gsm, col=s.col)        # the sidecar (singlet.celltype_nn.output)
    up = pred.space.rollup(P)                           # probability of every ancestor term (cells x closure terms)

``call``: the most specific term whose rolled-up probability is at least 0.5 (so mass spread over several subtypes still
supports their common parent); ``argmax``: the single output term with the most mass. Neither is forced: a cell whose mass is
spread over unrelated branches gets a general call with a low probability.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sps

from .features import GeneVocab
from .output import COLUMNS
from .ontology import TermSpace

TOP_K = 5


class CellTypePredictor:
    def __init__(self, net, meta: dict):
        self.net = net
        self.meta = meta
        self.vocab = GeneVocab.from_dict(meta["vocab"])
        self.space = TermSpace(meta["terms"])
        self.version = str(meta.get("model_version") or meta.get("version") or "unversioned")

    @classmethod
    def load(cls, path, version: Optional[str] = None) -> "CellTypePredictor":
        from . import model as M

        net, meta = M.load(path)
        if version:
            meta["model_version"] = version
        return cls(net, meta)

    def predict(
        self, X, gene_ids: Sequence[str], species: str, tissue: Optional[str] = None, batch: int = 8192, threads: Optional[int] = None
    ) -> np.ndarray:
        """Calibrated probabilities, cells x output terms. ``X``: raw counts (cells x genes); ``gene_ids``: Ensembl ids;
        ``tissue``: the sample's tissue text (as :func:`singlet.celllabels.io.tissue_text` gives it), used by models trained
        with tissue context; None, or text naming no organ, means no context."""
        import torch

        from .features import tissue_context, transform

        if threads:
            torch.set_num_threads(threads)
        V = self.vocab.project(sps.csr_matrix(X), gene_ids, species)
        ctx = None
        if self.net.n_context:
            ctx = torch.from_numpy(tissue_context([tissue], self.meta["tissue_groups"]))
        out = np.zeros((V.shape[0], len(self.space)), np.float32)
        with torch.no_grad():
            for s in range(0, V.shape[0], batch):
                c = torch.from_numpy(V[s : s + batch].toarray())
                cx = None if ctx is None else ctx.expand(c.shape[0], -1)
                out[s : s + batch] = self.net.proba(transform(c, self.net.transform), cx).numpy()
        return out

    def summarise(self, P: np.ndarray, threshold: float = 0.5, k: int = TOP_K) -> dict:
        """Per cell: call (closure term) and its rolled-up probability, argmax term and its probability, top-k terms."""
        j, pj = self.space.call(P, threshold)
        call = np.array(self.space.closure, dtype=object)[j]
        am = P.argmax(1)
        top = np.argsort(-P, axis=1)[:, :k]
        terms = np.array(self.space.terms, dtype=object)
        return dict(
            call=call,
            call_prob=pj.astype(np.float32),
            argmax=terms[am],
            argmax_prob=P[np.arange(len(am)), am].astype(np.float32),
            topk=terms[top],
            topk_prob=np.take_along_axis(P, top, 1).astype(np.float32),
        )

    def table(self, P: np.ndarray, gsm: str, col: np.ndarray, threshold: float = 0.5):
        """The per-cell sidecar table (``singlet.celltype_nn.output.COLUMNS``)."""
        import pandas as pd

        s = self.summarise(P, threshold)
        root = {t for t in self.space.closure if not self.space.o.terms.get(t, {}).get("is_a")}
        call = np.array([None if c in root else c for c in s["call"]], dtype=object)
        d = {
            "gsm": np.full(len(col), gsm, dtype=object),
            "col": np.asarray(col, np.int64),
            "cell_type_cl": call,
            "cell_type_label": np.array([self.space.name(c) if c else None for c in call], dtype=object),
            "cell_type_prob": np.where(pd.isna(call), np.float32(np.nan), s["call_prob"]).astype(np.float32),
            "argmax_cl": s["argmax"],
            "argmax_prob": s["argmax_prob"],
        }
        for i in range(s["topk"].shape[1]):
            d[f"top{i + 1}_cl"] = s["topk"][:, i]
        for i in range(s["topk"].shape[1]):
            d[f"top{i + 1}_prob"] = s["topk_prob"][:, i]
        d["model_version"] = np.full(len(col), self.version, dtype=object)
        return pd.DataFrame(d)[[c for c, _ in COLUMNS]]
