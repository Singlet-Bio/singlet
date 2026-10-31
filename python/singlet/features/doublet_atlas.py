# SPDX-License-Identifier: MIT
"""A small atlas-trained encoder for the doublet stacker: numpy only, 0.5 MB, no torch.

A one-hidden-layer network on a 1,000-gene panel, *distilled* from the atlas cell type network (``singlet.celltype_nn``, m1_notissue,
trained on 2.8M CELLxGENE cells) by ``scripts/doublet_encoder_build.py``: ``out = relu(((x - mu) / sd) @ W1 + b1) @ W2 + b2`` where ``x`` is
the panel's log1p CP10k profile. ``out[:, :32]`` approximates the network's hidden-layer embedding projected on its 32 leading principal
axes; ``out[:, 32]`` its cell type entropy. The student was fitted on atlas reference cells only (never on the doublet evaluation
datasets), on real cells and on sums of random pairs, both under the network's own count augmentation (depth thinning, protocol bias,
ambient RNA, intron reads).

Features per sample (cells x genes raw counts):

``atlas_knn``      fraction of simulated doublets (summed raw counts of random cell pairs) among a cell's k nearest neighbours in the
                   embedding. The reference is fixed in size: at most ``REF_CAP`` of the sample's cells and twice as many simulated
                   doublets, so the cost is O(n * REF_CAP); a cell's own reference row is excluded;
``atlas_entropy``  the entropy output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import hashlib

import numpy as np
import scipy.sparse as sp

from .doublet_util import simulate_doublets

DATA = Path(__file__).parent / "data"
ENCODER_NPZ = DATA / "doublet_v2_encoder.npz"
REF_CAP = 3000
SIM_RATIO = 2
K_MULT = 2.0
K_EMB = 32


class Atlas:
    def __init__(self, a: Dict[str, np.ndarray]):
        self.a = a
        self.mu = a["mu"].astype(np.float32)
        self.sd = a["sd"].astype(np.float32)
        self.W1 = a["W1"].astype(np.float32)
        self.b1 = a["b1"].astype(np.float32)
        self.W2 = a["W2"].astype(np.float32)
        self.b2 = a["b2"].astype(np.float32)
        self.K = self.W2.shape[1] - 1
        self._hum = {str(g): i for i, g in enumerate(a["panel_ids"])}
        self._mou = {str(g): i for i, g in enumerate(a["panel_mouse"]) if str(g)}
        self._sym = {str(s).upper(): i for i, s in enumerate(a["panel_symbols"])}

    @classmethod
    def load(cls, path=ENCODER_NPZ) -> "Atlas":
        z = np.load(path)
        return cls({k: z[k] for k in z.files})

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.a):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.a[k]).tobytes())
        return h.hexdigest()

    def columns(self, gene_ids, species: Optional[str] = None) -> np.ndarray:
        """Panel position of each input gene (-1 = not on the panel). Ensembl ids (version suffix and ``hg19_``-style prefixes
        ignored; mouse ids through the one-to-one ortholog table) or, when most names are not Ensembl ids, gene symbols (upper case)."""
        g = [str(x) for x in gene_ids]
        g = [x.split("_", 1)[1] if x[:5] in ("hg19_", "mm10_", "GRCh3", "GRCm3", "hg38_") and "_" in x else x for x in g]
        g = [x.split(".")[0] for x in g]
        if np.mean([x.startswith("ENS") for x in g]) > 0.5:
            return np.array([self._hum.get(x, self._mou.get(x, -1)) for x in g], np.int64)
        return np.array([self._sym.get(x.upper(), -1) for x in g], np.int64)

    def panel_counts(self, X: sp.csr_matrix, cols: np.ndarray) -> sp.csr_matrix:
        keep = np.flatnonzero(cols >= 0)
        P = sp.csr_matrix((np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(cols), self.W1.shape[0]))
        return sp.csr_matrix(sp.csr_matrix(X, dtype=np.float32) @ P)

    def outputs(self, C: sp.csr_matrix, block: int = 4096) -> np.ndarray:
        out = np.empty((C.shape[0], self.K + 1), np.float32)
        tot = np.asarray(C.sum(1)).ravel()
        for s in range(0, C.shape[0], block):
            d = C[s : s + block].toarray()
            d *= (1e4 / np.maximum(tot[s : s + block], 1.0))[:, None]
            np.log1p(d, out=d)
            d -= self.mu
            d /= self.sd
            out[s : s + block] = np.maximum(d @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2
        return out

    @staticmethod
    def knn_ref(Q: np.ndarray, ref: np.ndarray, m: int, k: int, pos: np.ndarray, block: int = 4096) -> np.ndarray:
        """Fraction of simulated doublets (rows ``>= m`` of ``ref``) among the k nearest reference rows of each query; ``pos[i]``
        is the reference row that is query i itself (-1 if none), excluded."""
        rq = (ref * ref).sum(1)
        out = np.empty(len(Q))
        for s in range(0, len(Q), block):
            q = Q[s : s + block]
            d = (q * q).sum(1)[:, None] + rq[None, :] - 2.0 * (q @ ref.T)
            sel = np.flatnonzero(pos[s : s + block] >= 0)
            d[sel, pos[s : s + block][sel]] = np.inf
            nb = np.argpartition(d, k, axis=1)[:, :k]
            out[s : s + block] = (nb >= m).mean(1)
        return out

    def features(self, X, gene_ids, species: Optional[str] = None, *, seed: int = 0) -> Dict[str, np.ndarray]:
        """``{"atlas_knn": ..., "atlas_entropy": ...}`` for the rows of ``X`` (raw counts, cells x genes named by ``gene_ids``)."""
        rng = np.random.default_rng(seed)
        X = sp.csr_matrix(X, dtype=np.float32)
        n = X.shape[0]
        C = self.panel_counts(X, self.columns(gene_ids, species))
        o = self.outputs(C)
        E = o[:, :K_EMB]
        sub = np.sort(rng.choice(n, REF_CAP, replace=False)) if n > REF_CAP else np.arange(n)
        m = len(sub)
        S, _ = simulate_doublets(C, int(SIM_RATIO * m), rng)
        ref = np.vstack([E[sub], self.outputs(S)[:, :K_EMB]])
        pos = np.full(n, -1)
        pos[sub] = np.arange(m)
        k = max(10, int(round(K_MULT * 0.5 * np.sqrt(m))))
        return {"atlas_knn": self.knn_ref(E, ref, m, k, pos), "atlas_entropy": o[:, self.K].astype(np.float64)}
