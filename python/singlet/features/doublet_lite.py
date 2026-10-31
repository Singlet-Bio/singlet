# SPDX-License-Identifier: MIT
"""A small fixed atlas encoder for the doublet stacker: numpy only, no torch, well under 1 MB.

It replaces the 77 MB cell type network of ``doublet_atlas.py`` (docs/cell_features.md section 8). Fitted once on the CELLxGENE
training corpus of ``singlet.celltype_nn`` (atlas reference cells, never on the doublet evaluation datasets):

* a gene panel of G genes (human Ensembl ids, symbols, mouse one-to-one orthologs) and a linear encoder, PCA of the panel's
  log1p CP10k profiles: ``z = (log1p(1e4 c / sum(c over panel)) - mu) @ W``;
* a closed-form Gaussian classifier in that space over the atlas cell types (class means, one shared covariance), with a
  temperature fitted by held-out likelihood on atlas datasets that were not used for the means.

Features per sample (cells x genes raw counts):

``lite_entropy``  entropy of the classifier's posterior over atlas cell types (a low-rank stand-in for ``ct_entropy``);
``lite_knn``      fraction of simulated doublets (summed counts of random pairs, encoded the same way) among a cell's k nearest
                  neighbours in the fixed space (stand-in for ``atlas_knn``).

Cost is linear in cells apart from the neighbour search, which is blocked brute force (``n * (n + m) * dim`` multiply-adds)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from .doublet_util import simulate_doublets

DATA = Path(__file__).parent / "data"
MODEL_NPZ = DATA / "doublet_lite_v1.npz"


class Lite:
    def __init__(self, a: Dict[str, np.ndarray]):
        self.a = a
        self.dim = int(a["W"].shape[1])
        self.mu = a["mu"].astype(np.float32)
        self.W = a["W"].astype(np.float32)
        self.means = a["means"].astype(np.float32)
        self.L = a["prec_chol"].astype(np.float32)  # shared class precision = L L^T
        self.temp = float(a["temp"])
        self._hum = {str(g): i for i, g in enumerate(a["panel_ids"])}
        self._mou = {str(g): i for i, g in enumerate(a["panel_mouse"]) if str(g)}
        self._sym = {str(s).upper(): i for i, s in enumerate(a["panel_symbols"])}

    @classmethod
    def load(cls, path=MODEL_NPZ) -> "Lite":
        z = np.load(path)
        return cls({k: z[k] for k in z.files})

    # ------------------------------------------------------------------------------------------------------------ encoding
    def columns(self, gene_ids, species: Optional[str]) -> np.ndarray:
        """Panel position of each input gene (-1 = not on the panel). Ensembl ids (version suffix and ``hg19_``-style prefixes
        ignored; mouse ids through the ortholog table) or, when most names are not Ensembl ids, gene symbols (upper case)."""
        g = [str(x) for x in gene_ids]
        g = [x.split("_", 1)[1] if x[:5] in ("hg19_", "mm10_", "GRCh3", "GRCm3", "hg38_") and "_" in x else x for x in g]
        g = [x.split(".")[0] for x in g]
        if np.mean([x.startswith("ENS") for x in g]) > 0.5:
            return np.array([self._hum.get(x, self._mou.get(x, -1)) for x in g], np.int64)
        return np.array([self._sym.get(x.upper(), -1) for x in g], np.int64)

    def panel_counts(self, X: sp.csr_matrix, cols: np.ndarray) -> sp.csr_matrix:
        keep = np.flatnonzero(cols >= 0)
        P = sp.csr_matrix(
            (np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(cols), self.W.shape[0])
        )
        return sp.csr_matrix(sp.csr_matrix(X, dtype=np.float32) @ P)

    def encode_panel(self, C: sp.csr_matrix, block: int = 4096) -> np.ndarray:
        """Cells x dim coordinates of panel counts ``C`` (cells x G)."""
        out = np.empty((C.shape[0], self.dim), np.float32)
        tot = np.asarray(C.sum(1)).ravel()
        for s in range(0, C.shape[0], block):
            d = C[s : s + block].toarray()
            d *= (1e4 / np.maximum(tot[s : s + block], 1.0))[:, None]
            np.log1p(d, out=d)
            d -= self.mu
            out[s : s + block] = d @ self.W
        return out

    # ----------------------------------------------------------------------------------------------------------- features
    def space(self, Z: np.ndarray, kind: str = "pca", d: int = 0) -> np.ndarray:
        """Coordinates for the neighbour search: ``pca`` (as encoded); ``white`` (scaled by the within-class precision, so
        within-type variation counts as unit noise); ``lda`` (whitened, then the ``d`` directions along which the cell type
        means differ most)."""
        if kind == "pca":
            return Z
        Y = Z @ self.L
        if kind == "white":
            return Y
        if not hasattr(self, "_lda"):
            B = self.means @ self.L
            B = B[np.abs(self.means).max(1) < 1e3]
            B = B - B.mean(0)
            _, V = np.linalg.eigh(B.T @ B / len(B))
            self._lda = V[:, ::-1].astype(np.float32)
        return Y @ self._lda[:, : (d or self.dim)]

    def log_posterior(self, Z: np.ndarray) -> np.ndarray:
        """Log posterior over atlas cell types (uniform prior): Gaussian classes with a shared covariance, temperature ``temp``."""
        A = Z @ self.L
        B = self.means @ self.L
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
        logit = -0.5 * d2 / self.temp
        logit -= logit.max(1, keepdims=True)
        return logit - np.log(np.exp(logit).sum(1, keepdims=True))

    def entropy(self, Z: np.ndarray, block: int = 8192) -> np.ndarray:
        out = np.empty(len(Z))
        for s in range(0, len(Z), block):
            lp = self.log_posterior(Z[s : s + block])
            out[s : s + block] = -(np.exp(lp) * lp).sum(1)
        return out

    @staticmethod
    def knn_fraction(Z: np.ndarray, n_real: int, k: int, block: int = 2048) -> np.ndarray:
        """For each of the first ``n_real`` rows of ``Z``: fraction of its k nearest neighbours (self excluded) among rows
        ``>= n_real`` (the simulated doublets)."""
        sq = (Z * Z).sum(1)
        out = np.empty(n_real)
        for s in range(0, n_real, block):
            e = min(s + block, n_real)
            d = sq[s:e, None] + sq[None, :] - 2.0 * (Z[s:e] @ Z.T)
            d[np.arange(e - s), np.arange(s, e)] = np.inf
            nb = np.argpartition(d, k, axis=1)[:, :k]
            out[s:e] = (nb >= n_real).mean(1)
        return out

    def features(
        self,
        X: sp.csr_matrix,
        gene_ids,
        species: Optional[str],
        *,
        n_sim: Optional[int] = None,
        seed: int = 0,
        space: str = "pca",
        space_dim: int = 0,
    ) -> Dict[str, np.ndarray]:
        """``{"lite_entropy": ..., "lite_knn": ...}`` for the rows of ``X`` (raw counts). ``n_sim``: simulated doublets (default
        2 n, as the atlas encoder's feature)."""
        rng = np.random.default_rng(seed)
        X = sp.csr_matrix(X, dtype=np.float32)
        n = X.shape[0]
        cols = self.columns(gene_ids, species)
        C = self.panel_counts(X, cols)
        Z = self.encode_panel(C)
        S, _ = simulate_doublets(C, int(n_sim or 2 * n), rng)
        Zs = self.encode_panel(S)
        k = max(10, int(round(0.5 * np.sqrt(n))))
        knn = self.knn_fraction(self.space(np.vstack([Z, Zs]), space, space_dim), n, k)
        return {"lite_entropy": self.entropy(Z), "lite_knn": knn}


class Distilled(Lite):
    """The same panel and counts handling, with a one-hidden-layer network distilled from the atlas cell type network
    (``scripts/doublet_distill.py``): ``out = relu(((x - mu) / sd) @ W1 + b1) @ W2 + b2`` where ``x`` is the panel's log1p CP10k profile.
    ``out[:, :K]`` approximates the network's hidden-layer embedding projected on its global top-K principal axes (the
    neighbour-search space of ``atlas_knn``); ``out[:, K]`` its cell type entropy (``ct_entropy``)."""

    def __init__(self, a):
        self.a = a
        self.mu = a["mu"].astype(np.float32)
        self.sd = a["sd"].astype(np.float32)
        self.W1 = a["W1"].astype(np.float32)
        self.b1 = a["b1"].astype(np.float32)
        self.W2 = a["W2"].astype(np.float32)
        self.b2 = a["b2"].astype(np.float32)
        self.W = self.W1  # panel size = W.shape[0]
        self.K = self.W2.shape[1] - 1
        self._hum = {str(g): i for i, g in enumerate(a["panel_ids"])}
        self._mou = {str(g): i for i, g in enumerate(a["panel_mouse"]) if str(g)}
        self._sym = {str(s).upper(): i for i, s in enumerate(a["panel_symbols"])}

    def outputs(self, C: sp.csr_matrix, block: int = 4096) -> np.ndarray:
        out = np.empty((C.shape[0], self.K + 1), np.float32)
        tot = np.asarray(C.sum(1)).ravel()
        for s in range(0, C.shape[0], block):
            d = C[s : s + block].toarray()
            d *= (1e4 / np.maximum(tot[s : s + block], 1.0))[:, None]
            np.log1p(d, out=d)
            d -= self.mu
            d /= self.sd
            h = np.maximum(d @ self.W1 + self.b1, 0.0)
            out[s : s + block] = h @ self.W2 + self.b2
        return out

    def features(self, X, gene_ids, species, *, n_sim=None, seed=0, dim=0, **_):
        rng = np.random.default_rng(seed)
        X = sp.csr_matrix(X, dtype=np.float32)
        n = X.shape[0]
        C = self.panel_counts(X, self.columns(gene_ids, species))
        S, _ = simulate_doublets(C, int(n_sim or 2 * n), rng)
        o = self.outputs(C)
        os_ = self.outputs(S)
        k = max(10, int(round(0.5 * np.sqrt(n))))
        d = dim or self.K
        knn = self.knn_fraction(np.vstack([o[:, :d], os_[:, :d]]), n, k)
        return {"lite_entropy": o[:, self.K].astype(np.float64), "lite_knn": knn}

    @classmethod
    def load(cls, path) -> "Distilled":
        z = np.load(path)
        return cls({k: z[k] for k in z.files})
