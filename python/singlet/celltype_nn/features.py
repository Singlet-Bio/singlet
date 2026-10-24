# SPDX-License-Identifier: MIT
"""Model inputs: a fixed gene vocabulary keyed by human Ensembl id, the count transform, and training augmentation.

* **Vocabulary.** Human Ensembl gene ids. Mouse genes enter through Ensembl one-to-one orthologs (other mouse genes are dropped),
  so both species share one input layer. Gene symbols are never used to align inputs.
* **Transform.** ``log1p(1e4 * c / sum_vocab(c))``: counts are scaled by the cell's total over the vocabulary only, so genes
  outside the vocabulary (mitochondrial, ribosomal, intronic-only, pipeline-specific) cannot shift the scaling.
* **Augmentation** (training only, applied to counts before the transform) simulates what changes when Singlet reprocesses a
  study: depth (binomial thinning to 5-100%), intronic reads (each gene gains reads in proportion to its counts and its
  intron/exon ratio measured on Singlet's own spliced/unspliced layers), ambient RNA (counts drawn from the batch's mean
  profile), per-gene protocol bias (log-normal factors from a fixed bank of "protocols") and genes missing from an annotation
  (random gene dropout).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sps


@dataclass
class GeneVocab:
    genes: List[str]  # human Ensembl ids, in input order
    symbols: List[str]
    mouse: Dict[str, str] = field(default_factory=dict)  # mouse Ensembl id -> human Ensembl id (one-to-one orthologs)

    def __post_init__(self):
        self.index = {g: i for i, g in enumerate(self.genes)}

    def __len__(self) -> int:
        return len(self.genes)

    def columns(self, gene_ids: Sequence[str], species: str) -> np.ndarray:
        """Vocabulary position of each input gene (-1 = not in the vocabulary). Version suffixes are ignored."""
        out = np.full(len(gene_ids), -1, np.int64)
        for j, g in enumerate(gene_ids):
            g = str(g).split(".")[0]
            if species == "mouse":
                g = self.mouse.get(g, "")
            out[j] = self.index.get(g, -1)
        return out

    def project(self, X, gene_ids: Sequence[str], species: str) -> sps.csr_matrix:
        """Cells x vocabulary counts (duplicates summed, missing genes zero)."""
        cols = self.columns(gene_ids, species)
        keep = np.flatnonzero(cols >= 0)
        M = sps.csr_matrix((np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(cols), len(self)))
        return sps.csr_matrix(sps.csr_matrix(X, dtype=np.float32) @ M)

    def to_dict(self) -> dict:
        return dict(genes=self.genes, symbols=self.symbols, mouse=self.mouse)

    @classmethod
    def from_dict(cls, d: dict) -> "GeneVocab":
        return cls(list(d["genes"]), list(d["symbols"]), dict(d.get("mouse") or {}))


def read_orthologs(path: Path) -> Dict[str, str]:
    """mouse Ensembl id -> human Ensembl id from ``orthologs_human_mouse.tsv.gz`` (scripts/celltype_nn/corpus.py orthologs)."""
    out = {}
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            h, _, m, _ = line.rstrip("\n").split("\t")
            out[m] = h
    return out


def transform(counts: np.ndarray) -> np.ndarray:
    """log1p(1e4 * c / total) per row; works on numpy arrays and torch tensors."""
    tot = counts.sum(1, keepdims=True) if isinstance(counts, np.ndarray) else counts.sum(1, keepdim=True)
    scale = 1e4 / (tot + 1e-6) if isinstance(counts, np.ndarray) else 1e4 / tot.clamp_min(1e-6)
    x = counts * scale
    return np.log1p(x) if isinstance(counts, np.ndarray) else x.log1p()


@dataclass
class AugmentConfig:
    p_thin: float = 0.8
    thin_min: float = 0.05
    p_intron: float = 0.5
    intron_max: float = 1.5
    p_ambient: float = 0.5
    ambient_max: float = 0.15
    p_bias: float = 0.7
    bias_sd: float = 0.3
    n_protocols: int = 64
    p_dropout: float = 0.5
    dropout_max: float = 0.1


class Augmenter:
    """Batch augmentation on a torch device. ``intron_ratio`` is per vocabulary gene (unspliced / spliced on Singlet counts)."""

    def __init__(self, n_genes: int, intron_ratio: Optional[np.ndarray], cfg: AugmentConfig, device, seed: int = 0):
        import torch

        self.cfg, self.device = cfg, device
        g = torch.Generator().manual_seed(seed)
        self.bias = torch.exp(cfg.bias_sd * torch.randn(cfg.n_protocols, n_genes, generator=g)).to(device)
        r = np.zeros(n_genes, np.float32) if intron_ratio is None else np.asarray(intron_ratio, np.float32)
        self.intron = torch.as_tensor(np.clip(r, 0, 20), device=device)

    def __call__(self, c):
        import torch

        cfg, B, dev = self.cfg, c.shape[0], c.device

        def coin(p):
            return (torch.rand(B, 1, device=dev) < p).float()

        # intronic reads: each gene gains reads in proportion to its counts and its intron/exon ratio
        s = coin(cfg.p_intron) * torch.rand(B, 1, device=dev) * cfg.intron_max
        rate = c * (1 + s * self.intron)
        # protocol bias
        k = torch.randint(0, cfg.n_protocols, (B,), device=dev)
        use = coin(cfg.p_bias)
        rate = rate * (use * self.bias[k] + (1 - use))
        # ambient RNA from the batch mean profile
        prof = c.sum(0, keepdim=True)
        prof = prof / prof.sum().clamp_min(1)
        a = coin(cfg.p_ambient) * torch.rand(B, 1, device=dev) * cfg.ambient_max
        rate = rate + a * c.sum(1, keepdim=True) * prof
        # depth: thin to a log-uniform fraction
        lo = np.log(cfg.thin_min)
        f = torch.exp(lo * torch.rand(B, 1, device=dev))
        f = torch.where(coin(cfg.p_thin) > 0, f, torch.ones_like(f))
        out = torch.poisson(rate * f)
        # genes missing from the annotation
        d = coin(cfg.p_dropout) * torch.rand(B, 1, device=dev) * cfg.dropout_max
        out = out * (torch.rand_like(out) >= d).float()
        return out
