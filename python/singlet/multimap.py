# SPDX-License-Identifier: MIT
"""Multi-mapping reads at retrieval.

The gene count matrices hold only molecules whose reads are assigned to one gene. Molecules whose reads are compatible with two
genes are kept beside them in ``multimap_counts`` (classes x barcodes: one row per gene set, named ``"<gene_id>_<name>|<gene_id>_<name>"``
or ``"<gene_id>_<name>"`` for a molecule whose reads share a single gene; see ``include/singlet/pileup/multimap_record.h``).

:func:`distribute` turns that record into per-gene counts the way STARsolo's ``--soloMultiMappers`` does in
``SoloFeature_collapseUMIall.cpp``, cell by cell:

* ``"Uniform"``: each molecule adds 1/n to each of its n genes.
* ``"EM"``: the expectation-maximization STARsolo runs, started from unique + uniform counts, with its constants (counts below 0.01
  are set to zero before each step; it stops when no gene changes by 0.01 or more, or after 101 steps). The unique counts of the cell
  are the prior, so a molecule shared by a highly and a lowly expressed gene goes mostly to the first.

The result holds only the multimapper part (STARsolo's UniqueAndMult matrix minus its Unique matrix): add it to the unique counts to
include multi-mapping reads, or leave it out.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

METHODS = ("EM", "Uniform")
_EM_ZERO = 0.01  # STARsolo: gEMold below this is set to 0 before each step
_EM_TOL = 0.01  # STARsolo: stop when max |new - old| < this
_EM_MAX = 100  # STARsolo: stop after iterI > 100


def class_genes(names: Sequence[str], gene_row: Dict[str, int]) -> List[Tuple[int, ...]]:
    """The gene rows of each class name (``gene_id_name`` parts joined by ``|``); a class with an unknown gene gets ``()``."""
    out: List[Tuple[int, ...]] = []
    for nm in names:
        rows = []
        for part in str(nm).split("|"):
            r = gene_row.get(part.split("_", 1)[0], -1)
            if r < 0:
                rows = []
                break
            rows.append(r)
        out.append(tuple(sorted(set(rows))))
    return out


def distribute(unique, record, classes: Sequence[Tuple[int, ...]], method: str = "EM"):
    """Per-gene multimapper counts (cells x genes, float64 CSR) from the record.

    Parameters
    ----------
    unique
        cells x genes unique counts (sparse or dense); the EM prior.
    record
        cells x classes molecule counts (sparse or dense), the same cells in the same order.
    classes
        per class (column of ``record``) its gene rows (:func:`class_genes`); empty classes are skipped.
    method
        ``"EM"`` (default) or ``"Uniform"``.
    """
    import scipy.sparse as sp

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    U = sp.csr_matrix(unique, dtype=np.float64)
    M = sp.coo_matrix(record)
    n_cells, n_genes = U.shape  # pyright: ignore  (scipy stubs type the shape as optional)
    if M.shape[0] != n_cells:
        raise ValueError("unique and record must have the same cells")
    sizes = np.array([len(c) for c in classes], dtype=np.int64)
    keep = (M.data > 0) & (sizes[M.col] > 0)
    cell, cls, cnt = (
        M.row[keep].astype(np.int64),
        M.col[keep].astype(np.int64),
        M.data[keep].astype(np.float64),
    )
    if cell.size == 0:
        return sp.csr_matrix((n_cells, n_genes), dtype=np.float64)
    g1 = np.array([c[0] if c else -1 for c in classes], dtype=np.int64)[cls]
    g2 = np.array([c[1] if len(c) > 1 else -1 for c in classes], dtype=np.int64)[cls]
    two = g2 >= 0

    # the (cell, gene) variables: STARsolo's genesM per cell (the genes of that cell's multi-gene molecules)
    keys = np.concatenate([cell * n_genes + g1, cell[two] * n_genes + g2[two]])
    var_keys, inv = np.unique(keys, return_inverse=True)
    v1 = inv[: cell.size]
    v2 = np.full(cell.size, -1, dtype=np.int64)
    v2[two] = inv[cell.size :]
    var_cell, var_gene = var_keys // n_genes, var_keys % n_genes
    nv = var_keys.size

    w = np.where(two, 0.5, 1.0) * cnt
    uniform = np.bincount(v1, w, nv) + np.bincount(v2[two], w[two], nv)
    if method == "Uniform":
        val = uniform
    else:
        prior = np.asarray(U[var_cell, var_gene]).ravel()
        old = uniform + prior
        result = np.zeros(nv)
        active = np.ones(
            n_cells, dtype=bool
        )  # each cell stops on its own, as STARsolo runs one EM per cell
        it = 0
        while True:
            it += 1
            o = np.where(old < _EM_ZERO, 0.0, old)
            norm = o[v1] + np.where(two, o[np.maximum(v2, 0)], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = cnt / norm
            new = (
                prior
                + np.bincount(v1, o[v1] * f, nv)
                + np.bincount(v2[two], (o[v2[two]] * f[two]), nv)
            )
            change = np.zeros(n_cells)
            np.maximum.at(change, var_cell, np.abs(new - o))
            done = active & ((change < _EM_TOL) | (it > _EM_MAX))
            dv = done[var_cell]
            result[dv] = new[dv]
            active &= ~done
            if not active.any():
                break
            old = np.where(active[var_cell], new, old)
        val = result - prior
    return sp.csr_matrix((val, (var_cell, var_gene)), shape=(n_cells, n_genes))
