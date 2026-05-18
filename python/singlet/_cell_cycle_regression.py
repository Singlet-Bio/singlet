# SPDX-License-Identifier: MIT
"""Cell cycle regression.

Provides singlet.cell_cycle_regression() — score cell cycle phases and
optionally regress out cell cycle effects from expression data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def cell_cycle_regression(
    adata: "AnnData",
    *,
    s_genes: list[str] | None = None,
    g2m_genes: list[str] | None = None,
    regress: bool = True,
    layer: str | None = None,
) -> "AnnData":
    """Score cell cycle and regress out cell cycle effects.

    First scores S and G2M phase activity per cell, assigns cell cycle
    phase labels, and optionally regresses out the cell cycle signal
    from the expression matrix.

    Parameters
    ----------
    adata
        Annotated data matrix (log-normalized recommended).
    s_genes
        S-phase marker genes. If None, uses Tirosh 2016 human gene list.
    g2m_genes
        G2/M marker genes. If None, uses Tirosh 2016 human gene list.
    regress
        If True, regress out S_score and G2M_score from adata.X using
        linear regression per gene.
    layer
        Expression layer for regression. None uses .X.

    Returns
    -------
    AnnData
        The input adata (modified in place) with added:
        - adata.obs['cell_cycle_phase']: categorical (G1, S, G2M)
        - adata.obs['S_score']: S-phase score per cell
        - adata.obs['G2M_score']: G2/M-phase score per cell
        If regress=True, adata.X is modified to remove cell cycle signal.
        If layer is set, that layer is modified instead.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.cell_cycle_regression(adata)
    >>> # Cell cycle effect removed from adata.X
    >>> print(adata.obs['cell_cycle_phase'].value_counts())
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse

    from singlet._cell_cycle import G2M_GENES_HUMAN, S_GENES_HUMAN

    if s_genes is None:
        s_genes = S_GENES_HUMAN
    if g2m_genes is None:
        g2m_genes = G2M_GENES_HUMAN

    # Filter to genes present in data
    s_genes_present = [g for g in s_genes if g in adata.var_names]
    g2m_genes_present = [g for g in g2m_genes if g in adata.var_names]

    # Get expression matrix for scoring
    X = adata.layers[layer] if layer is not None else adata.X
    if issparse(X):
        X_dense = np.asarray(X.todense(), dtype=np.float64)
    else:
        X_dense = np.asarray(X, dtype=np.float64)

    # Score S phase: mean(S genes) - mean(control genes)
    gene_means = X_dense.mean(axis=0)
    overall_mean = gene_means.mean()

    if len(s_genes_present) > 0:
        s_idx = [list(adata.var_names).index(g) for g in s_genes_present]
        s_score = X_dense[:, s_idx].mean(axis=1) - overall_mean
    else:
        s_score = np.zeros(adata.n_obs)

    if len(g2m_genes_present) > 0:
        g2m_idx = [list(adata.var_names).index(g) for g in g2m_genes_present]
        g2m_score = X_dense[:, g2m_idx].mean(axis=1) - overall_mean
    else:
        g2m_score = np.zeros(adata.n_obs)

    # Assign phases
    phases = []
    for s_val, g2m_val in zip(s_score, g2m_score):
        if s_val > g2m_val and s_val > 0:
            phases.append("S")
        elif g2m_val > s_val and g2m_val > 0:
            phases.append("G2M")
        else:
            phases.append("G1")

    adata.obs["S_score"] = s_score.astype(np.float32)
    adata.obs["G2M_score"] = g2m_score.astype(np.float32)
    adata.obs["cell_cycle_phase"] = pd.Categorical(phases, categories=["G1", "S", "G2M"])

    # Regress out cell cycle scores
    if regress:
        # Check if there's any cell cycle signal to regress
        s_has_signal = np.any(s_score != 0)
        g2m_has_signal = np.any(g2m_score != 0)

        if not s_has_signal and not g2m_has_signal:
            # No cell cycle signal detected, nothing to regress
            return adata

        # Build design matrix with columns that have variance
        design_cols = [np.ones(adata.n_obs)]
        if s_has_signal and np.std(s_score) > 0:
            design_cols.append(s_score)
        if g2m_has_signal and np.std(g2m_score) > 0:
            design_cols.append(g2m_score)

        if len(design_cols) == 1:
            # Only intercept, nothing to regress
            return adata

        design = np.column_stack(design_cols)

        # Solve least squares: X = design @ beta + residuals
        # Use QR decomposition for numerical stability
        q_mat, r_mat = np.linalg.qr(design)

        # Check for rank deficiency
        diag_r = np.abs(np.diag(r_mat))
        if np.any(diag_r < 1e-10):
            # Design matrix is rank-deficient, use lstsq
            beta, _, _, _ = np.linalg.lstsq(design, X_dense, rcond=None)
        else:
            beta = np.linalg.solve(r_mat, q_mat.T @ X_dense)

        # Residuals (keep intercept, remove CC effects)
        # X_corrected = intercept + residuals = X - (CC_scores * beta_CC)
        cc_effect = design[:, 1:] @ beta[1:, :]
        X_corrected = X_dense - cc_effect

        # Store corrected expression
        if layer is not None:
            adata.layers[layer] = X_corrected.astype(np.float32)
        else:
            adata.X = X_corrected.astype(np.float32)

    return adata
