# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.io.donor — helpers for loading singlify donor assignment outputs.

Singlify writes ``donor_assignments.tsv`` for samples processed with
``--snps`` + ``--pipeline``.  This module provides a convenience loader
that returns a ``pd.Series`` indexed by cell barcode, ready to assign to
``adata.obs['donor_id']`` before calling
``singlet_gpu.de.pseudobulk_de``.

TSV schema (singlify donor_assignments.tsv)
-------------------------------------------
barcode       — cell barcode (index)
donor_id      — assigned donor label (str, e.g. "donor_0", "donor_1", …)
prob_max      — posterior probability for the top donor call (float32)
prob_doublet  — posterior doublet probability (float32)

API
---
load_donor_assignments(tsv_path) -> pd.Series
    Load donor_assignments.tsv and return a Series indexed by barcode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def load_donor_assignments(
    tsv_path: Union[str, "Path"],
    *,
    barcode_col: str = "barcode",
    donor_col: str = "donor_id",
    sep: str = "\t",
    min_prob: Optional[float] = None,
    exclude_doublets: bool = False,
    doublet_col: str = "prob_doublet",
    doublet_threshold: float = 0.5,
) -> "pd.Series":
    """
    Load singlify's ``donor_assignments.tsv`` into a ``pd.Series``.

    Returns a ``pd.Series`` indexed by cell barcode with values equal to
    the donor label.  Cells that fail quality filters are returned with
    ``NaN`` so that the Series can be used directly with
    ``adata.obs[sample_col]`` and downstream ``pseudobulk_de``.

    Parameters
    ----------
    tsv_path : str or Path
        Path to the ``donor_assignments.tsv`` file produced by singlify.
    barcode_col : str, default ``"barcode"``
        Column name for cell barcodes.
    donor_col : str, default ``"donor_id"``
        Column name for donor labels.
    sep : str, default ``"\t"``
        Field separator.
    min_prob : float or None, default None
        Minimum posterior probability (``prob_max``) for a cell to be
        assigned a donor label.  Cells below this threshold are set to
        ``NaN``.  ``None`` → no filtering.
    exclude_doublets : bool, default False
        If ``True``, cells with ``prob_doublet ≥ doublet_threshold`` are
        set to ``NaN``.
    doublet_col : str, default ``"prob_doublet"``
        Column name for the doublet probability.  Used when
        *exclude_doublets* is ``True``.
    doublet_threshold : float, default 0.5
        Posterior doublet probability threshold for exclusion when
        *exclude_doublets* is ``True``.

    Returns
    -------
    pd.Series
        Index: cell barcodes (str).
        Values: donor labels (str), with ``NaN`` for excluded cells.
        Series name: value of *donor_col*.

    Raises
    ------
    FileNotFoundError
        If *tsv_path* does not exist.
    KeyError
        If *barcode_col* or *donor_col* is not present in the TSV.

    Examples
    --------
    Basic load::

        from singlet_gpu.io.donor import load_donor_assignments
        donor_series = load_donor_assignments(
            "/path/to/GSM4037629/donor_assignments.tsv"
        )
        adata.obs['donor_id'] = donor_series

    With doublet filtering::

        donor_series = load_donor_assignments(tsv_path, exclude_doublets=True)

    With minimum confidence::

        donor_series = load_donor_assignments(tsv_path, min_prob=0.9)
    """
    import pandas as pd

    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"donor_assignments.tsv not found: {path}.  "
            "Singlify produces this file when run with --snps + --pipeline."
        )

    df = pd.read_csv(str(path), sep=sep)

    missing_cols = [c for c in (barcode_col, donor_col) if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Required columns {missing_cols} not found in {path.name}.  "
            f"Available columns: {list(df.columns)}"
        )

    # Set barcode as index.
    df = df.set_index(barcode_col)

    # Extract donor Series.
    series: pd.Series = df[donor_col].copy()
    series.name = donor_col

    # Apply min_prob filter.
    if min_prob is not None:
        prob_col = "prob_max"
        if prob_col not in df.columns:
            import warnings
            warnings.warn(
                f"min_prob filtering requested but column '{prob_col}' not "
                f"found in {path.name}.  Skipping min_prob filter.",
                UserWarning,
                stacklevel=2,
            )
        else:
            mask_low_prob = df[prob_col] < float(min_prob)
            series[mask_low_prob] = None  # type: ignore[call-overload]

    # Apply doublet filter.
    if exclude_doublets:
        if doublet_col not in df.columns:
            import warnings
            warnings.warn(
                f"exclude_doublets=True but column '{doublet_col}' not found "
                f"in {path.name}.  Skipping doublet filter.",
                UserWarning,
                stacklevel=2,
            )
        else:
            mask_doublet = df[doublet_col] >= float(doublet_threshold)
            series[mask_doublet] = None  # type: ignore[call-overload]

    return series


__all__ = ["load_donor_assignments"]
