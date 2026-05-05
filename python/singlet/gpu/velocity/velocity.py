# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.velocity.velocity — GPU-native steady-state RNA velocity.

Underlying C++ cycle: cycle 15 (preprocess/velocity_prep.h —
``singlet::gpu::preprocess::velocity_prep_compute`` kernel).

Drop-in for ``scvelo.tl.velocity`` with ``mode='steady_state'``.  All
parameter names and defaults match the scvelo 0.3 API for this mode.

Steady-state velocity estimates the gene-specific kinetic ratio (gamma) by
linear regression of unspliced on spliced counts near the top/bottom
percentile extremes, then computes velocity as:
    velocity = unspliced − gamma × spliced

Only ``mode='steady_state'`` is implemented in cycle 15.  Dynamical and
stochastic modes require cycle 24+ (Cell2fate-style ODE inference).

Result location (matches scvelo)
---------------------------------
``adata.layers['velocity']``           — float32 velocity matrix (cells × genes).
``adata.var['velocity_gamma']``        — float32 per-gene slope (gamma).
``adata.var['velocity_r2']``           — float32 per-gene R² of the fit.
``adata.uns['velocity_params']``       — run parameters.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.velocity_prep_compute``
must be exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_layer(
    adata: "anndata.AnnData",
    layer_key: str,
    fallback_key: Optional[str] = None,
) -> object:
    """
    Return adata.layers[layer_key], with an optional fallback.
    Raises KeyError with a helpful message if absent.
    """
    if layer_key in adata.layers:
        return adata.layers[layer_key]
    if fallback_key is not None and fallback_key in adata.layers:
        return adata.layers[fallback_key]
    available = list(adata.layers.keys())
    raise KeyError(
        f"Layer '{layer_key}' not found in adata.layers.  "
        f"Available layers: {available}.  "
        "Run singlet.gpu.velocity.moments() first (which creates 'Ms', 'Mu') "
        "or set layer_spliced/layer_unspliced to the correct keys."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def velocity(
    adata: "anndata.AnnData",
    *,
    mode: str = "steady_state",
    fit_offset: bool = False,
    perc: List[int] = None,
    layer_spliced: str = "spliced",
    layer_unspliced: str = "unspliced",
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native steady-state RNA velocity estimation (cycle-15 kernel).

    Mirrors ``scvelo.tl.velocity`` for ``mode='steady_state'`` — parameter
    names and defaults are identical for this mode.

    For each gene, the gamma (kinetic ratio) is estimated by a robust linear
    regression of smoothed unspliced vs smoothed spliced counts in the
    extreme percentile bands (controlled by *perc*).  The velocity is then:

        ``velocity = Mu − gamma × Ms``

    where ``Ms`` / ``Mu`` are the smoothed spliced / unspliced layers
    (written by ``singlet.gpu.velocity.moments``).  If ``Ms`` / ``Mu`` are
    not present, the unsmoothed layers are used as a fallback.

    Parameters
    ----------
    adata : AnnData
        Must contain:

        - ``adata.layers['Ms']`` (or ``layer_spliced``)   — smoothed spliced.
        - ``adata.layers['Mu']`` (or ``layer_unspliced``) — smoothed unspliced.

        Produced by ``singlet.gpu.velocity.moments`` or ``scvelo.pp.moments``.
    mode : str, default ``"steady_state"``
        Velocity estimation mode.  Only ``'steady_state'`` is supported.
        ``'dynamical'`` and ``'stochastic'`` raise ``NotImplementedError``.
    fit_offset : bool, default False
        Whether to include an offset term (intercept) in the linear fit.
        ``False`` (default) forces the regression line through the origin
        (scvelo default for steady-state).
    perc : list of int or None, default None
        Percentile bounds [lower, upper] used to select the extreme cells
        for the gamma regression.  ``None`` → ``[5, 95]`` (scvelo default).
    layer_spliced : str, default ``"spliced"``
        Key for spliced (or smoothed spliced ``'Ms'``) layer.  ``'Ms'`` is
        preferred if present; *layer_spliced* is used as fallback.
    layer_unspliced : str, default ``"unspliced"``
        Key for unspliced (or smoothed unspliced ``'Mu'``) layer.
    copy : bool, default False
        Return a modified copy of *adata*.  When ``False`` (default),
        *adata* is modified in-place and ``None`` is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.velocity_prep_compute`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    NotImplementedError
        If *mode* is not ``'steady_state'``.
    KeyError
        If the required layers are missing.

    Notes
    -----
    **Result location** (matches scvelo):

    - ``adata.layers['velocity']``      — float32 (cells × genes).
    - ``adata.var['velocity_gamma']``   — float32 per-gene slope.
    - ``adata.var['velocity_r2']``      — float32 per-gene R².
    - ``adata.uns['velocity_params']``  — run parameters dict.

    **Mode limitations**: only ``mode='steady_state'`` is implemented.
    Dynamical mode (Cell2fate ODE inference) and stochastic mode are
    deferred to cycle 24+.

    Examples
    --------
    Default steady-state velocity::

        import singlet.gpu.velocity as sgv
        sgv.moments(adata)
        sgv.velocity(adata)
        print(adata.layers['velocity'].shape)  # (n_cells, n_genes)

    With offset term::

        sgv.velocity(adata, fit_offset=True)
    """
    if mode != "steady_state":
        raise NotImplementedError(
            f"mode='{mode}' is not yet implemented.  "
            "Only 'steady_state' is supported in cycle 15.  "
            "Dynamical and stochastic modes are deferred to cycle 24+."
        )

    if perc is None:
        perc = [5, 95]
    if len(perc) != 2 or not (0 <= perc[0] < perc[1] <= 100):
        raise ValueError(
            f"perc must be [lower, upper] with 0 ≤ lower < upper ≤ 100, "
            f"got {perc!r}."
        )

    import singlet.gpu._core as _core

    if not hasattr(_core, "velocity_prep_compute"):
        raise AttributeError(
            "_core.velocity_prep_compute is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose velocity_prep_compute "
            "before singlet.gpu.velocity.velocity() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    # Prefer smoothed layers (Ms/Mu) produced by moments(); fall back to raw.
    spliced_mat   = _get_layer(working, "Ms",   fallback_key=layer_spliced)
    unspliced_mat = _get_layer(working, "Mu",   fallback_key=layer_unspliced)

    # C++ kernel: velocity_prep_compute(spliced, unspliced,
    #                                    perc_lower, perc_upper, fit_offset)
    # Returns struct with fields:
    #   .velocity   — (n_cells × n_genes) float32 velocity matrix
    #   .gamma      — (n_genes,) float32 per-gene kinetic ratio
    #   .r2         — (n_genes,) float32 per-gene R²
    result = _core.velocity_prep_compute(
        spliced_mat,
        unspliced_mat,
        int(perc[0]),
        int(perc[1]),
        bool(fit_offset),
    )

    working.layers["velocity"] = np.asarray(result.velocity, dtype=np.float32)
    working.var["velocity_gamma"] = np.asarray(result.gamma, dtype=np.float32)
    working.var["velocity_r2"]    = np.asarray(result.r2,    dtype=np.float32)
    working.uns["velocity_params"] = {
        "mode": "steady_state",
        "fit_offset": fit_offset,
        "perc": list(perc),
        "layer_spliced": layer_spliced,
        "layer_unspliced": layer_unspliced,
    }

    return working if copy else None


__all__ = ["velocity"]
