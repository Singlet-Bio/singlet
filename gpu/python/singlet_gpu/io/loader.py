# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.io.loader — high-level .1pz ↔ AnnData I/O wrappers.

Underlying C++ cycle: cycle 1 (io/pz_device_loader.h — zero-copy .1pz →
DeviceCSC) and cycle 18 (pybind11 _core module exposing load_pz,
DeviceCsc, Metadata, PzDeviceMatrix).

The core .1pz → AnnData path was initially shipped in ``singlet_gpu/io.py``
as part of cycle 18.  This module supersedes that file by refactoring the
logic into the canonical ``singlet_gpu/io/`` package layout expected by
cycles 19+.  The old ``singlet_gpu/io.py`` is retained for backwards compat
and simply re-imports from here.

CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: the cycle 18 ``_core`` pybind11
module does not expose ``from_cupy_csr`` or ``to_cupy_csr`` helper functions.
Functions in this file that need to marshal a ``cupy.sparse.csr_matrix`` back
to a C++ ``DeviceCSC`` (e.g. for write-back) work around this by using the
``__cuda_array_interface__`` protocol directly or a host round-trip.  A
dedicated cycle should add ``_core.from_cupy_csr`` / ``_core.to_cupy_csr``
and remove the workaround.  The workaround path is explicitly marked.

API
---
read_pz_to_anndata(pz_dir, *, modality="exon", keep_host_pinned=False) -> AnnData
    Load a .1pz sample directory (or file) and return a GPU-resident AnnData.

write_anndata_to_pz(adata, pz_dir, *, modality="exon") -> None
    Round-trip: write a (modified) AnnData back to a .1pz file.
    NOTE: requires host copy in the absence of ``_core.from_cupy_csr``.
    See CYCLE-19-FOLLOWUP note above.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import anndata

# ---------------------------------------------------------------------------
# Modality → canonical filename stem mapping.
# Mirrors the singlify output artifact table.
# ---------------------------------------------------------------------------
_MODALITY_TO_STEM: dict[str, str] = {
    "exon":      "exon_counts",
    "intron":    "intron_counts",
    "gene":      "gene_counts",
    "sj":        "splice_junctions",
    "snp_ad":    "snp_ad",
    "snp_dp":    "snp_dp",
    "mt":        "mt_alleles",
    "adt":       "adt",
    "fragments": "fragments",
}


def _resolve_pz_path(pz_dir: Union[str, Path], modality: str) -> str:
    """
    Resolve a ``.1pz`` file path from a file or sample directory.

    Parameters
    ----------
    pz_dir:
        Direct path to a ``.1pz`` file, or a sample directory containing
        ``{stem}.1pz`` matching *modality*.
    modality:
        Modality key (see ``_MODALITY_TO_STEM``).

    Returns
    -------
    str
        Absolute path to the resolved ``.1pz`` file.

    Raises
    ------
    ValueError
        If *modality* is not recognised.
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    p = Path(pz_dir)
    if p.is_file():
        return str(p)
    if p.is_dir():
        stem = _MODALITY_TO_STEM.get(modality)
        if stem is None:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Recognised values: {sorted(_MODALITY_TO_STEM)}"
            )
        candidate = p / f"{stem}.1pz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected {candidate} for modality='{modality}' "
                f"but file not found in {p}"
            )
        return str(candidate)
    raise FileNotFoundError(f"Path does not exist: {pz_dir}")


def _build_anndata(device_csc, metadata) -> "anndata.AnnData":
    """
    Wrap a :class:`~singlet_gpu.DeviceCsc` + :class:`~singlet_gpu.Metadata`
    into an AnnData with a zero-copy ``cupy.sparse.csr_matrix`` at ``.X``.

    Parameters
    ----------
    device_csc : singlet_gpu.DeviceCsc
        Device CSC matrix (genes × cells).
    metadata : singlet_gpu.Metadata
        Embedded GEO metadata.

    Returns
    -------
    anndata.AnnData
        ``adata.X``  — cupy.sparse.csr_matrix (cells × genes), zero-copy.
        ``adata.obs`` — DataFrame indexed by cell barcodes.
        ``adata.var`` — DataFrame indexed by gene names.
        ``adata.uns['singlify']`` — GEO + provenance dict.
    """
    try:
        import anndata as ad
        import pandas as pd
        import cupy as cp
        # cupy >= 14 removed `cupy.sparse`; the new home is `cupyx.scipy.sparse`
        # which exposes the same csc_matrix / csr_matrix API.
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
    except ImportError as exc:
        raise ImportError(
            "read_pz_to_anndata requires anndata, cupy, and pandas. "
            "Install: pip install 'singlet-gpu[anndata]' cupy-cuda12x"
        ) from exc

    rows = device_csc.rows
    cols = device_csc.cols

    # Build zero-copy cupy 1-D arrays via __cuda_array_interface__ dicts
    # returned by the DeviceCsc view properties.  cupy.asarray() accepts
    # any object with __cuda_array_interface__ — zero device↔host traffic.
    cu_data    = cp.asarray(device_csc.data_view)
    cu_indices = cp.asarray(device_csc.indices_view)
    cu_indptr  = cp.asarray(device_csc.indptr_view)

    # .1pz stores (genes × cells) CSC.  AnnData convention is (cells × genes)
    # with X in CSR layout.  .T on a csc_matrix returns a view — no data copy.
    genes_x_cells_csc = csp.csc_matrix(
        (cu_data, cu_indices, cu_indptr), shape=(rows, cols)
    )
    cells_x_genes_csr = genes_x_cells_csc.T.tocsr()

    obs_names = list(metadata.colnames) if metadata.colnames else [str(i) for i in range(cols)]
    var_names = list(metadata.rownames) if metadata.rownames else [str(i) for i in range(rows)]

    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=var_names)

    singlify_meta = dict(metadata.to_dict())
    singlify_meta["n_genes"] = rows
    singlify_meta["n_cells"] = cols

    adata = ad.AnnData(
        X=cells_x_genes_csr,
        obs=obs,
        var=var,
        uns={"singlify": singlify_meta},
    )
    # Keep device allocation alive: store a reference in uns so Python GC
    # does not collect it while adata.X is in use.
    adata.uns["_singlet_gpu_device_ref"] = device_csc
    return adata


def read_pz_to_anndata(
    pz_dir: Union[str, "os.PathLike[str]"],
    *,
    modality: str = "exon",
    keep_host_pinned: bool = False,
) -> "anndata.AnnData":
    """
    Load a ``.1pz`` sample directory (or file) and return a GPU-resident AnnData.

    ``adata.X`` is a ``cupy.sparse.csr_matrix`` backed by device memory —
    no data is ever copied to the host during load.

    Parameters
    ----------
    pz_dir : str or path-like
        Path to a sample directory (containing ``{modality_stem}.1pz``)
        or directly to a ``.1pz`` file.
    modality : str, default ``"exon"``
        Which modality to load when *pz_dir* is a directory.
        Choices: ``"exon"``, ``"intron"``, ``"gene"``, ``"sj"``,
        ``"snp_ad"``, ``"snp_dp"``, ``"mt"``, ``"adt"``, ``"fragments"``.
    keep_host_pinned : bool, default ``False``
        Retain pinned host copies after device transfer.  Enables use of
        host-pointer SVD adapters without a second copy.

    Returns
    -------
    anndata.AnnData
        ``adata.X`` — cupy.sparse.csr_matrix (cells × genes), zero-copy.
        ``adata.uns['singlify']`` — GEO + provenance dict.

    Examples
    --------
    Load by sample directory::

        adata = read_pz_to_anndata("quant/scrna/GSE127/GSE127918/GSM4037629/")

    Load unspliced counts::

        adata_u = read_pz_to_anndata(sample_dir, modality="intron")
    """
    try:
        from singlet_gpu import load_pz as _load_pz
    except (ImportError, TypeError) as exc:
        raise ImportError(
            "singlet_gpu._core is not available — the C++ extension must be "
            "compiled on a CUDA-capable node.  Run: pip install -e python/ "
            "on a GPU node."
        ) from exc

    path = _resolve_pz_path(pz_dir, modality)
    result = _load_pz(path, keep_host_pinned=keep_host_pinned)
    return _build_anndata(result.mat, result.meta)


def write_anndata_to_pz(
    adata: "anndata.AnnData",
    pz_dir: Union[str, "os.PathLike[str]"],
    *,
    modality: str = "exon",
) -> None:
    """
    Write an AnnData back to a ``.1pz`` file (round-trip support).

    .. warning::
        This function requires a host copy of ``adata.X`` in the current
        implementation because ``_core.from_cupy_csr`` is not yet exposed by
        the cycle-18 binding.  A future cycle will add that binding and remove
        the host copy.  See ``CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData whose ``X`` is a ``cupy.sparse.csr_matrix`` (cells × genes).
    pz_dir : str or path-like
        Target directory.  Created if it does not exist.
    modality : str, default ``"exon"``
        Used to derive the output filename ``{stem}.1pz``.

    Raises
    ------
    NotImplementedError
        Always raised in cycle 19.  The round-trip write path requires
        ``_core.from_cupy_csr`` which is not yet implemented.  This stub
        documents the intended API surface.
    """
    # CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: implement once _core exposes
    # from_cupy_csr(data_ptr, indices_ptr, indptr_ptr, rows, cols, nnz) ->
    # DeviceCsc and a matching pz_write() function.
    stem = _MODALITY_TO_STEM.get(modality)
    if stem is None:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Recognised values: {sorted(_MODALITY_TO_STEM)}"
        )
    out_path = Path(pz_dir) / f"{stem}.1pz"
    raise NotImplementedError(
        f"write_anndata_to_pz → {out_path}: not yet implemented.  "
        "Waiting on _core.from_cupy_csr / pz_write binding "
        "(CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE)."
    )


__all__ = ["read_pz_to_anndata", "write_anndata_to_pz"]
