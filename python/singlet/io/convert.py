"""Format conversions: .1pz/.spz <-> h5ad, zarr, TileDB-SOMA, MTX, CSC."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata
    import scipy.sparse

# ─── Export from AnnData ─────────────────────────────────────────────────────


def to_h5ad(adata, path: str | Path, *, compression: str = "gzip") -> None:
    """Write AnnData to HDF5 (.h5ad) format.

    Parameters
    ----------
    adata : anndata.AnnData
    path : str or Path
    compression : str
        HDF5 compression filter. Default ``"gzip"``.
    """
    adata.write_h5ad(Path(path).expanduser(), compression=compression)


def to_zarr(adata, path: str | Path, *, chunks: Optional[tuple] = None) -> None:
    """Write AnnData to Zarr store.

    Parameters
    ----------
    adata : anndata.AnnData
    path : str or Path
        Zarr store directory.
    chunks : tuple, optional
        Chunk shape for the X matrix.
    """
    adata.write_zarr(Path(path).expanduser(), chunks=chunks)


def to_csc(adata, *, layer: Optional[str] = None) -> scipy.sparse.csc_matrix:
    """Extract the count matrix as a scipy CSC sparse matrix.

    Parameters
    ----------
    adata : anndata.AnnData
    layer : str, optional
        Use this layer instead of X.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse matrix (cells × genes).
    """
    import scipy.sparse as sp

    mat = adata.layers[layer] if layer else adata.X
    if sp.issparse(mat):
        return mat.tocsc()
    return sp.csc_matrix(mat)


def to_tiledb(adata, uri: str, *, measurement_name: str = "RNA") -> None:
    """Write AnnData to a TileDB-SOMA experiment.

    Requires ``tiledbsoma`` to be installed.

    Parameters
    ----------
    adata : anndata.AnnData
    uri : str
        TileDB-SOMA URI (local path or cloud URI).
    measurement_name : str
        SOMA measurement name. Default ``"RNA"``.
    """
    import tiledbsoma
    import tiledbsoma.io

    tiledbsoma.io.from_anndata(
        uri,
        adata,
        measurement_name=measurement_name,
    )


def to_mtx(adata, directory: str | Path, *, layer: Optional[str] = None) -> None:
    """Write AnnData to Market Exchange (MTX) format.

    Creates ``matrix.mtx.gz``, ``barcodes.tsv.gz``, ``features.tsv.gz``
    in the target directory.

    Parameters
    ----------
    adata : anndata.AnnData
    directory : str or Path
    layer : str, optional
        Use this layer instead of X.
    """
    import gzip

    import scipy.io
    import scipy.sparse as sp

    out = Path(directory).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)

    # Write matrix
    with gzip.open(out / "matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, mat.T)  # MTX convention: genes × cells

    # Write barcodes
    with gzip.open(out / "barcodes.tsv.gz", "wt") as f:
        for bc in adata.obs_names:
            f.write(f"{bc}\n")

    # Write features/genes
    with gzip.open(out / "features.tsv.gz", "wt") as f:
        for gene in adata.var_names:
            f.write(f"{gene}\t{gene}\tGene Expression\n")


# ─── Import to AnnData ──────────────────────────────────────────────────────


def from_h5ad(path: str | Path, *, backed: str | None = None) -> anndata.AnnData:
    """Read an .h5ad file into AnnData.

    Parameters
    ----------
    path : str or Path
    backed : str, optional
        ``"r"`` for read-only backed mode, ``None`` for in-memory.

    Returns
    -------
    anndata.AnnData
    """
    import anndata as ad

    return ad.read_h5ad(Path(path).expanduser(), backed=backed)  # type: ignore[arg-type]


def from_zarr(path: str | Path) -> anndata.AnnData:
    """Read a Zarr store into AnnData.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    anndata.AnnData
    """
    import anndata as ad

    return ad.read_zarr(Path(path).expanduser())


def from_tiledb(uri: str, *, measurement_name: str = "RNA") -> anndata.AnnData:
    """Read a TileDB-SOMA experiment into AnnData.

    Requires ``tiledbsoma`` to be installed.

    Parameters
    ----------
    uri : str
        TileDB-SOMA URI.
    measurement_name : str
        SOMA measurement name.

    Returns
    -------
    anndata.AnnData
    """
    import tiledbsoma
    import tiledbsoma.io

    exp = tiledbsoma.Experiment.open(uri)
    return tiledbsoma.io.to_anndata(exp, measurement_name)


def from_mtx(directory: str | Path) -> anndata.AnnData:
    """Read Market Exchange (MTX) format into AnnData.

    Expects ``matrix.mtx`` or ``matrix.mtx.gz``, plus optional
    ``barcodes.tsv(.gz)`` and ``features.tsv(.gz)`` / ``genes.tsv(.gz)``.

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
    anndata.AnnData
    """
    import anndata as ad
    import pandas as pd
    import scipy.io

    d = Path(directory).expanduser()

    # Find matrix file
    for name in ["matrix.mtx.gz", "matrix.mtx"]:
        if (d / name).exists():
            mat = scipy.io.mmread(d / name).T.tocsr()  # genes×cells → cells×genes
            break
    else:
        raise FileNotFoundError(f"No matrix.mtx(.gz) found in {d}")

    adata = ad.AnnData(X=mat)

    # Barcodes
    for name in ["barcodes.tsv.gz", "barcodes.tsv"]:
        if (d / name).exists():
            barcodes = pd.read_csv(d / name, header=None, sep="\t")[0].values
            adata.obs_names = pd.Index(barcodes)
            break

    # Features/genes
    for name in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        if (d / name).exists():
            features = pd.read_csv(d / name, header=None, sep="\t")
            adata.var_names = pd.Index(features[0].values)
            if features.shape[1] > 1:
                adata.var["gene_name"] = features[1].values
            break

    return adata


# ─── .1pz / .spz round-trip convenience ─────────────────────────────────────


def pz_to_h5ad(pz_path: str | Path, h5ad_path: str | Path, **kwargs) -> None:
    """Convert a .1pz file to .h5ad."""
    from singlet._io import read_1pz

    adata = read_1pz(pz_path)
    to_h5ad(adata, h5ad_path, **kwargs)


def h5ad_to_pz(h5ad_path: str | Path, pz_path: str | Path, **kwargs) -> None:
    """Convert a .h5ad file to .1pz."""
    from singlet._io import write_1pz

    adata = from_h5ad(h5ad_path)
    write_1pz(adata, pz_path, **kwargs)


def spz_to_h5ad(spz_path: str | Path, h5ad_path: str | Path, **kwargs) -> None:
    """Convert a .spz file to .h5ad (legacy compat)."""
    from singlet._io import read_spz

    adata = read_spz(spz_path)
    to_h5ad(adata, h5ad_path, **kwargs)


def h5ad_to_spz(h5ad_path: str | Path, spz_path: str | Path, **kwargs) -> None:
    """Convert a .h5ad file to .spz (legacy compat)."""
    from singlet._io import write_spz

    adata = from_h5ad(h5ad_path)
    write_spz(adata, spz_path, **kwargs)
