"""Metadata extraction from R objects (Seurat / SingleCellExperiment).

Uses `rpy2 <https://rpy2.github.io/>`_ to call R's ``readRDS()`` and
extract cell-level metadata from:

- **Seurat objects** — ``object@meta.data``
- **SingleCellExperiment** — ``colData(object)``
- **Plain data.frame / matrix** — coerced with ``as.data.frame()``

All metadata columns are preserved — no filtering.

Requirements:
    - ``rpy2`` Python package
    - ``R`` installation with ``Seurat`` and ``SingleCellExperiment`` packages
    - For Seurat v5, ``SeuratObject`` package is also required
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Cache R initialisation state
_R_INITIALISED = False


def _init_r():
    """Lazily initialise rpy2 and load required R packages."""
    global _R_INITIALISED
    if _R_INITIALISED:
        return

    import rpy2.robjects as ro
    from rpy2.robjects import r

    # Suppress R startup messages
    r("suppressPackageStartupMessages(library(methods))")

    _R_INITIALISED = True


def _r_to_dataframe(r_df) -> pd.DataFrame:
    """Convert an R data.frame to a pandas DataFrame.

    Handles factors, character vectors, numeric, logical, and integer
    columns.  Returns all columns — no filtering.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import r

    colnames = list(r("colnames")(r_df))
    rownames = list(r("rownames")(r_df))

    data = {}
    for col in colnames:
        vec = r_df.rx2(col)
        # Factors → character
        if r("is.factor")(vec)[0]:
            vec = r("as.character")(vec)
        data[col] = list(vec)

    df = pd.DataFrame(data)
    df["barcode"] = rownames
    return df


def extract_metadata_from_rds(path: Path) -> pd.DataFrame:
    """Extract all cell-level metadata from an RDS file.

    Detects the object class (Seurat, SingleCellExperiment, or
    data.frame) and extracts the appropriate metadata slot.
    Returns **all** columns from the metadata — cell type, clusters,
    QC metrics, donor IDs, condition labels, UMAP coordinates, etc.

    Args:
        path: Path to a ``.rds`` or ``.rds.gz`` file.

    Returns:
        DataFrame with a ``barcode`` column and all metadata columns.
        Empty DataFrame on failure or if rpy2 is not available.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import r
    except ImportError:
        logger.warning(
            "rpy2 not installed; cannot read RDS file %s. "
            "Install with: pip install rpy2",
            path.name,
        )
        return pd.DataFrame()

    try:
        _init_r()

        # Read the RDS file
        safe_path = str(path).replace("\\", "/")
        r(f'obj <- readRDS("{safe_path}")')

        # Detect class
        classes = list(r("class(obj)"))
        logger.info("RDS %s: R class(es) = %s", path.name, classes)

        if "Seurat" in classes or "seurat" in classes:
            return _extract_seurat(path)
        elif "SingleCellExperiment" in classes:
            return _extract_sce(path)
        elif "data.frame" in classes:
            return _extract_dataframe(path)
        elif "matrix" in classes or "dgCMatrix" in classes or "dgTMatrix" in classes:
            logger.info(
                "RDS %s: sparse/dense matrix — no cell metadata to extract",
                path.name,
            )
            return pd.DataFrame()
        else:
            # Try to coerce to data.frame
            return _extract_unknown(path, classes)

    except Exception as e:
        logger.warning("Failed to read RDS %s: %s", path.name, e)
        return pd.DataFrame()
    finally:
        # Clean up R memory
        try:
            r("if (exists('obj')) rm(obj); gc()")
        except Exception:
            pass


def _extract_seurat(path: Path) -> pd.DataFrame:
    """Extract meta.data from a Seurat object."""
    from rpy2.robjects import r

    try:
        # Try Seurat v5 first, then v3/v4
        r("""
            tryCatch({
                suppressPackageStartupMessages(library(SeuratObject))
            }, error = function(e) {
                suppressPackageStartupMessages(library(Seurat))
            })
        """)
    except Exception:
        pass

    # Get meta.data
    r("meta <- obj@meta.data")
    n_cells = r("nrow(meta)")[0]
    n_cols = r("ncol(meta)")[0]

    logger.info(
        "Seurat %s: %d cells × %d metadata columns",
        path.name, int(n_cells), int(n_cols),
    )

    meta_df = r("meta")
    df = _r_to_dataframe(meta_df)

    logger.info(
        "Seurat %s: extracted columns: %s",
        path.name, list(df.columns[:10]),
    )
    return df


def _extract_sce(path: Path) -> pd.DataFrame:
    """Extract colData from a SingleCellExperiment object."""
    from rpy2.robjects import r

    try:
        r("suppressPackageStartupMessages(library(SingleCellExperiment))")
    except Exception:
        pass

    r("meta <- as.data.frame(colData(obj))")
    n_cells = r("nrow(meta)")[0]
    n_cols = r("ncol(meta)")[0]

    logger.info(
        "SCE %s: %d cells × %d metadata columns",
        path.name, int(n_cells), int(n_cols),
    )

    meta_df = r("meta")
    df = _r_to_dataframe(meta_df)

    logger.info(
        "SCE %s: extracted columns: %s",
        path.name, list(df.columns[:10]),
    )
    return df


def _extract_dataframe(path: Path) -> pd.DataFrame:
    """Extract from a plain R data.frame."""
    from rpy2.robjects import r

    n_rows = r("nrow(obj)")[0]
    n_cols = r("ncol(obj)")[0]

    logger.info(
        "data.frame %s: %d rows × %d columns",
        path.name, int(n_rows), int(n_cols),
    )

    df = _r_to_dataframe(r("obj"))
    return df


def _extract_unknown(path: Path, classes: list) -> pd.DataFrame:
    """Attempt to extract metadata from an unknown R object type."""
    from rpy2.robjects import r

    logger.info(
        "RDS %s: unknown class %s, attempting as.data.frame()",
        path.name, classes,
    )

    try:
        r("meta <- as.data.frame(obj)")
        n_rows = r("nrow(meta)")[0]
        if n_rows > 0:
            df = _r_to_dataframe(r("meta"))
            logger.info(
                "RDS %s: coerced successfully, %d rows × %d columns",
                path.name, int(n_rows), len(df.columns),
            )
            return df
    except Exception:
        pass

    # Try extracting @meta.data or metadata() generically
    for slot_expr in ["obj@meta.data", "obj@metadata", "metadata(obj)"]:
        try:
            r(f"meta <- as.data.frame({slot_expr})")
            n_rows = r("nrow(meta)")[0]
            if n_rows > 0:
                df = _r_to_dataframe(r("meta"))
                logger.info(
                    "RDS %s: extracted via %s, %d rows",
                    path.name, slot_expr, int(n_rows),
                )
                return df
        except Exception:
            continue

    logger.warning("RDS %s: could not extract metadata from class %s", path.name, classes)
    return pd.DataFrame()


def extract_metadata_from_rdata(path: Path) -> pd.DataFrame:
    """Extract cell-level metadata from an RData/RDA/RObj file.

    RData files can contain multiple R objects.  This function loads
    the workspace and tries to find a Seurat, SingleCellExperiment,
    or data.frame object with cell-level metadata.

    Args:
        path: Path to a .RData, .Rdata, .rda, or .Robj file
            (optionally .gz compressed).

    Returns:
        DataFrame with a ``barcode`` column and all metadata columns.
        Empty DataFrame on failure.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import r
    except ImportError:
        logger.warning("rpy2 not installed; cannot read RData %s", path.name)
        return pd.DataFrame()

    try:
        _init_r()

        safe_path = str(path).replace("\\", "/")

        # Handle .gz decompression in R
        if safe_path.endswith(".gz"):
            r(f'con <- gzfile("{safe_path}")')
            r('loaded_names <- load(con)')
            r('close(con)')
        else:
            r(f'loaded_names <- load("{safe_path}")')

        obj_names = list(r("loaded_names"))
        logger.info("RData %s: loaded objects: %s", path.name, obj_names)

        if not obj_names:
            return pd.DataFrame()

        # Try each object, prefer Seurat > SCE > data.frame
        best_df = pd.DataFrame()
        best_score = -1

        for obj_name in obj_names:
            try:
                r(f"obj <- {obj_name}")
                classes = list(r("class(obj)"))
                logger.info("RData %s: object '%s' class = %s",
                            path.name, obj_name, classes)

                if "Seurat" in classes or "seurat" in classes:
                    df = _extract_seurat(path)
                    if len(df) > len(best_df):
                        best_df = df
                        best_score = 3
                elif "SingleCellExperiment" in classes and best_score < 3:
                    df = _extract_sce(path)
                    if len(df) > len(best_df):
                        best_df = df
                        best_score = 2
                elif "data.frame" in classes and best_score < 2:
                    n_rows = int(r("nrow(obj)")[0])
                    n_cols = int(r("ncol(obj)")[0])
                    if n_rows > 10 and n_cols < 500:
                        df = _r_to_dataframe(r("obj"))
                        if len(df) > len(best_df):
                            best_df = df
                            best_score = 1
            except Exception as e:
                logger.debug("RData %s: failed on '%s': %s",
                             path.name, obj_name, e)
                continue

        return best_df

    except Exception as e:
        logger.warning("Failed to read RData %s: %s", path.name, e)
        return pd.DataFrame()
    finally:
        try:
            for name in obj_names:
                r(f"if (exists('{name}')) rm({name})")
            r("if (exists('obj')) rm(obj); gc()")
        except Exception:
            pass
