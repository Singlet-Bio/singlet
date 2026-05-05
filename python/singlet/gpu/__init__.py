"""singlet.gpu — GPU-accelerated single-cell analysis.

Requires: pip install singlet[gpu]

Submodules:
    singlet.gpu.pp          Preprocessing (normalize, hvg, scale)
    singlet.gpu.reduce      Dimensionality reduction (PCA, NMF, SVD)
    singlet.gpu.tools       Analysis tools (neighbors, leiden, umap)
    singlet.gpu.de          Differential expression
    singlet.gpu.enrich      Gene set enrichment
    singlet.gpu.spatial     Spatial analysis
    singlet.gpu.velocity    RNA velocity
    singlet.gpu.integrate   Batch integration
"""
try:
    import cupy as _cp
except ImportError:
    raise ImportError(
        "singlet.gpu requires CuPy with CUDA 12+. Install with: pip install singlet[gpu]"
    )

from singlet.gpu._core import load_pz, DeviceCsc, Metadata, device_count, set_device

__all__ = ["load_pz", "DeviceCsc", "Metadata", "device_count", "set_device"]
