# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu — GPU-native single-cell analysis.

Public API
----------
DeviceCsc       -- CSC sparse matrix on the GPU.
Metadata        -- GEO metadata embedded in a .1pz file.
PzDeviceMatrix  -- Combined result of load_pz (DeviceCsc + Metadata).
load_pz         -- Load a .1pz file to GPU device (zero-copy).
io.read_anndata -- High-level helper: .1pz directory → AnnData (cupy-backed).
io.to_anndata   -- Wrap a DeviceCsc + Metadata into an AnnData.

Device utilities
----------------
device_count()     -- Number of visible CUDA devices.
cuda_device_name() -- Name of a specific CUDA device.
current_device()   -- Index of the currently active CUDA device.
set_device()       -- Switch the active CUDA device.
"""

from singlet_gpu.version import __version__

# CYCLE-110: Import submodules at top level so users can do
#   import singlet_gpu as sg
#   sg.qc.calculate_qc_metrics(adata, ...)
# without having to know the full `from singlet_gpu.qc import ...` form.
# Wrapped in try/except so tooling envs missing optional deps still see the
# names (will raise meaningful AttributeError on use).
def _import_submodule(name):
    try:
        return __import__(f"singlet_gpu.{name}", fromlist=["*"])
    except ImportError:
        return None

preprocess = _import_submodule("preprocess")
reduce     = _import_submodule("reduce")
qc         = _import_submodule("qc")
pp         = _import_submodule("pp")
tools      = _import_submodule("tools")
streaming  = _import_submodule("streaming")
io         = _import_submodule("io")  # io/ subdir (read_anndata, to_anndata helpers)
# Wrapper modules surfaced to the public namespace so tests / users can call
# `singlet_gpu.lineage.detect_clones(...)` etc without explicit deep imports.
# (Each submodule remains optional — _import_submodule returns None on
# ImportError so the package still imports if a sub-tree is missing deps.)
enrich       = _import_submodule("enrich")
enrichment   = _import_submodule("enrichment")  # legacy/alt name — separate dir
integrate    = _import_submodule("integrate")
embed        = _import_submodule("embed")
lineage      = _import_submodule("lineage")
fate         = _import_submodule("fate")
de           = _import_submodule("de")
spatial      = _import_submodule("spatial")
generative   = _import_submodule("generative")
perturbation = _import_submodule("perturbation")
abundance    = _import_submodule("abundance")
ase          = _import_submodule("ase")
atac         = _import_submodule("atac")
cna          = _import_submodule("cna")
comm         = _import_submodule("comm")
disease      = _import_submodule("disease")
eqtl         = _import_submodule("eqtl")
grn          = _import_submodule("grn")
network      = _import_submodule("network")
del _import_submodule

# Core C++ extension — compiled by CMake/pybind11.
# ImportError here usually means the wheel was not built yet
# (no nvcc on login nodes); run: pip install -e python/ on a GPU node.
try:
    from singlet_gpu._core import (
        DeviceCsc,
        Metadata,
        PzDeviceMatrix,
        load_pz,
        device_count,
        cuda_device_name,
        current_device,
        set_device,
    )
except ImportError as _e:
    import warnings
    warnings.warn(
        f"singlet_gpu._core could not be imported ({_e}). "
        "The C++ extension must be compiled on a CUDA-capable node before use. "
        "Run: pip install -e singlet-gpu/python/ on a GPU node.",
        ImportWarning,
        stacklevel=2,
    )
    # Provide stub names so that `import singlet_gpu` succeeds for tooling
    # (linters, doc builders) on nodes without CUDA.
    DeviceCsc = None       # type: ignore[assignment]
    Metadata = None        # type: ignore[assignment]
    PzDeviceMatrix = None  # type: ignore[assignment]
    load_pz = None         # type: ignore[assignment]
    device_count = None    # type: ignore[assignment]
    cuda_device_name = None  # type: ignore[assignment]
    current_device = None  # type: ignore[assignment]
    set_device = None      # type: ignore[assignment]

__all__ = [
    "__version__",
    "DeviceCsc",
    "Metadata",
    "PzDeviceMatrix",
    "load_pz",
    "device_count",
    "cuda_device_name",
    "current_device",
    "set_device",
]
