---
feature: python_wrapper_foundation
roadmap_id: 18
module: python/src/_singlet_gpu_core.cpp + python/singlet_gpu/__init__.py
status: design
tolerance: round-trip CSC matrix bit-identical (numpy.array_equal on indptr/indices, fp32 equality on data) between AnnData.X and the device-loaded matrix
target_perf: load_pz wheel call ≤50ms overhead vs the C++ direct call (pybind11 overhead only, no real compute)
ooc_plan: same as cycle 2 loader (PzChunkIterator wrapped as a Python iterator yielding cupy.sparse chunks)
---

## Why this is the wrapper sprint kickoff

The original 17-feature roadmap is complete (cycle 17). Per absolute rule §⛔10, wrappers come AFTER the kernels reach the Pareto frontier. We are now in Phase A of the post-roadmap plan: cycles 18–22 are Python + R wrappers.

Cycle 18 is the **foundation** binding — exposing `core::DeviceCSC`, `load_pz`, the Metadata struct, and the AnnData/Scanpy interop layer. Cycles 19–21 then add per-feature thin bindings on top. Cycle 22 starts the R wrapper.

## Architecture

The Python wrapper is a `pybind11` C++ extension module compiled against singlet-gpu's header-only library. It produces a `singlet_gpu` Python package distributable via PyPI.

```
python/
├── pyproject.toml           # PEP 517 build, scikit-build-core backend
├── CMakeLists.txt           # delegates to singlet-gpu's main CMakeLists
├── src/
│   ├── _singlet_gpu_core.cpp        # pybind11 entry: PYBIND11_MODULE(_core, m) { ... }
│   ├── _bind_loader.hpp             # pz loader bindings
│   ├── _bind_metadata.hpp           # Metadata struct binding
│   ├── _bind_anndata.hpp            # AnnData interop helpers
│   └── _cupy_interop.hpp            # cuda-array-interface protocol implementation
├── singlet_gpu/
│   ├── __init__.py                  # public API: from singlet_gpu._core import ...
│   ├── io.py                        # high-level read_anndata, write_anndata
│   └── version.py
└── tests/
    └── test_core.py                 # pytest suite
```

## Key design decisions

### 1. cuda-array-interface for zero-copy interop with cupy

The `core::DeviceCSC` (= `factornet::gpu::SparseMatrixGPU<float>`) holds three device pointers: `indptr`, `indices`, `values`. We expose them via the **cuda-array-interface protocol** (`__cuda_array_interface__` attribute), which cupy / numba / pytorch all recognize.

This means:
```python
m = singlet_gpu.io.load_pz("counts.1pz")
csr = cupy.sparse.csr_matrix((m.data_view, m.indices_view, m.indptr_view), shape=(rows, cols))
```
Zero copy. The `.view` properties return Python objects with `__cuda_array_interface__` describing the device pointer + dtype + shape.

For AnnData users:
```python
adata = singlet_gpu.io.read_anndata("counts.1pz")
# adata.X is a cupy.sparse.csr_matrix view
```

### 2. Lifetime management

Device memory is owned by `core::DeviceMemory<T>` (RAII inside factornet). The pybind11 wrapper holds a `shared_ptr<DeviceCSC>` inside a Python class. When the Python object is GC'd, the shared_ptr drops, factornet frees.

The cupy view shares the same lifetime via a Python-level reference: the cupy.sparse object holds a reference to the singlet_gpu `DeviceCsc` object via `__cuda_array_interface__`'s `mask` field (or via a Python-side `ndarray.base` chain).

### 3. AnnData interop

Two helpers in `singlet_gpu/io.py`:

```python
def read_anndata(pz_path: str, *, modality: str = "exon") -> AnnData:
    """Load a .1pz directory (which has multiple .1pz files for exon/intron/sj/etc.) and return an AnnData with X = the requested modality."""

def to_anndata(device_csc, metadata) -> AnnData:
    """Wrap a singlet_gpu device CSC + metadata into an AnnData."""
```

The AnnData's `obs` and `var` come from the Metadata struct's `colnames` (cells) and `rownames` (genes). The embedded GEO context (`gsm_id`, `gse_id`, `protocol`, etc.) populates `adata.uns['singlify']`.

### 4. No copy back to host unless explicitly requested

Per the §⛔ rules, we never copy device data to host implicitly. If the user wants a numpy/scipy view, they call `.to_host()` explicitly:

```python
csr_host = m.to_host()  # explicit; returns scipy.sparse.csr_matrix
```

The default `repr` says "DeviceCSC on cuda:0, shape (m, n), nnz N — call .to_host() for a numpy view."

### 5. PyPI distribution

- `pyproject.toml` uses `scikit-build-core` for CMake-based builds.
- Wheels built on `manylinux2014` runners with CUDA 12.4 statically linked.
- One wheel per CUDA major version (CUDA 12 only initially).
- Conda-forge feedstock auto-tracks PyPI.

## API surface (cycle 18 minimal)

```python
import singlet_gpu

# Core types
m: singlet_gpu.DeviceCsc = singlet_gpu.io.load_pz("path/to/exon_counts.1pz")
print(m)  # "DeviceCsc(rows=30000, cols=11560, nnz=27000000, dtype=float32, device='cuda:0')"

# Metadata
print(m.meta.gsm_id, m.meta.gse_id, m.meta.protocol)
print(m.meta.rownames[:5])  # first 5 gene names
print(m.meta.colnames[:5])  # first 5 cell barcodes

# Zero-copy cupy view
import cupy.sparse as csp
mat = csp.csr_matrix((m.data_view, m.indices_view, m.indptr_view), shape=(m.rows, m.cols))

# AnnData interop
adata = singlet_gpu.io.read_anndata("path/to/sample/")  # auto-detects modality
print(adata)
print(adata.uns['singlify']['gsm_id'])

# Explicit host copy
csr_host = m.to_host()
```

## Implementation notes (for cycle 18 kernel-dev)

- Header path: `python/src/_singlet_gpu_core.cpp` (~400 LOC) + `python/src/_bind_*.hpp` (~100 LOC each).
- Use `pybind11::class_<core::DeviceCsc, std::shared_ptr<core::DeviceCsc>>` so the Python object owns a shared_ptr.
- Implement `__cuda_array_interface__` as a Python-side @property on the binding.
- Test infra: `python/tests/test_core.py` with pytest fixtures loading GSM4037629.

## Constraints

- **NO host copy in the binding hot path.** The default load + view operation never moves device data to host. Only the explicit `.to_host()` call does.
- **Use shared_ptr for lifetime safety** so the Python GC interacts cleanly with factornet's RAII.
- pybind11 `PYBIND11_MODULE(_core, m)` declares the C extension; the `singlet_gpu` Python package then re-exports as `from singlet_gpu._core import DeviceCsc, load_pz, ...`.
- `pyproject.toml` declares the build dependency on `scikit-build-core`, runtime dep on `cupy-cuda12x>=13.0`, optional dep on `anndata`.

## Correctness test spec

Test file: `python/tests/test_core.py`.

Test cases:
1. **`test_load_pz_basic`**: load GSM4037629 exon_counts.1pz, confirm rows/cols/nnz match the C++ result.
2. **`test_metadata_fields`**: confirm `m.meta.gsm_id == "GSM4037629"`, etc.
3. **`test_cuda_array_interface_zero_copy`**: build a `cupy.sparse.csr_matrix` from the views, confirm `cupy.allclose` to a host-loaded scipy reference.
4. **`test_anndata_roundtrip`**: `read_anndata` → modify some cells → `to_anndata` → confirm round-trip.
5. **`test_to_host_explicit_copy`**: `.to_host()` returns a scipy.sparse.csr_matrix matching the device data.
6. **`test_lifetime_safety`**: GC the binding, confirm the cupy view is invalidated (or holds the underlying shared_ptr alive — pick one and document).

Tolerance: bit-identical (no math involved).

## Build / test

- Cycle 18 produces only source. The first GPU dispatch session must `pip install -e python/` and run `pytest python/tests/`.
- No nvcc available locally — same constraint as cycles 2–17.
