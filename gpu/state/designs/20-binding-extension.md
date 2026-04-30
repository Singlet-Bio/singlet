---
feature: cycle_18_binding_extension
roadmap_id: 20
module: python/src/_bind_kernels.hpp + python/src/_bind_cupy_ingest.hpp + extend python/src/_singlet_gpu_core.cpp
status: design
tolerance: each binding round-trip element-wise bit-identical to the C++ direct call (no math involved — pure marshaling)
target_perf: ≤1µs marshal overhead per binding call (pybind11's typical for small structs)
ooc_plan: not applicable (host-side bindings only)
---

## Why this exists

Cycle 19 finished writing all the Python wrapper modules (lognorm, hvg, svd, nmf), but cycle 18 only exposed the loader + Metadata + DeviceCsc class through pybind11 — NOT the per-kernel functions. The cycle 19 wrappers raise `AttributeError("_core has no attribute 'normalize_total'")` until we extend the binding.

This is NOT a re-do of cycle 18 — it's a focused extension that adds 13 missing entry points. After this cycle:
- `_core.from_cupy_csr(csr_matrix) -> DeviceCsc` (the missing converter)
- `_core.to_cupy_csr(device_csc) -> dict` (the missing reverse)
- `_core.normalize_total(device_csc, target_sum, ...) -> NormalizeResult`
- `_core.log1p(device_csc, base) -> None` (in-place)
- `_core.highly_variable_genes(device_csc, n_top_genes, flavor, ...) -> HvgResult`
- `_core.pca(device_csc, n_comps, zero_center, scale, rng_seed) -> PcaResult`
- `_core.svd_lanczos(device_csc, k, ...)` (and `irlba`, `randomized`, `krylov`, `deflation`)
- `_core.nmf(device_csc, k, ...)`
- `_core.nmf_chunked(loader, k, ...)`
- `_core.nmf_graph_factorize(...)`

## Architecture

We add ONE new binding header `_bind_kernels.hpp` and ONE small helper header `_bind_cupy_ingest.hpp`. The main `_singlet_gpu_core.cpp` adds these to its `PYBIND11_MODULE` block.

### `_bind_cupy_ingest.hpp` (~150 LOC)

Implements the **inverse** of cycle 18's `__cuda_array_interface__` export: ingestion of a Python `cupy.sparse.csr_matrix` into our `singlet_gpu::core::DeviceCSC`.

```cpp
namespace singlet_gpu::python {

// Accept any object exposing `.indptr`, `.indices`, `.data` with __cuda_array_interface__.
// Validate dtypes (int32, int32, float32). Build a DeviceCSC that holds non-owning
// device_span<> aliases — the lifetime is bound to the source Python object via a
// shared_ptr capsule.
core::DeviceCSC from_cupy_csr(pybind11::object csr_matrix);

// Reverse: build a Python dict that scipy / cupy can wrap. Returns:
//   { "indptr": cupy_ndarray_view, "indices": cupy_ndarray_view, "data": cupy_ndarray_view,
//     "shape": (rows, cols), "_owner": <wrapped DeviceCsc> }
pybind11::dict to_cupy_csr(const core::DeviceCSC& mat);

}  // namespace
```

The `from_cupy_csr` parses the `__cuda_array_interface__` dict on the input arrays, extracts the device pointer via `(int_ptr, read_only)` tuple, and constructs a `DeviceCSC` with non-owning spans. Lifetime: the Python source object MUST stay alive. We enforce this by adding the source object as a member of a `std::shared_ptr<pybind11::object>` lifetime anchor inside the binding's `DeviceCsc` Python wrapper class.

### `_bind_kernels.hpp` (~600 LOC)

One pybind11 function per kernel. Each is ~30-50 LOC (parse args, call C++ kernel, wrap result).

```cpp
namespace singlet_gpu::python {

void bind_kernels(pybind11::module_& m) {
    // -- preprocess --
    m.def("normalize_total", &py_normalize_total,
          py::arg("mat"), py::kw_only(),
          py::arg("target_sum") = 0.0f,
          py::arg("approximate_median") = false);

    m.def("log1p", &py_log1p,
          py::arg("mat"), py::kw_only(),
          py::arg("base") = 0.0f);  // 0 = natural log

    m.def("highly_variable_genes", &py_highly_variable_genes,
          py::arg("mat"), py::kw_only(),
          py::arg("n_top_genes") = 2000,
          py::arg("flavor") = std::string("seurat_v3"),
          py::arg("min_mean") = 0.0125f,
          py::arg("max_mean") = 3.0f,
          py::arg("pearson_theta") = 100.0f);

    // -- reduce/svd (5 backends + auto) --
    m.def("svd_auto_select", &py_svd_auto_select,
          py::arg("mat"), py::arg("k"), py::kw_only(),
          py::arg("center") = true, py::arg("scale") = false,
          py::arg("tol") = 1e-6f, py::arg("max_iter") = 100,
          py::arg("seed") = uint64_t{0});

    m.def("svd_lanczos", &py_svd_lanczos, /* same args */);
    m.def("svd_irlba", &py_svd_irlba, /* same args */);
    m.def("svd_randomized", &py_svd_randomized, /* same args */);
    m.def("svd_krylov", &py_svd_krylov, /* + constraint args */);
    m.def("svd_deflation", &py_svd_deflation, /* + constraint args */);

    // pca = thin alias to svd_auto_select with center=true
    m.def("pca", &py_pca, /* same args, defaults zero_center=true */);

    // -- reduce/nmf --
    m.def("nmf", &py_nmf,
          py::arg("mat"), py::arg("rank"), py::kw_only(),
          py::arg("loss") = std::string("MSE"),
          py::arg("solver_mode") = 3,
          py::arg("init_mode") = 2,
          py::arg("max_iter") = 100,
          py::arg("tol") = 1e-5f,
          py::arg("seed") = uint64_t{0});

    m.def("nmf_chunked", &py_nmf_chunked,
          py::arg("loader"), py::arg("rank"), py::kw_only(),
          /* same kwargs */);

    m.def("nmf_graph_factorize", &py_nmf_graph_factorize,
          py::arg("inputs"), py::arg("rank"), py::kw_only(),
          py::arg("topology") = std::string("shared_h"));
}

}  // namespace
```

Each `py_*` function:
1. Unpacks the `DeviceCsc` (or list of them for the graph variant).
2. Constructs the C++ `*Config` struct from the kwargs.
3. Calls the C++ kernel.
4. Wraps the C++ result struct into a pybind11-exposed Python class (or a plain dict for simple results).

Result wrapper classes (also added to `_singlet_gpu_core.cpp`):
- `class NormalizeResult { float target_used; cupy_view<float> size_factors; cupy_view<uint8> qc_mask; }`
- `class HvgResult { cupy_view<int> indices; cupy_view<float> scores; cupy_view<float> means; cupy_view<float> variances; }`
- `class SvdResult { cupy_view<float> U; cupy_view<float> d; cupy_view<float> V; int k_selected; }`
- `class NmfResult { cupy_view<float> W; cupy_view<float> d; cupy_view<float> H; int iterations; bool converged; }`

Each result class exposes its members via `__cuda_array_interface__` Python @property (same pattern as cycle 18 DeviceCsc.data_view).

### Update `_singlet_gpu_core.cpp`

Add to the `PYBIND11_MODULE(_core, m)` block:

```cpp
PYBIND11_MODULE(_core, m) {
    // ... existing cycle 18 bindings (DeviceCsc, Metadata, PzDeviceMatrix, load_pz) ...

    m.def("from_cupy_csr", &singlet_gpu::python::from_cupy_csr,
          py::arg("csr_matrix"));
    m.def("to_cupy_csr", &singlet_gpu::python::to_cupy_csr,
          py::arg("device_csc"));

    singlet_gpu::python::bind_kernels(m);

    // Result classes
    py::class_<NormalizeResult, std::shared_ptr<NormalizeResult>>(m, "NormalizeResult")
        .def_readonly("target_used", &NormalizeResult::target_used)
        .def_property_readonly("size_factors_view", &NormalizeResult::size_factors_view)
        .def_property_readonly("qc_mask_view", &NormalizeResult::qc_mask_view);
    // ... HvgResult, SvdResult, NmfResult similarly ...
}
```

## Constraints

- **NO host data copies** in any binding. The cupy ingest produces non-owning device pointers. The result classes return cupy views via `__cuda_array_interface__`.
- **Lifetime safety**: cupy.sparse.csr_matrix passed to `from_cupy_csr` must stay alive while the resulting `DeviceCsc` is used. We enforce via `std::shared_ptr<pybind11::object>` capsule.
- Each `py_*` function is < 60 LOC; pure marshaling.
- pybind11 ≥ 2.11.

## Build / test

- No nvcc on this login node. Source-only delivery.
- The cycle 20 validator will write smoke tests that confirm `_core.normalize_total` etc. are callable (just `hasattr` checks + signature inspection — no GPU execution).

## Return format (≤30 lines, exact)

```
## gpu-kernel-dev — cycle 20 (binding extension)
Files written:
  - python/src/_bind_kernels.hpp ({LOC})
  - python/src/_bind_cupy_ingest.hpp ({LOC})
Files modified:
  - python/src/_singlet_gpu_core.cpp (+{LOC})
Total new: {LOC}
Build: SKIPPED (no nvcc)
Existing tests: SKIPPED
Workspace budget: pure marshal, no GPU workspace
Streams used: factornet's
Precision: fp32
Determinism: inherits from C++
Self-check: no host copies in any binding; cupy ingest uses non-owning device spans: CONFIRMED
Bindings exposed: 13 functions + 4 result classes
Notes: {1-3 lines}
```

Nothing else.

## Risks

1. **`__cuda_array_interface__` ingest is non-trivial**: parsing the dict, validating dtype/strides, constructing the device span. Need careful testing.
2. **Lifetime bugs**: a cupy.sparse.csr_matrix that's GC'd while a DeviceCsc references it = use-after-free. Test with `gc.collect()` patterns.
3. **Result class proliferation**: 4 result classes need pybind11 bindings + `__cuda_array_interface__` properties. Bulky but necessary.
4. **Signature parity (CYCLE-19-FOLLOWUP-SCANPY-SIG-PARITY)** is a separate followup — this cycle uses our internal C++ parameter names. The Python wrapper layer (cycle 19) translates to scanpy names.
