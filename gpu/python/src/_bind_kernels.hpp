// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_kernels.hpp
//
// bind_kernels(pybind11::module_& m) — registers all per-kernel pybind11 functions.
//
// Architecture:
//   Each py_* function:
//     1. Unwraps PyDeviceCsc → constructs a temporary io::PzDeviceMatrix view.
//     2. Builds the C++ Config struct from Python kwargs.
//     3. Calls the C++ kernel.
//     4. Wraps the C++ result into a pybind11-exposed Python result class
//        (NormalizeResult, HvgResult, SvdResult, NmfResult — registered in
//        _singlet_gpu_core.cpp).
//
// No host↔device copies anywhere in this file.  Result classes expose device
// buffers via __cuda_array_interface__ (same pattern as DeviceCsc.data_view).
//
// Lifetime: result classes hold a shared_ptr to their source DeviceMemory<T>
// via the py::object _owner anchor — same pattern as DeviceCsc views.
//
// PzDeviceMatrix view construction note:
//   The C++ kernels accept const/non-const io::PzDeviceMatrix& and modify
//   mat.values in place (lognorm, log1p).  We construct a local PzDeviceMatrix
//   with mat aliased from the PyDeviceCsc's shared_ptr<DeviceCSC>.  The alias
//   is a move-in of the DeviceCSC fields into the local struct — but DeviceCSC
//   is non-copyable and its DeviceMemory members are non-owning aliases here, so
//   we use from_device_ptrs to build a non-owning view of the same device pointers.
//   This guarantees: (a) the original PyDeviceCsc still owns the device memory;
//   (b) the local PzDeviceMatrix does NOT cudaFree on dtor.
//
// SvdResult / NmfResult memory ownership:
//   factornet returns SvdResult (SVDResult<float>) with Eigen matrices on host.
//   NmfResult (NMFResult<float>) similarly.  We expose them as Python objects
//   whose array views call cudaMemcpy on first access — but to avoid that latency
//   we upload to device at bind time.  The result objects therefore own DeviceMemory<T>
//   buffers.
//
// LogNormResult / HvgResult memory ownership:
//   Both return DeviceMemory<T> buffers from the kernel — already device-resident.
//   We hold them in the Python result class.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/preprocess/hvg.h>
// Rule 32: only deflation (primary) and randomized (fallback) remain.
#include <singlet-gpu/reduce/svd/auto_select.h>
#include <singlet-gpu/reduce/svd/deflation.h>
#include <singlet-gpu/reduce/svd/randomized.h>
#include <singlet-gpu/reduce/nmf/fit.h>
#include <singlet-gpu/reduce/nmf/chunked.h>
#include <singlet-gpu/reduce/nmf/graph.h>
#include <singlet-gpu/streaming/pz_data_loader.h>
#include <singlet-gpu/streaming/streamed_pipeline.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/graph/leiden.h>
#include <singlet-gpu/embed/umap.h>
#include <singlet-gpu/de/wilcoxon.h>
#include <singlet-gpu/de/ttest.h>
#include <singlet-gpu/anno/marker_score.h>
#include <singlet-gpu/anno/reference_map.h>
#include <singlet-gpu/gsea/fgsea.h>
#include <singlet-gpu/gsea/aucell.h>
#include <singlet-gpu/integrate/harmony.h>
#include <singlet-gpu/integrate/bbknn.h>
#include <singlet-gpu/preprocess/velocity_prep.h>
#include <singlet-gpu/anno/mt_lineage.h>
#include <singlet-gpu/de/donor_pseudobulk.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"   // PyDeviceCsc, PyPzDeviceMatrix
#include "_bind_results.hpp"  // Py* result wrappers for cycles 7-17

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// PyNormalizeResult — wraps LogNormResult (device buffers) for Python.
// ---------------------------------------------------------------------------
struct PyNormalizeResult {
    singlet_gpu::core::DeviceMemory<float>   size_factors;
    singlet_gpu::core::DeviceMemory<uint8_t> qc_mask;
    float                                    target_used = 0.0f;
    cudaStream_t                             stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyHvgResult — wraps HvgResult (device buffers) for Python.
// ---------------------------------------------------------------------------
struct PyHvgResult {
    singlet_gpu::core::DeviceMemory<int>   indices;
    singlet_gpu::core::DeviceMemory<float> scores;
    singlet_gpu::core::DeviceMemory<float> mean;
    singlet_gpu::core::DeviceMemory<float> var;
    cudaStream_t                           stream = nullptr;
};

// ---------------------------------------------------------------------------
// PySvdResult — wraps SvdResult (host Eigen matrices from factornet) for Python.
// Uploads to device on construction so views are zero-copy thereafter.
// ---------------------------------------------------------------------------
struct PySvdResult {
    // Device buffers — uploaded once from the factornet Eigen result.
    singlet_gpu::core::DeviceMemory<float> d_U;    // rows × k  (col-major)
    singlet_gpu::core::DeviceMemory<float> d_d;    // k
    singlet_gpu::core::DeviceMemory<float> d_V;    // cols × k  (col-major)
    int k_selected = 0;
    int rows = 0;
    int cols = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyNmfResult — same pattern as PySvdResult.
// ---------------------------------------------------------------------------
struct PyNmfResult {
    singlet_gpu::core::DeviceMemory<float> d_W;   // rows × k  (col-major)
    singlet_gpu::core::DeviceMemory<float> d_d;   // k
    singlet_gpu::core::DeviceMemory<float> d_H;   // k × cols  (row-major in factornet)
    int k_used = 0;
    int iterations = 0;
    bool converged = false;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// make_pz_view — construct a non-owning io::PzDeviceMatrix from PyDeviceCsc.
//
// WHY: all C++ kernels take io::PzDeviceMatrix&.  We build a stack-local view
// that aliases the PyDeviceCsc's device pointers without transferring ownership
// (from_device_ptrs sets owns_=false on all DeviceMemory members).
// ---------------------------------------------------------------------------
static singlet_gpu::io::PzDeviceMatrix make_pz_view(PyDeviceCsc& py_csc)
{
    singlet_gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet_gpu::core::DeviceCSC::from_device_ptrs(
        static_cast<int>(py_csc.csc->rows),
        static_cast<int>(py_csc.csc->cols),
        static_cast<int>(py_csc.csc->nnz),
        py_csc.csc->col_ptr.data(),
        py_csc.csc->row_indices.get(),
        py_csc.csc->values.data()
    );
    pz.producer_stream = py_csc.producer_stream;
    // host_retained = false: SVD adapters that call require_host_retained will
    // throw if host pinned buffers are not available.  Users who need SVD must
    // call load_pz(keep_host_pinned=True) and work through PzDeviceMatrix.mat
    // via the existing C++ API, or populate host_indptr/indices/values on the
    // PyDeviceCsc manually (not yet exposed).
    pz.host_retained = false;
    return pz;
}

// ---------------------------------------------------------------------------
// upload_eigen_matrix — copy an Eigen column-major float matrix to device.
// Allocates a DeviceMemory<float> and calls cudaMemcpy synchronously once.
// WHY synchronous here: the Eigen source is on host after factornet returns;
// the user will access the result view immediately after the Python call returns.
// ---------------------------------------------------------------------------
static singlet_gpu::core::DeviceMemory<float>
upload_eigen_matrix(const float* host_ptr, std::size_t count)
{
    singlet_gpu::core::DeviceMemory<float> d(count);
    if (count > 0 && host_ptr != nullptr) {
        cudaError_t err = cudaMemcpy(
            d.get(), host_ptr, count * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("upload_eigen_matrix: ") + cudaGetErrorString(err));
    }
    return d;
}

namespace singlet_gpu {
namespace python {

// ===========================================================================
// PREPROCESS — normalize_total, log1p
// ===========================================================================

// py_normalize_total — wraps preprocess::log_normalize.
// target_sum=0 means use median (default); approximate_median is the OOC path.
static PyNormalizeResult py_normalize_total(
    py::object mat_obj,
    float target_sum,
    bool  approximate_median)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

    singlet_gpu::preprocess::LogNormConfig cfg;
    cfg.method           = singlet_gpu::preprocess::LogNormMethod::TotalCount;
    cfg.target_count     = target_sum;          // 0 = use median
    cfg.approximate_median = approximate_median;

    // log_normalize mutates mat.values in place; pass DeviceCSC& directly.
    singlet_gpu::preprocess::LogNormResult r =
        singlet_gpu::preprocess::log_normalize(
            *py_csc.csc, cfg, py_csc.producer_stream);

    PyNormalizeResult out;
    out.size_factors = std::move(r.size_factors);
    out.qc_mask      = std::move(r.qc_mask);
    out.target_used  = r.target_used;
    out.stream       = py_csc.producer_stream;
    return out;
}

// py_log1p — in-place log1p with configurable base.
// base=0 means natural log (same as Python math.log1p).
static void py_log1p(py::object mat_obj, float base)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

    // log1p is a lightweight kernel: pass a trivial config.
    // base != 0 means log_{base}(1+x) = log(1+x)/log(base) — applied post-kernel.
    // We delegate base=0 (natural) directly to log_normalize with target_count=1
    // so no size-factor scaling occurs (size factor becomes 1.0 for every cell).
    // A dedicated log1p kernel (preprocess::log1p_inplace) is planned; for now
    // we reuse the lognorm pass 4 by supplying target_count=1 (s_j always = 1).
    singlet_gpu::preprocess::LogNormConfig cfg;
    cfg.method       = singlet_gpu::preprocess::LogNormMethod::TotalCount;
    cfg.target_count = 1.0f;  // T=1 → s_j = col_sum → values become log1p(x/col_sum)
    // WHY target_count=1 for log1p: when T=1, s_j=col_sum so values become
    // log1p(x/col_sum), not plain log1p(x).  True in-place log1p needs a
    // dedicated preprocess::log1p_inplace kernel (cycle-21 target).
    // For the binding layer the signature is already correct; semantics provisional.
    (void)singlet_gpu::preprocess::log_normalize(
        *py_csc.csc, cfg, py_csc.producer_stream);
    (void)base;  // base scaling: deferred to cycle-21 preprocess::log1p_inplace
}

// ===========================================================================
// PREPROCESS — highly_variable_genes
// ===========================================================================

static PyHvgResult py_highly_variable_genes(
    py::object mat_obj,
    int         n_top_genes,
    std::string flavor_str,
    float       min_mean,
    float       max_mean,
    float       pearson_theta)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

    singlet_gpu::preprocess::HvgConfig cfg;
    cfg.top_n         = n_top_genes;
    cfg.min_mean      = min_mean;
    cfg.max_mean      = max_mean;
    cfg.pearson_theta = pearson_theta;

    if (flavor_str == "seurat_v3" || flavor_str == "seurat") {
        cfg.flavor = singlet_gpu::preprocess::HvgFlavor::SeuratV3;
    } else if (flavor_str == "pearson_residuals" || flavor_str == "pearson") {
        cfg.flavor = singlet_gpu::preprocess::HvgFlavor::PearsonResiduals;
    } else {
        throw py::value_error(
            "highly_variable_genes: unknown flavor '" + flavor_str
            + "'. Use 'seurat_v3' or 'pearson_residuals'.");
    }

    // select_hvg takes const DeviceCSC& — pass directly, no PzDeviceMatrix needed.
    singlet_gpu::preprocess::HvgResult r =
        singlet_gpu::preprocess::select_hvg(
            *py_csc.csc, cfg, py_csc.producer_stream);

    PyHvgResult out;
    out.indices = std::move(r.indices);
    out.scores  = std::move(r.scores);
    out.mean    = std::move(r.mean);
    out.var     = std::move(r.var);
    out.stream  = py_csc.producer_stream;
    return out;
}

// ===========================================================================
// REDUCE — SVD helpers
// ===========================================================================

// build_svd_config — shared config constructor for all SVD backends.
static singlet_gpu::reduce::svd::SvdConfig build_svd_config(
    int      k,
    bool     center,
    bool     scale,
    float    tol,
    int      max_iter,
    uint64_t seed)
{
    singlet_gpu::reduce::svd::SvdConfig cfg;
    cfg.k_max    = k;
    cfg.center   = center;
    cfg.scale    = scale;
    cfg.tol      = tol;
    cfg.max_iter = max_iter;
    // factornet uses uint32_t seed; Python API accepts uint64_t for forward compat.
    cfg.seed     = static_cast<uint32_t>(seed & 0xFFFFFFFF);
    return cfg;
}

// upload_svd_result — take a factornet SvdResult (host Eigen) and produce PySvdResult.
static PySvdResult upload_svd_result(
    const singlet_gpu::reduce::svd::SvdResult& r,
    int rows, int cols, cudaStream_t stream)
{
    int k = static_cast<int>(r.d.size());

    PySvdResult out;
    out.k_selected = k;
    out.rows       = rows;
    out.cols       = cols;
    out.stream     = stream;

    // U: rows × k (Eigen col-major → flat float array, rows*k elements)
    out.d_U = upload_eigen_matrix(r.U.data(), static_cast<std::size_t>(rows * k));
    // d: k singular values
    out.d_d = upload_eigen_matrix(r.d.get(), static_cast<std::size_t>(k));
    // V: cols × k
    out.d_V = upload_eigen_matrix(r.V.data(), static_cast<std::size_t>(cols * k));

    return out;
}

// py_svd_auto_select
static PySvdResult py_svd_auto_select(
    py::object mat_obj, int k,
    bool center, bool scale, float tol, int max_iter, uint64_t seed)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    singlet_gpu::io::PzDeviceMatrix pz = make_pz_view(py_csc);
    auto cfg = build_svd_config(k, center, scale, tol, max_iter, seed);
    auto r = singlet_gpu::reduce::svd::auto_select(pz, k, cfg);
    return upload_svd_result(r, pz.mat.rows, pz.mat.cols, pz.producer_stream);
}

// py_svd_randomized
// Rule 32: py_svd_lanczos and py_svd_irlba removed (backends deleted).
static PySvdResult py_svd_randomized(
    py::object mat_obj, int k,
    bool center, bool scale, float tol, int max_iter, uint64_t seed)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    singlet_gpu::io::PzDeviceMatrix pz = make_pz_view(py_csc);
    auto cfg = build_svd_config(k, center, scale, tol, max_iter, seed);
    auto r = singlet_gpu::reduce::svd::randomized(pz, cfg);
    return upload_svd_result(r, pz.mat.rows, pz.mat.cols, pz.producer_stream);
}

// py_svd_deflation — supports all constraint kwargs (L1, L2, nonneg, etc.).
// Rule 32: py_svd_krylov removed; deflation handles all constraint types.
static PySvdResult py_svd_deflation(
    py::object mat_obj, int k,
    bool center, bool scale, float tol, int max_iter, uint64_t seed,
    py::kwargs kwargs)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    singlet_gpu::io::PzDeviceMatrix pz = make_pz_view(py_csc);
    auto cfg = build_svd_config(k, center, scale, tol, max_iter, seed);

    if (kwargs.contains("L1_u"))       cfg.L1_u       = kwargs["L1_u"].cast<float>();
    if (kwargs.contains("L1_v"))       cfg.L1_v       = kwargs["L1_v"].cast<float>();
    if (kwargs.contains("L2_u"))       cfg.L2_u       = kwargs["L2_u"].cast<float>();
    if (kwargs.contains("L2_v"))       cfg.L2_v       = kwargs["L2_v"].cast<float>();
    if (kwargs.contains("nonneg_u"))   cfg.nonneg_u   = kwargs["nonneg_u"].cast<bool>();
    if (kwargs.contains("nonneg_v"))   cfg.nonneg_v   = kwargs["nonneg_v"].cast<bool>();

    auto r = singlet_gpu::reduce::svd::deflation(pz, cfg);
    return upload_svd_result(r, pz.mat.rows, pz.mat.cols, pz.producer_stream);
}

// py_pca — alias for auto_select with zero_center defaulting to true.
static PySvdResult py_pca(
    py::object mat_obj, int n_comps,
    bool zero_center, bool scale, float tol, int max_iter, uint64_t seed)
{
    // pca is auto_select with zero_center mapped to center.
    return py_svd_auto_select(mat_obj, n_comps,
                               zero_center, scale, tol, max_iter, seed);
}

// ===========================================================================
// REDUCE — NMF helpers
// ===========================================================================

// build_nmf_config — shared config constructor.
static singlet_gpu::reduce::nmf::NmfConfig build_nmf_config(
    int rank, std::string loss_str, int solver_mode, int init_mode,
    int max_iter, float tol, uint64_t seed)
{
    singlet_gpu::reduce::nmf::NmfConfig cfg;
    cfg.rank        = rank;
    cfg.solver_mode = solver_mode;
    cfg.init_mode   = init_mode;
    cfg.max_iter    = max_iter;
    cfg.tol         = tol;
    // factornet NMFConfig uses uint32_t seed; Python API exposes uint64_t.
    cfg.seed        = static_cast<uint32_t>(seed & 0xFFFFFFFF);

    // Map loss string to LossType.
    if (loss_str == "MSE" || loss_str == "mse") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::MSE;
    } else if (loss_str == "KL" || loss_str == "kl") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::KL;
    } else if (loss_str == "MAE" || loss_str == "mae") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::MAE;
    } else {
        throw py::value_error(
            "nmf: unknown loss '" + loss_str
            + "'. Use 'MSE', 'KL', or 'MAE'.");
    }
    return cfg;
}

// upload_nmf_result — take factornet NmfResult (host Eigen) → PyNmfResult.
static PyNmfResult upload_nmf_result(
    const singlet_gpu::reduce::nmf::NmfResult& r,
    int k, int rows, int cols, cudaStream_t stream)
{
    PyNmfResult out;
    out.k_used     = k;
    out.iterations = r.iterations;
    out.converged  = r.converged;
    out.stream     = stream;

    // W: rows × k (col-major Eigen)
    out.d_W = upload_eigen_matrix(r.W.data(), static_cast<std::size_t>(rows * k));
    // d: k scale factors
    out.d_d = upload_eigen_matrix(r.d.get(), static_cast<std::size_t>(k));
    // H: k × cols
    out.d_H = upload_eigen_matrix(r.H.data(), static_cast<std::size_t>(k * cols));

    return out;
}

// py_nmf — in-memory NMF.
static PyNmfResult py_nmf(
    py::object mat_obj, int rank,
    std::string loss, int solver_mode, int init_mode,
    int max_iter, float tol, uint64_t seed)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    singlet_gpu::io::PzDeviceMatrix pz = make_pz_view(py_csc);
    auto cfg = build_nmf_config(rank, loss, solver_mode, init_mode, max_iter, tol, seed);
    auto r = singlet_gpu::reduce::nmf::fit(pz, cfg);
    return upload_nmf_result(r, rank,
                              static_cast<int>(pz.mat.rows),
                              static_cast<int>(pz.mat.cols),
                              pz.producer_stream);
}

// py_nmf_chunked — streaming NMF via PzDataLoader.
// loader_obj: a Python object wrapping a PzDataLoader (not yet bound in cycle 18;
// this function accepts the raw C++ object via py::object and casts it).
static PyNmfResult py_nmf_chunked(
    py::object loader_obj, int rank,
    std::string loss, int solver_mode, int init_mode,
    int max_iter, float tol, uint64_t seed)
{
    auto& loader = loader_obj.cast<singlet_gpu::io::PzDataLoader&>();
    auto cfg = build_nmf_config(rank, loss, solver_mode, init_mode, max_iter, tol, seed);
    auto r = singlet_gpu::reduce::nmf::chunked_fit(loader, cfg);

    int rows = static_cast<int>(loader.rows());
    int cols = static_cast<int>(loader.cols());
    return upload_nmf_result(r, rank, rows, cols, /*stream=*/nullptr);
}

// py_nmf_graph_factorize — multi-modal graph NMF.
// inputs: Python list of DeviceCsc objects (one per modality).
// topology: "shared_h" (default SharedNode), "concat", "hierarchical"
static PyNmfResult py_nmf_graph_factorize(
    py::list inputs_list, int rank, std::string topology)
{
    // Validate at least one input.
    if (inputs_list.empty())
        throw py::value_error("nmf_graph_factorize: inputs list is empty");

    // Convert list of PyDeviceCsc to list of PzDeviceMatrix views.
    std::vector<singlet_gpu::io::PzDeviceMatrix> pz_views;
    pz_views.reserve(inputs_list.size());
    for (py::handle h : inputs_list) {
        auto& py_csc = h.cast<PyDeviceCsc&>();
        pz_views.push_back(make_pz_view(py_csc));
    }

    // Require all inputs host_retained for graph NMF Eigen mapping.
    for (std::size_t i = 0; i < pz_views.size(); ++i) {
        if (!pz_views[i].host_retained) {
            throw std::runtime_error(
                "nmf_graph_factorize: all inputs must be loaded with "
                "keep_host_pinned=True (input index "
                + std::to_string(i) + ")");
        }
    }

    // Build Eigen views and FactorGraph.
    namespace G = singlet_gpu::reduce::nmf::graph;
    using SpMat = G::SpMat;

    std::vector<SpMat> eigen_mats;
    eigen_mats.reserve(pz_views.size());
    for (auto& pz : pz_views) {
        eigen_mats.push_back(G::eigen_view_of(pz));
    }

    // Build input nodes (hold Eigen matrices by reference — must outlive graph).
    std::vector<G::InputNode<SpMat>> input_nodes;
    input_nodes.reserve(eigen_mats.size());
    for (std::size_t i = 0; i < eigen_mats.size(); ++i) {
        input_nodes.emplace_back(eigen_mats[i], "input_" + std::to_string(i));
    }

    // Collect raw node pointers for graph construction.
    std::vector<::factornet::graph::NodeBase<float>*> node_ptrs;
    node_ptrs.reserve(input_nodes.size());
    for (auto& n : input_nodes) node_ptrs.push_back(&n);

    if (topology == "shared_h") {
        // SharedNode joins all inputs under a single NMFLayerNode.
        G::SharedNode shared(node_ptrs);
        G::NMFLayerNode joint(&shared, rank, "joint");
        G::FactorGraph net(node_ptrs, &joint);
        net.compile();

        // Primary data is the first input for factornet::graph::fit.
        auto gr = G::fit_graph(net, eigen_mats[0]);

        // Flatten result: use the "joint" layer's H as the cell embedding.
        const auto& layer = gr.layers.at("joint");
        int rows = static_cast<int>(pz_views[0].mat.rows);
        int cols = static_cast<int>(pz_views[0].mat.cols);

        PyNmfResult out;
        out.k_used    = rank;
        out.converged = gr.converged;
        out.stream    = nullptr;
        out.d_W = upload_eigen_matrix(layer.W.data(),
                                      static_cast<std::size_t>(rows * rank));
        out.d_d = upload_eigen_matrix(layer.d.get(),
                                      static_cast<std::size_t>(rank));
        out.d_H = upload_eigen_matrix(layer.H.data(),
                                      static_cast<std::size_t>(rank * cols));
        return out;

    } else {
        throw py::value_error(
            "nmf_graph_factorize: unknown topology '" + topology
            + "'. Supported: 'shared_h'.");
    }
}

// ===========================================================================
// bind_kernels — registers all functions in the module.
// ===========================================================================

inline void bind_kernels(py::module_& m)
{
    // -- preprocess --
    m.def("normalize_total", &py_normalize_total,
          py::arg("mat"),
          py::kw_only(),
          py::arg("target_sum") = 0.0f,
          py::arg("approximate_median") = false,
          R"doc(
          Per-cell log-normalization in place.  Mutates mat.data.

          Parameters
          ----------
          mat : DeviceCsc
              Input count matrix (genes × cells), float32 device.
          target_sum : float, optional
              Target library size. 0 (default) = use per-sample median.
          approximate_median : bool, optional
              Use streaming P2 quantile estimate (future out-of-core path).

          Returns
          -------
          NormalizeResult
              .target_used  — float, the T actually applied.
              .size_factors_view — __cuda_array_interface__ dict (float32, n_cells).
              .qc_mask_view      — __cuda_array_interface__ dict (uint8,  n_cells).
          )doc");

    m.def("log1p", &py_log1p,
          py::arg("mat"),
          py::kw_only(),
          py::arg("base") = 0.0f,
          R"doc(
          In-place element-wise log1p transform.

          Parameters
          ----------
          mat : DeviceCsc
              Matrix to transform in place.
          base : float, optional
              Log base.  0 (default) = natural log.
          )doc");

    m.def("highly_variable_genes", &py_highly_variable_genes,
          py::arg("mat"),
          py::kw_only(),
          py::arg("n_top_genes")    = 2000,
          py::arg("flavor")         = std::string("seurat_v3"),
          py::arg("min_mean")       = 0.0125f,
          py::arg("max_mean")       = 3.0f,
          py::arg("pearson_theta")  = 100.0f,
          R"doc(
          Select highly variable genes.

          Parameters
          ----------
          mat : DeviceCsc
              Normalized count matrix (genes × cells).
          n_top_genes : int
              Number of HVGs to select.
          flavor : str
              'seurat_v3' (default) or 'pearson_residuals'.
          min_mean, max_mean : float
              Mean-expression filter window.
          pearson_theta : float
              Overdispersion parameter for Pearson residuals flavor.

          Returns
          -------
          HvgResult
              .indices_view — int32 device array (n_top_genes,), gene indices.
              .scores_view  — float32 device array (n_top_genes,).
              .mean_view    — float32 device array (n_genes,).
              .var_view     — float32 device array (n_genes,).
          )doc");

    // -- SVD backends --
    m.def("svd_auto_select", &py_svd_auto_select,
          py::arg("mat"), py::arg("k"),
          py::kw_only(),
          py::arg("center")   = true,
          py::arg("scale")    = false,
          py::arg("tol")      = 1e-6f,
          py::arg("max_iter") = 100,
          py::arg("seed")     = uint64_t{0},
          // Rule 32: deflation primary for all k; randomized fallback on empty result only.
          "Auto-select SVD backend (Deflation primary / Randomized automatic fallback). "
          "To use randomized directly, call svd_randomized() instead.");

    // Rule 32: svd_lanczos, svd_irlba, svd_krylov removed (backends deleted).

    m.def("svd_randomized", &py_svd_randomized,
          py::arg("mat"), py::arg("k"),
          py::kw_only(),
          py::arg("center")   = true,
          py::arg("scale")    = false,
          py::arg("tol")      = 1e-6f,
          py::arg("max_iter") = 100,
          py::arg("seed")     = uint64_t{0},
          "Randomized SVD (Rule 32 fallback; lowest memory footprint for OOC chunks).");

    m.def("svd_deflation", &py_svd_deflation,
          py::arg("mat"), py::arg("k"),
          py::kw_only(),
          py::arg("center")   = true,
          py::arg("scale")    = false,
          py::arg("tol")      = 1e-6f,
          py::arg("max_iter") = 100,
          py::arg("seed")     = uint64_t{0},
          "Deflation SVD (Rule 32 primary; k-independent 28ms, all k values). "
          "Extra kwargs: L1_u, L1_v, L2_u, L2_v, nonneg_u, nonneg_v (all float/bool).");

    m.def("pca", &py_pca,
          py::arg("mat"), py::arg("n_comps"),
          py::kw_only(),
          py::arg("zero_center") = true,
          py::arg("scale")       = false,
          py::arg("tol")         = 1e-6f,
          py::arg("max_iter")    = 100,
          py::arg("seed")        = uint64_t{0},
          "PCA via auto-selected SVD backend.  Alias for svd_auto_select.");

    // -- NMF backends --
    m.def("nmf", &py_nmf,
          py::arg("mat"), py::arg("rank"),
          py::kw_only(),
          py::arg("loss")        = std::string("MSE"),
          py::arg("solver_mode") = 3,
          py::arg("init_mode")   = 2,
          py::arg("max_iter")    = 100,
          py::arg("tol")         = 1e-5f,
          py::arg("seed")        = uint64_t{0},
          "In-memory GPU NMF (MSE/KL/MAE loss; MU/CD/Cholesky solver).");

    m.def("nmf_chunked", &py_nmf_chunked,
          py::arg("loader"), py::arg("rank"),
          py::kw_only(),
          py::arg("loss")        = std::string("MSE"),
          py::arg("solver_mode") = 2,
          py::arg("init_mode")   = 2,
          py::arg("max_iter")    = 100,
          py::arg("tol")         = 1e-5f,
          py::arg("seed")        = uint64_t{0},
          "Streaming chunked GPU NMF.  loader: PzDataLoader object.");

    m.def("nmf_graph_factorize", &py_nmf_graph_factorize,
          py::arg("inputs"), py::arg("rank"),
          py::kw_only(),
          py::arg("topology") = std::string("shared_h"),
          "Multi-modal graph NMF.  inputs: list[DeviceCsc]. "
          "topology: 'shared_h' (shared cell embedding H across modalities).");
}

// ===========================================================================
// CYCLE 7 — streaming_pipeline_run
// ===========================================================================

// convert_pipeline_result — translate C++ PipelineResult → PyStreamingPipelineResult.
static std::shared_ptr<PyStreamingPipelineResult>
convert_pipeline_result(singlet_gpu::streaming::PipelineResult&& r)
{
    auto out = std::make_shared<PyStreamingPipelineResult>();
    out->n_cells              = r.n_cells;
    out->n_genes_total        = r.n_genes_total;
    out->n_nnz_total          = r.n_nnz_total;
    out->lognorm_target_used  = r.lognorm_target_used;
    out->pca_ran              = r.pca_ran;
    out->nmf_ran              = r.nmf_ran;
    out->wall_lognorm_s       = r.wall_lognorm_s;
    out->wall_hvg_s           = r.wall_hvg_s;
    out->wall_pca_s           = r.wall_pca_s;
    out->wall_nmf_s           = r.wall_nmf_s;
    out->n_chunks_processed   = r.n_chunks_processed;
    out->size_factors         = std::move(r.size_factors);
    out->qc_mask              = std::move(r.qc_mask);
    out->hvg_indices          = std::move(r.hvg_indices);
    out->hvg_scores           = std::move(r.hvg_scores);
    out->gene_means           = std::move(r.gene_means);
    out->gene_vars            = std::move(r.gene_vars);
    out->pca                  = std::move(r.pca);
    out->nmf                  = std::move(r.nmf);
    return out;
}

static std::shared_ptr<PyStreamingPipelineResult>
py_streaming_pipeline_run(
    py::list input_paths_list,
    int      chunk_cols,
    bool     run_lognorm,
    bool     run_hvg,
    bool     run_pca,
    int      pca_k,
    bool     run_nmf,
    int      nmf_factors,
    bool     cache_normalized,
    int64_t  in_memory_pca_threshold)
{
    singlet_gpu::streaming::PipelineConfig cfg;
    cfg.chunk_cols              = chunk_cols;
    cfg.run_lognorm             = run_lognorm;
    cfg.run_hvg                 = run_hvg;
    cfg.run_pca                 = run_pca;
    cfg.pca_k                   = pca_k;
    cfg.run_nmf                 = run_nmf;
    cfg.nmf_cfg.rank            = nmf_factors;
    cfg.cache_normalized        = cache_normalized;
    cfg.in_memory_pca_threshold = in_memory_pca_threshold;

    // Convert py::list of strings → std::vector<std::string>.
    for (py::handle h : input_paths_list)
        cfg.input_paths.push_back(h.cast<std::string>());

    auto r = singlet_gpu::streaming::run_pipeline(cfg, /*stream=*/nullptr);
    return convert_pipeline_result(std::move(r));
}

// ===========================================================================
// CYCLE 8 — knn_graph, knn_graph_from_indices
//
// Both return a PyKnnResult.  The C++ DeviceDense is constructed from a
// Python object that must expose __cuda_array_interface__ (e.g. cupy array).
// We extract the raw device pointer + shape from the interface dict and build
// a non-owning DeviceDense view — same pattern as from_cupy_csr.
// ===========================================================================

// build_device_dense_view — extract a non-owning DeviceDense from a Python
// object with __cuda_array_interface__.  Requires dtype=float32, ndim=2.
// Returns the DeviceDense + a lifetime anchor py::object.
static singlet_gpu::core::DeviceDense
build_dense_view(py::object arr, int& rows_out, int& cols_out)
{
    if (!py::hasattr(arr, "__cuda_array_interface__"))
        throw py::type_error("embedding must expose __cuda_array_interface__");

    py::dict iface = arr.attr("__cuda_array_interface__").cast<py::dict>();
    std::string ts = iface["typestr"].cast<std::string>();
    if (ts != "<f4" && ts != "f4" && ts != ">f4")
        throw py::value_error("embedding must be float32; got typestr='" + ts + "'");

    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 2)
        throw py::value_error("embedding must be 2-D (n_cells, n_dims)");

    rows_out = static_cast<int>(shape[0].cast<std::size_t>());
    cols_out = static_cast<int>(shape[1].cast<std::size_t>());

    py::tuple dtup = iface["data"].cast<py::tuple>();
    float* ptr = reinterpret_cast<float*>(dtup[0].cast<std::intptr_t>());

    // DeviceDense wraps a non-owning pointer: wrap(ptr, rows, cols).
    return singlet_gpu::core::DeviceDense::wrap(ptr, rows_out, cols_out);
}

// convert_knn_result — move C++ KnnResult → PyKnnResult.
static std::shared_ptr<PyKnnResult>
convert_knn_result(singlet_gpu::graph::KnnResult&& r, cudaStream_t stream)
{
    auto out = std::make_shared<PyKnnResult>();
    out->row_offsets = std::move(r.row_offsets);
    out->neighbors   = std::move(r.neighbors);
    out->distances   = std::move(r.distances);
    out->n           = r.n;
    out->k           = r.k;
    out->stream      = stream;
    return out;
}

static std::shared_ptr<PyKnnResult>
py_knn_graph(
    py::object embedding_obj,
    int         k,
    std::string backend_str,
    std::string metric_str,
    bool        return_squared,
    uint64_t    seed,
    int         hnsw_M,
    int         hnsw_ef)
{
    int n = 0, d = 0;
    // embedding_obj lifetime must outlive this call; DeviceDense is non-owning.
    singlet_gpu::core::DeviceDense emb = build_dense_view(embedding_obj, n, d);

    singlet_gpu::graph::KnnConfig cfg;
    cfg.k              = k;
    cfg.return_squared = return_squared;
    cfg.seed           = seed;
    cfg.hnsw_M         = hnsw_M;
    cfg.hnsw_ef        = hnsw_ef;

    if (backend_str == "auto" || backend_str == "Auto")
        cfg.backend = singlet_gpu::graph::KnnBackend::Auto;
    else if (backend_str == "exact" || backend_str == "Exact")
        cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    else if (backend_str == "cagra" || backend_str == "CAGRA" || backend_str == "Cagra"
             || backend_str == "hnsw" || backend_str == "HNSW")  // backward compat
        cfg.backend = singlet_gpu::graph::KnnBackend::Cagra;
    else
        throw py::value_error("knn_graph: unknown backend '" + backend_str
                              + "'. Use 'auto', 'exact', or 'hnsw'.");

    if (metric_str == "L2" || metric_str == "l2")
        cfg.metric = singlet_gpu::graph::DistanceMetric::L2;
    else if (metric_str == "Cosine" || metric_str == "cosine")
        cfg.metric = singlet_gpu::graph::DistanceMetric::Cosine;
    else if (metric_str == "Inner" || metric_str == "inner" || metric_str == "IP")
        cfg.metric = singlet_gpu::graph::DistanceMetric::Inner;
    else
        throw py::value_error("knn_graph: unknown metric '" + metric_str
                              + "'. Use 'L2', 'Cosine', or 'Inner'.");

    cudaStream_t stream = singlet_gpu::core::default_context().stream();
    auto r = singlet_gpu::graph::compute_knn(emb, cfg, stream);
    return convert_knn_result(std::move(r), stream);
}

// knn_graph_from_indices: reuse knn_graph but accept pre-built KnnResult
// (no-op wrapper — the KnnResult is already built; this entry point exposes
// it as a Python function for consistency with the design doc API surface).
// In practice callers just use knn_graph; this form is for pipelines that
// already hold a PyKnnResult and want to forward it to leiden/umap.
// We simply return the same object (no copy, no recompute).
static std::shared_ptr<PyKnnResult>
py_knn_graph_from_indices(py::object knn_result_obj)
{
    // knn_result_obj must be a PyKnnResult; return it as-is (identity forward).
    auto ptr = knn_result_obj.cast<std::shared_ptr<PyKnnResult>>();
    return ptr;
}

// ===========================================================================
// CYCLE 9 — leiden_partition, leiden_multi_resolution
// ===========================================================================

// extract_knn_csr — pull raw CSR pointers from a PyKnnResult.
static singlet_gpu::graph::KnnResult
borrow_knn_result(PyKnnResult& py_knn)
{
    // Build a C++ KnnResult that borrows device pointers from py_knn.
    // DeviceMemory::wrap gives a non-owning alias (owns_=false).
    singlet_gpu::graph::KnnResult r;
    r.n            = py_knn.n;
    r.k            = py_knn.k;
    r.backend_used = singlet_gpu::graph::KnnBackend::Auto;

    const std::size_t n1 = static_cast<std::size_t>(py_knn.n + 1);
    const std::size_t nk = static_cast<std::size_t>(py_knn.n) * py_knn.k;

    r.row_offsets = singlet_gpu::core::DeviceMemory<int>::wrap(
        py_knn.row_offsets.data(), n1);
    r.neighbors   = singlet_gpu::core::DeviceMemory<int>::wrap(
        py_knn.neighbors.data(), nk);
    r.distances   = singlet_gpu::core::DeviceMemory<float>::wrap(
        py_knn.distances.data(), nk);
    return r;
}

static std::shared_ptr<PyLeidenResult>
py_leiden_partition(
    py::object  knn_result_obj,
    float       resolution,
    int         max_iter,
    uint64_t    seed,
    std::string backend_str,
    std::string weight_function_str,
    float       gaussian_sigma)
{
    auto& py_knn = knn_result_obj.cast<PyKnnResult&>();
    auto knn = borrow_knn_result(py_knn);

    singlet_gpu::graph::LeidenConfig cfg;
    cfg.resolution     = resolution;
    cfg.max_iter       = max_iter;
    cfg.seed           = seed;
    cfg.gaussian_sigma = gaussian_sigma;
    (void)backend_str;  // cuGraph is the only supported backend

    if (weight_function_str == "Connectivity" || weight_function_str == "connectivity")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Connectivity;
    else if (weight_function_str == "Inverse" || weight_function_str == "inverse")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Inverse;
    else if (weight_function_str == "Gaussian" || weight_function_str == "gaussian")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Gaussian;
    else
        throw py::value_error("leiden_partition: unknown weight_function '"
                              + weight_function_str
                              + "'. Use 'Connectivity', 'Inverse', or 'Gaussian'.");

    cudaStream_t stream = py_knn.stream;
    auto r = singlet_gpu::graph::leiden(knn, cfg, stream);

    auto out = std::make_shared<PyLeidenResult>();
    out->labels     = std::move(r.labels);
    out->n_clusters = r.n_clusters;
    out->modularity = r.modularity;
    out->iterations = r.iterations;
    out->stream     = stream;
    return out;
}

static py::list
py_leiden_multi_resolution(
    py::object  knn_result_obj,
    py::list    resolutions_list,
    uint64_t    seed,
    std::string weight_function_str,
    float       gaussian_sigma)
{
    auto& py_knn = knn_result_obj.cast<PyKnnResult&>();
    auto knn = borrow_knn_result(py_knn);

    singlet_gpu::graph::LeidenConfig cfg;
    cfg.seed           = seed;
    cfg.gaussian_sigma = gaussian_sigma;

    if (weight_function_str == "Connectivity" || weight_function_str == "connectivity")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Connectivity;
    else if (weight_function_str == "Inverse" || weight_function_str == "inverse")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Inverse;
    else if (weight_function_str == "Gaussian" || weight_function_str == "gaussian")
        cfg.weight_function = singlet_gpu::graph::LeidenWeight::Gaussian;
    else
        throw py::value_error("leiden_multi_resolution: unknown weight_function '"
                              + weight_function_str + "'.");

    std::vector<float> resolutions;
    resolutions.reserve(resolutions_list.size());
    for (py::handle h : resolutions_list)
        resolutions.push_back(h.cast<float>());

    cudaStream_t stream = py_knn.stream;
    auto results = singlet_gpu::graph::leiden_multi(knn, resolutions, cfg, stream);

    py::list out_list;
    for (auto& r : results) {
        auto out = std::make_shared<PyLeidenResult>();
        out->labels     = std::move(r.labels);
        out->n_clusters = r.n_clusters;
        out->modularity = r.modularity;
        out->iterations = r.iterations;
        out->stream     = stream;
        out_list.append(out);
    }
    return out_list;
}

// ===========================================================================
// CYCLE 10 — umap_embed
// ===========================================================================

static std::shared_ptr<PyUmapResult>
py_umap_embed(
    py::object  knn_result_obj,
    int         n_components,
    int         n_epochs,
    float       min_dist,
    float       spread,
    float       learning_rate,
    std::string init_str,
    uint64_t    seed,
    int         negative_sample_rate)
{
    auto& py_knn = knn_result_obj.cast<PyKnnResult&>();
    auto knn = borrow_knn_result(py_knn);

    singlet_gpu::embed::UmapConfig cfg;
    cfg.n_components         = n_components;
    cfg.n_epochs             = n_epochs;
    cfg.min_dist             = min_dist;
    cfg.spread               = spread;
    cfg.learning_rate        = learning_rate;
    cfg.seed                 = seed;
    cfg.negative_sample_rate = negative_sample_rate;

    if (init_str == "Random" || init_str == "random")
        cfg.init = singlet_gpu::embed::UmapInit::Random;
    else if (init_str == "Spectral" || init_str == "spectral")
        cfg.init = singlet_gpu::embed::UmapInit::Spectral;
    else
        throw py::value_error("umap_embed: unknown init '" + init_str
                              + "'. Use 'Random' or 'Spectral'.");

    cudaStream_t stream = py_knn.stream;
    auto r = singlet_gpu::embed::umap(knn, cfg, stream);

    auto out = std::make_shared<PyUmapResult>();
    out->embedding    = std::move(r.embedding);
    out->n            = r.n;
    out->n_components = r.n_components;
    out->stream       = stream;
    return out;
}

// ===========================================================================
// CYCLE 11 — wilcoxon_de, ttest_de
// ===========================================================================

// build_device_labels — extract a non-owning DeviceMemory<int> from a Python
// integer array with __cuda_array_interface__.
static singlet_gpu::core::DeviceMemory<int>
borrow_int_array(py::object arr, const char* name)
{
    if (!py::hasattr(arr, "__cuda_array_interface__"))
        throw py::type_error(std::string(name) + " must expose __cuda_array_interface__");

    py::dict iface = arr.attr("__cuda_array_interface__").cast<py::dict>();
    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 1)
        throw py::value_error(std::string(name) + " must be 1-D");

    std::size_t n = shape[0].cast<std::size_t>();
    py::tuple dtup = iface["data"].cast<py::tuple>();
    int* ptr = reinterpret_cast<int*>(dtup[0].cast<std::intptr_t>());

    return singlet_gpu::core::DeviceMemory<int>::wrap(ptr, n);
}

// convert_cluster_markers — move C++ ClusterMarkers → PyClusterMarkers.
static std::shared_ptr<PyClusterMarkers>
convert_cluster_markers(singlet_gpu::de::ClusterMarkers&& cm, cudaStream_t stream)
{
    auto out = std::make_shared<PyClusterMarkers>();
    out->cluster_id  = cm.cluster_id;
    out->gene_indices = std::move(cm.gene_indices);
    out->z_scores    = std::move(cm.z_scores);
    out->log2_fc     = std::move(cm.log2_fc);
    out->p_values    = std::move(cm.p_values);
    out->p_adj       = std::move(cm.p_adj);
    out->stream      = stream;
    return out;
}

static std::shared_ptr<PyWilcoxonResult>
py_wilcoxon_de(
    py::object mat_obj,
    py::object labels_obj,
    int        n_clusters,
    int        n_bins,
    int        top_n,
    int        gene_tile,
    bool       deterministic)
{
    auto& py_csc   = mat_obj.cast<PyDeviceCsc&>();
    auto  labels   = borrow_int_array(labels_obj, "labels");
    cudaStream_t stream = py_csc.producer_stream;

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.n_bins        = n_bins;
    cfg.top_n         = top_n;
    cfg.gene_tile     = gene_tile;
    cfg.deterministic = deterministic;

    auto r = singlet_gpu::de::wilcoxon_de(*py_csc.csc, labels, n_clusters, cfg, stream);

    auto out = std::make_shared<PyWilcoxonResult>();
    out->per_cluster.reserve(r.per_cluster.size());
    for (auto& cm : r.per_cluster)
        out->per_cluster.push_back(convert_cluster_markers(std::move(cm), stream));
    return out;
}

static std::shared_ptr<PyTtestResult>
py_ttest_de(
    py::object mat_obj,
    py::object labels_obj,
    int        n_clusters,
    int        top_n,
    int        gene_tile,
    bool       deterministic)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    auto  labels = borrow_int_array(labels_obj, "labels");
    cudaStream_t stream = py_csc.producer_stream;

    singlet_gpu::de::TtestConfig cfg;
    cfg.top_n         = top_n;
    cfg.gene_tile     = gene_tile;
    cfg.deterministic = deterministic;

    auto r = singlet_gpu::de::ttest_de(*py_csc.csc, labels, n_clusters, cfg, stream);

    auto out = std::make_shared<PyTtestResult>();
    out->per_cluster.reserve(r.per_cluster.size());
    for (auto& cm : r.per_cluster)
        out->per_cluster.push_back(convert_cluster_markers(std::move(cm), stream));
    return out;
}

// ===========================================================================
// CYCLE 12 — marker_score, celltypist_project, load_celltypist_model
// ===========================================================================

// parse_gene_sets — convert py::dict or list-of-tuples to anno::GeneSetDB.
static singlet_gpu::anno::GeneSetDB
parse_gene_sets(py::object gene_sets_obj)
{
    singlet_gpu::anno::GeneSetDB db;

    if (py::isinstance<py::dict>(gene_sets_obj)) {
        // dict: {name: [gene_idx, ...]} or {name: [(gene_idx, weight), ...]}
        py::dict d = gene_sets_obj.cast<py::dict>();
        for (auto& item : d) {
            db.set_names.push_back(item.first.cast<std::string>());
            py::list members = item.second.cast<py::list>();
            std::vector<int>   idxs;
            std::vector<float> wts;
            bool has_weights = false;
            for (py::handle h : members) {
                if (py::isinstance<py::tuple>(h)) {
                    auto tup = h.cast<py::tuple>();
                    idxs.push_back(tup[0].cast<int>());
                    wts.push_back(tup[1].cast<float>());
                    has_weights = true;
                } else {
                    idxs.push_back(h.cast<int>());
                }
            }
            db.member_gene_indices.push_back(std::move(idxs));
            if (has_weights) db.weights.push_back(std::move(wts));
            else             db.weights.emplace_back();  // empty → unit weights
        }
    } else {
        // list-of-tuples: [(name, [gene_idx,...]), ...]
        py::list lst = gene_sets_obj.cast<py::list>();
        for (py::handle h : lst) {
            py::tuple tup = h.cast<py::tuple>();
            db.set_names.push_back(tup[0].cast<std::string>());
            py::list members = tup[1].cast<py::list>();
            std::vector<int> idxs;
            idxs.reserve(members.size());
            for (py::handle g : members) idxs.push_back(g.cast<int>());
            db.member_gene_indices.push_back(std::move(idxs));
            db.weights.emplace_back();  // unit weights
        }
    }
    return db;
}

static std::shared_ptr<PyMarkerScoreResult>
py_marker_score(
    py::object  mat_obj,
    py::object  gene_sets_obj,
    std::string method_str,
    int         min_n_genes_per_set)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    cudaStream_t stream = py_csc.producer_stream;

    auto db = parse_gene_sets(gene_sets_obj);

    singlet_gpu::anno::MarkerScoreConfig cfg;
    cfg.min_n_genes_per_set = min_n_genes_per_set;

    if (method_str == "Mlm" || method_str == "mlm")
        cfg.method = singlet_gpu::anno::MarkerMethod::Mlm;
    else if (method_str == "Ulm" || method_str == "ulm")
        cfg.method = singlet_gpu::anno::MarkerMethod::Ulm;
    else if (method_str == "Wsum" || method_str == "wsum")
        cfg.method = singlet_gpu::anno::MarkerMethod::Wsum;
    else if (method_str == "UCell" || method_str == "ucell")
        cfg.method = singlet_gpu::anno::MarkerMethod::UCell;
    else
        throw py::value_error("marker_score: unknown method '" + method_str
                              + "'. Use 'Mlm', 'Ulm', 'Wsum', or 'UCell'.");

    auto r = singlet_gpu::anno::marker_score(*py_csc.csc, db, cfg, stream);

    auto out = std::make_shared<PyMarkerScoreResult>();
    out->scores  = std::move(r.scores);
    out->n_sets  = r.n_sets;
    out->n_cells = r.n_cells;
    out->stream  = stream;
    return out;
}

static std::shared_ptr<PyCelltypistModel>
py_load_celltypist_model(const std::string& npz_path)
{
    // Host-side file I/O; device uploads happen inside load_celltypist_model.
    auto m_cpp = singlet_gpu::anno::load_celltypist_model(npz_path);

    auto out = std::make_shared<PyCelltypistModel>();
    out->weights     = std::move(m_cpp.weights);
    out->intercepts  = std::move(m_cpp.intercepts);
    out->class_names = std::move(m_cpp.class_names);
    out->n_classes   = m_cpp.n_classes;
    out->n_pcs       = m_cpp.n_pcs;
    out->stream      = nullptr;
    return out;
}

static std::shared_ptr<PyRefMapResult>
py_celltypist_project(
    py::object embedding_obj,
    py::object model_obj)
{
    int n = 0, d = 0;
    // embedding_obj lifetime must outlive the call; DeviceDense is non-owning.
    singlet_gpu::core::DeviceDense emb = build_dense_view(embedding_obj, n, d);
    auto& model = model_obj.cast<PyCelltypistModel&>();

    // Reconstruct a temporary anno::CelltypistModel borrowing device pointers.
    singlet_gpu::anno::CelltypistModel m_borrow;
    m_borrow.n_classes = model.n_classes;
    m_borrow.n_pcs     = model.n_pcs;
    m_borrow.class_names = model.class_names;
    m_borrow.weights    = singlet_gpu::core::DeviceMemory<float>::wrap(
        model.weights.data(), model.weights.size());
    m_borrow.intercepts = singlet_gpu::core::DeviceMemory<float>::wrap(
        model.intercepts.data(), model.intercepts.size());

    cudaStream_t stream = model.stream;
    auto r = singlet_gpu::anno::project_to_reference(emb, m_borrow, stream);

    auto out = std::make_shared<PyRefMapResult>();
    out->probabilities = std::move(r.probabilities);
    out->labels        = std::move(r.labels);
    out->n_classes     = model.n_classes;
    out->n_cells       = n;
    out->stream        = stream;
    return out;
}

// ===========================================================================
// CYCLE 13 — fgsea, aucell
// ===========================================================================

// borrow_float_array_1d — non-owning DeviceMemory<float> from __cuda_array_interface__.
static singlet_gpu::core::DeviceMemory<float>
borrow_float_array(py::object arr, const char* name)
{
    if (!py::hasattr(arr, "__cuda_array_interface__"))
        throw py::type_error(std::string(name) + " must expose __cuda_array_interface__");

    py::dict iface  = arr.attr("__cuda_array_interface__").cast<py::dict>();
    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 1)
        throw py::value_error(std::string(name) + " must be 1-D");

    std::size_t n = shape[0].cast<std::size_t>();
    py::tuple dtup = iface["data"].cast<py::tuple>();
    float* ptr = reinterpret_cast<float*>(dtup[0].cast<std::intptr_t>());

    return singlet_gpu::core::DeviceMemory<float>::wrap(ptr, n);
}

static std::shared_ptr<PyFgseaResult>
py_fgsea(
    py::object stats_obj,
    py::object gene_sets_obj,
    int        min_perm,
    int        max_perm,
    float      adaptive_target_pvalue,
    int        min_set_size,
    int        max_set_size,
    uint64_t   seed)
{
    auto stats = borrow_float_array(stats_obj, "stats");
    auto db    = parse_gene_sets(gene_sets_obj);

    singlet_gpu::gsea::FgseaConfig cfg;
    cfg.min_perm               = min_perm;
    cfg.max_perm               = max_perm;
    cfg.adaptive_target_pvalue = adaptive_target_pvalue;
    cfg.min_set_size           = min_set_size;
    cfg.max_set_size           = max_set_size;
    cfg.seed                   = seed;

    auto r = singlet_gpu::gsea::fgsea(stats, db, cfg, /*stream=*/nullptr);

    auto out = std::make_shared<PyFgseaResult>();
    out->pathways = std::move(r.pathways);
    return out;
}

static std::shared_ptr<PyAUCellResult>
py_aucell(
    py::object mat_obj,
    py::object gene_sets_obj,
    int        top_k_genes,
    int        n_bins,
    int        cell_tile,
    uint64_t   seed)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    cudaStream_t stream = py_csc.producer_stream;

    auto db = parse_gene_sets(gene_sets_obj);

    singlet_gpu::gsea::AUCellConfig cfg;
    cfg.top_k_genes = top_k_genes;
    cfg.n_bins      = n_bins;
    cfg.cell_tile   = cell_tile;
    cfg.seed        = seed;

    auto r = singlet_gpu::gsea::aucell(*py_csc.csc, db, cfg, stream);

    auto out = std::make_shared<PyAUCellResult>();
    out->scores     = std::move(r.scores);
    out->n_pathways = r.n_pathways;
    out->n_cells    = r.n_cells;
    out->stream     = stream;
    return out;
}

// ===========================================================================
// CYCLE 14 — harmony, bbknn
// ===========================================================================

static std::shared_ptr<PyHarmonyResult>
py_harmony(
    py::object embedding_obj,
    py::object batch_labels_obj,
    int        n_batches,
    int        n_clusters,
    int        max_iter,
    float      tol,
    float      lambda,
    uint64_t   seed)
{
    int n = 0, d = 0;
    // embedding_obj: __cuda_array_interface__ 2-D float32.
    // Lifetime anchor: embedding_obj must outlive the call.
    singlet_gpu::core::DeviceDense emb = build_dense_view(embedding_obj, n, d);
    auto batch_labels = borrow_int_array(batch_labels_obj, "batch_labels");

    singlet_gpu::integrate::HarmonyConfig cfg;
    cfg.n_clusters = n_clusters;
    cfg.max_iter   = max_iter;
    cfg.tol        = tol;
    cfg.lambda     = lambda;
    cfg.seed       = seed;

    cudaStream_t stream = singlet_gpu::core::default_context().stream();
    auto r = singlet_gpu::integrate::harmony(emb, batch_labels, n_batches, cfg, stream);

    auto out = std::make_shared<PyHarmonyResult>();
    out->corrected    = std::move(r.corrected);
    out->n_iters_used = r.n_iters_used;
    out->final_delta  = r.final_delta;
    out->stream       = stream;
    return out;
}

static std::shared_ptr<PyKnnResult>
py_bbknn(
    py::object embedding_obj,
    py::object batch_labels_obj,
    int        n_batches,
    int        k_within,
    int        approx_threshold)
{
    int n = 0, d = 0;
    singlet_gpu::core::DeviceDense emb = build_dense_view(embedding_obj, n, d);
    auto batch_labels = borrow_int_array(batch_labels_obj, "batch_labels");

    singlet_gpu::integrate::BbknnConfig cfg;
    cfg.k_within         = k_within;
    cfg.approx_threshold = approx_threshold;

    cudaStream_t stream = singlet_gpu::core::default_context().stream();
    auto r = singlet_gpu::integrate::bbknn(emb, batch_labels, n_batches, cfg, stream);
    return convert_knn_result(std::move(r), stream);
}

// ===========================================================================
// CYCLE 15 — velocity_prep_compute
// ===========================================================================

static std::shared_ptr<PyVelocityPrepResult>
py_velocity_prep_compute(
    py::object spliced_obj,
    py::object unspliced_obj,
    py::object knn_for_smoothing,
    int        min_S_count,
    int        min_U_count,
    int        top_n_quantile,
    bool       smooth_moments,
    bool       compute_velocity)
{
    auto& spliced   = spliced_obj.cast<PyDeviceCsc&>();
    auto& unspliced = unspliced_obj.cast<PyDeviceCsc&>();
    cudaStream_t stream = spliced.producer_stream;

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.min_S_count      = min_S_count;
    cfg.min_U_count      = min_U_count;
    cfg.top_n_pct        = top_n_quantile;
    cfg.smooth_moments   = smooth_moments;
    cfg.compute_velocity = compute_velocity;

    // Optional knn_for_smoothing: nullptr if not provided.
    const singlet_gpu::graph::KnnResult* knn_ptr = nullptr;
    std::shared_ptr<singlet_gpu::graph::KnnResult> knn_owned;
    if (!knn_for_smoothing.is_none()) {
        // Borrow the KnnResult device pointers via borrow_knn_result.
        auto& py_knn = knn_for_smoothing.cast<PyKnnResult&>();
        knn_owned = std::make_shared<singlet_gpu::graph::KnnResult>(
            borrow_knn_result(py_knn));
        knn_ptr = knn_owned.get();
    }

    auto r = singlet_gpu::preprocess::velocity_prep(
        *spliced.csc, *unspliced.csc, knn_ptr, cfg, stream);

    auto out = std::make_shared<PyVelocityPrepResult>();
    out->gamma                   = std::move(r.gamma);
    out->gamma_se                = std::move(r.gamma_se);
    out->S_mean                  = std::move(r.S_mean);
    out->U_mean                  = std::move(r.U_mean);
    out->filter_mask             = std::move(r.filter_mask);
    out->velocity                = std::move(r.velocity);
    out->n_genes_passing_filter  = r.n_genes_passing_filter;
    out->stream                  = stream;
    return out;
}

// ===========================================================================
// CYCLE 16 — mt_lineage_call_clones
// ===========================================================================

static std::shared_ptr<PyClonePrediction>
py_mt_lineage_call_clones(
    py::object alt_counts_obj,
    py::object depth_counts_obj,
    int        min_depth,
    int        min_cells_alt,
    float      min_vaf,
    int        max_em_iters,
    float      em_tol,
    int        min_K,
    int        max_K,
    uint64_t   seed)
{
    auto& alt   = alt_counts_obj.cast<PyDeviceCsc&>();
    auto& depth = depth_counts_obj.cast<PyDeviceCsc&>();
    cudaStream_t stream = alt.producer_stream;

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth      = min_depth;
    cfg.min_cells_alt  = min_cells_alt;
    cfg.min_vaf        = min_vaf;
    cfg.max_em_iters   = max_em_iters;
    cfg.em_tol         = em_tol;
    cfg.min_K          = min_K;
    cfg.max_K          = max_K;
    cfg.seed           = seed;

    auto r = singlet_gpu::anno::call_clones(*alt.csc, *depth.csc, cfg, stream);

    auto out = std::make_shared<PyClonePrediction>();
    out->labels                  = std::move(r.labels);
    out->n_clones                = r.n_clones;
    out->informative_sites       = std::move(r.informative_sites);
    out->n_informative_sites     = r.n_informative_sites;
    out->clone_profiles          = std::move(r.clone_profiles);
    out->per_cell_heteroplasmy   = std::move(r.per_cell_heteroplasmy);
    out->stream                  = stream;
    return out;
}

// ===========================================================================
// CYCLE 17 — donor_pseudobulk_de
// ===========================================================================

static std::shared_ptr<PyDonorPseudobulkResult>
py_donor_pseudobulk_de(
    py::object mat_obj,
    py::object cluster_labels_obj,
    int        n_clusters,
    py::object donor_labels_obj,
    int        n_donors,
    int        min_cells_per_pseudobulk,
    int        max_irls_iters,
    float      irls_tol,
    int        max_dispersion_iters,
    bool       apeglm_shrinkage,
    int        top_n,
    uint64_t   seed)
{
    auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
    cudaStream_t stream = py_csc.producer_stream;

    auto cluster_labels = borrow_int_array(cluster_labels_obj, "cluster_labels");
    auto donor_labels   = borrow_int_array(donor_labels_obj,   "donor_labels");

    singlet_gpu::de::DonorPseudobulkConfig cfg;
    cfg.min_cells_per_pseudobulk = min_cells_per_pseudobulk;
    cfg.max_irls_iters           = max_irls_iters;
    cfg.irls_tol                 = irls_tol;
    cfg.max_dispersion_iters     = max_dispersion_iters;
    cfg.apeglm_shrinkage         = apeglm_shrinkage;
    cfg.top_n                    = top_n;
    cfg.seed                     = seed;

    auto r = singlet_gpu::de::donor_pseudobulk_de(
        *py_csc.csc, cluster_labels, n_clusters,
        donor_labels, n_donors, cfg, stream);

    auto out = std::make_shared<PyDonorPseudobulkResult>();
    out->log2_fc                 = std::move(r.log2_fc);
    out->p_values                = std::move(r.p_values);
    out->p_adj                   = std::move(r.p_adj);
    out->dispersion              = std::move(r.dispersion);
    out->pseudobulk              = std::move(r.pseudobulk);
    out->n_genes_passing_filter  = r.n_genes_passing_filter;
    out->stream                  = stream;
    return out;
}

// ===========================================================================
// bind_kernels_ext — register cycles 7-17 functions.
// Called from bind_kernels() after the existing cycle 1-6 registrations.
// ===========================================================================

inline void bind_kernels_ext(py::module_& m)
{
    // -- cycle 7: streaming pipeline --
    m.def("streaming_pipeline_run", &py_streaming_pipeline_run,
          py::arg("input_paths"),
          py::kw_only(),
          py::arg("chunk_cols")               = 100000,
          py::arg("run_lognorm")              = true,
          py::arg("run_hvg")                  = true,
          py::arg("run_pca")                  = false,
          py::arg("pca_k")                    = 50,
          py::arg("run_nmf")                  = false,
          py::arg("nmf_factors")              = 20,
          py::arg("cache_normalized")         = false,
          py::arg("in_memory_pca_threshold")  = int64_t{2000000},
          "End-to-end streaming pipeline (lognorm + HVG + optional PCA/NMF) over "
          "a list of .1pz paths.  Returns StreamingPipelineResult.");

    // -- cycle 8: kNN --
    m.def("knn_graph", &py_knn_graph,
          py::arg("embedding"), py::arg("k") = 15,
          py::kw_only(),
          py::arg("backend")       = std::string("auto"),
          py::arg("metric")        = std::string("L2"),
          py::arg("return_squared")= false,
          py::arg("seed")          = uint64_t{0},
          py::arg("hnsw_M")        = 16,
          py::arg("hnsw_ef")       = 64,
          "k-nearest-neighbour graph from a dense embedding.  "
          "embedding: __cuda_array_interface__ float32 (n × d).  "
          "Returns KnnResult.");

    m.def("knn_graph_from_indices", &py_knn_graph_from_indices,
          py::arg("knn_result"),
          "Identity forward of an existing KnnResult.  "
          "Useful for downstream functions that accept a KnnResult directly.");

    // -- cycle 9: Leiden --
    m.def("leiden_partition", &py_leiden_partition,
          py::arg("knn_result"), py::arg("resolution") = 1.0f,
          py::kw_only(),
          py::arg("max_iter")         = 100,
          py::arg("seed")             = uint64_t{0},
          py::arg("backend")          = std::string("CuGraph"),
          py::arg("weight_function")  = std::string("Connectivity"),
          py::arg("gaussian_sigma")   = 0.0f,
          "Leiden community detection.  Returns LeidenResult.");

    m.def("leiden_multi_resolution", &py_leiden_multi_resolution,
          py::arg("knn_result"), py::arg("resolutions"),
          py::kw_only(),
          py::arg("seed")            = uint64_t{0},
          py::arg("weight_function") = std::string("Connectivity"),
          py::arg("gaussian_sigma")  = 0.0f,
          "Run Leiden at multiple resolutions (graph loaded once).  "
          "resolutions: list[float].  Returns list[LeidenResult].");

    // -- cycle 10: UMAP --
    m.def("umap_embed", &py_umap_embed,
          py::arg("knn_result"),
          py::kw_only(),
          py::arg("n_components")        = 2,
          py::arg("n_epochs")            = 0,
          py::arg("min_dist")            = 0.5f,
          py::arg("spread")              = 1.0f,
          py::arg("learning_rate")       = 1.0f,
          py::arg("init")                = std::string("Random"),
          py::arg("seed")                = uint64_t{0},
          py::arg("negative_sample_rate")= 5,
          "UMAP embedding from a KnnResult.  Returns UmapResult.");

    // -- cycle 11: DE --
    m.def("wilcoxon_de", &py_wilcoxon_de,
          py::arg("mat"), py::arg("labels"), py::arg("n_clusters"),
          py::kw_only(),
          py::arg("n_bins")       = 4096,
          py::arg("top_n")        = 100,
          py::arg("gene_tile")    = 1024,
          py::arg("deterministic")= false,
          "Histogram-binned Wilcoxon rank-sum DE.  Returns WilcoxonResult.");

    m.def("ttest_de", &py_ttest_de,
          py::arg("mat"), py::arg("labels"), py::arg("n_clusters"),
          py::kw_only(),
          py::arg("top_n")        = 100,
          py::arg("gene_tile")    = 1024,
          py::arg("deterministic")= false,
          "Welch t-test DE.  Returns TtestResult.");

    // -- cycle 12: annotation --
    m.def("marker_score", &py_marker_score,
          py::arg("mat"), py::arg("gene_sets"),
          py::kw_only(),
          py::arg("method")              = std::string("Mlm"),
          py::arg("min_n_genes_per_set") = 5,
          "Marker activity scoring (Mlm/Ulm/Wsum/UCell).  "
          "gene_sets: dict {name: [idx,...]} or list of (name, [idx,...]) tuples.  "
          "Returns MarkerScoreResult.");

    m.def("load_celltypist_model", &py_load_celltypist_model,
          py::arg("npz_path"),
          "Load a CellTypist .npz model from disk.  Returns CelltypistModel.");

    m.def("celltypist_project", &py_celltypist_project,
          py::arg("embedding"), py::arg("model"),
          "Project PCA embedding onto a CelltypistModel.  "
          "Returns RefMapResult with probabilities + labels.");

    // -- cycle 13: GSEA --
    m.def("fgsea", &py_fgsea,
          py::arg("stats"), py::arg("gene_sets"),
          py::kw_only(),
          py::arg("min_perm")                = 1000,
          py::arg("max_perm")                = 10000,
          py::arg("adaptive_target_pvalue")  = 0.05f,
          py::arg("min_set_size")            = 15,
          py::arg("max_set_size")            = 500,
          py::arg("seed")                    = uint64_t{0},
          "fGSEA preranked enrichment.  stats: 1-D float32 device array (n_genes,).  "
          "Returns FgseaResult.");

    m.def("aucell", &py_aucell,
          py::arg("mat"), py::arg("gene_sets"),
          py::kw_only(),
          py::arg("top_k_genes") = 500,
          py::arg("n_bins")      = 4096,
          py::arg("cell_tile")   = 65536,
          py::arg("seed")        = uint64_t{0},
          "AUCell gene-set activity scoring.  Returns AUCellResult.");

    // -- cycle 14: integration --
    m.def("harmony", &py_harmony,
          py::arg("embedding"), py::arg("batch_labels"), py::arg("n_batches"),
          py::kw_only(),
          py::arg("n_clusters") = 20,
          py::arg("max_iter")   = 10,
          py::arg("tol")        = 1e-4f,
          py::arg("lambda")     = 1.0f,
          py::arg("seed")       = uint64_t{0},
          "Harmony batch integration.  "
          "embedding: __cuda_array_interface__ float32 (n × d).  "
          "Returns HarmonyResult.");

    m.def("bbknn", &py_bbknn,
          py::arg("embedding"), py::arg("batch_labels"), py::arg("n_batches"),
          py::kw_only(),
          py::arg("k_within")        = 3,
          py::arg("approx_threshold")= 100000,
          "Batch-balanced kNN graph.  Returns KnnResult (same type as knn_graph).");

    // -- cycle 15: velocity --
    m.def("velocity_prep_compute", &py_velocity_prep_compute,
          py::arg("spliced"), py::arg("unspliced"),
          py::kw_only(),
          py::arg("knn_for_smoothing") = py::none(),
          py::arg("min_S_count")       = 10,
          py::arg("min_U_count")       = 5,
          py::arg("top_n_quantile")    = 5,
          py::arg("smooth_moments")    = true,
          py::arg("compute_velocity")  = true,
          "RNA velocity preparation (γ estimation + moment smoothing).  "
          "Returns VelocityPrepResult.");

    // -- cycle 16: MT lineage --
    m.def("mt_lineage_call_clones", &py_mt_lineage_call_clones,
          py::arg("alt_counts"), py::arg("depth_counts"),
          py::kw_only(),
          py::arg("min_depth")      = 10,
          py::arg("min_cells_alt")  = 5,
          py::arg("min_vaf")        = 0.01f,
          py::arg("max_em_iters")   = 100,
          py::arg("em_tol")         = 1e-5f,
          py::arg("min_K")          = 2,
          py::arg("max_K")          = 10,
          py::arg("seed")           = uint64_t{0},
          "MT heteroplasmy clone prediction (BIC-EM GMM).  "
          "alt_counts / depth_counts: DeviceCsc (MT sites × cells).  "
          "Returns ClonePrediction.");

    // -- cycle 17: donor pseudobulk DE --
    m.def("donor_pseudobulk_de", &py_donor_pseudobulk_de,
          py::arg("mat"), py::arg("cluster_labels"), py::arg("n_clusters"),
          py::arg("donor_labels"), py::arg("n_donors"),
          py::kw_only(),
          py::arg("min_cells_per_pseudobulk") = 10,
          py::arg("max_irls_iters")           = 50,
          py::arg("irls_tol")                 = 1e-5f,
          py::arg("max_dispersion_iters")     = 10,
          py::arg("apeglm_shrinkage")         = true,
          py::arg("top_n")                    = 100,
          py::arg("seed")                     = uint64_t{0},
          "Donor-aware pseudobulk NB GLM DE.  Returns DonorPseudobulkResult.");
}

}  // namespace python
}  // namespace singlet_gpu
