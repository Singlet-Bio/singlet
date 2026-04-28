// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/graph/graph_all.hpp — FactorGraph<float>, graph::fit<float>
//             factornet/graph/node.hpp      — InputNode, NMFLayerNode, SVDLayerNode,
//                                             SharedNode, ConcatNode, AddNode, ConditionNode
//             factornet/graph/result.hpp    — GraphResult<float>, LayerResult<float>
//
// Adapter: singlet_gpu::reduce::nmf::graph
//
// The FactorGraph abstraction enables hierarchical and multi-modal NMF by wiring
// factornet node objects into a DAG.  This header is a thin re-export layer:
//   - Node type aliases in singlet_gpu::reduce::nmf::graph namespace.
//   - fit_graph() adapter that constructs an Eigen::Map view of a PzDeviceMatrix
//     and calls factornet::graph::fit().
//
// IMPORTANT: FactorGraph does NOT have a built-in shared-H joint factorization API.
//   Multi-modal joint NMF (RNA + ATAC + ADT sharing cell embedding) is built by
//   concatenating feature axes with SharedNode or ConcatNode, then applying a
//   single NMFLayerNode.  The SharedNode pattern is the preferred approach:
//
//   Shared-H multi-modal (SharedNode — recommended for RNA+ATAC+ADT):
//     Eigen::SparseMatrix<float> rna_eig  = to_eigen(rna_pz);   // genes × cells
//     Eigen::SparseMatrix<float> atac_eig = to_eigen(atac_pz);  // peaks × cells
//     Eigen::SparseMatrix<float> adt_eig  = to_eigen(adt_pz);   // tags  × cells
//     InputNode<float, SpMat> rna_in (rna_eig,  "RNA");
//     InputNode<float, SpMat> atac_in(atac_eig, "ATAC");
//     InputNode<float, SpMat> adt_in (adt_eig,  "ADT");
//     SharedNode<float> shared({&rna_in, &atac_in, &adt_in});
//     NMFLayerNode<float> joint(&shared, 20, "joint");
//     FactorGraph<float> net({&rna_in, &atac_in, &adt_in}, &joint);
//     net.compile();
//     auto result = factornet::graph::fit(net, concat_eigen);
//     // result["joint"].W_splits["RNA"] — RNA-specific W rows
//     // result["joint"].H              — shared cell embedding (k × cells)
//
//   Hierarchical NMF (two-layer deep):
//     NMFLayerNode<float> L1(&inp, 64, "encoder");
//     NMFLayerNode<float> L2(&L1,  10, "bottleneck");
//     FactorGraph<float> net({&inp}, &L2);
//     net.compile();
//     auto result = factornet::graph::fit(net, A_eig);
//
// Eigen::Map alignment note (design doc §Risks):
//   Our pinned host buffers (cudaMallocHost) are 16-byte aligned, which satisfies
//   Eigen's default alignment requirement.  If alignment fails at runtime (assertion
//   in Eigen::Map constructor), fall back to constructing a full Eigen::SparseMatrix
//   from the host buffers (a copy, but only at graph-setup time, not in the hot loop).
//
// Streams: factornet graph::fit creates its own GPUContext internally.
//   No external stream is accepted.  CPU-only path for multi-layer BCD.
//
// Precision: fp32 throughout.
// OOC plan: single-layer graphs delegate to nmf_fit_gpu (OOC via chunked_fit).
//   Multi-layer graphs perform in-memory BCD — full matrix must fit in host RAM.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/graph/graph_all.hpp>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/svd/deflation.h>  // require_host_retained (canonical owner)

#include <Eigen/Sparse>
#include <stdexcept>

namespace singlet_gpu {
namespace reduce {
namespace nmf {
namespace graph {

// ---------------------------------------------------------------------------
// Node type aliases — expose factornet node types under singlet_gpu namespace.
// ---------------------------------------------------------------------------

// SpMat — the Eigen sparse matrix type expected by InputNode and fit().
// Must be ColMajor to match our CSC host buffers.
using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor, int>;

template<typename MatrixType = SpMat>
using InputNode    = ::factornet::graph::InputNode<float, MatrixType>;
using NMFLayerNode = ::factornet::graph::NMFLayerNode<float>;
using SVDLayerNode = ::factornet::graph::SVDLayerNode<float>;
using SharedNode   = ::factornet::graph::SharedNode<float>;
using ConcatNode   = ::factornet::graph::ConcatNode<float>;
using AddNode      = ::factornet::graph::AddNode<float>;
using ConditionNode= ::factornet::graph::ConditionNode<float>;
using FactorGraph  = ::factornet::graph::FactorGraph<float>;
using GraphResult  = ::factornet::graph::GraphResult<float>;
using LayerResult  = ::factornet::graph::LayerResult<float>;

// ---------------------------------------------------------------------------
// eigen_view_of — construct an Eigen::Map<SpMat> over PzDeviceMatrix host buffers.
//
// WHY Map not SparseMatrix: avoids copying the (potentially large) host CSC arrays.
// The Map shares memory with the pinned buffers — it is valid as long as the
// PzDeviceMatrix's shared_ptr keeps them alive (i.e., for the duration of fit_graph).
//
// Alignment: cudaMallocHost returns 16-byte aligned memory on all CUDA platforms,
// satisfying Eigen's alignment contract for Map.  If a runtime failure occurs
// (e.g., unusual CUDA driver version), construct a full SpMat as a fallback.
// ---------------------------------------------------------------------------
inline SpMat eigen_view_of(const io::PzDeviceMatrix& m)
{
    ::singlet_gpu::reduce::svd::require_host_retained(m, "nmf::graph::eigen_view_of");

    const int rows = static_cast<int>(m.mat.rows);
    const int cols = static_cast<int>(m.mat.cols);
    const int nnz  = static_cast<int>(m.mat.nnz);

    // Construct from raw pointers into pinned host buffers.
    // Eigen::SparseMatrix<float, ColMajor, int> constructor from existing CSC arrays.
    SpMat eig(rows, cols);
    eig.reserve(nnz);
    // Use Map-from-CSC form: Map(rows, cols, nnz, outer_ptr, inner_idx, values).
    // This is a non-owning view — no allocation.
    return Eigen::Map<const SpMat>(
        rows, cols, nnz,
        m.host_indptr.get(),   // outer (col) pointers, size cols+1
        m.host_indices.get(),  // inner (row) indices, size nnz
        m.host_values.get()    // values, size nnz
    );
}

// ---------------------------------------------------------------------------
// fit_graph — adapter for a pre-built FactorGraph.
//
// The caller constructs nodes, wires them together, calls net.compile(), then
// passes the graph + primary data matrix here.  For multi-modal graphs the
// primary data is the concatenated or first-input matrix; InputNodes hold
// references to their own Eigen sparse matrices (caller's responsibility).
//
// W_init: optional warm-start for the first layer's W (nullptr = random init).
// ---------------------------------------------------------------------------
inline GraphResult fit_graph(const FactorGraph& net,
                             const SpMat& data,
                             const DenseMatrix* W_init = nullptr)
{
    if (!net.is_compiled())
        throw std::runtime_error(
            "fit_graph: FactorGraph not compiled. Call net.compile() first.");

    return ::factornet::graph::fit<float, SpMat>(net, data, W_init);
}

// Convenience overload: takes PzDeviceMatrix directly (single-input graphs).
// Constructs the Eigen view internally; the view is valid for the duration of fit.
inline GraphResult fit_graph(const FactorGraph& net,
                             const io::PzDeviceMatrix& m,
                             const DenseMatrix* W_init = nullptr)
{
    SpMat eig = eigen_view_of(m);
    return fit_graph(net, eig, W_init);
}

}  // namespace graph
}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
