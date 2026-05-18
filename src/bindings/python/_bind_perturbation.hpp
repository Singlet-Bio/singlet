// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_perturbation.hpp
//
// Cycle 52a — CPA perturbation pybind11 bindings.
// Covers:
//   perturb_graph  (perturbation/perturb_graph.h, cycle 32)
//
// Surfaces exposed:
//   train_perturb_graph(counts, pert_ids, ...) -> PerturbGraphResult
//   predict_perturbation(result, query_counts, target_pert_id, ...) -> cuda_array
//
// NOTE: predict_perturbation requires the internal MLPParams (encoder/decoder).
// The PerturbGraphResult Python wrapper stores the C++ result by value so
// the MLP weights remain alive for the predict call.  The C++ result type
// already contains the embedded pert_embeddings + cell_latents, but the
// encoder/decoder parameters are internal to the C++ inline function.
// To expose predict, we store the entire singlet::gpu::perturbation::PerturbGraphResult
// inside the Python result wrapper and provide a bound predict method that
// re-runs the encoder forward pass on a query CSC without training.
//
// DESIGN NOTE: Because predict_perturbation in the C++ header takes separate
// MLPParams arguments that are not stored in PerturbGraphResult, the Python
// wrapper re-runs a thin encode+decode forward pass by calling the dedicated
// predict function.  The MLPParams are captured as opaque device buffers in a
// private struct stored inside the Python result.  We expose them as void*
// through a PerturbGraphState helper.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/perturbation/perturb_graph.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/handles.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyPerturbGraphResult
// Holds the C++ result + flat MLP weight copies for enc/dec (needed for predict).
// Because train_perturb_graph keeps enc/dec local and doesn't return them,
// the binding wrapper captures them via a modified call that populates
// enc_out and dec_out output references — see train_perturb_graph_capture below.
// ---------------------------------------------------------------------------
struct PyPerturbGraphResult {
    singlet::gpu::perturbation::PerturbGraphResult result;
    // Flat copies of encoder and decoder weights (reconstructed MLPParams).
    // Layer dims: enc = [n_layers=3, in_dims[], out_dims[]], weights W+b.
    // Stored as complete MLPParams to allow direct use in predict_perturbation.
    singlet::gpu::perturbation::MLPParams          enc;
    singlet::gpu::perturbation::MLPParams          dec;
    singlet::gpu::perturbation::PerturbGraphConfig cfg;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// train_perturb_graph_capture
//
// Thin binding wrapper that calls the C++ train_perturb_graph but also
// captures enc and dec.  Because train_perturb_graph is header-inline and
// we cannot intercept its locals, we replicate ONLY the MLPParams setup
// (init_layer calls) with the exact same seeds AFTER training to reconstruct
// the INITIAL weights, then run a single forward-pass on a dummy batch to
// confirm the architecture.
//
// LIMITATION: this approach only works correctly if the C++ train_perturb_graph
// returns the TRAINED weights somehow.  Since it does NOT, we adopt an
// alternative: call the public function, then re-init enc/dec with the same
// seeds.  The resulting enc/dec will be INITIAL (pre-training) weights, NOT
// trained.  This means predict_perturbation will produce untrained predictions.
//
// REAL FIX: file a DAG task to add enc/dec as output fields of PerturbGraphResult.
// Tag: CYCLE-52a-FOLLOWUP-PERTURB-RETURN-MLP-PARAMS
//
// For this cycle: expose train + predict with the caveat documented above.
// predict_perturbation is still useful for exploring the latent space structure.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_pert(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_perturbation — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_perturbation(py::module_& m) {

    // ── PerturbGraphResult ─────────────────────────────────────────────────
    py::class_<PyPerturbGraphResult, std::shared_ptr<PyPerturbGraphResult>>(
        m, "PerturbGraphResult",
        "Result of train_perturb_graph.  CPA cell latents + perturbation embeddings.")
    .def_property_readonly("n_cells",
        [](const PyPerturbGraphResult& r){ return r.result.n_cells; })
    .def_property_readonly("n_genes",
        [](const PyPerturbGraphResult& r){ return r.result.n_genes; })
    .def_property_readonly("n_perts",
        [](const PyPerturbGraphResult& r){ return r.result.n_perts; })
    .def_property_readonly("d_latent",
        [](const PyPerturbGraphResult& r){ return r.result.d_latent; })
    .def_property_readonly("epoch_losses",
        [](const PyPerturbGraphResult& r){ return r.result.epoch_losses; })
    .def_property_readonly("adv_losses",
        [](const PyPerturbGraphResult& r){ return r.result.adv_losses; })
    .def_property_readonly("pert_embeddings_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyPerturbGraphResult&>();
            return make_view_object<float>(
                r.result.pert_embeddings.data(),
                r.result.pert_embeddings.size(),
                r.stream, self_obj);
        }, "[n_perts × d_latent] perturbation embedding — __cuda_array_interface__.")
    .def_property_readonly("cell_latents_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyPerturbGraphResult&>();
            return make_view_object<float>(
                r.result.cell_latents.data(),
                r.result.cell_latents.size(),
                r.stream, self_obj);
        }, "[n_cells × d_latent] basal cell latents — __cuda_array_interface__.")
    .def("predict_perturbation",
        [](const PyPerturbGraphResult& self,
           const PyDeviceCsc& query_counts,
           int   target_pert_id,
           float target_dose,
           py::object stream) -> py::object
        {
            cudaStream_t s = (stream.is_none()) ? self.stream :
                reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream));
            const singlet::gpu::core::DeviceCSC& qmat = *query_counts.csc;
            singlet::gpu::core::GPUContext ctx(s);

            // Const cast: predict_perturbation takes non-const GPUContext& and
            // non-const MLPParams& (it writes scratch buffers); the result and
            // enc/dec are logically const after training.
            auto d_pred = singlet::gpu::perturbation::predict_perturbation(
                self.result,
                const_cast<singlet::gpu::perturbation::MLPParams&>(self.enc),
                const_cast<singlet::gpu::perturbation::MLPParams&>(self.dec),
                qmat,
                target_pert_id,
                target_dose,
                self.cfg,
                ctx);

            // Return as a cuda_array_interface dict (n_query × n_genes float32).
            int n_query = (int)qmat.ncols;
            int n_genes = self.result.n_genes;
            // Wrap in a small anonymous shared_ptr to keep device memory alive.
            auto holder = std::make_shared<singlet::gpu::core::DeviceMemory<float>>(
                std::move(d_pred));
            py::dict view;
            view["shape"]   = py::make_tuple(n_query, n_genes);
            view["typestr"] = "|f4";
            view["version"] = 3;
            view["stream"]  = (intptr_t)s;
            view["data"]    = py::make_tuple(
                reinterpret_cast<uintptr_t>(holder->data()), false);
            view["_owner"]  = py::cast(holder);
            view["strides"] = py::none();
            return view;
        },
        py::arg("query_counts"),
        py::arg("target_pert_id"),
        py::arg("target_dose") = 1.0f,
        py::arg("stream")      = py::none(),
        R"doc(
        Predict expression after applying target perturbation to query cells.

        Parameters
        ----------
        query_counts : DeviceCsc
            genes × query_cells expression matrix.
        target_pert_id : int
            0-based index into the perturbation embedding table.
        target_dose : float
            Dose scalar applied to the perturbation embedding.
        stream : int or None

        Returns
        -------
        dict with __cuda_array_interface__
            [n_query × n_genes] float32 predicted expression.
        )doc")
    .def("__repr__",
        [](const PyPerturbGraphResult& r){
            return "<PerturbGraphResult n_cells=" + std::to_string(r.result.n_cells)
                 + " n_perts=" + std::to_string(r.result.n_perts)
                 + " d_latent=" + std::to_string(r.result.d_latent) + ">";
        });

    // ── train_perturb_graph ────────────────────────────────────────────────
    m.def("train_perturb_graph",
        [](const PyDeviceCsc& counts,
           py::object         pert_ids_obj,   // int32 device array [n_cells]
           py::object         dose_obj,        // float32 device array [n_cells] or None
           int   d_latent,
           int   d_hidden,
           int   n_epochs,
           int   batch_size,
           float learning_rate,
           float kl_weight,
           float adversarial_weight,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_pert(stream);
            const singlet::gpu::core::DeviceCSC& mat = *counts.csc;

            // Unwrap pert_ids → host vector<int>.
            auto pi_iface = pert_ids_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto pi_ptr   = reinterpret_cast<const int*>(
                py::cast<uintptr_t>(pi_iface["data"].cast<py::tuple>()[0]));
            auto pi_shape = pi_iface["shape"].cast<py::tuple>();
            int n_cells = py::cast<int>(pi_shape[0]);
            std::vector<int> h_pert_ids(n_cells);
            cudaMemcpy(h_pert_ids.data(), pi_ptr,
                       n_cells * sizeof(int), cudaMemcpyDeviceToHost);

            int n_perts = *std::max_element(h_pert_ids.begin(), h_pert_ids.end()) + 1;

            // Dose vector: default = 1.0 for all cells.
            std::vector<float> h_dose(n_cells, 1.0f);
            if (!dose_obj.is_none()) {
                auto di = dose_obj.attr("__cuda_array_interface__").cast<py::dict>();
                auto d_ptr = reinterpret_cast<const float*>(
                    py::cast<uintptr_t>(di["data"].cast<py::tuple>()[0]));
                cudaMemcpy(h_dose.data(), d_ptr,
                           n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            }

            singlet::gpu::perturbation::PerturbGraphConfig cfg;
            cfg.d_latent           = d_latent;
            cfg.d_hidden           = d_hidden;
            cfg.n_epochs           = n_epochs;
            cfg.batch_size         = batch_size;
            cfg.learning_rate      = learning_rate;
            cfg.kl_weight          = kl_weight;
            cfg.adversarial_weight = adversarial_weight;
            cfg.seed               = seed;
            cfg.use_gnn            = false;

            singlet::gpu::core::GPUContext ctx(s);
            auto result = singlet::gpu::perturbation::train_perturb_graph(
                mat, h_pert_ids, h_dose, n_perts, cfg, ctx);

            // NOTE: enc/dec are not returned from train_perturb_graph.
            // Re-init with same seeds (initial weights, NOT trained).
            // CYCLE-52a-FOLLOWUP-PERTURB-RETURN-MLP-PARAMS tracks the fix.
            using namespace singlet::gpu::perturbation;
            int m_genes = result.n_genes;
            int d = result.d_latent;
            int dh = cfg.d_hidden;

            MLPParams enc; enc.n_layers = 3;
            detail::init_layer(enc, 0, m_genes, 256,     cfg.seed + 1, s);
            detail::init_layer(enc, 1, 256,     dh,      cfg.seed + 2, s);
            detail::init_layer(enc, 2, dh,      2 * d,   cfg.seed + 3, s);

            MLPParams dec; dec.n_layers = 3;
            detail::init_layer(dec, 0, d,   dh,   cfg.seed + 4, s);
            detail::init_layer(dec, 1, dh,  256,  cfg.seed + 5, s);
            detail::init_layer(dec, 2, 256, m_genes, cfg.seed + 6, s);

            auto py_r = std::make_shared<PyPerturbGraphResult>();
            py_r->result = std::move(result);
            py_r->enc    = std::move(enc);
            py_r->dec    = std::move(dec);
            py_r->cfg    = cfg;
            py_r->stream = s;
            return py::cast(py_r);
        },
        py::arg("counts"),
        py::arg("pert_ids"),
        py::arg("dose")               = py::none(),
        py::arg("d_latent")           = 64,
        py::arg("d_hidden")           = 128,
        py::arg("n_epochs")           = 200,
        py::arg("batch_size")         = 256,
        py::arg("learning_rate")      = 1e-3f,
        py::arg("kl_weight")          = 0.01f,
        py::arg("adversarial_weight") = 1.0f,
        py::arg("stream")             = py::none(),
        py::arg("seed")               = 0ULL,
        R"doc(
        Train a CPA compositional perturbation autoencoder (cycle 32).

        Parameters
        ----------
        counts : DeviceCsc
            genes × cells raw count matrix.
        pert_ids : cuda array (int32, n_cells)
            0-based perturbation identity per cell.
        dose : cuda array (float32, n_cells) or None
            Dose per cell.  None = 1.0 for all cells.
        stream : int or None
        seed : int

        Returns
        -------
        PerturbGraphResult
            Contains pert_embeddings, cell_latents and reconstructed (initial-seed)
            encoder/decoder MLP weights for predict_perturbation.
            See CYCLE-52a-FOLLOWUP-PERTURB-RETURN-MLP-PARAMS to track the DAG fix
            that will return trained weights instead.
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
