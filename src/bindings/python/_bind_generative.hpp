// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_generative.hpp
//
// Cycle 52a — Generative model pybind11 bindings.
// Covers:
//   discrete_diffusion  (generative/discrete_diffusion.h, cycle 30)
//
// Surfaces exposed:
//   train_discrete_diffusion(counts, ...) -> DiscreteDiffusionResult
//   sample_cells(model, n_samples, ...) -> cuda_array_interface (float32)
//
// The trained model weights are host-side (std::vector<float>) so they can
// be serialised / moved across Python without device lifetime concerns.
// The sample_cells output is device-resident DeviceMemory<float>.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/generative/discrete_diffusion.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <string>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyDiscreteDiffusionResult
// Host-side model weights + metadata for serialisation; device sampling
// buffer returned separately from sample_cells.
// ---------------------------------------------------------------------------
struct PyDiscreteDiffusionResult {
    std::vector<float>  model_weights;
    std::vector<int>    model_shape;
    std::vector<float>  loss_history;
    std::vector<float>  bin_edges;
    int                 n_epochs_used = 0;
    // Cached config so sample_cells can be called without re-specifying defaults.
    singlet::gpu::generative::DiscreteDiffusionConfig cfg;
};

// ---------------------------------------------------------------------------
// PyDDSamples — wraps the device-resident float32 sample matrix.
// ---------------------------------------------------------------------------
struct PyDDSamples {
    singlet::gpu::core::DeviceMemory<float> data;  // n_samples × n_genes, row-major
    int n_samples = 0;
    int n_genes   = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// Redeclared here as a local so this header is self-contained.
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_gen(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_generative — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_generative(py::module_& m) {

    // ── DiscreteDiffusionResult ────────────────────────────────────────────
    py::class_<PyDiscreteDiffusionResult,
               std::shared_ptr<PyDiscreteDiffusionResult>>(
        m, "DiscreteDiffusionResult",
        "Trained D3PM discrete-diffusion model.  Weights are host-side (serialisable).")
    .def_property_readonly("model_weights",
        [](const PyDiscreteDiffusionResult& r){ return r.model_weights; },
        "Flat host float32 vector of all model parameters.")
    .def_property_readonly("model_shape",
        [](const PyDiscreteDiffusionResult& r){ return r.model_shape; },
        "Shape table (flat int32): [layer_id, param_id, rows, cols, ...].")
    .def_property_readonly("loss_history",
        [](const PyDiscreteDiffusionResult& r){ return r.loss_history; },
        "Mean masked cross-entropy per training epoch.")
    .def_property_readonly("bin_edges",
        [](const PyDiscreteDiffusionResult& r){ return r.bin_edges; },
        "Token bin boundaries [vocab_size+1].")
    .def_property_readonly("n_epochs_used",
        [](const PyDiscreteDiffusionResult& r){ return r.n_epochs_used; })
    .def("__repr__",
        [](const PyDiscreteDiffusionResult& r){
            return "<DiscreteDiffusionResult epochs=" + std::to_string(r.n_epochs_used)
                 + " n_params=" + std::to_string(r.model_weights.size()) + ">";
        });

    // ── DDSamples ──────────────────────────────────────────────────────────
    py::class_<PyDDSamples, std::shared_ptr<PyDDSamples>>(
        m, "DDSamples",
        "Sampled cells from a trained DiscreteDiffusionResult.")
    .def_property_readonly("n_samples",
        [](const PyDDSamples& r){ return r.n_samples; })
    .def_property_readonly("n_genes",
        [](const PyDDSamples& r){ return r.n_genes; })
    .def_property_readonly("data_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyDDSamples&>();
            return make_view_object<float>(r.data.get(), r.data.size(),
                                           r.stream, self_obj);
        }, "[n_samples × n_genes] float32 reconstructed expression — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyDDSamples& r){
            return "<DDSamples n_samples=" + std::to_string(r.n_samples)
                 + " n_genes=" + std::to_string(r.n_genes) + ">";
        });

    // ── train_discrete_diffusion ───────────────────────────────────────────
    m.def("train_discrete_diffusion",
        [](const PyDeviceCsc& counts,
           int    n_timesteps,
           int    vocab_size,
           int    d_hidden,
           int    n_heads,
           int    n_layers,
           int    n_epochs,
           int    batch_size,
           float  learning_rate,
           std::string token_binning,
           float  beta_start,
           float  beta_end,
           int    d_ffn_mult,
           float  adam_beta1,
           float  adam_beta2,
           float  adam_eps,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_gen(stream);
            const singlet::gpu::core::DeviceCSC& mat = *counts.csc;

            singlet::gpu::generative::DiscreteDiffusionConfig cfg;
            cfg.n_timesteps   = n_timesteps;
            cfg.vocab_size    = vocab_size;
            cfg.d_hidden      = d_hidden;
            cfg.n_heads       = n_heads;
            cfg.n_layers      = n_layers;
            cfg.n_epochs      = n_epochs;
            cfg.batch_size    = batch_size;
            cfg.learning_rate = learning_rate;
            cfg.token_binning = token_binning;
            cfg.seed          = seed;
            cfg.beta_start    = beta_start;
            cfg.beta_end      = beta_end;
            cfg.d_ffn_mult    = d_ffn_mult;
            cfg.adam_beta1    = adam_beta1;
            cfg.adam_beta2    = adam_beta2;
            cfg.adam_eps      = adam_eps;

            auto r = singlet::gpu::generative::train_discrete_diffusion(mat, cfg, s);

            auto py_r = std::make_shared<PyDiscreteDiffusionResult>();
            py_r->model_weights  = std::move(r.model_weights);
            py_r->model_shape    = std::move(r.model_shape);
            py_r->loss_history   = std::move(r.loss_history);
            py_r->bin_edges      = std::move(r.bin_edges);
            py_r->n_epochs_used  = r.n_epochs_used;
            py_r->cfg            = cfg;
            return py::cast(py_r);
        },
        py::arg("counts"),
        py::arg("n_timesteps")   = 100,
        py::arg("vocab_size")    = 16,
        py::arg("d_hidden")      = 256,
        py::arg("n_heads")       = 4,
        py::arg("n_layers")      = 4,
        py::arg("n_epochs")      = 200,
        py::arg("batch_size")    = 256,
        py::arg("learning_rate") = 1e-4f,
        py::arg("token_binning") = "log2",
        py::arg("beta_start")    = 1e-4f,
        py::arg("beta_end")      = 0.02f,
        py::arg("d_ffn_mult")    = 4,
        py::arg("adam_beta1")    = 0.9f,
        py::arg("adam_beta2")    = 0.999f,
        py::arg("adam_eps")      = 1e-8f,
        py::arg("stream")        = py::none(),
        py::arg("seed")          = 0ULL,
        R"doc(
        Train a D3PM discrete-diffusion model on single-cell counts (cycle 30).

        Parameters
        ----------
        counts : DeviceCsc
            genes × cells raw count matrix.
        stream : int or None
        seed : int

        Returns
        -------
        DiscreteDiffusionResult
            Host-side model weights suitable for pickling and serialisation.
        )doc");

    // ── sample_cells ──────────────────────────────────────────────────────
    m.def("sample_discrete_diffusion",
        [](const PyDiscreteDiffusionResult& model,
           int   n_samples,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_gen(stream);
            // Reconstruct C++ result from Py wrapper.
            singlet::gpu::generative::DiscreteDiffusionResult cpp_model;
            cpp_model.model_weights = model.model_weights;
            cpp_model.model_shape   = model.model_shape;
            cpp_model.loss_history  = model.loss_history;
            cpp_model.bin_edges     = model.bin_edges;
            cpp_model.n_epochs_used = model.n_epochs_used;

            // Patch seed in config copy.
            singlet::gpu::generative::DiscreteDiffusionConfig cfg = model.cfg;
            cfg.seed = seed;

            // Infer n_genes from model_shape (gene embedding dim entry).
            // Fallback: the gene ID embed table shape is [n_genes × d_hidden].
            // We search model_shape for the entry with param_id=1 (gene embed).
            // If not found, n_genes cannot be inferred here — caller must pass.
            // For the device sample buffer size we use the bin_edges trick:
            // the reconstructed output is float (not tokenized), so n_genes is
            // embedded as the 2nd dimension of the first model_shape entry.
            int n_genes = 0;
            if (model.model_shape.size() >= 4) {
                // model_shape layout: [layer_id, param_id, rows, cols, ...]
                // The gene embedding (layer_id=0, param_id=1) has rows=n_genes.
                for (size_t i = 0; i + 3 < model.model_shape.size(); i += 4) {
                    if (model.model_shape[i+1] == 1) {
                        n_genes = model.model_shape[i+2];
                        break;
                    }
                }
            }

            auto d_samples = singlet::gpu::generative::sample_cells(cpp_model, n_samples, cfg, s);

            auto py_s = std::make_shared<PyDDSamples>();
            py_s->data     = std::move(d_samples);
            py_s->n_samples = n_samples;
            py_s->n_genes   = n_genes;
            py_s->stream    = s;
            return py::cast(py_s);
        },
        py::arg("model"),
        py::arg("n_samples"),
        py::arg("stream") = py::none(),
        py::arg("seed")   = 0ULL,
        R"doc(
        Sample synthetic cells from a trained DiscreteDiffusionResult.

        Parameters
        ----------
        model : DiscreteDiffusionResult
            Trained model from train_discrete_diffusion.
        n_samples : int
            Number of cells to generate.
        stream : int or None
        seed : int
            Philox seed for sampling.

        Returns
        -------
        DDSamples
            Device-resident float32 buffer [n_samples × n_genes].
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
