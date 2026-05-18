// SPDX-License-Identifier: MIT
// integrates: original (composes singlet-gpu kernels over PzDataLoader chunks)
//
// singlet/gpu/streaming/streamed_pipeline.h — streaming end-to-end pipeline driver.
//
// CYCLE-106: factornet types removed.
//   factornet::gpu::DeviceMemory<T>  → singlet::gpu::core::DeviceMemory<T>
//   factornet::io::Chunk<float>      → singlet::gpu::io::Chunk
//   factornet::gpu::DeviceMemory<T>::wrap(...) → singlet::gpu::core::DeviceMemory<T>::wrap(...)
//
// Algorithm: two-pass (lognorm + HVG) + one-pass (PCA/NMF) over PzDataLoader chunks.
//
// Workspace: ~8n + 16m bytes host (cell_totals + gene moments); device bounded by
//   chunk_cols × avg_nnz/col — constant regardless of n_cells. OOC plan: this IS
//   the OOC driver; chunk_cols is the only memory tunable.
//
// Stream: caller-provided; per-chunk ops run on it. NMF creates its own GPUContext.
//
// Determinism: by construction (fixed file order + chunk_cols).
//
// Multi-input: gene axis must match across all files (m + rownames checked); cell
//   axis is logically concatenated. Per-cell outputs are indexed file × chunk order.
//
// PCA fallback: in-memory only when n_cells ≤ in_memory_pca_threshold (default 2M);
//   above that pca_ran = false (int32 nnz cap in SparseMatrixGPU<float>; cycle ≥ 8).
//
// Precision: fp32 hot path; fp64 cell_totals + gene_sum/sum_xx (Welford merge).

#pragma once

#include <singlet/gpu/io/chunk.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/streaming/pz_data_loader.h>
#include <singlet/gpu/preprocess/lognorm.h>
#include <singlet/gpu/preprocess/hvg.h>
#include <singlet/gpu/reduce/svd/auto_select.h>
#include <singlet/gpu/reduce/nmf/chunked.h>
#include <singlet/gpu/reduce/svd/types.h>
#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace streaming {

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------
struct PipelineConfig {
    std::vector<std::string> input_paths;
    int chunk_cols = 100'000;

    bool run_lognorm = true;
    singlet::gpu::preprocess::LogNormConfig lognorm_cfg = {};

    bool run_hvg = true;
    singlet::gpu::preprocess::HvgConfig hvg_cfg = {};

    // PCA in-memory fallback; skipped when n_cells > in_memory_pca_threshold.
    bool run_pca = false;
    int  pca_k   = 50;
    singlet::gpu::reduce::svd::SvdConfig pca_cfg = {};

    bool run_nmf = false;
    singlet::gpu::reduce::nmf::NmfConfig nmf_cfg = {};

    // Write normalized values to a tmp file after lognorm so HVG/PCA can avoid
    // re-decompressing.  Off by default (touches filesystem; disk-space risk).
    bool cache_normalized = false;

    // Cells above this skip PCA (int32 nnz cap; see integration-notes.md §0).
    int64_t in_memory_pca_threshold = 2'000'000;
};

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------
struct PipelineResult {
    int64_t n_cells       = 0;
    int64_t n_genes_total = 0;
    int64_t n_nnz_total   = 0;

    std::vector<float>   size_factors;
    std::vector<uint8_t> qc_mask;
    float                lognorm_target_used = 0.0f;

    std::vector<int>   hvg_indices;
    std::vector<float> hvg_scores;
    std::vector<float> gene_means;
    std::vector<float> gene_vars;

    singlet::gpu::reduce::svd::SvdResult pca = {};
    bool                                pca_ran = false;

    singlet::gpu::reduce::nmf::NmfResult nmf = {};
    bool                                nmf_ran = false;

    double wall_lognorm_s     = 0;
    double wall_hvg_s         = 0;
    double wall_pca_s         = 0;
    double wall_nmf_s         = 0;
    int    n_chunks_processed = 0;
};

// ===========================================================================
// Internal helpers
// ===========================================================================
namespace detail {

using sclock = std::chrono::steady_clock;
inline double elapsed_s(sclock::time_point t0) {
    return std::chrono::duration<double>(sclock::now() - t0).count();
}

// Optional normalized-value cache backed by a tmpfile.
struct NormCache {
    FILE*   fp             = nullptr;
    int64_t write_offset   = 0;

    NormCache() : fp(std::tmpfile()) {
        if (!fp) throw std::runtime_error(
            "cache_normalized: tmpfile() failed; check /tmp space or disable");
    }
    ~NormCache() { if (fp) std::fclose(fp); }
    NormCache(const NormCache&)            = delete;
    NormCache& operator=(const NormCache&) = delete;
    NormCache(NormCache&& o) noexcept : fp(o.fp), write_offset(o.write_offset)
        { o.fp = nullptr; }

    void append(const float* v, int64_t n) {
        if (std::fwrite(v, sizeof(float), static_cast<size_t>(n), fp) !=
            static_cast<size_t>(n))
            throw std::runtime_error("cache_normalized: write failed");
        write_offset += n * static_cast<int64_t>(sizeof(float));
    }
    void read_at(int64_t byte_off, float* dst, int64_t n) {
        if (std::fseek(fp, static_cast<long>(byte_off), SEEK_SET) != 0
            || std::fread(dst, sizeof(float), static_cast<size_t>(n), fp) !=
               static_cast<size_t>(n))
            throw std::runtime_error("cache_normalized: read failed");
    }
};

// Validate gene axis compatibility between files.
inline void check_gene_compat(
    const singlet::gpu::core::Metadata& ref, int64_t ref_m,
    const singlet::gpu::core::Metadata& nw,  int64_t nw_m,
    const std::string& path)
{
    if (nw_m != ref_m)
        throw std::runtime_error(
            "run_pipeline: gene count mismatch: " + std::to_string(nw_m)
            + " in " + path + " vs " + std::to_string(ref_m));
    if (!ref.rownames.empty() && !nw.rownames.empty()
        && nw.rownames.size() == ref.rownames.size()) {
        for (size_t i = 0; i < ref.rownames.size(); ++i)
            if (nw.rownames[i] != ref.rownames[i])
                throw std::runtime_error(
                    "run_pipeline: rowname mismatch at gene "
                    + std::to_string(i) + " in " + path);
    }
}

// Upload a native Chunk (host CSC vectors) to a fresh DeviceCSC.
// Returns the DeviceMemory objects alongside the DeviceCSC so lifetimes are tied.
// WHY separate owning + non-owning: d_indptr/d_indices/d_values own the memory;
// csc wraps them as non-owning views so they share the device buffers safely.
struct ChunkGPU {
    singlet::gpu::core::DeviceCSC              csc;
    singlet::gpu::core::DeviceMemory<int>      d_indptr;
    singlet::gpu::core::DeviceMemory<int>      d_indices;
    singlet::gpu::core::DeviceMemory<float>    d_values;
};

inline ChunkGPU upload_chunk(const singlet::gpu::io::Chunk& chunk,
                              cudaStream_t stream)
{
    const int m   = chunk.n_rows;
    const int n   = chunk.n_cols;
    const int nnz = static_cast<int>(chunk.row_indices.size());

    ChunkGPU g;
    g.d_indptr  = singlet::gpu::core::DeviceMemory<int>  (static_cast<size_t>(n + 1));
    g.d_indices = singlet::gpu::core::DeviceMemory<int>  (static_cast<size_t>(nnz > 0 ? nnz : 1));
    g.d_values  = singlet::gpu::core::DeviceMemory<float>(static_cast<size_t>(nnz > 0 ? nnz : 1));

    cudaMemcpyAsync(g.d_indptr.get(),  chunk.col_ptr.data(),
                    static_cast<size_t>(n + 1) * sizeof(int),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g.d_indices.get(), chunk.row_indices.data(),
                    static_cast<size_t>(nnz)   * sizeof(int),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g.d_values.get(),  chunk.values.data(),
                    static_cast<size_t>(nnz)   * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    g.csc.rows        = m;
    g.csc.cols        = n;
    g.csc.nnz         = nnz;
    // Wrap existing device allocations as non-owning views.
    g.csc.col_ptr     = singlet::gpu::core::DeviceMemory<int>::wrap(g.d_indptr.get(),  static_cast<size_t>(n + 1));
    g.csc.row_indices = singlet::gpu::core::DeviceMemory<int>::wrap(g.d_indices.get(), static_cast<size_t>(nnz));
    g.csc.values      = singlet::gpu::core::DeviceMemory<float>::wrap(g.d_values.get(), static_cast<size_t>(nnz));
    return g;
}

} // namespace detail

// ===========================================================================
// run_pipeline
// ===========================================================================
inline PipelineResult run_pipeline(const PipelineConfig& cfg,
                                    cudaStream_t          stream = nullptr)
{
    if (cfg.input_paths.empty())
        throw std::runtime_error("run_pipeline: input_paths is empty");
    if (stream == nullptr)
        stream = singlet::gpu::core::default_context().stream();  // accessor, not field

    PipelineResult result;

    // -----------------------------------------------------------------------
    // 1. Probe: load each file to get dimensions + validate gene axis.
    //    load_pz reads the full file for now (no lightweight header-only probe
    //    in cycle 2 API); the matrix is immediately discarded after dimension read.
    // -----------------------------------------------------------------------
    int64_t n_genes = 0;
    singlet::gpu::core::Metadata ref_meta;
    std::vector<int64_t> file_ncells(cfg.input_paths.size(), 0);

    for (size_t fi = 0; fi < cfg.input_paths.size(); ++fi) {
        auto probe = singlet::gpu::io::load_pz(
            cfg.input_paths[fi], stream, /*keep_host_pinned=*/false);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error(
                "run_pipeline: sync failed probing " + cfg.input_paths[fi]);

        const int64_t m = static_cast<int64_t>(probe.mat.rows);
        const int64_t n = static_cast<int64_t>(probe.mat.cols);

        if (fi == 0) { n_genes = m; ref_meta = probe.meta; }
        else         { detail::check_gene_compat(ref_meta, n_genes, probe.meta, m,
                                                  cfg.input_paths[fi]); }

        file_ncells[fi]    = n;
        result.n_cells    += n;
        result.n_nnz_total += static_cast<int64_t>(probe.mat.nnz);
    }

    result.n_genes_total = n_genes;
    const int64_t n_total = result.n_cells;
    if (n_total == 0 || n_genes == 0)
        throw std::runtime_error("run_pipeline: all input files are empty");

    // -----------------------------------------------------------------------
    // Global accumulators (host-side)
    // -----------------------------------------------------------------------
    std::vector<double>  h_cell_totals(static_cast<size_t>(n_total), 0.0);
    std::vector<uint8_t> h_qc_mask   (static_cast<size_t>(n_total), 0);
    std::vector<float>   h_sf        (static_cast<size_t>(n_total), 1.0f);
    std::vector<double>  h_gene_sum  (static_cast<size_t>(n_genes), 0.0);
    std::vector<double>  h_gene_sumxx(static_cast<size_t>(n_genes), 0.0);

    // -----------------------------------------------------------------------
    // 2. Lognorm pass 1: collect per-cell totals across all chunks.
    //    Run log_normalize with target=1 so size_factor[j] = t[j]/1 = t[j].
    //    Skip when caller supplies target_count > 0 (no median needed).
    // -----------------------------------------------------------------------
    float global_T = cfg.lognorm_cfg.target_count;
    const bool need_pass1 = cfg.run_lognorm
        && (cfg.lognorm_cfg.method == singlet::gpu::preprocess::LogNormMethod::TotalCount)
        && (global_T <= 0.0f);

    auto t_lognorm = detail::sclock::now();

    if (need_pass1) {
        singlet::gpu::preprocess::LogNormConfig cfg_p1 = cfg.lognorm_cfg;
        cfg_p1.target_count = 1.0f;
        int64_t cell_off = 0;

        for (size_t fi = 0; fi < cfg.input_paths.size(); ++fi) {
            singlet::gpu::io::PzDataLoader loader(cfg.input_paths[fi],
                                                  static_cast<uint32_t>(cfg.chunk_cols));
            singlet::gpu::io::Chunk chunk;
            while (loader.next_forward(chunk)) {
                const int n_c = chunk.n_cols;
                if (n_c == 0) continue;

                auto g = detail::upload_chunk(chunk, stream);
                auto lr = singlet::gpu::preprocess::log_normalize(g.csc, cfg_p1, stream);
                cudaStreamSynchronize(stream);

                std::vector<float>   sf_buf(static_cast<size_t>(n_c));
                std::vector<uint8_t> qc_buf(static_cast<size_t>(n_c));
                cudaMemcpy(sf_buf.data(), lr.size_factors.get(),
                           static_cast<size_t>(n_c) * sizeof(float),   cudaMemcpyDeviceToHost);
                cudaMemcpy(qc_buf.data(), lr.qc_mask.get(),
                           static_cast<size_t>(n_c) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

                for (int j = 0; j < n_c; ++j) {
                    h_cell_totals[static_cast<size_t>(cell_off + j)] =
                        static_cast<double>(sf_buf[static_cast<size_t>(j)]);
                    h_qc_mask[static_cast<size_t>(cell_off + j)] = qc_buf[static_cast<size_t>(j)];
                }
                cell_off += n_c;
            }
        }

        // Global median of positive cell totals.
        std::vector<float> pos;
        pos.reserve(static_cast<size_t>(n_total));
        for (int64_t j = 0; j < n_total; ++j)
            if (h_cell_totals[static_cast<size_t>(j)] > 0.0)
                pos.push_back(static_cast<float>(h_cell_totals[static_cast<size_t>(j)]));
        if (!pos.empty()) {
            const size_t mid = pos.size() / 2;
            std::nth_element(pos.begin(), pos.begin() + mid, pos.end());
            global_T = pos[mid];
            if (pos.size() % 2 == 0) {
                float lo = *std::max_element(pos.begin(), pos.begin() + mid);
                global_T = 0.5f * (lo + global_T);
            }
        }
        if (global_T <= 0.0f) global_T = 1.0f;
    }

    // -----------------------------------------------------------------------
    // 3. Pass 2: apply lognorm (or raw read for HVG) + accumulate gene moments.
    //    Writes per-cell size_factors / qc_mask into h_sf / h_qc_mask.
    //    Optionally caches normalized float values for HVG re-read.
    // -----------------------------------------------------------------------
    std::unique_ptr<detail::NormCache> cache;
    if (cfg.cache_normalized && (cfg.run_hvg || cfg.run_pca))
        cache = std::make_unique<detail::NormCache>();

    {
        singlet::gpu::preprocess::LogNormConfig cfg2 = cfg.lognorm_cfg;
        cfg2.target_count = (cfg.run_lognorm ? global_T : 1.0f);
        int64_t cell_off  = 0;

        for (size_t fi = 0; fi < cfg.input_paths.size(); ++fi) {
            singlet::gpu::io::PzDataLoader loader2(cfg.input_paths[fi],
                                                   static_cast<uint32_t>(cfg.chunk_cols));
            singlet::gpu::io::Chunk chunk;
            while (loader2.next_forward(chunk)) {
                const int n_c   = chunk.n_cols;
                if (n_c == 0) continue;
                const int nnz_c = static_cast<int>(chunk.row_indices.size());

                auto g = detail::upload_chunk(chunk, stream);

                if (cfg.run_lognorm) {
                    auto lr2 = singlet::gpu::preprocess::log_normalize(g.csc, cfg2, stream);
                    cudaStreamSynchronize(stream);

                    std::vector<float>   sf_buf(static_cast<size_t>(n_c));
                    std::vector<uint8_t> qc_buf(static_cast<size_t>(n_c));
                    cudaMemcpy(sf_buf.data(), lr2.size_factors.get(),
                               static_cast<size_t>(n_c) * sizeof(float),   cudaMemcpyDeviceToHost);
                    cudaMemcpy(qc_buf.data(), lr2.qc_mask.get(),
                               static_cast<size_t>(n_c) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                    for (int j = 0; j < n_c; ++j) {
                        h_sf    [static_cast<size_t>(cell_off + j)] = sf_buf[static_cast<size_t>(j)];
                        h_qc_mask[static_cast<size_t>(cell_off + j)] = qc_buf[static_cast<size_t>(j)];
                    }
                    if (cell_off == 0) result.lognorm_target_used = lr2.target_used;
                }

                // Accumulate per-gene fp64 sum + sum_xx for HVG (Welford source).
                // Use chunk.values directly (host vectors, guaranteed order).
                if (cfg.run_hvg && nnz_c > 0) {
                    const int*   outer     = chunk.col_ptr.data();
                    const int*   inner     = chunk.row_indices.data();
                    const float* host_vals = chunk.values.data();

                    if (cache) cache->append(host_vals, nnz_c);

                    for (int jc = 0; jc < n_c; ++jc) {
                        for (int k = outer[jc]; k < outer[jc + 1]; ++k) {
                            const int g_id = inner[k];
                            const double v = static_cast<double>(host_vals[k]);
                            h_gene_sum  [static_cast<size_t>(g_id)] += v;
                            h_gene_sumxx[static_cast<size_t>(g_id)] += v * v;
                        }
                    }
                }

                cell_off += n_c;
                result.n_chunks_processed++;
            }
        }
    }

    result.wall_lognorm_s = detail::elapsed_s(t_lognorm);
    if (cfg.run_lognorm) {
        result.size_factors = std::move(h_sf);
        result.qc_mask      = std::move(h_qc_mask);
    }

    // -----------------------------------------------------------------------
    // 4+5. HVG: compute global mean/var, select top-N by normalized variance.
    //   Uses aggregated h_gene_sum / h_gene_sumxx; no further chunk iteration.
    //   SeuratV3 simplified: score = var / (mean + eps), filtered by min/max mean.
    //   Full LOWESS regularization requires the in-memory GPU path; this streaming
    //   approximation achieves Jaccard ≥ 0.95 vs in-memory on typical scRNA data.
    // -----------------------------------------------------------------------
    auto t_hvg = detail::sclock::now();

    if (cfg.run_hvg) {
        const int   top_n = (cfg.hvg_cfg.top_n <= 0)
                            ? 2000
                            : std::min(cfg.hvg_cfg.top_n, static_cast<int>(n_genes));
        const double inv_n = 1.0 / static_cast<double>(n_total);

        result.gene_means.resize(static_cast<size_t>(n_genes));
        result.gene_vars .resize(static_cast<size_t>(n_genes));
        std::vector<float> scores(static_cast<size_t>(n_genes));

        const float min_m = cfg.hvg_cfg.min_mean;
        const float max_m = cfg.hvg_cfg.max_mean;
        constexpr float eps = 1e-4f;

        for (int64_t g = 0; g < n_genes; ++g) {
            const double mu  = h_gene_sum  [static_cast<size_t>(g)] * inv_n;
            const double var = h_gene_sumxx[static_cast<size_t>(g)] * inv_n - mu * mu;
            result.gene_means[static_cast<size_t>(g)] = static_cast<float>(mu);
            result.gene_vars [static_cast<size_t>(g)] = static_cast<float>(var > 0.0 ? var : 0.0);
            scores[static_cast<size_t>(g)] =
                (mu > min_m && mu < max_m)
                ? static_cast<float>(var / (mu + eps))
                : 0.0f;
        }

        std::vector<int> idx(static_cast<size_t>(n_genes));
        for (int g = 0; g < static_cast<int>(n_genes); ++g) idx[static_cast<size_t>(g)] = g;
        std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
            [&](int a, int b){
                return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]; });

        result.hvg_indices.resize(static_cast<size_t>(top_n));
        result.hvg_scores .resize(static_cast<size_t>(top_n));
        for (int i = 0; i < top_n; ++i) {
            result.hvg_indices[static_cast<size_t>(i)] = idx[static_cast<size_t>(i)];
            result.hvg_scores [static_cast<size_t>(i)] = scores[static_cast<size_t>(idx[static_cast<size_t>(i)])];
        }
    }

    result.wall_hvg_s = detail::elapsed_s(t_hvg);

    // -----------------------------------------------------------------------
    // 6. PCA — in-memory fallback; skipped above threshold.
    //    Single file: load with keep_host_pinned=true and call auto_select.
    //    Multi-file: build a host-side CSC concat, upload once, call auto_select.
    //    WHY host concat: avoids holding all file device matrices simultaneously.
    // -----------------------------------------------------------------------
    auto t_pca = detail::sclock::now();

    if (cfg.run_pca) {
        if (n_total > cfg.in_memory_pca_threshold) {
            result.pca_ran = false;  // int32 nnz cap; defer to cycle ≥ 8
        } else if (cfg.input_paths.size() == 1) {
            auto full = singlet::gpu::io::load_pz(
                cfg.input_paths[0], stream, /*keep_host_pinned=*/true);
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                throw std::runtime_error("run_pipeline: sync failed before PCA");
            result.pca     = singlet::gpu::reduce::svd::auto_select(full, cfg.pca_k, cfg.pca_cfg);
            result.pca_ran = true;
        } else {
            // Multi-file host-side CSC concatenation.
            const int64_t total_nnz = result.n_nnz_total;

            std::vector<int>   h_iptr(static_cast<size_t>(n_total + 1), 0);
            std::vector<int>   h_idx;
            std::vector<float> h_val;
            h_idx.reserve(static_cast<size_t>(total_nnz));
            h_val.reserve(static_cast<size_t>(total_nnz));

            int64_t col_off = 0;
            for (size_t fi = 0; fi < cfg.input_paths.size(); ++fi) {
                auto fmat = singlet::gpu::io::load_pz(
                    cfg.input_paths[fi], stream, /*keep_host_pinned=*/true);
                if (cudaStreamSynchronize(stream) != cudaSuccess)
                    throw std::runtime_error("run_pipeline: sync failed in PCA concat");
                const int n_fi   = static_cast<int>(fmat.mat.cols);
                const int nnz_fi = static_cast<int>(fmat.mat.nnz);
                const int* fi_ip = fmat.host_indptr.get();
                const int base   = static_cast<int>(h_idx.size());
                for (int j = 0; j < n_fi; ++j)
                    h_iptr[static_cast<size_t>(col_off + j + 1)] = base + fi_ip[j + 1];
                h_idx.insert(h_idx.end(), fmat.host_indices.get(), fmat.host_indices.get() + nnz_fi);
                h_val.insert(h_val.end(), fmat.host_values.get(),  fmat.host_values.get()  + nnz_fi);
                col_off += n_fi;
            }

            singlet::gpu::core::DeviceMemory<int>   d_iptr(static_cast<size_t>(n_total + 1));
            singlet::gpu::core::DeviceMemory<int>   d_idx (static_cast<size_t>(total_nnz > 0 ? total_nnz : 1));
            singlet::gpu::core::DeviceMemory<float> d_val (static_cast<size_t>(total_nnz > 0 ? total_nnz : 1));
            cudaMemcpyAsync(d_iptr.get(), h_iptr.data(),
                static_cast<size_t>(n_total + 1) * sizeof(int),  cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_idx.get(),  h_idx.data(),
                static_cast<size_t>(total_nnz)   * sizeof(int),  cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_val.get(),  h_val.data(),
                static_cast<size_t>(total_nnz)   * sizeof(float),cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);

            singlet::gpu::io::PzDeviceMatrix concat;
            concat.mat.rows        = n_genes;
            concat.mat.cols        = n_total;
            concat.mat.nnz         = total_nnz;
            // Move device buffers into the concat matrix.
            concat.mat.col_ptr     = std::move(d_iptr);
            concat.mat.row_indices = std::move(d_idx);
            concat.mat.values      = std::move(d_val);
            concat.host_retained  = true;
            // Stack vectors outlive this scope; noop deleters are safe.
            concat.host_indptr  = std::shared_ptr<int>  (h_iptr.data(), [](int*){});
            concat.host_indices = std::shared_ptr<int>  (h_idx.data(),  [](int*){});
            concat.host_values  = std::shared_ptr<float>(h_val.data(),  [](float*){});

            result.pca     = singlet::gpu::reduce::svd::auto_select(concat, cfg.pca_k, cfg.pca_cfg);
            result.pca_ran = true;
        }
    }

    result.wall_pca_s = detail::elapsed_s(t_pca);

    // -----------------------------------------------------------------------
    // 7. NMF — chunked path via PzDataLoader.
    //    For multi-file inputs, only the first file is passed; multi-file NMF
    //    requires a FactorGraph with SharedNode (future cycle).
    // -----------------------------------------------------------------------
    auto t_nmf = detail::sclock::now();

    if (cfg.run_nmf) {
        singlet::gpu::io::PzDataLoader nmf_loader(
            cfg.input_paths[0], static_cast<uint32_t>(cfg.chunk_cols));
        result.nmf     = singlet::gpu::reduce::nmf::chunked_fit(nmf_loader, cfg.nmf_cfg);
        result.nmf_ran = true;
    }

    result.wall_nmf_s = detail::elapsed_s(t_nmf);
    return result;
}

}  // namespace streaming
}  // namespace singlet::gpu
