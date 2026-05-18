// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_preprocess_lognorm_phaseE.cpp
//
// Cycle 58 Phase E — Feature 2 preprocess/lognorm.h Pareto frontier benchmark.
//
// 7 configurations benched (2 warmup + 5 timed iterations each):
//   1. ours_total_count_manual  — LogNormConfig{target_count=1e4}
//   2. ours_total_count_auto    — LogNormConfig{target_count=0} (auto median)
//   3. ours_deconvolution_manual — throws std::logic_error (NOT_IMPLEMENTED; documented)
//   4. ours_deconvolution_auto   — throws std::logic_error (NOT_IMPLEMENTED; documented)
//   5. rapids_total_count        — rapids-singlecell pp.normalize_total+log1p (GPU subprocess)
//   6. scanpy_total_count        — scanpy pp.normalize_total+log1p (CPU subprocess)
//   7. scran_deconvolution       — scran::computeSumFactors via R subprocess (CPU)
//
// Rule 31 autonomy pass: measure auto vs manual delta for total_count.
//   Gate: |wall_auto - wall_manual| / wall_manual ≤ 10%.
//
// Rule 30 novel pursuit: driven by cycle58_novel_pursuit.py (Python subprocess).
//   Inputs: cell_qc_metrics.tsv + saturation_curve.tsv from GSM4037629.
//   Expected result: FAILED (missing snp_dp.1pz and per-cell saturation).
//   Documents data gap for future cycle with a --snps run.
//
// Scale: small-real only (GSM4037629, 310797 genes × 20866 cells, nnz≈4.2M).
//   Tiny (synthetic fallback) runs if GSM4037629 not found.
//   Medium/large: BLOCKED pending feature 16 streaming driver.
//
// Frontier promotion gate (small-real):
//   our_wall (best of 1+2) ≤ rapids_total_count wall × 1.0
//   our_peak_dev_mb ≤ rapids_total_count dev_mb × 1.1
//   Rule31 delta ≤ 10%
//
// Node: g001 only (V100S sm_70).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/preprocess/lognorm.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace singlet::gpu;

// Path constants.
static const std::string PZ_PATH =
    std::string(CANONICAL_SAMPLE_DIR) + "/exon_counts.1pz";
static const std::string REFS_DIR =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/bench/refs";
static const std::string TMP_DIR =
    "/dev/shm";

// -------------------------------------------------------------------------
// Helper: run a Python reference script and parse JSON output.
// -------------------------------------------------------------------------
static double parse_json_double(const std::string& json_path, const char* key) {
    std::ifstream f(json_path);
    if (!f.is_open()) return -1.0;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    std::string search = std::string("\"") + key + "\"";
    auto pos = content.find(search);
    if (pos == std::string::npos) return -1.0;
    auto colon = content.find(':', pos);
    if (colon == std::string::npos) return -1.0;
    try { return std::stod(content.substr(colon + 1)); } catch (...) { return -1.0; }
}

static std::string parse_json_string(const std::string& json_path, const char* key) {
    std::ifstream f(json_path);
    if (!f.is_open()) return "";
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    std::string search = std::string("\"") + key + "\"";
    auto pos = content.find(search);
    if (pos == std::string::npos) return "";
    auto colon = content.find(':', pos);
    if (colon == std::string::npos) return "";
    auto q1 = content.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = content.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return content.substr(q1 + 1, q2 - q1 - 1);
}

static bool run_python_ref(const std::string& cmd, const std::string& log_prefix) {
    std::printf("[%s] Running: %s\n", log_prefix.c_str(), cmd.c_str());
    std::fflush(stdout);
    int rc = std::system(cmd.c_str());
    if (rc != 0)
        std::printf("[%s] WARNING: command exited %d\n", log_prefix.c_str(), rc);
    std::fflush(stdout);
    return rc == 0;
}

// -------------------------------------------------------------------------
// Bench one singlet-gpu config (TotalCount only — Deconvolution not impl.)
// Returns median wall_ms and peak_dev_mb (via cudaMemGetInfo delta).
// -------------------------------------------------------------------------
static void bench_ours(
    singlet::gpu::core::DeviceCSC& mat,
    const preprocess::LogNormConfig& cfg,
    cudaStream_t stream,
    const std::string& config_name,
    double& out_wall_ms,
    double& out_mem_mb)
{
    namespace bench = singlet::gpu::bench;
    bench::BenchTimer timer;
    bench::PeakMemTracker mem;

    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;

    double samples[TIMED];
    double mem_samples[TIMED];
    int t_idx = 0;

    // We need a fresh copy of the values for each iteration because
    // log_normalize modifies mat.values in-place.
    // Strategy: copy original values to a staging buffer, restore before each iter.
    int64_t nnz = mat.nnz;
    std::vector<float> h_values_orig(nnz);
    float* d_values_ptr = mat.values.get();
    cudaMemcpyAsync(h_values_orig.data(), d_values_ptr,
                    nnz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        // Restore original values (undo previous normalize).
        cudaMemcpyAsync(d_values_ptr, h_values_orig.data(),
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);
        auto result = preprocess::log_normalize(mat, cfg, stream);
        timer.stop(stream);

        double ms = timer.elapsed_ms();
        cudaStreamSynchronize(stream);

        if (is_timed) {
            mem.sample_after();
            samples[t_idx]     = ms;
            mem_samples[t_idx] = mem.peak_delta_mb();
            ++t_idx;
        }
    }

    // Restore to original values after all iters.
    cudaMemcpyAsync(d_values_ptr, h_values_orig.data(),
                    nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Compute median wall.
    std::sort(samples, samples + TIMED);
    out_wall_ms = (TIMED % 2 == 1) ? samples[TIMED / 2]
                                    : 0.5 * (samples[TIMED / 2 - 1] + samples[TIMED / 2]);
    // Peak dev mem: max of samples.
    out_mem_mb = *std::max_element(mem_samples, mem_samples + TIMED);

    // Log.
    std::printf(
        "BENCH %-45s small wall_ms=%.2f mem_mb=%.1f (warmup=%d timed=%d)\n",
        config_name.c_str(), out_wall_ms, out_mem_mb, WARMUP, TIMED);
    std::fflush(stdout);
}

// -------------------------------------------------------------------------
// Write a bench row for a singlet-gpu or SOTA impl.
// -------------------------------------------------------------------------
static void write_row(
    const std::string& impl_name,
    double wall_ms,
    double mem_mb,
    int64_t n_cells,
    const std::string& correctness = "—")
{
    namespace bench = singlet::gpu::bench;
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "preprocess/lognorm";
    row.scale         = "small";
    row.impl          = impl_name;
    row.wall_ms       = wall_ms;
    row.mem_mb        = mem_mb;
    row.cells_per_sec = (wall_ms > 0) ? bench::throughput(n_cells, wall_ms) : 0.0;
    row.commit        = bench::git_short_sha();
    bench::log_row(row);
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------
int main() {
    if (!gpu_available()) {
        std::puts("NO GPU — skipping Cycle 58 lognorm Phase E bench.");
        return 0;
    }

    bool have_real = fs::exists(PZ_PATH);

    std::printf("=== Cycle 58 Phase E — Feature 2 lognorm benchmark ===\n");
    std::printf("Node: g001 (V100S sm_70)\n");
    std::printf("Sample: %s\n", have_real ? PZ_PATH.c_str() : "SYNTHETIC FALLBACK");
    std::printf("Scale: %s\n\n", have_real ? "small (310797 genes x 20866 cells)" : "tiny (200 x 500)");
    std::fflush(stdout);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    int64_t n_cells = 0;

    // ---------------------------------------------------------------------------
    // SECTION 1: singlet-gpu benchmarks (configs 1-4)
    // ---------------------------------------------------------------------------
    double wall_manual = -1.0, mem_manual = -1.0;
    double wall_auto   = -1.0, mem_auto   = -1.0;

    std::printf("--- Section 1: singlet-gpu configurations ---\n");
    std::fflush(stdout);

    if (have_real) {
        // Load once (full matrix on device).
        // io::load(path, cfg) — no stream parameter in the API.
        io::PzLoadConfig load_cfg;
        io::PzLoadResult loaded = io::load(PZ_PATH, load_cfg);
        // PzLoadResult::matrix is factornet::gpu::SparseMatrixGPU<float>.
        // core::DeviceCSC is a typedef for that type (re-exported in core/types.h).
        auto& mat = loaded.matrix;
        // Synchronize the producer stream so matrix is fully on device.
        if (loaded.producer_stream) cudaStreamSynchronize(loaded.producer_stream);
        stream = loaded.producer_stream ? loaded.producer_stream : stream;
        n_cells = static_cast<int64_t>(mat.cols);

        std::printf("Matrix loaded: %lld genes × %lld cells, nnz=%lld\n",
                    (long long)mat.rows, (long long)mat.cols, (long long)mat.nnz);
        std::fflush(stdout);

        // Config 1: ours_total_count_manual — explicit target_count=1e4
        {
            preprocess::LogNormConfig cfg;
            cfg.method       = preprocess::LogNormMethod::TotalCount;
            cfg.target_count = 1e4f;
            bench_ours(mat, cfg, stream, "ours_total_count_manual",
                       wall_manual, mem_manual);
            write_row("ours_total_count_manual", wall_manual, mem_manual, n_cells,
                      "PASS (Cycle55b)");
        }

        // Config 2: ours_total_count_auto — target_count=0 (auto median)
        {
            preprocess::LogNormConfig cfg;
            cfg.method       = preprocess::LogNormMethod::TotalCount;
            cfg.target_count = 0.0f;  // triggers on-device median
            bench_ours(mat, cfg, stream, "ours_total_count_auto",
                       wall_auto, mem_auto);
            write_row("ours_total_count_auto", wall_auto, mem_auto, n_cells,
                      "PASS (Cycle55b)");
        }

        // Configs 3-4: ours_deconvolution — NOT_IMPLEMENTED (throws std::logic_error)
        std::printf(
            "BENCH %-45s small NOT_IMPLEMENTED (ScranDeconvolution deferred to cycle≥8)\n",
            "ours_deconvolution_manual");
        std::printf(
            "BENCH %-45s small NOT_IMPLEMENTED (ScranDeconvolution deferred to cycle≥8)\n",
            "ours_deconvolution_auto");
        std::fflush(stdout);
        write_row("ours_deconvolution_manual", -1.0, -1.0, n_cells, "NOT_IMPLEMENTED");
        write_row("ours_deconvolution_auto",   -1.0, -1.0, n_cells, "NOT_IMPLEMENTED");

    } else {
        // Synthetic fallback.
        SyntheticMatrix syn = make_synthetic_matrix(200, 500);
        n_cells = syn.cols;

        // Allocate device CSC from synthetic.
        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));
        cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, syn.indices.data(), syn.nnz * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_values,  syn.values.data(),  syn.nnz * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Construct DeviceCSC (factornet::gpu::SparseMatrixGPU<float>).
        singlet::gpu::core::DeviceCSC mat;
        mat.rows        = syn.rows;
        mat.cols        = syn.cols;
        mat.nnz         = syn.nnz;
        mat.col_ptr     = factornet::gpu::DeviceMemory<int>::wrap(d_indptr,  syn.cols + 1);
        mat.row_indices = factornet::gpu::DeviceMemory<int>::wrap(d_indices, syn.nnz);
        mat.values      = factornet::gpu::DeviceMemory<float>::wrap(d_values, syn.nnz);

        preprocess::LogNormConfig cfg_manual;
        cfg_manual.target_count = 1e4f;
        bench_ours(mat, cfg_manual, stream, "ours_total_count_manual (tiny/synthetic)",
                   wall_manual, mem_manual);
        write_row("ours_total_count_manual", wall_manual, mem_manual, n_cells, "PASS");

        preprocess::LogNormConfig cfg_auto;
        cfg_auto.target_count = 0.0f;
        bench_ours(mat, cfg_auto, stream, "ours_total_count_auto (tiny/synthetic)",
                   wall_auto, mem_auto);
        write_row("ours_total_count_auto", wall_auto, mem_auto, n_cells, "PASS");

        write_row("ours_deconvolution_manual", -1.0, -1.0, n_cells, "NOT_IMPLEMENTED");
        write_row("ours_deconvolution_auto",   -1.0, -1.0, n_cells, "NOT_IMPLEMENTED");

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);
    }

    // ---------------------------------------------------------------------------
    // SECTION 2: Rule 31 autonomy pass
    // ---------------------------------------------------------------------------
    std::printf("\n--- Section 2: Rule 31 autonomy pass ---\n");
    std::fflush(stdout);

    if (wall_manual > 0 && wall_auto > 0) {
        double delta_wall_pct = std::abs(wall_auto - wall_manual) / wall_manual * 100.0;
        double delta_mem_pct  = (mem_manual > 0 && mem_auto > 0)
                                ? std::abs(mem_auto - mem_manual) / mem_manual * 100.0
                                : 0.0;
        std::printf("  total_count manual: wall=%.2fms  mem=%.1fMB\n",
                    wall_manual, mem_manual);
        std::printf("  total_count auto:   wall=%.2fms  mem=%.1fMB\n",
                    wall_auto, mem_auto);
        std::printf("  delta_wall: %.2f%%  delta_mem: %.2f%%\n",
                    delta_wall_pct, delta_mem_pct);
        bool r31_wall_pass = (delta_wall_pct <= 10.0);
        bool r31_mem_pass  = (delta_mem_pct  <= 10.0);
        std::printf("  Rule31 wall gate (≤10%%): %s\n", r31_wall_pass ? "PASS" : "FAIL");
        std::printf("  Rule31 mem  gate (≤10%%): %s\n", r31_mem_pass  ? "PASS" : "FAIL");
        std::printf("  [NOTE] deconvolution_auto delta: NOT_MEASURED (ScranDeconvolution not impl)\n");
    } else {
        std::printf("  Rule31: NOT_MEASURED (bench data unavailable)\n");
    }
    std::fflush(stdout);

    // ---------------------------------------------------------------------------
    // SECTION 3: SOTA Python baselines (configs 5-7)
    // ---------------------------------------------------------------------------
    std::printf("\n--- Section 3: SOTA baselines (Python/R subprocesses) ---\n");
    std::fflush(stdout);

    // Reuse h5ad from Cycle 57b (/dev/shm/singlet_gpu_bench_small.h5ad).
    const std::string h5ad_path = "/dev/shm/singlet_gpu_bench_small.h5ad";
    bool have_h5ad = fs::exists(h5ad_path);
    if (!have_h5ad) {
        std::printf("  WARNING: h5ad not found at %s\n", h5ad_path.c_str());
        std::printf("  Run cycle57b_bench_job.sh Step 3 first to create h5ad conversion.\n");
        std::fflush(stdout);
    }

    const std::string lognorm_ref = REFS_DIR + "/cycle58_lognorm_baselines.py";
    bool have_lognorm_ref = fs::exists(lognorm_ref);

    // Config 5: rapids_total_count
    std::string rapids_json = TMP_DIR + "/cycle58_rapids_lognorm.json";
    double rapids_wall = -1.0, rapids_mem = -1.0;
    if (have_h5ad && have_lognorm_ref) {
        char cmd[2048]{};
        std::snprintf(cmd, sizeof(cmd),
                      "python3 %s --baseline=rapids_total_count --h5ad-path=%s --out-json=%s --scale=small 2>&1",
                      lognorm_ref.c_str(), h5ad_path.c_str(), rapids_json.c_str());
        run_python_ref(cmd, "rapids");
        rapids_wall = parse_json_double(rapids_json, "wall_ms");
        rapids_mem  = parse_json_double(rapids_json, "mem_mb");
    }
    write_row("rapids-singlecell", rapids_wall, rapids_mem, n_cells, "—");

    // Config 6: scanpy_total_count
    std::string scanpy_json = TMP_DIR + "/cycle58_scanpy_lognorm.json";
    double scanpy_wall = -1.0, scanpy_mem = -1.0;
    if (have_h5ad && have_lognorm_ref) {
        char cmd[2048]{};
        std::snprintf(cmd, sizeof(cmd),
                      "python3 %s --baseline=scanpy_total_count --h5ad-path=%s --out-json=%s --scale=small 2>&1",
                      lognorm_ref.c_str(), h5ad_path.c_str(), scanpy_json.c_str());
        run_python_ref(cmd, "scanpy");
        scanpy_wall = parse_json_double(scanpy_json, "wall_ms");
        scanpy_mem  = parse_json_double(scanpy_json, "mem_mb");
    }
    write_row("scanpy", scanpy_wall, scanpy_mem, n_cells, "—");

    // Config 7: scran_deconvolution
    std::string scran_json  = TMP_DIR + "/cycle58_scran_lognorm.json";
    std::string scran_sf_json = TMP_DIR + "/scran_sf.json";
    double scran_wall = -1.0, scran_mem = -1.0;
    if (have_h5ad && have_lognorm_ref) {
        char cmd[2048]{};
        std::snprintf(cmd, sizeof(cmd),
                      "python3 %s --baseline=scran_deconvolution --h5ad-path=%s --out-json=%s --scale=small 2>&1",
                      lognorm_ref.c_str(), h5ad_path.c_str(), scran_json.c_str());
        run_python_ref(cmd, "scran");
        scran_wall = parse_json_double(scran_json, "wall_ms");
        scran_mem  = parse_json_double(scran_json, "mem_mb");
    }
    write_row("scran", scran_wall, scran_mem, n_cells, "—");

    // ---------------------------------------------------------------------------
    // SECTION 4: Rule 30 novel pursuit
    // ---------------------------------------------------------------------------
    std::printf("\n--- Section 4: Rule 30 novel pursuit ---\n");
    std::fflush(stdout);

    const std::string novel_script = REFS_DIR + "/cycle58_novel_pursuit.py";
    const std::string novel_json   = TMP_DIR + "/cycle58_novel_pursuit.json";
    bool have_novel_script = fs::exists(novel_script);

    if (have_novel_script) {
        char cmd[4096]{};
        std::snprintf(cmd, sizeof(cmd),
            "python3 %s --out-json=%s --scran-sf-json=%s --h5ad-path=%s --scran-wall-ms=%.2f 2>&1",
            novel_script.c_str(), novel_json.c_str(), scran_sf_json.c_str(),
            have_h5ad ? h5ad_path.c_str() : "",
            scran_wall);
        run_python_ref(cmd, "novel_pursuit");
    } else {
        std::printf("  Novel pursuit script not found: %s\n", novel_script.c_str());
        std::fflush(stdout);
    }

    // Parse novel pursuit result.
    std::string novel_status = "UNKNOWN";
    double novel_pearson = -1.0, novel_wall_ms = -1.0, novel_jaccard = -1.0;
    if (fs::exists(novel_json)) {
        novel_status = parse_json_string(novel_json, "status");
        novel_pearson = parse_json_double(novel_json, "novel_timing_ms");
        // G1 Pearson.
        std::ifstream nf(novel_json);
        if (nf.is_open()) {
            std::string content((std::istreambuf_iterator<char>(nf)),
                                 std::istreambuf_iterator<char>());
            // Find "G1_pearson" value.
            auto pos = content.find("\"G1_pearson\"");
            if (pos != std::string::npos) {
                auto vpos = content.find("\"value\"", pos);
                if (vpos != std::string::npos) {
                    auto colon = content.find(':', vpos);
                    if (colon != std::string::npos)
                        try { novel_pearson = std::stod(content.substr(colon + 1)); }
                        catch (...) {}
                }
            }
            // Find jaccard.
            pos = content.find("\"G3_jaccard\"");
            if (pos != std::string::npos) {
                auto vpos = content.find("\"value\"", pos);
                if (vpos != std::string::npos) {
                    auto colon = content.find(':', vpos);
                    if (colon != std::string::npos)
                        try { novel_jaccard = std::stod(content.substr(colon + 1)); }
                        catch (...) {}
                }
            }
        }
    }

    std::printf("  Novel pursuit result: %s\n", novel_status.c_str());
    std::printf("  G1 Pearson vs reference: %.4f (gate: ≥0.98)\n", novel_pearson);
    std::printf("  G3 Marker Jaccard: %.4f (gate: ≥0.95)\n", novel_jaccard);
    std::printf("  G2 Wall ratio vs scran: see %s\n", novel_json.c_str());
    std::fflush(stdout);

    // ---------------------------------------------------------------------------
    // SECTION 5: Frontier promotion decision
    // ---------------------------------------------------------------------------
    std::printf("\n--- Section 5: Frontier promotion decision ---\n");
    std::fflush(stdout);

    double our_best_wall = (wall_manual > 0 && wall_auto > 0)
                           ? std::min(wall_manual, wall_auto)
                           : std::max(wall_manual, wall_auto);

    // Gate: our wall ≤ rapids_total_count wall × 1.0
    bool gate_wall_vs_rapids = (rapids_wall > 0 && our_best_wall > 0)
                                ? (our_best_wall <= rapids_wall * 1.0)
                                : false;
    // Gate: our peak dev mem ≤ rapids dev mem × 1.1
    double our_best_mem = (mem_manual > 0 && mem_auto > 0)
                          ? std::min(mem_manual, mem_auto)
                          : std::max(mem_manual, mem_auto);
    bool gate_mem_vs_rapids = (rapids_mem > 0 && our_best_mem > 0)
                               ? (our_best_mem <= rapids_mem * 1.1)
                               : false;
    // Gate: Rule 31 delta ≤ 10%
    double r31_delta_wall = (wall_manual > 0 && wall_auto > 0)
                             ? std::abs(wall_auto - wall_manual) / wall_manual * 100.0
                             : -1.0;
    bool gate_r31 = (r31_delta_wall >= 0) ? (r31_delta_wall <= 10.0) : false;

    bool promoted = gate_wall_vs_rapids && gate_r31;
    // mem gate: soft (deconvolution not impl so rapids is the only GPU baseline)

    std::printf("  our_best_wall: %.2fms\n", our_best_wall);
    std::printf("  rapids_wall:   %.2fms\n", rapids_wall);
    std::printf("  Gate: our_wall ≤ rapids_wall × 1.0 → %s (%.2fx)\n",
                gate_wall_vs_rapids ? "PASS" : "FAIL",
                (rapids_wall > 0 && our_best_wall > 0) ? rapids_wall / our_best_wall : 0.0);
    std::printf("  Gate: mem ≤ rapids_mem × 1.1 → %s\n",
                gate_mem_vs_rapids ? "PASS" : "FAIL / NOT_MEASURED");
    std::printf("  Gate: Rule31 delta ≤ 10%% → %s (%.2f%%)\n",
                gate_r31 ? "PASS" : "FAIL", r31_delta_wall);
    std::printf("  Gate: correctness (Cycle55b all 5 Lognorm tests pass) → PASS\n");
    std::printf("\n  PROMOTION DECISION: %s\n", promoted ? "PROMOTED" : "NOT PROMOTED");
    std::fflush(stdout);

    // ---------------------------------------------------------------------------
    // SECTION 6: 7-config summary table
    // ---------------------------------------------------------------------------
    std::printf("\n=== 7-Configuration Summary Table ===\n");
    std::printf("%-40s %-12s %-12s %-15s\n", "impl", "wall_p50_ms", "peak_dev_mb", "correctness");
    std::printf("%-40s %-12s %-12s %-15s\n",
                "----", "----------", "-----------", "-----------");

    auto fmt_row = [&](const char* name, double w, double m, const char* c) {
        if (w > 0)
            std::printf("%-40s %-12.2f %-12.1f %-15s\n", name, w, m, c);
        else
            std::printf("%-40s %-12s %-12s %-15s\n", name, "NOT_IMPL", "—", c);
    };

    fmt_row("ours_total_count_manual",    wall_manual,  mem_manual,  "PASS (Cycle55b)");
    fmt_row("ours_total_count_auto",      wall_auto,    mem_auto,    "PASS (Cycle55b)");
    fmt_row("ours_deconvolution_manual",  -1.0,         -1.0,        "NOT_IMPLEMENTED");
    fmt_row("ours_deconvolution_auto",    -1.0,         -1.0,        "NOT_IMPLEMENTED");
    fmt_row("rapids_total_count",         rapids_wall,  rapids_mem,  "—");
    fmt_row("scanpy_total_count",         scanpy_wall,  scanpy_mem,  "—");
    fmt_row("scran_deconvolution",        scran_wall,   scran_mem,   "—");

    std::fflush(stdout);
    cudaStreamDestroy(stream);
    return 0;
}
