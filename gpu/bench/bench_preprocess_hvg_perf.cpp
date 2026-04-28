// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_preprocess_hvg_perf.cpp
//
// Benchmark: preprocess/hvg  vs  scanpy.pp.highly_variable_genes (3 flavors)
//
// What is measured:
//   - select_hvg() time for both SeuratV3 and PearsonResiduals flavors.
//   - Excludes load_pz time.
//   - Peak device memory: gene-moment arrays + sort workspace.
//   - Throughput in genes/s (HVG selection is gene-centric).
//
// SOTA reference: scanpy.pp.highly_variable_genes (seurat, seurat_v3, pearson)
// via hvg_ref.py subprocess.
//
// Scale: 10k — GSM4037629 (11,560 cells, ~33k genes).
// Fallback: synthetic 200 genes × 500 cells.

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/preprocess/hvg.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet_gpu;

static constexpr const char* REF_SCRIPT =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/refs/hvg_ref.py";

// Helper: run one flavor and return BenchRow fields (populates timer + mem externally).
static void bench_hvg_flavor(
        const io::PzDeviceMatrix& mat,
        preprocess::HvgFlavor flavor,
        const char* flavor_name,
        const char* scale,
        float sota_wall, float sota_mem,
        cudaStream_t stream) {

    int64_t n_genes = mat.mat.rows;

    preprocess::HvgConfig cfg;
    cfg.flavor = flavor;
    cfg.top_n  = 2000;

    BenchTimer        timer(stream);
    PeakMemoryTracker mem;

    for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
        bool timed = (iter >= BENCH_WARMUP_ITERS);
        if (timed) mem.snapshot_before();
        timer.start();
        auto result = preprocess::select_hvg(mat.mat, cfg, stream);
        cudaStreamSynchronize(stream);
        timer.stop();
        if (timed) { timer.record(); mem.snapshot_after(); }
    }

    // feature label: "preprocess/hvg-{flavor}"
    std::string feature = std::string("preprocess/hvg-") + flavor_name;

    BenchRow row;
    row.date          = current_date_str();
    row.feature       = feature;
    row.scale         = scale;
    row.impl          = "singlet-gpu";
    row.wall_ms_min   = timer.min_ms();
    row.wall_ms_med   = timer.median_ms();
    row.wall_ms_max   = timer.max_ms();
    row.mem_mb_peak   = mem.peak_mb();
    // HVG is gene-centric; report genes/s.
    row.cells_per_sec = throughput(n_genes, row.wall_ms_med);
    row.sota_wall_sec = sota_wall;
    row.sota_mem_mb   = sota_mem;
    log_row(row);
}

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/counts.1pz";
    bool have_real = fs::exists(pz_path);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // SOTA reference subprocess — runs once for all 3 flavors.
    float sota_wall = -1.0f, sota_mem = -1.0f;
    if (have_real && fs::exists(REF_SCRIPT)) {
        char cmd[1024]{};
        std::snprintf(cmd, sizeof(cmd),
            "python3 %s %s 2>/dev/null", REF_SCRIPT, CANONICAL_SAMPLE_DIR);
        FILE* fp = popen(cmd, "r");
        if (fp) {
            char line[256]{};
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0) sota_wall = std::atof(line + 14);
                if (strncmp(line, "SOTA_MEM_MB=",   12) == 0) sota_mem  = std::atof(line + 12);
            }
            pclose(fp);
        }
    }

    if (have_real) {
        io::PzDeviceMatrix mat = io::load_pz(pz_path, stream);
        cudaStreamSynchronize(stream);

        bench_hvg_flavor(mat, preprocess::HvgFlavor::SeuratV3,         "seurat_v3",         "10k", sota_wall, sota_mem, stream);
        bench_hvg_flavor(mat, preprocess::HvgFlavor::PearsonResiduals,  "pearson_residuals",  "10k", sota_wall, sota_mem, stream);

    } else {
        skip_no_sample(pz_path);

        // Synthetic: build minimal PzDeviceMatrix.
        SyntheticMatrix syn = make_synthetic_matrix(200, 500);

        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));
        cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, syn.indices.data(), syn.nnz        * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_values,  syn.values.data(),  syn.nnz        * sizeof(float), cudaMemcpyHostToDevice);

        io::PzDeviceMatrix mat;
        mat.mat.rows    = syn.rows;
        mat.mat.cols    = syn.cols;
        mat.mat.nnz     = syn.nnz;
        mat.mat.col_ptr     = factornet::gpu::DeviceMemory<int>::wrap(d_indptr,  syn.cols + 1);
        mat.mat.row_indices = factornet::gpu::DeviceMemory<int>::wrap(d_indices, syn.nnz);
        mat.mat.values      = factornet::gpu::DeviceMemory<float>::wrap(d_values, syn.nnz);
        mat.producer_stream = stream;

        bench_hvg_flavor(mat, preprocess::HvgFlavor::SeuratV3,        "seurat_v3",        "tiny", -1.0f, -1.0f, stream);
        bench_hvg_flavor(mat, preprocess::HvgFlavor::PearsonResiduals, "pearson_residuals", "tiny", -1.0f, -1.0f, stream);

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);
    }

    cudaStreamDestroy(stream);
    return 0;
}
