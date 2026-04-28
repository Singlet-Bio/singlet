// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_de_ttest_perf.cpp
//
// Performance benchmark: singlet_gpu::de::ttest_de at 3 scales.
// Timing scope: ttest_de() only; labels are assigned deterministically (cell % n_clusters).
// 3 warmup + 5 timed iterations per scale. Scanpy t-test CPU reference via subprocess.
// Scales: small (500c×200g, 4cl), medium (11560c×36601g, 5cl from .1pz), large (100000c×30000g, 8cl).
// Large scale: skip gracefully on OOM or >120s.

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/de/ttest.h>
#include <singlet-gpu/de/types.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// exon_counts.1pz is the canonical scRNA output (counts.1pz was the old name).
static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/exon_counts.1pz";

// Build synthetic CSC (genes x cells) on device.
static singlet_gpu::core::DeviceCSC make_csc(int n_genes, int n_cells, uint64_t seed,
                                              cudaStream_t stream) {
    using namespace singlet_gpu::core;
    std::mt19937_64 rng(seed);
    const double density = 0.05;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<int> col_ptr(n_cells + 1, 0);
    std::vector<int> row_idx;
    std::vector<float> vals;
    row_idx.reserve((size_t)(n_genes * n_cells * density * 1.2));
    vals.reserve(row_idx.capacity());
    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = (int)row_idx.size();
        for (int g = 0; g < n_genes; ++g) {
            if (ud(rng) < density) {
                row_idx.push_back(g);
                vals.push_back((float)((int)(ud(rng) * 19) + 1));
            }
        }
    }
    col_ptr[n_cells] = (int)row_idx.size();
    int64_t nnz = (int64_t)row_idx.size();
    DeviceMemory<int> d_col(n_cells + 1);
    DeviceMemory<int> d_row(nnz);
    DeviceMemory<float> d_val(nnz);
    cudaMemcpyAsync(d_col.get(), col_ptr.data(), (n_cells+1)*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row.get(), row_idx.data(), nnz*4,        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_val.get(), vals.data(),    nnz*4,        cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    DeviceCSC mat;
    mat.rows = n_genes; mat.cols = n_cells; mat.nnz = nnz;
    mat.col_ptr = std::move(d_col); mat.row_indices = std::move(d_row); mat.values = std::move(d_val);
    return mat;
}

// Upload labels (cell % n_clusters) to device.
static singlet_gpu::core::DeviceMemory<int> make_labels(int n_cells, int n_clusters,
                                                          cudaStream_t stream) {
    std::vector<int> h(n_cells);
    for (int i = 0; i < n_cells; ++i) h[i] = i % n_clusters;
    singlet_gpu::core::DeviceMemory<int> d(n_cells);
    cudaMemcpyAsync(d.get(), h.data(), n_cells*4, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// Write CSC binary for Python reference (same format as wilcoxon_ref.py).
static void write_csc_bin(const std::string& path, const float* hv, const int* hp,
                           const int* hi, int nr, int nc, int64_t nnz) {
    std::ofstream f(path, std::ios::binary);
    uint32_t magic = 0x43535343u, nru = nr, ncu = nc;
    uint64_t nz = nnz;
    f.write((char*)&magic,4); f.write((char*)&nru,4); f.write((char*)&ncu,4);
    f.write((char*)&nz,8);
    f.write((char*)hv, nnz*4); f.write((char*)hp, (nc+1)*4); f.write((char*)hi, nnz*4);
}
static void write_npy_i32(const std::string& path, const int* d, int n) {
    std::string hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (" + std::to_string(n) + ",), }";
    int pad = ((int(hdr.size())+10+63)/64)*64 - 10;
    while (int(hdr.size()) < pad-1) hdr += ' '; hdr += '\n';
    uint16_t hl = uint16_t(hdr.size());
    std::ofstream f(path, std::ios::binary);
    const char mg[] = "\x93NUMPY"; f.write(mg,6); f.put(1); f.put(0);
    f.write((char*)&hl,2); f.write(hdr.data(),hl); f.write((char*)d,n*4);
}

struct ScaleResult { double gpu_ms; double scanpy_ms; double mem_mb; bool skipped; std::string skip_reason; };

static ScaleResult run_scale(const char* label, int n_genes, int n_cells, int n_clusters,
                              bool use_file, cudaStream_t stream) {
    namespace bench = singlet_gpu::bench;
    ScaleResult r{};

    singlet_gpu::core::DeviceCSC mat;
    int64_t nnz = 0;
    if (use_file) {
        if (fs::exists(kSamplePath)) {
            try {
                auto loaded = singlet_gpu::io::load_pz(kSamplePath, stream);
                cudaStreamSynchronize(stream);
                if (loaded.mat.cols > 0 && loaded.mat.rows > 0) {
                    n_cells = loaded.mat.cols; n_genes = loaded.mat.rows; nnz = loaded.mat.nnz;
                    mat = std::move(loaded.mat);
                    std::printf("[ttest_bench %s] .1pz: %d genes × %d cells nnz=%lld\n",
                                label, n_genes, n_cells, (long long)nnz);
                } else {
                    std::fprintf(stderr, "[ttest_bench %s] .1pz returned empty mat — synthetic\n", label);
                    use_file = false;
                }
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "[ttest_bench %s] load failed: %s\n", label, ex.what());
                use_file = false;
            }
        } else {
            std::fprintf(stderr, "[ttest_bench %s] .1pz not found at %s — synthetic\n", label, kSamplePath);
            use_file = false;
        }
    }
    if (!use_file) {
        // Check device memory headroom: need ~(n_cells*n_genes*density)*4*3 bytes
        size_t free_b = 0, total = 0; cudaMemGetInfo(&free_b, &total);
        double needed_gb = (double)n_genes * n_cells * 0.05 * 4 * 3 / 1e9;
        double avail_gb = free_b / 1e9;
        if (needed_gb > avail_gb * 0.9) {
            r.skipped = true;
            r.skip_reason = "OOM-predicted (need ~" + std::to_string(needed_gb) + " GB, have " + std::to_string(avail_gb) + " GB free)";
            return r;
        }
        try {
            mat = make_csc(n_genes, n_cells, 0xBEEFC0DE, stream);
            nnz = mat.nnz;
        } catch (const std::exception& ex) {
            r.skipped = true; r.skip_reason = std::string("OOM: ") + ex.what(); return r;
        }
        std::printf("[ttest_bench %s] synthetic %d genes × %d cells nnz=%lld\n",
                    label, n_genes, n_cells, (long long)nnz);
    }

    auto d_labels = make_labels(n_cells, n_clusters, stream);

    singlet_gpu::de::TtestConfig cfg;
    cfg.top_n = 100; cfg.gene_tile = 1024; cfg.deterministic = false;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        auto res = singlet_gpu::de::ttest_de(mat, d_labels, n_clusters, cfg, stream);
        cudaStreamSynchronize(stream); (void)res;
    }

    // Timed
    bench::BenchTimer timer;
    bench::PeakMemTracker mem;
    std::vector<double> wall_ms(5);
    for (int i = 0; i < 5; ++i) {
        mem.sample_before();
        timer.start(stream);
        auto res = singlet_gpu::de::ttest_de(mat, d_labels, n_clusters, cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem.sample_after();
        wall_ms[i] = timer.elapsed_ms();
        std::printf("[ttest_bench %s] iter %d: %.1f ms\n", label, i, wall_ms[i]);
        (void)res;
    }
    std::sort(wall_ms.begin(), wall_ms.end());
    r.gpu_ms = wall_ms[2]; r.mem_mb = mem.peak_delta_mb();

    // Scanpy CPU reference
    fs::create_directories("/tmp/sg_ttest_bench");
    std::string mat_p = std::string("/tmp/sg_ttest_bench/") + label + "_mat.bin";
    std::string lbl_p = std::string("/tmp/sg_ttest_bench/") + label + "_labels.npy";
    std::string out_p = std::string("/tmp/sg_ttest_bench/") + label + "_ref.json";
    {
        std::vector<float> hv(nnz); std::vector<int> hp(n_cells+1), hi(nnz), hl(n_cells);
        cudaMemcpy(hv.data(), mat.values.get(),      nnz*4,           cudaMemcpyDeviceToHost);
        cudaMemcpy(hp.data(), mat.col_ptr.get(),     (n_cells+1)*4,   cudaMemcpyDeviceToHost);
        cudaMemcpy(hi.data(), mat.row_indices.get(), nnz*4,           cudaMemcpyDeviceToHost);
        cudaMemcpy(hl.data(), d_labels.get(),        n_cells*4,       cudaMemcpyDeviceToHost);
        write_csc_bin(mat_p, hv.data(), hp.data(), hi.data(), n_genes, n_cells, nnz);
        write_npy_i32(lbl_p, hl.data(), n_cells);
    }
    std::string script = std::string(BENCH_REFS_DIR) + "/ttest_ref.py";
    std::string cmd = "python3 " + script + " --input " + mat_p + " --labels " + lbl_p +
                      " --timing-json " + out_p + " 2>/dev/null";
    auto ref = bench::run_python_reference(cmd, out_p);
    r.scanpy_ms = ref.wall_ms;

    bench::BenchRow row;
    row.date = bench::today_iso(); row.feature = "de/ttest"; row.scale = label;
    row.impl = "singlet-gpu"; row.wall_ms = r.gpu_ms; row.mem_mb = r.mem_mb;
    row.cells_per_sec = bench::throughput(n_cells, r.gpu_ms);
    row.sota_wall = r.scanpy_ms; row.sota_mem = ref.mem_mb;
    row.ratio_wall = (r.scanpy_ms > 0 && r.gpu_ms > 0) ? r.scanpy_ms / r.gpu_ms : -1.0;
    row.commit = bench::git_short_sha();
    bench::log_row(row);
    return r;
}

int main() {
    namespace bench = singlet_gpu::bench;
    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev == 0) { bench::skip("bench_de_ttest_perf", "no CUDA GPU"); return 0; }
    cudaStream_t stream = nullptr; cudaStreamCreate(&stream);

    // Small: 200 genes × 500 cells, 4 clusters
    auto s = run_scale("small-500c-200g-4cl",  200,   500,  4, false, stream);
    // Medium: GSM4037629 real data (~11560 cells × ~36601 genes), 5 clusters
    auto m = run_scale("medium-11k-real-5cl",  36601,11560,  5, true,  stream);
    // Large: 30000 genes × 100000 cells, 8 clusters — skip gracefully
    auto l = run_scale("large-100k-30kg-8cl",  30000,100000, 8, false, stream);

    std::printf("\n=== SUMMARY bench_de_ttest_perf ===\n");
    std::printf("small  : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                s.gpu_ms, s.scanpy_ms,
                (s.scanpy_ms>0&&s.gpu_ms>0)?s.scanpy_ms/s.gpu_ms:-1.0, s.mem_mb);
    std::printf("medium : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                m.gpu_ms, m.scanpy_ms,
                (m.scanpy_ms>0&&m.gpu_ms>0)?m.scanpy_ms/m.gpu_ms:-1.0, m.mem_mb);
    if (l.skipped)
        std::printf("large  : SKIPPED — %s\n", l.skip_reason.c_str());
    else
        std::printf("large  : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                    l.gpu_ms, l.scanpy_ms,
                    (l.scanpy_ms>0&&l.gpu_ms>0)?l.scanpy_ms/l.gpu_ms:-1.0, l.mem_mb);

    cudaStreamDestroy(stream);
    return 0;
}
