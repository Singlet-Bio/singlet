// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_de_wilcoxon_perf_c85.cpp
//
// Cycle 85: multi-scale Wilcoxon DE benchmark (small / medium / large).
// Mirrors bench_de_ttest_perf.cpp — labels assigned deterministically (cell % n_clusters).
// Timing scope: wilcoxon_de() only. 3 warmup + 5 timed iterations per scale.

#include <singlet/gpu/bench/harness.h>

#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/de/wilcoxon.h>
#include <singlet/gpu/de/types.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
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
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/exon_counts.1pz";

static singlet::gpu::core::DeviceCSC make_csc(int n_genes, int n_cells, uint64_t seed,
                                              cudaStream_t stream) {
    using namespace singlet::gpu::core;
    std::mt19937_64 rng(seed);
    const double density = 0.05;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<int> cp(n_cells + 1, 0), ri; std::vector<float> vv;
    ri.reserve((size_t)(n_genes * n_cells * density * 1.2));
    vv.reserve(ri.capacity());
    for (int c = 0; c < n_cells; ++c) {
        cp[c] = (int)ri.size();
        for (int g = 0; g < n_genes; ++g) {
            if (ud(rng) < density) { ri.push_back(g); vv.push_back((float)((int)(ud(rng)*19)+1)); }
        }
    }
    cp[n_cells] = (int)ri.size();
    int64_t nnz = (int64_t)ri.size();
    DeviceMemory<int> dc(n_cells+1), dr(nnz); DeviceMemory<float> dv(nnz);
    cudaMemcpyAsync(dc.get(), cp.data(),(n_cells+1)*4,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(dr.get(), ri.data(),nnz*4,        cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(dv.get(), vv.data(),nnz*4,        cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
    DeviceCSC mat; mat.rows=n_genes; mat.cols=n_cells; mat.nnz=nnz;
    mat.col_ptr=std::move(dc); mat.row_indices=std::move(dr); mat.values=std::move(dv);
    return mat;
}

static singlet::gpu::core::DeviceMemory<int> make_labels(int n_cells, int n_clusters,
                                                          cudaStream_t stream) {
    std::vector<int> h(n_cells);
    for (int i = 0; i < n_cells; ++i) h[i] = i % n_clusters;
    singlet::gpu::core::DeviceMemory<int> d(n_cells);
    cudaMemcpyAsync(d.get(), h.data(), n_cells*4, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// write_csc_bin (CSC binary for Python reference) now lives in the shared
// harness: singlet::gpu::bench::write_csc_bin.
static void write_npy_i32(const std::string& p, const int* d, int n) {
    std::string hdr="{'descr': '<i4', 'fortran_order': False, 'shape': ("+std::to_string(n)+",), }";
    int pad=((int(hdr.size())+10+63)/64)*64-10;
    while(int(hdr.size())<pad-1) hdr+=' '; hdr+='\n';
    uint16_t hl=uint16_t(hdr.size());
    std::ofstream f(p,std::ios::binary);
    const char mg[]="\x93NUMPY"; f.write(mg,6);f.put(1);f.put(0);
    f.write((char*)&hl,2);f.write(hdr.data(),hl);f.write((char*)d,n*4);
}

struct ScaleResult { double gpu_ms; double scanpy_ms; double mem_mb; bool skipped; std::string skip_reason; };

static ScaleResult run_scale(const char* label, int n_genes, int n_cells, int n_clusters,
                              bool use_file, cudaStream_t stream) {
    namespace bench = singlet::gpu::bench;
    ScaleResult r{};

    singlet::gpu::core::DeviceCSC mat;
    int64_t nnz = 0;
    if (use_file) {
        if (fs::exists(kSamplePath)) {
            try {
                auto loaded = singlet::gpu::io::load_pz(kSamplePath, stream);
                cudaStreamSynchronize(stream);
                if (loaded.mat.cols > 0 && loaded.mat.rows > 0) {
                    n_cells = loaded.mat.cols; n_genes = loaded.mat.rows; nnz = loaded.mat.nnz;
                    mat = std::move(loaded.mat);
                    std::printf("[wilcoxon_bench %s] .1pz: %d genes × %d cells nnz=%lld\n",
                                label, n_genes, n_cells, (long long)nnz);
                } else {
                    std::fprintf(stderr,"[wilcoxon_bench %s] .1pz empty — synthetic\n",label);
                    use_file = false;
                }
            } catch (const std::exception& ex) {
                std::fprintf(stderr,"[wilcoxon_bench %s] load: %s\n",label,ex.what());
                use_file = false;
            }
        } else {
            std::fprintf(stderr,"[wilcoxon_bench %s] .1pz not found at %s — synthetic\n",label,kSamplePath);
            use_file = false;
        }
    }
    if (!use_file) {
        size_t free_b=0,total=0; cudaMemGetInfo(&free_b,&total);
        double need = (double)n_genes*n_cells*0.05*4*3/1e9;
        double avail = free_b/1e9;
        if (need > avail*0.9) { r.skipped=true; r.skip_reason="OOM-predicted (need "+std::to_string(need)+"GB, avail "+std::to_string(avail)+"GB)"; return r; }
        try { mat=make_csc(n_genes,n_cells,0xC0FFEE42,stream); nnz=mat.nnz; }
        catch(const std::exception& ex){ r.skipped=true; r.skip_reason=std::string("OOM: ")+ex.what(); return r; }
        std::printf("[wilcoxon_bench %s] synthetic %d genes × %d cells nnz=%lld\n",
                    label,n_genes,n_cells,(long long)nnz);
    }

    auto d_labels = make_labels(n_cells, n_clusters, stream);

    singlet::gpu::de::WilcoxonConfig cfg;
    cfg.n_bins=4096; cfg.top_n=100; cfg.gene_tile=1024; cfg.deterministic=false;

    for (int i=0;i<3;++i){ auto r2=singlet::gpu::de::wilcoxon_de(mat,d_labels,n_clusters,cfg,stream); cudaStreamSynchronize(stream);(void)r2; }

    bench::BenchTimer timer;
    bench::PeakMemTracker mem;
    std::vector<double> wall_ms(5);
    for (int i=0;i<5;++i){
        mem.sample_before(); timer.start(stream);
        auto r2=singlet::gpu::de::wilcoxon_de(mat,d_labels,n_clusters,cfg,stream);
        timer.stop(stream); cudaStreamSynchronize(stream); mem.sample_after();
        wall_ms[i]=timer.elapsed_ms();
        std::printf("[wilcoxon_bench %s] iter %d: %.1f ms\n",label,i,wall_ms[i]);
        (void)r2;
    }
    std::sort(wall_ms.begin(),wall_ms.end());
    r.gpu_ms=wall_ms[2]; r.mem_mb=mem.peak_delta_mb();

    fs::create_directories("/tmp/sg_wilcoxon_bench");
    std::string mat_p=std::string("/tmp/sg_wilcoxon_bench/")+label+"_mat.bin";
    std::string lbl_p=std::string("/tmp/sg_wilcoxon_bench/")+label+"_labels.npy";
    std::string out_p=std::string("/tmp/sg_wilcoxon_bench/")+label+"_ref.json";
    {
        std::vector<float> hv(nnz); std::vector<int> hp(n_cells+1),hi(nnz),hl(n_cells);
        cudaMemcpy(hv.data(),mat.values.get(),nnz*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hp.data(),mat.col_ptr.get(),(n_cells+1)*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hi.data(),mat.row_indices.get(),nnz*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hl.data(),d_labels.get(),n_cells*4,cudaMemcpyDeviceToHost);
        bench::write_csc_bin(mat_p,hv.data(),hp.data(),hi.data(),n_genes,n_cells,nnz);
        write_npy_i32(lbl_p,hl.data(),n_cells);
    }
    std::string script=std::string(BENCH_REFS_DIR)+"/wilcoxon_ref.py";
    std::string cmd="python3 "+script+" --input "+mat_p+" --labels "+lbl_p+" --timing-json "+out_p+" 2>/dev/null";
    auto ref=bench::run_python_reference(cmd,out_p);
    r.scanpy_ms=ref.wall_ms;

    bench::BenchRow row;
    row.date=bench::today_iso(); row.feature="de/wilcoxon"; row.scale=label;
    row.impl="singlet-gpu"; row.wall_ms=r.gpu_ms; row.mem_mb=r.mem_mb;
    row.cells_per_sec=bench::throughput(n_cells,r.gpu_ms);
    row.sota_wall=r.scanpy_ms; row.sota_mem=ref.mem_mb;
    row.ratio_wall=(r.scanpy_ms>0&&r.gpu_ms>0)?r.scanpy_ms/r.gpu_ms:-1.0;
    row.commit=bench::git_short_sha();
    bench::log_row(row);
    return r;
}

int main() {
    namespace bench = singlet::gpu::bench;
    int ndev=0; cudaGetDeviceCount(&ndev);
    if(ndev==0){ bench::skip("bench_de_wilcoxon_perf_c85","no CUDA GPU"); return 0; }
    cudaStream_t stream=nullptr; cudaStreamCreate(&stream);

    auto s=run_scale("small-500c-200g-4cl",  200,   500,  4, false, stream);
    auto m=run_scale("medium-11k-real-5cl",  36601,11560,  5, true,  stream);
    auto l=run_scale("large-100k-30kg-8cl",  30000,100000, 8, false, stream);

    std::printf("\n=== SUMMARY bench_de_wilcoxon_perf_c85 ===\n");
    std::printf("small  : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                s.gpu_ms,s.scanpy_ms,(s.scanpy_ms>0&&s.gpu_ms>0)?s.scanpy_ms/s.gpu_ms:-1.0,s.mem_mb);
    std::printf("medium : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                m.gpu_ms,m.scanpy_ms,(m.scanpy_ms>0&&m.gpu_ms>0)?m.scanpy_ms/m.gpu_ms:-1.0,m.mem_mb);
    if(l.skipped)
        std::printf("large  : SKIPPED — %s\n",l.skip_reason.c_str());
    else
        std::printf("large  : GPU=%.1fms  scanpy=%.1fms  ratio=%.1fx  mem=%.1fMB\n",
                    l.gpu_ms,l.scanpy_ms,(l.scanpy_ms>0&&l.gpu_ms>0)?l.scanpy_ms/l.gpu_ms:-1.0,l.mem_mb);

    cudaStreamDestroy(stream);
    return 0;
}
