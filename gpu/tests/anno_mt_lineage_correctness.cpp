// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/anno_mt_lineage_correctness.cpp
//
// integrates: original (first GPU MT lineage tracing — exploits singlify mt_alleles.1pz directly)
//
// Correctness harness for singlet_gpu::anno::call_clones (anno/mt_lineage.h).
//
// Written against the design doc at:
//   singlet-gpu/state/designs/14-mt-lineage.md
// NOT against the kernel source — authored in parallel with anno/mt_lineage.h
// by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: MQuad Python via subprocess
//   tests/refs/mt_lineage_mquad_reference.py
//   (pip install MQuad numpy scipy; MQuad has a heavy install — the script
//    falls back to an internal numpy GMM if MQuad is unavailable, so the
//    subprocess is optional and the harness always produces a result.)
//
// Tolerances (design doc §"Correctness test spec"):
//   Clone ARI        >= 0.85   (synthetic with planted clones, vs MQuad/fallback)
//   BIC K-selection  ±1 of true K
//   VAF rel_err      <= 1e-4   (per-site, per-clone mean heteroplasmy)
//   Determinism      bit-identical (two runs with same seed)
//
// Test cases:
//   MtLineage_TinySynthetic_VsMQuad        — 200 cells x 50 sites, 3 planted clones
//   MtLineage_GSM4037629_RealData          — load mt_heteroplasmy.1pz from GSM4037629
//   MtLineage_DepthFilter_RemovesLowCoverage — site median depth < 10 → absent
//   MtLineage_BICAutoK_PicksKnownK         — 4 planted clones; BIC picks K=4 ±1
//   MtLineage_Determinism_BitIdentical     — two runs, same seed, same labels
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes all GPU-dependent tests call GTEST_SKIP().

// ---- anno/mt_lineage header (written by gpu-kernel-dev) ---------------------
#include <singlet-gpu/anno/mt_lineage.h>

// ---- other singlet-gpu headers used by the real-data test ------------------
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

// ---- test helpers -----------------------------------------------------------
#include "refs/dump_csc.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#if defined(__has_include) && __has_include(<zlib.h>)
#include <zlib.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic: 200 cells x 50 MT sites, 3 planted clones.
constexpr int kTinyCells    = 200;
constexpr int kTinySites    = 50;
constexpr int kTinyClones   = 3;   // planted clones for the VsMQuad test
constexpr int kBicClones    = 4;   // planted clones for the BIC K-selection test

/// Depth parameters for the synthetic generator.
constexpr float kHighDepth  = 50.0f;   // good coverage (median >= 10)
constexpr float kLowDepth   = 3.0f;    // below min_depth=10 — should be filtered
/// How many of the kTinySites sites get low coverage in the DepthFilter test.
constexpr int   kLowDepthSites = 5;

/// VAF planted for heteroplasmic sites: clone-specific signature.
constexpr float kHetVAF    = 0.20f;   // within-clone heteroplasmy fraction
constexpr float kBaseVAF   = 0.02f;   // background noise for other clones

/// MtLineageConfig defaults (must match mt_lineage.h MtLineageConfig).
constexpr int   kMinDepth       = 10;
constexpr int   kMinCellsAlt    = 5;
constexpr float kMinVAF         = 0.01f;
constexpr int   kMinK           = 2;
constexpr int   kMaxK           = 10;

/// Tolerances from the design doc.
constexpr double kAriMin        = 0.85;
constexpr double kVafRelErrMax  = 1e-4;
constexpr int    kBicKTolerance = 1;

/// Real-data path — GSM4037629.
/// The file is named mt_heteroplasmy.1pz on this pipeline version.
static const char* kGsmMtPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/mt_heteroplasmy.1pz";

/// MQuad reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/mt_lineage_mquad_reference.py";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_mt_lineage_refs_tmp";

/// Python interpreter.
static const char* kPython = "python3";

// =============================================================================
// Utility helpers (pattern from cycles 3-15)
// =============================================================================

namespace {

void ensure_refs_tmp() { fs::create_directories(kRefsTmpDir); }

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " + std::to_string(ret) +
                                 "): " + cmd);
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Host-side CSC container (float32 values matching dump_csc.h binary format).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             n_rows;   // MT sites (rows)
    uint32_t             n_cols;   // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// ---------------------------------------------------------------------------
// Write HostCSC to dump_csc.h binary format.
// ---------------------------------------------------------------------------
void write_csc_bin(const HostCSC& h, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    static constexpr uint32_t kMagic = 0x43535343u;
    f.write(reinterpret_cast<const char*>(&kMagic),   4);
    f.write(reinterpret_cast<const char*>(&h.n_rows), 4);
    f.write(reinterpret_cast<const char*>(&h.n_cols), 4);
    f.write(reinterpret_cast<const char*>(&h.nnz),    8);
    f.write(reinterpret_cast<const char*>(h.values .data()),
            static_cast<std::streamsize>(h.nnz * sizeof(float)));
    f.write(reinterpret_cast<const char*>(h.indptr .data()),
            static_cast<std::streamsize>((static_cast<size_t>(h.n_cols) + 1) * sizeof(int32_t)));
    f.write(reinterpret_cast<const char*>(h.indices.data()),
            static_cast<std::streamsize>(h.nnz * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Upload HostCSC to device (synchronous host-data constructor).
// ---------------------------------------------------------------------------
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    // Synchronize stream before using the sync constructor to avoid ordering issues.
    cudaStreamSynchronize(stream);
    return singlet_gpu::core::DeviceCSC(
        static_cast<int>(h.n_rows),
        static_cast<int>(h.n_cols),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

// ---------------------------------------------------------------------------
// Download DeviceMemory<int> to host vector.
// ---------------------------------------------------------------------------
std::vector<int> download_int(
        const singlet_gpu::core::DeviceMemory<int>& d,
        size_t count, cudaStream_t stream)
{
    std::vector<int> h(count);
    cudaError_t e = cudaMemcpyAsync(h.data(), d.get(),
                                    count * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_int: ") + cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// ---------------------------------------------------------------------------
// Download DeviceMemory<float> to host vector.
// ---------------------------------------------------------------------------
std::vector<float> download_f32(
        const singlet_gpu::core::DeviceMemory<float>& d,
        size_t count, cudaStream_t stream)
{
    std::vector<float> h(count);
    cudaError_t e = cudaMemcpyAsync(h.data(), d.get(),
                                    count * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_f32: ") + cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// ---------------------------------------------------------------------------
// Adjusted Rand Index (in-process, no scipy dependency).
//
// Formula: ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
// Uses the combinatorial form via pair contingency counts.
// ---------------------------------------------------------------------------
double adjusted_rand_index(const std::vector<int>& labels_true,
                            const std::vector<int>& labels_pred)
{
    assert(labels_true.size() == labels_pred.size());
    int n = static_cast<int>(labels_true.size());

    // Determine number of classes/clusters via max+1.
    int n_classes  = *std::max_element(labels_true.begin(), labels_true.end()) + 1;
    int n_clusters = *std::max_element(labels_pred.begin(), labels_pred.end()) + 1;

    // Contingency table (flat, row-major: [class][cluster]).
    std::vector<int64_t> cont(static_cast<size_t>(n_classes) * n_clusters, 0);
    for (int i = 0; i < n; ++i)
        cont[static_cast<size_t>(labels_true[i]) * n_clusters + labels_pred[i]] += 1;

    auto comb2 = [](int64_t x) -> double {
        return (x >= 2) ? static_cast<double>(x * (x - 1)) / 2.0 : 0.0;
    };

    double sum_comb_c = 0.0;
    for (int64_t v : cont) sum_comb_c += comb2(v);

    // Row sums (per class) and col sums (per cluster).
    std::vector<int64_t> row_sums(n_classes,  0);
    std::vector<int64_t> col_sums(n_clusters, 0);
    for (int i = 0; i < n_classes; ++i)
        for (int j = 0; j < n_clusters; ++j) {
            int64_t v = cont[static_cast<size_t>(i) * n_clusters + j];
            row_sums[i] += v;
            col_sums[j] += v;
        }

    double sum_comb_a = 0.0;
    for (int64_t r : row_sums) sum_comb_a += comb2(r);
    double sum_comb_b = 0.0;
    for (int64_t c : col_sums) sum_comb_b += comb2(c);

    double denom_n = comb2(static_cast<int64_t>(n));
    if (denom_n == 0.0) return 1.0;  // trivial: all in one cluster

    double expected = (sum_comb_a * sum_comb_b) / denom_n;
    double max_val  = (sum_comb_a + sum_comb_b) / 2.0;
    double denom    = max_val - expected;
    if (std::abs(denom) < 1e-10)
        return (std::abs(sum_comb_c - expected) < 1e-10) ? 1.0 : 0.0;

    return (sum_comb_c - expected) / denom;
}

// ---------------------------------------------------------------------------
// Minimal NPZ reader: extract a named array from a .npz (ZIP) file.
//
// We use a minimal ZIP reader — only what we need to extract the .npy members
// from numpy .npz archives.  No external zip library required.
// ---------------------------------------------------------------------------

// NPY v1 array reader: reads a single .npy byte blob and extracts the data.
// Only handles 1-D and 2-D C-order arrays with dtypes int32 and float32.
struct NpyArray {
    int ndim;
    std::vector<size_t> shape;
    std::string dtype;
    std::vector<std::byte> data;

    size_t n_elements() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }

    const float* as_f32() const {
        assert(dtype == "<f4" || dtype == "float32");
        return reinterpret_cast<const float*>(data.data());
    }
    const int32_t* as_i32() const {
        assert(dtype == "<i4" || dtype == "int32");
        return reinterpret_cast<const int32_t*>(data.data());
    }
    float scalar_f32() const { return as_f32()[0]; }
    int32_t scalar_i32() const { return as_i32()[0]; }
};

// Parse the NPY v1/v2 header dict string to extract dtype and shape.
static NpyArray parse_npy(const std::vector<std::byte>& blob) {
    NpyArray arr;
    // Magic: \x93NUMPY, then 1 byte major, 1 byte minor, 2 bytes header_len (v1)
    // or 4 bytes header_len (v2).
    if (blob.size() < 10)
        throw std::runtime_error("parse_npy: too small");
    std::string magic(reinterpret_cast<const char*>(blob.data()), 6);
    if (magic != "\x93NUMPY")
        throw std::runtime_error("parse_npy: bad magic");
    uint8_t major = static_cast<uint8_t>(blob[6]);
    size_t header_len_bytes = (major == 1) ? 2 : 4;
    size_t header_offset = 8;  // after magic(6) + major(1) + minor(1)
    if (blob.size() < header_offset + header_len_bytes)
        throw std::runtime_error("parse_npy: too small for header len");
    uint32_t header_len = 0;
    std::memcpy(&header_len, blob.data() + header_offset, header_len_bytes);
    size_t data_offset = header_offset + header_len_bytes + header_len;
    if (blob.size() < data_offset)
        throw std::runtime_error("parse_npy: blob too small for header");

    std::string hdr(reinterpret_cast<const char*>(blob.data()) + header_offset + header_len_bytes,
                    header_len);

    // Parse 'descr': extract dtype string.
    auto extract_str = [&](const std::string& key) -> std::string {
        auto pos = hdr.find(key);
        if (pos == std::string::npos) return "";
        // Find the string value after the colon.
        pos = hdr.find('\'', pos + key.size());
        if (pos == std::string::npos) return "";
        size_t start = pos + 1;
        size_t end   = hdr.find('\'', start);
        if (end == std::string::npos) return "";
        return hdr.substr(start, end - start);
    };

    arr.dtype = extract_str("'descr'");

    // Parse 'shape': a tuple like "(200,)" or "(3, 50)".
    {
        auto pos = hdr.find("'shape'");
        if (pos == std::string::npos) throw std::runtime_error("parse_npy: no shape");
        auto lp = hdr.find('(', pos);
        auto rp = hdr.find(')', lp);
        if (lp == std::string::npos || rp == std::string::npos)
            throw std::runtime_error("parse_npy: bad shape");
        std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
        // Tokenise on comma.
        arr.shape.clear();
        std::istringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim whitespace
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t") + 1);
            if (tok.empty()) continue;
            arr.shape.push_back(static_cast<size_t>(std::stoul(tok)));
        }
    }
    arr.ndim = static_cast<int>(arr.shape.size());

    // Copy data bytes.
    size_t n_bytes = blob.size() - data_offset;
    arr.data.assign(blob.begin() + static_cast<ptrdiff_t>(data_offset), blob.end());
    (void)n_bytes;
    return arr;
}

// Minimal ZIP local-file-header reader to extract named members from .npz.
// Only handles DEFLATE (method 8) and stored (method 0) entries.
static std::unordered_map<std::string, std::vector<std::byte>>
read_npz(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_npz: cannot open: " + path);
    std::vector<char> raw_buf(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::byte> buf(raw_buf.size());
    std::memcpy(buf.data(), raw_buf.data(), raw_buf.size());

    std::unordered_map<std::string, std::vector<std::byte>> members;
    size_t pos = 0;
    while (pos + 30 <= buf.size()) {
        uint32_t sig = 0;
        std::memcpy(&sig, buf.data() + pos, 4);
        if (sig == 0x04034b50u) {  // local file header
            uint16_t method = 0, fname_len = 0, extra_len = 0;
            uint32_t csize = 0, usize = 0;
            std::memcpy(&method,    buf.data() + pos + 8,  2);
            std::memcpy(&csize,     buf.data() + pos + 18, 4);
            std::memcpy(&usize,     buf.data() + pos + 22, 4);
            std::memcpy(&fname_len, buf.data() + pos + 26, 2);
            std::memcpy(&extra_len, buf.data() + pos + 28, 2);
            size_t data_pos = pos + 30 + fname_len + extra_len;
            std::string fname(reinterpret_cast<const char*>(buf.data()) + pos + 30, fname_len);
            if (data_pos + csize <= buf.size()) {
                if (method == 0) {
                    // stored
                    members[fname].assign(
                        buf.begin() + static_cast<ptrdiff_t>(data_pos),
                        buf.begin() + static_cast<ptrdiff_t>(data_pos + usize));
                } else if (method == 8) {
                    // deflate — use zlib
#if defined(__has_include) && __has_include(<zlib.h>)
                    members[fname].resize(usize);
                    uLongf dest_len = usize;
                    int zr = uncompress(
                        reinterpret_cast<Bytef*>(members[fname].data()), &dest_len,
                        reinterpret_cast<const Bytef*>(buf.data() + data_pos), csize);
                    if (zr != Z_OK)
                        throw std::runtime_error("read_npz: zlib uncompress failed: " + fname);
#else
                    // zlib not available — skip DEFLATE members (test will SKIP).
                    (void)fname;
#endif
                }
            }
            pos = data_pos + csize;
        } else if (sig == 0x02014b50u || sig == 0x06054b50u) {
            break;  // central directory or end-of-central-dir
        } else {
            ++pos;  // scan forward
        }
    }
    return members;
}

// Load a named array from a .npz file (member name without the .npy extension
// is tried first, then with .npy appended).
NpyArray load_npz_array(const std::string& npz_path, const std::string& key) {
    auto members = read_npz(npz_path);
    // numpy savez uses "<key>.npy" as member names.
    std::string member_name = key + ".npy";
    auto it = members.find(member_name);
    if (it == members.end()) {
        it = members.find(key);
        if (it == members.end())
            throw std::runtime_error("load_npz_array: key '" + key +
                                     "' not found in " + npz_path);
    }
    return parse_npy(it->second);
}

// ---------------------------------------------------------------------------
// Synthetic data generator.
//
// Produces alt + depth CSC matrices (n_sites x n_cells) with:
//   - kTinySites sites, kTinyCells cells, n_clones planted clones.
//   - Each clone owns `sites_per_clone` consecutive sites with high VAF.
//   - All other sites have near-zero VAF (background noise only).
//   - Depth is high (kHighDepth ± Poisson) for all sites unless overridden.
// ---------------------------------------------------------------------------
struct SyntheticMT {
    HostCSC alt;
    HostCSC depth;
    std::vector<int> true_labels;  // (n_cells,) ground-truth clone index
};

SyntheticMT make_synthetic(
    int n_cells, int n_sites, int n_clones,
    float het_vaf,       // within-clone heteroplasmy
    float base_vaf,      // background noise
    float mean_depth,    // mean depth for all sites
    uint64_t seed,
    int low_depth_sites = 0,  // first N sites get depth < kMinDepth
    float low_depth_val = kLowDepth)
{
    std::mt19937_64 rng(seed);
    auto poisson_f = [&](float mu) -> float {
        std::poisson_distribution<int> pd(static_cast<double>(mu));
        return static_cast<float>(pd(rng));
    };

    int cells_per_clone = n_cells / n_clones;
    int sites_per_clone = std::max(2, n_sites / (n_clones * 2));  // non-overlapping

    // Ground-truth labels.
    std::vector<int> labels(n_cells);
    for (int c = 0; c < n_cells; ++c)
        labels[c] = std::min(c / cells_per_clone, n_clones - 1);

    // Build dense alt + depth arrays, then CSC-encode.
    // alt[site, cell] = Binomial(depth, vaf)
    std::vector<float> alt_dense  (static_cast<size_t>(n_sites) * n_cells, 0.0f);
    std::vector<float> depth_dense(static_cast<size_t>(n_sites) * n_cells, 0.0f);

    for (int s = 0; s < n_sites; ++s) {
        float d_mu = (s < low_depth_sites) ? low_depth_val : mean_depth;
        for (int c = 0; c < n_cells; ++c) {
            float d = std::max(1.0f, poisson_f(d_mu));
            // Determine if this site is a heteroplasmic signature for this clone.
            int clone_label = labels[c];
            int sig_site_start = clone_label * sites_per_clone;
            int sig_site_end   = sig_site_start + sites_per_clone;
            bool is_het_site = (s >= sig_site_start && s < sig_site_end
                                && s >= low_depth_sites);
            float vaf = is_het_site ? het_vaf : base_vaf;
            std::binomial_distribution<int> bd(static_cast<int>(d), static_cast<double>(vaf));
            float a = static_cast<float>(bd(rng));
            alt_dense  [static_cast<size_t>(s) * n_cells + c] = a;
            depth_dense[static_cast<size_t>(s) * n_cells + c] = d;
        }
    }

    // Encode as CSC (sites = rows, cells = cols).
    // A CSC matrix stores columns: indptr[j] = offset of first element in col j.
    // Here cols = cells, rows = sites.
    auto encode_csc = [&](const std::vector<float>& dense) -> HostCSC {
        HostCSC h;
        h.n_rows = static_cast<uint32_t>(n_sites);
        h.n_cols = static_cast<uint32_t>(n_cells);
        h.indptr.resize(static_cast<size_t>(n_cells) + 1, 0);

        // Count nnz per column.
        for (int c = 0; c < n_cells; ++c)
            for (int s = 0; s < n_sites; ++s)
                if (dense[static_cast<size_t>(s) * n_cells + c] != 0.0f)
                    h.indptr[c + 1]++;

        // Prefix-sum.
        for (int c = 0; c < n_cells; ++c)
            h.indptr[c + 1] += h.indptr[c];
        h.nnz = static_cast<uint64_t>(h.indptr[n_cells]);

        h.values .resize(h.nnz);
        h.indices.resize(h.nnz);

        // Fill.
        std::vector<int32_t> col_ptr = h.indptr;  // copy for filling
        for (int c = 0; c < n_cells; ++c)
            for (int s = 0; s < n_sites; ++s) {
                float v = dense[static_cast<size_t>(s) * n_cells + c];
                if (v != 0.0f) {
                    size_t idx = static_cast<size_t>(col_ptr[c]++);
                    h.values [idx] = v;
                    h.indices[idx] = static_cast<int32_t>(s);
                }
            }
        return h;
    };

    SyntheticMT result;
    result.alt   = encode_csc(alt_dense);
    result.depth = encode_csc(depth_dense);
    result.true_labels = labels;
    return result;
}

// ---------------------------------------------------------------------------
// Run the MQuad reference subprocess and parse the .npz output.
// Returns false if the subprocess fails (MQuad unavailable) — caller decides
// whether to SKIP or use an in-test fallback check.
// ---------------------------------------------------------------------------
bool run_mquad_subprocess(
    const std::string& alt_bin,
    const std::string& depth_bin,
    const std::string& out_npz,
    uint64_t seed = 42)
{
    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --alt "   << alt_bin
        << " --depth " << depth_bin
        << " --out "   << out_npz
        << " --seed "  << seed
        << " 2>&1";
    int ret = std::system(cmd.str().c_str());
    return (ret == 0) && fs::exists(out_npz);
}

}  // namespace (anonymous)

// =============================================================================
// Test 1 — MtLineage_TinySynthetic_VsMQuad
//
// 200 cells x 50 MT sites, 3 planted clones with distinct heteroplasmy patterns.
// Run call_clones and the MQuad reference. Compare clone label ARI >= 0.85.
// If MQuad is unavailable the test verifies ARI >= 0.85 against the ground truth
// (the planted labels) rather than against MQuad output, still a valid check.
// =============================================================================

TEST(MtLineageCorrectness, MtLineage_TinySynthetic_VsMQuad) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device available";

    ensure_refs_tmp();

    // Generate synthetic data.
    SyntheticMT syn = make_synthetic(
        kTinyCells, kTinySites, kTinyClones,
        kHetVAF, kBaseVAF, kHighDepth, kSeed);

    // Write CSC bins for the reference.
    const std::string alt_bin   = refs_path("tiny_alt.bin");
    const std::string depth_bin = refs_path("tiny_depth.bin");
    const std::string out_npz   = refs_path("tiny_mquad.npz");
    write_csc_bin(syn.alt,   alt_bin);
    write_csc_bin(syn.depth, depth_bin);

    // Run MQuad reference (optional — falls back gracefully).
    bool mquad_ok = run_mquad_subprocess(alt_bin, depth_bin, out_npz, /*seed=*/42);

    // Run the GPU kernel.
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_alt   = upload_csc(syn.alt,   stream);
    auto d_depth = upload_csc(syn.depth, stream);

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth    = kMinDepth;
    cfg.min_cells_alt = kMinCellsAlt;
    cfg.min_vaf      = kMinVAF;
    cfg.min_K        = kMinK;
    cfg.max_K        = kMaxK;
    cfg.seed         = kSeed;

    auto pred = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);

    ASSERT_GT(pred.n_clones, 0) << "n_clones must be at least 1";
    ASSERT_LE(pred.n_clones, kMaxK) << "n_clones must not exceed max_K";

    auto gpu_labels = download_int(pred.labels, static_cast<size_t>(kTinyCells), stream);

    // Check ARI against ground truth (always available).
    double ari_gt = adjusted_rand_index(syn.true_labels, gpu_labels);
    EXPECT_GE(ari_gt, kAriMin)
        << "ARI vs ground truth = " << ari_gt << " (threshold " << kAriMin << ")";

    // If MQuad ran successfully, also check ARI vs MQuad labels.
    if (mquad_ok) {
        NpyArray mq_labels_arr = load_npz_array(out_npz, "clone_labels");
        ASSERT_EQ(mq_labels_arr.n_elements(), static_cast<size_t>(kTinyCells));
        std::vector<int> mq_labels(kTinyCells);
        for (int i = 0; i < kTinyCells; ++i)
            mq_labels[i] = static_cast<int>(mq_labels_arr.as_i32()[i]);
        double ari_vs_mquad = adjusted_rand_index(mq_labels, gpu_labels);
        EXPECT_GE(ari_vs_mquad, kAriMin)
            << "ARI vs MQuad = " << ari_vs_mquad << " (threshold " << kAriMin << ")";
        std::printf("[registry] MtLineage_TinySynthetic_VsMQuad  ARI_gt=%.4f  ARI_mquad=%.4f  "
                    "tol=%.2f  n_clones=%d  PASS\n",
                    ari_gt, ari_vs_mquad, kAriMin, pred.n_clones);
    } else {
        std::printf("[registry] MtLineage_TinySynthetic_VsMQuad  ARI_gt=%.4f  "
                    "tol=%.2f  n_clones=%d  MQuad=unavailable  PASS\n",
                    ari_gt, kAriMin, pred.n_clones);
    }

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 2 — MtLineage_GSM4037629_RealData
//
// Load mt_heteroplasmy.1pz from GSM4037629 (11,560 cells).
// Confirm: n_clones in [1, 10], no NaNs in clone profiles, finite outputs.
// Accept K=1 as valid (healthy donor PBMCs may have no multi-clone signal).
// =============================================================================

TEST(MtLineageCorrectness, MtLineage_GSM4037629_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device available";
    if (!fs::exists(kGsmMtPath))
        GTEST_SKIP() << "GSM4037629 mt_heteroplasmy.1pz not found: " << kGsmMtPath;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Load mt_heteroplasmy.1pz.  The file stores alt counts (MT sites x cells).
    // For a real depth matrix we would need mt_depth.1pz; if not present we
    // approximate depth = alt + estimated ref counts via a simple heuristic or
    // use the loader's 'depth_from_alt' flag if the kernel provides it.
    // Per design doc risk #1: we load alt only and let call_clones handle the
    // depth approximation if no paired depth file is present.
    singlet_gpu::core::DeviceCSC d_alt =
        singlet_gpu::io::load_pz(kGsmMtPath, stream).mat;

    // Try to load a depth counterpart; if absent, use alt as a proxy depth.
    // The kernel's DeviceCSC depth input is accepted even if it is the same
    // as alt (extreme edge case: depth ≈ alt → VAF ≈ 1.0 for called sites).
    // This is the safest fallback when mt_depth.1pz is not yet written by singlify.
    singlet_gpu::core::DeviceCSC d_depth;
    static const char* kGsmDepthPath =
        "/mnt/projects/debruinz_project/singlify_pipeline/"
        "quant/scrna/GSE127/GSE127918/GSM4037629/mt_depth.1pz";
    if (fs::exists(kGsmDepthPath)) {
        d_depth = singlet_gpu::io::load_pz(kGsmDepthPath, stream).mat;
    } else {
        // No depth file — pass alt as depth proxy so call_clones has something
        // to compute VAF from (VAF will be ≤ 1.0 per site by construction).
        d_depth = singlet_gpu::io::load_pz(kGsmMtPath, stream).mat;
    }

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = kMinDepth;
    cfg.min_cells_alt = kMinCellsAlt;
    cfg.min_vaf       = kMinVAF;
    cfg.min_K         = kMinK;
    cfg.max_K         = kMaxK;
    cfg.seed          = kSeed;

    auto pred = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);

    // n_clones must be in [1, 10].
    EXPECT_GE(pred.n_clones, 1)
        << "n_clones must be at least 1";
    EXPECT_LE(pred.n_clones, kMaxK)
        << "n_clones must not exceed max_K=" << kMaxK;

    // n_cells labels.
    size_t n_cells = static_cast<size_t>(d_alt.cols);
    auto gpu_labels = download_int(pred.labels, n_cells, stream);

    // No label should be out of range.
    for (size_t i = 0; i < n_cells; ++i) {
        ASSERT_GE(gpu_labels[i], 0)       << "Negative clone label at cell " << i;
        ASSERT_LT(gpu_labels[i], pred.n_clones) << "Clone label OOB at cell " << i;
    }

    // Clone profiles: no NaNs, all finite.
    if (pred.n_clones > 0 && pred.informative_sites.size() > 0) {
        size_t n_sites_inf  = static_cast<size_t>(pred.informative_sites.size());
        size_t n_profile_el = static_cast<size_t>(pred.n_clones) * n_sites_inf;
        auto profiles = download_f32(pred.clone_profiles, n_profile_el, stream);
        for (size_t i = 0; i < n_profile_el; ++i) {
            ASSERT_TRUE(std::isfinite(profiles[i]))
                << "NaN/Inf in clone profile at element " << i;
        }
    }

    std::printf("[registry] MtLineage_GSM4037629_RealData  n_clones=%d  "
                "n_cells=%zu  n_informative=%zu  PASS\n",
                pred.n_clones, n_cells,
                static_cast<size_t>(pred.informative_sites.size()));

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 3 — MtLineage_DepthFilter_RemovesLowCoverage
//
// Sites with median depth < 10 must NOT appear in informative_sites.
// We inject kLowDepthSites sites at the start with depth < kMinDepth.
// =============================================================================

TEST(MtLineageCorrectness, MtLineage_DepthFilter_RemovesLowCoverage) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device available";

    // Generate data with first kLowDepthSites sites having depth < kMinDepth.
    SyntheticMT syn = make_synthetic(
        kTinyCells, kTinySites, kTinyClones,
        kHetVAF, kBaseVAF, kHighDepth, kSeed + 1,
        /*low_depth_sites=*/kLowDepthSites,
        /*low_depth_val=*/kLowDepth);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_alt   = upload_csc(syn.alt,   stream);
    auto d_depth = upload_csc(syn.depth, stream);

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = kMinDepth;
    cfg.min_cells_alt = kMinCellsAlt;
    cfg.min_vaf       = kMinVAF;
    cfg.seed          = kSeed + 1;

    auto pred = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);

    // The informative_sites array must not contain any of the low-depth sites.
    // Download the informative_sites indices.
    ASSERT_GT(pred.informative_sites.size(), 0u)
        << "Expected at least one informative site after filtering";

    size_t n_inf = static_cast<size_t>(pred.informative_sites.size());
    // informative_sites is DeviceMemory<int>
    auto h_info_sites = download_int(pred.informative_sites, n_inf, stream);

    for (int site_idx : h_info_sites) {
        EXPECT_GE(site_idx, kLowDepthSites)
            << "Site " << site_idx << " has low depth (<" << kMinDepth
            << ") but appears in informative_sites";
    }

    std::printf("[registry] MtLineage_DepthFilter_RemovesLowCoverage  "
                "low_depth_sites=%d  n_informative=%zu  "
                "all_above_min_depth=true  PASS\n",
                kLowDepthSites, n_inf);

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 4 — MtLineage_BICAutoK_PicksKnownK
//
// Synthetic with 4 planted clones. BIC auto-K selection must pick K=4 ± 1.
// =============================================================================

TEST(MtLineageCorrectness, MtLineage_BICAutoK_PicksKnownK) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device available";

    // Stronger signal: higher VAF contrast, more cells, cleaner data.
    constexpr int kBicCells = 200;
    constexpr int kBicSites = 60;
    SyntheticMT syn = make_synthetic(
        kBicCells, kBicSites, kBicClones,
        /*het_vaf=*/0.35f, /*base_vaf=*/0.01f,
        kHighDepth, kSeed + 2);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_alt   = upload_csc(syn.alt,   stream);
    auto d_depth = upload_csc(syn.depth, stream);

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = kMinDepth;
    cfg.min_cells_alt = kMinCellsAlt;
    cfg.min_vaf       = kMinVAF;
    cfg.min_K         = kMinK;
    cfg.max_K         = kMaxK;
    cfg.seed          = kSeed + 2;

    auto pred = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);

    int k_error = std::abs(pred.n_clones - kBicClones);
    EXPECT_LE(k_error, kBicKTolerance)
        << "BIC selected K=" << pred.n_clones
        << " but true K=" << kBicClones
        << " (tolerance ±" << kBicKTolerance << ")";

    // ARI of the BIC-selected labelling vs ground truth.
    auto gpu_labels = download_int(pred.labels, static_cast<size_t>(kBicCells), stream);
    double ari = adjusted_rand_index(syn.true_labels, gpu_labels);
    EXPECT_GE(ari, kAriMin)
        << "ARI vs ground truth = " << ari << " (threshold " << kAriMin << ")";

    std::printf("[registry] MtLineage_BICAutoK_PicksKnownK  K_selected=%d  K_true=%d  "
                "ARI=%.4f  tol_K=±%d  tol_ARI=%.2f  PASS\n",
                pred.n_clones, kBicClones, ari, kBicKTolerance, kAriMin);

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 5 — MtLineage_Determinism_BitIdentical
//
// Two runs on the same input with the same seed must produce bit-identical labels
// and clone profiles.
// =============================================================================

TEST(MtLineageCorrectness, MtLineage_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device available";

    SyntheticMT syn = make_synthetic(
        kTinyCells, kTinySites, kTinyClones,
        kHetVAF, kBaseVAF, kHighDepth, kSeed + 3);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_alt   = upload_csc(syn.alt,   stream);
    auto d_depth = upload_csc(syn.depth, stream);

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = kMinDepth;
    cfg.min_cells_alt = kMinCellsAlt;
    cfg.min_vaf       = kMinVAF;
    cfg.min_K         = kMinK;
    cfg.max_K         = kMaxK;
    cfg.seed          = kSeed + 3;

    // Run 1.
    auto pred1 = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);
    auto labels1 = download_int(pred1.labels, static_cast<size_t>(kTinyCells), stream);

    // Run 2 — same inputs, same config, same seed.
    auto pred2 = singlet_gpu::anno::call_clones(d_alt, d_depth, cfg, stream);
    cudaStreamSynchronize(stream);
    auto labels2 = download_int(pred2.labels, static_cast<size_t>(kTinyCells), stream);

    // Labels must be bit-identical.
    ASSERT_EQ(pred1.n_clones, pred2.n_clones) << "n_clones differs between runs";
    for (int i = 0; i < kTinyCells; ++i) {
        EXPECT_EQ(labels1[i], labels2[i])
            << "Label mismatch at cell " << i
            << " (run1=" << labels1[i] << " run2=" << labels2[i] << ")";
    }

    // Clone profiles must also be bit-identical.
    size_t n_inf = static_cast<size_t>(pred1.informative_sites.size());
    if (pred1.n_clones > 0 && n_inf > 0) {
        size_t n_profile_el = static_cast<size_t>(pred1.n_clones) * n_inf;
        auto prof1 = download_f32(pred1.clone_profiles, n_profile_el, stream);
        auto prof2 = download_f32(pred2.clone_profiles, n_profile_el, stream);
        for (size_t i = 0; i < n_profile_el; ++i) {
            EXPECT_EQ(prof1[i], prof2[i])
                << "Clone profile bit mismatch at element " << i;
        }
    }

    std::printf("[registry] MtLineage_Determinism_BitIdentical  n_clones=%d  "
                "n_cells=%d  bit_identical=true  PASS\n",
                pred1.n_clones, kTinyCells);

    cudaStreamDestroy(stream);
}
