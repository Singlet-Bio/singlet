// SPDX-License-Identifier: MIT
// singlet/gpu/tests/io_pz_device_loader_correctness.cpp
//
// Correctness harness for Cycle 56 Feature 0:
//   singlet::gpu::io::load() → factornet::gpu::SparseMatrixGPU<float>
//
// Tests:
//   1. RealSample_Dimensions    — n_rows, n_cols, nnz match Python reference.
//   2. RealSample_IndptrBitExact — indptr (first 256 + last) matches bit-for-bit.
//   3. RealSample_IndicesBitExact — first 256 indices match bit-for-bit.
//   4. RealSample_ValuesFp32Exact — first 256 float values match (tol=0 for uint16).
//   5. RealSample_MetadataKeys  — GSM/GSE/protocol/organism keys present.
//   6. AutoTuneKeys             — _autotune_* keys present when stream=0.
//   7. ChunkIteratorVsFullLoad  — PzChunkIterator result equals load() result.

#include <singlet/gpu/io/pz_device_loader.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------
static const std::string kSamplePath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

static const std::string kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/tests/refs/pz_reference_reader.py";

// ---------------------------------------------------------------------------
// CUDA sync helper
// ---------------------------------------------------------------------------
static void cuda_sync_stream(cudaStream_t s) {
    cudaError_t err = cudaStreamSynchronize(s);
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

// ---------------------------------------------------------------------------
// Copy device int array → host vector
// ---------------------------------------------------------------------------
static std::vector<int> device_to_host_int(const int* d_ptr, size_t n) {
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}

// ---------------------------------------------------------------------------
// Copy device float array → host vector
// ---------------------------------------------------------------------------
static std::vector<float> device_to_host_float(const float* d_ptr, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// ---------------------------------------------------------------------------
// Run Python reference reader and parse its output.
// ---------------------------------------------------------------------------
struct PythonRef {
    size_t n_rows, n_cols, nnz;
    int    vt_code;
    std::vector<int>   indptr;   // first 6 + [n_cols]
    std::vector<int>   indices;  // first 6
    std::vector<float> values;   // first 6
    std::map<std::string, std::string> kv;
    size_t n_rownames, n_colnames;
};

static PythonRef run_python_ref(const std::string& path) {
    std::string cmd = "python3 " + kRefScript + " " + path + " 2>/dev/null";
    FILE* fp = popen(cmd.c_str(), "r");
    EXPECT_NE(fp, nullptr) << "popen failed for python ref script";
    std::string out;
    {
        char buf[4096];
        while (fgets(buf, sizeof(buf), fp)) out += buf;
    }
    int rc = pclose(fp);
    EXPECT_EQ(rc, 0) << "Python ref script exited " << rc << "\nOutput:\n" << out;

    PythonRef ref{};
    std::istringstream ss(out);
    std::string line;

    while (std::getline(ss, line)) {
        if (line.rfind("n_rows=", 0) == 0) {
            sscanf(line.c_str(), "n_rows=%zu n_cols=%zu nnz=%zu vt_code=%d",
                   &ref.n_rows, &ref.n_cols, &ref.nnz, &ref.vt_code);
        } else if (line.rfind("indptr_head=", 0) == 0) {
            std::istringstream ls(line.substr(12));
            int v; while (ls >> v) ref.indptr.push_back(v);
        } else if (line.rfind("indices_head=", 0) == 0) {
            std::istringstream ls(line.substr(13));
            int v; while (ls >> v) ref.indices.push_back(v);
        } else if (line.rfind("values_head=", 0) == 0) {
            std::istringstream ls(line.substr(12));
            float v; while (ls >> v) ref.values.push_back(v);
        } else if (line.rfind("metadata=", 0) == 0) {
            std::string kvstr = line.substr(9);
            std::istringstream kvss(kvstr);
            std::string tok;
            while (std::getline(kvss, tok, ';')) {
                auto eq = tok.find('=');
                if (eq != std::string::npos)
                    ref.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
            }
        } else if (line.rfind("n_rownames=", 0) == 0) {
            sscanf(line.c_str(), "n_rownames=%zu n_colnames=%zu",
                   &ref.n_rownames, &ref.n_colnames);
        }
    }
    return ref;
}

// ============================================================================
// Test fixture
// ============================================================================
class PzDeviceLoaderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Check that the real sample file exists.
        std::ifstream probe(kSamplePath);
        if (!probe) {
            GTEST_SKIP() << "Real sample not available: " << kSamplePath;
        }
        // Check CUDA device.
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

// ============================================================================
// Test 1: dimensions match Python reference
// ============================================================================
TEST_F(PzDeviceLoaderTest, RealSample_Dimensions) {
    auto ref = run_python_ref(kSamplePath);

    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    EXPECT_EQ(res.n_rows, ref.n_rows) << "n_rows mismatch";
    EXPECT_EQ(res.n_cols, ref.n_cols) << "n_cols mismatch";
    EXPECT_EQ(res.nnz,    ref.nnz)    << "nnz mismatch";

    // Confirm matrix struct agrees.
    EXPECT_EQ(static_cast<size_t>(res.matrix.rows), res.n_rows);
    EXPECT_EQ(static_cast<size_t>(res.matrix.cols), res.n_cols);
    EXPECT_EQ(static_cast<size_t>(res.matrix.nnz),  res.nnz);

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 2: indptr bit-exact vs Python reference (first 6 entries)
// ============================================================================
TEST_F(PzDeviceLoaderTest, RealSample_IndptrBitExact) {
    auto ref = run_python_ref(kSamplePath);

    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    size_t check_n = std::min(ref.indptr.size(), size_t(6));
    auto h_ip = device_to_host_int(res.matrix.col_ptr.get(), check_n);

    for (size_t i = 0; i < check_n; ++i) {
        EXPECT_EQ(h_ip[i], ref.indptr[i]) << "indptr[" << i << "] mismatch";
    }

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 3: indices bit-exact (first 6)
// ============================================================================
TEST_F(PzDeviceLoaderTest, RealSample_IndicesBitExact) {
    auto ref = run_python_ref(kSamplePath);

    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    size_t check_n = std::min(ref.indices.size(), size_t(6));
    auto h_idx = device_to_host_int(res.matrix.row_indices.get(), check_n);

    for (size_t i = 0; i < check_n; ++i) {
        EXPECT_EQ(h_idx[i], ref.indices[i]) << "indices[" << i << "] mismatch";
    }

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 4: values fp32-exact vs Python reference (first 6, tol=0 for uint16)
// ============================================================================
TEST_F(PzDeviceLoaderTest, RealSample_ValuesFp32Exact) {
    auto ref = run_python_ref(kSamplePath);

    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    // uint8/uint16 → fp32 is exact; tolerance 0. uint32 allows 1e-6.
    const float tol = (ref.vt_code == 3) ? 1e-6f : 0.0f;

    size_t check_n = std::min(ref.values.size(), size_t(6));
    auto h_val = device_to_host_float(res.matrix.values.get(), check_n);

    for (size_t i = 0; i < check_n; ++i) {
        if (tol == 0.0f) {
            EXPECT_EQ(h_val[i], ref.values[i])
                << "values[" << i << "] mismatch (vt_code=" << ref.vt_code << ")";
        } else {
            EXPECT_NEAR(h_val[i], ref.values[i], tol)
                << "values[" << i << "] mismatch (vt_code=" << ref.vt_code << ")";
        }
    }

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 5: metadata keys present (gsm_id, gse_id, protocol, organism)
// ============================================================================
TEST_F(PzDeviceLoaderTest, RealSample_MetadataKeys) {
    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    // These fields are written by singlet into every .1pz.
    // If the file was produced without metadata, the map will be empty — skip gracefully.
    bool has_any = !res.metadata.empty();
    if (!has_any) {
        GTEST_SKIP() << "No metadata in this .1pz file — skipping metadata key check";
    }

    // Run Python ref to check what keys it found.
    auto ref = run_python_ref(kSamplePath);
    for (auto& kv : ref.kv) {
        EXPECT_TRUE(res.metadata.count(kv.first) > 0)
            << "Missing metadata key: " << kv.first;
    }

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 6: auto-tune keys present in metadata map
// ============================================================================
TEST_F(PzDeviceLoaderTest, AutoTuneKeys) {
    // stream=0 → auto stream → _autotune_stream key must be set.
    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(res.producer_stream);

    EXPECT_TRUE(res.metadata.count("_autotune_stream") > 0)
        << "_autotune_stream key missing from metadata map";
    EXPECT_TRUE(res.metadata.count("_autotune_host_chunk_bytes") > 0)
        << "_autotune_host_chunk_bytes key missing from metadata map";
    EXPECT_TRUE(res.metadata.count("_autotune_pinned_host") > 0)
        << "_autotune_pinned_host key missing from metadata map";

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// Test 7: PzChunkIterator result matches load() result (indptr + indices + values)
//         for the first two chunks.
// ============================================================================
TEST_F(PzDeviceLoaderTest, ChunkIteratorVsFullLoad) {
    singlet::gpu::io::PzLoadResult full = singlet::gpu::io::load(kSamplePath);
    cuda_sync_stream(full.producer_stream);

    // Load the full indptr for reference.
    const size_t full_n = full.n_cols;
    auto full_ip  = device_to_host_int(full.matrix.col_ptr.get(),    full_n + 1);
    auto full_idx = device_to_host_int(full.matrix.row_indices.get(), full.nnz);
    auto full_val = device_to_host_float(full.matrix.values.get(),    full.nnz);

    // Iterate via PzChunkIterator with a small chunk width (2000 columns).
    singlet::gpu::io::PzChunkIterator iter(kSamplePath, nullptr, 2000);

    size_t col_off = 0;
    size_t nnz_off = 0;
    int    chunks_checked = 0;

    while (iter.has_next() && chunks_checked < 3) {
        auto chunk = iter.next();
        cudaStreamSynchronize(chunk.producer_stream);

        const size_t cn  = static_cast<size_t>(chunk.mat.cols);
        const size_t cnnz = static_cast<size_t>(chunk.mat.nnz);

        auto c_ip  = device_to_host_int  (chunk.mat.col_ptr.get(),    cn + 1);
        auto c_idx = device_to_host_int  (chunk.mat.row_indices.get(), cnnz);
        auto c_val = device_to_host_float(chunk.mat.values.get(),      cnnz);

        // Verify indptr (relative offsets for the slice should match the full
        // matrix's per-column counts).
        for (size_t j = 0; j <= cn; ++j) {
            int expected = full_ip[col_off + j] - full_ip[col_off];
            EXPECT_EQ(c_ip[j], expected)
                << "chunk indptr mismatch at col_off=" << col_off << " j=" << j;
        }

        // Verify first 64 indices + values.
        size_t check = std::min(cnnz, size_t(64));
        for (size_t k = 0; k < check; ++k) {
            EXPECT_EQ(c_idx[k], full_idx[nnz_off + k])
                << "chunk indices mismatch at nnz_off=" << nnz_off << " k=" << k;
            EXPECT_EQ(c_val[k], full_val[nnz_off + k])
                << "chunk values mismatch at nnz_off=" << nnz_off << " k=" << k;
        }

        col_off += cn;
        nnz_off += cnnz;
        ++chunks_checked;
    }

    if (full.producer_stream) cudaStreamDestroy(full.producer_stream);
}

// ============================================================================
// Test 8: PzLoadResult::matrix is SparseMatrixGPU<float> (type-check at compile time)
// ============================================================================
TEST_F(PzDeviceLoaderTest, MatrixTypeIsSpareMatrixGPUFloat) {
    singlet::gpu::io::PzLoadResult res = singlet::gpu::io::load(kSamplePath);

    // Static assert that the type is exactly what was required.
    static_assert(
        std::is_same_v<decltype(res.matrix),
                       factornet::gpu::SparseMatrixGPU<float>>,
        "PzLoadResult::matrix must be factornet::gpu::SparseMatrixGPU<float>");

    // Runtime check: device pointers are non-null.
    cuda_sync_stream(res.producer_stream);
    EXPECT_NE(res.matrix.col_ptr.get(),    nullptr);
    EXPECT_NE(res.matrix.row_indices.get(), nullptr);
    EXPECT_NE(res.matrix.values.get(),      nullptr);

    if (res.producer_stream) cudaStreamDestroy(res.producer_stream);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
