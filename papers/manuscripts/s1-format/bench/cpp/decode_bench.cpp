// SPDX-License-Identifier: MIT
// Decode timing of one pz2 matrix member in C++ (no Python in the loop), for the decode stage and Fig. 4.
//   decode_bench <member.pz2> <threads> <isa: scalar|avx2|avx512|auto> <mode: native|canonical> <repeats> <warmup>
// Prints one JSON line: {"nnz":..., "times":[...], "isa":..., "mode":..., "threads":...}.
// The member is read into memory first; the timed region is Matrix::open + Matrix::decode into caller-owned CSC arrays,
// which is the complete work a C++ reader does (the calls shown in Fig. 4).
#include <singlet/pz2/pz2.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace pz = singlet::pz2;

static std::vector<uint8_t> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: decode_bench member.pz2 threads isa mode repeats warmup\n"); return 2; }
    std::vector<uint8_t> buf = read_file(argv[1]);
    pz::DecodeOptions opt;
    opt.threads = std::atoi(argv[2]);
    std::string isa = argv[3], mode = argv[4];
    opt.isa = isa == "scalar" ? pz::Isa::Scalar : isa == "avx2" ? pz::Isa::Avx2 : isa == "avx512" ? pz::Isa::Avx512 : pz::Isa::Auto;
    opt.mode = mode == "canonical" ? pz::Mode::Canonical : pz::Mode::Native;
    const int reps = std::atoi(argv[5]), warm = std::atoi(argv[6]);
    pz::ByteSource src(buf);
    pz::Matrix m0 = pz::Matrix::open(src);
    const uint64_t nnz = m0.header().nnz;
    const uint32_t n = m0.header().n;
    std::vector<uint64_t> indptr(n + 1);
    std::vector<uint32_t> indices(nnz);
    std::vector<uint32_t> data(nnz);
    std::string times;
    for (int k = 0; k < warm + reps; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        pz::Matrix m = pz::Matrix::open(src);
        pz::Output out;
        out.indptr = indptr.data(); out.indptr_type = pz::IntType::U64; out.indptr_capacity = indptr.size();
        out.indices = indices.data(); out.indices_type = pz::IntType::U32; out.indices_capacity = indices.size();
        out.data = data.data(); out.data_type = pz::DType::U32; out.data_capacity = data.size();
        m.decode(src, out, opt);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (k >= warm) times += (times.empty() ? "" : ",") + std::to_string(s);
    }
    // isa: what was asked for; isa_used: the kernel that ran (a request the CPU cannot serve falls back, e.g. AVX-512
    // without VBMI2 runs AVX2)
    std::printf("{\"nnz\":%llu,\"threads\":%d,\"isa\":\"%s\",\"isa_used\":\"%s\",\"mode\":\"%s\",\"times\":[%s]}\n",
                (unsigned long long)nnz, opt.threads, isa.c_str(), pz::detail::isa_name(pz::resolve_isa(opt.isa)), mode.c_str(),
                times.c_str());
    return 0;
}
