// SPDX-License-Identifier: MIT
// HLA prefilter bench: where the in-process prefilter over every read spends its time, and whether a faster form writes the same reads.
//   prefilter_bench T1K_DIR RUN.1fq THREADS OUTDIR [MODES]
// MODES (default "decode decodeq ref new"), each a full pass over the .1fq on THREADS threads:
//   decode   blocks decoded as the prefilter decoded them before (quality unpacked), nothing else
//   decodeq  blocks decoded with the quality column left packed
//   ref      the prefilter as released (per-k-mer hits_ref, unpacked quality): OUTDIR/ref.fq
//   new      singlet::hla::prefilter_onefq (rolling hits, packed quality): OUTDIR/new.fq
// Prints per mode: wall s, CPU s (getrusage of the process), reads scanned and kept. Counts and times only.
// Build: g++ -std=gnu++17 -O3 -DNDEBUG -march=x86-64-v3 -pthread -I<repo>/include prefilter_bench.cpp -lzstd -lz
#include "singlet/pipeline/hla_typing.h"

#include <sys/resource.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
double cpu_s() { rusage u{}; getrusage(RUSAGE_SELF, &u); return u.ru_utime.tv_sec + u.ru_stime.tv_sec + 1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec); }

// The released prefilter_onefq, verbatim apart from hits_ref and the decode-only switches.
bool prefilter_ref(const singlet::hla::KmerSet& ks, const std::string& input, bool cdna_is_r1, int threads, FILE* out, uint64_t& next_index,
                   singlet::hla::Result& res, int mode /*0 full, 1 decode only, 2 decode packed*/) {
    singlet::fq::RunSet rs;
    try { rs.open({input}); } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return false; }
    const uint16_t bc_len = rs.bc_length();
    std::vector<std::string> dict;
    if (rs.has_bc_dict()) {
        const auto& flat = rs.bc_dict_flat();
        dict.resize(rs.bc_dict_size());
        for (size_t d = 0; d < dict.size(); ++d) {
            dict[d].resize(bc_len);
            for (uint16_t j = 0; j < bc_len; ++j) { const uint8_t b = flat[d * bc_len + j]; dict[d][j] = "ACGTN"[b < 5 ? b : 4]; }
        }
    }
    const int T = std::max(1, threads);
    const uint32_t nblk = rs.block_count();
    const uint32_t W = static_cast<uint32_t>(T) * 4;
    std::vector<std::string> buf(W);
    std::vector<uint64_t> nscan(W), nkept(W);
    const uint64_t file_base = next_index;
    for (uint32_t w0 = 0; w0 < nblk; w0 += W) {
        const uint32_t nb = std::min(W, nblk - w0);
        std::atomic<uint32_t> nx{0};
        auto work = [&]() {
            singlet::fq::DecodedBlock B;
            B.keep_packed_qual = mode == 2;
            for (uint32_t b; (b = nx++) < nb;) {
                B.clear(); rs.decode(w0 + b, B);
                const uint64_t base = file_base + rs.block_read_offset(w0 + b);
                std::string& o = buf[b]; o.clear(); nscan[b] = nkept[b] = 0;
                for (uint32_t i = 0; i < B.n_reads; ++i) {
                    const uint32_t dup = B.dup_counts.empty() ? 1u : std::max<uint32_t>(1u, B.dup_counts[i]);
                    nscan[b] += dup;
                    if (mode) continue;
                    const uint8_t* s = cdna_is_r1 ? B.r1_seq(i) : B.r2_seq(i);
                    const int L = cdna_is_r1 ? B.r1_len(i) : B.r2_len(i);
                    if (!ks.hits_ref(s, L)) continue;
                    std::string bc;
                    if (!dict.empty() && !B.bc_indices.empty() && B.bc_indices[i] < dict.size()) bc = dict[B.bc_indices[i]];
                    else {
                        const uint8_t* r = cdna_is_r1 ? B.r2_seq(i) : B.r1_seq(i);
                        const int rl = std::min<int>(bc_len ? bc_len : 16, cdna_is_r1 ? B.r2_len(i) : B.r1_len(i));
                        for (int j = 0; j < rl; ++j) bc += "ACGTN"[r[j] < 5 ? r[j] : 4];
                    }
                    const uint8_t* q = cdna_is_r1 ? nullptr : B.r2_quality(i);
                    std::string seq(L, 'N'), qual(L, 'I');
                    for (int j = 0; j < L; ++j) { seq[j] = "ACGTN"[s[j] < 5 ? s[j] : 4]; if (q) qual[j] = static_cast<char>(33 + std::min<int>(q[j], 93)); }
                    for (uint32_t d = 0; d < dup; ++d) {
                        o += "@" + std::to_string(base + i) + (dup > 1 ? "." + std::to_string(d) : std::string()) + "_" + bc + "\n" + seq + "\n+\n" + qual + "\n";
                        ++nkept[b];
                    }
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < T; ++t) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
        for (uint32_t b = 0; b < nb; ++b) { if (out) std::fwrite(buf[b].data(), 1, buf[b].size(), out); res.reads_scanned += nscan[b]; res.reads_kept += nkept[b]; }
    }
    next_index = file_base + rs.stored_reads();
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: prefilter_bench T1K_DIR RUN.1fq THREADS OUTDIR [MODES]\n"; return 2; }
    const std::string t1k = argv[1], fq = argv[2], od = argv[4];
    const int T = std::atoi(argv[3]);
    const std::string modes = argc > 5 ? argv[5] : "decode decodeq ref new";
    singlet::hla::KmerSet ks; size_t na = 0;
    auto t0 = std::chrono::steady_clock::now();
    if (!singlet::hla::load_fasta_kmers(t1k + "/hlaidx/hlaidx_rna_seq.fa", ks, na)) { std::cerr << "cannot read the index\n"; return 1; }
    std::printf("index: %zu alleles, %zu k-mers, %.1f s\n", na, ks.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::stringstream ss(modes); std::string m;
    while (ss >> m) {
        singlet::hla::Result res; uint64_t next = 0; std::string why;
        FILE* out = (m == "ref" || m == "new") ? std::fopen((od + "/" + m + ".fq").c_str(), "wb") : nullptr;
        const double c0 = cpu_s(); const auto w0 = std::chrono::steady_clock::now();
        bool ok;
        if (m == "new") ok = singlet::hla::prefilter_onefq(ks, fq, false, T, out, next, res, why);
        else ok = prefilter_ref(ks, fq, false, T, out, next, res, m == "ref" ? 0 : m == "decode" ? 1 : 2);
        const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), c = cpu_s() - c0;
        if (out) std::fclose(out);
        std::printf("MODE %-8s ok %d wall %.2f s cpu %.1f s scanned %llu kept %llu  cpu ns/read %.1f\n", m.c_str(), ok ? 1 : 0, w, c,
                    (unsigned long long)res.reads_scanned, (unsigned long long)res.reads_kept, res.reads_scanned ? 1e9 * c / res.reads_scanned : 0.0);
        std::fflush(stdout);
    }
    return 0;
}
