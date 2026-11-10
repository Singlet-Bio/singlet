// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_replay.h
// Speed work on the all-site pass (allsite.h) without whole runs: SINGLET_ALLSITE_DUMP_DIR=DIR on a run writes the pass's input records,
// parameters and contig names to DIR; `singlet allsite-replay DIR` runs the pass alone on them and prints its log line, its timing and a
// digest of the result (sites, statistics, both matrices over every barcode), so two kernels or thread counts compare in seconds.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "allsite.h"

namespace singlet { namespace allsite { namespace replay {

// one array of the input as up to two parts (PileupCat's pre part and vector part), written back to back
inline bool write_parts(const std::string& path, const void* a, uint64_t abytes, const void* b, uint64_t bbytes) {
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
    bool ok = (abytes == 0 || std::fwrite(a, 1, abytes, f) == abytes) && (bbytes == 0 || std::fwrite(b, 1, bbytes, f) == bbytes);
    return std::fclose(f) == 0 && ok;
}

inline bool write_meta(const std::string& dir, const Params& P, uint32_t n_tid, uint32_t n_barcodes, const std::string& genome_dir,
                       const std::vector<std::string>& tid_names) {
    std::ofstream m(dir + "/meta.txt");
    m << "n_tid " << n_tid << "\nn_barcodes " << n_barcodes << "\nmin_mapq " << P.min_mapq << "\nmin_alt_molecules " << P.min_alt_molecules
      << "\nmin_alt_cells " << P.min_alt_cells << "\nskip_tid " << P.skip_tid << "\nuse_qual " << P.use_qual << "\nqwords " << P.qwords
      << "\nskip_n " << P.skip_n << "\numi_dedup " << P.umi_dedup << "\ndirectional " << P.directional << "\numi_len " << P.umi_len
      << "\nthreads " << P.threads << "\nblock_hits " << P.block_hits << "\ngenome_dir " << genome_dir << "\n";
    std::ofstream t(dir + "/tids.txt"); for (const auto& n : tid_names) t << n << "\n";
    return static_cast<bool>(m) && static_cast<bool>(t);
}

struct RSig { uint32_t aln_off; uint8_t n_aln; uint8_t nh; uint16_t rlen; uint64_t seq_off; };
struct RAln { int32_t tid; int32_t pos; uint32_t ops_off; uint16_t n_ops; uint8_t flags; uint8_t mapq; };
struct RRead { uint32_t sig; int32_t bc; uint64_t umi3; };
static_assert(sizeof(RSig) == 16 && sizeof(RAln) == 16 && sizeof(RRead) == 16, "record sizes");

class Map {   // one file mapped read-only
public:
    ~Map() { if (p_ && p_ != MAP_FAILED) munmap(p_, n_); }
    bool open(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) return false;
        struct stat st{}; if (fstat(fd, &st) != 0) { close(fd); return false; }
        n_ = static_cast<size_t>(st.st_size);
        if (n_ > 0) { p_ = mmap(nullptr, n_, PROT_READ, MAP_PRIVATE, fd, 0); if (p_ == MAP_FAILED) { p_ = nullptr; close(fd); return false; } }
        close(fd); return true;
    }
    template <class T> const T* as() const { return static_cast<const T*>(p_); }
    size_t bytes() const { return n_; }
private:
    void* p_ = nullptr; size_t n_ = 0;
};

struct Input {
    Map S, A, Rd, O, Q, L;
    Params P; uint32_t n_tid = 0, n_barcodes = 0; std::string genome_dir; std::vector<std::string> tids;
    uint64_t n_sigs() const { return S.bytes() / sizeof(RSig); }
    uint64_t n_reads() const { return Rd.bytes() / sizeof(RRead); }
    const RSig& sig(uint64_t i) const { return S.as<RSig>()[i]; }
    const RAln& aln(uint64_t i) const { return A.as<RAln>()[i]; }
    const RRead& read(uint64_t i) const { return Rd.as<RRead>()[i]; }
    const uint32_t* ops_at(uint64_t off) const { return O.as<uint32_t>() + off; }
    const uint8_t* seq_at(uint64_t off) const { return Q.as<uint8_t>() + off; }
    uint64_t lowq_word(uint64_t i) const { return L.as<uint64_t>()[i]; }
    bool load(const std::string& dir, std::string* err) {
        for (auto& [m, f] : std::vector<std::pair<Map*, const char*>>{{&S, "sigs.bin"}, {&A, "alns.bin"}, {&Rd, "reads.bin"}, {&O, "ops.bin"}, {&Q, "seq.bin"}, {&L, "lowq.bin"}})
            if (!m->open(dir + "/" + f)) { if (err) *err = dir + "/" + f; return false; }
        std::ifstream m(dir + "/meta.txt"); std::string k;
        while (m >> k) {
            if (k == "genome_dir") { m >> genome_dir; continue; }
            long long v; m >> v;
            if (k == "n_tid") n_tid = static_cast<uint32_t>(v); else if (k == "n_barcodes") n_barcodes = static_cast<uint32_t>(v);
            else if (k == "min_mapq") P.min_mapq = static_cast<int>(v); else if (k == "min_alt_molecules") P.min_alt_molecules = static_cast<int>(v);
            else if (k == "min_alt_cells") P.min_alt_cells = static_cast<int>(v); else if (k == "skip_tid") P.skip_tid = static_cast<int32_t>(v);
            else if (k == "use_qual") P.use_qual = static_cast<int>(v); else if (k == "qwords") P.qwords = static_cast<int>(v);
            else if (k == "skip_n") P.skip_n = static_cast<int>(v); else if (k == "umi_dedup") P.umi_dedup = static_cast<int>(v);
            else if (k == "directional") P.directional = static_cast<int>(v); else if (k == "umi_len") P.umi_len = static_cast<int>(v);
            else if (k == "threads") P.threads = static_cast<unsigned>(v); else if (k == "block_hits") P.block_hits = static_cast<uint64_t>(v);
        }
        std::ifstream t(dir + "/tids.txt"); std::string l; while (std::getline(t, l)) tids.push_back(l);
        if (n_tid == 0 || genome_dir.empty() || tids.size() != n_tid) { if (err) *err = dir + "/meta.txt or tids.txt"; return false; }
        return true;
    }
};

struct Fnv {
    uint64_t h = 1469598103934665603ull;
    void add(const void* p, size_t n) { const uint8_t* b = static_cast<const uint8_t*>(p); for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; } }
    template <class T> void v(const std::vector<T>& x) { const uint64_t n = x.size(); add(&n, 8); if (n) add(x.data(), n * sizeof(T)); }
    template <class T> void s(T x) { add(&x, sizeof(T)); }
};

// digests of the sites, the statistics and the matrices; r must be finalized over every barcode (Result::finalize(nullptr))
inline std::string digest(const Result& r) {
    Fnv a, b, c;
    for (const auto& x : r.sites) { a.s(x.tid); a.s(x.pos); a.s(x.ref); a.s(x.alt); a.s(x.alt_molecules); a.s(x.alt_cells); }
    for (uint64_t x : {r.reads_used, r.mismatch_obs, r.molecules, r.ambiguous, r.other, r.dp_records, r.ad_records}) b.s(x);
    b.v(r.site_dp); b.v(r.site_ad);
    for (const auto* m : {&r.ad, &r.dp}) { c.s(m->nrows); c.s(m->ncols); c.v(m->indptr); c.v(m->indices); c.v(m->data); }
    char buf[96]; std::snprintf(buf, sizeof buf, "sites %016llx stats %016llx matrices %016llx", (unsigned long long) a.h, (unsigned long long) b.h, (unsigned long long) c.h);
    return buf;
}

}}}  // namespace singlet::allsite::replay
