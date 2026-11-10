#pragma once
// D-101: HLA typing per sample and per donor with T1K (Song et al., Genome Research 2023; MIT; github.com/mourisl/T1K).
//
// The reads are deleted after the campaign, so the run types HLA from its own input reads:
//   1. one pass over the input .1fq keeps every cDNA read that shares a 19-mer (either strand, sampled every 4 bases) with any
//      allele of the IPD-IMGT/HLA index T1K uses. T1K's genotyper (preset hla) counts a read only at >= 97% identity, so a counted
//      read of length L has at most floor(0.03 L) mismatches and an exact run of at least L (0.97) / (floor(0.03 L) + 1) bases,
//      which is >= 22 for every L from 44 to 300: a 19-mer sampled every 4 bases always falls inside it. The prefilter therefore
//      keeps every read T1K would count, and T1K's own extractor and genotyper decide from there (validated against T1K on the
//      full reads: docs/pipeline-outputs.md, bench/hla/).
//   2. run-t1k --preset hla on those reads gives the sample's genotype;
//   3. per donor (donor demux, >= 2 donors): run-t1k --noExtraction on the candidate reads of that donor's called cells
//      (cell_status cell, prob_max >= 0.9, the D-52 genotype rule), matched by the read's raw cell barcode.
// The pass reads only the input, so the CPU and GPU engines give the same HLA output by construction.
//
// Outputs, all owner-only (D-101: HLA types are genotypes): private/hla/
//   hla_genotype.tsv            T1K genotype table of the sample (gene, num_diff_alleles, allele_1, abundance_1, quality_1, allele_2, ...)
//   hla_allele.tsv, hla_allele.vcf   T1K's representative alleles and novel variants
//   donors/donor<k>_genotype.tsv     the same table per donor
//   hla_candidate_reads.fq.gz        the prefiltered reads (name: <index>_<raw cell barcode>), so HLA can be typed again later
//   hla_summary.json            T1K and IPD-IMGT/HLA manifest, read counts, calls per gene for the sample and each donor
//
// Deferred route (--hla-defer, then `singlet hla --pending`): the run does the prefilter only and leaves private/hla/pending/ (the candidate
// reads and manifest.json), so T1K can run later on a CPU node after the input reads are gone. run_pending types from there, writes the
// outputs above byte for byte as the direct route does, and deletes pending/ once hla_summary.json has landed. pending/ holds reads: it is
// owner-only, under private/, and never part of a public object.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <cerrno>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zlib.h>

#include "../fq/read_source.h"
#include "../fq/reader.h"

namespace singlet::hla {

constexpr int kK = 19;          // prefilter k-mer
constexpr int kStride = 4;      // sample every 4th k-mer of a read (a power of two: hits() masks with it)
constexpr double kDonorMinProb = 0.9;

inline bool file_exists(const std::string& p) { struct stat st{}; return stat(p.c_str(), &st) == 0; }

/// Canonical 19-mers (2-bit, A=0 C=1 G=2 T=3) of a set of sequences, in an open-addressing table.
class KmerSet {
public:
    void add_sequence(const std::string& s) {
        uint64_t f = 0, r = 0; int valid = 0;
        const uint64_t mask = (1ull << (2 * kK)) - 1;
        for (char ch : s) {
            int b = code(ch);
            if (b < 0) { valid = 0; f = r = 0; continue; }
            f = ((f << 2) | uint64_t(b)) & mask;
            r = (r >> 2) | (uint64_t(3 - b) << (2 * (kK - 1)));
            if (++valid >= kK) keys_.push_back(std::min(f, r));
        }
    }
    void build() {
        std::sort(keys_.begin(), keys_.end());
        keys_.erase(std::unique(keys_.begin(), keys_.end()), keys_.end());
        size_t cap = 1; while (cap < keys_.size() * 2 + 16) cap <<= 1;
        table_.assign(cap, 0); mask_ = cap - 1;
        bits_.assign(size_t(1) << (kBitsLog2 - 6), 0);
        for (uint64_t k : keys_) {
            const uint64_t hk = hash(k);
            size_t h = hk & mask_;
            while (table_[h] != 0) h = (h + 1) & mask_;
            table_[h] = k + 1;
            bits_[hk >> (64 - kBitsLog2 + 6)] |= uint64_t(1) << ((hk >> (64 - kBitsLog2)) & 63);
        }
        n_ = keys_.size();
        keys_.clear(); keys_.shrink_to_fit();
    }
    bool contains(uint64_t k) const {
        // the bitset (512 KB, in L2) answers most misses; the table decides the rest, so membership is exact
        const uint64_t hk = hash(k);
        if (!((bits_[hk >> (64 - kBitsLog2 + 6)] >> ((hk >> (64 - kBitsLog2)) & 63)) & 1)) return false;
        size_t h = hk & mask_;
        while (table_[h] != 0) { if (table_[h] == k + 1) return true; h = (h + 1) & mask_; }
        return false;
    }
    size_t size() const { return n_; }
    /// true when a 19-mer starting at a multiple of kStride (and the last one) of `seq` (codes 0..3, 4 = N) is in the set. One rolling
    /// pass over the read builds both strands' codes; a 19-mer that holds an N is skipped. Same k-mers as hits_ref, so the same answer.
    bool hits(const uint8_t* seq, int len) const {
        if (len < kK) return false;
        const int last = len - kK;
        constexpr uint64_t mask = (1ull << (2 * kK)) - 1;
        uint64_t f = 0, r = 0;
        int run = 0;
        for (int i = 0; i < len; ++i) {
            const uint8_t b = seq[i];
            if (b > 3) { run = 0; continue; }
            f = ((f << 2) | b) & mask;
            r = (r >> 2) | (uint64_t(3 - b) << (2 * (kK - 1)));
            if (++run >= kK) {
                const int p = i - (kK - 1);
                if (((p & (kStride - 1)) == 0 || p == last) && contains(std::min(f, r))) return true;
            }
        }
        return false;
    }
    /// The k-mers rebuilt one by one at each sampled start (the form before hits); kept as the reference of the tests and the bench.
    bool hits_ref(const uint8_t* seq, int len) const {
        if (len < kK) return false;
        const int last = len - kK;
        for (int p = 0;; p += kStride) {
            if (p > last) p = last;
            uint64_t f = 0, r = 0; bool ok = true;
            for (int i = 0; i < kK; ++i) {
                const uint8_t b = seq[p + i];
                if (b > 3) { ok = false; break; }
                f = (f << 2) | b;
                r |= uint64_t(3 - b) << (2 * i);
            }
            if (ok && contains(std::min(f, r))) return true;
            if (p == last) return false;
        }
    }
    static int code(char c) {
        switch (c) { case 'A': case 'a': return 0; case 'C': case 'c': return 1; case 'G': case 'g': return 2; case 'T': case 't': return 3; }
        return -1;
    }
private:
    static uint64_t hash(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
    static constexpr int kBitsLog2 = 22;   // 2^22-bit membership filter in front of the table (~5% of misses pass it)
    std::vector<uint64_t> keys_, table_, bits_;
    size_t mask_ = 0, n_ = 0;
};

inline bool load_fasta_kmers(const std::string& path, KmerSet& ks, size_t& n_seq) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line, seq; n_seq = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '>') { if (!seq.empty()) { ks.add_sequence(seq); ++n_seq; } seq.clear(); }
        else seq += line;
    }
    if (!seq.empty()) { ks.add_sequence(seq); ++n_seq; }
    ks.build();
    return n_seq > 0;
}

struct Config {
    std::string t1k_dir;            // bin/{run-t1k,fastq-extractor,genotyper,analyzer}, hlaidx/hlaidx_rna_seq.fa, manifest.json
    std::vector<std::string> onefq; // the run's input reads, one .1fq per run (read in order)
    bool cdna_is_r1 = false;        // reads_inverted: the barcode is in R2
    std::string out_prefix;         // the run directory; outputs go to <out_prefix>/private/hla/
    std::string donor_assignments;  // <out_prefix>/donor_assignments.tsv (empty or absent: no per-donor typing)
    int threads = 8;
    // where the T1K work files go; empty: <out_prefix>/private/hla/.t1k. The pipeline passes a directory outside the run directory, so
    // the run's files do not change while the sample stage runs beside STAR (sample_stage / finish_stage below).
    std::string tmp_dir;
    int min_cells = 2;              // frequency prior: type against alleles recorded in >= this many cells of IPD-IMGT/HLA (0: all alleles)
    bool defer = false;             // prefilter only: leave private/hla/pending/ for run_pending (--hla-defer)
};

struct Result {
    bool ran = false;
    std::string why_not;
    uint64_t reads_scanned = 0, reads_kept = 0;
    size_t n_kmers = 0, n_alleles = 0, n_alleles_typed = 0;
    int n_donors_typed = 0, n_donors_failed = 0;   // failed: a donor with candidate reads whose T1K call failed
    double seconds_prefilter = 0, seconds_t1k = 0;
    bool deferred = false;   // private/hla/pending/ was written (Config::defer)
    bool landed = false;     // run_pending: hla_summary.json is complete and pending/ is gone
};

/// One line for the run log.
inline std::string status_text(const Result& r) {
    if (r.deferred) return "deferred: candidate reads and manifest in private/hla/pending/";
    return r.ran ? std::string("typed") : "not typed: " + r.why_not;
}

inline std::string shq(const std::string& s) {   // single-quote for /bin/sh
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}

/// `sh -c cmd` with only stdin, stdout and stderr inherited; returns as std::system does (0 on success). The HLA sample stage runs on
/// a thread beside the STAR pipe and the read FIFO, so a child that kept their descriptors would hold back the EOF their readers wait
/// for. Only async-signal-safe calls run between fork and exec.
inline int run_sh(const std::string& cmd) {
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
#ifdef SYS_close_range
        if (syscall(SYS_close_range, 3u, ~0u, 0u) != 0)
#endif
        {
            struct rlimit rl{};
            const long top = getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY ? static_cast<long>(rl.rlim_cur) : 65536L;
            for (long fd = 3; fd < top; ++fd) close(static_cast<int>(fd));
        }
        execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0) if (errno != EINTR) return -1;
    return st;
}

inline std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

/// T1K genotype table -> JSON object {gene: {"alleles": [a1, a2], "quality": [q1, q2], "abundance": [x1, x2]}} (quality <= 0 kept;
/// T1K's README: ignore calls of quality <= 0, which the reader does with the quality field).
inline std::string genotype_json(const std::string& tsv_path) {
    std::ifstream f(tsv_path);
    std::string line, o = "{";
    bool first = true;
    while (std::getline(f, line)) {
        std::vector<std::string> v; std::stringstream ss(line); std::string x;
        while (std::getline(ss, x, '\t')) v.push_back(x);
        if (v.size() < 8) continue;
        auto q = [](const std::string& s) { std::string e; for (char c : s) { if (c == '"' || c == '\\') e += '\\'; e += c; } return "\"" + e + "\""; };
        auto num = [](const std::string& s) { char* end = nullptr; std::strtod(s.c_str(), &end); return (end && *end == 0 && !s.empty()) ? s : std::string("null"); };
        o += std::string(first ? "" : ", ") + q(v[0]) + ": {\"alleles\": [" + q(v[2]) + ", " + q(v[5]) + "], \"abundance\": [" + num(v[3]) + ", " +
             num(v[6]) + "], \"quality\": [" + num(v[4]) + ", " + num(v[7]) + "]}";
        first = false;
    }
    return o + "}";
}

/// raw cell barcode (no suffix) -> donor index, for called cells assigned to one donor with prob_max >= 0.9
inline std::unordered_map<std::string, int> read_donors(const std::string& path, int& n_donors) {
    std::unordered_map<std::string, int> m; n_donors = 0;
    std::ifstream f(path);
    std::string line;
    if (!std::getline(f, line)) return m;
    while (std::getline(f, line)) {
        std::vector<std::string> v; std::stringstream ss(line); std::string x;
        while (std::getline(ss, x, '\t')) v.push_back(x);
        if (v.size() < 7 || v[6] != "cell" || v[1].rfind("donor", 0) != 0) continue;
        if (std::atof(v[2].c_str()) < kDonorMinProb) continue;
        const int k = std::atoi(v[1].c_str() + 5);
        if (v[1] != "donor" + std::to_string(k)) continue;
        std::string bc = v[0];
        const auto dash = bc.find('-'); if (dash != std::string::npos) bc.resize(dash);
        m[bc] = k; n_donors = std::max(n_donors, k + 1);
    }
    return m;
}

/// The same prefilter on FASTQ(.gz) input (the bulk benchmark, singlet hla-prefilter): keeps a pair when either mate hits.
/// Writes <prefix>_1.fq (+ <prefix>_2.fq for pairs). Returns {pairs scanned, pairs kept}, or {0, 0} on an open failure.
inline std::pair<uint64_t, uint64_t> prefilter_fastq(const KmerSet& ks, const std::string& r1, const std::string& r2, const std::string& prefix) {
    gzFile g1 = gzopen(r1.c_str(), "rb"), g2 = r2.empty() ? nullptr : gzopen(r2.c_str(), "rb");
    if (!g1 || (!r2.empty() && !g2)) { if (g1) gzclose(g1); if (g2) gzclose(g2); return {0, 0}; }
    gzbuffer(g1, 1 << 20); if (g2) gzbuffer(g2, 1 << 20);
    FILE* o1 = std::fopen((prefix + "_1.fq").c_str(), "wb");
    FILE* o2 = g2 ? std::fopen((prefix + "_2.fq").c_str(), "wb") : nullptr;
    auto getl = [](gzFile g, std::string& s) {
        s.clear(); char b[4096];
        while (gzgets(g, b, sizeof b)) { s += b; if (!s.empty() && s.back() == '\n') { s.pop_back(); if (!s.empty() && s.back() == '\r') s.pop_back(); return true; } }
        return !s.empty();
    };
    auto rec = [&](gzFile g, std::string* r) { return getl(g, r[0]) && getl(g, r[1]) && getl(g, r[2]) && getl(g, r[3]); };
    std::string a[4], b[4];
    std::vector<uint8_t> c;
    auto hit = [&](const std::string& s) {
        c.resize(s.size());
        for (size_t i = 0; i < s.size(); ++i) { const int x = KmerSet::code(s[i]); c[i] = x < 0 ? 4 : uint8_t(x); }
        return ks.hits(c.data(), static_cast<int>(c.size()));
    };
    uint64_t n = 0, k = 0;
    while (rec(g1, a) && (!g2 || rec(g2, b))) {
        ++n;
        if (!(hit(a[1]) || (g2 && hit(b[1])))) continue;
        ++k;
        std::fprintf(o1, "%s\n%s\n+\n%s\n", a[0].c_str(), a[1].c_str(), a[3].c_str());
        if (o2) std::fprintf(o2, "%s\n%s\n+\n%s\n", b[0].c_str(), b[1].c_str(), b[3].c_str());
    }
    gzclose(g1); if (g2) gzclose(g2);
    std::fclose(o1); if (o2) std::fclose(o2);
    return {n, k};
}

// Frequency prior (D-101): 3' reads cannot separate alleles of one group, so T1K picks rare, mostly unconfirmed alleles over the common one.
// Keep the alleles that IPD-IMGT/HLA records in >= k cells (Allele_status.txt of the index's release); alleles missing from the file stay in.
// Returns the number kept, or 0 when the file is absent or nothing would be removed (the caller then uses the full index).
inline size_t filter_index_by_cells(const std::string& fa, const std::string& status, int k, const std::string& out, size_t& total) {
    std::ifstream sf(status), in(fa);
    total = 0;
    if (!sf || !in) return 0;
    std::unordered_map<std::string, int> cells;
    std::string line;
    while (std::getline(sf, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("Allele,", 0) == 0) continue;
        const size_t c1 = line.find(','); if (c1 == std::string::npos) continue;
        const size_t c2 = line.find(',', c1 + 1);
        const std::string n = line.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
        int v = 0; bool dig = !n.empty();
        for (char ch : n) { if (ch < '0' || ch > '9') { dig = false; break; } v = v * 10 + (ch - '0'); }
        cells[line.substr(0, c1)] = dig ? v : 0;
    }
    std::ofstream of(out);
    if (!of) return 0;
    bool keep = false; size_t kept = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') {
            ++total;
            std::string name = line.substr(1, line.find_first_of(" \t") == std::string::npos ? std::string::npos : line.find_first_of(" \t") - 1);
            if (name.rfind("HLA-", 0) == 0) name = name.substr(4);
            const auto it = cells.find(name);
            keep = it == cells.end() || it->second >= k;
            kept += keep;
        }
        if (keep) of << line << '\n';
    }
    return kept;
}

// The run in two stages, so the pipeline can type the sample beside STAR and the pileup (the sample needs only the input reads) and
// leave only the donors, which need the demux, for the end:
//   sample_stage: the prefilter and the sample's T1K, everything in Session::tmp (Config::tmp_dir: outside the run directory, so the
//                 run's files are the same while it runs; empty: <out>/private/hla/.t1k as before);
//   finish_stage: the donors' T1K and every output under private/hla/, as run() wrote them in one go.
// run() = both, one after the other. The outputs do not depend on when the sample stage ran.
struct Session {
    Config cfg;
    Result res;
    std::string dir, tmp, kept_fq, log, tref, T;
    bool made = false;         // the temporary directory exists (the input checks passed)
    bool stopped = false;      // a prefilter error after it was made: finish_stage returns res as it is
    double t1k_sample_s = 0;   // the sample's T1K (with the index prior), part of seconds_t1k
    // hla_candidate_reads.fq.gz (level 6, one deflate stream, so it cannot be split) is compressed on one thread beside T1K, which
    // leaves a core idle for most of its serial stretches; finish_stage moves the file in. A failed or absent compression is redone there.
    std::future<bool> gz;
    std::string gz_tmp;
    bool wait_gz() { return gz.valid() ? gz.get() : false; }
    // The donor stage (donor_stage): T1K on each donor's candidate reads, in tmp. It needs only the sample stage and the donor
    // assignments, so the pipeline starts it as soon as demux has written them; finish_stage runs it when nobody did and moves its
    // results in. donor_reads[k] counts donor k's candidate reads, donor_ok[k] says its T1K call succeeded.
    bool donors_run = false;
    int n_donors = 0;
    std::vector<uint64_t> donor_reads;
    std::vector<char> donor_ok;
    double donor_s = 0;        // wall seconds of the donor stage
};

/// Move a file, across file systems too (Config::tmp_dir may be on another one than the run directory).
inline void move_file(const std::string& from, const std::string& to) {
    if (std::rename(from.c_str(), to.c_str()) == 0) return;
    std::ifstream in(from, std::ios::binary);
    if (!in) return;
    { std::ofstream out(to, std::ios::binary | std::ios::trunc); out << in.rdbuf(); }
    std::remove(from.c_str());
}

inline bool t1k_call(const Session& s, const std::string& in, const std::string& prefix, bool no_extract, const std::string& threads = "") {
    const std::string runt1k = s.cfg.t1k_dir + "/bin/run-t1k";
    const std::string cmd = "perl " + shq(runt1k) + " -u " + shq(in) + " --preset hla -f " + shq(s.tref) + " -t " + (threads.empty() ? s.T : threads) + " -o " + shq(prefix) +
                            " --od " + shq(s.tmp) + (no_extract ? " --noExtraction" : "") + " >> " + shq(s.log) + " 2>&1";
    return run_sh(cmd.c_str()) == 0;
}

/// Prefilter of one .1fq into `out`: blocks decoded and scanned on `threads` threads (RunSet::decode is thread-safe), written in block
/// order, reads named by their index in the whole input (`next_index` continues across the runs). Returns false (why) on an open error.
inline bool prefilter_onefq(const KmerSet& ks, const std::string& input, bool cdna_is_r1, int threads, FILE* out, uint64_t& next_index,
                            Result& res, std::string& why) {
    singlet::fq::RunSet rs;
    try { rs.open({input}); } catch (const std::exception& e) { why = std::string("cannot open input: ") + e.what(); return false; }
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
    const uint32_t W = static_cast<uint32_t>(T) * 4;   // blocks per window: decoded and scanned in parallel, then written in order
    std::vector<std::string> buf(W);
    std::vector<uint64_t> nscan(W), nkept(W);
    const uint64_t file_base = next_index;
    for (uint32_t w0 = 0; w0 < nblk; w0 += W) {
        const uint32_t nb = std::min(W, nblk - w0);
        std::atomic<uint32_t> nx{0};
        auto work = [&]() {
            singlet::fq::DecodedBlock B;
            // the quality column stays packed where the format allows it (2 bits a base, BINNED4 and ILLUMINA4): only the few kept reads
            // need their Phred values, which are unpacked from it below exactly as the decoder unpacks them
            B.keep_packed_qual = true;
            std::vector<uint8_t> qbuf;
            for (uint32_t b; (b = nx++) < nb;) {
                B.clear(); rs.decode(w0 + b, B);   // a staged source does not clear the block itself
                const uint64_t base = file_base + rs.block_read_offset(w0 + b);
                std::string& o = buf[b]; o.clear(); nscan[b] = nkept[b] = 0;
                const bool packed = !B.r2_qual_packed.empty() && B.r2_qual_packed_phred;
                size_t qoff = 0;   // read i's packed quality starts here (reads in order, packed_bytes(len) each)
                for (uint32_t i = 0; i < B.n_reads; ++i) {
                    const uint32_t dup = B.dup_counts.empty() ? 1u : std::max<uint32_t>(1u, B.dup_counts[i]);
                    nscan[b] += dup;
                    const size_t qthis = qoff;
                    if (packed) qoff += singlet::fq::pack::packed_bytes(B.r2_len(i));
                    const uint8_t* s = cdna_is_r1 ? B.r1_seq(i) : B.r2_seq(i);
                    const int L = cdna_is_r1 ? B.r1_len(i) : B.r2_len(i);
                    if (!ks.hits(s, L)) continue;
                    std::string bc;
                    if (!dict.empty() && !B.bc_indices.empty() && B.bc_indices[i] < dict.size()) bc = dict[B.bc_indices[i]];
                    else {
                        const uint8_t* r = cdna_is_r1 ? B.r2_seq(i) : B.r1_seq(i);
                        const int rl = std::min<int>(bc_len ? bc_len : 16, cdna_is_r1 ? B.r2_len(i) : B.r1_len(i));
                        for (int j = 0; j < rl; ++j) bc += "ACGTN"[r[j] < 5 ? r[j] : 4];
                    }
                    const uint8_t* q = nullptr;
                    if (!cdna_is_r1 && packed) {
                        const int n = B.r2_len(i);
                        if (qthis + singlet::fq::pack::packed_bytes(n) <= B.r2_qual_packed.size()) {
                            qbuf.resize(n);
                            const uint8_t* src = B.r2_qual_packed.data() + qthis;
                            for (int j = 0; j < n; ++j) qbuf[j] = B.r2_qual_packed_phred[(src[j >> 2] >> (6 - 2 * (j & 3))) & 3];
                            q = qbuf.data();
                        }
                    } else if (!cdna_is_r1) q = B.r2_quality(i);
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
        for (uint32_t b = 0; b < nb; ++b) { std::fwrite(buf[b].data(), 1, buf[b].size(), out); res.reads_scanned += nscan[b]; res.reads_kept += nkept[b]; }
    }
    next_index = file_base + rs.stored_reads();
    return true;
}

constexpr const char* kPendingReads = "candidate_reads.fq.gz";
constexpr const char* kPendingFormat = "singlet-hla-pending-1";

/// gzip a file (mode: gzopen's, e.g. "wb1"); false when anything failed.
inline bool gzip_file(const std::string& from, const std::string& to, const char* mode) {
    std::ifstream in(from, std::ios::binary);
    gzFile g = gzopen(to.c_str(), mode);
    if (!in || !g) { if (g) gzclose(g); return false; }
    std::vector<char> b(1 << 20);
    bool ok = true;
    while (ok && in) { in.read(b.data(), b.size()); const int n = static_cast<int>(in.gcount()); if (n > 0 && gzwrite(g, b.data(), static_cast<unsigned>(n)) != n) ok = false; }
    return gzclose(g) == Z_OK && ok && !in.bad();
}

/// The sample's T1K on the prefiltered reads (Session::kept_fq), with the frequency prior.
inline void type_sample(Session& s) {
    const Config& cfg = s.cfg;
    Result& res = s.res;
    const std::string ref = cfg.t1k_dir + "/hlaidx/hlaidx_rna_seq.fa";
    auto t1 = std::chrono::steady_clock::now();
    // the sample
    s.T = std::to_string(std::max(1, cfg.threads));
    s.log = s.tmp + "/t1k.log";
    // typing index: the K-cell frequency prior when the index ships Allele_status.txt, else the full index (reported either way)
    s.tref = ref;
    res.n_alleles_typed = res.n_alleles;
    const std::string status = cfg.t1k_dir + "/hlaidx/Allele_status.txt";
    if (cfg.min_cells > 0 && file_exists(status)) {
        size_t total = 0;
        const size_t kept = filter_index_by_cells(ref, status, cfg.min_cells, s.tmp + "/hlaidx.prior.fa", total);
        if (kept > 0) { s.tref = s.tmp + "/hlaidx.prior.fa"; res.n_alleles_typed = kept; }
        std::cerr << "[hla] frequency prior (>= " << cfg.min_cells << " cells): " << (kept > 0 ? std::to_string(kept) + " of " + std::to_string(total) + " alleles" : std::string("not applied")) << "\n";
    } else if (cfg.min_cells > 0) {
        std::cerr << "[hla] frequency prior not applied: no Allele_status.txt in " << cfg.t1k_dir << "/hlaidx\n";
    }
    if (res.reads_kept > 0 && !s.gz.valid()) {
        s.gz_tmp = s.tmp + "/hla_candidate_reads.fq.gz";
        const std::string from = s.kept_fq, to = s.gz_tmp;
        s.gz = std::async(std::launch::async, [from, to] { return gzip_file(from, to, "wb6"); });
    }
    if (res.reads_kept == 0) { res.why_not = "no read shares a k-mer with the HLA index"; }
    else if (!t1k_call(s, s.kept_fq, "sample", false)) { res.why_not = "run-t1k failed (" + s.log + ")"; }
    else res.ran = true;
    s.t1k_sample_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

inline void sample_stage(Session& s) {
    const Config& cfg = s.cfg;
    Result& res = s.res;
    const std::string ref = cfg.t1k_dir + "/hlaidx/hlaidx_rna_seq.fa", runt1k = cfg.t1k_dir + "/bin/run-t1k";
    s.dir = cfg.out_prefix + "/private/hla";
    s.tmp = cfg.tmp_dir.empty() ? s.dir + "/.t1k" : cfg.tmp_dir;
    if (!file_exists(ref) || !file_exists(runt1k)) { res.why_not = "T1K or its HLA index not found in " + cfg.t1k_dir; return; }
    if (cfg.onefq.empty()) { res.why_not = "no .1fq input"; return; }
    for (const auto& f : cfg.onefq) if (!file_exists(f)) { res.why_not = "missing input " + f; return; }
    run_sh(("mkdir -p " + shq(s.tmp)).c_str());
    s.made = true;
    s.stopped = true;   // until the prefilter completes

    // 1. prefilter
    auto t0 = std::chrono::steady_clock::now();
    KmerSet ks;
    if (!load_fasta_kmers(ref, ks, res.n_alleles)) { res.why_not = "cannot read " + ref; return; }
    res.n_kmers = ks.size();
    s.kept_fq = s.tmp + "/prefilter.fq";
    FILE* out = std::fopen(s.kept_fq.c_str(), "wb");
    if (!out) { res.why_not = "cannot write " + s.kept_fq; return; }
    uint64_t next_index = 0;
    for (const auto& input : cfg.onefq) {
        std::string why;
        if (!prefilter_onefq(ks, input, cfg.cdna_is_r1, cfg.threads, out, next_index, res, why)) { std::fclose(out); res.why_not = why; return; }
    }
    std::fclose(out);
    s.stopped = false;
    res.seconds_prefilter = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "[hla] prefilter: " << res.reads_kept << " of " << res.reads_scanned << " reads share a 19-mer with " << res.n_alleles
              << " HLA alleles (" << res.n_kmers << " k-mers), " << res.seconds_prefilter << " s\n";

    if (cfg.defer) {
        if (res.reads_kept > 0 && !gzip_file(s.kept_fq, s.tmp + "/" + kPendingReads, "wb1")) {
            s.stopped = true; res.why_not = "cannot compress the candidate reads in " + s.tmp; return;
        }
        std::remove(s.kept_fq.c_str());
        return;
    }
    type_sample(s);
}

inline std::string json_quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

/// A field of the manifest (one key per line); false when absent. String values are unquoted.
inline bool manifest_field(const std::string& text, const std::string& key, std::string& value) {
    const std::string k = "\n  \"" + key + "\": ";
    auto p = text.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    auto e = text.find('\n', p);
    std::string v = text.substr(p, e == std::string::npos ? std::string::npos : e - p);
    if (!v.empty() && v.back() == ',') v.pop_back();
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
        value.clear();
        for (size_t i = 1; i + 1 < v.size(); ++i) { if (v[i] == '\\' && i + 2 < v.size()) ++i; value += v[i]; }
    } else value = v;
    return true;
}

/// Deferred route, end of the run: the candidate reads and the manifest into private/hla/pending/. The manifest is written last, after
/// the reads are in place, so a pending/ with a manifest is complete.
inline Result write_pending(Session& s) {
    const Config& cfg = s.cfg;
    Result& res = s.res;
    const std::string pend = s.dir + "/pending", reads = pend + "/" + kPendingReads, staged = s.tmp + "/" + kPendingReads;
    run_sh(("rm -rf " + shq(pend) + " && mkdir -p " + shq(pend)).c_str());
    ::chmod((cfg.out_prefix + "/private").c_str(), 0700);
    auto fail = [&](const std::string& why) -> Result& { res.why_not = why; run_sh(("rm -rf " + shq(pend)).c_str()); if (!std::getenv("SINGLET_HLA_KEEP_TMP")) run_sh(("rm -rf " + shq(s.tmp)).c_str()); return res; };
    long long bytes = 0;
    if (res.reads_kept > 0) {
        struct stat a{}, b{};
        if (stat(staged.c_str(), &a) != 0) return fail("no candidate reads at " + staged);
        move_file(staged, reads);
        if (stat(reads.c_str(), &b) != 0 || a.st_size != b.st_size) return fail("cannot write " + reads);
        bytes = static_cast<long long>(b.st_size);
    }
    // the donor file relative to the run directory, which may move before the CPU job runs
    std::string donors = cfg.donor_assignments;
    const std::string pre = cfg.out_prefix + "/";
    if (donors.rfind(pre, 0) == 0) donors = donors.substr(pre.size());
    {
        std::ofstream j(pend + "/manifest.json.tmp");
        j << "{\n  \"format\": " << json_quote(kPendingFormat) << ",\n  \"t1k_dir\": " << json_quote(cfg.t1k_dir) << ",\n  \"cdna_is_r1\": " << (cfg.cdna_is_r1 ? "true" : "false")
          << ",\n  \"donor_assignments\": " << json_quote(donors) << ",\n  \"min_cells\": " << cfg.min_cells << ",\n  \"reads_scanned\": " << res.reads_scanned
          << ",\n  \"reads_kept\": " << res.reads_kept << ",\n  \"n_kmers\": " << res.n_kmers << ",\n  \"n_alleles\": " << res.n_alleles
          << ",\n  \"seconds_prefilter\": " << res.seconds_prefilter << ",\n  \"candidate_bytes\": " << bytes << "\n}\n";
        j.close();
        if (!j || std::rename((pend + "/manifest.json.tmp").c_str(), (pend + "/manifest.json").c_str()) != 0) return fail("cannot write the manifest in " + pend);
    }
    if (!std::getenv("SINGLET_HLA_KEEP_TMP")) run_sh(("rm -rf " + shq(s.tmp)).c_str());
    run_sh(("chmod -R go-rwx " + shq(s.dir)).c_str());
    res.deferred = true;
    return res;
}

/// The donor stage: the sample's candidate reads (T1K's sample_candidate.fq) split by donor (called cells of one donor with prob_max
/// >= 0.9), then T1K --noExtraction on each donor's reads, all in Session::tmp; finish_stage moves the genotype tables in. Runs once.
/// Several donors are typed at a time, each with a share of the threads; every T1K call gets the same reads and index as one call per
/// donor in turn, and T1K's results do not depend on its thread count. SINGLET_HLA_DONOR_JOBS=N caps the concurrent calls (1: in turn).
/// Writes nothing into the run directory, so it may run while export is still writing there.
inline void donor_stage(Session& s) {
    if (s.donors_run) return;
    s.donors_run = true;
    const Config& cfg = s.cfg;
    const Result& res = s.res;
    if (!s.made || s.stopped || !res.ran) return;
    if (cfg.donor_assignments.empty() || !file_exists(cfg.donor_assignments)) return;
    const auto t0 = std::chrono::steady_clock::now();
    const std::string& tmp = s.tmp;
    int n_donors = 0;
    const std::unordered_map<std::string, int> bc2d = read_donors(cfg.donor_assignments, n_donors);
    if (n_donors < 2) return;
    s.n_donors = n_donors;
    s.donor_reads.assign(n_donors, 0);
    s.donor_ok.assign(n_donors, 0);
    {
        std::vector<FILE*> fo(n_donors, nullptr);
        for (int k = 0; k < n_donors; ++k) fo[k] = std::fopen((tmp + "/donor" + std::to_string(k) + ".fq").c_str(), "wb");
        std::ifstream cf(tmp + "/sample_candidate.fq");
        std::string h, sq, p, q;
        while (std::getline(cf, h) && std::getline(cf, sq) && std::getline(cf, p) && std::getline(cf, q)) {
            std::string bc = h.substr(h.find('_') == std::string::npos ? h.size() : h.find('_') + 1);
            const auto sp = bc.find_first_of(" \t"); if (sp != std::string::npos) bc.resize(sp);
            auto it = bc2d.find(bc);
            if (it == bc2d.end() || !fo[it->second]) continue;
            std::fprintf(fo[it->second], "%s\n%s\n%s\n%s\n", h.c_str(), sq.c_str(), p.c_str(), q.c_str());
            ++s.donor_reads[it->second];
        }
        for (auto* f : fo) if (f) std::fclose(f);
    }
    // the donors with reads, largest first; P calls at a time with T / P threads each (at least 4 threads per call)
    std::vector<int> order;
    for (int k = 0; k < n_donors; ++k) if (s.donor_reads[k] > 0) order.push_back(k);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s.donor_reads[a] > s.donor_reads[b]; });
    const int T = std::max(1, cfg.threads);
    int P = std::max(1, std::min<int>(static_cast<int>(order.size()), T / 4));
    if (const char* e = std::getenv("SINGLET_HLA_DONOR_JOBS")) { const int j = std::atoi(e); if (j > 0) P = std::min(P, j); }
    const std::string per = P == 1 ? s.T : std::to_string(std::max(1, T / P));
    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (size_t i; (i = next++) < order.size();) {
            const int k = order[i];
            const std::string name = "donor" + std::to_string(k);
            s.donor_ok[k] = t1k_call(s, tmp + "/" + name + ".fq", name, true, per) ? 1 : 0;
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < P; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    s.donor_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "[hla] donor stage: " << order.size() << " of " << n_donors << " donors with candidate reads, " << P << " T1K calls at a time ("
              << per << " threads each), " << s.donor_s << " s\n";
}

inline Result finish_stage(Session& s) {
    const Config& cfg = s.cfg;
    Result& res = s.res;
    if (!s.made) return res;   // stopped at the input checks, before any directory was made
    if (cfg.defer && !s.stopped) return write_pending(s);
    const std::string& dir = s.dir;
    const std::string& tmp = s.tmp;
    run_sh(("mkdir -p " + shq(dir + "/donors")).c_str());
    ::chmod((cfg.out_prefix + "/private").c_str(), 0700);
    if (s.stopped) {
        if (!cfg.tmp_dir.empty() && !std::getenv("SINGLET_HLA_KEEP_TMP")) run_sh(("rm -rf " + shq(tmp)).c_str());
        return res;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (res.ran) {
        move_file(tmp + "/sample_genotype.tsv", dir + "/hla_genotype.tsv");
        move_file(tmp + "/sample_allele.tsv", dir + "/hla_allele.tsv");
        move_file(tmp + "/sample_allele.vcf", dir + "/hla_allele.vcf");
    }

    // 3. donors: candidate reads of each donor's cells (typed by donor_stage, here unless the pipeline started it after demux)
    donor_stage(s);
    std::string donors_json = "{}";
    if (s.n_donors >= 2) {
        donors_json = "{";
        for (int k = 0; k < s.n_donors; ++k) {
            const std::string name = "donor" + std::to_string(k);
            const uint64_t nr = s.donor_reads[k];
            std::string calls = "null";
            if (nr > 0 && s.donor_ok[k]) {
                move_file(tmp + "/" + name + "_genotype.tsv", dir + "/donors/" + name + "_genotype.tsv");
                calls = genotype_json(dir + "/donors/" + name + "_genotype.tsv");
                ++res.n_donors_typed;
            } else if (nr > 0) {
                ++res.n_donors_failed;
                std::cerr << "[hla] T1K failed for " << name << "; end of its log:\n" << std::flush;
                run_sh(("tail -n 20 " + shq(s.log) + " 1>&2").c_str());
            }
            donors_json += std::string(k ? ", " : "") + "\"" + name + "\": {\"candidate_reads\": " + std::to_string(nr) + ", \"calls\": " + calls + "}";
        }
        donors_json += "}";
    }
    auto t2 = std::chrono::steady_clock::now();
    res.seconds_t1k = s.t1k_sample_s + std::chrono::duration<double>(t2 - t1).count();

    // keep the prefiltered reads (owner-only) so HLA can be typed again once the reads are gone
    if (res.reads_kept > 0) {
        const std::string gzout = dir + "/hla_candidate_reads.fq.gz";
        if (s.gz.valid() && s.wait_gz()) move_file(s.gz_tmp, gzout);
        else gzip_file(s.kept_fq, gzout, "wb6");
    }
    std::string manifest = slurp(cfg.t1k_dir + "/manifest.json");
    while (!manifest.empty() && (manifest.back() == '\n' || manifest.back() == ' ')) manifest.pop_back();
    if (manifest.empty()) manifest = "null";
    {
        std::ofstream j(dir + "/hla_summary.json.tmp");   // renamed once complete: a hla_summary.json is always whole
        j << "{\n  \"t1k\": " << manifest << ",\n  \"ran\": " << (res.ran ? "true" : "false") << ",\n  \"why_not\": \"" << res.why_not
          << "\",\n  \"prefilter\": {\"k\": " << kK << ", \"stride\": " << kStride << ", \"hla_alleles\": " << res.n_alleles << ", \"typed_alleles\": " << res.n_alleles_typed << ", \"min_cells\": " << cfg.min_cells << ", \"kmers\": " << res.n_kmers
          << ", \"reads_scanned\": " << res.reads_scanned << ", \"reads_kept\": " << res.reads_kept
          << "},\n  \"run\": {\"prefilter_seconds\": " << res.seconds_prefilter << ", \"t1k_seconds\": " << res.seconds_t1k << "},\n  \"sample\": " << (res.ran ? genotype_json(dir + "/hla_genotype.tsv") : std::string("null"))
          << ",\n  \"donor_rule\": \"called cells (cell_status cell) of one donor with prob_max >= 0.9, matched by raw cell barcode\""
          << ",\n  \"donors\": " << donors_json << "\n}\n";
    }
    std::rename((dir + "/hla_summary.json.tmp").c_str(), (dir + "/hla_summary.json").c_str());
    if (!std::getenv("SINGLET_HLA_KEEP_TMP")) run_sh(("rm -rf " + shq(tmp)).c_str());
    run_sh(("chmod -R go-rwx " + shq(dir)).c_str());
    return res;
}

inline Result run(const Config& cfg) {
    Session s;
    s.cfg = cfg;
    sample_stage(s);
    return finish_stage(s);
}

/// Deferred route, the CPU job: `pending_dir` is <run>/private/hla/pending. t1k_dir overrides the manifest's install when not empty; tmp_dir
/// as in Config (empty: <run>/private/hla/.t1k). Result::landed is true when hla_summary.json is complete and pending/ has been deleted (a
/// sample with no candidate read lands with ran = false, as the direct route writes it). On any failure pending/ stays, so the job can run
/// again, and no hla_summary.json is left.
inline Result run_pending(const std::string& pending_dir, const std::string& t1k_dir, int threads, const std::string& tmp_dir) {
    Result bad;
    std::string pend = pending_dir;
    while (pend.size() > 1 && pend.back() == '/') pend.pop_back();
    const std::string suffix = "/private/hla/pending";
    if (pend.size() <= suffix.size() || pend.compare(pend.size() - suffix.size(), suffix.size(), suffix) != 0) { bad.why_not = "not a <run>/private/hla/pending directory: " + pend; return bad; }
    const std::string man = slurp(pend + "/manifest.json");
    std::string v;
    if (man.empty()) { bad.why_not = "no manifest in " + pend; return bad; }
    if (!manifest_field(man, "format", v) || v != kPendingFormat) { bad.why_not = "unknown pending format in " + pend; return bad; }
    auto num = [&](const char* k, unsigned long long& out) { std::string x; if (!manifest_field(man, k, x)) return false; out = std::strtoull(x.c_str(), nullptr, 10); return true; };

    Session s;
    Config& cfg = s.cfg;
    Result& res = s.res;
    cfg.out_prefix = pend.substr(0, pend.size() - suffix.size());
    cfg.threads = threads;
    cfg.tmp_dir = tmp_dir;
    cfg.t1k_dir = t1k_dir;
    if (cfg.t1k_dir.empty() && !manifest_field(man, "t1k_dir", cfg.t1k_dir)) { bad.why_not = "the manifest names no T1K install"; return bad; }
    if (manifest_field(man, "cdna_is_r1", v)) cfg.cdna_is_r1 = v == "true";
    if (manifest_field(man, "donor_assignments", v) && !v.empty()) cfg.donor_assignments = v[0] == '/' ? v : cfg.out_prefix + "/" + v;
    unsigned long long mc = 0, scanned = 0, kept = 0, nk = 0, na = 0;
    if (!num("min_cells", mc) || !num("reads_scanned", scanned) || !num("reads_kept", kept) || !num("n_kmers", nk) || !num("n_alleles", na)) { bad.why_not = "incomplete manifest in " + pend; return bad; }
    cfg.min_cells = static_cast<int>(mc);
    res.reads_scanned = scanned; res.reads_kept = kept; res.n_kmers = nk; res.n_alleles = na;
    if (manifest_field(man, "seconds_prefilter", v)) res.seconds_prefilter = std::atof(v.c_str());

    s.dir = cfg.out_prefix + "/private/hla";
    s.tmp = tmp_dir.empty() ? s.dir + "/.t1k" : tmp_dir;
    if (!file_exists(cfg.t1k_dir + "/hlaidx/hlaidx_rna_seq.fa") || !file_exists(cfg.t1k_dir + "/bin/run-t1k")) { bad.why_not = "T1K or its HLA index not found in " + cfg.t1k_dir; return bad; }
    run_sh(("mkdir -p " + shq(s.tmp)).c_str());
    auto clean = [&]() { if (s.gz.valid()) s.gz.wait(); if (!std::getenv("SINGLET_HLA_KEEP_TMP")) run_sh(("rm -rf " + shq(s.tmp)).c_str()); };
    s.kept_fq = s.tmp + "/prefilter.fq";
    if (kept > 0) {   // the reads as the prefilter wrote them, checked against the manifest
        gzFile g = gzopen((pend + "/" + kPendingReads).c_str(), "rb");
        FILE* out = g ? std::fopen(s.kept_fq.c_str(), "wb") : nullptr;
        uint64_t lines = 0;
        bool ok = g && out;
        std::vector<char> b(1 << 20);
        for (int n; ok && (n = gzread(g, b.data(), static_cast<unsigned>(b.size()))) != 0;) {
            if (n < 0 || std::fwrite(b.data(), 1, static_cast<size_t>(n), out) != static_cast<size_t>(n)) { ok = false; break; }
            lines += static_cast<uint64_t>(std::count(b.data(), b.data() + n, '\n'));
        }
        if (g) gzclose(g);
        if (out && std::fclose(out) != 0) ok = false;
        if (!ok || lines != 4 * kept) { bad.why_not = "the candidate reads in " + pend + " are damaged (" + std::to_string(lines / 4) + " reads, the manifest says " + std::to_string(kept) + ")"; clean(); return bad; }
    }
    s.made = true;
    type_sample(s);
    if (kept > 0 && !res.ran) {
        bad.why_not = res.why_not;
        std::cerr << "[hla] end of T1K's log:\n" << std::flush;
        run_sh(("tail -n 20 " + shq(s.log) + " 1>&2").c_str());
        clean();
        return bad;
    }
    Result out = finish_stage(s);
    const std::string summary = s.dir + "/hla_summary.json";
    if (out.n_donors_failed > 0) {
        std::remove(summary.c_str());
        out.ran = false;
        out.why_not = std::to_string(out.n_donors_failed) + " donor(s) failed in T1K";
        return out;
    }
    if (!file_exists(summary)) { out.why_not = "hla_summary.json was not written"; return out; }
    run_sh(("rm -rf " + shq(pend)).c_str());
    out.landed = !file_exists(pend);
    if (!out.landed) out.why_not = "cannot delete " + pend;
    return out;
}

}  // namespace singlet::hla
