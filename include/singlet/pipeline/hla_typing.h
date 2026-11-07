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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zlib.h>

#include "../fq/reader.h"

namespace singlet::hla {

constexpr int kK = 19;          // prefilter k-mer
constexpr int kStride = 4;      // sample every 4th k-mer of a read
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
        for (uint64_t k : keys_) {
            size_t h = hash(k) & mask_;
            while (table_[h] != 0) h = (h + 1) & mask_;
            table_[h] = k + 1;
        }
        n_ = keys_.size();
        keys_.clear(); keys_.shrink_to_fit();
    }
    bool contains(uint64_t k) const {
        size_t h = hash(k) & mask_;
        while (table_[h] != 0) { if (table_[h] == k + 1) return true; h = (h + 1) & mask_; }
        return false;
    }
    size_t size() const { return n_; }
    /// true when a 19-mer starting at a multiple of kStride (and the last one) of `seq` (codes 0..3, 4 = N) is in the set
    bool hits(const uint8_t* seq, int len) const {
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
    std::vector<uint64_t> keys_, table_;
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
};

struct Result {
    bool ran = false;
    std::string why_not;
    uint64_t reads_scanned = 0, reads_kept = 0;
    size_t n_kmers = 0, n_alleles = 0;
    int n_donors_typed = 0;
    double seconds_prefilter = 0, seconds_t1k = 0;
};

inline std::string shq(const std::string& s) {   // single-quote for /bin/sh
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
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

inline Result run(const Config& cfg) {
    Result res;
    const std::string ref = cfg.t1k_dir + "/hlaidx/hlaidx_rna_seq.fa", runt1k = cfg.t1k_dir + "/bin/run-t1k";
    if (!file_exists(ref) || !file_exists(runt1k)) { res.why_not = "T1K or its HLA index not found in " + cfg.t1k_dir; return res; }
    if (cfg.onefq.empty()) { res.why_not = "no .1fq input"; return res; }
    for (const auto& f : cfg.onefq) if (!file_exists(f)) { res.why_not = "missing input " + f; return res; }
    const std::string dir = cfg.out_prefix + "/private/hla", tmp = dir + "/.t1k";
    std::system(("mkdir -p " + shq(tmp) + " " + shq(dir + "/donors")).c_str());
    ::chmod((cfg.out_prefix + "/private").c_str(), 0700);

    // 1. prefilter
    auto t0 = std::chrono::steady_clock::now();
    KmerSet ks;
    if (!load_fasta_kmers(ref, ks, res.n_alleles)) { res.why_not = "cannot read " + ref; return res; }
    res.n_kmers = ks.size();
    const std::string kept_fq = tmp + "/prefilter.fq";
    FILE* out = std::fopen(kept_fq.c_str(), "wb");
    if (!out) { res.why_not = "cannot write " + kept_fq; return res; }
    uint64_t next_index = 0;
    for (const auto& input : cfg.onefq) {
        singlet::fq::Reader rd;
        try { rd.open(input.c_str()); } catch (const std::exception& e) { std::fclose(out); res.why_not = std::string("cannot open input: ") + e.what(); return res; }
        const uint16_t bc_len = rd.bc_length();
        std::vector<std::string> dict;
        if (rd.has_bc_dict()) {
            const auto& flat = rd.bc_dict_flat();
            dict.resize(rd.bc_dict_size());
            for (size_t d = 0; d < dict.size(); ++d) {
                dict[d].resize(bc_len);
                for (uint16_t j = 0; j < bc_len; ++j) { const uint8_t b = flat[d * bc_len + j]; dict[d][j] = "ACGTN"[b < 5 ? b : 4]; }
            }
        }
        const int T = std::max(1, cfg.threads);
        std::vector<singlet::fq::DecodedBlock> blk(static_cast<size_t>(T) * 2);
        std::vector<std::string> buf(blk.size());
        std::vector<uint64_t> base(blk.size()), nscan(blk.size()), nkept(blk.size());
        bool more = true;
        while (more) {
            size_t nb = 0;
            for (; nb < blk.size(); ++nb) {
                if (!rd.read_block(blk[nb])) { more = false; break; }
                base[nb] = next_index; next_index += blk[nb].n_reads;
            }
            if (nb == 0) break;
            std::atomic<size_t> nx{0};
            auto work = [&]() {
                for (size_t b; (b = nx++) < nb;) {
                    const auto& B = blk[b]; std::string& o = buf[b]; o.clear(); nscan[b] = nkept[b] = 0;
                    for (uint32_t i = 0; i < B.n_reads; ++i) {
                        const uint32_t dup = B.dup_counts.empty() ? 1u : std::max<uint32_t>(1u, B.dup_counts[i]);
                        nscan[b] += dup;
                        const uint8_t* s = cfg.cdna_is_r1 ? B.r1_seq(i) : B.r2_seq(i);
                        const int L = cfg.cdna_is_r1 ? B.r1_len(i) : B.r2_len(i);
                        if (!ks.hits(s, L)) continue;
                        std::string bc;
                        if (!dict.empty() && !B.bc_indices.empty() && B.bc_indices[i] < dict.size()) bc = dict[B.bc_indices[i]];
                        else {
                            const uint8_t* r = cfg.cdna_is_r1 ? B.r2_seq(i) : B.r1_seq(i);
                            const int rl = std::min<int>(bc_len ? bc_len : 16, cfg.cdna_is_r1 ? B.r2_len(i) : B.r1_len(i));
                            for (int j = 0; j < rl; ++j) bc += "ACGTN"[r[j] < 5 ? r[j] : 4];
                        }
                        const uint8_t* q = cfg.cdna_is_r1 ? nullptr : B.r2_quality(i);
                        std::string seq(L, 'N'), qual(L, 'I');
                        for (int j = 0; j < L; ++j) { seq[j] = "ACGTN"[s[j] < 5 ? s[j] : 4]; if (q) qual[j] = static_cast<char>(33 + std::min<int>(q[j], 93)); }
                        for (uint32_t d = 0; d < dup; ++d) {
                            o += "@" + std::to_string(base[b] + i) + (dup > 1 ? "." + std::to_string(d) : std::string()) + "_" + bc + "\n" + seq + "\n+\n" + qual + "\n";
                            ++nkept[b];
                        }
                    }
                }
            };
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) th.emplace_back(work);
            for (auto& t : th) t.join();
            for (size_t b = 0; b < nb; ++b) { std::fwrite(buf[b].data(), 1, buf[b].size(), out); res.reads_scanned += nscan[b]; res.reads_kept += nkept[b]; }
        }
    }
    std::fclose(out);
    auto t1 = std::chrono::steady_clock::now();
    res.seconds_prefilter = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[hla] prefilter: " << res.reads_kept << " of " << res.reads_scanned << " reads share a 19-mer with " << res.n_alleles
              << " HLA alleles (" << res.n_kmers << " k-mers), " << res.seconds_prefilter << " s\n";

    // 2. the sample
    const std::string T = std::to_string(std::max(1, cfg.threads));
    const std::string log = tmp + "/t1k.log";
    auto t1k = [&](const std::string& in, const std::string& prefix, bool no_extract) {
        const std::string cmd = "perl " + shq(runt1k) + " -u " + shq(in) + " --preset hla -f " + shq(ref) + " -t " + T + " -o " + shq(prefix) +
                                " --od " + shq(tmp) + (no_extract ? " --noExtraction" : "") + " >> " + shq(log) + " 2>&1";
        return std::system(cmd.c_str()) == 0;
    };
    if (res.reads_kept == 0) { res.why_not = "no read shares a k-mer with the HLA index"; }
    else if (!t1k(kept_fq, "sample", false)) { res.why_not = "run-t1k failed (" + log + ")"; }
    else {
        res.ran = true;
        std::rename((tmp + "/sample_genotype.tsv").c_str(), (dir + "/hla_genotype.tsv").c_str());
        std::rename((tmp + "/sample_allele.tsv").c_str(), (dir + "/hla_allele.tsv").c_str());
        std::rename((tmp + "/sample_allele.vcf").c_str(), (dir + "/hla_allele.vcf").c_str());
    }

    // 3. donors: candidate reads of each donor's cells
    std::string donors_json = "{}";
    int n_donors = 0;
    std::unordered_map<std::string, int> bc2d;
    if (res.ran && !cfg.donor_assignments.empty() && file_exists(cfg.donor_assignments)) bc2d = read_donors(cfg.donor_assignments, n_donors);
    if (res.ran && n_donors >= 2) {
        std::vector<FILE*> fo(n_donors, nullptr);
        std::vector<uint64_t> nr(n_donors, 0);
        for (int k = 0; k < n_donors; ++k) fo[k] = std::fopen((tmp + "/donor" + std::to_string(k) + ".fq").c_str(), "wb");
        std::ifstream cf(tmp + "/sample_candidate.fq");
        std::string h, s, p, q;
        while (std::getline(cf, h) && std::getline(cf, s) && std::getline(cf, p) && std::getline(cf, q)) {
            std::string bc = h.substr(h.find('_') == std::string::npos ? h.size() : h.find('_') + 1);
            const auto sp = bc.find_first_of(" \t"); if (sp != std::string::npos) bc.resize(sp);
            auto it = bc2d.find(bc);
            if (it == bc2d.end() || !fo[it->second]) continue;
            std::fprintf(fo[it->second], "%s\n%s\n%s\n%s\n", h.c_str(), s.c_str(), p.c_str(), q.c_str());
            ++nr[it->second];
        }
        for (auto* f : fo) if (f) std::fclose(f);
        donors_json = "{";
        for (int k = 0; k < n_donors; ++k) {
            const std::string name = "donor" + std::to_string(k);
            std::string calls = "null";
            if (nr[k] > 0 && t1k(tmp + "/" + name + ".fq", name, true)) {
                std::rename((tmp + "/" + name + "_genotype.tsv").c_str(), (dir + "/donors/" + name + "_genotype.tsv").c_str());
                calls = genotype_json(dir + "/donors/" + name + "_genotype.tsv");
                ++res.n_donors_typed;
            }
            donors_json += std::string(k ? ", " : "") + "\"" + name + "\": {\"candidate_reads\": " + std::to_string(nr[k]) + ", \"calls\": " + calls + "}";
        }
        donors_json += "}";
    }
    auto t2 = std::chrono::steady_clock::now();
    res.seconds_t1k = std::chrono::duration<double>(t2 - t1).count();

    // keep the prefiltered reads (owner-only) so HLA can be typed again once the reads are gone
    if (res.reads_kept > 0) {
        gzFile g = gzopen((dir + "/hla_candidate_reads.fq.gz").c_str(), "wb6");
        std::ifstream kf(kept_fq, std::ios::binary);
        std::vector<char> b(1 << 20);
        while (g && kf) { kf.read(b.data(), b.size()); if (kf.gcount() > 0) gzwrite(g, b.data(), static_cast<unsigned>(kf.gcount())); }
        if (g) gzclose(g);
    }
    std::string manifest = slurp(cfg.t1k_dir + "/manifest.json");
    while (!manifest.empty() && (manifest.back() == '\n' || manifest.back() == ' ')) manifest.pop_back();
    if (manifest.empty()) manifest = "null";
    {
        std::ofstream j(dir + "/hla_summary.json");
        j << "{\n  \"t1k\": " << manifest << ",\n  \"ran\": " << (res.ran ? "true" : "false") << ",\n  \"why_not\": \"" << res.why_not
          << "\",\n  \"prefilter\": {\"k\": " << kK << ", \"stride\": " << kStride << ", \"hla_alleles\": " << res.n_alleles << ", \"kmers\": " << res.n_kmers
          << ", \"reads_scanned\": " << res.reads_scanned << ", \"reads_kept\": " << res.reads_kept << ", \"seconds\": " << res.seconds_prefilter
          << "},\n  \"t1k_seconds\": " << res.seconds_t1k << ",\n  \"sample\": " << (res.ran ? genotype_json(dir + "/hla_genotype.tsv") : std::string("null"))
          << ",\n  \"donor_rule\": \"called cells (cell_status cell) of one donor with prob_max >= 0.9, matched by raw cell barcode\""
          << ",\n  \"donors\": " << donors_json << "\n}\n";
    }
    if (!std::getenv("SINGLET_HLA_KEEP_TMP")) std::system(("rm -rf " + shq(tmp)).c_str());
    std::system(("chmod -R go-rwx " + shq(dir)).c_str());
    return res;
}

}  // namespace singlet::hla
