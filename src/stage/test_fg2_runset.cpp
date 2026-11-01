// test_fg2_runset: the gate for Fg2RunSet. The decoded blocks of a .1fg2 must equal, read for read, the .1fq RunSet's blocks of the same reads
// (barcode as sequenced + UMI, cDNA, decoded quality); reads the stage drops (R1 too short, empty R2) are skipped on the .1fq side.
//
//   test_fg2_runset --whitelist-dir D --fg2 FILE.1fg2 [--ordered] [--threads N] RUN1.1fq [RUN2.1fq ...]
//
// Always compares an order-independent digest (sum over reads of a 64-bit hash of the three byte strings) and the read count; with --ordered also compares
// read k with read k (same order: a .1fg2 built from the same .1fq). --threads decodes the .1fg2 chunks concurrently to exercise the thread safety.
// Exit 0 = identical.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <singlet/fq/fg2_runset.h>
#include <singlet/fq/read_source.h>

namespace sq = singlet::fq;

static uint64_t mix(uint64_t h, uint64_t w) { h = (h ^ w) * 0x9E3779B97F4A7C15ull; return h ^ (h >> 31); }
static uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t h) { for (size_t i = 0; i < n; ++i) h = mix(h, p[i]); return mix(h, n); }

struct Rec { std::vector<uint8_t> r1, r2, q; };
static void get(const sq::DecodedBlock& b, uint32_t i, Rec& r) {
    r.r1.assign(b.r1_seq(i), b.r1_seq(i) + b.r1_len(i));
    r.r2.assign(b.r2_seq(i), b.r2_seq(i) + b.r2_len(i));
    const uint8_t* q = b.r2_qual.empty() ? nullptr : b.r2_quality(i);
    r.q.assign(q ? q : r.r2.data(), (q ? q : r.r2.data()) + (q ? b.r2_len(i) : 0));
}
static uint64_t rec_hash(const Rec& r) { return hash_bytes(r.q.data(), r.q.size(), hash_bytes(r.r2.data(), r.r2.size(), hash_bytes(r.r1.data(), r.r1.size(), 0x51ED))); }

int main(int argc, char** argv) {
    std::vector<std::string> fqs, wl;
    std::string fg2path;
    bool ordered = false;
    int threads = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--fg2" && i + 1 < argc) fg2path = argv[++i];
        else if (a == "--whitelist-dir" && i + 1 < argc) wl.push_back(argv[++i]);
        else if (a == "--ordered") ordered = true;
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else fqs.push_back(a);
    }
    if (fg2path.empty() || fqs.empty()) { std::fprintf(stderr, "usage: test_fg2_runset --whitelist-dir D --fg2 F.1fg2 [--ordered] [--threads N] RUN.1fq...\n"); return 2; }
    sq::RunSet rs; rs.open(fqs);
    sq::Fg2RunSet fr; fr.open({fg2path}, wl);
    std::printf("fq blocks %u (%llu reads), fg2 chunks %u (%llu reads, %llu distinct sequences, q plane %d)\n", rs.block_count(), (unsigned long long)rs.stored_reads(),
                fr.block_count(), (unsigned long long)fr.stored_reads(), (unsigned long long)fr.table().n, (int)fr.has_quality());
    const int bcl = fr.bc_length(), umil = fr.fg2_header().umi_len, need = std::max(fr.fg2_header().cb_start + bcl, fr.fg2_header().umi_start + umil);

    // .1fq side: flatten the selected reads (order kept) into a digest and, when ordered, a vector of hashes
    uint64_t fq_digest = 0, fq_n = 0;
    std::vector<uint64_t> fq_hashes;
    for (uint32_t g = 0; g < rs.block_count(); ++g) {
        sq::DecodedBlock b; rs.decode(g, b); Rec r;
        for (uint32_t i = 0; i < b.n_reads; ++i) {
            if (b.r1_len(i) < need || b.r2_len(i) == 0) continue;
            get(b, i, r);
            r.r1.resize(bcl + umil);   // the stage keeps barcode + UMI; the .1fq read may carry more
            uint64_t h = rec_hash(r);
            fq_digest += h; fq_n++;
            if (ordered) fq_hashes.push_back(h);
        }
    }
    // .1fg2 side, optionally with several decoder threads
    std::vector<uint64_t> part_digest(fr.block_count(), 0), part_n(fr.block_count(), 0);
    std::vector<std::vector<uint64_t>> part_hashes(ordered ? fr.block_count() : 0);
    std::atomic<uint32_t> next{0};
    std::atomic<int> errors{0};
    auto work = [&]() {
        for (uint32_t g; (g = next.fetch_add(1)) < fr.block_count();) {
            try {
                sq::DecodedBlock b; fr.decode(g, b); Rec r;
                for (uint32_t i = 0; i < b.n_reads; ++i) {
                    get(b, i, r);
                    // dictionary semantics: bc_indices[i] < dict_n  <=>  exact hit, and then the dictionary entry is the barcode as sequenced
                    const uint32_t bi = b.bc_indices[i];
                    if (bi < fr.bc_dict_size()) { if (std::memcmp(&fr.bc_dict_flat()[(size_t)bi * bcl], b.r1_seq(i), bcl) != 0) { errors++; std::fprintf(stderr, "dict entry != barcode (chunk %u read %u)\n", g, i); } }
                    else if (bi != fr.bc_dict_size()) { errors++; std::fprintf(stderr, "bad bc index\n"); }
                    uint64_t h = rec_hash(r);
                    part_digest[g] += h; part_n[g]++;
                    if (ordered) part_hashes[g].push_back(h);
                }
            } catch (const std::exception& e) { std::fprintf(stderr, "chunk %u: %s\n", g, e.what()); errors++; }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    uint64_t fg_digest = 0, fg_n = 0;
    for (size_t g = 0; g < part_digest.size(); ++g) { fg_digest += part_digest[g]; fg_n += part_n[g]; }
    std::printf("fq reads %llu digest %016llx | fg2 reads %llu digest %016llx\n", (unsigned long long)fq_n, (unsigned long long)fq_digest,
                (unsigned long long)fg_n, (unsigned long long)fg_digest);
    bool ok = errors == 0 && fq_n == fg_n && fq_digest == fg_digest;
    if (ordered) {
        uint64_t k = 0, bad = 0;
        for (size_t g = 0; g < part_hashes.size(); ++g)
            for (uint64_t h : part_hashes[g]) { if (k >= fq_hashes.size() || fq_hashes[k] != h) { if (bad++ < 3) std::fprintf(stderr, "ordered mismatch at read %llu\n", (unsigned long long)k); } ++k; }
        std::printf("ordered: %llu reads compared, %llu mismatches\n", (unsigned long long)k, (unsigned long long)bad);
        ok = ok && bad == 0 && k == fq_hashes.size();
    }
    {   // metadata: the .1fg2 carries the source .1fq's JSON verbatim (plus the encoder's own key at the end)
        sq::Reader r0; r0.open(fqs[0].c_str());
        std::string want = r0.read_metadata(), have = fr.metadata();
        while (!want.empty() && (want.back() == ' ' || want.back() == '\n' || want.back() == '\r' || want.back() == '\t')) want.pop_back();
        const std::string prefix = want.empty() ? std::string() : want.substr(0, want.size() - 1);
        const bool meta_ok = want.empty() || have.compare(0, prefix.size(), prefix) == 0;
        std::printf("metadata: %zu bytes in the .1fq, %zu in the .1fg2, verbatim prefix %s\n", want.size(), have.size(), meta_ok ? "OK" : "MISMATCH");
        ok = ok && meta_ok;
    }
    std::printf("%s\n", ok ? "IDENTICAL" : "DIFFERENT");
    return ok ? 0 : 1;
}
