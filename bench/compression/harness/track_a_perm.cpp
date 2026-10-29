// Track A deliverable 3: what does the row permutation cost, and what are the alternatives?
//
//   track_a_perm [--out results/track_a_perm.tsv] corpus.tsv corpus_dir
//
// For every corpus matrix:
//   today        perm block as the production writer stores it (m u32, zstd-3)
//   rowcounts    per-row nnz counts (LEB128) with the best of {plain, zigzag-delta} x {zstd-19, xz, brotli};
//                the decoder re-derives the permutation (counting sort desc, ties ascending row id)
//   entropy      order-0 entropy of the row counts (a floor for any coder that ignores row context)
//   ref order    fixed reference row order = descending mean nnz share over OTHER studies with the same row
//                vocabulary (leave-one-study-out; the two mouse samples are one study so they fall back to the
//                other sample of the same study, flagged). Cost: nothing per file, model = the order table.
//   identity     no permutation at all (vocabulary / GTF order)
// and the effect of each on the chunk payload (zstd-3, chunk_cols 1024, exactly the production entropy stage).
#include "common.h"
#include "track_a.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace cs;
using namespace cs::ta;

struct Sample { std::string id, gse; };

static double entropy_bytes(const std::vector<uint32_t>& v) {
    std::map<uint32_t, size_t> h;
    for (auto x : v) h[x]++;
    double e = 0, n = double(v.size());
    for (auto& [k, c] : h) { (void)k; double p = double(c) / n; e -= double(c) * std::log2(p); }
    return e / 8;
}

int main(int argc, char** argv) {
    std::string out = "results/track_a_perm.tsv";
    std::vector<std::string> pos;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--out" && i + 1 < argc) out = argv[++i]; else if (a == "--check-only") check_only = true; else pos.push_back(a); }
    std::vector<Sample> samples;
    {
        std::ifstream f(pos[0]); std::string line; std::getline(f, line);
        while (std::getline(f, line)) { std::stringstream ss(line); std::string id, gse; std::getline(ss, id, '\t'); std::getline(ss, gse, '\t'); if (!id.empty()) samples.push_back({id, gse}); }
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tm\tnnz\tperm_today_file\tperm_today_z3\trowcounts_best\trowcounts_z3_plain\trowcounts_order0_entropy\tref_order_model_z3\tref_training\tchunks_own_perm\tchunks_ref_perm\tchunks_identity_perm\n";
    for (const char* kind : {"exon", "intron"}) {
        struct Item { Sample s; uint64_t vhash; std::vector<uint32_t> rnnz; uint64_t nnz; CanonicalMatrix M; size_t perm_file; std::vector<uint32_t> perm_stored; };
        std::vector<Item> items;
        for (auto& s : samples) {
            Item it; it.s = s;
            std::string path = pos[1] + "/" + s.id + "/" + kind + "_counts.1pz";
            it.M = load_1pz(path);
            it.vhash = names_hash(it.M.rownames);
            it.rnnz.assign(it.M.m, 0);
            for (auto r : it.M.indices) it.rnnz[r]++;
            it.nnz = it.M.nnz;
            auto file = read_file(path);
            singlet::pz::PZHeader h; std::memcpy(&h, file.data(), sizeof(h));
            it.perm_file = h.perm_z_sz;
            {   // the permutation exactly as stored in the original file
                std::vector<uint8_t> rawp(size_t(h.m) * 4);
                size_t z = ZSTD_decompress(rawp.data(), rawp.size(), file.data() + sizeof(h), h.perm_z_sz);
                if (ZSTD_isError(z)) throw std::runtime_error("perm block");
                it.perm_stored.resize(h.m);
                std::memcpy(it.perm_stored.data(), rawp.data(), rawp.size());
            }
            items.push_back(std::move(it));
        }
        for (auto& it : items) {
            const CanonicalMatrix& M = it.M;
            if (check_only) {
                std::vector<uint32_t> rn(M.m, 0);
                for (auto r : M.indices) rn[r]++;
                bool same = perm_from_rnnz(rn) == it.perm_stored;
                std::printf("%-22s %-7s perm re-derived from row counts == stored permutation: %s\n", it.s.id.c_str(), kind, same ? "IDENTICAL" : "DIFFERENT");
                continue;
            }
            // own permutation: production
            Prepared P = prepare(M, 1024, nullptr, 2);
            PermStats ps0; enc_perm(P, 0, &ps0);
            PermStats ps1; auto rc = enc_perm(P, 1, &ps1);
            // plain varint + zstd-3 for reference
            std::vector<uint8_t> raw; for (auto v : P.rnnz) pz::varint_push(raw, v);
            size_t rc_z3 = z_compress(ec_zstd(3), raw).size();
            double ent = entropy_bytes(P.rnnz);
            auto chunk_bytes = [&](const Prepared& Q) { size_t s = 0; for (auto& b : entropy_chunks(Q, ec_zstd(3))) s += b.size(); return s; };
            size_t own = chunk_bytes(P);
            // reference order from other studies (fallback: other sample of the same study)
            std::vector<double> score(M.m, 0.0);
            int ntrain = 0; bool same_study = false;
            for (int pass = 0; pass < 2 && ntrain == 0; ++pass)
                for (auto& o : items) {
                    if (&o == &it || o.vhash != it.vhash) continue;
                    if (pass == 0 && o.s.gse == it.s.gse) continue;
                    if (pass == 1 && o.s.gse != it.s.gse) continue;
                    if (pass == 1) same_study = true;
                    for (uint32_t i = 0; i < M.m; ++i) score[i] += double(o.rnnz[i]) / double(o.nnz);
                    ++ntrain;
                }
            std::vector<uint32_t> ref(M.m);
            std::iota(ref.begin(), ref.end(), 0u);
            std::stable_sort(ref.begin(), ref.end(), [&](uint32_t a, uint32_t b) { return score[a] > score[b]; });
            std::vector<uint8_t> rawref(size_t(M.m) * 4);
            std::memcpy(rawref.data(), ref.data(), rawref.size());
            size_t ref_model = z_compress(ec_zstd(3), rawref).size();
            Prepared PR = prepare(M, 1024, &ref, 2);
            size_t refb = chunk_bytes(PR);
            std::vector<uint32_t> idp(M.m); std::iota(idp.begin(), idp.end(), 0u);
            Prepared PI = prepare(M, 1024, &idp, 2);
            size_t idb = chunk_bytes(PI);
            std::string tr = ntrain ? (std::to_string(ntrain) + (same_study ? " samples, SAME study" : " samples, other studies")) : "none";
            std::printf("%-22s %-7s perm today %8zu (our z3 %8zu) | rowcounts best %7zu (z3 plain %7zu, order0 entropy %7.0f) | ref-order model %8zu (%s) | chunks own %9zu ref %9zu (%+.1f%%) identity %9zu (%+.1f%%)\n",
                        it.s.id.c_str(), kind, it.perm_file, ps0.bytes, ps1.bytes, rc_z3, ent, ref_model, tr.c_str(), own, refb, 100.0 * (double(refb) / own - 1), idb, 100.0 * (double(idb) / own - 1));
            tsv << it.s.id << '\t' << kind << "_counts.1pz\t" << M.m << '\t' << M.nnz << '\t' << it.perm_file << '\t' << ps0.bytes << '\t' << ps1.bytes << '\t' << rc_z3 << '\t' << ent << '\t' << ref_model << '\t' << tr << '\t' << own << '\t' << refb << '\t' << idb << '\n';
            tsv.flush();
        }
    }
    return 0;
}
