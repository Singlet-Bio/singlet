// Track A deliverable 4: entropy-coder x chunk-size sweep on the exact prefilter bytes pz_writer.h hands to zstd.
//
//   track_a_sweep --out results/track_a_sweep.tsv [--chunks 256,1024,4096,0] [--coders z1,z3,...] [--parts] file.1pz ...
//
// chunk 0 = whole matrix in one chunk. Coder labels: z<level>, z22l (zstd 22, windowLog 27, long-distance matching),
// xz, br, bz2, zd<level> (zstd with a dictionary trained on other studies, looked up in the dict manifest).
// For every (file, chunk_cols, coder) one row:
//   payload bytes (sum of entropy-coder output), bits/nnz of that payload, encode ms, decode ms of the entropy stage
//   alone, and the estimated full decode time = (decode work that is the same for every coder, measured once with the
//   store coder through the whole container decode: bitmap unpack, bit planes, varint walk, per-column re-sort)
//   + (that coder's own entropy-stage decode time). The chunk table and 20-byte blob headers are reported separately
//   (they are what random access costs). Every coder's output is verified chunk by chunk against the original bytes.
#include "common.h"
#include "track_a.h"
#include <iostream>
#include <sstream>

using namespace cs;
using namespace cs::ta;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

static bool make_ec(const std::string& l, const CanonicalMatrix& M, EC& ec) {
    if (l.size() > 2 && l[0] == 'z' && l[1] == 'd') {   // zd3, zd19
        int lev = std::atoi(l.c_str() + 2);
        auto dp = load_dict_for(M);
        if (!dp) return false;
        auto d2 = std::make_shared<DictPair>(dp->raw, lev);   // re-create the CDict at the requested level
        dict_registry()[d2->id] = d2;
        ec = ec_zstd(lev); ec.dict = d2; ec.label = l;
        return true;
    }
    if (l == "z22l") { ec = ec_zstd(22, 27, true); ec.label = l; return true; }
    if (l == "z22w") { ec = ec_zstd(22, 27, false); ec.label = l; return true; }
    if (l[0] == 'z') { ec = ec_zstd(std::atoi(l.c_str() + 1)); ec.label = l; return true; }
    if (l == "xz") { ec = ec_xz(); ec.label = l; return true; }
    if (l == "br") { ec = ec_brotli(11, 24); ec.label = l; return true; }
    if (l == "bz2") { ec = ec_bzip2(); ec.label = l; return true; }
    if (l == "store") { ec = ec_store(); ec.label = l; return true; }
    return false;
}

static double time_entropy_decode(const std::vector<std::vector<uint8_t>>& blobs, const EC& ec, const Prepared* verify, bool& ok) {
    double best = 1e300;
    std::vector<uint8_t> scratch;
    for (int r = 0; r < 3; ++r) {
        Timer td;
        for (size_t c = 0; c < blobs.size(); ++c) {
            uint32_t zsz, psz;
            std::memcpy(&zsz, blobs[c].data() + 8, 4); std::memcpy(&psz, blobs[c].data() + 16, 4);
            scratch.resize(psz);
            ec_decompress(ec, blobs[c].data() + 20, zsz, scratch.data(), psz);
        }
        best = std::min(best, td.ms());
    }
    if (verify) {   // untimed verification pass
        for (size_t c = 0; c < blobs.size(); ++c) {
            uint32_t zsz, psz;
            std::memcpy(&zsz, blobs[c].data() + 8, 4); std::memcpy(&psz, blobs[c].data() + 16, 4);
            scratch.resize(psz);
            ec_decompress(ec, blobs[c].data() + 20, zsz, scratch.data(), psz);
            if (scratch != verify->chunks[c].packed) ok = false;
        }
    }
    return best;
}

int main(int argc, char** argv) {
    std::string out = "results/track_a_sweep.tsv";
    std::vector<int> chunks = {256, 1024, 4096, 0};
    std::vector<std::string> coders = {"z1", "z3", "z6", "z9", "z12", "z15", "z19", "z22", "z22l", "xz", "br", "bz2"};
    std::vector<std::string> files;
    bool parts = false;
    auto split = [](const std::string& s) { std::vector<std::string> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) v.push_back(t); return v; };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--chunks" && i + 1 < argc) { chunks.clear(); for (auto& t : split(argv[++i])) chunks.push_back(std::atoi(t.c_str())); }
        else if (a == "--coders" && i + 1 < argc) coders = split(argv[++i]);
        else if (a == "--parts") parts = true;
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tchunk_cols\tcoder\tnnz\tprefilter_bytes\tpacked_bytes\tpayload_bytes\thdr_bytes\tbits_per_nnz_payload\tenc_ms\tdec_entropy_ms\tdec_full_ms_est\tdec_full_Mnnz_s\tdec_entropy_Mnnz_s\tok\tmodel_bytes\n";
    for (auto& f : files) {
        auto M = load_1pz(f);
        auto Mn = M; Mn.rownames.clear(); Mn.colnames.clear();   // matrix-only comparison (names_mode 2)
        std::string sample = parent_name(f), member = base_name(f);
        for (int cc : chunks) {
            int chunk_cols = (cc == 0 || cc > int(M.n)) ? int(M.n) : cc;
            Prepared P = prepare(M, chunk_cols, nullptr, 2);
            size_t packed = P.packed_total();
            size_t pf = 0;
            for (auto& c : P.chunks) pf += P.pf_size(c);
            std::printf("== %s/%s chunk_cols=%d nc=%d packed=%zu\n", sample.c_str(), member.c_str(), cc, P.nc, packed);
            if (parts) {
                // each logical stream compressed alone with zstd-19: varint metadata / bit planes of the low gap byte / upper gap bytes
                size_t m_ = 0, p_ = 0, u_ = 0, mraw = 0, praw = 0, uraw = 0;
                for (auto& c : P.chunks) {
                    auto sp = split_chunk(P, c);
                    m_ += z_compress(ec_zstd(19), sp.meta).size(); p_ += z_compress(ec_zstd(19), sp.planes).size(); u_ += z_compress(ec_zstd(19), sp.upper).size();
                    mraw += sp.meta.size(); praw += sp.planes.size(); uraw += sp.upper.size();
                }
                std::printf("   parts (zstd19 each alone): meta %zu->%zu  planes %zu->%zu  upper %zu->%zu\n", mraw, m_, praw, p_, uraw, u_);
                tsv << sample << '\t' << member << '\t' << cc << "\tparts_z19\t" << M.nnz << '\t' << pf << '\t' << packed << '\t' << (m_ + p_ + u_) << "\t0\t"
                    << (m_ + p_ + u_) * 8.0 / M.nnz << "\t0\t0\t0\t0\t0\t1\tmeta=" << m_ << ";planes=" << p_ << ";upper=" << u_ << "\n";
            }
            // The decode work that does not depend on the entropy coder, measured once through the real container decoder.
            double full_store = 1e300, store_entropy = 0;
            {
                EC st = ec_store();
                Plan pl; pl.ec = st; pl.chunk_cols = chunk_cols; pl.names_mode = 2; pl.perm_mode = 0;
                auto stream = encode_plan(M, pl, nullptr, &P);
                for (int r = 0; r < 3; ++r) {
                    Timer td;
                    auto D = decode_plan(stream);
                    full_store = std::min(full_store, td.ms());
                    if (r == 0 && !(D == Mn)) { std::printf("   STORE ROUNDTRIP FAIL\n"); return 1; }
                }
                bool ok = true;
                store_entropy = time_entropy_decode(entropy_chunks(P, st), st, nullptr, ok);
                double ne = full_store - store_entropy;
                std::printf("   decode work independent of the entropy coder: %.1f ms (%.1f Mnnz/s ceiling)\n", ne, double(M.nnz) / ne / 1e3);
                tsv << sample << '\t' << member << '\t' << cc << "\tnon_entropy_decode\t" << M.nnz << '\t' << pf << '\t' << packed << "\t0\t0\t0\t0\t0\t" << ne << '\t'
                    << double(M.nnz) / ne / 1e3 << "\t0\t1\t0\n";
            }
            for (auto& label : coders) {
                EC ec;
                if (!make_ec(label, M, ec)) { std::printf("   %-6s skipped (no model)\n", label.c_str()); continue; }
                Timer te;
                auto blobs = entropy_chunks(P, ec);
                double enc_ms = te.ms();
                size_t payload = 0, hdr = 24 * blobs.size();
                for (auto& b : blobs) payload += b.size() - 20;
                bool ok = true;
                double dec_e = time_entropy_decode(blobs, ec, &P, ok);
                double dec_f = full_store - store_entropy + dec_e;
                double bpn = payload * 8.0 / double(M.nnz);
                size_t mb = ec.dict ? ec.dict->raw.size() : 0;
                std::printf("   %-6s payload %10zu B  %6.3f bits/nnz  enc %9.1f ms  dec(entropy) %8.1f ms  dec(full est) %8.1f ms (%.1f Mnnz/s) %s\n",
                            label.c_str(), payload, bpn, enc_ms, dec_e, dec_f, double(M.nnz) / dec_f / 1e3, ok ? "OK" : "MISMATCH");
                tsv << sample << '\t' << member << '\t' << cc << '\t' << label << '\t' << M.nnz << '\t' << pf << '\t' << packed << '\t' << payload << '\t' << hdr << '\t'
                    << bpn << '\t' << enc_ms << '\t' << dec_e << '\t' << dec_f << '\t' << double(M.nnz) / dec_f / 1e3 << '\t' << double(M.nnz) / dec_e / 1e3 << '\t'
                    << (ok ? 1 : 0) << '\t' << mb << '\n';
                tsv.flush();
            }
        }
    }
    return 0;
}
