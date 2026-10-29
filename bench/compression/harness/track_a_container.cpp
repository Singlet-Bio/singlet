// Track A deliverable 6: why does re-emitting a .1pz with our writer not reproduce the original bytes?
//
//   track_a_container [--out results/track_a_container.tsv] file.1pz ...
//
// For every file: re-emit the decoded matrix with the production writer (zstd linked into this binary), then
// compare section by section (perm, ptr, chunk table, every chunk blob, colsums, metadata). A section whose
// *decompressed* bytes agree but whose compressed bytes differ is a pure zstd-encoder difference. For each
// original zstd frame we print ZSTD_getFrameHeader fields (window size, checksum flag, content size).
#define ZSTD_STATIC_LINKING_ONLY
#include "common.h"
#include "track_a.h"
#include <iostream>

using namespace cs;
using namespace singlet::pz;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

struct Sec { std::string name; size_t off, len; };

static std::vector<Sec> sections(const std::vector<uint8_t>& b, PZHeader& h) {
    std::memcpy(&h, b.data(), sizeof(h));
    std::vector<Sec> s;
    size_t off = 96;
    s.push_back({"perm", off, h.perm_z_sz}); off += h.perm_z_sz;
    s.push_back({"ptr", off, h.ptr_z_sz}); off += h.ptr_z_sz;
    s.push_back({"ctable", off, 4ull * h.num_chunks});
    std::vector<uint32_t> ct(h.num_chunks);
    std::memcpy(ct.data(), b.data() + off, 4ull * h.num_chunks);
    off += 4ull * h.num_chunks;
    for (uint32_t c = 0; c < h.num_chunks; ++c) { s.push_back({"chunk" + std::to_string(c), off, ct[c]}); off += ct[c]; }
    s.push_back({"colsums", off, h.colsums_z_sz}); off += h.colsums_z_sz;
    s.push_back({"meta", off, h.metadata_z_sz});
    return s;
}

static const char* strat_name(int s) {
    static const char* n[] = {"?", "fast", "dfast", "greedy", "lazy", "lazy2", "btlazy2", "btopt", "btultra", "btultra2"};
    return s >= 1 && s <= 9 ? n[s] : "?";
}

int main(int argc, char** argv) {
    std::string out = "results/track_a_container.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--out" && i + 1 < argc) out = argv[++i]; else files.push_back(a); }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\torig_bytes\tre_bytes\tdelta_bytes\tidentical\tsections\tsec_identical\tsec_zstd_only_diff\tsec_content_diff\tperm_orig\tperm_re\tptr_orig\tptr_re\tchunks_orig\tchunks_re\tcolsums_orig\tcolsums_re\tmeta_orig\tmeta_re\tmax_windowLog\tchecksum_frames\tcodec_level\n";
    for (auto& f : files) {
        auto orig = read_file(f);
        auto M = load_1pz(f);
        std::string tmp = "results/.track_a_container.tmp.1pz";
        auto re = rewrite_1pz(M, tmp);
        std::remove(tmp.c_str());
        PZHeader ho, hr;
        auto so = sections(orig, ho), sr = sections(re, hr);
        std::printf("== %s/%s: orig %zu B, re-emitted %zu B (%+lld), identical=%d, header level=%u chunk_cols=%u\n", parent_name(f).c_str(),
                    base_name(f).c_str(), orig.size(), re.size(), (long long)re.size() - (long long)orig.size(), orig == re ? 1 : 0, ho.codec_level, ho.chunk_cols);
        int same = 0, zonly = 0, content = 0, total = 0;
        size_t po = 0, pr = 0, qo = 0, qr = 0, co = 0, cr = 0, so_ = 0, sr_ = 0, mo = 0, mr = 0;
        unsigned max_wlog = 0, ck = 0;
        ZSTD_DCtx* d = ZSTD_createDCtx();
        for (size_t i = 0; i < so.size() && i < sr.size(); ++i) {
            auto& a = so[i]; auto& r = sr[i];
            if (a.name == "ctable") continue;
            ++total;
            const uint8_t* ap = orig.data() + a.off; const uint8_t* rp = re.data() + r.off;
            // chunk blobs have a 20-byte header before the frame
            size_t hdr = a.name.rfind("chunk", 0) == 0 ? 20 : 0;
            bool eq = (a.len == r.len) && std::memcmp(ap, rp, a.len) == 0;
            if (a.name == "perm") { po = a.len; pr = r.len; }
            else if (a.name == "ptr") { qo = a.len; qr = r.len; }
            else if (a.name == "colsums") { so_ = a.len; sr_ = r.len; }
            else if (a.name == "meta") { mo = a.len; mr = r.len; }
            else { co += a.len; cr += r.len; }
            ZSTD_frameHeader fh;
            if (a.len > hdr && ZSTD_getFrameHeader(&fh, ap + hdr, a.len - hdr) == 0) {
                max_wlog = std::max<unsigned>(max_wlog, (unsigned)(31 - __builtin_clz((unsigned)std::max<unsigned long long>(1, fh.windowSize))));
                ck += fh.checksumFlag;
                if (i < 4 || a.name == "colsums" || a.name == "meta")
                    std::printf("   %-8s frame: content %llu window %llu blockMax %u checksum %u dictID %u frameType %d | orig %zu re %zu %s\n", a.name.c_str(), fh.frameContentSize,
                                fh.windowSize, fh.blockSizeMax, fh.checksumFlag, fh.dictID, (int)fh.frameType, a.len, r.len, eq ? "IDENTICAL" : "differs");
            }
            if (eq) { ++same; continue; }
            // same decompressed content?
            auto dec = [&](const uint8_t* p, size_t n, std::vector<uint8_t>& o) {
                unsigned long long cs = ZSTD_getFrameContentSize(p, n);
                if (cs == ZSTD_CONTENTSIZE_ERROR || cs == ZSTD_CONTENTSIZE_UNKNOWN) return false;
                o.resize(size_t(cs));
                size_t z = ZSTD_decompressDCtx(d, o.data(), o.size(), p, n);
                return !ZSTD_isError(z) && z == o.size();
            };
            std::vector<uint8_t> x, y;
            bool ok = a.len > hdr && r.len > hdr && dec(ap + hdr, a.len - hdr, x) && dec(rp + hdr, r.len - hdr, y);
            if (ok && x == y) ++zonly; else ++content;
        }
        ZSTD_freeDCtx(d);
        std::printf("   sections compared %d: byte-identical %d, same content but different zstd bytes %d, different content %d\n", total, same, zonly, content);
        std::printf("   sizes orig/re: perm %zu/%zu ptr %zu/%zu chunks %zu/%zu colsums %zu/%zu meta %zu/%zu; max windowLog in orig frames %u, frames with checksum %u\n", po, pr, qo, qr, co, cr, so_, sr_, mo, mr, max_wlog, ck);
        tsv << parent_name(f) << '\t' << base_name(f) << '\t' << orig.size() << '\t' << re.size() << '\t' << (long long)re.size() - (long long)orig.size() << '\t' << (orig == re ? 1 : 0) << '\t' << total << '\t' << same
            << '\t' << zonly << '\t' << content << '\t' << po << '\t' << pr << '\t' << qo << '\t' << qr << '\t' << co << '\t' << cr << '\t' << so_ << '\t' << sr_ << '\t' << mo << '\t' << mr << '\t' << max_wlog << '\t' << ck << '\t' << (unsigned)ho.codec_level << '\n';
        tsv.flush();
    }
    std::printf("linked zstd: %s\n", ZSTD_versionString());
    return 0;
}
