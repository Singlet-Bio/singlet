// singlet-stage: fetch-tier encoder  .1fq (one or more runs of a sample)  ->  .1fg2  (staged GPU-native read container, src/stage/fmt2.h).
//
//   singlet-stage --whitelist-dir D [--whitelist-dir D2 ...] [--accessions SRR1,SRR2] [--level 3] [--scratch DIR] [--json FILE] -o OUT.1fg2 RUN1.1fq [RUN2.1fq ...]
//
// One source of truth: the protocol (CandidateSpec from fq/protocol.h, by the protocol id in the .1fq header), the barcode geometry, the whitelist file
// and the R11 correction (CbCorrector::resolve, STAR 1MM_multi_Nbase_pseudocounts with the sample's exact histogram) are the engine's own headers; the .1fq
// is read with the engine's Reader. No clipping or trimming here: R2 is stored exactly as the .1fq holds it (the engine clips on the packed codes).
//
// Pass 1  (per .1fq block): R1 -> raw barcode key + UMI; exact-barcode histogram (exact whitelist hit and a UMI with no N that is not a homopolymer, as
//         singlet.cpp pass 1); R2 -> 2-bit slot, deduplicated into the sample's sequence table (first occurrence order); binned quality plane (ILLUMINA4
//         bins by Phred: <10, 10-19, 20-29, >=30); the per-read record is spilled to node-local scratch.
// Pass 2  (per spill chunk of 32,768 reads): resolve the barcode with the final histogram (cached per raw barcode), write the planes, one zstd frame each, CRC32C.
// Then the sequence blocks, the run table, the index and the footer. Output: one JSON line on stdout (timings, CPU seconds, bytes per plane, B/pair).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <zstd.h>

#include <singlet/fq/protocol.h>
#include <singlet/fq/reader.h>
#include <singlet/pileup/cb_correct.h>
#include "fmt2.h"
#include "stage_verify.h"

using namespace fg2;
namespace sq = singlet::fq;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double cpu_s() {
    rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}
[[noreturn]] static void die(int code, const std::string& m) { std::fprintf(stderr, "[singlet-stage] %s\n", m.c_str()); std::exit(code); }

// ── SHA-256 (whitelist identity in the header) ───────────────────────────────────────────────────────────────────────
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t len = 0; uint8_t buf[64]; size_t nb = 0;
    static uint32_t rr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) { uint32_t s0 = rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3), s1 = rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10); w[i] = w[i-16]+s0+w[i-7]+s1; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rr(e,6)^rr(e,11)^rr(e,25), ch = (e&f)^(~e&g), t1 = hh+S1+ch+K[i]+w[i];
            uint32_t S0 = rr(a,2)^rr(a,13)^rr(a,22), mj = (a&b)^(a&c)^(b&c), t2 = S0+mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n) { size_t k = std::min(n, 64 - nb); std::memcpy(buf + nb, p, k); nb += k; p += k; n -= k; if (nb == 64) { block(buf); nb = 0; } }
    }
    void finish(uint8_t out[32]) {
        uint64_t bits = len * 8; uint8_t pad = 0x80; update(&pad, 1); uint8_t z = 0;
        while (nb != 56) update(&z, 1);
        uint8_t l[8]; for (int i = 0; i < 8; i++) l[i] = (uint8_t)(bits >> (56 - 8 * i)); update(l, 8);
        for (int i = 0; i < 8; i++) { out[4*i] = h[i] >> 24; out[4*i+1] = h[i] >> 16; out[4*i+2] = h[i] >> 8; out[4*i+3] = h[i]; }
    }
};

// ── sequence table: open addressing over packed 2-bit sequences (N coded as A, N positions kept separately and part of the key) ───────────
struct SeqTable {
    static constexpr int NMB = 32;       // N-position bitmask bytes (255 bits)
    static constexpr int SLOT = 64;      // 255 nt * 2 bit = 510 bits
    std::vector<uint8_t> data, len, hasn;
    std::unordered_map<uint32_t, std::array<uint8_t, NMB>> nmask;
    std::vector<uint32_t> ht;            // id + 1, 0 = empty
    uint64_t mask = 0;
    uint64_t n() const { return len.size(); }
    SeqTable() { ht.assign(1u << 20, 0); mask = ht.size() - 1; }
    static uint64_t mix(uint64_t h, uint64_t w) { h = (h ^ w) * 0x9E3779B97F4A7C15ull; return h ^ (h >> 29); }
    static uint64_t hash(const uint8_t* slot, uint8_t l, const uint8_t* nm) {
        uint64_t h = 0x243F6A8885A308D3ull, w;
        for (int i = 0; i < SLOT; i += 8) { std::memcpy(&w, slot + i, 8); h = mix(h, w); }
        h = mix(h, l);
        if (nm) { for (int i = 0; i < NMB; i += 8) { std::memcpy(&w, nm + i, 8); h = mix(h, w); } }
        return h;
    }
    void grow() {
        std::vector<uint32_t> nt(ht.size() * 2, 0);
        uint64_t nmk = nt.size() - 1;
        for (uint64_t id = 0; id < n(); id++) {
            const uint8_t* nm = hasn[id] ? nmask.at((uint32_t)id).data() : nullptr;
            uint64_t p = hash(&data[id * SLOT], len[id], nm) & nmk;
            while (nt[p]) p = (p + 1) & nmk;
            nt[p] = (uint32_t)id + 1;
        }
        ht.swap(nt); mask = nmk;
    }
    uint32_t get_or_add(const uint8_t* slot, uint8_t l, const uint8_t* nm /* null if the sequence has no N */) {
        if ((n() + 1) * 2 > ht.size()) grow();
        uint64_t p = hash(slot, l, nm) & mask;
        for (;;) {
            uint32_t e = ht[p];
            if (!e) break;
            uint32_t id = e - 1;
            if (len[id] == l && std::memcmp(&data[(uint64_t)id * SLOT], slot, SLOT) == 0 && (bool)hasn[id] == (nm != nullptr) &&
                (!nm || std::memcmp(nmask.at(id).data(), nm, NMB) == 0)) return id;
            p = (p + 1) & mask;
        }
        uint32_t id = (uint32_t)n();
        if (id == 0xFFFFFFFEu) die(5, "more than 2^32-2 distinct sequences");
        data.insert(data.end(), slot, slot + SLOT);
        len.push_back(l);
        hasn.push_back(nm ? 1 : 0);
        if (nm) { std::array<uint8_t, NMB> a; std::memcpy(a.data(), nm, NMB); nmask.emplace(id, a); }
        ht[p] = id + 1;
        return id;
    }
};

#pragma pack(push, 1)
struct Rec { uint64_t cbkey; uint32_t umi; uint32_t seq; uint16_t umin; uint8_t len; uint8_t bad; };   // 20 bytes; followed by SLOT quality bytes if the sample has them
#pragma pack(pop)

// ── output helper ────────────────────────────────────────────────────────────────────────────────────────────────────
struct Out {
    FILE* f; uint64_t off = 0;
    void write(const void* p, size_t n) { if (n && std::fwrite(p, 1, n, f) != n) die(4, "write failed"); off += n; }
    void pad_to(uint64_t a) { static const uint8_t z[4096] = {0}; uint64_t t = align_up(off, a); while (off < t) write(z, std::min<uint64_t>(4096, t - off)); }
};

struct Plane { std::vector<uint8_t> raw; };

// compress one plane; returns stored bytes (zstd frame, or raw if no gain) and sets `rawflag`
static std::vector<uint8_t> pack_plane(const std::vector<uint8_t>& raw, int level, bool& rawflag) {
    rawflag = false;
    if (raw.empty()) return {};
    std::vector<uint8_t> out(ZSTD_compressBound(raw.size()));
    size_t c = ZSTD_compress(out.data(), out.size(), raw.data(), raw.size(), level);
    if (ZSTD_isError(c) || c >= raw.size()) { rawflag = true; return raw; }
    out.resize(c);
    return out;
}

int main(int argc, char** argv) {
    std::vector<std::string> wl_dirs, inputs, accs;
    std::string out_path, json_path, scratch = "/tmp";
    int level = 3;
    bool do_verify = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) die(2, "missing value for " + a); return argv[++i]; };
        if (a == "--whitelist-dir") wl_dirs.push_back(val());
        else if (a == "-o") out_path = val();
        else if (a == "--level") level = std::atoi(val().c_str());
        else if (a == "--scratch") scratch = val();
        else if (a == "--json") json_path = val();
        else if (a == "--verify") do_verify = true;
        else if (a == "--accessions") { std::string s = val(); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); accs.push_back(s.substr(p, q - p)); p = q + 1; } }
        else if (!a.empty() && a[0] == '-') die(2, "unknown option " + a);
        else inputs.push_back(a);
    }
    if (out_path.empty() || inputs.empty()) die(2, "usage: singlet-stage --whitelist-dir D -o OUT.1fg2 RUN1.1fq [RUN2.1fq ...]");
    if (wl_dirs.empty() && std::getenv("SINGLET_WHITELIST_DIR")) wl_dirs.push_back(std::getenv("SINGLET_WHITELIST_DIR"));
    double t0 = now_s(), c0 = cpu_s();

    // protocol, geometry, whitelist from the first run (all runs must agree)
    sq::Reader r0; r0.open(inputs[0].c_str());
    const uint8_t protocol_id = r0.header().protocol_id;
    const sq::CandidateSpec* spec = nullptr;
    for (const auto& s : sq::known_protocols()) if (s.protocol_id == protocol_id) { spec = &s; break; }
    if (!spec || spec->whitelist_file.empty() || !spec->per_seg_whitelist_files.empty() || spec->barcode_in_r2)
        die(6, "protocol " + std::to_string(protocol_id) + " has no single universal whitelist in R1: not staged (use the .1fq path)");
    std::string wl_path;
    for (const auto& d : wl_dirs) { std::string p = d + "/" + spec->whitelist_file; if (std::ifstream(p).good()) { wl_path = p; break; } }
    if (wl_path.empty()) die(6, "whitelist " + spec->whitelist_file + " not found in --whitelist-dir");
    singlet::CbCorrector corr;
    if (!corr.load_whitelist(wl_path)) die(6, "cannot load whitelist " + wl_path);
    const int bcl = r0.bc_length(), bc0 = spec->bc_offset;
    const int umil = r0.umi_length();
    const int umi0 = spec->umi_len > 0 ? spec->umi_offset : bcl;
    if (bcl <= 0 || bcl > 21 || bcl != corr.length()) die(6, "barcode length " + std::to_string(bcl) + " does not match the whitelist (" + std::to_string(corr.length()) + ")");
    if (umil <= 0 || umil > 16) die(6, "UMI length " + std::to_string(umil) + " not supported (1..16)");
    const bool has_q = r0.header().qual_mode != (uint8_t)sq::QualMode::NONE;
    r0.close();
    Sha256 sha; { std::ifstream wf(wl_path, std::ios::binary); char b[1 << 16]; while (wf.read(b, sizeof b) || wf.gcount()) sha.update((const uint8_t*)b, (size_t)wf.gcount()); }
    uint8_t wl_sha[32]; sha.finish(wl_sha);

    // ── pass 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────
    const std::string spill_path = scratch + "/s9-stage-spill-" + std::to_string(getpid());
    FILE* spill = std::fopen(spill_path.c_str(), "wb+");
    if (!spill) die(4, "cannot create spill file " + spill_path);
    SeqTable tab;
    std::vector<uint64_t> hist(corr.size(), 0);
    std::unordered_map<uint64_t, int64_t> cbfind;
    std::vector<RunEntry> runs;
    uint64_t n_in = 0, n_dropped = 0, n_stored = 0, hist_reads = 0, src_bytes = 0;
    uint32_t max_len = 0;
    const size_t recsz = sizeof(Rec) + (has_q ? SeqTable::SLOT : 0);
    std::vector<uint8_t> sbuf; sbuf.reserve(recsz * 65536);
    for (size_t ri = 0; ri < inputs.size(); ri++) {
        sq::Reader rd; rd.open(inputs[ri].c_str());
        if (rd.header().protocol_id != protocol_id || rd.bc_length() != bcl || rd.umi_length() != umil) die(6, "runs disagree on protocol/geometry: " + inputs[ri]);
        if ((rd.header().qual_mode != (uint8_t)sq::QualMode::NONE) != has_q) die(6, "runs disagree on quality mode: " + inputs[ri]);
        struct stat st; if (stat(inputs[ri].c_str(), &st) == 0) src_bytes += st.st_size;
        RunEntry re; std::memset(&re, 0, sizeof re);
        std::string acc = ri < accs.size() ? accs[ri] : inputs[ri].substr(inputs[ri].find_last_of('/') + 1);
        std::strncpy(re.accession, acc.c_str(), sizeof re.accession - 1);
        re.first_read = n_stored; re.raw_bytes = st.st_size;
        sq::DecodedBlock blk;
        while (rd.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                n_in++;
                const uint16_t r1l = blk.r1_len(i), r2l = blk.r2_len(i);
                if (r1l < bc0 + bcl || r1l < umi0 + umil || r2l == 0) { n_dropped++; continue; }
                if (r2l > MAX_SEQ_LEN) die(3, "cDNA longer than 255 nt (" + std::to_string(r2l) + "): not stageable");
                const uint8_t* r1 = blk.r1_seq(i);
                uint64_t cbkey = 0; bool cbN = false;
                for (int j = 0; j < bcl; j++) { uint8_t c = std::min<uint8_t>(r1[bc0 + j], 4); cbN |= c > 3; cbkey = (cbkey << 3) | c; }
                uint32_t umi = 0; uint16_t umin = 0; bool homo = true;
                for (int j = 0; j < umil; j++) { uint8_t c = r1[umi0 + j]; if (c > 3) { umin |= (uint16_t)(1u << j); c = 0; } umi = (umi << 2) | c; if (r1[umi0 + j] != r1[umi0]) homo = false; }
                const bool bad = umin != 0 || homo;
                if (!bad && !cbN) {
                    auto it = cbfind.find(cbkey);
                    int64_t w;
                    if (it == cbfind.end()) {
                        char s[32]; for (int j = 0; j < bcl; j++) s[j] = "ACGTN"[(cbkey >> (3 * (bcl - 1 - j))) & 7];
                        s[bcl] = 0; w = corr.find(s); cbfind.emplace(cbkey, w);
                    } else w = it->second;
                    if (w >= 0) { hist[w]++; hist_reads++; }
                }
                // cDNA -> slot
                uint8_t slot[SeqTable::SLOT] = {0}; uint8_t nm[SeqTable::NMB] = {0}; bool anyN = false;
                const uint8_t* r2 = blk.r2_seq(i);
                for (uint16_t j = 0; j < r2l; j++) { uint8_t c = r2[j]; if (c > 3) { anyN = true; nm[j >> 3] |= (uint8_t)(1u << (j & 7)); c = 0; } slot[j >> 2] |= (uint8_t)(c << (6 - 2 * (j & 3))); }
                const uint32_t sid = tab.get_or_add(slot, (uint8_t)r2l, anyN ? nm : nullptr);
                max_len = std::max<uint32_t>(max_len, r2l);
                Rec rec{cbkey, umi, sid, umin, (uint8_t)r2l, (uint8_t)bad};
                const size_t at = sbuf.size(); sbuf.resize(at + recsz);
                std::memcpy(&sbuf[at], &rec, sizeof rec);
                if (has_q) {
                    uint8_t* q = &sbuf[at + sizeof rec]; std::memset(q, 0, SeqTable::SLOT);
                    const uint8_t* qp = blk.r2_quality(i);
                    for (uint16_t j = 0; j < r2l; j++) { uint8_t p = qp ? qp[j] : 0; uint8_t b = p >= 30 ? 3 : p >= 20 ? 2 : p >= 10 ? 1 : 0; q[j >> 2] |= (uint8_t)(b << (6 - 2 * (j & 3))); }
                }
                n_stored++;
            }
            if (sbuf.size() >= recsz * 32768) { if (std::fwrite(sbuf.data(), 1, sbuf.size(), spill) != sbuf.size()) die(4, "spill write failed"); sbuf.clear(); }
        }
        re.n_reads = n_stored - re.first_read;
        runs.push_back(re);
    }
    if (!sbuf.empty()) { if (std::fwrite(sbuf.data(), 1, sbuf.size(), spill) != sbuf.size()) die(4, "spill write failed"); sbuf.clear(); }
    std::fflush(spill);
    for (uint64_t w = 0; w < hist.size(); w++) if (hist[w]) corr.add_exact((int64_t)w, hist[w]);
    double t1 = now_s(), c1 = cpu_s();

    // ── pass 2 ────────────────────────────────────────────────────────────────────────────────────────────────────────
    const uint32_t seq_slot = (uint32_t)align_up(std::max<uint32_t>((max_len + 3) / 4, 1), 8);
    Out o; o.f = std::fopen(out_path.c_str(), "wb"); if (!o.f) die(4, "cannot create " + out_path);
    FileHeader H; std::memset(&H, 0, sizeof H);
    o.write(&H, sizeof H);   // placeholder
    o.pad_to(ALIGN);
    H.chunks_offset = o.off;
    std::vector<IndexEntry> cidx, bidx;
    uint64_t plane_bytes[NPLANE] = {0}, plane_raw[NPLANE] = {0}, status_counts[8] = {0};
    std::unordered_map<uint64_t, std::pair<int64_t, uint8_t>> cbres;
    std::rewind(spill);
    std::vector<uint8_t> rbuf(recsz * CHUNK_READS);
    uint64_t first_read = 0;
    for (;;) {
        size_t got = std::fread(rbuf.data(), 1, rbuf.size(), spill);
        if (got == 0) break;
        if (got % recsz) die(4, "spill file truncated");
        const uint32_t n = (uint32_t)(got / recsz);
        Plane pl[NPLANE];
        pl[PL_CB].raw.resize(4 * n); pl[PL_CBST].raw.resize(n); pl[PL_UMI].raw.resize(4 * n); pl[PL_SEQ].raw.resize(4 * n); pl[PL_LEN].raw.resize(n);
        if (has_q) pl[PL_Q].raw.assign((size_t)n * seq_slot, 0);
        std::vector<uint32_t> exc;
        for (uint32_t i = 0; i < n; i++) {
            Rec rec; std::memcpy(&rec, &rbuf[i * recsz], sizeof rec);
            auto it = cbres.find(rec.cbkey);
            if (it == cbres.end()) {
                bool hasN = false; char s[32], outb[32];
                for (int j = 0; j < bcl; j++) { uint8_t c = (rec.cbkey >> (3 * (bcl - 1 - j))) & 7; hasN |= c > 3; s[j] = "ACGTN"[c]; }
                s[bcl] = 0;
                int st = corr.resolve(s, nullptr, outb);
                (void)hasN;
                int64_t idx = st >= 0 ? corr.find(std::string(outb, bcl).c_str()) : -1;
                it = cbres.emplace(rec.cbkey, std::make_pair(idx, (uint8_t)(st + 4))).first;
            }
            const int64_t idx = it->second.first;
            const uint8_t status = (uint8_t)(it->second.second | (rec.bad ? CBST_BAD_UMI : 0));
            const uint32_t cb = idx >= 0 ? (uint32_t)idx : NO_CB;
            std::memcpy(&pl[PL_CB].raw[4 * i], &cb, 4); pl[PL_CBST].raw[i] = status;
            std::memcpy(&pl[PL_UMI].raw[4 * i], &rec.umi, 4); std::memcpy(&pl[PL_SEQ].raw[4 * i], &rec.seq, 4); pl[PL_LEN].raw[i] = rec.len;
            status_counts[it->second.second & 7]++; if (rec.bad) status_counts[7]++;
            for (int j = 0; j < umil; j++) if (rec.umin >> j & 1) exc.push_back(exc_pack(i, EXC_UMI_N, (uint32_t)j));
            if (has_q) std::memcpy(&pl[PL_Q].raw[(size_t)i * seq_slot], &rbuf[i * recsz + sizeof rec], seq_slot);
        }
        pl[PL_EXC].raw.resize(4 * exc.size()); if (!exc.empty()) std::memcpy(pl[PL_EXC].raw.data(), exc.data(), 4 * exc.size());
        // chunk
        std::vector<uint8_t> body; ChunkHeader ch; std::memset(&ch, 0, sizeof ch);
        ch.magic = CHUNK_MAGIC; ch.n_reads = n; ch.n_exc = (uint32_t)exc.size(); ch.first_read = first_read;
        body.resize(sizeof ch);
        for (int p = 0; p < NPLANE; p++) {
            bool rawflag; std::vector<uint8_t> st = pack_plane(pl[p].raw, level, rawflag);
            if (rawflag) ch.flags |= 1u << p;
            while (body.size() % SEC_ALIGN) body.push_back(0);
            ch.sec[p] = {(uint32_t)st.size(), (uint32_t)pl[p].raw.size(), fg2::crc32c(st.data(), st.size()), (uint32_t)body.size()};
            body.insert(body.end(), st.begin(), st.end());
            plane_bytes[p] += st.size(); plane_raw[p] += pl[p].raw.size();
        }
        std::memcpy(body.data(), &ch, sizeof ch);
        o.pad_to(ALIGN);
        cidx.push_back({o.off, (uint32_t)body.size(), n});
        o.write(body.data(), body.size());
        first_read += n;
    }
    std::fclose(spill); std::remove(spill_path.c_str());
    // sequence blocks
    o.pad_to(ALIGN);
    H.blocks_offset = o.off;
    uint64_t seq_bytes[NSEQPLANE] = {0};
    for (uint64_t b0 = 0; b0 < tab.n(); b0 += BLOCK_SEQS) {
        const uint32_t n = (uint32_t)std::min<uint64_t>(BLOCK_SEQS, tab.n() - b0);
        Plane sp[NSEQPLANE];
        sp[SP_SEQ2].raw.resize((size_t)n * seq_slot); sp[SP_SLEN].raw.resize(n);
        std::vector<uint32_t> sexc;
        for (uint32_t i = 0; i < n; i++) {
            const uint64_t id = b0 + i;
            std::memcpy(&sp[SP_SEQ2].raw[(size_t)i * seq_slot], &tab.data[id * SeqTable::SLOT], seq_slot);
            sp[SP_SLEN].raw[i] = tab.len[id];
            if (tab.hasn[id]) { const auto& m = tab.nmask.at((uint32_t)id); for (uint32_t j = 0; j < tab.len[id]; j++) if (m[j >> 3] >> (j & 7) & 1) sexc.push_back(i | (j << 15)); }
        }
        sp[SP_SEXC].raw.resize(4 * sexc.size()); if (!sexc.empty()) std::memcpy(sp[SP_SEXC].raw.data(), sexc.data(), 4 * sexc.size());
        std::vector<uint8_t> body; BlockHeader bh; std::memset(&bh, 0, sizeof bh);
        bh.magic = SEQB_MAGIC; bh.n_seqs = n; bh.n_sexc = (uint32_t)sexc.size(); bh.first_seq = b0;
        body.resize(sizeof bh);
        for (int p = 0; p < NSEQPLANE; p++) {
            bool rawflag; std::vector<uint8_t> st = pack_plane(sp[p].raw, level, rawflag);
            if (rawflag) bh.flags |= 1u << p;
            while (body.size() % SEC_ALIGN) body.push_back(0);
            bh.sec[p] = {(uint32_t)st.size(), (uint32_t)sp[p].raw.size(), fg2::crc32c(st.data(), st.size()), (uint32_t)body.size()};
            body.insert(body.end(), st.begin(), st.end());
            seq_bytes[p] += st.size();
        }
        std::memcpy(body.data(), &bh, sizeof bh);
        o.pad_to(ALIGN);
        bidx.push_back({o.off, (uint32_t)body.size(), n});
        o.write(body.data(), body.size());
    }
    // meta, run table, index, footer
    char meta[1024];
    int ml = std::snprintf(meta, sizeof meta, "{\"tool\":\"singlet-stage\",\"whitelist\":\"%s\",\"runs\":%zu,\"cb_hist_reads\":%llu,\"level\":%d}",
                           spec->whitelist_file.c_str(), runs.size(), (unsigned long long)hist_reads, level);
    std::vector<uint8_t> mz(ZSTD_compressBound(ml)); size_t mzn = ZSTD_compress(mz.data(), mz.size(), meta, ml, level);
    o.pad_to(ALIGN); H.meta_offset = o.off; H.meta_comp_size = (uint32_t)mzn; H.meta_raw_size = (uint32_t)ml; o.write(mz.data(), mzn);
    o.pad_to(16); H.runs_offset = o.off; H.n_runs = (uint32_t)runs.size(); o.write(runs.data(), runs.size() * sizeof(RunEntry));
    o.pad_to(16); H.index_offset = o.off;
    std::vector<IndexEntry> allidx = cidx; allidx.insert(allidx.end(), bidx.begin(), bidx.end());
    o.write(allidx.data(), allidx.size() * sizeof(IndexEntry));
    Footer F{FOOT_MAGIC, (uint32_t)cidx.size(), (uint32_t)bidx.size(), fg2::crc32c(allidx.data(), allidx.size() * sizeof(IndexEntry)), H.index_offset};
    o.write(&F, sizeof F);
    // final header
    H.magic = FILE_MAGIC; H.version = VERSION; H.header_size = sizeof H; H.codec = CODEC_ZSTD; H.codec_level = (uint8_t)level;
    H.flags = F_CRC | F_CB_FIXED | (has_q ? F_QPLANE : 0);
    H.chunk_reads = CHUNK_READS; H.block_seqs = BLOCK_SEQS; H.n_reads = n_stored; H.n_reads_in = n_in; H.n_dropped = n_dropped; H.n_unique = tab.n();
    H.n_chunks = (uint32_t)cidx.size(); H.n_blocks = (uint32_t)bidx.size(); H.seq_slot = (uint16_t)seq_slot; H.r2_max_len = (uint16_t)max_len;
    H.r2_clip5p = 0; H.protocol_id = protocol_id; H.cb_len = (uint8_t)bcl; H.cb_start = (uint8_t)bc0; H.umi_len = (uint8_t)umil; H.umi_start = (uint8_t)umi0;
    H.r1_len = (uint8_t)std::max(bc0 + bcl, umi0 + umil);
    if (has_q) { H.q_edges[0] = 2; H.q_edges[1] = 11; H.q_edges[2] = 25; H.q_edges[3] = 37; H.q_bins = 4; }
    H.wl_entries = (uint32_t)corr.size(); std::memcpy(H.wl_sha256, wl_sha, 32);
    std::strncpy(H.wl_name, spec->whitelist_file.c_str(), sizeof H.wl_name - 1);
    H.cb_hist_reads = hist_reads; std::memcpy(H.status_counts, status_counts, sizeof status_counts);
    H.source_bytes = src_bytes;
    H.header_crc32c = fg2::crc32c(&H, offsetof(FileHeader, header_crc32c));
    std::fflush(o.f); std::fseek(o.f, 0, SEEK_SET); std::fwrite(&H, 1, sizeof H, o.f); std::fclose(o.f);
    double t2 = now_s(), c2 = cpu_s();

    uint64_t total_planes = 0; for (int p = 0; p < NPLANE; p++) total_planes += plane_bytes[p];
    uint64_t total_seq = 0; for (int p = 0; p < NSEQPLANE; p++) total_seq += seq_bytes[p];
    struct stat fst; stat(out_path.c_str(), &fst);
    const double M = n_stored / 1e6;
    char js[4096];
    std::snprintf(js, sizeof js,
        "{\"n_reads\":%llu,\"n_in\":%llu,\"n_dropped\":%llu,\"n_unique\":%llu,\"unique_fraction\":%.4f,\"protocol_id\":%u,\"whitelist\":\"%s\",\"cb_hist_reads\":%llu,"
        "\"seq_slot\":%u,\"max_len\":%u,\"has_q\":%d,\"file_bytes\":%lld,\"bytes_per_read\":%.3f,\"read_plane_bytes\":{\"cb\":%llu,\"cbst\":%llu,\"umi\":%llu,\"seq\":%llu,\"len\":%llu,\"q\":%llu,\"exc\":%llu},"
        "\"seq_table_bytes\":{\"seq2\":%llu,\"slen\":%llu,\"sexc\":%llu},\"status_counts\":[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu],"
        "\"pass1_s\":%.2f,\"pass2_s\":%.2f,\"wall_s\":%.2f,\"cpu_s\":%.2f,\"cpu_pass1_s\":%.2f,\"cpu_pass2_s\":%.2f,\"core_s_per_Mreads\":%.3f,\"table_mem_mb\":%.0f}",
        (unsigned long long)n_stored, (unsigned long long)n_in, (unsigned long long)n_dropped, (unsigned long long)tab.n(), n_stored ? (double)tab.n() / n_stored : 0.0,
        protocol_id, spec->whitelist_file.c_str(), (unsigned long long)hist_reads, seq_slot, max_len, (int)has_q, (long long)fst.st_size, n_stored ? (double)fst.st_size / n_stored : 0.0,
        (unsigned long long)plane_bytes[PL_CB], (unsigned long long)plane_bytes[PL_CBST], (unsigned long long)plane_bytes[PL_UMI], (unsigned long long)plane_bytes[PL_SEQ],
        (unsigned long long)plane_bytes[PL_LEN], (unsigned long long)plane_bytes[PL_Q], (unsigned long long)plane_bytes[PL_EXC],
        (unsigned long long)seq_bytes[0], (unsigned long long)seq_bytes[1], (unsigned long long)seq_bytes[2],
        (unsigned long long)status_counts[0], (unsigned long long)status_counts[1], (unsigned long long)status_counts[2], (unsigned long long)status_counts[3],
        (unsigned long long)status_counts[4], (unsigned long long)status_counts[5], (unsigned long long)status_counts[6], (unsigned long long)status_counts[7],
        t1 - t0, t2 - t1, t2 - t0, c2 - c0, c1 - c0, c2 - c1, M > 0 ? (c2 - c0) / M : 0.0, (tab.data.size() + tab.ht.size() * 4.0) / 1e6);
    std::puts(js);
    if (!json_path.empty()) { std::ofstream jf(json_path); jf << js << "\n"; }
    (void)total_planes; (void)total_seq;
    if (do_verify) return stage_verify::verify_file(out_path, inputs, corr, bcl, bc0, umil, umi0, has_q);
    return 0;
}
