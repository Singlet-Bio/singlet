// stage::Encoder: reads of one sample -> .1fg2 (include/singlet/fq/fg2_format.h). Header-only; used by singlet-stage (from .1fq) and by `singlet download --stage-out`
// (straight from the SRA, no .1fq). The read-level logic is one code path for both:
//   begin_run / add_read ... end_run   (pass 1: exact-barcode histogram, distinct-cDNA table, spill of the per-read records)
//   finish(out_path)                   (pass 2: R11 barcode correction with the final histogram, planes, zstd, CRC32C, table blocks, run table, index)
// Protocol, geometry and whitelist come from the engine's own tables (known_protocols), the correction is singlet::CbCorrector.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <zstd.h>

#include <singlet/fq/protocol.h>
#include <singlet/fq/types.h>
#include <singlet/pileup/cb_correct.h>
#include <singlet/fq/fg2_format.h>

namespace stage {
using namespace fg2;

[[noreturn]] inline void die(int code, const std::string& m) { std::fprintf(stderr, "[singlet-stage] %s\n", m.c_str()); std::exit(code); }
inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline double cpu_s() {
    rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

// Distinct cDNA sequences: open addressing over packed 2-bit sequences (N coded as A; the N positions are part of the key).
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
    uint32_t get_or_add(const uint8_t* slot, uint8_t l, const uint8_t* nm) {
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
struct Rec { uint64_t cbkey; uint32_t umi; uint32_t seq; uint16_t umin; uint8_t len; uint8_t bad; };   // 20 bytes; + SLOT quality bytes if the sample has them
#pragma pack(pop)

struct Out {
    FILE* f; uint64_t off = 0;
    void write(const void* p, size_t n) { if (n && std::fwrite(p, 1, n, f) != n) die(4, "write failed"); off += n; }
    void pad_to(uint64_t a) { static const uint8_t z[4096] = {0}; uint64_t t = align_up(off, a); while (off < t) write(z, std::min<uint64_t>(4096, t - off)); }
};

inline std::vector<uint8_t> pack_plane(const std::vector<uint8_t>& raw, int level, bool& rawflag) {
    rawflag = false;
    if (raw.empty()) return {};
    std::vector<uint8_t> out(ZSTD_compressBound(raw.size()));
    size_t c = ZSTD_compress(out.data(), out.size(), raw.data(), raw.size(), level);
    if (ZSTD_isError(c) || c >= raw.size()) { rawflag = true; return raw; }
    out.resize(c);
    return out;
}

// A "part" is one run's pass-1 state on disk, so a multi-run sample can be built run by run (`singlet download --stage-out F --stage-part`) and merged
// (`singlet-stage --merge-parts`) into ONE .1fg2 with one exact-barcode histogram and one distinct-cDNA table. Layout (little endian):
//   PartHdr | RunEntry[n_runs] | hist pairs (u64 whitelist index, u64 count) | len[n_unique] | hasn[n_unique] | n_nmask u64 + (u32 id, 32 B mask)* | data[n_unique*SLOT] | records[n_stored*recsz]
#pragma pack(push, 1)
struct PartHdr {
    uint32_t magic, version;
    uint8_t protocol_id, bcl, bc0, umil, umi0, has_q; uint16_t pad;
    uint64_t n_stored, n_in, n_dropped, hist_reads, n_unique, n_hist_pairs, src_bytes, recsz;
    uint32_t max_len, n_runs;
    char wl_name[64];
    uint64_t meta_len;   // bytes of the source metadata JSON that follow the header
};
#pragma pack(pop)
static constexpr uint32_t PART_MAGIC = 0x54505331u;   // "1SPT"

class Encoder {
public:
    singlet::CbCorrector corr;
    uint8_t protocol_id = 0;
    int bcl = 0, bc0 = 0, umil = 0, umi0 = 0;
    bool has_q = false;
    std::string wl_name;
    uint64_t n_in = 0, n_dropped = 0, n_stored = 0, hist_reads = 0, src_bytes = 0;
    double t_start = 0, c_start = 0;

    // protocol_id -> universal single whitelist in R1, or die(6) (the caller then stays on the .1fq path)
    void init(uint8_t pid, int bc_len, int bc_off, int umi_len, int umi_off, bool with_q, const std::vector<std::string>& wl_dirs,
              const std::string& scratch_dir) {
        protocol_id = pid; bcl = bc_len; bc0 = bc_off; umil = umi_len; umi0 = umi_off; has_q = with_q; scratch = scratch_dir;
        const singlet::fq::CandidateSpec* spec = nullptr;
        for (const auto& s : singlet::fq::known_protocols()) if (s.protocol_id == pid) { spec = &s; break; }
        if (!spec || spec->whitelist_file.empty() || !spec->per_seg_whitelist_files.empty() || spec->barcode_in_r2)
            die(6, "protocol " + std::to_string(pid) + " has no single universal whitelist in R1: not staged (use the .1fq path)");
        wl_name = spec->whitelist_file;
        for (const auto& d : wl_dirs) { std::string p = d + "/" + wl_name; if (std::ifstream(p).good()) { wl_path = p; break; } }
        if (wl_path.empty()) die(6, "whitelist " + wl_name + " not found in the whitelist dirs");
        if (!corr.load_whitelist(wl_path)) die(6, "cannot load whitelist " + wl_path);
        if (bcl <= 0 || bcl > 16 || bcl != corr.length()) die(6, "barcode length " + std::to_string(bcl) + " does not match the whitelist (" + std::to_string(corr.length()) + ")");
        if (umil <= 0 || umil > 16) die(6, "UMI length " + std::to_string(umil) + " not supported (1..16)");
        hist.assign(corr.size(), 0);
        recsz = sizeof(Rec) + (has_q ? SeqTable::SLOT : 0);
        spill_path = scratch + "/s9-stage-spill-" + std::to_string(getpid());
        spill = std::fopen(spill_path.c_str(), "wb+");
        if (!spill) die(4, "cannot create spill file " + spill_path);
        sbuf.reserve(recsz * 65536);
        t_start = now_s(); c_start = cpu_s();
        ready = true;
    }
    bool initialised() const { return ready; }

    // The source .1fq's metadata JSON (Reader::read_metadata() / the SRA encoder's finish() argument), kept verbatim in the .1fg2: the engine configures itself from it.
    void set_source_metadata(const std::string& m) { if (source_meta.empty()) source_meta = m; }

    void begin_run(const std::string& accession, uint64_t raw_bytes) {
        RunEntry re; std::memset(&re, 0, sizeof re);
        std::strncpy(re.accession, accession.c_str(), sizeof re.accession - 1);
        re.first_read = n_stored; re.raw_bytes = raw_bytes;
        runs.push_back(re); src_bytes += raw_bytes;
    }
    void end_run() { runs.back().n_reads = n_stored - runs.back().first_read; }
    void note_dropped() { n_in++; n_dropped++; }

    // One read. bc[bcl] and umi[umil] byte-numeric (0-3, 4 = N), r2[r2l] byte-numeric, qual = raw Phred per base of r2 (null: all zero).
    void add_read(const uint8_t* bc, const uint8_t* umi_in, const uint8_t* r2, uint16_t r2l, const uint8_t* qual) {
        n_in++;
        if (r2l > MAX_SEQ_LEN) die(3, "cDNA longer than 255 nt (" + std::to_string(r2l) + "): not stageable");
        uint64_t cbkey = 0; bool cbN = false;
        for (int j = 0; j < bcl; j++) { uint8_t c = std::min<uint8_t>(bc[j], 4); cbN |= c > 3; cbkey = (cbkey << 3) | c; }
        uint32_t umi = 0; uint16_t umin = 0; bool homo = true;
        for (int j = 0; j < umil; j++) { uint8_t c = umi_in[j]; if (c > 3) { umin |= (uint16_t)(1u << j); c = 0; } umi = (umi << 2) | c; if (umi_in[j] != umi_in[0]) homo = false; }
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
        uint8_t slot[SeqTable::SLOT] = {0}; uint8_t nm[SeqTable::NMB] = {0}; bool anyN = false;
        for (uint16_t j = 0; j < r2l; j++) { uint8_t c = r2[j]; if (c > 3) { anyN = true; nm[j >> 3] |= (uint8_t)(1u << (j & 7)); c = 0; } slot[j >> 2] |= (uint8_t)(c << (6 - 2 * (j & 3))); }
        const uint32_t sid = tab.get_or_add(slot, (uint8_t)r2l, anyN ? nm : nullptr);
        max_len = std::max<uint32_t>(max_len, r2l);
        Rec rec{cbkey, umi, sid, umin, (uint8_t)r2l, (uint8_t)bad};
        const size_t at = sbuf.size(); sbuf.resize(at + recsz);
        std::memcpy(&sbuf[at], &rec, sizeof rec);
        if (has_q) {
            uint8_t* q = &sbuf[at + sizeof rec]; std::memset(q, 0, SeqTable::SLOT);
            for (uint16_t j = 0; j < r2l; j++) {
                uint8_t p = qual ? qual[j] : 0, b = p <= 9 ? 0 : p <= 19 ? 1 : p <= 29 ? 2 : 3;   // singlet::fq::qual::phred_to_ilmn4
                q[j >> 2] |= (uint8_t)(b << (6 - 2 * (j & 3)));
            }
        }
        n_stored++;
        if (sbuf.size() >= recsz * 32768) flush_spill();
    }

    // Pass 2 and the file. Returns the JSON report (one line).
    std::string finish(const std::string& out_path, int level, double extra_cpu_s = 0, double extra_wall_s = 0) {
        flush_spill(); std::fflush(spill);
        for (uint64_t w = 0; w < hist.size(); w++) if (hist[w]) corr.add_exact((int64_t)w, hist[w]);
        double t1 = now_s(), c1 = cpu_s();
        fg2::Sha256 sha; { std::ifstream wf(wl_path, std::ios::binary); char b[1 << 16]; while (wf.read(b, sizeof b) || wf.gcount()) sha.update((const uint8_t*)b, (size_t)wf.gcount()); }
        uint8_t wl_sha[32]; sha.finish(wl_sha);
        const uint32_t seq_slot = (uint32_t)align_up(std::max<uint32_t>((max_len + 3) / 4, 1), 8);
        Out o; o.f = std::fopen(out_path.c_str(), "wb"); if (!o.f) die(4, "cannot create " + out_path);
        FileHeader H; std::memset(&H, 0, sizeof H);
        o.write(&H, sizeof H); o.pad_to(ALIGN);
        H.chunks_offset = o.off;
        std::vector<IndexEntry> cidx, bidx;
        uint64_t plane_bytes[NPLANE] = {0}, status_counts[8] = {0}, rawcb_entries = 0;
        std::unordered_map<uint64_t, std::pair<int64_t, uint8_t>> cbres;
        std::rewind(spill);
        std::vector<uint8_t> rbuf(recsz * CHUNK_READS);
        uint64_t first_read = 0;
        for (;;) {
            size_t got = std::fread(rbuf.data(), 1, rbuf.size(), spill);
            if (got == 0) break;
            if (got % recsz) die(4, "spill file truncated");
            const uint32_t n = (uint32_t)(got / recsz);
            std::vector<uint8_t> pl[NPLANE];
            pl[PL_CB].resize(4 * n); pl[PL_CBST].resize(n); pl[PL_UMI].resize(4 * n); pl[PL_SEQ].resize(4 * n); pl[PL_LEN].resize(n);
            if (has_q) pl[PL_Q].assign((size_t)n * seq_slot, 0);
            std::vector<uint32_t> exc, rawcb;
            for (uint32_t i = 0; i < n; i++) {
                Rec rec; std::memcpy(&rec, &rbuf[i * recsz], sizeof rec);
                auto it = cbres.find(rec.cbkey);
                if (it == cbres.end()) {
                    char s[32], outb[32];
                    for (int j = 0; j < bcl; j++) s[j] = "ACGTN"[(rec.cbkey >> (3 * (bcl - 1 - j))) & 7];
                    s[bcl] = 0;
                    int st = corr.resolve(s, nullptr, outb);
                    int64_t idx = st >= 0 ? corr.find(std::string(outb, bcl).c_str()) : -1;
                    it = cbres.emplace(rec.cbkey, std::make_pair(idx, (uint8_t)(st + 4))).first;
                }
                const int64_t idx = it->second.first;
                const uint8_t status = (uint8_t)(it->second.second | (rec.bad ? CBST_BAD_UMI : 0));
                const uint32_t cb = idx >= 0 ? (uint32_t)idx : NO_CB;
                std::memcpy(&pl[PL_CB][4 * i], &cb, 4); pl[PL_CBST][i] = status;
                std::memcpy(&pl[PL_UMI][4 * i], &rec.umi, 4); std::memcpy(&pl[PL_SEQ][4 * i], &rec.seq, 4); pl[PL_LEN][i] = rec.len;
                status_counts[it->second.second & 7]++; if (rec.bad) status_counts[7]++;
                if ((it->second.second & 7) != CBST_EXACT) {   // raw barcode of every read that was not an exact whitelist hit
                    uint32_t word = 0, nmask = 0;
                    for (int j = 0; j < bcl; j++) { uint32_t c = (uint32_t)((rec.cbkey >> (3 * (bcl - 1 - j))) & 7); if (c > 3) { nmask |= 1u << j; c = 0; } word = (word << 2) | c; }
                    rawcb.push_back(word); rawcb.push_back(i | (nmask << 15));
                }
                for (int j = 0; j < umil; j++) if ((rec.umin >> j) & 1) exc.push_back(exc_pack(i, EXC_UMI_N, (uint32_t)j));
                if (has_q) std::memcpy(&pl[PL_Q][(size_t)i * seq_slot], &rbuf[i * recsz + sizeof rec], seq_slot);
            }
            pl[PL_EXC].resize(4 * exc.size()); if (!exc.empty()) std::memcpy(pl[PL_EXC].data(), exc.data(), 4 * exc.size());
            pl[PL_RAWCB].resize(4 * rawcb.size()); if (!rawcb.empty()) std::memcpy(pl[PL_RAWCB].data(), rawcb.data(), 4 * rawcb.size());
            rawcb_entries += rawcb.size() / 2;
            std::vector<uint8_t> body; ChunkHeader ch; std::memset(&ch, 0, sizeof ch);
            ch.magic = CHUNK_MAGIC; ch.n_reads = n; ch.n_exc = (uint32_t)exc.size(); ch.n_rawcb = (uint32_t)(rawcb.size() / 2); ch.first_read = first_read;
            body.resize(sizeof ch);
            for (int p = 0; p < NPLANE; p++) {
                bool rawflag; std::vector<uint8_t> st = pack_plane(pl[p], level, rawflag);
                if (rawflag) ch.flags |= 1u << p;
                while (body.size() % SEC_ALIGN) body.push_back(0);
                ch.sec[p] = {(uint32_t)st.size(), (uint32_t)pl[p].size(), fg2::crc32c(st.data(), st.size()), (uint32_t)body.size()};
                body.insert(body.end(), st.begin(), st.end());
                plane_bytes[p] += st.size();
            }
            std::memcpy(body.data(), &ch, sizeof ch);
            o.pad_to(ALIGN); cidx.push_back({o.off, (uint32_t)body.size(), n}); o.write(body.data(), body.size());
            first_read += n;
        }
        std::fclose(spill); spill = nullptr; std::remove(spill_path.c_str());
        o.pad_to(ALIGN); H.blocks_offset = o.off;
        uint64_t seq_bytes[NSEQPLANE] = {0};
        for (uint64_t b0 = 0; b0 < tab.n(); b0 += BLOCK_SEQS) {
            const uint32_t n = (uint32_t)std::min<uint64_t>(BLOCK_SEQS, tab.n() - b0);
            std::vector<uint8_t> sp[NSEQPLANE];
            sp[SP_SEQ2].resize((size_t)n * seq_slot); sp[SP_SLEN].resize(n);
            std::vector<uint32_t> sexc;
            for (uint32_t i = 0; i < n; i++) {
                const uint64_t id = b0 + i;
                std::memcpy(&sp[SP_SEQ2][(size_t)i * seq_slot], &tab.data[id * SeqTable::SLOT], seq_slot);
                sp[SP_SLEN][i] = tab.len[id];
                if (tab.hasn[id]) { const auto& m = tab.nmask.at((uint32_t)id); for (uint32_t j = 0; j < tab.len[id]; j++) if ((m[j >> 3] >> (j & 7)) & 1) sexc.push_back(i | (j << 15)); }
            }
            sp[SP_SEXC].resize(4 * sexc.size()); if (!sexc.empty()) std::memcpy(sp[SP_SEXC].data(), sexc.data(), 4 * sexc.size());
            std::vector<uint8_t> body; BlockHeader bh; std::memset(&bh, 0, sizeof bh);
            bh.magic = SEQB_MAGIC; bh.n_seqs = n; bh.n_sexc = (uint32_t)sexc.size(); bh.first_seq = b0;
            body.resize(sizeof bh);
            for (int p = 0; p < NSEQPLANE; p++) {
                bool rawflag; std::vector<uint8_t> st = pack_plane(sp[p], level, rawflag);
                if (rawflag) bh.flags |= 1u << p;
                while (body.size() % SEC_ALIGN) body.push_back(0);
                bh.sec[p] = {(uint32_t)st.size(), (uint32_t)sp[p].size(), fg2::crc32c(st.data(), st.size()), (uint32_t)body.size()};
                body.insert(body.end(), st.begin(), st.end());
                seq_bytes[p] += st.size();
            }
            std::memcpy(body.data(), &bh, sizeof bh);
            o.pad_to(ALIGN); bidx.push_back({o.off, (uint32_t)body.size(), n}); o.write(body.data(), body.size());
        }
        std::string meta;
        {
            char key[512];
            std::snprintf(key, sizeof key, "\"s9_stage\":{\"tool\":\"singlet-stage\",\"whitelist\":\"%s\",\"runs\":%zu,\"cb_hist_reads\":%llu,\"level\":%d}",
                          wl_name.c_str(), runs.size(), (unsigned long long)hist_reads, level);
            // the source .1fq's metadata JSON verbatim (an object), our own key appended; without it just ours
            std::string src = source_meta;
            while (!src.empty() && (src.back() == ' ' || src.back() == '\n' || src.back() == '\r' || src.back() == '\t')) src.pop_back();
            size_t lb = src.find_first_not_of(" \n\r\t");
            if (lb != std::string::npos && src[lb] == '{' && src.back() == '}') {
                std::string inner = src.substr(0, src.size() - 1);
                bool empty_obj = inner.find_first_not_of(" \n\r\t{") == std::string::npos;
                meta = inner + (empty_obj ? "" : ",") + key + "}";
            } else {
                meta = std::string("{") + key + "}";
            }
        }
        int ml = (int)meta.size();
        std::vector<uint8_t> mz(ZSTD_compressBound(ml)); size_t mzn = ZSTD_compress(mz.data(), mz.size(), meta.data(), ml, level);
        o.pad_to(ALIGN); H.meta_offset = o.off; H.meta_comp_size = (uint32_t)mzn; H.meta_raw_size = (uint32_t)ml; o.write(mz.data(), mzn);
        o.pad_to(16); H.runs_offset = o.off; H.n_runs = (uint32_t)runs.size(); o.write(runs.data(), runs.size() * sizeof(RunEntry));
        o.pad_to(16); H.index_offset = o.off;
        std::vector<IndexEntry> allidx = cidx; allidx.insert(allidx.end(), bidx.begin(), bidx.end());
        o.write(allidx.data(), allidx.size() * sizeof(IndexEntry));
        Footer F{FOOT_MAGIC, (uint32_t)cidx.size(), (uint32_t)bidx.size(), fg2::crc32c(allidx.data(), allidx.size() * sizeof(IndexEntry)), H.index_offset};
        o.write(&F, sizeof F);
        H.magic = FILE_MAGIC; H.version = VERSION; H.header_size = sizeof H; H.codec = CODEC_ZSTD; H.codec_level = (uint8_t)level;
        H.flags = F_CRC | F_CB_FIXED | (has_q ? F_QPLANE : 0);
        H.chunk_reads = CHUNK_READS; H.block_seqs = BLOCK_SEQS; H.n_reads = n_stored; H.n_reads_in = n_in; H.n_dropped = n_dropped; H.n_unique = tab.n();
        H.n_chunks = (uint32_t)cidx.size(); H.n_blocks = (uint32_t)bidx.size(); H.seq_slot = (uint16_t)seq_slot; H.r2_max_len = (uint16_t)max_len;
        H.r2_clip5p = 0; H.protocol_id = protocol_id; H.cb_len = (uint8_t)bcl; H.cb_start = (uint8_t)bc0; H.umi_len = (uint8_t)umil; H.umi_start = (uint8_t)umi0;
        H.r1_len = (uint8_t)std::max(bc0 + bcl, umi0 + umil);
        if (has_q) { H.q_edges[0] = 2; H.q_edges[1] = 11; H.q_edges[2] = 25; H.q_edges[3] = 37; H.q_bins = 4; }
        H.wl_entries = (uint32_t)corr.size(); std::memcpy(H.wl_sha256, wl_sha, 32);
        std::strncpy(H.wl_name, wl_name.c_str(), sizeof H.wl_name - 1);
        H.cb_hist_reads = hist_reads; std::memcpy(H.status_counts, status_counts, sizeof status_counts);
        H.source_bytes = src_bytes;
        H.header_crc32c = fg2::crc32c(&H, offsetof(FileHeader, header_crc32c));
        std::fflush(o.f); std::fseek(o.f, 0, SEEK_SET); std::fwrite(&H, 1, sizeof H, o.f); std::fclose(o.f);
        double t2 = now_s(), c2 = cpu_s();
        struct stat fst; stat(out_path.c_str(), &fst);
        const double M = n_stored / 1e6;
        char js[4096];
        std::snprintf(js, sizeof js,
            "{\"n_reads\":%llu,\"n_in\":%llu,\"n_dropped\":%llu,\"n_unique\":%llu,\"unique_fraction\":%.4f,\"protocol_id\":%u,\"whitelist\":\"%s\",\"cb_hist_reads\":%llu,"
            "\"seq_slot\":%u,\"max_len\":%u,\"has_q\":%d,\"file_bytes\":%lld,\"bytes_per_read\":%.3f,\"read_plane_bytes\":{\"cb\":%llu,\"cbst\":%llu,\"umi\":%llu,\"seq\":%llu,\"len\":%llu,\"q\":%llu,\"exc\":%llu,\"rawcb\":%llu},\"rawcb_entries\":%llu,"
            "\"seq_table_bytes\":{\"seq2\":%llu,\"slen\":%llu,\"sexc\":%llu},\"status_counts\":[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu],"
            "\"encoder_wall_s\":%.2f,\"encoder_cpu_s\":%.2f,\"finish_cpu_s\":%.2f,\"core_s_per_Mreads\":%.3f,\"table_mem_mb\":%.0f}",
            (unsigned long long)n_stored, (unsigned long long)n_in, (unsigned long long)n_dropped, (unsigned long long)tab.n(), n_stored ? (double)tab.n() / n_stored : 0.0,
            protocol_id, wl_name.c_str(), (unsigned long long)hist_reads, seq_slot, max_len, (int)has_q, (long long)fst.st_size, n_stored ? (double)fst.st_size / n_stored : 0.0,
            (unsigned long long)plane_bytes[PL_CB], (unsigned long long)plane_bytes[PL_CBST], (unsigned long long)plane_bytes[PL_UMI], (unsigned long long)plane_bytes[PL_SEQ],
            (unsigned long long)plane_bytes[PL_LEN], (unsigned long long)plane_bytes[PL_Q], (unsigned long long)plane_bytes[PL_EXC], (unsigned long long)plane_bytes[PL_RAWCB], (unsigned long long)rawcb_entries,
            (unsigned long long)seq_bytes[0], (unsigned long long)seq_bytes[1], (unsigned long long)seq_bytes[2],
            (unsigned long long)status_counts[0], (unsigned long long)status_counts[1], (unsigned long long)status_counts[2], (unsigned long long)status_counts[3],
            (unsigned long long)status_counts[4], (unsigned long long)status_counts[5], (unsigned long long)status_counts[6], (unsigned long long)status_counts[7],
            t2 - t_start + extra_wall_s, c2 - c_start + extra_cpu_s, c2 - c1, M > 0 ? (c2 - c_start + extra_cpu_s) / M : 0.0, (tab.data.size() + tab.ht.size() * 4.0) / 1e6);
        return js;
    }

    // Write this run's state as a part file (instead of finish()).
    void write_part(const std::string& path) {
        flush_spill(); std::fflush(spill);
        FILE* f = std::fopen(path.c_str(), "wb"); if (!f) die(4, "cannot create " + path);
        PartHdr h; std::memset(&h, 0, sizeof h);
        h.magic = PART_MAGIC; h.version = 1; h.protocol_id = protocol_id; h.bcl = (uint8_t)bcl; h.bc0 = (uint8_t)bc0; h.umil = (uint8_t)umil; h.umi0 = (uint8_t)umi0;
        h.has_q = has_q; h.n_stored = n_stored; h.n_in = n_in; h.n_dropped = n_dropped; h.hist_reads = hist_reads; h.n_unique = tab.n(); h.src_bytes = src_bytes; h.recsz = recsz;
        h.max_len = max_len; h.n_runs = (uint32_t)runs.size(); std::strncpy(h.wl_name, wl_name.c_str(), sizeof h.wl_name - 1); h.meta_len = source_meta.size();
        std::vector<std::pair<uint64_t, uint64_t>> hp; for (uint64_t w = 0; w < hist.size(); w++) if (hist[w]) hp.push_back({w, hist[w]});
        h.n_hist_pairs = hp.size();
        auto W = [&](const void* p, size_t n) { if (n && std::fwrite(p, 1, n, f) != n) die(4, "part write failed"); };
        W(&h, sizeof h); W(source_meta.data(), source_meta.size()); W(runs.data(), runs.size() * sizeof(RunEntry)); W(hp.data(), hp.size() * 16);
        W(tab.len.data(), tab.len.size()); W(tab.hasn.data(), tab.hasn.size());
        uint64_t nn = tab.nmask.size(); W(&nn, 8);
        for (const auto& kv : tab.nmask) { W(&kv.first, 4); W(kv.second.data(), SeqTable::NMB); }
        W(tab.data.data(), tab.data.size());
        std::rewind(spill);
        std::vector<uint8_t> buf(1 << 22);
        for (size_t got; (got = std::fread(buf.data(), 1, buf.size(), spill)) > 0;) W(buf.data(), got);
        std::fclose(f); std::fclose(spill); spill = nullptr; std::remove(spill_path.c_str());
    }

    static PartHdr read_part_header(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb"); if (!f) die(4, "cannot open part " + path);
        PartHdr h; if (std::fread(&h, sizeof h, 1, f) != 1 || h.magic != PART_MAGIC) die(4, "not a part file: " + path);
        std::fclose(f); return h;
    }

    // Merge a part (an earlier run of the same sample) into this encoder as if its reads had been added here: histogram summed, sequences re-keyed into
    // this table (ids remapped in the spilled records), run entries appended with their first_read shifted.
    void merge_part(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb"); if (!f) die(4, "cannot open part " + path);
        auto R = [&](void* p, size_t n) { if (n && std::fread(p, 1, n, f) != n) die(4, "part truncated: " + path); };
        PartHdr h; R(&h, sizeof h);
        if (h.magic != PART_MAGIC || h.protocol_id != protocol_id || h.bcl != bcl || h.bc0 != bc0 || h.umil != umil || h.umi0 != umi0 || (bool)h.has_q != has_q ||
            h.recsz != recsz || wl_name != std::string(h.wl_name))
            die(6, "part " + path + " disagrees with the sample's protocol/geometry/quality mode");
        { std::string m(h.meta_len, '\0'); if (h.meta_len) R(&m[0], h.meta_len); set_source_metadata(m); }
        std::vector<RunEntry> pr(h.n_runs); R(pr.data(), pr.size() * sizeof(RunEntry));
        for (auto& r : pr) { r.first_read += n_stored; runs.push_back(r); }
        std::vector<std::pair<uint64_t, uint64_t>> hp(h.n_hist_pairs); R(hp.data(), hp.size() * 16);
        for (const auto& kv : hp) { if (kv.first >= hist.size()) die(4, "part histogram index out of range"); hist[kv.first] += kv.second; }
        hist_reads += h.hist_reads;
        std::vector<uint8_t> plen(h.n_unique), phasn(h.n_unique); R(plen.data(), plen.size()); R(phasn.data(), phasn.size());
        uint64_t nn; R(&nn, 8);
        std::unordered_map<uint32_t, std::array<uint8_t, SeqTable::NMB>> pnm;
        for (uint64_t i = 0; i < nn; i++) { uint32_t id; std::array<uint8_t, SeqTable::NMB> m; R(&id, 4); R(m.data(), SeqTable::NMB); pnm.emplace(id, m); }
        std::vector<uint8_t> pdata(h.n_unique * SeqTable::SLOT); R(pdata.data(), pdata.size());
        std::vector<uint32_t> remap(h.n_unique);
        for (uint64_t id = 0; id < h.n_unique; id++)
            remap[id] = tab.get_or_add(&pdata[id * SeqTable::SLOT], plen[id], phasn[id] ? pnm.at((uint32_t)id).data() : nullptr);
        std::vector<uint8_t> rb(recsz * 32768);
        for (uint64_t left = h.n_stored; left > 0;) {
            const size_t k = (size_t)std::min<uint64_t>(left, 32768);
            R(rb.data(), k * recsz);
            for (size_t i = 0; i < k; i++) { Rec rec; std::memcpy(&rec, &rb[i * recsz], sizeof rec); rec.seq = remap[rec.seq]; std::memcpy(&rb[i * recsz], &rec, sizeof rec); }
            sbuf.insert(sbuf.end(), rb.begin(), rb.begin() + k * recsz);
            if (sbuf.size() >= recsz * 32768) flush_spill();
            left -= k;
        }
        std::fclose(f);
        n_stored += h.n_stored; n_in += h.n_in; n_dropped += h.n_dropped; src_bytes += h.src_bytes; max_len = std::max<uint32_t>(max_len, h.max_len);
    }

private:
    void flush_spill() {
        if (sbuf.empty()) return;
        if (std::fwrite(sbuf.data(), 1, sbuf.size(), spill) != sbuf.size()) die(4, "spill write failed");
        sbuf.clear();
    }
    bool ready = false;
    std::string scratch, wl_path, spill_path;
    FILE* spill = nullptr;
    SeqTable tab;
    std::vector<uint64_t> hist;
    std::unordered_map<uint64_t, int64_t> cbfind;
    std::vector<RunEntry> runs;
    uint32_t max_len = 0;
    size_t recsz = 0;
    std::vector<uint8_t> sbuf;
    std::string source_meta;
};
}  // namespace stage
