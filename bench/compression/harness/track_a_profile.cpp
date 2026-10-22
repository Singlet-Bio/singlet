// Track A deliverable 5: where do the milliseconds of the production reader go?
//
//   track_a_profile [--reps 5] [--out results/track_a_profile.tsv] file.1pz ...
//
// An instrumented copy of singlet::pz::read_1pz_buffer (pz_reader.h). The logic is a line-for-line copy; the
// only additions are TSC timestamps between stages. Per-column stages (varint parse + gap reconstruction, sort,
// emit) are timed per column; the cost of the timestamp pair itself is measured and subtracted.
// The uninstrumented read_1pz_buffer is timed as well and compared with the sum of stages.
#include "common.h"
#include "track_a.h"
#include <cmath>
#include <iostream>
#include <x86intrin.h>

using namespace cs;
using namespace singlet::pz;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

enum Stage {
    S_FILE_READ, S_VALIDATE_CRC, S_PERM_ZSTD, S_PERM_CHECK, S_PTR, S_META_ZSTD, S_META_PARSE, S_INDPTR, S_OUT_ALLOC,
    S_CH_ZSTD, S_CH_CRC, S_CH_BITMAP, S_CH_BITPLANE, S_CH_UNSPLIT, S_COL_PARSE, S_COL_SORT, S_COL_EMIT, S_CH_OTHER, S_COUNT
};
static const char* kStage[] = {
    "file read (ifstream)", "validate_buffer (whole-file CRC32)", "perm: zstd decode", "perm: permutation check", "ptr: zstd + col counts",
    "meta: zstd decode (names TLV)", "meta: TLV parse / name splitting", "indptr build", "indices/data alloc+zero",
    "chunk: zstd decode", "chunk: CRC32 of packed", "chunk: bitmap_unpack", "chunk: bit_planes_decode (+memcpy)", "chunk: byte_unsplit",
    "column walk: varint parse + gap->row", "column walk: std::sort(rv)", "column walk: emit indices/data", "chunk: other (allocs, loop)"};

static uint64_t g_cyc[S_COUNT];
static inline uint64_t tsc() { return __rdtsc(); }

struct Out { double ns_total; double ns[S_COUNT]; };

static void profile_once(const std::vector<uint8_t>& file, Out& out, double cyc_ns, double tsc_overhead_cyc, uint64_t& nnz_out) {
    std::memset(g_cyc, 0, sizeof(g_cyc));
    const uint8_t* buf = file.data();
    const size_t size = file.size();
    ReadLimits limits;
    uint64_t t0 = tsc();
    PZHeader hdr = validate_buffer(buf, size, limits);
    g_cyc[S_VALIDATE_CRC] += tsc() - t0;
    nnz_out = hdr.nnz;
    ReadResult r;
    r.m = hdr.m; r.n = hdr.n; r.nnz = hdr.nnz; r.vt_code = hdr.vt_code;
    size_t body_offset = sizeof(PZHeader);
    std::vector<uint32_t> perm, cc;
    ZstdDCtx dctx;
    // ---- perm ----
    t0 = tsc();
    const size_t perm_bytes = size_t(hdr.m) * 4;
    auto raw = zstd_decompress(dctx.ctx, buf + body_offset, hdr.perm_z_sz, perm_bytes);
    perm.resize(hdr.m);
    std::memcpy(perm.data(), raw.data(), perm_bytes);
    uint64_t t1 = tsc(); g_cyc[S_PERM_ZSTD] += t1 - t0;
    {
        std::vector<uint8_t> seen(hdr.m, 0);
        for (uint32_t i = 0; i < hdr.m; ++i) {
            if (perm[i] >= hdr.m || seen[perm[i]]) throw PZReadError("perm");
            seen[perm[i]] = 1;
        }
    }
    uint64_t t2 = tsc(); g_cyc[S_PERM_CHECK] += t2 - t1;
    body_offset += hdr.perm_z_sz;
    // ---- ptr ----
    t0 = tsc();
    const size_t ptr_bytes = size_t(hdr.n) * hdr.ptr_width;
    auto raw_ptr = zstd_decompress(dctx.ctx, buf + body_offset, hdr.ptr_z_sz, ptr_bytes);
    cc.resize(hdr.n);
    for (uint32_t j = 0; j < hdr.n; ++j) {
        uint32_t v = 0;
        for (int b = 0; b < hdr.ptr_width; ++b) v |= uint32_t(raw_ptr[size_t(j) * hdr.ptr_width + b]) << (8 * b);
        cc[j] = v;
    }
    body_offset += hdr.ptr_z_sz;
    g_cyc[S_PTR] += tsc() - t0;
    // ---- metadata ----
    if (hdr.flags & FLAG_HAS_METADATA) {
        t0 = tsc();
        const size_t max_out = size_t(std::min<uint64_t>(uint64_t(hdr.metadata_z_sz) * 4096 + (1u << 20), limits.max_metadata_bytes));
        auto mraw = zstd_decompress(dctx.ctx, buf + hdr.metadata_offset, hdr.metadata_z_sz, max_out);
        t1 = tsc(); g_cyc[S_META_ZSTD] += t1 - t0;
        parse_metadata_tlv(mraw.data(), mraw.size(), r.rownames, r.colnames, r.user_kv, hdr.m, hdr.n, limits.max_kv_entries);
        g_cyc[S_META_PARSE] += tsc() - t1;
    }
    // ---- indptr ----
    t0 = tsc();
    r.indptr.assign(size_t(hdr.n) + 1, 0);
    uint64_t running = 0;
    for (uint32_t j = 0; j < hdr.n; ++j) { r.indptr[j] = uint32_t(running); running += cc[j]; }
    r.indptr[hdr.n] = uint32_t(running);
    g_cyc[S_INDPTR] += tsc() - t0;

    // ---- chunk section (copy of decode_vocsc_section) ----
    const size_t nc = hdr.num_chunks;
    const size_t chunk_table_offset = body_offset;
    const size_t body_end = size - sizeof(PZFooter);
    (void)body_end;
    constexpr uint32_t kBlobHdr = 20;
    std::vector<uint32_t> ctable(nc);
    if (nc) std::memcpy(ctable.data(), buf + chunk_table_offset, nc * 4);
    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;
    const uint32_t chunk_cols = hdr.chunk_cols, n = hdr.n;
    t0 = tsc();
    r.indices.assign(size_t(hdr.nnz), 0);
    r.data.assign(size_t(hdr.nnz), 0);
    g_cyc[S_OUT_ALLOC] += tsc() - t0;
    uint64_t blob_cursor = uint64_t(chunk_table_offset) + nc * 4;
    std::vector<uint8_t> rawb, packed, prefilter;
    std::vector<uint32_t> gaps;
    std::vector<std::pair<uint32_t, uint32_t>> rv;
    uint64_t ev_parse = 0, ev_sort = 0;
    uint64_t chunk_total_cyc = 0, chunk_timed_cyc = 0;
    for (size_t c = 0; c < nc; ++c) {
        uint64_t cs0 = tsc();
        const uint32_t blob_sz = ctable[c];
        const uint8_t* blob = buf + blob_cursor;
        uint32_t ng32, msz32, zsz, chunk_crc, psz32;
        std::memcpy(&ng32, blob + 0, 4); std::memcpy(&msz32, blob + 4, 4); std::memcpy(&zsz, blob + 8, 4);
        std::memcpy(&chunk_crc, blob + 12, 4); std::memcpy(&psz32, blob + 16, 4);
        const size_t col_start = c * size_t(chunk_cols);
        const size_t col_end = std::min(col_start + size_t(chunk_cols), size_t(n));
        const size_t raw_sz = size_t(msz32) + size_t(ng32) * gw;
        const size_t bp_sz = ng32 > 0 ? 8 * ((size_t(ng32) + 7) / 8) : 0;
        const size_t pf_sz = size_t(msz32) + bp_sz + (ng32 > 0 ? size_t(gw - 1) * ng32 : 0);
        t0 = tsc();
        packed = zstd_decompress(dctx.ctx, blob + kBlobHdr, zsz, psz32);
        t1 = tsc(); g_cyc[S_CH_ZSTD] += t1 - t0;
        if (CRC32::compute(packed.data(), packed.size()) != chunk_crc) throw PZReadError("crc");
        t2 = tsc(); g_cyc[S_CH_CRC] += t2 - t1;
        prefilter.assign(pf_sz, 0);
        bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);
        uint64_t t3 = tsc(); g_cyc[S_CH_BITMAP] += t3 - t2;
        rawb.resize(raw_sz);
        std::memcpy(rawb.data(), prefilter.data(), msz32);
        if (ng32 > 0) {
            bit_planes_decode(prefilter.data() + msz32, ng32, rawb.data() + msz32);
            if (gw > 1) std::memcpy(rawb.data() + msz32 + ng32, prefilter.data() + msz32 + bp_sz, size_t(gw - 1) * ng32);
        }
        uint64_t t4 = tsc(); g_cyc[S_CH_BITPLANE] += t4 - t3;
        gaps.assign(ng32, 0);
        if (ng32 > 0) {
            if (gw == 2) byte_unsplit_16(rawb.data() + msz32, ng32, gaps.data());
            else byte_unsplit_32(rawb.data() + msz32, ng32, gaps.data());
        }
        uint64_t t5 = tsc(); g_cyc[S_CH_UNSPLIT] += t5 - t4;
        size_t mp = 0, gp = 0;
        const uint8_t* meta = rawb.data();
        uint64_t walk_total = 0;
        for (size_t j = col_start; j < col_end; ++j) {
            uint64_t a = tsc();
            uint32_t ng_col = varint_read(meta, msz32, mp);
            if (ng_col == 0) { g_cyc[S_COL_PARSE] += tsc() - a; ++ev_parse; continue; }
            rv.clear();
            rv.reserve(cc[j]);
            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv = varint_read(meta, msz32, mp);
                uint32_t cnt = varint_read(meta, msz32, mp);
                uint64_t prev_row = 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    uint64_t prow = prev_row + gaps[gp++];
                    prev_row = prow + 1;
                    rv.emplace_back(perm[prow], cv);
                }
            }
            uint64_t b = tsc();
            std::sort(rv.begin(), rv.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            uint64_t d = tsc();
            const uint32_t base = r.indptr[j];
            for (size_t k = 0; k < rv.size(); ++k) { r.indices[base + k] = rv[k].first; r.data[base + k] = rv[k].second; }
            uint64_t e = tsc();
            g_cyc[S_COL_PARSE] += b - a; g_cyc[S_COL_SORT] += d - b; g_cyc[S_COL_EMIT] += e - d;
            walk_total += e - a;
            ++ev_parse; ++ev_sort;
        }
        (void)walk_total;
        blob_cursor += blob_sz;
        uint64_t ce = tsc();
        chunk_total_cyc += ce - cs0;
        chunk_timed_cyc += (t1 - t0) + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) + walk_total;
    }
    g_cyc[S_CH_OTHER] += chunk_total_cyc > chunk_timed_cyc ? chunk_total_cyc - chunk_timed_cyc : 0;
    // subtract timestamp overhead from the per-column stages
    double over = tsc_overhead_cyc;
    double total_ns = 0;
    for (int s = 0; s < S_COUNT; ++s) out.ns[s] = double(g_cyc[s]) * cyc_ns;
    out.ns[S_COL_PARSE] = std::max(0.0, out.ns[S_COL_PARSE] - over * double(ev_parse) * cyc_ns);
    out.ns[S_COL_SORT] = std::max(0.0, out.ns[S_COL_SORT] - over * double(ev_sort) * cyc_ns);
    out.ns[S_COL_EMIT] = std::max(0.0, out.ns[S_COL_EMIT] - over * double(ev_sort) * cyc_ns);
    for (int s = 0; s < S_COUNT; ++s) if (s != S_FILE_READ) total_ns += out.ns[s];
    out.ns_total = total_ns;
}

int main(int argc, char** argv) {
    int reps = 5;
    std::string out = "results/track_a_profile.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else files.push_back(a);
    }
    // calibrate TSC
    double cyc_ns;
    {
        auto t0 = std::chrono::steady_clock::now(); uint64_t c0 = tsc();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.2) {}
        uint64_t c1 = tsc(); auto t1 = std::chrono::steady_clock::now();
        cyc_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(c1 - c0);
    }
    double over;
    {
        uint64_t acc = 0; const int N = 1000000;
        for (int i = 0; i < N; ++i) { uint64_t a = tsc(); uint64_t b = tsc(); acc += b - a; }
        over = double(acc) / N;   // cycles per timestamp pair, i.e. per timed region boundary
    }
    std::printf("TSC %.3f GHz, timestamp pair costs %.1f cycles (%.1f ns)\n", 1.0 / cyc_ns, over, over * cyc_ns);
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) {
        tsv << "sample\tmember\tnnz\tn\tm\tstage\tms\tns_per_nnz\tpct_of_stage_sum\n";
    }
    for (auto& f : files) {
        auto file = read_file(f);
        Out best{1e300, {}};
        uint64_t nnz = 0;
        for (int r = 0; r < reps; ++r) {
            Out o;
            profile_once(file, o, cyc_ns, over, nnz);
            if (o.ns_total < best.ns_total) best = o;
        }
        // file read from disk (OS cache)
        double rd = 1e300;
        for (int r = 0; r < 3; ++r) { Timer t; auto b = read_file(f); (void)b; rd = std::min(rd, t.ms()); }
        best.ns[S_FILE_READ] = rd * 1e6;
        // uninstrumented reference
        double ref = 1e300;
        ReadResult rr;
        for (int r = 0; r < reps; ++r) { Timer t; rr = read_1pz_buffer(file.data(), file.size()); ref = std::min(ref, t.ms()); }
        uint32_t M_ = rr.m, N_ = rr.n;
        std::string sample = parent_name(f), member = base_name(f);
        std::printf("\n== %s/%s  nnz=%llu  m=%u n=%u  file %zu B\n", sample.c_str(), member.c_str(), (unsigned long long)nnz, M_, N_, file.size());
        std::printf("   uninstrumented read_1pz_buffer: %.1f ms  (%.1f Mnnz/s, %.1f ns/nnz); sum of instrumented stages (excl. file read): %.1f ms\n",
                    ref, double(nnz) / ref / 1e3, ref * 1e6 / double(nnz), best.ns_total / 1e6);
        for (int s = 0; s < S_COUNT; ++s) {
            double ms = best.ns[s] / 1e6;
            double pct = s == S_FILE_READ ? 0 : 100.0 * best.ns[s] / best.ns_total;
            std::printf("   %-40s %9.2f ms  %9.2f ns/nnz  %5.1f%%\n", kStage[s], ms, best.ns[s] / double(nnz), pct);
            tsv << sample << '\t' << member << '\t' << nnz << '\t' << N_ << '\t' << M_ << '\t' << kStage[s] << '\t' << ms << '\t' << best.ns[s] / double(nnz) << '\t' << pct << '\n';
        }
        tsv << sample << '\t' << member << '\t' << nnz << '\t' << N_ << '\t' << M_ << "\tTOTAL instrumented (excl file read)\t" << best.ns_total / 1e6 << '\t' << best.ns_total / double(nnz) << "\t100\n";
        tsv << sample << '\t' << member << '\t' << nnz << '\t' << N_ << '\t' << M_ << "\tTOTAL uninstrumented read_1pz_buffer\t" << ref << '\t' << ref * 1e6 / double(nnz) << "\t0\n";
        tsv.flush();
    }
    return 0;
}
