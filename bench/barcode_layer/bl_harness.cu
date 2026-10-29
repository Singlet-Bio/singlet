// Standalone harness for src/engine_gpu/barcode_layer.cuh (phase 1c prep). Instantiates every kernel in the header and writes
// per-read results as flat binary files, which bench/barcode_layer/run_class.py compares with the Python reference
// (barcode_layer_ref.py) and with STAR's own CB/UB.
//
//   nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -Isrc/engine_gpu bench/barcode_layer/bl_harness.cu -o bl_harness
//   bl_harness mode=simple|complex|discover|adt|guide|spot key=value ...
//
// Inputs (little endian):
//   reads.bin   'BLR1' u32, pad u32, n u64, off[n+1] u64, seq[off[n]], qual[off[n]]
//   u64 list    n u64, then n u64 (sorted unique 2-bit values)
// Outputs: <out>.<field>.<type>, raw arrays, one element per read unless noted.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "barcode_layer.cuh"

using namespace bl;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

typedef std::map<std::string, std::string> Args;
static std::string arg(const Args& a, const char* k, const char* d = "") { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; }
static long long argi(const Args& a, const char* k, long long d = 0) { auto it = a.find(k); return it == a.end() ? d : atoll(it->second.c_str()); }

struct Reads { u64 n = 0; std::vector<u64> off; std::vector<char> seq, qual; };

static bool load_reads(const std::string& path, Reads& R) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    u32 magic = 0, pad = 0; u64 n = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&pad, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1 || magic != 0x31524c42u) { fprintf(stderr, "bad reads file\n"); fclose(f); return false; }
    R.n = n; R.off.resize(n + 1);
    if (fread(R.off.data(), 8, n + 1, f) != n + 1) { fclose(f); return false; }
    R.seq.resize(R.off[n]); R.qual.resize(R.off[n]);
    bool ok = fread(R.seq.data(), 1, R.seq.size(), f) == R.seq.size() && fread(R.qual.data(), 1, R.qual.size(), f) == R.qual.size();
    fclose(f); return ok;
}

static std::vector<u64> load_u64(const std::string& path) {
    std::vector<u64> v; FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(3); }
    u64 n = 0; if (fread(&n, 8, 1, f) != 1) exit(3); v.resize(n); if (n && fread(v.data(), 8, n, f) != n) exit(3); fclose(f); return v;
}

static std::vector<std::string> load_lines(const std::string& path) {
    std::vector<std::string> out; std::ifstream f(path); std::string s;
    while (std::getline(f, s)) { while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back(); if (!s.empty()) out.push_back(s); }
    return out;
}

template <class T> static void write_arr(const std::string& prefix, const char* name, const char* type, const std::vector<T>& v) {
    std::string p = prefix + "." + name + "." + type; FILE* f = fopen(p.c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", p.c_str()); exit(4); }
    if (!v.empty()) fwrite(v.data(), sizeof(T), v.size(), f); fclose(f);
}

template <class T> static T* to_dev(const std::vector<T>& v) {
    T* d = nullptr; size_t b = std::max<size_t>(v.size() * sizeof(T), 8); CK(cudaMalloc(&d, b));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}

struct Timer {
    cudaEvent_t a, b; Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static u64 pack2(const std::string& s, int& pos_n) {       // host convertNuclStrToInt64
    u64 v = 0; pos_n = -1;
    for (size_t i = 0; i < s.size(); i++) {
        int c; switch (s[i]) { case 'A': case 'a': c = 0; break; case 'C': case 'c': c = 1; break; case 'G': case 'g': c = 2; break; case 'T': case 't': c = 3; break; default: c = 4; }
        if (c > 3) { if (pos_n >= 0) { pos_n = -2; return v; } pos_n = (int)i; c = 0; }
        v = (v << 2) + (u64)c;
    }
    return v;
}

static const int BLK = 256;
static inline unsigned grid(u64 n) { return (unsigned)((n + BLK - 1) / BLK); }

struct DevReads {
    char *seq = nullptr, *qual = nullptr; u64* off = nullptr; u64 n = 0;
    explicit DevReads(const Reads& R) { n = R.n; seq = to_dev(R.seq); qual = to_dev(R.qual); off = to_dev(R.off); }
};

static void dump_records(const std::string& out, const std::vector<BlRecord>& rec, const std::vector<BlRead>* rd) {
    size_t n = rec.size(); std::vector<i64> cell(n); std::vector<i32> st(n); std::vector<u64> um(n); std::vector<u32> ul(n);
    for (size_t i = 0; i < n; i++) { cell[i] = rec[i].cell_key; st[i] = rec[i].status; um[i] = rec[i].umi_key; ul[i] = rec[i].umi_len; }
    write_arr(out, "cell", "i64", cell); write_arr(out, "status", "i32", st); write_arr(out, "umi", "u64", um); write_arr(out, "umilen", "u32", ul);
    if (rd) {
        std::vector<u64> cv(n); std::vector<i32> pn(n);
        for (size_t i = 0; i < n; i++) { cv[i] = (*rd)[i].cb_val; pn[i] = (*rd)[i].pos_n; }
        write_arr(out, "cbval", "u64", cv); write_arr(out, "posn", "i32", pn);
    }
}

static SimpleDev simple_layout(const Args& a, const u64* d_wl, u64 wl_n) {
    SimpleDev L{};
    L.cb_start = (u32)argi(a, "cb_start", 0); L.cb_len = (u32)argi(a, "cb_len", 16);
    L.umi_start = (u32)argi(a, "umi_start", 16); L.umi_len = (u32)argi(a, "umi_len", 12);
    L.mm1 = (u32)argi(a, "mm1", 0); L.mm1_multi = (u32)argi(a, "mm1_multi", 0); L.mm1_multi_nbase = (u32)argi(a, "nbase", 0);
    L.one_exact = (u32)argi(a, "one_exact", 0); L.sam_tag_out = (u32)argi(a, "sam_tag_out", 0); L.has_wl = (u32)argi(a, "has_wl", 1);
    L.wl = d_wl; L.wl_n = wl_n; return L;
}

// ---------------------------------------------------------------------------------------------------------------------
static int mode_simple(const Args& a, Reads& R) {
    std::string out = arg(a, "out");
    int fq = (int)argi(a, "force_qual", 0); if (fq) std::fill(R.qual.begin(), R.qual.end(), (char)fq);
    std::vector<u64> wl; if (argi(a, "has_wl", 1)) wl = load_u64(arg(a, "wl"));
    u64 *d_wl = to_dev(wl); DevReads D(R); SimpleDev L = simple_layout(a, wl.empty() ? nullptr : d_wl, wl.size());
    std::vector<double> qtab(34); for (int q = 0; q < 34; q++) qtab[q] = pow(10.0, -q / 10.0);     // STAR: pow(10.0, -qin / 10.0)
    double* d_q = to_dev(qtab);
    BlRead* d_rd; BlRecord* d_rec; u32* d_hist; CK(cudaMalloc(&d_rd, R.n * sizeof(BlRead))); CK(cudaMalloc(&d_rec, R.n * sizeof(BlRecord)));
    CK(cudaMalloc(&d_hist, std::max<size_t>(wl.size(), 1) * 4)); CK(cudaMemset(d_hist, 0, std::max<size_t>(wl.size(), 1) * 4));
    Timer t; t.start();
    k_bl_extract_simple<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, L, d_rd); CK(cudaGetLastError());
    float t1 = t.stop(); t.start();
    if (L.has_wl) { k_bl_exact_hist<<<grid(R.n), BLK>>>(d_rd, R.n, true, d_hist); CK(cudaGetLastError()); }
    std::vector<u32> hist_exact(wl.size());
    if (!wl.empty()) CK(cudaMemcpy(hist_exact.data(), d_hist, wl.size() * 4, cudaMemcpyDeviceToHost));
    if (argi(a, "pseudocount", 0) && L.has_wl) { k_bl_add_pseudocount<<<grid(wl.size()), BLK>>>(d_hist, wl.size()); CK(cudaGetLastError()); }
    float t2 = t.stop(); t.start();
    k_bl_resolve_simple<<<grid(R.n), BLK>>>(D.seq, D.qual, D.off, R.n, L, d_hist, d_q, d_rd, d_rec); CK(cudaGetLastError());
    float t3 = t.stop();
    std::vector<BlRecord> rec(R.n); std::vector<BlRead> rd(R.n);
    CK(cudaMemcpy(rec.data(), d_rec, R.n * sizeof(BlRecord), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rd.data(), d_rd, R.n * sizeof(BlRead), cudaMemcpyDeviceToHost));
    dump_records(out, rec, &rd); write_arr(out, "histexact", "u32", hist_exact);
    printf("simple: %llu reads, extract %.2f ms, hist %.2f ms, resolve %.2f ms\n", (unsigned long long)R.n, t1, t2, t3);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// complex layout file:   adapter <seq|->      adapter_mm <n>      umi a0 d0 a1 d1      seg a0 d0 a1 d1 <whitelist text file>
struct HostSeg { i32 at[2], ad[2]; std::string wlfile; };

static SegDev build_seg(const HostSeg& hs, u64 factor, u64& total_out, std::vector<void*>& keep) {
    SegDev S{}; for (int k = 0; k < 2; k++) { S.anchor_type[k] = hs.at[k]; S.anchor_dist[k] = hs.ad[k]; }
    S.wl_factor = factor; S.min_len = 0xFFFFFFFFu;
    if (hs.wlfile.empty()) { total_out = 0; return S; }
    std::map<size_t, std::vector<u64>> by_len;
    for (auto& s : load_lines(hs.wlfile)) { int pn; u64 v = pack2(s, pn); if (pn == 0) continue; by_len[s.size()].push_back(v); }   // ParametersSolo.cpp:373-396
    size_t max_len = by_len.empty() ? 0 : by_len.rbegin()->first;
    std::vector<u64> wl; std::vector<u32> off(max_len + 2, 0), add(max_len + 1, 0); u64 total = 0;
    for (size_t ln = 0; ln <= max_len; ln++) {
        off[ln] = (u32)wl.size(); add[ln] = (u32)total;
        auto it = by_len.find(ln);
        if (ln >= 1 && it != by_len.end()) {
            S.min_len = std::min<u32>(S.min_len, (u32)ln);
            std::vector<u64>& v = it->second; std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end());
            wl.insert(wl.end(), v.begin(), v.end()); total += v.size();
        }
    }
    off[max_len + 1] = (u32)wl.size();
    S.n_len = (u32)(max_len + 1); total_out = total;
    S.wl = to_dev(wl); S.wl_off = to_dev(off); S.wl_add = to_dev(add);
    return S;
}

static int mode_complex(const Args& a, Reads& R) {
    std::string out = arg(a, "out");
    std::vector<HostSeg> segs; HostSeg umi{}; std::string adapter = "-"; int adapter_mm = 1;
    for (auto& line : load_lines(arg(a, "layout"))) {
        std::istringstream is(line); std::string key; is >> key;
        if (key == "adapter") is >> adapter;
        else if (key == "adapter_mm") is >> adapter_mm;
        else if (key == "umi") is >> umi.at[0] >> umi.ad[0] >> umi.at[1] >> umi.ad[1];
        else if (key == "seg") { HostSeg s; is >> s.at[0] >> s.ad[0] >> s.at[1] >> s.ad[1] >> s.wlfile; segs.push_back(s); }
    }
    ComplexDev C{}; C.n_seg = (u32)segs.size(); C.mm1 = 0; std::vector<void*> keep; u64 factor = 1;
    for (size_t g = 0; g < segs.size(); g++) { u64 tot; C.seg[g] = build_seg(segs[g], factor, tot, keep); factor *= tot; printf("segment %zu: %llu entries over all lengths, factor %llu\n", g, (unsigned long long)tot, (unsigned long long)C.seg[g].wl_factor); }
    C.umi = build_seg(umi, 1, factor, keep);
    std::vector<char> ad; if (adapter != "-") ad.assign(adapter.begin(), adapter.end());
    C.adapter = ad.empty() ? nullptr : to_dev(ad); C.adapter_len = (u32)ad.size(); C.adapter_mm_max = (u32)adapter_mm;
    DevReads D(R); BlRecord* d_rec; CK(cudaMalloc(&d_rec, R.n * sizeof(BlRecord)));
    Timer t; t.start();
    k_bl_extract_complex<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, C, d_rec); CK(cudaGetLastError());
    float ms = t.stop();
    std::vector<BlRecord> rec(R.n); CK(cudaMemcpy(rec.data(), d_rec, R.n * sizeof(BlRecord), cudaMemcpyDeviceToHost));
    dump_records(out, rec, nullptr);
    printf("complex: %llu reads, %zu segments, %.2f ms\n", (unsigned long long)R.n, segs.size(), ms);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// no-whitelist protocols: discovery (thrust sort + run-length + threshold + rank) and the per-read cell key
struct CountDescKeyAsc { __host__ __device__ bool operator()(const thrust::tuple<u32, u64>& x, const thrust::tuple<u32, u64>& y) const {
    if (thrust::get<0>(x) != thrust::get<0>(y)) return thrust::get<0>(x) > thrust::get<0>(y); return thrust::get<1>(x) < thrust::get<1>(y); } };
struct MinReads { u32 m; __host__ __device__ bool operator()(const thrust::tuple<u32, u64>& x) const { return thrust::get<0>(x) >= m; } };
struct NFreeSelect { const BlRead* rd; __device__ bool operator()(u64 i) const { return rd[i].pos_n == -1; } };
struct GetCb { const BlRead* rd; __device__ u64 operator()(u64 i) const { return rd[i].cb_val; } };

static int mode_discover(const Args& a, Reads& R) {
    std::string out = arg(a, "out"); u32 min_reads = (u32)argi(a, "min_reads", 5);
    DevReads D(R);
    // raw counting pass: no UMI check, no whitelist; N-free barcodes are the countable ones
    Args b = a; b["sam_tag_out"] = "1"; b["has_wl"] = "0";
    SimpleDev Lc = simple_layout(b, nullptr, 0);
    BlRead* d_rd; CK(cudaMalloc(&d_rd, R.n * sizeof(BlRead)));
    Timer t; t.start();
    k_bl_extract_simple<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, Lc, d_rd); CK(cudaGetLastError());
    thrust::device_vector<u64> idx(R.n); thrust::sequence(idx.begin(), idx.end());
    thrust::device_vector<u64> sel(R.n);
    size_t ns = thrust::copy_if(thrust::device, idx.begin(), idx.end(), sel.begin(), NFreeSelect{d_rd}) - sel.begin();
    thrust::device_vector<u64> keys(ns);
    thrust::transform(thrust::device, sel.begin(), sel.begin() + ns, keys.begin(), GetCb{d_rd});
    thrust::sort(thrust::device, keys.begin(), keys.end());
    thrust::device_vector<u64> uk(ns); thrust::device_vector<u32> cnt(ns);
    auto end = thrust::reduce_by_key(thrust::device, keys.begin(), keys.end(), thrust::constant_iterator<u32>(1), uk.begin(), cnt.begin());
    size_t nu = end.first - uk.begin();
    // keep count >= min_reads
    thrust::device_vector<u64> ck(nu); thrust::device_vector<u32> cc(nu);
    auto zi = thrust::make_zip_iterator(thrust::make_tuple(cnt.begin(), uk.begin()));
    auto zo = thrust::make_zip_iterator(thrust::make_tuple(cc.begin(), ck.begin()));
    size_t nk = thrust::copy_if(thrust::device, zi, zi + nu, zo, MinReads{min_reads}) - zo;
    ck.resize(nk); cc.resize(nk);
    // rank: count descending, then barcode ascending (stream rule, singlet.cpp 5165-5190)
    thrust::sort(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(cc.begin(), ck.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(cc.end(), ck.end())), CountDescKeyAsc());
    // value-sorted copy with rank
    thrust::device_vector<u64> vals(ck); thrust::device_vector<u32> rank(nk); thrust::sequence(rank.begin(), rank.end());
    thrust::sort_by_key(thrust::device, vals.begin(), vals.end(), rank.begin());
    float t1 = t.stop();
    // per-read records through the same extraction as the pileup sees (UMI check on)
    Args c = a; c["sam_tag_out"] = "0"; c["has_wl"] = "0"; SimpleDev Lr = simple_layout(c, nullptr, 0);
    k_bl_extract_simple<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, Lr, d_rd); CK(cudaGetLastError());
    BlRecord* d_rec; CK(cudaMalloc(&d_rec, R.n * sizeof(BlRecord)));
    k_bl_resolve_simple<<<grid(R.n), BLK>>>(D.seq, D.qual, D.off, R.n, Lr, nullptr, nullptr, d_rd, d_rec); CK(cudaGetLastError());
    k_bl_discovered_cell<<<grid(R.n), BLK>>>(d_rd, R.n, thrust::raw_pointer_cast(vals.data()), thrust::raw_pointer_cast(rank.data()), nk, d_rec); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<BlRecord> rec(R.n); std::vector<BlRead> rd(R.n);
    CK(cudaMemcpy(rec.data(), d_rec, R.n * sizeof(BlRecord), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rd.data(), d_rd, R.n * sizeof(BlRead), cudaMemcpyDeviceToHost));
    dump_records(out, rec, &rd);
    std::vector<u64> hk(nk); std::vector<u32> hc(nk); thrust::copy(ck.begin(), ck.end(), hk.begin()); thrust::copy(cc.begin(), cc.end(), hc.begin());
    write_arr(out, "disc_keys", "u64", hk); write_arr(out, "disc_counts", "u32", hc);
    printf("discover: %llu reads, %zu N-free, %zu distinct, %zu with >= %u reads, discovery %.2f ms\n", (unsigned long long)R.n, ns, nu, nk, min_reads, t1);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
static int mode_adt(const Args& a, Reads& R) {
    std::string out = arg(a, "out"); int max_off = (int)argi(a, "max_offset", 5);
    std::vector<std::string> tags; for (auto& s : load_lines(arg(a, "tags"))) { std::string u = s; for (auto& ch : u) ch = (char)toupper(ch); tags.push_back(u); }
    u32 tl = tags.empty() ? 0 : (u32)tags[0].size();
    std::map<u64, i32> ex, h1;      // adt_matcher.h build_index; tags of another length or with N cannot be a window value
    std::map<std::string, i32> exs;
    for (size_t i = 0; i < tags.size(); i++) exs[tags[i]] = (i32)i;
    for (auto& kv : exs) { int pn; u64 v = pack2(kv.first, pn); if (pn == -1 && kv.first.size() == tl) ex[v] = kv.second; }
    std::map<std::string, i32> h1s;
    for (size_t i = 0; i < tags.size(); i++) {
        const std::string& t = tags[i];
        for (size_t p = 0; p < t.size(); p++) for (char b : std::string("ACGT")) {
            if (b == t[p]) continue; std::string nb = t; nb[p] = b;
            if (exs.count(nb)) continue;
            auto it = h1s.find(nb);
            if (it == h1s.end()) h1s[nb] = (i32)i; else if (it->second != (i32)i) it->second = -2;
        }
    }
    for (auto& kv : h1s) { int pn; u64 v = pack2(kv.first, pn); if (pn == -1 && kv.first.size() == tl) h1[v] = kv.second; }
    std::vector<u64> exk, h1k; std::vector<i32> exi, h1i;
    for (auto& kv : ex) { exk.push_back(kv.first); exi.push_back(kv.second); } for (auto& kv : h1) { h1k.push_back(kv.first); h1i.push_back(kv.second); }
    TagDev T{}; T.tag_len = tl; T.ex_key = to_dev(exk); T.ex_id = to_dev(exi); T.ex_n = exk.size(); T.h1_key = to_dev(h1k); T.h1_id = to_dev(h1i); T.h1_n = h1k.size();
    DevReads D(R); i32* d_o; CK(cudaMalloc(&d_o, R.n * 4));
    Timer t; t.start(); k_bl_adt_match<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, T, max_off, d_o); CK(cudaGetLastError()); float ms = t.stop();
    std::vector<i32> o(R.n); CK(cudaMemcpy(o.data(), d_o, R.n * 4, cudaMemcpyDeviceToHost));
    write_arr(out, "tag", "i32", o);
    printf("adt: %llu reads, %zu tags, %zu exact keys, %zu hamming-1 keys, %.2f ms\n", (unsigned long long)R.n, tags.size(), exk.size(), h1k.size(), ms);
    return 0;
}

static int mode_guide(const Args& a, Reads& R) {
    std::string out = arg(a, "out"); std::vector<std::string> gs;
    for (auto& s : load_lines(arg(a, "guides"))) { std::string u = s; for (auto& ch : u) ch = (char)toupper(ch); gs.push_back(u); }
    const u32 stride = 64; std::vector<char> g(gs.size() * stride, 0); std::vector<u32> gl(gs.size());
    for (size_t i = 0; i < gs.size(); i++) { gl[i] = (u32)std::min<size_t>(gs[i].size(), stride); memcpy(&g[i * stride], gs[i].data(), gl[i]); }
    DevReads D(R); char* d_g = to_dev(g); u32* d_gl = to_dev(gl); i32* d_o; CK(cudaMalloc(&d_o, R.n * 4));
    Timer t; t.start(); k_bl_guide_match<<<grid(R.n), BLK>>>(D.seq, D.off, R.n, d_g, d_gl, (u32)gs.size(), stride, d_o); CK(cudaGetLastError()); float ms = t.stop();
    std::vector<i32> o(R.n); CK(cudaMemcpy(o.data(), d_o, R.n * 4, cudaMemcpyDeviceToHost));
    write_arr(out, "gid", "i32", o);
    printf("guide: %llu reads, %zu guides, %.2f ms\n", (unsigned long long)R.n, gs.size(), ms);
    return 0;
}

static int mode_spot(const Args& a, Reads&) {
    std::string out = arg(a, "out"); std::vector<u64> cells = load_u64(arg(a, "cells")), spots = load_u64(arg(a, "spots"));
    u64* d_c = to_dev(cells); u64* d_s = to_dev(spots); i32* d_o; CK(cudaMalloc(&d_o, std::max<size_t>(cells.size(), 1) * 4));
    k_bl_spot_gather<<<grid(cells.size()), BLK>>>(d_c, cells.size(), d_s, spots.size(), d_o); CK(cudaGetLastError());
    std::vector<i32> o(cells.size()); CK(cudaMemcpy(o.data(), d_o, cells.size() * 4, cudaMemcpyDeviceToHost));
    write_arr(out, "row", "i32", o);
    printf("spot: %zu cells, %zu spots\n", cells.size(), spots.size());
    return 0;
}

int main(int argc, char** argv) {
    Args a; for (int i = 1; i < argc; i++) { std::string s = argv[i]; size_t e = s.find('='); if (e == std::string::npos) continue; a[s.substr(0, e)] = s.substr(e + 1); }
    std::string mode = arg(a, "mode");
    int dev = 0; CK(cudaGetDevice(&dev)); cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev)); printf("device %s sm_%d%d\n", pr.name, pr.major, pr.minor);
    Reads R; if (mode != "spot" && !load_reads(arg(a, "reads"), R)) return 1;
    if (mode == "simple") return mode_simple(a, R);
    if (mode == "complex") return mode_complex(a, R);
    if (mode == "discover") return mode_discover(a, R);
    if (mode == "adt") return mode_adt(a, R);
    if (mode == "guide") return mode_guide(a, R);
    if (mode == "spot") return mode_spot(a, R);
    fprintf(stderr, "unknown mode '%s'\n", mode.c_str()); return 1;
}
