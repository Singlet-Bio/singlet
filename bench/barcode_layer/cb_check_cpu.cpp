// CPU comparator for the GPU barcode layer: W3b's R11 CbCorrector (cb_correct.h, copied verbatim from branch perf/bam-stream,
// commit bad77390) applied to the same reads file as bl_harness. Pass 1 builds STAR's exact histogram exactly as singlet.cpp
// does (exact whitelist hit, UMI without N and not a homopolymer, every read); pass 2 resolves each read.
//   cb_check_cpu reads=reads.bin wl=whitelist.txt out=prefix cb_start=0 cb_len=16 umi_start=16 umi_len=12 qual=F|file
// Writes <out>.cell.i64 (whitelist index, -1 none) and <out>.status.i32 (CbCorrector::Status, -3 bad UMI, -5 short read).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "cb_correct.h"

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; i++) { std::string s = argv[i]; size_t e = s.find('='); if (e != std::string::npos) a[s.substr(0, e)] = s.substr(e + 1); }
    int cb0 = atoi(a["cb_start"].c_str()), cbl = atoi(a["cb_len"].c_str()), um0 = atoi(a["umi_start"].c_str()), uml = atoi(a["umi_len"].c_str());
    bool real_q = a["qual"] == "file";
    FILE* f = fopen(a["reads"].c_str(), "rb"); if (!f) return 1;
    uint32_t magic, pad; uint64_t n; if (fread(&magic, 4, 1, f) != 1 || fread(&pad, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1) return 1;
    std::vector<uint64_t> off(n + 1); if (fread(off.data(), 8, n + 1, f) != n + 1) return 1;
    std::vector<char> seq(off[n]), qual(off[n]);
    if (fread(seq.data(), 1, seq.size(), f) != seq.size() || fread(qual.data(), 1, qual.size(), f) != qual.size()) return 1;
    fclose(f);
    singlet::CbCorrector cc;
    if (!cc.load_whitelist(a["wl"]) || cc.length() != cbl) { fprintf(stderr, "whitelist load failed\n"); return 2; }
    fprintf(stderr, "whitelist %zu x %d\n", cc.size(), cc.length());
    auto umi_ok = [&](const char* s, uint64_t len) {
        if ((uint64_t)(um0 + uml) > len) return false;      // short read: singlet's pass 1 skips it (r1len check)
        bool ok = true, homo = true; const char* u = s + um0;
        for (int j = 0; j < uml; ++j) { char c = u[j]; if (c != 'A' && c != 'C' && c != 'G' && c != 'T') { ok = false; break; } if (c != u[0]) homo = false; }
        return ok && !homo;
    };
    for (uint64_t i = 0; i < n; i++) {                                  // pass 1
        const char* s = &seq[off[i]]; uint64_t len = off[i + 1] - off[i];
        if ((uint64_t)(cb0 + cbl) > len || !umi_ok(s, len)) continue;
        int64_t w = cc.find(std::string(s + cb0, cbl).c_str());
        if (w >= 0) cc.add_exact(w, 1);
    }
    std::vector<int64_t> cell(n, -1); std::vector<int32_t> st(n, 0);
    char out[64]; std::string cb;
    for (uint64_t i = 0; i < n; i++) {                                  // pass 2
        const char* s = &seq[off[i]]; uint64_t len = off[i + 1] - off[i];
        if ((uint64_t)(cb0 + cbl) > len || !umi_ok(s, len)) { st[i] = -3; continue; }
        cb.assign(s + cb0, cbl);
        int r = cc.resolve(cb.c_str(), real_q ? &qual[off[i] + cb0] : nullptr, out);
        st[i] = r;
        if (r >= 0) cell[i] = cc.find(std::string(out, cbl).c_str());
    }
    std::string p = a["out"];
    FILE* o = fopen((p + ".cell.i64").c_str(), "wb"); fwrite(cell.data(), 8, n, o); fclose(o);
    o = fopen((p + ".status.i32").c_str(), "wb"); fwrite(st.data(), 4, n, o); fclose(o);
    uint64_t cnt[8] = {}; for (auto v : st) if (v >= -4 && v <= 2) cnt[v + 4]++;
    fprintf(stderr, "cb_check_cpu: %llu reads: exact=%llu one=%llu posterior=%llu rejected=%llu no_match=%llu two_N=%llu bad_umi/short=%llu\n", (unsigned long long)n,
            (unsigned long long)cnt[4], (unsigned long long)cnt[5], (unsigned long long)cnt[6], (unsigned long long)cnt[0], (unsigned long long)cnt[3], (unsigned long long)cnt[2], (unsigned long long)cnt[1]);
    return 0;
}
