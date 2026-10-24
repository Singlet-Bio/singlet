// Track D: dump the encode() output of every registered codec for every corpus file.
//   bin/track_d_dump.exe [--out-dir results/track_d_dump] [--codec a,b] [--skip a,b] [--skip-existing]
//                        [--include-cm] [--verify] file.1pz...
// Files are written as <out-dir>/<sample>__<member>__<codec>.bin where sample/member come from the
// path corpus/<sample>/<member>_counts.1pz.  Codecs named d_cm_* (already second-pass by construction)
// are skipped unless --include-cm.
#include "common.h"
#include "registry.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}
static std::string tag_of(const std::string& p) {
    auto norm = p;
    std::replace(norm.begin(), norm.end(), '\\', '/');
    auto s = norm.find_last_of('/');
    std::string base = s == std::string::npos ? norm : norm.substr(s + 1);
    std::string par;
    if (s != std::string::npos && s > 0) {
        auto t = norm.find_last_of('/', s - 1);
        size_t b = (t == std::string::npos) ? 0 : t + 1;
        par = norm.substr(b, s - b);
    }
    auto u = base.find("_counts");
    std::string member = u == std::string::npos ? base : base.substr(0, u);
    return par + "__" + member;
}

int main(int argc, char** argv) {
    std::vector<std::string> files, want, skip;
    std::string outdir = "results/track_d_dump";
    bool skip_existing = false, include_cm = false, verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out-dir" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--codec" && i + 1 < argc) want = split(argv[++i]);
        else if (a == "--skip" && i + 1 < argc) skip = split(argv[++i]);
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--include-cm") include_cm = true;
        else if (a == "--verify") verify = true;
        else files.push_back(a);
    }
    if (files.empty()) { std::cerr << "usage: track_d_dump [opts] file.1pz...\n"; return 2; }
    auto codecs = all_codecs();
    for (auto& f : files) {
        std::string tag = tag_of(f);
        CanonicalMatrix M = load_1pz(f);
        std::printf("== %s (nnz %llu)\n", tag.c_str(), (unsigned long long)M.nnz);
        for (auto& c : codecs) {
            if (!want.empty() && std::find(want.begin(), want.end(), c.name) == want.end()) continue;
            if (std::find(skip.begin(), skip.end(), c.name) != skip.end()) continue;
            if (!include_cm && want.empty() && c.name.rfind("d_cm_", 0) == 0) continue;
            std::string path = outdir + "/" + tag + "__" + c.name + ".bin";
            if (skip_existing && std::ifstream(path).good()) { std::printf("  %-28s exists\n", c.name.c_str()); continue; }
            try {
                Timer t;
                auto enc = c.encode(M);
                double ms = t.ms();
                write_file(path, enc);
                bool ok = true;
                if (verify) ok = (c.decode(enc) == M);
                std::printf("  %-28s %10zu B  enc %.0f ms %s\n", c.name.c_str(), enc.size(), ms,
                            verify ? (ok ? "verify=OK" : "verify=FAIL") : "");
            } catch (const std::exception& e) {
                std::printf("  %-28s ERROR %s\n", c.name.c_str(), e.what());
            }
            std::fflush(stdout);
        }
    }
    return 0;
}
