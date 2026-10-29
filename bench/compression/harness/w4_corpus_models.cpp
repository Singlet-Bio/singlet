// SPDX-License-Identifier: MIT
// w4_corpus: build per-reference-build vocabulary model files (Track A format, kernels/track_a.h) from the row-name lists
// found in the w4 corpus. One file per build: results/w4_corpus_models/track_a_vocab_<build>.bin with one list per
// distinct (member, variant). Also prints the size of every file (the model cost that the atlas pays once per build).
//
//   bin/w4_corpus_models.exe results/w4_corpus_vocab_manifest.tsv results/w4_corpus_models
//
// manifest columns: build kind variant hash file     (written by tools/w4_corpus_names.py; empty-name variants are skipped)
#include "common.h"
#include "track_a.h"
#include <iostream>
#include <sstream>

using namespace cs;

// Same file format as ta::build_vocab_file but zstd-19 only (coder tag 0): xz-9e/brotli-11 on 15 MB lists take minutes per list and
// only change the model *size* (Track A measured xz about 25% smaller than zstd-19), never the codec's behaviour.
static std::vector<uint8_t> build_vocab_z19(const std::vector<std::pair<std::string, std::vector<std::string>>>& lists) {
    struct L { std::string kind; uint32_t count; uint64_t hash, raw_len; std::vector<uint8_t> z; };
    std::vector<L> ls;
    for (auto& [kind, names] : lists) {
        L l; l.kind = kind; l.count = uint32_t(names.size()); l.hash = ta::names_hash(names);
        auto raw = ta::join_nul(names);
        l.raw_len = raw.size();
        l.z = ta::z_compress(ta::coder_for_tag(0), raw);
        ls.push_back(std::move(l));
    }
    size_t hdr = 8 + 4;
    for (auto& l : ls) hdr += 2 + l.kind.size() + 4 + 8 + 1 + 8 + 8 + 8;
    std::vector<uint8_t> o; ta::ByteW w(o);
    w.bytes("TAVOC001", 8); w.u32(uint32_t(ls.size()));
    uint64_t off = hdr;
    for (auto& l : ls) {
        w.u16(uint16_t(l.kind.size())); w.bytes(l.kind.data(), l.kind.size());
        w.u32(l.count); w.u64(l.hash); w.u8(0); w.u64(l.raw_len); w.u64(l.z.size()); w.u64(off);
        off += l.z.size();
    }
    for (auto& l : ls) w.bytes(l.z.data(), l.z.size());
    return o;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: w4_corpus_models manifest.tsv outdir\n"; return 2; }
    std::ifstream f(argv[1]);
    std::string line;
    std::getline(f, line);   // header
    std::map<std::string, std::vector<std::pair<std::string, std::vector<std::string>>>> by_build;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string t;
        while (std::getline(ss, t, '\t')) c.push_back(t);
        if (c.size() < 5) continue;
        const std::string& build = c[0];
        const std::string& kind = c[1];
        const std::string& path = c[4];
        auto M = load_1pz(path);
        if (M.rownames.empty()) { std::printf("skip empty list %s %s (%s)\n", build.c_str(), kind.c_str(), path.c_str()); continue; }
        std::printf("%s %-8s rows=%zu hash=%016llx (%s)\n", build.c_str(), kind.c_str(), M.rownames.size(),
                    (unsigned long long)ta::names_hash(M.rownames), path.c_str());
        by_build[build].emplace_back(kind, std::move(M.rownames));
    }
    std::string out = argv[2];
    std::string tsv = out + "/w4_corpus_model_sizes.tsv";
    std::ofstream o(tsv);
    o << "build\tkind\trows\tfile_bytes_all_lists\tfile\n";
    for (auto& [build, lists] : by_build) {
        auto bytes = build_vocab_z19(lists);
        std::string p = out + "/track_a_vocab_" + build + ".bin";
        write_file(p, bytes);
        std::printf("wrote %s  %zu B (%zu lists)\n", p.c_str(), bytes.size(), lists.size());
        for (auto& [kind, names] : lists) o << build << '\t' << kind << '\t' << names.size() << '\t' << bytes.size() << '\t' << p << '\n';
    }
    return 0;
}
