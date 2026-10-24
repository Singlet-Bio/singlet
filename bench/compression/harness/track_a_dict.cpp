// Track A: train zstd dictionaries for the VOCSC chunk payload, leave-one-study-out.
//
//   track_a_dict [--size 1048576] [--cap-mb 96] [--out-dir results] corpus.tsv corpus_dir
//
// For every corpus sample S and matrix kind K (exon / intron), the training set is the chunk payloads
// (bitmap-packed prefilter bytes, chunk_cols 1024) of kind K from samples that are NOT in S's study and
// have the same row vocabulary (same species/reference build). Samples without any such training data
// (the two mouse brain samples come from one study) are reported and get no dictionary.
// Writes results/track_a_dict_<kind>_<sample>_<size>.bin and the manifest results/track_a_dict_manifest[_<size>].tsv
// (matrix fingerprint -> dict file), which codecs a_vocsc_zstd_dict / track_a_sweep read.
#define ZDICT_STATIC_LINKING_ONLY
#include "common.h"
#include "track_a.h"
#include <iostream>
#include <sstream>

using namespace cs;
using namespace cs::ta;

struct Sample { std::string id, gse; };

int main(int argc, char** argv) {
    size_t dsize = 1u << 20;
    size_t cap_mb = 96;
    std::string out_dir = "results";
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) dsize = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--cap-mb" && i + 1 < argc) cap_mb = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else pos.push_back(a);
    }
    if (pos.size() < 2) { std::cerr << "usage: track_a_dict [opts] corpus.tsv corpus_dir\n"; return 2; }
    std::vector<Sample> samples;
    {
        std::ifstream f(pos[0]);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string id, gse;
            std::getline(ss, id, '\t'); std::getline(ss, gse, '\t');
            if (!id.empty()) samples.push_back({id, gse});
        }
    }
    std::string manifest = out_dir + "/track_a_dict_manifest" + (dsize == (1u << 20) ? "" : "_" + std::to_string(dsize)) + ".tsv";
    std::ofstream man(manifest);   // overwrite
    for (const char* kind : {"exon", "intron"}) {
        // load every matrix of this kind once (packed chunks only), keep vocabulary hash
        struct Item { Sample s; uint64_t vhash; std::string fp; std::vector<Chunk> chunks; };
        std::vector<Item> items;
        for (auto& s : samples) {
            auto M = load_1pz(pos[1] + "/" + s.id + "/" + kind + "_counts.1pz");
            Item it; it.s = s; it.vhash = names_hash(M.rownames); it.fp = fingerprint(M);
            auto P = prepare(M, 1024, nullptr, 2);
            it.chunks = std::move(P.chunks);
            items.push_back(std::move(it));
            std::printf("loaded %s %s: %zu chunks\n", s.id.c_str(), kind, items.back().chunks.size());
        }
        for (auto& held : items) {
            std::vector<const Item*> train;
            for (auto& o : items) if (o.s.gse != held.s.gse && o.vhash == held.vhash) train.push_back(&o);
            if (train.empty()) { std::printf("%s %s: no out-of-study training data with the same vocabulary, no dictionary\n", held.s.id.c_str(), kind); continue; }
            // sample chunks evenly: each training file contributes cap/len(train) bytes, spread over its chunks
            std::vector<uint8_t> buf;
            std::vector<size_t> sizes;
            size_t per_file = cap_mb * 1000000 / train.size();
            for (auto* t : train) {
                size_t total = 0;
                for (auto& c : t->chunks) total += c.packed.size();
                double keep = std::min(1.0, double(per_file) / double(total));
                double acc = 0;
                for (auto& c : t->chunks) {
                    acc += keep;
                    if (acc >= 1.0) { acc -= 1.0; buf.insert(buf.end(), c.packed.begin(), c.packed.end()); sizes.push_back(c.packed.size()); }
                }
            }
            std::vector<uint8_t> dict(dsize);
            ZDICT_fastCover_params_t fp;
            std::memset(&fp, 0, sizeof(fp));
            fp.steps = 6; fp.nbThreads = 1; fp.splitPoint = 0.75; fp.accel = 5; fp.d = 0; fp.k = 0;
            fp.zParams.compressionLevel = 19;
            Timer t;
            size_t r = ZDICT_optimizeTrainFromBuffer_fastCover(dict.data(), dict.size(), buf.data(), sizes.data(), unsigned(sizes.size()), &fp);
            if (ZDICT_isError(r)) { std::printf("%s %s: dictionary training failed: %s\n", held.s.id.c_str(), kind, ZDICT_getErrorName(r)); continue; }
            dict.resize(r);
            std::string fn = std::string("track_a_dict_") + kind + "_" + held.s.id + "_" + std::to_string(dsize) + ".bin";
            write_file(out_dir + "/" + fn, dict);
            man << held.fp << '\t' << fn << '\n';
            std::printf("%s %s: trained on %zu chunks (%zu MB) from %zu out-of-study samples -> %zu byte dict (k=%u d=%u) in %.1f s\n", held.s.id.c_str(), kind,
                        sizes.size(), buf.size() / 1000000, train.size(), dict.size(), fp.k, fp.d, t.ms() / 1000);
        }
    }
    return 0;
}
