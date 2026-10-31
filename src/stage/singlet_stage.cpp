// singlet-stage: fetch-tier encoder  .1fq (one or more runs of a sample)  ->  .1fg2  (staged GPU-native read container, include/singlet/fq/fg2_format.h).
//
//   singlet-stage --whitelist-dir D [--whitelist-dir D2 ...] [--accessions SRR1,SRR2] [--level 3] [--scratch DIR] [--json FILE] [--verify] -o OUT.1fg2 RUN1.1fq [RUN2.1fq ...]
//
// The two-step path (SRA -> .1fq -> .1fg2). The one-step path is `singlet download <SRR|file.sra> --stage-out OUT.1fg2` (same stage::Encoder, fed from the
// SRA writer's blocks, no .1fq written or re-read). One source of truth: protocol, geometry, whitelist and R11 correction are the engine's own headers;
// no clipping or trimming here (the engine clips on the packed codes). See stage_encoder.h for the passes.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <singlet/fq/protocol.h>
#include <singlet/fq/reader.h>
#include "stage_encoder.h"
#include "stage_verify.h"

namespace sq = singlet::fq;
using stage::die;

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == "--digest") {
        uint64_t d = 0, n = 0; int rc = stage_verify::digest_file(argv[i + 1], &d, &n);
        std::printf("digest=%016llx reads=%llu\n", (unsigned long long)d, (unsigned long long)n);
        return rc;
    }
    std::vector<std::string> wl_dirs, inputs, accs, parts;
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
        else if (a == "--merge-parts") { while (i + 1 < argc && argv[i + 1][0] != '-') parts.push_back(argv[++i]); }
        else if (a == "--accessions") { std::string s = val(); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); accs.push_back(s.substr(p, q - p)); p = q + 1; } }
        else if (!a.empty() && a[0] == '-') die(2, "unknown option " + a);
        else inputs.push_back(a);
    }
    if (out_path.empty() || (inputs.empty() && parts.empty())) die(2, "usage: singlet-stage --whitelist-dir D -o OUT.1fg2 (RUN1.1fq [RUN2.1fq ...] | --merge-parts PART1 [PART2 ...])");
    if (wl_dirs.empty() && std::getenv("SINGLET_WHITELIST_DIR")) wl_dirs.push_back(std::getenv("SINGLET_WHITELIST_DIR"));
    if (!parts.empty()) {   // multi-run sample built run by run with `singlet download --stage-out PART --stage-part`: one histogram, one table, no .1fq
        stage::PartHdr h0 = stage::Encoder::read_part_header(parts[0]);
        stage::Encoder enc;
        enc.init(h0.protocol_id, h0.bcl, h0.bc0, h0.umil, h0.umi0, h0.has_q != 0, wl_dirs, scratch);
        for (const auto& p : parts) enc.merge_part(p);
        std::string js = enc.finish(out_path, level);
        std::puts(js.c_str());
        if (!json_path.empty()) { std::ofstream jf(json_path); jf << js << "\n"; }
        return 0;
    }

    sq::Reader r0; r0.open(inputs[0].c_str());
    const uint8_t protocol_id = r0.header().protocol_id;
    const sq::CandidateSpec* spec = nullptr;
    for (const auto& s : sq::known_protocols()) if (s.protocol_id == protocol_id) { spec = &s; break; }
    if (!spec) die(6, "unknown protocol id " + std::to_string(protocol_id));
    const int bcl = r0.bc_length(), bc0 = spec->bc_offset, umil = r0.umi_length();
    const int umi0 = spec->umi_len > 0 ? spec->umi_offset : bcl;
    const bool has_q = r0.header().qual_mode != (uint8_t)sq::QualMode::NONE;
    r0.close();

    stage::Encoder enc;
    enc.init(protocol_id, bcl, bc0, umil, umi0, has_q, wl_dirs, scratch);

    for (size_t ri = 0; ri < inputs.size(); ri++) {
        sq::Reader rd; rd.open(inputs[ri].c_str());
        if (rd.header().protocol_id != protocol_id || rd.bc_length() != bcl || rd.umi_length() != umil) die(6, "runs disagree on protocol/geometry: " + inputs[ri]);
        if ((rd.header().qual_mode != (uint8_t)sq::QualMode::NONE) != has_q) die(6, "runs disagree on quality mode: " + inputs[ri]);
        struct stat st; uint64_t bytes = stat(inputs[ri].c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
        enc.begin_run(ri < accs.size() ? accs[ri] : inputs[ri].substr(inputs[ri].find_last_of('/') + 1), bytes);
        sq::DecodedBlock blk;
        while (rd.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                const uint16_t r1l = blk.r1_len(i), r2l = blk.r2_len(i);
                if (r1l < bc0 + bcl || r1l < umi0 + umil || r2l == 0) { enc.note_dropped(); continue; }
                const uint8_t* r1 = blk.r1_seq(i);
                enc.add_read(r1 + bc0, r1 + umi0, blk.r2_seq(i), r2l, has_q ? blk.r2_quality(i) : nullptr);
            }
        }
        enc.end_run();
    }
    std::string js = enc.finish(out_path, level);
    std::puts(js.c_str());
    if (!json_path.empty()) { std::ofstream jf(json_path); jf << js << "\n"; }
    if (do_verify) return stage_verify::verify_file(out_path, inputs, enc.corr, bcl, bc0, umil, umi0, has_q);
    return 0;
}
