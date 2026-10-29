// SPDX-License-Identifier: MIT
// Idea 3, part B driver: joint exon + intron coding of one sample.
//
//   bin/idea3_joint.exe [--reps 3] [--out results/idea3_joint.tsv] corpus/<id>/exon_counts.1pz corpus/<id>/intron_counts.1pz [...pairs]
//
// For each (exon, intron) pair:  exon coded with the idea3 gene->interval codec (self-described tables); the intron matrix coded
// (a) independently with the same codec, (b) conditionally on the decoded exon matrix.  Both matrices are decoded and verified
// against the originals (indptr/indices/data/names).  One TSV row per codec with the same column style as the harness.
#include "common.h"
#include "ref_zstd_csc.h"
#include "idea3_joint.h"
#include "idea3_time.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    size_t start = (t == std::string::npos) ? 0 : t + 1;
    return p.substr(start, s - start);
}
static std::string base_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

using u64 = uint64_t; using u32 = uint32_t;
struct Row { std::string sample, member, codec; size_t orig = 0, enc = 0; u64 nnz = 0; u32 m = 0, n = 0; double enc_ms = 0, dec_ms = 0, enc_cpu = 0, dec_cpu = 0; bool ok = false; i3::LastStats ls; };

static void emit(std::ofstream& tsv, const Row& r) {
    double nnzd = (double)r.nnz;
    tsv << r.sample << '\t' << r.member << '\t' << r.codec << '\t' << r.orig << '\t' << r.enc << '\t' << (double)r.enc / (double)r.orig << '\t'
        << r.enc * 8.0 / nnzd << '\t' << r.nnz << '\t' << r.m << '\t' << r.n << '\t' << r.enc_ms << '\t' << r.dec_ms << '\t'
        << nnzd / r.dec_ms / 1e3 << '\t' << (r.ok ? 1 : 0) << '\t' << -1 << '\t' << 0 << '\t'
        << r.ls.payload_bytes << '\t' << r.ls.payload_bytes * 8.0 / nnzd << '\t' << r.ls.names_bytes << '\t' << r.ls.side_bytes << '\t'
        << r.ls.bits[i3::C_K] << '\t' << r.ls.bits[i3::C_PRES] << '\t' << r.ls.bits[i3::C_CNT] << '\t' << r.ls.bits[i3::C_SPLIT] << '\t'
        << r.ls.nnz_gene << '\t' << r.ls.umi << '\t' << r.enc_cpu << '\t' << r.dec_cpu << '\t' << nnzd / r.dec_cpu / 1e3 << '\n';
    tsv.flush();
}

int main(int argc, char** argv) {
    std::vector<std::string> files;
    std::string out = "results/idea3_joint.tsv";
    int reps = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else files.push_back(a);
    }
    if (files.size() < 2 || files.size() % 2) { std::cerr << "usage: idea3_joint [--out tsv] [--reps n] exon.1pz intron.1pz [...]\n"; return 2; }
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tratio_vs_1pz\tbits_per_nnz\tnnz\tm\tn\t"
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\t"
               "payload_bytes\tpayload_bits_per_nnz\tnames_bytes\tside_bytes\tk_bits\tpres_bits\tcnt_bits\tsplit_bits\tnnz_gene\tumi\tenc_cpu_ms\tdec_cpu_ms_min\tdec_cpu_Mnnz_s\n";
    for (size_t pi = 0; pi < files.size(); pi += 2) {
        const std::string &fe = files[pi], &fi = files[pi + 1];
        CanonicalMatrix E = load_1pz(fe), I = load_1pz(fi);
        std::string sample = parent_name(fe);
        if (E.colnames != I.colnames) { std::printf("== %s: colnames differ between exon and intron; skipping\n", sample.c_str()); continue; }
        size_t origE = read_file(fe).size(), origI = read_file(fi).size();
        std::printf("== %s  exon nnz=%llu  intron nnz=%llu  n=%u\n", sample.c_str(), (unsigned long long)E.nnz, (unsigned long long)I.nnz, E.n);

        // exon
        i3::Params P;
        Row rE; rE.sample = sample; rE.member = "exon_counts.1pz"; rE.codec = "idea3_joint_exon_self"; rE.orig = origE; rE.nnz = E.nnz; rE.m = E.m; rE.n = E.n;
        Timer t0; double cp0 = i3::cpu_ms();
        auto encE = i3::encode_matrix(E, P, nullptr);
        rE.enc_ms = t0.ms(); rE.enc_cpu = i3::cpu_ms() - cp0; rE.ls = i3::last_stats(); rE.enc = encE.size();
        CanonicalMatrix Ed; double dmin = 1e300, cmin = 1e300;
        for (int r = 0; r < reps; ++r) { Timer t; double c0 = i3::cpu_ms(); Ed = i3::decode_matrix(encE, P, nullptr); dmin = std::min(dmin, t.ms()); cmin = std::min(cmin, i3::cpu_ms() - c0); }
        rE.dec_ms = dmin; rE.dec_cpu = cmin; rE.ok = (Ed == E);
        std::printf("  exon  %-26s payload %9zu B (%.3f b/nnz)  total %9zu B  dec %.0f ms  %s\n", rE.codec.c_str(), rE.ls.payload_bytes, rE.ls.payload_bytes * 8.0 / E.nnz, rE.enc, rE.dec_ms, rE.ok ? "OK" : "FAIL");
        emit(tsv, rE);

        // intron independent
        Row rI; rI.sample = sample; rI.member = "intron_counts.1pz"; rI.codec = "idea3_joint_intron_indep"; rI.orig = origI; rI.nnz = I.nnz; rI.m = I.m; rI.n = I.n;
        Timer t1; cp0 = i3::cpu_ms();
        auto encI = i3::encode_matrix(I, P, nullptr);
        rI.enc_ms = t1.ms(); rI.enc_cpu = i3::cpu_ms() - cp0; rI.ls = i3::last_stats(); rI.enc = encI.size();
        CanonicalMatrix Id; dmin = 1e300; cmin = 1e300;
        for (int r = 0; r < reps; ++r) { Timer t; double c0 = i3::cpu_ms(); Id = i3::decode_matrix(encI, P, nullptr); dmin = std::min(dmin, t.ms()); cmin = std::min(cmin, i3::cpu_ms() - c0); }
        rI.dec_ms = dmin; rI.dec_cpu = cmin; rI.ok = (Id == I);
        std::printf("  intron %-25s payload %9zu B (%.3f b/nnz)  total %9zu B  dec %.0f ms  %s\n", rI.codec.c_str(), rI.ls.payload_bytes, rI.ls.payload_bytes * 8.0 / I.nnz, rI.enc, rI.dec_ms, rI.ok ? "OK" : "FAIL");
        emit(tsv, rI);

        // intron conditional on exon (decoder uses the DECODED exon matrix)
        for (int cond = 0; cond < 2; ++cond) {
            i3::JointOpts o; o.use_cond = (cond == 1);
            Row rC; rC.sample = sample; rC.member = "intron_counts.1pz"; rC.codec = cond ? "idea3_joint_intron_cond" : "idea3_joint_intron_nocond"; rC.orig = origI; rC.nnz = I.nnz; rC.m = I.m; rC.n = I.n;
            Timer t2; cp0 = i3::cpu_ms();
            auto encC = i3::encode_intron_cond(I, E, o);
            rC.enc_ms = t2.ms(); rC.enc_cpu = i3::cpu_ms() - cp0; rC.ls = i3::last_stats(); rC.enc = encC.size();
            CanonicalMatrix Cd; dmin = 1e300; cmin = 1e300;
            for (int r = 0; r < reps; ++r) { Timer t; double c0 = i3::cpu_ms(); Cd = i3::decode_intron_cond(encC, Ed, o); dmin = std::min(dmin, t.ms()); cmin = std::min(cmin, i3::cpu_ms() - c0); }
            rC.dec_ms = dmin; rC.dec_cpu = cmin; rC.ok = (Cd == I);
            std::printf("  intron %-25s payload %9zu B (%.3f b/nnz; K %.3f P %.3f C %.3f S %.3f side %.3f)  total %9zu B  dec %.0f ms  %s\n", rC.codec.c_str(), rC.ls.payload_bytes,
                        rC.ls.payload_bytes * 8.0 / I.nnz, rC.ls.bits[i3::C_K] / I.nnz, rC.ls.bits[i3::C_PRES] / I.nnz, rC.ls.bits[i3::C_CNT] / I.nnz, rC.ls.bits[i3::C_SPLIT] / I.nnz,
                        rC.ls.bits[i3::C_SIDE] / I.nnz, rC.enc, rC.dec_ms, rC.ok ? "OK" : "FAIL");
            emit(tsv, rC);
            if (cond) {
                double saved = (double)rI.ls.payload_bytes - (double)rC.ls.payload_bytes;
                std::printf("  => conditioning saves %.0f bytes of intron payload = %.3f bits per intron nnz (%.2f%%); joint payload %.0f B vs separate %.0f B\n",
                            saved, saved * 8.0 / I.nnz, 100.0 * saved / rI.ls.payload_bytes, (double)rE.ls.payload_bytes + rC.ls.payload_bytes, (double)rE.ls.payload_bytes + rI.ls.payload_bytes);
            }
        }
        std::fflush(stdout);
    }
    return 0;
}
