// SPDX-License-Identifier: MIT
// w4_bugs: exercise the PROPOSED fix (results/w4_bugs_patches/psi_writer_and_donor_priority.patch) on a shipped sj matrix.
// Build with -Icorpus_w4_bugs/fix/b/include first so that splice_psi.h is the patched one.
//   w4_bugs_psi_fix <sj_counts.1pz> <out_prefix>  -> <out_prefix>.fix_f32.bin, <out_prefix>.fix_bp.1pz
#include <singlet/pileup/pz_reader.h>
#include <singlet/pileup/splice_psi.h>
#include <cmath>
#include <fstream>
struct CSC { uint32_t nrows, ncols; std::vector<int32_t> indptr, indices; std::vector<uint16_t> data; };
int main(int, char** argv) {
    auto r = singlet::pz::read_1pz(argv[1]);
    CSC c; c.nrows = r.m; c.ncols = r.n; c.indptr.assign(r.indptr.begin(), r.indptr.end()); c.indices.assign(r.indices.begin(), r.indices.end());
    c.data.assign(r.data.begin(), r.data.end());
    auto psi = singlet::compute_psi(c, r.rownames);
    std::string pre = argv[2];
    { std::ofstream f(pre + ".fix_f32.bin", std::ios::binary); f.write((const char*)psi.data.data(), psi.data.size() * 4); }
    // identical to the lines added to export.h by the patch
    std::vector<uint16_t> psi_bp(psi.data.size());
    for (size_t i = 0; i < psi.data.size(); ++i) psi_bp[i] = static_cast<uint16_t>(std::lround(psi.data[i] * 10000.0f));
    std::map<std::string, std::string> psi_meta; psi_meta["value_scale"] = "10000"; psi_meta["value_kind"] = "psi";
    return singlet::pz::write_1pz<uint16_t>(pre + ".fix_bp.1pz", psi.nrows, psi.ncols, psi.indptr, psi.indices, psi_bp, r.rownames, r.colnames, 3, 1024, 1, psi_meta) ? 0 : 1;
}
