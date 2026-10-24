// SPDX-License-Identifier: MIT
// w4_bugs: drive the PRODUCTION compute_psi() (include/singlet/pileup/splice_psi.h) and the PRODUCTION write_1pz()
// (pz_writer.h) on a shipped sj_counts.1pz, so that every statement about what "the in-tree writer" does is measured.
//
//   w4_bugs_psi <sj_counts.1pz> <out_prefix>
// writes
//   <out_prefix>.psi_f32.bin    float32 PSI in the order compute_psi emits them (cols outer, rows as in sj CSC)
//   <out_prefix>.psi_idx.bin    int32 row indices, <out_prefix>.psi_ptr.bin int32 indptr (n+1)
//   <out_prefix>.intree.1pz     what export.h:1123 -> atomic_write_1pz -> write_1pz<float> writes today (truncating cast)
//   <out_prefix>.events.tsv     event_id, type, junction names (as splice_events.tsv would)
//   <out_prefix>.assign.tsv     junction row, name, event_id actually used by compute_psi (junction_event[]), "" if none
#include <singlet/pileup/pz_reader.h>
#include <singlet/pileup/splice_psi.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

struct CSC {  // the minimal shape compute_psi requires (SparseAccumulator::CSCMatrix)
    uint32_t nrows, ncols;
    std::vector<int32_t> indptr, indices;
    std::vector<uint16_t> data;  // the in-tree sj matrix is uint16 (splice_psi.h header comment)
};

template <typename T>
static void dump(const std::string& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s sj.1pz out_prefix\n", argv[0]); return 2; }
    auto r = singlet::pz::read_1pz(argv[1]);
    CSC c;
    c.nrows = r.m; c.ncols = r.n;
    c.indptr.assign(r.indptr.begin(), r.indptr.end());
    c.indices.assign(r.indices.begin(), r.indices.end());
    c.data.resize(r.data.size());
    for (size_t i = 0; i < r.data.size(); ++i) {
        if (r.data[i] > 65535) { std::fprintf(stderr, "sj value %u > uint16\n", r.data[i]); return 3; }
        c.data[i] = static_cast<uint16_t>(r.data[i]);
    }
    auto psi = singlet::compute_psi(c, r.rownames);
    std::string pre = argv[2];
    dump(pre + ".psi_f32.bin", psi.data);
    dump(pre + ".psi_idx.bin", psi.indices);
    dump(pre + ".psi_ptr.bin", psi.indptr);

    // The in-tree export path: export.h:1123 atomic_write_1pz -> write_1pz<float>(..., psi.data, ...)
    bool ok = singlet::pz::write_1pz<float>(pre + ".intree.1pz", psi.nrows, psi.ncols, psi.indptr, psi.indices,
                                            psi.data, r.rownames, r.colnames, 3, 1024, 1, {});
    if (!ok) return 4;

    // The variant that reproduces the shipped bytes: round-half-even before the (truncating) cast.
    {
        std::vector<float> rd(psi.data.size());
        for (size_t i = 0; i < rd.size(); ++i) rd[i] = std::nearbyint(psi.data[i]);
        singlet::pz::write_1pz<float>(pre + ".rint.1pz", psi.nrows, psi.ncols, psi.indptr, psi.indices, rd,
                                      r.rownames, r.colnames, 3, 1024, 1, {});
    }
    singlet::write_splice_events(pre + ".events.tsv", psi.events, r.rownames);
    // Re-derive which event each junction was assigned to (same loop as compute_psi, which keeps it private).
    std::vector<int32_t> je(r.m, -1);
    for (int32_t e = 0; e < (int32_t)psi.events.size(); ++e)
        for (uint32_t j : psi.events[e].junction_indices)
            if (j < je.size() && je[j] == -1) je[j] = e;
    std::ofstream a(pre + ".assign.tsv");
    a << "row\tname\tevent_id\n";
    for (uint32_t j = 0; j < r.m; ++j)
        a << j << "\t" << r.rownames[j] << "\t" << (je[j] >= 0 ? psi.events[je[j]].event_id : std::string()) << "\n";
    std::fprintf(stderr, "n_events=%zu psi_entries=%zu sj_nnz=%zu\n", psi.events.size(), psi.data.size(), r.data.size());
    return 0;
}
