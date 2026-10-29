// SPDX-License-Identifier: MIT
// second translation unit of test_pz2_multi_tu
#include <singlet/pz2/pz2.h>
using namespace singlet::pz2;
int helper_b() {
    std::vector<uint32_t> ptr{0, 2}, idx{1, 5}, val{3, 1};
    CSC c; c.m = 10; c.n = 1; c.nnz = 2; c.indptr = ptr.data(); c.indices = idx.data(); c.data = val.data();
    EncodeOptions o; o.row_names.kind = NamesKind::None; o.col_names.kind = NamesKind::None;
    auto b = encode_matrix(c, o);
    std::vector<uint32_t> ip(2), ix(2), dv(2);
    Output out; out.indptr = ip.data(); out.indptr_capacity = 2; out.indices = ix.data(); out.indices_capacity = 2; out.data = dv.data(); out.data_capacity = 2;
    DecodeOptions d; d.mode = Mode::Canonical;
    decode(b.data(), b.size(), out, d);
    return (ix == idx && dv == val && ip == ptr) ? 0 : 1;
}
