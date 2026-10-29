// SPDX-License-Identifier: MIT
// pz2 public API smoke test: header / chunk index / byte source / ISA reporting / column planning.
// Codec round trips live in test_pz2_roundtrip.cpp.
#define SINGLET_TEST_HARNESS_QUIET
#include "test_harness.h"
#include <singlet/pz2/pz2.h>

#include <cstdio>

using namespace singlet::pz2;

int main() {
    // ISA reporting
    CHECK(isa_supported(Isa::Scalar), "scalar always supported");
    CHECK(isa_supported(best_isa()), "best isa is supported");
    CHECK(resolve_isa(Isa::Auto) == selected_isa(), "Auto resolves to selected_isa");
    CHECK(!isa_report().empty(), "isa_report");
    std::printf("%s\n", isa_report().c_str());

    // names helpers
    auto blob = join_names({"a", "bc", ""});
    CHECK(blob.size() == 6 && split_names(blob, 3)[1] == "bc" && split_names(blob, 3)[2].empty(), "join/split names");

    // synthetic header + chunk index: n=1000 columns, 3 chunks of 400/400/200, nnz 5000
    Header h;
    h.codec = CodecId::Generic; h.row_order = 0; h.m = 100; h.n = 1000; h.nnz = 5000; h.chunk_cols = 400; h.n_chunks = 3;
    std::vector<uint8_t> idx;
    const uint32_t fc[4] = {0, 400, 800, 1000};
    const uint64_t nb[4] = {0, 2000, 4000, 5000}, off[4] = {0, 700, 1300, 1500};
    for (int c = 0; c < 4; ++c) {
        detail::put_u64(idx, off[c]); detail::put_u32(idx, c < 3 ? uint32_t(off[c + 1] - off[c]) : 0); detail::put_u32(idx, fc[c]); detail::put_u64(idx, nb[c]);
    }
    h.section[kChunkIndex] = {kHeaderBytes, idx.size()};
    h.section[kChunks] = {kHeaderBytes + idx.size(), 1500};
    h.total_len = kHeaderBytes + idx.size() + 1500;
    h.rows_kind = NamesKind::Inline; h.cols_kind = NamesKind::Cells; h.col_names_id = detail::cells_pack_id();
    h.digest.b[3] = 0xAB;
    auto hb = detail::write_header(h);
    Header g = read_header(hb.data(), hb.size());
    CHECK(g.m == 100 && g.n == 1000 && g.nnz == 5000 && g.n_chunks == 3 && g.codec == CodecId::Generic && g.cols_kind == NamesKind::Cells, "header round trip");
    CHECK(g.digest.b[3] == 0xAB && g.total_len == h.total_len && g.prefix_len() == kHeaderBytes + idx.size(), "header digest / prefix_len");
    auto bad = hb;
    bad[10] ^= 1;
    bool threw = false;
    try { read_header(bad.data(), bad.size()); } catch (const Error& e) { threw = e.code == Errc::Corrupt; }
    CHECK(threw, "header CRC detects corruption");
    threw = false;
    try { read_header(hb.data(), 100); } catch (const Error& e) { threw = e.code == Errc::Truncated; }
    CHECK(threw, "truncated header");

    ChunkIndex ci = read_chunk_index(g, idx.data(), idx.size());
    CHECK(ci.n_chunks() == 3 && ci.chunk_of_column(0) == 0 && ci.chunk_of_column(399) == 0 && ci.chunk_of_column(400) == 1 && ci.chunk_of_column(999) == 2, "chunk_of_column");
    CHECK(ci.cols_in_chunk(2) == 200 && ci.nnz_in_chunk(1) == 2000, "chunk geometry");

    // ByteSource with partial ranges + NeedRange
    std::vector<uint8_t> member(size_t(h.total_len), 0);
    std::memcpy(member.data(), hb.data(), hb.size());
    std::memcpy(member.data() + kHeaderBytes, idx.data(), idx.size());
    ByteSource part;
    part.add(0, member.data(), size_t(g.prefix_len()));
    Matrix mx = Matrix::open(part);
    CHECK(mx.valid() && mx.header().n == 1000 && mx.chunk_index().n_chunks() == 3, "Matrix::open from the prefix only");
    ColumnPlan pl = mx.plan_columns(450, 900);
    CHECK(pl.chunk0 == 1 && pl.chunk1 == 3 && pl.byte_off == g.section[kChunks].off + 700 && pl.byte_len == 800 && pl.max_nnz == 3000 && pl.first_col == 400, "plan_columns spans chunks 1..2");
    ColumnPlan p1 = mx.plan_columns(0, 1);
    CHECK(p1.chunk0 == 0 && p1.chunk1 == 1 && p1.byte_len == 700, "plan_columns single column");
    threw = false;
    try { ByteSource s2; s2.add(0, member.data(), 100); Matrix::open(s2); } catch (const Error& e) { threw = e.code == Errc::NeedRange; }
    CHECK(threw, "Matrix::open on too little data reports NeedRange");
    threw = false;
    try { mx.plan_columns(10, 5); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
    CHECK(threw, "plan_columns rejects reversed range");

    std::printf("test_pz2_api: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
