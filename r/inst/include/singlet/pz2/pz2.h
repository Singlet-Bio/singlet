// SPDX-License-Identifier: MIT
// singlet::pz2 -- the v2 sparse-matrix member codec ("pz2"), header-only C++17.
//
// Format: docs/format-v2.md section 4 (member layout) and section 5 (codecs).  Notes and status: bench/compression/notes/m_w1.md.
//
// What a pz2 member is: one CSC matrix (m rows = features, n columns = cells/barcodes, nnz non-zeros), cut into column
// chunks that are independently decodable.  Layout, in byte order:
//     header (256 bytes) | chunk index | tables | rows | chunks | cols | kv | extra
// A client that wants any column range needs only the first four parts (`Header::prefix_len()` bytes) plus the chunks that
// cover the range (`Matrix::plan_columns`).  Nothing here does I/O: callers hand the library bytes (`ByteSource`) and
// pre-allocated output buffers (`Output`).
//
// Two codecs share the container:
//   codec 1 ("rans")    class-order 32-lane rANS; integer only; scalar / AVX2 / AVX-512 paths that decode to identical
//                       output; m < 2^21 rows; any u32 value (escapes); the fast one.
//   codec 2 ("generic") per chunk: LEB128 counts, row-id gaps and values, each stream zstd-19; any size; the fallback.
// `CodecId::Auto` picks codec 1 whenever it is allowed (m < 2^21, not an f32-with-fractions matrix) and codec 2 otherwise
// or when codec 2 is smaller (tiny matrices).
//
// Build: C++17, links zstd.  No -march flag is needed; the SIMD paths are compiled with per-function target attributes and
// chosen at run time (cpuid).  `PZ2_ISA=scalar|avx2|avx512` forces a path (never beyond what the CPU supports);
// `DecodeOptions::isa` overrides per call.  On non-x86 targets only the scalar path exists.
//
// Row spaces.  Codec 1 stores rows in "class order" (rows sorted by nnz class descending, canonical id ascending inside a
// class).  `Mode::Native` returns row ids in that stored space plus `Matrix::row_perm()` (stored id -> canonical id); it
// costs no sort.  `Mode::Canonical` returns exactly the legacy matrix (canonical ids ascending in every column), which costs a
// per-column sort.  For codec 2 and for row_order 0 both modes are identical and `row_perm()` is empty (identity).
//
// Thread safety.  All free functions and const members are thread-safe; `Matrix` is cheap to copy (shared immutable state).
#pragma once

#include <array>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace singlet {
namespace pz2 {

// ===========================================================================================================
// constants
// ===========================================================================================================
inline constexpr uint16_t kFormatVersion = 1;
inline constexpr size_t kHeaderBytes = 256;              // fixed header size; member starts with magic "PZ2M"
inline constexpr size_t kChunkRecBytes = 24;             // one chunk-index record
inline constexpr uint32_t kRansMaxRows = (1u << 21) - 1; // codec 1 handles m <= kRansMaxRows (m < 2^21)
inline constexpr uint32_t kMaxChunkNnz = 1u << 27;       // encoder never builds a chunk with more non-zeros than this
/// Default EncodeOptions::chunk_nnz_max: a chunk ends before the column that would take it past this many non-zeros, so a
/// chunk of densely populated columns (the called cells at the start of a UMI-ordered matrix) is not wider than the sparse
/// rest. Readers never depend on it (chunk widths are writer choices, docs/format-v2.md 4.2).
inline constexpr uint32_t kDefaultChunkNnzMax = 49152;

// ===========================================================================================================
// enums
// ===========================================================================================================

/// Element type a matrix is *stored* as.  For 1..3 this is identical to the legacy `.1pz` vt_code.  F32 values travel as
/// u32 bit patterns (codec 1 recodes exactly-integral f32 matrices as integers, see Header::flags).
enum class ValueType : uint8_t { Auto = 0, U8 = 1, U16 = 2, U32 = 3, F32 = 4 };

enum class CodecId : uint16_t { Auto = 0, Rans = 1, Generic = 2 };

/// Instruction-set path of the decoder.  `Auto` means "selected_isa()".
enum class Isa : uint8_t { Scalar = 0, Avx2 = 1, Avx512 = 2, Auto = 255 };

enum class Mode : uint8_t { Native = 0, Canonical = 1 };

/// Element type of a caller buffer holding matrix values.  U8/U16/U32/I32 are numeric conversions of integer values (the
/// decoder fails with Errc::TypeMismatch if a value does not fit); F32/F64 convert numerically from integer matrices.  For
/// a matrix stored as ValueType::F32, F32 and U32 return the stored bit pattern unchanged and F64 widens the float.
enum class DType : uint8_t { U8 = 1, U16 = 2, U32 = 3, F32 = 4, I32 = 5, F64 = 6 };

/// Integer width of indptr / indices buffers.  Row ids are < 2^21, so U32 and I32 hold identical bits; indptr may be 64-bit
/// for matrices with nnz >= 2^31.  `indices` accepts only U32 / I32.
enum class IntType : uint8_t { U32 = 0, I32 = 1, U64 = 2, I64 = 3 };

/// How the row / column names of a member are stored.
enum class NamesKind : uint8_t {
    Inline = 0,      ///< NUL-joined names stored in the member (zstd)
    Pack = 1,        ///< names are exactly the list of the pack with the given 8-byte id; nothing stored
    PackSubset = 2,  ///< names are pack[index[i]]; the index list is stored (varint deltas + zstd)
    Cells = 3,       ///< (columns only) names are the sample `cells` table's barcode column; nothing stored
    None = 4         ///< the matrix has no names at all (count 0; digest hashes zero names).  Opt-in: the default is Inline.
};

/// Section ids used by Header::section.  Physical byte order is: chunk_index, tables, rows, chunks, cols, kv, extra.
enum Section : int { kRows = 0, kCols = 1, kKv = 2, kTables = 3, kChunkIndex = 4, kChunks = 5, kExtra = 6, kNumSections = 7 };

// ===========================================================================================================
// errors
// ===========================================================================================================
enum class Errc {
    BadArgument,    ///< invalid call (null buffers, col range out of order, ...)
    BadMagic,       ///< not a pz2 member
    Unsupported,    ///< valid but not handled (unknown version/codec, m too large for codec 1, ...)
    Truncated,      ///< input shorter than the format requires
    Corrupt,        ///< structurally invalid or CRC mismatch
    NeedRange,      ///< partial ByteSource lacks bytes: Error::need_off / need_len say which member range to fetch
    OutputTooSmall, ///< caller buffer capacity is smaller than required
    TypeMismatch,   ///< a value does not fit the requested output type
    NotImplemented
};

class Error : public std::runtime_error {
public:
    Error(Errc c, const std::string& msg, uint64_t off = 0, uint64_t len = 0)
        : std::runtime_error("pz2: " + msg), code(c), need_off(off), need_len(len) {}
    Errc code;
    uint64_t need_off;  ///< Errc::NeedRange / OutputTooSmall: member byte offset / element count that is missing
    uint64_t need_len;  ///< Errc::NeedRange: number of bytes needed from need_off
};

// ===========================================================================================================
// small value types
// ===========================================================================================================
using PackId = std::array<uint8_t, 8>;  ///< first 8 bytes of a pack's digest16 (docs/format-v2.md section 6)

struct Digest16 {
    std::array<uint8_t, 16> b{};
    std::string hex() const {
        static const char* d = "0123456789abcdef";
        std::string s(32, '0');
        for (int i = 0; i < 16; ++i) { s[2 * i] = d[b[i] >> 4]; s[2 * i + 1] = d[b[i] & 15]; }
        return s;
    }
    bool is_zero() const { for (auto x : b) if (x) return false; return true; }
    bool operator==(const Digest16& o) const { return b == o.b; }
    bool operator!=(const Digest16& o) const { return b != o.b; }
};

/// Row or column names of a matrix as the caller sees them.
struct NamesSpec {
    NamesKind kind = NamesKind::Inline;
    PackId pack_id{};                      ///< Pack / PackSubset
    std::vector<uint32_t> pack_index;      ///< PackSubset: index into the pack's list for every name
    /// Logical names, each followed by a NUL ("a\0b\0"), count = m (rows) or n (columns).  Encoder: stored iff kind==Inline,
    /// otherwise optional and used only to compute the canonical digest.  Decoder: filled iff kind==Inline.
    std::vector<uint8_t> blob;
    uint64_t count = 0;                    ///< number of names the matrix has (m or n)
};
std::vector<uint8_t> join_names(const std::vector<std::string>& names);                        ///< -> NUL-terminated blob
std::vector<std::string> split_names(const std::vector<uint8_t>& blob, size_t count);          ///< inverse of join_names

using KvMap = std::map<std::string, std::string>;  ///< legacy user key/value pairs (sorted by key)

// ===========================================================================================================
// input matrix
// ===========================================================================================================

/// Canonical CSC input.  Row ids strictly ascending inside each column, all < m.  Exactly one of `indptr` (n+1 entries,
/// indptr[0]==0, indptr[n]==nnz; requires nnz < 2^32) or `counts` (n entries; any nnz) is given.  `indices` is U32/I32 bits.
/// `data` holds nnz elements of `data_type` (U8/U16/U32/F32 only); F32 holds float values.  All stored values must be > 0 for
/// meaningful matrices, but explicit zeros are accepted and round-trip (value 0 is a first-class value in both codecs).
struct CSC {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    const uint32_t* indptr = nullptr;
    const uint32_t* counts = nullptr;
    const uint32_t* indices = nullptr;
    const void* data = nullptr;
    DType data_type = DType::U32;
};

// ===========================================================================================================
// encode
// ===========================================================================================================
struct EncodeOptions {
    CodecId codec = CodecId::Auto;
    /// Maximum columns per chunk (the header's nominal `chunk_cols`; a chunk is narrower when it reaches chunk_nnz_max or
    /// kMaxChunkNnz non-zeros, or at the end). 0 = library default: about `chunk_nnz_target` non-zeros per chunk at the
    /// matrix's mean column density (default_chunk_cols()).
    uint32_t chunk_cols = 0;
    uint32_t chunk_nnz_target = 0;   ///< target non-zeros per chunk when chunk_cols == 0; 0 = library default (see notes/m_w1.md)
    /// A chunk ends before the column that would take it past this many non-zeros (a chunk always holds at least one column).
    /// 0 = kDefaultChunkNnzMax; UINT32_MAX = no budget (fixed-width chunks of chunk_cols columns, the behaviour before).
    uint32_t chunk_nnz_max = 0;
    uint8_t classes_per_octave = 4;  ///< codec 1 row-class resolution; 0 = exact frequency rank.  1, 2, 4, 8 are tested.
    uint8_t lanes = 32;              ///< codec 1 rANS lanes (16, 32 or 64).  The AVX paths are tuned for 32.
    uint8_t prob_bits = 10;          ///< codec 1 gap/flag table precision (8..12)
    uint8_t value_prob_bits = 10;    ///< codec 1 residual-value table precision (8..12)
    int ctx_kind = -1;               ///< codec 1 context configuration 0..4; -1 = pick the smallest by estimate
    ValueType value_type = ValueType::Auto;  ///< Auto = smallest of U8/U16/U32 that fits (F32 for F32 input)
    NamesSpec row_names;             ///< kind Inline needs a blob of m names; Pack/PackSubset need pack_id
    NamesSpec col_names;             ///< also allows NamesKind::Cells
    KvMap kv;
    /// Canonical digest (docs/format-v2.md 4.6).  If `digest` is set it is stored as given.  Otherwise, when
    /// `compute_digest` is true, it is computed from the matrix, the logical names (NamesSpec::blob must then be present
    /// for rows and columns even when they are stored as Pack/Cells; if either is missing and its count > 0 the call throws
    /// BadArgument) and kv.  With compute_digest false the digest is left zero.
    bool compute_digest = true;
    bool has_digest = false;
    Digest16 digest;
    int threads = 1;                 ///< encoder worker threads (0 = hardware concurrency, capped at 8)
    int zstd_level = 19;             ///< for the side streams (names, rows, tables)
};

struct EncodeStats {
    CodecId codec = CodecId::Rans;
    ValueType value_type = ValueType::U32;
    uint32_t chunk_cols = 0, n_chunks = 0;
    uint64_t total = 0, header = 0, chunk_index = 0, tables = 0, rows = 0, chunks = 0, cols = 0, kv = 0, extra = 0;
    uint64_t codec1_bytes = 0, codec2_bytes = 0;   ///< sizes of both candidates when Auto tried both (0 = not tried)
};

/// Smallest-payload chunk width policy used when EncodeOptions::chunk_cols == 0.
uint32_t default_chunk_cols(uint32_t n, uint64_t nnz, uint32_t target_nnz = 0);

/// Encode one matrix into a pz2 member.  Deterministic: same input and options give the same bytes on every machine.
std::vector<uint8_t> encode_matrix(const CSC& in, const EncodeOptions& opt = EncodeOptions(), EncodeStats* stats = nullptr);

// ===========================================================================================================
// parsed header and chunk index
// ===========================================================================================================
struct SectionRef { uint64_t off = 0, len = 0; };   ///< byte range relative to the member start

struct Header {
    uint16_t version = kFormatVersion;
    CodecId codec = CodecId::Rans;
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    ValueType value_type = ValueType::U32;
    uint8_t row_order = 1;            ///< 0 = canonical ids stored, 1 = class order
    uint8_t classes_per_octave = 4;
    uint8_t flags = 0;                ///< bit0: F32 matrix coded as integers (all values exactly integral and < 2^24)
    uint32_t chunk_cols = 0, n_chunks = 0;
    uint8_t lanes = 32, prob_bits = 10, value_prob_bits = 10, ctx_kind = 0;   ///< codec 1 parameters
    PackId row_names_id{}, col_names_id{};   ///< pack id prefix (all zero for Inline; "CELLS\0\0\0" for NamesKind::Cells)
    Digest16 digest;
    SectionRef section[kNumSections];
    uint64_t total_len = 0;           ///< byte length of the whole member
    NamesKind rows_kind = NamesKind::Inline, cols_kind = NamesKind::Inline;
    /// Bytes from the member start that `Matrix::open` needs: header + chunk index + tables + rows.
    uint64_t prefix_len() const {
        uint64_t e = kHeaderBytes;
        for (int s : {kChunkIndex, kTables, kRows}) if (section[s].len && section[s].off + section[s].len > e) e = section[s].off + section[s].len;
        return e;
    }
};

/// Parse and validate the 256-byte header (magic, version, header CRC, section bounds).  Needs n >= kHeaderBytes.
Header read_header(const uint8_t* p, size_t n);

struct ChunkRec {
    uint64_t off = 0;        ///< byte offset inside the chunks section
    uint32_t len = 0;        ///< byte length of the chunk stream
    uint32_t first_col = 0;  ///< first column of the chunk
    uint64_t nnz_before = 0; ///< non-zeros in all earlier chunks
};

struct ChunkIndex {
    std::vector<ChunkRec> rec;   ///< n_chunks + 1 records; the last is a sentinel {off = chunks length, len 0, first_col = n, nnz_before = nnz}
    size_t n_chunks() const { return rec.empty() ? 0 : rec.size() - 1; }
    /// Chunk containing column `col` (col < n).
    size_t chunk_of_column(uint32_t col) const {
        size_t lo = 0, hi = n_chunks();
        while (hi - lo > 1) { size_t mid = (lo + hi) / 2; if (rec[mid].first_col <= col) lo = mid; else hi = mid; }
        return lo;
    }
    uint32_t cols_in_chunk(size_t c) const { return rec[c + 1].first_col - rec[c].first_col; }
    uint64_t nnz_in_chunk(size_t c) const { return rec[c + 1].nnz_before - rec[c].nnz_before; }
};

/// Parse the chunk-index section; `p` points at its first byte (Header::section[kChunkIndex]), n is the bytes available.
ChunkIndex read_chunk_index(const Header& h, const uint8_t* p, size_t n);

// ===========================================================================================================
// byte source: a whole member in memory, or any set of fetched ranges of it
// ===========================================================================================================
class ByteSource {
public:
    ByteSource() = default;
    /// A complete member (bytes [0, n)).  The memory is borrowed and must outlive the source.
    ByteSource(const uint8_t* p, size_t n) { add(0, p, n); }
    explicit ByteSource(const std::vector<uint8_t>& v) { add(0, v.data(), v.size()); }
    /// Register bytes [member_offset, member_offset + n) of the member.  Borrowed, not copied.  Ranges may be added in any order.
    void add(uint64_t member_offset, const uint8_t* p, size_t n) { if (n) parts_.push_back({member_offset, p, n}); }
    bool has(uint64_t off, uint64_t len) const { return find(off, len) != nullptr; }
    /// Pointer to member bytes [off, off+len) which must lie inside one registered range; else throws Errc::NeedRange.
    const uint8_t* get(uint64_t off, uint64_t len) const {
        const uint8_t* q = find(off, len);
        if (!q) throw Error(Errc::NeedRange, "byte range not available", off, len);
        return q;
    }
private:
    struct Part { uint64_t off; const uint8_t* p; size_t n; };
    std::vector<Part> parts_;
    const uint8_t* find(uint64_t off, uint64_t len) const {
        if (len == 0) return parts_.empty() ? reinterpret_cast<const uint8_t*>("") : parts_[0].p;
        for (const Part& q : parts_)
            if (off >= q.off && off - q.off <= q.n && len <= q.n - (off - q.off)) return q.p + (off - q.off);
        return nullptr;
    }
};

// ===========================================================================================================
// decode
// ===========================================================================================================

/// Caller-owned output buffers.  Capacities are in elements and are checked (Errc::OutputTooSmall); pass nullptr to skip a
/// buffer.  Whole matrix: indptr n+1, indices / data nnz.  Column range [c0,c1): indptr c1-c0+1 (starting at 0), indices /
/// data at least `ColumnPlan::max_nnz`.
struct Output {
    void* indptr = nullptr;  IntType indptr_type = IntType::U32;  size_t indptr_capacity = 0;
    void* indices = nullptr; IntType indices_type = IntType::U32; size_t indices_capacity = 0;
    void* data = nullptr;    DType data_type = DType::U32;        size_t data_capacity = 0;
};

struct DecodeOptions {
    Mode mode = Mode::Native;
    int threads = 1;           ///< worker threads over chunks (1 = caller's thread; 0 = min(4, hardware concurrency))
    Isa isa = Isa::Auto;       ///< Auto = selected_isa() (honours PZ2_ISA); an unsupported choice falls back to the best supported
};

/// Which bytes to fetch for a column range.
struct ColumnPlan {
    uint32_t col0 = 0, col1 = 0;        ///< requested columns [col0, col1)
    size_t chunk0 = 0, chunk1 = 0;      ///< chunks [chunk0, chunk1) cover the range
    uint64_t byte_off = 0, byte_len = 0;///< ONE contiguous member byte range holding exactly those chunks
    uint64_t max_nnz = 0;               ///< non-zeros in the covering chunks (an upper bound for the range's nnz)
    uint32_t first_col = 0;             ///< first column of chunk0 (output starts at col0, not here)
};

/// An opened member: header, chunk index, entropy tables and row permutation parsed from the prefix.
class Matrix {
public:
    struct Impl;
    Matrix() = default;
    /// Parse header, chunk index, tables and rows from `src`, which must contain bytes [0, Header::prefix_len()).  Throws
    /// Errc::NeedRange (with the missing range) if it does not.  Builds the decoding tables (about 3 ms for large ones).
    static Matrix open(const ByteSource& src);
    bool valid() const { return static_cast<bool>(impl_); }
    const Header& header() const;
    const ChunkIndex& chunk_index() const;
    /// Stored row id -> canonical row id (size m) when row_order == 1; empty (identity) otherwise.
    const std::vector<uint32_t>& row_perm() const;
    /// Canonical -> stored (inverse of row_perm), built lazily; empty for identity.
    const std::vector<uint32_t>& row_inv() const;
    /// Names / kv.  These read the cols / kv / rows sections, which are not part of the prefix for cols and kv.
    NamesSpec row_names(const ByteSource& src) const;
    NamesSpec col_names(const ByteSource& src) const;
    KvMap kv(const ByteSource& src) const;
    /// Chunks and bytes needed for columns [col0, col1).  Pure index arithmetic.
    ColumnPlan plan_columns(uint32_t col0, uint32_t col1) const;
    /// Decode every column.  `src` must hold the whole chunks section (a complete member is the usual case).
    void decode(const ByteSource& src, Output& out, const DecodeOptions& opt = DecodeOptions()) const;
    /// Decode columns [col0, col1).  `src` needs only the plan's byte range.  Returns the exact nnz of the range.
    uint64_t decode_columns(const ByteSource& src, uint32_t col0, uint32_t col1, Output& out, const DecodeOptions& opt = DecodeOptions()) const;
    /// Exact column counts of [col0, col1) (n entries).  Cheap: reads only the count streams of the covering chunks.
    std::vector<uint32_t> column_counts(const ByteSource& src, uint32_t col0, uint32_t col1) const;
private:
    std::shared_ptr<const Impl> impl_;
};

/// One-shot convenience for a complete member in memory.
void decode(const uint8_t* member, size_t n, Output& out, const DecodeOptions& opt = DecodeOptions());

// ===========================================================================================================
// canonical digest (docs/format-v2.md 4.6)
// ===========================================================================================================

/// SHA-256 over the canonical content, first 16 bytes.  `m`, `n`, `nnz`, indptr/counts, indices, data come from `csc`;
/// `data` of any DType is widened to the u32 value (F32: the bit pattern).  Names are the logical NUL-joined blobs (count m
/// and n); kv is hashed sorted.  Identical digests mean identical canonical matrices.
Digest16 canonical_digest(const CSC& csc, ValueType vt, const std::vector<uint8_t>& row_names_blob, const std::vector<uint8_t>& col_names_blob, const KvMap& kv);

// ===========================================================================================================
// instruction set selection
// ===========================================================================================================
const char* isa_name(Isa isa);                 ///< "scalar", "avx2", "avx512"
bool isa_compiled(Isa isa);                    ///< is the path part of this build (always true for Scalar)
bool isa_supported(Isa isa);                   ///< compiled in AND reported by cpuid/XGETBV
Isa best_isa();                                ///< best supported path
Isa selected_isa();                            ///< best_isa() unless PZ2_ISA forces a lower one (read once)
Isa resolve_isa(Isa requested);                ///< Auto -> selected_isa(); unsupported -> best supported not above the request
std::string isa_report();                      ///< one line: selected, best, compiled paths, PZ2_ISA value

}  // namespace pz2
}  // namespace singlet

// Implementation, split by topic (all inline; included once, in this order).
#include "detail/common.h"
#include "detail/isa.h"
#include "detail/simd.h"
#include "detail/container.h"
#include "detail/values.h"
#include "detail/sha256.h"
#include "detail/rans1.h"
#include "detail/rans1_enc.h"
#include "detail/rans1_dec.h"
#include "detail/rans1_avx512.h"
#include "detail/rans1_avx2.h"
#include "detail/codec2.h"
#include "detail/api.h"
