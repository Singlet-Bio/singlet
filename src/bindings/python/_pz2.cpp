// SPDX-License-Identifier: MIT
/**
 * @file _pz2.cpp
 * @brief pybind11 bindings for the store matrix codec (include/singlet/pz2/pz2.h): module `singlet._pz2`.
 *
 * Reading (zero-copy where the format allows):
 *   read_header(buf)                        -> dict           parse the 256-byte member header
 *   Member(source, fetch=None)              -> opened member  header, chunk index, tables and row permutation from the prefix
 *       .plan(c0, c1)                       -> dict           chunk run and the ONE member byte range to fetch for columns [c0, c1)
 *       .add_range(off, buf)                                  register fetched member bytes (borrowed, kept alive)
 *       .decode(c0, c1, mode=, threads=, dtype=, cols=, gene_map=, n_genes=) -> dict(indptr, indices, data, ...)
 *       .row_names() .col_names() .kv()                       names / key-values (fetch callable supplies missing ranges)
 *   decode(buf, ...)                        one-shot whole-matrix decode of a complete member
 *   decode_columns(source_or_fetch, c0, c1, ...)
 *
 * Output arrays are allocated once by numpy and filled by the decoder: indptr int32 (int64 when nnz >= 2^31), indices int32,
 * data in the stored width (uint8 / uint16 / uint32 / float32) or the requested dtype.  The GIL is released while decoding.
 * mode: 'native' (rows in the stored class order; `row_perm` maps stored -> canonical), 'canonical' (legacy row order, costs a
 * per-column sort) or 'gene' (rows aggregated into genes through `gene_map`, a canonical-row -> gene array; sums are int32).
 *
 * Writing (for the migration):
 *   encode(indptr, indices, data, m, n, **options) -> (bytes, stats)
 *   digest(indptr, indices, data, m, n, value_type, row_names, col_names, kv) -> bytes (16)
 *
 * Build: `pip install .` (setup.py) or src/bindings/python/build_pz2_mingw.sh (Rtools GCC, no MSVC).
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <singlet/pz2/pz2.h>
#include <singlet/pz2_client.h>

namespace py = pybind11;
namespace pz = singlet::pz2;
namespace pc = singlet::pz2c;

// ============================================================================
// errors
// ============================================================================
static PyObject* g_pz2_error = nullptr;      // singlet._pz2.Pz2Error (RuntimeError)
static PyObject* g_pz2_needrange = nullptr;  // singlet._pz2.NeedRange (Pz2Error): .off, .len

static const char* errc_name(pz::Errc c) {
    switch (c) {
        case pz::Errc::BadArgument: return "BadArgument";
        case pz::Errc::BadMagic: return "BadMagic";
        case pz::Errc::Unsupported: return "Unsupported";
        case pz::Errc::Truncated: return "Truncated";
        case pz::Errc::Corrupt: return "Corrupt";
        case pz::Errc::NeedRange: return "NeedRange";
        case pz::Errc::OutputTooSmall: return "OutputTooSmall";
        case pz::Errc::TypeMismatch: return "TypeMismatch";
        case pz::Errc::NotImplemented: return "NotImplemented";
    }
    return "Unknown";
}

static void translate_pz_error(const pz::Error& e) {
    PyObject* cls = e.code == pz::Errc::NeedRange ? g_pz2_needrange : g_pz2_error;
    PyObject* inst = PyObject_CallFunction(cls, "s", e.what());
    if (!inst) return;
    PyObject* code = PyUnicode_FromString(errc_name(e.code));
    PyObject_SetAttrString(inst, "code", code);
    Py_XDECREF(code);
    PyObject* off = PyLong_FromUnsignedLongLong(e.need_off);
    PyObject* len = PyLong_FromUnsignedLongLong(e.need_len);
    PyObject_SetAttrString(inst, "off", off);
    PyObject_SetAttrString(inst, "len", len);
    Py_XDECREF(off);
    Py_XDECREF(len);
    PyErr_SetObject(cls, inst);
    Py_DECREF(inst);
}

// ============================================================================
// buffers
// ============================================================================
/// RAII view of a Python bytes-like object (bytes, bytearray, memoryview, numpy uint8 array, mmap).
struct PyBuf {
    Py_buffer view{};
    bool ok = false;
    explicit PyBuf(py::handle h) {
        if (PyObject_GetBuffer(h.ptr(), &view, PyBUF_SIMPLE) != 0) throw py::error_already_set();
        ok = true;
    }
    PyBuf(const PyBuf&) = delete;
    PyBuf& operator=(const PyBuf&) = delete;
    ~PyBuf() {
        if (ok) {
            PyGILState_STATE s = PyGILState_Ensure();
            PyBuffer_Release(&view);
            PyGILState_Release(s);
        }
    }
    const uint8_t* data() const { return static_cast<const uint8_t*>(view.buf); }
    size_t size() const { return size_t(view.len); }
};

// ============================================================================
// small helpers
// ============================================================================
static std::string lower(std::string s) {
    for (auto& c : s) c = char(::tolower((unsigned char)c));
    return s;
}

static pz::Isa parse_isa(const py::object& o) {
    if (o.is_none()) return pz::Isa::Auto;
    std::string s = lower(py::cast<std::string>(o));
    if (s == "auto" || s.empty()) return pz::Isa::Auto;
    if (s == "scalar") return pz::Isa::Scalar;
    if (s == "avx2") return pz::Isa::Avx2;
    if (s == "avx512") return pz::Isa::Avx512;
    throw py::value_error("isa must be 'auto', 'scalar', 'avx2' or 'avx512'");
}

static pz::Mode parse_mode(const std::string& s0, bool& gene) {
    std::string s = lower(s0);
    gene = false;
    if (s == "native") return pz::Mode::Native;
    if (s == "canonical") return pz::Mode::Canonical;
    if (s == "gene") { gene = true; return pz::Mode::Native; }
    throw py::value_error("mode must be 'native', 'canonical' or 'gene'");
}

struct DtypeSpec {
    pz::DType dt;
    const char* name;
    size_t size;
};

static DtypeSpec stored_dtype(pz::ValueType vt) {
    switch (vt) {
        case pz::ValueType::U8: return {pz::DType::U8, "uint8", 1};
        case pz::ValueType::U16: return {pz::DType::U16, "uint16", 2};
        case pz::ValueType::U32: return {pz::DType::U32, "uint32", 4};
        case pz::ValueType::F32: return {pz::DType::F32, "float32", 4};
        default: break;
    }
    return {pz::DType::U32, "uint32", 4};
}

static DtypeSpec parse_dtype(const py::object& o, pz::ValueType stored) {
    if (o.is_none()) return stored_dtype(stored);
    std::string s;
    if (py::isinstance<py::str>(o)) s = py::cast<std::string>(o);
    else s = py::cast<std::string>(py::str(py::module_::import("numpy").attr("dtype")(o).attr("name")));
    s = lower(s);
    if (s == "native" || s == "stored") return stored_dtype(stored);
    if (s == "uint8" || s == "u8") return {pz::DType::U8, "uint8", 1};
    if (s == "uint16" || s == "u16") return {pz::DType::U16, "uint16", 2};
    if (s == "uint32" || s == "u32") return {pz::DType::U32, "uint32", 4};
    if (s == "int32" || s == "i32") return {pz::DType::I32, "int32", 4};
    if (s == "float32" || s == "f32") return {pz::DType::F32, "float32", 4};
    if (s == "float64" || s == "f64" || s == "double") return {pz::DType::F64, "float64", 8};
    throw py::value_error("dtype must be one of native, uint8, uint16, uint32, int32, float32, float64");
}

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; ++i) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}

static const char* value_type_name(pz::ValueType v) {
    switch (v) {
        case pz::ValueType::U8: return "u8";
        case pz::ValueType::U16: return "u16";
        case pz::ValueType::U32: return "u32";
        case pz::ValueType::F32: return "f32";
        default: return "auto";
    }
}
static const char* names_kind_name(pz::NamesKind k) {
    switch (k) {
        case pz::NamesKind::Inline: return "inline";
        case pz::NamesKind::Pack: return "pack";
        case pz::NamesKind::PackSubset: return "pack_subset";
        case pz::NamesKind::Cells: return "cells";
        case pz::NamesKind::None: return "none";
    }
    return "inline";
}

static py::dict header_dict(const pz::Header& h) {
    py::dict d;
    d["version"] = int(h.version);
    d["codec"] = int(h.codec);
    d["m"] = uint64_t(h.m);
    d["n"] = uint64_t(h.n);
    d["nnz"] = uint64_t(h.nnz);
    d["value_type"] = value_type_name(h.value_type);
    d["value_dtype"] = stored_dtype(h.value_type).name;
    d["row_order"] = int(h.row_order);
    d["classes_per_octave"] = int(h.classes_per_octave);
    d["flags"] = int(h.flags);
    d["chunk_cols"] = uint64_t(h.chunk_cols);
    d["n_chunks"] = uint64_t(h.n_chunks);
    d["lanes"] = int(h.lanes);
    d["prob_bits"] = int(h.prob_bits);
    d["value_prob_bits"] = int(h.value_prob_bits);
    d["ctx_kind"] = int(h.ctx_kind);
    d["row_names_kind"] = names_kind_name(h.rows_kind);
    d["col_names_kind"] = names_kind_name(h.cols_kind);
    d["row_names_id"] = hex(h.row_names_id.data(), 8);
    d["col_names_id"] = hex(h.col_names_id.data(), 8);
    d["digest"] = h.digest.hex();
    d["total_len"] = uint64_t(h.total_len);
    d["prefix_len"] = uint64_t(h.prefix_len());
    static const char* sec[pz::kNumSections] = {"rows", "cols", "kv", "tables", "chunk_index", "chunks", "extra"};
    py::dict s;
    for (int i = 0; i < pz::kNumSections; ++i) s[sec[i]] = py::make_tuple(uint64_t(h.section[i].off), uint64_t(h.section[i].len));
    d["sections"] = s;
    return d;
}

template <class T>
static py::array_t<T> make_array(size_t n) {
    return py::array_t<T>(py::ssize_t(n));
}

/// Shrink a freshly allocated, C-contiguous 1-D numpy array in place (numpy realloc).
static void shrink(py::array& a, size_t n) {
    if (size_t(a.size()) == n) return;
    py::ssize_t shape = py::ssize_t(n);
    a.resize(std::vector<py::ssize_t>{shape}, false);
}

// ============================================================================
// Member
// ============================================================================
class PyMember {
public:
    PyMember(py::object source, py::object fetch, size_t prefix_hint) : fetch_(std::move(fetch)), prefix_hint_(prefix_hint) {
        if (!source.is_none()) add_range(0, source);
        open_loop();
    }

    void add_range(uint64_t off, py::object buf) {
        auto pb = std::make_unique<PyBuf>(buf);
        if (pb->size() == 0) return;
        Rng r{off, pb->data(), pb->size()};
        bufs_.push_back(std::move(pb));
        rngs_.push_back(r);
    }

    bool has(uint64_t off, uint64_t len) const { return snapshot().has(off, len); }

    // ---- header-level accessors
    py::dict header() const { return header_dict(mx_.header()); }
    uint32_t m() const { return mx_.header().m; }
    uint32_t n() const { return mx_.header().n; }
    uint64_t nnz() const { return mx_.header().nnz; }
    uint64_t prefix_len() const { return mx_.header().prefix_len(); }
    std::string value_dtype() const { return stored_dtype(mx_.header().value_type).name; }

    py::object row_perm() const {
        const auto& p = mx_.row_perm();
        if (p.empty()) return py::none();
        auto a = make_array<uint32_t>(p.size());
        std::memcpy(a.mutable_data(), p.data(), p.size() * sizeof(uint32_t));
        return std::move(a);
    }

    py::array chunk_index() const {
        const auto& ix = mx_.chunk_index();
        py::array_t<uint64_t> a({py::ssize_t(ix.rec.size()), py::ssize_t(4)});
        auto* o = a.mutable_data();
        for (size_t i = 0; i < ix.rec.size(); ++i) {
            o[4 * i] = ix.rec[i].off;
            o[4 * i + 1] = ix.rec[i].len;
            o[4 * i + 2] = ix.rec[i].first_col;
            o[4 * i + 3] = ix.rec[i].nnz_before;
        }
        return std::move(a);
    }

    py::dict plan(uint32_t c0, uint32_t c1) const {
        pz::ColumnPlan p = mx_.plan_columns(c0, c1);
        py::dict d;
        d["col0"] = p.col0;
        d["col1"] = p.col1;
        d["chunk0"] = p.chunk0;
        d["chunk1"] = p.chunk1;
        d["byte_off"] = p.byte_off;
        d["byte_len"] = p.byte_len;
        d["max_nnz"] = p.max_nnz;
        d["first_col"] = p.first_col;
        return d;
    }

    py::array column_counts(uint32_t c0, uint32_t c1) {
        pz::ColumnPlan p = mx_.plan_columns(c0, c1);
        ensure(p.byte_off, p.byte_len);
        pz::ByteSource src = snapshot();
        std::vector<uint32_t> v;
        {
            py::gil_scoped_release nogil;
            v = mx_.column_counts(src, c0, c1);
        }
        auto a = make_array<uint32_t>(v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(uint32_t));
        return std::move(a);
    }

    // ---- names
    py::dict names_dict(bool rows) {
        pz::NamesSpec ns;
        for (int tries = 0; tries < 8; ++tries) {
            pz::ByteSource src = snapshot();
            try {
                py::gil_scoped_release nogil;
                ns = rows ? mx_.row_names(src) : mx_.col_names(src);
                break;
            } catch (const pz::Error& e) {
                if (e.code != pz::Errc::NeedRange || fetch_.is_none()) throw;
                ensure(e.need_off, e.need_len);
            }
        }
        py::dict d;
        d["kind"] = names_kind_name(ns.kind);
        d["pack_id"] = hex(ns.pack_id.data(), 8);
        d["count"] = uint64_t(ns.count);
        if (ns.kind == pz::NamesKind::Inline) {
            d["names"] = py::cast(pz::split_names(ns.blob, size_t(ns.count)));
        } else {
            d["names"] = py::none();
        }
        if (ns.kind == pz::NamesKind::PackSubset) {
            auto a = make_array<uint32_t>(ns.pack_index.size());
            if (!ns.pack_index.empty()) std::memcpy(a.mutable_data(), ns.pack_index.data(), ns.pack_index.size() * 4);
            d["pack_index"] = std::move(a);
        } else {
            d["pack_index"] = py::none();
        }
        return d;
    }

    py::dict kv() {
        pz::KvMap kv;
        for (int tries = 0; tries < 8; ++tries) {
            pz::ByteSource src = snapshot();
            try {
                py::gil_scoped_release nogil;
                kv = mx_.kv(src);
                break;
            } catch (const pz::Error& e) {
                if (e.code != pz::Errc::NeedRange || fetch_.is_none()) throw;
                ensure(e.need_off, e.need_len);
            }
        }
        py::dict d;
        for (auto& kvp : kv) d[py::str(kvp.first)] = py::str(kvp.second);
        return d;
    }

    // ---- decode
    py::dict decode(py::object c0_o, py::object c1_o, const std::string& mode_s, int threads, py::object dtype_o, py::object index_dtype_o,
                    py::object isa_o, py::object cols_o, py::object gene_map_o, int n_genes) {
        const pz::Header& h = mx_.header();
        bool gene = false;
        pz::Mode mode = parse_mode(mode_s, gene);
        const uint32_t c0 = c0_o.is_none() ? 0 : py::cast<uint32_t>(c0_o);
        const uint32_t c1 = c1_o.is_none() ? h.n : py::cast<uint32_t>(c1_o);
        if (c0 > c1 || c1 > h.n) throw py::value_error("column range out of bounds");
        const uint32_t ncol = c1 - c0;
        const bool whole = (c0 == 0 && c1 == h.n);

        // column selection (relative to c0), strictly ascending
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> cols;
        const uint32_t* cols_p = nullptr;
        size_t n_sel = 0;
        if (!cols_o.is_none()) {
            cols = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(cols_o);
            if (!cols) throw py::value_error("cols must be an integer array");
            cols_p = cols.data();
            n_sel = size_t(cols.size());
        }

        // gene map (canonical row -> gene)
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> gmap;
        std::vector<int32_t> stored_gmap;
        if (gene) {
            if (gene_map_o.is_none()) throw py::value_error("mode='gene' needs gene_map");
            gmap = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(gene_map_o);
            if (!gmap) throw py::value_error("gene_map must be an integer array");
            if (n_genes <= 0) throw py::value_error("mode='gene' needs n_genes > 0");
            const auto& perm = mx_.row_perm();
            stored_gmap = pc::compose_gene_map(gmap.data(), size_t(gmap.size()), perm.empty() ? nullptr : perm.data(), size_t(h.m));
        }

        // output types
        DtypeSpec ds = gene ? DtypeSpec{pz::DType::I32, "int32", 4} : parse_dtype(dtype_o, h.value_type);
        if (gene && !dtype_o.is_none()) {
            DtypeSpec want = parse_dtype(dtype_o, h.value_type);
            if (want.dt != pz::DType::I32 && want.dt != pz::DType::F64) throw py::value_error("mode='gene' supports dtype int32 or float64");
            ds = want;
        }
        const std::string idx_name = index_dtype_o.is_none() ? "int32" : lower(py::cast<std::string>(py::str(index_dtype_o)));
        if (idx_name != "int32" && idx_name != "uint32") throw py::value_error("index dtype must be int32 or uint32");

        // make sure the bytes are there (may call the fetch callable; GIL held)
        pz::ColumnPlan plan;
        if (whole) {
            const pz::SectionRef& ch = h.section[pz::kChunks];
            if (ch.len) ensure(ch.off, ch.len);
            plan.max_nnz = h.nnz;
        } else {
            plan = mx_.plan_columns(c0, c1);
            if (plan.byte_len) ensure(plan.byte_off, plan.byte_len);
        }
        const uint64_t bound = whole ? h.nnz : plan.max_nnz;
        const bool wide = bound >= (uint64_t(1) << 31);

        // allocate once
        py::array indptr = wide ? py::array(make_array<int64_t>(size_t(ncol) + 1)) : py::array(make_array<int32_t>(size_t(ncol) + 1));
        py::array indices = (idx_name == "uint32") ? py::array(make_array<uint32_t>(size_t(bound))) : py::array(make_array<int32_t>(size_t(bound)));
        py::array data;
        switch (ds.dt) {
            case pz::DType::U8: data = make_array<uint8_t>(size_t(bound)); break;
            case pz::DType::U16: data = make_array<uint16_t>(size_t(bound)); break;
            case pz::DType::U32: data = make_array<uint32_t>(size_t(bound)); break;
            case pz::DType::I32: data = make_array<int32_t>(size_t(bound)); break;
            case pz::DType::F32: data = make_array<float>(size_t(bound)); break;
            case pz::DType::F64: data = make_array<double>(size_t(bound)); break;
        }

        pz::Output out;
        out.indptr = indptr.mutable_data();
        out.indptr_type = wide ? pz::IntType::I64 : pz::IntType::I32;
        out.indptr_capacity = size_t(ncol) + 1;
        out.indices = indices.mutable_data();
        out.indices_type = (idx_name == "uint32") ? pz::IntType::U32 : pz::IntType::I32;
        out.indices_capacity = size_t(bound);
        out.data = data.mutable_data();
        out.data_type = ds.dt;
        out.data_capacity = size_t(bound);

        pz::DecodeOptions opt;
        opt.mode = mode;
        opt.threads = threads;
        opt.isa = parse_isa(isa_o);
        pz::ByteSource src = snapshot();

        uint64_t nnz_out = 0;
        py::array out_indptr = indptr;
        {
            py::gil_scoped_release nogil;
            if (whole) {
                mx_.decode(src, out, opt);
                nnz_out = h.nnz;
            } else {
                nnz_out = mx_.decode_columns(src, c0, c1, out, opt);
            }
            if (nnz_out > bound) throw pz::Error(pz::Errc::Corrupt, "decoder reported more non-zeros than the plan allows");
        }

        size_t ncol_out = ncol;
        const bool idx_unsigned = (idx_name == "uint32");
        void* ip_raw = indptr.mutable_data();
        void* idx_raw = indices.mutable_data();
        void* dat_raw = data.mutable_data();
        if (n_sel) {
            // keep only the selected columns, then continue on the compacted matrix
            py::array sel_indptr = wide ? py::array(make_array<int64_t>(n_sel + 1)) : py::array(make_array<int32_t>(n_sel + 1));
            void* sel_raw = sel_indptr.mutable_data();
            {
                py::gil_scoped_release nogil;
                if (wide) nnz_out = uint64_t(select_dispatch<int64_t>(ip_raw, ncol, cols_p, n_sel, idx_raw, idx_unsigned, dat_raw, ds.dt, sel_raw));
                else nnz_out = uint64_t(select_dispatch<int32_t>(ip_raw, ncol, cols_p, n_sel, idx_raw, idx_unsigned, dat_raw, ds.dt, sel_raw));
            }
            out_indptr = sel_indptr;
            ip_raw = sel_raw;
            ncol_out = n_sel;
        }
        if (gene) {
            py::gil_scoped_release nogil;
            if (wide) nnz_out = uint64_t(gene_dispatch<int64_t>(ip_raw, ncol_out, idx_raw, idx_unsigned, dat_raw, ds.dt, stored_gmap.data(), n_genes, threads));
            else nnz_out = uint64_t(gene_dispatch<int32_t>(ip_raw, ncol_out, idx_raw, idx_unsigned, dat_raw, ds.dt, stored_gmap.data(), n_genes, threads));
        }
        shrink(indices, size_t(nnz_out));
        shrink(data, size_t(nnz_out));

        py::dict r;
        r["indptr"] = out_indptr;
        r["indices"] = indices;
        r["data"] = data;
        r["n_rows"] = gene ? uint64_t(n_genes) : uint64_t(h.m);
        r["n_cols"] = uint64_t(ncol_out);
        r["nnz"] = nnz_out;
        r["mode"] = mode_s;
        if (!gene && mode == pz::Mode::Native && !mx_.row_perm().empty()) r["row_perm"] = row_perm();
        else r["row_perm"] = py::none();
        return r;
    }

private:
    struct Rng { uint64_t off; const uint8_t* p; size_t n; };
    py::object fetch_;
    size_t prefix_hint_;
    std::vector<std::unique_ptr<PyBuf>> bufs_;
    std::vector<Rng> rngs_;
    pz::Matrix mx_;

    pz::ByteSource snapshot() const {
        pz::ByteSource s;
        for (const Rng& r : rngs_) s.add(r.off, r.p, r.n);
        return s;
    }

    /// Make member bytes [off, off+len) available, calling the fetch callable when needed (GIL held).
    void ensure(uint64_t off, uint64_t len) {
        if (len == 0 || snapshot().has(off, len)) return;
        if (fetch_.is_none()) throw pz::Error(pz::Errc::NeedRange, "byte range not available", off, len);
        uint64_t want = len;
        if (off == 0 && want < prefix_hint_) want = prefix_hint_;
        py::object b = fetch_(off, want);
        auto pb = std::make_unique<PyBuf>(b);
        if (pb->size() < len) throw std::runtime_error("fetch returned " + std::to_string(pb->size()) + " bytes, needed " + std::to_string(len));
        Rng r{off, pb->data(), pb->size()};
        bufs_.push_back(std::move(pb));
        rngs_.push_back(r);
        keep_.append(b);
    }

    void open_loop() {
        for (int tries = 0; tries < 8; ++tries) {
            pz::ByteSource src = snapshot();
            try {
                mx_ = pz::Matrix::open(src);
                return;
            } catch (const pz::Error& e) {
                if (e.code != pz::Errc::NeedRange || fetch_.is_none()) throw;
                ensure(e.need_off, e.need_len);
            }
        }
        throw std::runtime_error("pz2: could not assemble the member prefix");
    }

    py::list keep_;  // objects returned by the fetch callable (their buffers are viewed by bufs_)

    template <class PtrT>
    static PtrT select_dispatch(void* indptr, size_t n_in, const uint32_t* cols, size_t n_sel, void* indices, bool idx_unsigned, void* data, pz::DType dt,
                                void* sel_indptr) {
        PtrT* ip = static_cast<PtrT*>(indptr);
        PtrT* op = static_cast<PtrT*>(sel_indptr);
        auto go = [&](auto* idx) -> PtrT {
            switch (dt) {
                case pz::DType::U8: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<uint8_t*>(data), op);
                case pz::DType::U16: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<uint16_t*>(data), op);
                case pz::DType::U32: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<uint32_t*>(data), op);
                case pz::DType::I32: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<int32_t*>(data), op);
                case pz::DType::F32: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<float*>(data), op);
                case pz::DType::F64: return pc::select_columns_inplace<PtrT>(ip, n_in, cols, n_sel, idx, static_cast<double*>(data), op);
            }
            return 0;
        };
        if (idx_unsigned) return go(static_cast<uint32_t*>(indices));
        return go(static_cast<int32_t*>(indices));
    }

    template <class PtrT>
    static PtrT gene_dispatch(void* indptr, size_t n, void* indices, bool idx_unsigned, void* data, pz::DType dt, const int32_t* map, int n_genes, int threads) {
        PtrT* ip = static_cast<PtrT*>(indptr);
        if (idx_unsigned) throw std::runtime_error("gene mode needs int32 indices");
        int32_t* idx = static_cast<int32_t*>(indices);
        if (dt == pz::DType::I32) return pc::aggregate_genes<PtrT, int32_t>(ip, n, idx, static_cast<int32_t*>(data), map, n_genes, threads);
        if (dt == pz::DType::F64) return pc::aggregate_genes<PtrT, double>(ip, n, idx, static_cast<double*>(data), map, n_genes, threads);
        throw std::runtime_error("gene mode needs int32 or float64 values");
    }
};

// ============================================================================
// functional wrappers
// ============================================================================
static py::dict py_read_header(py::object buf) {
    PyBuf b(buf);
    pz::Header h = pz::read_header(b.data(), b.size());
    return header_dict(h);
}

/// Source may be bytes-like (complete member or its prefix) or a fetch callable.
static std::unique_ptr<PyMember> open_member(py::object source, py::object fetch, size_t prefix_hint) {
    if (PyCallable_Check(source.ptr()) && !PyObject_CheckBuffer(source.ptr())) {
        return std::make_unique<PyMember>(py::none(), source, prefix_hint);
    }
    return std::make_unique<PyMember>(source, fetch, prefix_hint);
}

static py::dict py_decode(py::object source, const std::string& mode, int threads, py::object dtype, py::object index_dtype, py::object isa, py::object cols,
                          py::object gene_map, int n_genes, py::object fetch) {
    auto m = open_member(source, fetch, 65536);
    return m->decode(py::none(), py::none(), mode, threads, dtype, index_dtype, isa, cols, gene_map, n_genes);
}

static py::dict py_decode_columns(py::object source, uint32_t c0, uint32_t c1, const std::string& mode, int threads, py::object dtype, py::object index_dtype,
                                  py::object isa, py::object cols, py::object gene_map, int n_genes, py::object fetch) {
    auto m = open_member(source, fetch, 65536);
    return m->decode(py::int_(c0), py::int_(c1), mode, threads, dtype, index_dtype, isa, cols, gene_map, n_genes);
}

// ---- encode ---------------------------------------------------------------
static pz::NamesSpec parse_names(const py::object& o, uint64_t count) {
    pz::NamesSpec ns;
    ns.count = count;
    if (o.is_none()) return ns;  // inline, no blob (digest then needs compute_digest=False or an explicit digest)
    // accepted: list[str] (inline names), or dict {kind, pack_id (bytes8|hex), pack_index, names}
    if (py::isinstance<py::dict>(o)) {
        py::dict d = py::reinterpret_borrow<py::dict>(o);
        std::string kind = d.contains("kind") ? lower(py::cast<std::string>(d["kind"])) : "inline";
        if (kind == "inline") ns.kind = pz::NamesKind::Inline;
        else if (kind == "pack") ns.kind = pz::NamesKind::Pack;
        else if (kind == "pack_subset") ns.kind = pz::NamesKind::PackSubset;
        else if (kind == "cells") ns.kind = pz::NamesKind::Cells;
        else if (kind == "none") ns.kind = pz::NamesKind::None;
        else throw py::value_error("names kind must be inline, pack, pack_subset, cells or none");
        if (d.contains("pack_id") && !d["pack_id"].is_none()) {
            py::object pid = d["pack_id"];
            std::string raw;
            if (py::isinstance<py::bytes>(pid)) raw = py::cast<std::string>(pid);
            else {
                std::string hx = py::cast<std::string>(pid);
                if (hx.size() < 16) throw py::value_error("pack_id hex needs at least 16 characters");
                for (int i = 0; i < 8; ++i) raw.push_back(char(std::stoi(hx.substr(2 * i, 2), nullptr, 16)));
            }
            if (raw.size() < 8) throw py::value_error("pack_id needs 8 bytes");
            std::memcpy(ns.pack_id.data(), raw.data(), 8);
        }
        if (d.contains("pack_index") && !d["pack_index"].is_none()) {
            auto a = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(d["pack_index"]);
            if (!a) throw py::value_error("pack_index must be an integer array");
            ns.pack_index.assign(a.data(), a.data() + a.size());
        }
        if (d.contains("names") && !d["names"].is_none()) ns.blob = pz::join_names(py::cast<std::vector<std::string>>(d["names"]));
    } else {
        ns.kind = pz::NamesKind::Inline;
        ns.blob = pz::join_names(py::cast<std::vector<std::string>>(o));
    }
    return ns;
}

static pz::ValueType parse_value_type(const std::string& s0) {
    std::string s = lower(s0);
    if (s == "auto" || s.empty()) return pz::ValueType::Auto;
    if (s == "u8" || s == "uint8") return pz::ValueType::U8;
    if (s == "u16" || s == "uint16") return pz::ValueType::U16;
    if (s == "u32" || s == "uint32") return pz::ValueType::U32;
    if (s == "f32" || s == "float32") return pz::ValueType::F32;
    throw py::value_error("value_type must be auto, u8, u16, u32 or f32");
}

struct CscArgs {
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> indices;
    py::array data;
    std::vector<uint32_t> counts;
    pz::CSC csc;
};

static void fill_csc(CscArgs& A, py::object indptr_o, py::object indices_o, py::object data_o, uint32_t m, uint32_t n) {
    auto ip = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(indptr_o);
    if (!ip || ip.ndim() != 1 || uint64_t(ip.size()) != uint64_t(n) + 1) throw py::value_error("indptr must be a 1-D integer array of length n+1");
    A.indices = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(indices_o);
    if (!A.indices) throw py::value_error("indices must be an integer array");
    const int64_t* p = ip.data();
    if (p[0] != 0) throw py::value_error("indptr[0] must be 0");
    A.counts.resize(n);
    for (uint32_t j = 0; j < n; ++j) {
        if (p[j + 1] < p[j]) throw py::value_error("indptr must be non-decreasing");
        A.counts[j] = uint32_t(p[j + 1] - p[j]);
    }
    const uint64_t nnz = uint64_t(p[n]);
    if (uint64_t(A.indices.size()) != nnz) throw py::value_error("indices length != indptr[n]");
    py::array d = py::array::ensure(data_o);
    if (!d || uint64_t(d.size()) != nnz) throw py::value_error("data length != indptr[n]");
    py::dtype dt = d.dtype();
    pz::DType dtype;
    if (dt.is(py::dtype::of<uint8_t>())) dtype = pz::DType::U8;
    else if (dt.is(py::dtype::of<uint16_t>())) dtype = pz::DType::U16;
    else if (dt.is(py::dtype::of<uint32_t>())) dtype = pz::DType::U32;
    else if (dt.is(py::dtype::of<float>())) dtype = pz::DType::F32;
    else {
        // widen any other integer dtype to uint32 (negative values are an error)
        auto w = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(d);
        if (!w) throw py::value_error("data must be an integer or float32 array");
        auto u = py::array_t<uint32_t>(py::ssize_t(nnz));
        const int64_t* s = w.data();
        uint32_t* o = u.mutable_data();
        for (uint64_t k = 0; k < nnz; ++k) {
            if (s[k] < 0 || s[k] > int64_t(0xFFFFFFFFll)) throw py::value_error("data values must fit uint32");
            o[k] = uint32_t(s[k]);
        }
        d = u;
        dtype = pz::DType::U32;
    }
    A.data = py::array::ensure(d, py::array::c_style);
    A.csc.m = m;
    A.csc.n = n;
    A.csc.nnz = nnz;
    A.csc.indptr = nullptr;
    A.csc.counts = A.counts.data();
    A.csc.indices = A.indices.data();
    A.csc.data = A.data.data();
    A.csc.data_type = dtype;
}

static py::tuple py_encode(py::object indptr, py::object indices, py::object data, uint32_t m, uint32_t n, const std::string& codec, uint32_t chunk_cols,
                           int64_t chunk_nnz_max, int classes_per_octave, int lanes, int prob_bits, int value_prob_bits, int ctx_kind, const std::string& value_type,
                           py::object row_names, py::object col_names, py::object kv, py::object digest, bool compute_digest, int threads, int zstd_level) {
    CscArgs A;
    fill_csc(A, indptr, indices, data, m, n);
    pz::EncodeOptions opt;
    std::string c = lower(codec);
    if (c == "auto") opt.codec = pz::CodecId::Auto;
    else if (c == "rans" || c == "1") opt.codec = pz::CodecId::Rans;
    else if (c == "generic" || c == "2") opt.codec = pz::CodecId::Generic;
    else throw py::value_error("codec must be auto, rans or generic");
    opt.chunk_cols = chunk_cols;
    // 0 = library default, -1 = no budget (fixed-width chunks)
    opt.chunk_nnz_max = chunk_nnz_max < 0 ? UINT32_MAX : uint32_t(std::min<int64_t>(chunk_nnz_max, int64_t(UINT32_MAX) - 1));
    opt.classes_per_octave = uint8_t(classes_per_octave);
    opt.lanes = uint8_t(lanes);
    opt.prob_bits = uint8_t(prob_bits);
    opt.value_prob_bits = uint8_t(value_prob_bits);
    opt.ctx_kind = ctx_kind;
    opt.value_type = parse_value_type(value_type);
    opt.row_names = parse_names(row_names, m);
    opt.col_names = parse_names(col_names, n);
    if (!kv.is_none()) {
        for (auto item : py::reinterpret_borrow<py::dict>(kv))
            opt.kv[py::cast<std::string>(py::str(item.first))] = py::cast<std::string>(py::str(item.second));
    }
    if (!digest.is_none()) {
        std::string raw = py::cast<std::string>(digest);
        if (raw.size() != 16) throw py::value_error("digest must be 16 bytes");
        std::memcpy(opt.digest.b.data(), raw.data(), 16);
        opt.has_digest = true;
    }
    opt.compute_digest = compute_digest;
    opt.threads = threads;
    opt.zstd_level = zstd_level;
    std::vector<uint8_t> blob;
    pz::EncodeStats st;
    {
        py::gil_scoped_release nogil;
        blob = pz::encode_matrix(A.csc, opt, &st);
    }
    py::dict stats;
    stats["codec"] = int(st.codec);
    stats["value_type"] = value_type_name(st.value_type);
    stats["chunk_cols"] = st.chunk_cols;
    stats["n_chunks"] = st.n_chunks;
    stats["total"] = st.total;
    stats["header"] = st.header;
    stats["chunk_index"] = st.chunk_index;
    stats["tables"] = st.tables;
    stats["rows"] = st.rows;
    stats["chunks"] = st.chunks;
    stats["cols"] = st.cols;
    stats["kv"] = st.kv;
    stats["extra"] = st.extra;
    stats["codec1_bytes"] = st.codec1_bytes;
    stats["codec2_bytes"] = st.codec2_bytes;
    return py::make_tuple(py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size()), stats);
}

static py::bytes py_digest(py::object indptr, py::object indices, py::object data, uint32_t m, uint32_t n, const std::string& value_type, py::object row_names,
                           py::object col_names, py::object kv) {
    CscArgs A;
    fill_csc(A, indptr, indices, data, m, n);
    pz::ValueType vt = parse_value_type(value_type);
    if (vt == pz::ValueType::Auto) {
        vt = A.csc.data_type == pz::DType::U8 ? pz::ValueType::U8 : A.csc.data_type == pz::DType::U16 ? pz::ValueType::U16
           : A.csc.data_type == pz::DType::F32 ? pz::ValueType::F32 : pz::ValueType::U32;
    }
    std::vector<uint8_t> rb = parse_names(row_names, m).blob, cb = parse_names(col_names, n).blob;
    pz::KvMap kvm;
    if (!kv.is_none())
        for (auto item : py::reinterpret_borrow<py::dict>(kv))
            kvm[py::cast<std::string>(py::str(item.first))] = py::cast<std::string>(py::str(item.second));
    pz::Digest16 d;
    {
        py::gil_scoped_release nogil;
        d = pz::canonical_digest(A.csc, vt, rb, cb, kvm);
    }
    return py::bytes(reinterpret_cast<const char*>(d.b.data()), 16);
}

// ---- helpers over already decoded CSC data (also what the tests use to check the aggregation without a codec) ----------
/// aggregate_genes(indptr, indices, data, row_to_gene, n_genes, threads=1) -> (indptr, indices, data): rows summed into genes.
static py::tuple py_aggregate_genes(py::object indptr_o, py::object indices_o, py::object data_o, py::object map_o, int n_genes, int threads) {
    auto ip64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(indptr_o);
    auto idx = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(indices_o);
    auto mp = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(map_o);
    if (!ip64 || !idx || !mp) throw py::value_error("indptr, indices and row_to_gene must be integer arrays");
    const size_t n = size_t(ip64.size()) - 1;
    const size_t nnz = size_t(ip64.data()[n]);
    if (size_t(idx.size()) != nnz) throw py::value_error("indices length != indptr[-1]");
    for (size_t k = 0; k < nnz; ++k)
        if (uint32_t(idx.data()[k]) >= uint32_t(mp.size())) throw py::value_error("row id outside row_to_gene");
    py::array d = py::array::ensure(data_o);
    if (!d || size_t(d.size()) != nnz) throw py::value_error("data length != indptr[-1]");
    const bool as_double = d.dtype().kind() == 'f';
    const bool wide = nnz >= (size_t(1) << 31);
    py::array out_ip = wide ? py::array(make_array<int64_t>(n + 1)) : py::array(make_array<int32_t>(n + 1));
    for (size_t j = 0; j <= n; ++j) {
        if (wide) static_cast<int64_t*>(out_ip.mutable_data())[j] = ip64.data()[j];
        else static_cast<int32_t*>(out_ip.mutable_data())[j] = int32_t(ip64.data()[j]);
    }
    auto out_idx = make_array<int32_t>(nnz);
    if (nnz) std::memcpy(out_idx.mutable_data(), idx.data(), nnz * sizeof(int32_t));
    py::array out_d;
    uint64_t nout = 0;
    if (as_double) {
        auto dd = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(d);
        auto o = make_array<double>(nnz);
        if (nnz) std::memcpy(o.mutable_data(), dd.data(), nnz * sizeof(double));
        void* ipr = out_ip.mutable_data();
        {
            py::gil_scoped_release nogil;
            nout = wide ? uint64_t(pc::aggregate_genes<int64_t, double>(static_cast<int64_t*>(ipr), n, out_idx.mutable_data(), o.mutable_data(), mp.data(), n_genes, threads))
                        : uint64_t(pc::aggregate_genes<int32_t, double>(static_cast<int32_t*>(ipr), n, out_idx.mutable_data(), o.mutable_data(), mp.data(), n_genes, threads));
        }
        out_d = o;
    } else {
        auto dd = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(d);
        auto o = make_array<int32_t>(nnz);
        if (nnz) std::memcpy(o.mutable_data(), dd.data(), nnz * sizeof(int32_t));
        void* ipr = out_ip.mutable_data();
        {
            py::gil_scoped_release nogil;
            nout = wide ? uint64_t(pc::aggregate_genes<int64_t, int32_t>(static_cast<int64_t*>(ipr), n, out_idx.mutable_data(), o.mutable_data(), mp.data(), n_genes, threads))
                        : uint64_t(pc::aggregate_genes<int32_t, int32_t>(static_cast<int32_t*>(ipr), n, out_idx.mutable_data(), o.mutable_data(), mp.data(), n_genes, threads));
        }
        out_d = o;
    }
    py::array oi = out_idx;
    shrink(oi, size_t(nout));
    shrink(out_d, size_t(nout));
    return py::make_tuple(out_ip, oi, out_d);
}

/// select_columns(indptr, indices, data, cols) -> (indptr, indices, data) restricted to the (strictly ascending) columns.
static py::tuple py_select_columns(py::object indptr_o, py::object indices_o, py::object data_o, py::object cols_o) {
    auto ip64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(indptr_o);
    auto idx = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(indices_o);
    auto dd = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(data_o);
    auto cols = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(cols_o);
    if (!ip64 || !idx || !dd || !cols) throw py::value_error("indptr, indices, data and cols must be integer arrays");
    const size_t n = size_t(ip64.size()) - 1, nnz = size_t(ip64.data()[n]);
    auto oi = make_array<int32_t>(nnz), od = make_array<int32_t>(nnz);
    if (nnz) { std::memcpy(oi.mutable_data(), idx.data(), nnz * 4); std::memcpy(od.mutable_data(), dd.data(), nnz * 4); }
    auto oip = make_array<int64_t>(size_t(cols.size()) + 1);
    int64_t w = pc::select_columns_inplace<int64_t>(ip64.data(), n, cols.data(), size_t(cols.size()), oi.mutable_data(), od.mutable_data(), oip.mutable_data());
    py::array a = oi, b = od;
    shrink(a, size_t(w));
    shrink(b, size_t(w));
    return py::make_tuple(oip, a, b);
}

// ============================================================================
// module
// ============================================================================
PYBIND11_MODULE(_pz2, mod) {
    mod.doc() = "singlet._pz2: bindings for the store matrix codec (pz2). See src/bindings/python/_pz2.cpp for the API.";

    g_pz2_error = PyErr_NewException("singlet._pz2.Pz2Error", PyExc_RuntimeError, nullptr);
    g_pz2_needrange = PyErr_NewException("singlet._pz2.NeedRange", g_pz2_error, nullptr);
    Py_INCREF(g_pz2_error);
    Py_INCREF(g_pz2_needrange);
    mod.attr("Pz2Error") = py::reinterpret_borrow<py::object>(g_pz2_error);
    mod.attr("NeedRange") = py::reinterpret_borrow<py::object>(g_pz2_needrange);
    py::register_exception_translator([](std::exception_ptr p) {
        if (!p) return;
        try {
            std::rethrow_exception(p);
        } catch (const pz::Error& e) {
            translate_pz_error(e);
        }
    });

    mod.attr("HEADER_BYTES") = py::int_(pz::kHeaderBytes);
    mod.attr("FORMAT_VERSION") = py::int_(int(pz::kFormatVersion));
    mod.def("isa_report", []() { return pz::isa_report(); }, "One line: selected, best and compiled instruction-set paths and the PZ2_ISA value.");
    mod.def("selected_isa", []() { return std::string(pz::isa_name(pz::selected_isa())); });
    mod.def("best_isa", []() { return std::string(pz::isa_name(pz::best_isa())); });
    mod.def("isa_supported", [](const std::string& name) { return pz::isa_supported(parse_isa(py::str(name))); }, py::arg("name"));

    mod.def("read_header", &py_read_header, py::arg("buf"), "Parse and validate the 256-byte header of a pz2 member (bytes-like, at least 256 bytes).");

    py::class_<PyMember>(mod, "Member")
        .def(py::init([](py::object source, py::object fetch, size_t prefix_hint) {
                 if (PyCallable_Check(source.ptr()) && !PyObject_CheckBuffer(source.ptr())) return std::make_unique<PyMember>(py::none(), source, prefix_hint);
                 return std::make_unique<PyMember>(source, fetch, prefix_hint);
             }),
             py::arg("source") = py::none(), py::arg("fetch") = py::none(), py::arg("prefix_hint") = 65536,
             "Open a member from bytes-like `source` (a complete member or at least its prefix) and/or a `fetch(off, len)` callable returning member bytes.")
        .def("add_range", &PyMember::add_range, py::arg("off"), py::arg("buf"), "Register member bytes [off, off+len(buf)) (borrowed; the object is kept alive).")
        .def("has", &PyMember::has, py::arg("off"), py::arg("len"))
        .def_property_readonly("header", &PyMember::header)
        .def_property_readonly("m", &PyMember::m)
        .def_property_readonly("n", &PyMember::n)
        .def_property_readonly("nnz", &PyMember::nnz)
        .def_property_readonly("prefix_len", &PyMember::prefix_len)
        .def_property_readonly("value_dtype", &PyMember::value_dtype)
        .def_property_readonly("row_perm", &PyMember::row_perm, "stored row id -> canonical row id (uint32), or None for identity")
        .def("chunk_index", &PyMember::chunk_index, "(n_chunks+1, 4) uint64: off, len, first_col, nnz_before")
        .def("plan", &PyMember::plan, py::arg("c0"), py::arg("c1"))
        .def("column_counts", &PyMember::column_counts, py::arg("c0"), py::arg("c1"))
        .def("row_names", [](PyMember& s) { return s.names_dict(true); })
        .def("col_names", [](PyMember& s) { return s.names_dict(false); })
        .def("kv", &PyMember::kv)
        .def("decode", &PyMember::decode, py::arg("c0") = py::none(), py::arg("c1") = py::none(), py::kw_only(), py::arg("mode") = "native",
             py::arg("threads") = 1, py::arg("dtype") = py::none(), py::arg("index_dtype") = py::none(), py::arg("isa") = py::none(),
             py::arg("cols") = py::none(), py::arg("gene_map") = py::none(), py::arg("n_genes") = 0,
             "Decode columns [c0, c1) (default all) into freshly allocated numpy arrays; returns a dict with indptr, indices, data, n_rows, n_cols, nnz, row_perm.");

    mod.def("decode", &py_decode, py::arg("source"), py::kw_only(), py::arg("mode") = "native", py::arg("threads") = 1, py::arg("dtype") = py::none(),
            py::arg("index_dtype") = py::none(), py::arg("isa") = py::none(), py::arg("cols") = py::none(), py::arg("gene_map") = py::none(),
            py::arg("n_genes") = 0, py::arg("fetch") = py::none(),
            "Decode a whole member (bytes-like) or, with a fetch callable as `source`, the whole matrix fetched on demand.");
    mod.def("decode_columns", &py_decode_columns, py::arg("source"), py::arg("c0"), py::arg("c1"), py::kw_only(), py::arg("mode") = "native",
            py::arg("threads") = 1, py::arg("dtype") = py::none(), py::arg("index_dtype") = py::none(), py::arg("isa") = py::none(),
            py::arg("cols") = py::none(), py::arg("gene_map") = py::none(), py::arg("n_genes") = 0, py::arg("fetch") = py::none(),
            "Decode columns [c0, c1) of a member given as bytes-like (prefix plus the plan's range at its member offset, or whole) or as a fetch callable.");

    mod.def("aggregate_genes", &py_aggregate_genes, py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("row_to_gene"), py::arg("n_genes"),
            py::arg("threads") = 1, "Sum the rows of a CSC into genes through row_to_gene (-1 drops a row); returns new (indptr, indices, data) with gene ids ascending per column.");
    mod.def("select_columns", &py_select_columns, py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("cols"),
            "Keep the strictly ascending columns `cols` of a CSC; returns new (indptr, indices, data).");
    mod.def("encode", &py_encode, py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("m"), py::arg("n"), py::kw_only(), py::arg("codec") = "auto",
            py::arg("chunk_cols") = 0, py::arg("chunk_nnz_max") = 0, py::arg("classes_per_octave") = 4, py::arg("lanes") = 32, py::arg("prob_bits") = 10, py::arg("value_prob_bits") = 10,
            py::arg("ctx_kind") = -1, py::arg("value_type") = "auto", py::arg("row_names") = py::none(), py::arg("col_names") = py::none(),
            py::arg("kv") = py::none(), py::arg("digest") = py::none(), py::arg("compute_digest") = true, py::arg("threads") = 1, py::arg("zstd_level") = 19,
            "Encode a canonical CSC (row ids strictly ascending per column) into a pz2 member; returns (bytes, stats).\n"
            "chunk_cols: maximum columns per chunk (0 = default policy); chunk_nnz_max: a chunk ends before the column that would take it\n"
            "past this many non-zeros (0 = library default 49152, -1 = no budget).\n"
            "names: list[str] (inline) or {'kind': 'inline'|'pack'|'pack_subset'|'cells', 'pack_id': bytes8|hex, 'pack_index': array, 'names': list[str]}.");
    mod.def("digest", &py_digest, py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("m"), py::arg("n"), py::arg("value_type") = "auto",
            py::arg("row_names") = py::none(), py::arg("col_names") = py::none(), py::arg("kv") = py::none(),
            "Canonical digest (docs/format-store.md 4.6), 16 bytes.");
}
