/**
 * @file _singlepress.cpp
 * @brief pybind11 bindings for SinglePress sparse matrix compression.
 *
 * Exposes read/write/info/compress/decompress plus PyTorch zero-copy tensor
 * creation directly from compressed data.
 *
 * Build (standalone):
 *   c++ -O2 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
 *       -I../include src/_singlepress.cpp \
 *       -o singlet/_singlepress$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <singlepress/singlepress.h>

#include <cstring>
#include <fstream>

namespace py = pybind11;
using namespace singlepress;

// ============================================================================
// Helper: compute file CRC32
// ============================================================================
static uint32_t py_file_crc32(const std::string& path) {
    auto data = read_file(path);
    if (data.size() <= FOOTER_SIZE)
        throw std::runtime_error("File too small for CRC32");
    return CRC32::compute(data.data(), data.size() - FOOTER_SIZE);
}

// ============================================================================
// sp_write: CSC arrays → .spz file (float64 values)
// ============================================================================
static py::dict sp_write_py(
    py::array_t<int32_t> indptr,
    py::array_t<int32_t> indices,
    py::array_t<double>  data_arr,
    int32_t nrows,
    const std::string& path,
    bool row_sort = false,
    const std::string& precision = "auto",
    py::object rownames_obj = py::none(),
    py::object colnames_obj = py::none(),
    int verbose = 0)
{
    auto p_buf = indptr.request();
    auto i_buf = indices.request();
    auto x_buf = data_arr.request();

    if (p_buf.ndim != 1 || i_buf.ndim != 1 || x_buf.ndim != 1)
        throw std::runtime_error("All inputs must be 1-D arrays");

    int32_t* p_ptr = static_cast<int32_t*>(p_buf.ptr);
    int32_t* i_ptr = static_cast<int32_t*>(i_buf.ptr);
    double*  x_ptr = static_cast<double*>(x_buf.ptr);

    uint32_t n = static_cast<uint32_t>(p_buf.shape[0] - 1);
    uint64_t nnz = static_cast<uint64_t>(i_buf.shape[0]);
    uint32_t m = static_cast<uint32_t>(nrows);

    // Build CSCMatrix
    CSCMatrix mat(m, n, nnz);
    for (uint32_t j = 0; j <= n; ++j)
        mat.p[j] = static_cast<uint32_t>(p_ptr[j]);
    for (uint64_t k = 0; k < nnz; ++k)
        mat.i[k] = static_cast<uint32_t>(i_ptr[k]);
    std::memcpy(mat.x.data(), x_ptr, nnz * sizeof(double));

    CompressConfig cfg;
    cfg.precision = precision;
    cfg.row_sort = row_sort;
    cfg.verbose = verbose;

    CompressStats stats;
    auto compressed = compress(mat, cfg, &stats);

    // Inject dimnames
    Metadata meta;
    auto hdr = FileHeader::deserialize(compressed.data());
    if (hdr.metadata_offset > 0 && hdr.metadata_offset < compressed.size() - FOOTER_SIZE) {
        size_t avail = compressed.size() - FOOTER_SIZE - hdr.metadata_offset;
        meta = Metadata::deserialize(compressed.data() + hdr.metadata_offset, avail);
    }

    bool has_names = false;
    if (!rownames_obj.is_none()) {
        meta.set_rownames(rownames_obj.cast<std::vector<std::string>>());
        has_names = true;
    }
    if (!colnames_obj.is_none()) {
        meta.set_colnames(colnames_obj.cast<std::vector<std::string>>());
        has_names = true;
    }
    if (has_names) inject_metadata(compressed, meta);

    write_file(path, compressed);

    py::dict result;
    result["raw_bytes"] = stats.raw_size;
    result["compressed_bytes"] = compressed.size();
    result["ratio"] = (compressed.size() > 0)
        ? static_cast<double>(stats.raw_size) / compressed.size() : 0.0;
    result["num_chunks"] = stats.num_chunks;
    result["rows"] = m;
    result["cols"] = n;
    result["nnz"] = nnz;
    return result;
}

// ============================================================================
// sp_write_int: CSC arrays with int32 values (avoids double conversion)
// ============================================================================
static py::dict sp_write_int_py(
    py::array_t<int32_t> indptr,
    py::array_t<int32_t> indices,
    py::array_t<int32_t> data_arr,
    int32_t nrows,
    const std::string& path,
    bool row_sort = false,
    py::object rownames_obj = py::none(),
    py::object colnames_obj = py::none(),
    int verbose = 0)
{
    auto p_buf = indptr.request();
    auto i_buf = indices.request();
    auto x_buf = data_arr.request();

    int32_t* p_ptr = static_cast<int32_t*>(p_buf.ptr);
    int32_t* i_ptr = static_cast<int32_t*>(i_buf.ptr);
    int32_t* x_ptr = static_cast<int32_t*>(x_buf.ptr);

    uint32_t n = static_cast<uint32_t>(p_buf.shape[0] - 1);
    uint64_t nnz = static_cast<uint64_t>(i_buf.shape[0]);
    uint32_t m = static_cast<uint32_t>(nrows);

    CSCMatrix mat(m, n, nnz);
    for (uint32_t j = 0; j <= n; ++j)
        mat.p[j] = static_cast<uint32_t>(p_ptr[j]);
    for (uint64_t k = 0; k < nnz; ++k)
        mat.i[k] = static_cast<uint32_t>(i_ptr[k]);
    for (uint64_t k = 0; k < nnz; ++k)
        mat.x[k] = static_cast<double>(x_ptr[k]);

    CompressConfig cfg;
    cfg.precision = "auto";
    cfg.row_sort = row_sort;
    cfg.verbose = verbose;

    CompressStats stats;
    auto compressed = compress(mat, cfg, &stats);

    Metadata meta;
    auto hdr = FileHeader::deserialize(compressed.data());
    if (hdr.metadata_offset > 0 && hdr.metadata_offset < compressed.size() - FOOTER_SIZE) {
        size_t avail = compressed.size() - FOOTER_SIZE - hdr.metadata_offset;
        meta = Metadata::deserialize(compressed.data() + hdr.metadata_offset, avail);
    }

    bool has_names = false;
    if (!rownames_obj.is_none()) {
        meta.set_rownames(rownames_obj.cast<std::vector<std::string>>());
        has_names = true;
    }
    if (!colnames_obj.is_none()) {
        meta.set_colnames(colnames_obj.cast<std::vector<std::string>>());
        has_names = true;
    }
    if (has_names) inject_metadata(compressed, meta);

    write_file(path, compressed);

    py::dict result;
    result["raw_bytes"] = stats.raw_size;
    result["compressed_bytes"] = compressed.size();
    result["ratio"] = (compressed.size() > 0)
        ? static_cast<double>(stats.raw_size) / compressed.size() : 0.0;
    result["num_chunks"] = stats.num_chunks;
    result["rows"] = m;
    result["cols"] = n;
    result["nnz"] = nnz;
    return result;
}

// ============================================================================
// sp_read: .spz → CSC components dict
// ============================================================================
static py::dict sp_read_py(
    const std::string& path, bool reorder = true, int verbose = 0)
{
    auto data = read_file(path);

    DecompressConfig cfg;
    cfg.reorder = reorder;
    cfg.verbose = verbose;

    Metadata meta;
    CSCMatrix mat = decompress(data.data(), data.size(), cfg, &meta);

    py::array_t<int32_t> indptr(mat.n + 1);
    py::array_t<int32_t> indices(mat.nnz);
    py::array_t<double>  values(mat.nnz);

    auto p_out = indptr.mutable_unchecked<1>();
    auto i_out = indices.mutable_unchecked<1>();
    auto x_out = values.mutable_unchecked<1>();

    for (uint32_t j = 0; j <= mat.n; ++j)
        p_out(j) = static_cast<int32_t>(mat.p[j]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        i_out(k) = static_cast<int32_t>(mat.i[k]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        x_out(k) = mat.x[k];

    py::dict result;
    result["indptr"] = indptr;
    result["indices"] = indices;
    result["data"] = values;
    result["shape"] = py::make_tuple(mat.m, mat.n);

    if (meta.has_rownames())
        result["rownames"] = meta.get_rownames();
    if (meta.has_colnames())
        result["colnames"] = meta.get_colnames();

    return result;
}

// ============================================================================
// sp_read_columns: partial column range read (streaming support)
// ============================================================================
static py::dict sp_read_columns_py(
    const std::string& path, uint32_t col_start, uint32_t col_end,
    bool reorder = true)
{
    auto data = read_file(path);

    DecompressConfig cfg;
    cfg.reorder = reorder;

    Metadata meta;
    CSCMatrix mat = decompress_columns(
        data.data(), data.size(), col_start, col_end, cfg, &meta);

    py::array_t<int32_t> indptr(mat.n + 1);
    py::array_t<int32_t> indices(mat.nnz);
    py::array_t<double>  values(mat.nnz);

    auto p_out = indptr.mutable_unchecked<1>();
    auto i_out = indices.mutable_unchecked<1>();
    auto x_out = values.mutable_unchecked<1>();

    for (uint32_t j = 0; j <= mat.n; ++j)
        p_out(j) = static_cast<int32_t>(mat.p[j]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        i_out(k) = static_cast<int32_t>(mat.i[k]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        x_out(k) = mat.x[k];

    py::dict result;
    result["indptr"] = indptr;
    result["indices"] = indices;
    result["data"] = values;
    result["shape"] = py::make_tuple(mat.m, mat.n);

    if (meta.has_rownames())
        result["rownames"] = meta.get_rownames();
    if (meta.has_colnames()) {
        auto& names = meta.get_colnames();
        uint32_t end = std::min(col_end, static_cast<uint32_t>(names.size()));
        std::vector<std::string> subset(names.begin() + col_start,
                                         names.begin() + end);
        result["colnames"] = subset;
    }

    return result;
}

// ============================================================================
// sp_info: header-only metadata
// ============================================================================
static py::dict sp_info_py(const std::string& path) {
    auto info = file_info(path);

    py::dict result;
    result["version"] = info.version;
    result["rows"] = info.rows;
    result["cols"] = info.cols;
    result["nnz"] = info.nnz;
    result["density_pct"] = info.density * 100.0;
    result["value_type"] = info.value_type;
    result["num_chunks"] = info.num_chunks;
    result["chunk_cols"] = info.chunk_cols;
    result["row_sorted"] = info.row_sorted;
    result["file_bytes"] = info.file_bytes;
    result["raw_bytes"] = info.raw_bytes;
    result["ratio"] = info.ratio;
    result["has_transpose"] = info.has_transpose;
    result["crc32"] = info.crc32;
    result["crc32_valid"] = info.crc32_valid;
    return result;
}

// ============================================================================
// In-memory compress/decompress
// ============================================================================
static py::bytes sp_compress_py(
    py::array_t<int32_t> indptr,
    py::array_t<int32_t> indices,
    py::array_t<double>  data_arr,
    int32_t nrows,
    bool row_sort = false,
    const std::string& precision = "auto")
{
    auto p_buf = indptr.request();
    auto i_buf = indices.request();
    auto x_buf = data_arr.request();

    int32_t* p_ptr = static_cast<int32_t*>(p_buf.ptr);
    int32_t* i_ptr = static_cast<int32_t*>(i_buf.ptr);
    double*  x_ptr = static_cast<double*>(x_buf.ptr);

    uint32_t n = static_cast<uint32_t>(p_buf.shape[0] - 1);
    uint64_t nnz = static_cast<uint64_t>(i_buf.shape[0]);
    uint32_t m = static_cast<uint32_t>(nrows);

    CSCMatrix mat(m, n, nnz);
    for (uint32_t j = 0; j <= n; ++j)
        mat.p[j] = static_cast<uint32_t>(p_ptr[j]);
    for (uint64_t k = 0; k < nnz; ++k)
        mat.i[k] = static_cast<uint32_t>(i_ptr[k]);
    std::memcpy(mat.x.data(), x_ptr, nnz * sizeof(double));

    CompressConfig cfg;
    cfg.precision = precision;
    cfg.row_sort = row_sort;

    auto compressed = compress(mat, cfg);
    return py::bytes(
        reinterpret_cast<const char*>(compressed.data()), compressed.size());
}

static py::dict sp_decompress_py(py::bytes blob) {
    std::string raw = blob;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(raw.data());
    size_t sz = raw.size();

    DecompressConfig cfg;
    cfg.reorder = true;

    Metadata meta;
    CSCMatrix mat = decompress(data, sz, cfg, &meta);

    py::array_t<int32_t> indptr(mat.n + 1);
    py::array_t<int32_t> indices(mat.nnz);
    py::array_t<double>  values(mat.nnz);

    auto p_out = indptr.mutable_unchecked<1>();
    auto i_out = indices.mutable_unchecked<1>();
    auto x_out = values.mutable_unchecked<1>();

    for (uint32_t j = 0; j <= mat.n; ++j)
        p_out(j) = static_cast<int32_t>(mat.p[j]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        i_out(k) = static_cast<int32_t>(mat.i[k]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        x_out(k) = mat.x[k];

    py::dict result;
    result["indptr"] = indptr;
    result["indices"] = indices;
    result["data"] = values;
    result["shape"] = py::make_tuple(mat.m, mat.n);

    if (meta.has_rownames())
        result["rownames"] = meta.get_rownames();
    if (meta.has_colnames())
        result["colnames"] = meta.get_colnames();

    return result;
}

// ============================================================================
// PyTorch sparse tensor creation (zero-copy where possible)
// ============================================================================
static py::object sp_to_torch_coo_py(
    const std::string& path, const std::string& dtype = "float32",
    const std::string& device = "cpu")
{
    // Import torch lazily
    py::module_ torch = py::module_::import("torch");

    auto data = read_file(path);
    DecompressConfig cfg;
    cfg.reorder = true;

    Metadata meta;
    CSCMatrix mat = decompress(data.data(), data.size(), cfg, &meta);

    // Convert CSC to COO: expand column pointers
    uint64_t nnz = mat.nnz;
    py::array_t<int64_t> row_idx(nnz);
    py::array_t<int64_t> col_idx(nnz);
    auto row_buf = row_idx.mutable_unchecked<1>();
    auto col_buf = col_idx.mutable_unchecked<1>();

    for (uint32_t j = 0; j < mat.n; ++j) {
        for (uint32_t k = mat.p[j]; k < mat.p[j + 1]; ++k) {
            row_buf(k) = static_cast<int64_t>(mat.i[k]);
            col_buf(k) = static_cast<int64_t>(j);
        }
    }

    // Stack into [2, nnz] index tensor
    py::list idx_list;
    idx_list.append(torch.attr("from_numpy")(row_idx));
    idx_list.append(torch.attr("from_numpy")(col_idx));
    py::object indices_tensor = torch.attr("stack")(idx_list);

    // Values tensor
    py::object values_tensor;
    if (dtype == "float64") {
        py::array_t<double> vals(nnz);
        std::memcpy(vals.mutable_data(), mat.x.data(), nnz * sizeof(double));
        values_tensor = torch.attr("from_numpy")(vals);
    } else {
        py::array_t<float> vals(nnz);
        auto v = vals.mutable_unchecked<1>();
        for (uint64_t k = 0; k < nnz; ++k) v(k) = static_cast<float>(mat.x[k]);
        values_tensor = torch.attr("from_numpy")(vals);
    }

    py::object size = py::make_tuple(mat.m, mat.n);
    py::object sparse = torch.attr("sparse_coo_tensor")(
        indices_tensor, values_tensor, size);

    if (device != "cpu")
        sparse = sparse.attr("to")(device);

    return sparse;
}

// CSR tensor for PyTorch (preferred for GPU — torch.sparse_csr_tensor)
static py::object sp_to_torch_csr_py(
    const std::string& path, const std::string& dtype = "float32",
    const std::string& device = "cpu")
{
    py::module_ torch = py::module_::import("torch");

    auto data = read_file(path);
    DecompressConfig cfg;
    cfg.reorder = true;
    Metadata meta;
    CSCMatrix mat = decompress(data.data(), data.size(), cfg, &meta);

    // Convert CSC → CSR by transposing: the .spz stores genes×cells,
    // but AnnData is cells×genes, so CSC(genes×cells) = CSR(cells×genes)
    // This means the mat.p are "row pointers" for cells×genes CSR

    // Actually, for cells×genes we need to transpose the matrix.
    // CSC(m=genes, n=cells) → CSR(cells, genes) = transpose CSC
    // CSR crow_indices = old col pointers, CSR col_indices = old row indices

    // crow_indices: n+1 elements (cells+1)
    py::array_t<int64_t> crow(mat.n + 1);
    py::array_t<int64_t> ccol(mat.nnz);
    auto crow_buf = crow.mutable_unchecked<1>();
    auto ccol_buf = ccol.mutable_unchecked<1>();

    for (uint32_t j = 0; j <= mat.n; ++j)
        crow_buf(j) = static_cast<int64_t>(mat.p[j]);
    for (uint64_t k = 0; k < mat.nnz; ++k)
        ccol_buf(k) = static_cast<int64_t>(mat.i[k]);

    py::object crow_t = torch.attr("from_numpy")(crow);
    py::object ccol_t = torch.attr("from_numpy")(ccol);

    py::object values_tensor;
    if (dtype == "float64") {
        py::array_t<double> vals(mat.nnz);
        std::memcpy(vals.mutable_data(), mat.x.data(), mat.nnz * sizeof(double));
        values_tensor = torch.attr("from_numpy")(vals);
    } else {
        py::array_t<float> vals(mat.nnz);
        auto v = vals.mutable_unchecked<1>();
        for (uint64_t k = 0; k < mat.nnz; ++k)
            v(k) = static_cast<float>(mat.x[k]);
        values_tensor = torch.attr("from_numpy")(vals);
    }

    // CSR shape is (cells, genes) = (n, m)
    py::object size = py::make_tuple(mat.n, mat.m);
    py::object sparse = torch.attr("sparse_csr_tensor")(
        crow_t, ccol_t, values_tensor, size);

    if (device != "cpu")
        sparse = sparse.attr("to")(device);

    return sparse;
}

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(_singlepress, m) {
    m.doc() = "SinglePress: High-compression sparse matrix format for single-cell data.\n"
              "~10x compression, zero-copy GPU streaming, PyTorch native sparse tensors.";

    m.def("sp_write", &sp_write_py,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("nrows"), py::arg("path"),
          py::arg("row_sort") = false,
          py::arg("precision") = "auto",
          py::arg("rownames") = py::none(),
          py::arg("colnames") = py::none(),
          py::arg("verbose") = 0,
          "Write a CSC sparse matrix to a .spz file.");

    m.def("sp_write_int", &sp_write_int_py,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("nrows"), py::arg("path"),
          py::arg("row_sort") = false,
          py::arg("rownames") = py::none(),
          py::arg("colnames") = py::none(),
          py::arg("verbose") = 0,
          "Write a CSC sparse matrix with int32 values to a .spz file.");

    m.def("sp_read", &sp_read_py,
          py::arg("path"), py::arg("reorder") = true, py::arg("verbose") = 0,
          "Read a .spz file and return CSC components as a dict.");

    m.def("sp_read_columns", &sp_read_columns_py,
          py::arg("path"), py::arg("col_start"), py::arg("col_end"),
          py::arg("reorder") = true,
          "Read a column range from a .spz file (streaming support).");

    m.def("sp_info", &sp_info_py,
          py::arg("path"),
          "Read .spz file header without decompression.");

    m.def("sp_compress", &sp_compress_py,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("nrows"), py::arg("row_sort") = false,
          py::arg("precision") = "auto",
          "Compress CSC arrays to a raw bytes blob (in-memory).");

    m.def("sp_decompress", &sp_decompress_py,
          py::arg("blob"),
          "Decompress a raw .spz bytes blob to CSC arrays.");

    m.def("sp_to_torch_coo", &sp_to_torch_coo_py,
          py::arg("path"), py::arg("dtype") = "float32",
          py::arg("device") = "cpu",
          "Read .spz → PyTorch sparse COO tensor (cells × genes).");

    m.def("sp_to_torch_csr", &sp_to_torch_csr_py,
          py::arg("path"), py::arg("dtype") = "float32",
          py::arg("device") = "cpu",
          "Read .spz → PyTorch sparse CSR tensor (cells × genes).\n"
          "CSR is preferred for GPU operations via cuSPARSE.");

    m.def("file_crc32", &py_file_crc32,
          py::arg("path"),
          "Compute CRC32 of a .spz file (excluding footer).");
}
