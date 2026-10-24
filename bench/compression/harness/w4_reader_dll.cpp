// w4_reader_dll: C ABI over kernels/w4_reader_dec.h for ctypes / R. Build: tools/w4_reader_build.sh
#include <cstring>
#include "w4_reader_dec.h"

#ifdef _WIN32
#define W4_EXPORT extern "C" __declspec(dllexport)
#else
#define W4_EXPORT extern "C"
#endif

static void seterr(char* err, int errlen, const char* msg) {
    if (err && errlen > 0) { std::strncpy(err, msg, errlen - 1); err[errlen - 1] = 0; }
}

W4_EXPORT void* w4r_open(const uint8_t* buf, uint64_t n, int verify_file, char* err, int errlen) {
    try { auto* H = new w4r::Handle(); try { w4r::open(*H, buf, (size_t)n, verify_file != 0); } catch (...) { delete H; throw; } return H; }
    catch (const std::exception& e) { seterr(err, errlen, e.what()); return nullptr; }
}
// out[0..5] = m, n, nnz, vt_code, num_chunks, chunk_cols
W4_EXPORT void w4r_shape(void* h, uint64_t* out) {
    auto* H = (w4r::Handle*)h;
    out[0] = H->hdr.m; out[1] = H->hdr.n; out[2] = H->hdr.nnz; out[3] = H->hdr.vt_code; out[4] = H->hdr.num_chunks; out[5] = H->hdr.chunk_cols;
}
W4_EXPORT void w4r_indptr(void* h, int32_t* out) {
    auto* H = (w4r::Handle*)h;
    for (size_t j = 0; j <= H->hdr.n; ++j) out[j] = (int32_t)H->indptr[j];
}
// value_kind: 1 = uint8, 2 = uint16, 4 = int32, 5 = float32, 8 = float64. Chunks [c0, c1); c1 < 0 means all.
W4_EXPORT int w4r_decode(void* h, int64_t c0, int64_t c1, int32_t* indices, void* values, int value_kind, int threads, int packed_sort, char* err, int errlen) {
    auto* H = (w4r::Handle*)h;
    try {
        size_t a = (size_t)c0, b = c1 < 0 ? H->hdr.num_chunks : (size_t)c1;
        switch (value_kind) {
            case 1: w4r::decode_range<uint8_t>(*H, a, b, indices, (uint8_t*)values, threads, packed_sort != 0); break;
            case 2: w4r::decode_range<uint16_t>(*H, a, b, indices, (uint16_t*)values, threads, packed_sort != 0); break;
            case 4: w4r::decode_range<int32_t>(*H, a, b, indices, (int32_t*)values, threads, packed_sort != 0); break;
            case 5: w4r::decode_range<float>(*H, a, b, indices, (float*)values, threads, packed_sort != 0); break;
            case 8: w4r::decode_range<double>(*H, a, b, indices, (double*)values, threads, packed_sort != 0); break;
            default: seterr(err, errlen, "bad value_kind"); return 2;
        }
        return 0;
    } catch (const std::exception& e) { seterr(err, errlen, e.what()); return 1; }
}
W4_EXPORT void w4r_close(void* h) { delete (w4r::Handle*)h; }
