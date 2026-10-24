// SPDX-License-Identifier: MIT
// pz2 internals: value representation (what the codecs see) and conversion to/from the caller's element types.
//
// The codecs code unsigned 32-bit numbers ("coded values").  Three modes:
//   Int     value_type U8/U16/U32: the coded value is the number itself.
//   F32Int  value_type F32 and header flag bit0: every value is a non-negative integer < 2^24 whose float bit pattern round
//           trips exactly; the coded value is the integer.
//   F32Raw  value_type F32 otherwise: the coded value is the float's bit pattern (codec 2 only).
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {

enum class VMode { Int, F32Int, F32Raw };

inline VMode vmode_of(const Header& h) {
    if (h.value_type != ValueType::F32) return VMode::Int;
    return (h.flags & 1) ? VMode::F32Int : VMode::F32Raw;
}
inline uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

/// True when bit pattern `u` is a non-negative integer < 2^24 that survives integer -> float -> bits exactly.
inline bool f32_is_integral(uint32_t u, uint32_t& v) {
    float f = bits_f32(u);
    if (!(f >= 0.0f && f < 16777216.0f)) return false;
    uint32_t i = uint32_t(f);
    if (f32_bits(float(i)) != u) return false;   // also rejects -0.0
    v = i;
    return true;
}

inline size_t dtype_size(DType t) {
    switch (t) { case DType::U8: return 1; case DType::U16: return 2; case DType::U32: case DType::F32: case DType::I32: return 4; case DType::F64: return 8; }
    return 0;
}
inline size_t inttype_size(IntType t) { return (t == IntType::U64 || t == IntType::I64) ? 8 : 4; }

/// Convert n coded values to the caller's type.  Throws TypeMismatch if a value does not fit or the combination is not allowed.
inline void write_values(DType t, VMode vm, const uint32_t* s, size_t n, void* dst) {
    auto bad = [](const char* what) { throw Error(Errc::TypeMismatch, what); };
    if (vm == VMode::Int) {
        switch (t) {
            case DType::U32: if (dst != static_cast<const void*>(s)) std::memcpy(dst, s, n * 4); return;
            case DType::U8: { uint8_t* d = static_cast<uint8_t*>(dst); uint32_t any = 0; for (size_t i = 0; i < n; ++i) { any |= s[i]; d[i] = uint8_t(s[i]); } if (any > 255) bad("value does not fit u8"); return; }
            case DType::U16: { uint16_t* d = static_cast<uint16_t*>(dst); uint32_t any = 0; for (size_t i = 0; i < n; ++i) { any |= s[i]; d[i] = uint16_t(s[i]); } if (any > 65535) bad("value does not fit u16"); return; }
            case DType::I32: { int32_t* d = static_cast<int32_t*>(dst); uint32_t any = 0; for (size_t i = 0; i < n; ++i) { any |= s[i]; d[i] = int32_t(s[i]); } if (any >> 31) bad("value does not fit i32"); return; }
            case DType::F32: { float* d = static_cast<float*>(dst); for (size_t i = 0; i < n; ++i) d[i] = float(s[i]); return; }
            case DType::F64: { double* d = static_cast<double*>(dst); for (size_t i = 0; i < n; ++i) d[i] = double(s[i]); return; }
        }
        return;
    }
    const bool integral = vm == VMode::F32Int;
    switch (t) {
        case DType::F32: { float* d = static_cast<float*>(dst); for (size_t i = 0; i < n; ++i) d[i] = integral ? float(s[i]) : bits_f32(s[i]); return; }
        case DType::F64: { double* d = static_cast<double*>(dst); for (size_t i = 0; i < n; ++i) d[i] = double(integral ? float(s[i]) : bits_f32(s[i])); return; }
        case DType::U32: { uint32_t* d = static_cast<uint32_t*>(dst); for (size_t i = 0; i < n; ++i) d[i] = integral ? f32_bits(float(s[i])) : s[i]; return; }
        default: bad("an f32 matrix can only be decoded to f32, f64 or u32 bit patterns");
    }
}

/// The u32 "digest value" (integer value, or float bit pattern) of coded value v.
inline uint32_t digest_value(VMode vm, uint32_t v) { return vm == VMode::F32Int ? f32_bits(float(v)) : v; }

}  // namespace detail
}  // namespace pz2
}  // namespace singlet
