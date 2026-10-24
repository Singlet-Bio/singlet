// SPDX-License-Identifier: MIT
// pz2 internals: CPU feature detection and ISA selection.  No -march flag is needed anywhere: SIMD kernels carry per-function
// target attributes and are only called after isa_supported() said yes.
#pragma once
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define PZ2_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace singlet {
namespace pz2 {
namespace detail {

struct CpuFeatures {
    bool avx2 = false, bmi2 = false, popcnt = false;
    bool avx512 = false;   // F, BW, DQ, CD, VL, VBMI2 (the set the AVX-512 kernel needs)
    bool sha = false;      // SHA extensions + SSSE3 + SSE4.1 (used by the SHA-256 of the canonical digest)
};

#ifdef PZ2_X86
inline void cpuid_ex(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
    int a[4];
    __cpuidex(a, int(leaf), int(sub));
    for (int i = 0; i < 4; ++i) r[i] = uint32_t(a[i]);
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
inline uint64_t xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (uint64_t(hi) << 32) | lo;
#endif
}
#endif

inline CpuFeatures detect_cpu() {
    CpuFeatures f;
#ifdef PZ2_X86
    uint32_t r[4] = {0, 0, 0, 0};
    cpuid_ex(0, 0, r);
    const uint32_t maxleaf = r[0];
    if (maxleaf < 7) return f;
    cpuid_ex(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    f.popcnt = (r[2] >> 23) & 1;
    const bool ssse3 = (r[2] >> 9) & 1, sse41 = (r[2] >> 19) & 1;
    if (!osxsave || !avx) return f;
    const uint64_t xcr0 = xgetbv0();
    const bool ymm = (xcr0 & 0x6) == 0x6;
    const bool zmm = (xcr0 & 0xE6) == 0xE6;   // XMM, YMM, opmask, ZMM_Hi256, Hi16_ZMM
    cpuid_ex(7, 0, r);
    const uint32_t b = r[1], c = r[2];
    f.bmi2 = (b >> 8) & 1;
    f.sha = ((b >> 29) & 1) && ssse3 && sse41;
    f.avx2 = ymm && ((b >> 5) & 1) && f.bmi2 && f.popcnt;
    const bool avx512f = (b >> 16) & 1, avx512dq = (b >> 17) & 1, avx512cd = (b >> 28) & 1, avx512bw = (b >> 30) & 1, avx512vl = (b >> 31) & 1;
    const bool vbmi2 = (c >> 6) & 1;
    f.avx512 = zmm && f.avx2 && avx512f && avx512dq && avx512cd && avx512bw && avx512vl && vbmi2;
#endif
    return f;
}
inline const CpuFeatures& cpu() { static const CpuFeatures f = detect_cpu(); return f; }

// Which SIMD paths this build contains (set by the kernel headers that are compiled in).
#if defined(PZ2_X86) && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)) && !defined(PZ2_SCALAR_ONLY)
#define PZ2_COMPILED_AVX2 1
#define PZ2_COMPILED_AVX512 1
#define PZ2_COMPILED_SHANI 1
#endif

}  // namespace detail

inline const char* isa_name(Isa i) { return i == Isa::Scalar ? "scalar" : i == Isa::Avx2 ? "avx2" : i == Isa::Avx512 ? "avx512" : "auto"; }
inline bool isa_compiled(Isa i) {
    if (i == Isa::Scalar) return true;
#ifdef PZ2_COMPILED_AVX2
    if (i == Isa::Avx2) return true;
#endif
#ifdef PZ2_COMPILED_AVX512
    if (i == Isa::Avx512) return true;
#endif
    return false;
}
inline bool isa_supported(Isa i) {
    if (i == Isa::Scalar) return true;
    if (!isa_compiled(i)) return false;
    if (i == Isa::Avx2) return detail::cpu().avx2;
    if (i == Isa::Avx512) return detail::cpu().avx512;
    return false;
}
inline Isa best_isa() {
    if (isa_supported(Isa::Avx512)) return Isa::Avx512;
    if (isa_supported(Isa::Avx2)) return Isa::Avx2;
    return Isa::Scalar;
}
inline Isa selected_isa() {
    static const Isa sel = [] {
        Isa best = best_isa();
        const char* e = std::getenv("PZ2_ISA");
        if (!e || !*e) return best;
        std::string s(e);
        Isa want = best;
        if (s == "scalar") want = Isa::Scalar;
        else if (s == "avx2") want = Isa::Avx2;
        else if (s == "avx512") want = Isa::Avx512;
        else return best;   // unknown value: ignore
        // never beyond what the CPU supports
        while (want != Isa::Scalar && !isa_supported(want)) want = (want == Isa::Avx512) ? Isa::Avx2 : Isa::Scalar;
        return want;
    }();
    return sel;
}
inline Isa resolve_isa(Isa r) {
    if (r == Isa::Auto) return selected_isa();
    while (r != Isa::Scalar && !isa_supported(r)) r = (r == Isa::Avx512) ? Isa::Avx2 : Isa::Scalar;
    return r;
}
inline std::string isa_report() {
    std::string s = "selected=" + std::string(isa_name(selected_isa())) + " best=" + isa_name(best_isa()) + " compiled=scalar";
    if (isa_compiled(Isa::Avx2)) s += ",avx2";
    if (isa_compiled(Isa::Avx512)) s += ",avx512";
    const char* e = std::getenv("PZ2_ISA");
    s += std::string(" PZ2_ISA=") + (e ? e : "(unset)");
    return s;
}

}  // namespace pz2
}  // namespace singlet
