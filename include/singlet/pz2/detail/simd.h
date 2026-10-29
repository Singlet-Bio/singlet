// SPDX-License-Identifier: MIT
// pz2 internals: per-function ISA targeting without -march flags.
//   PZ2_BEGIN_AVX2 / PZ2_BEGIN_AVX512 ... PZ2_END_TARGET  wrap code that may use those instruction sets.  Code inside must be
//   plain functions and templates DEFINED inside the region; never instantiate std:: templates there (a COMDAT instance compiled
//   with AVX-512 could be picked for generic callers).  Kernels are only ever called after isa_supported() said yes.
#pragma once

#if defined(PZ2_COMPILED_AVX2) || defined(PZ2_COMPILED_AVX512) || defined(PZ2_COMPILED_SHANI)
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define PZ2_AINLINE __forceinline
#define PZ2_BEGIN_AVX2
#define PZ2_BEGIN_AVX512
#define PZ2_END_TARGET
#elif defined(__clang__)
#define PZ2_AINLINE inline __attribute__((always_inline))
#define PZ2_BEGIN_AVX2 _Pragma("clang attribute push (__attribute__((target(\"avx2,bmi,bmi2,popcnt,lzcnt\"))), apply_to=function)")
#define PZ2_BEGIN_AVX512 _Pragma("clang attribute push (__attribute__((target(\"avx2,bmi,bmi2,popcnt,lzcnt,avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx512vbmi2\"))), apply_to=function)")
#define PZ2_END_TARGET _Pragma("clang attribute pop")
#else
#define PZ2_AINLINE inline __attribute__((always_inline))
#define PZ2_BEGIN_AVX2 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,bmi,bmi2,popcnt,lzcnt\")")
#define PZ2_BEGIN_AVX512 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,bmi,bmi2,popcnt,lzcnt,avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx512vbmi2\")")
#define PZ2_END_TARGET _Pragma("GCC pop_options")
#endif
