// SPDX-License-Identifier: MIT
// tests/cpp/test_harness.h — shared hand-rolled test scaffolding
//
// These ~108 non-GPU C++ tests do NOT use a test framework; each is a
// standalone executable with its own `int main()`. Historically every file
// also hand-rolled an identical pass/fail counter pair plus a `CHECK` macro.
// This header consolidates ONLY that boilerplate — the counters, the `CHECK`
// macro, and the optional `CHECK_NEAR` float-compare helper. Each test still
// owns its own `main()` and its own summary print; nothing about test
// behavior or output changes.
//
// Three historically-distinct, byte-identical macro families existed. To keep
// every converted file behaving EXACTLY as before, select the family with a
// `#define` *before* including this header:
//
//   Family B (most common, ~20 files) — verbose, prints PASS+FAIL to stdout,
//   counters named n_pass / n_fail. This is the DEFAULT (no #define needed):
//       #include "test_harness.h"
//       // CHECK(cond, msg)            — prints "  PASS: msg" / "  FAIL: msg [line N]"
//       // CHECK_NEAR(a,b,eps,msg)
//
//   Family A (~9 files) — terse, 1-arg, prints only FAIL to stderr with the
//   stringized condition, counters named g_pass / g_fail:
//       #define SINGLET_TEST_HARNESS_TERSE
//       #include "test_harness.h"
//       // CHECK(cond)                 — prints "FAIL: cond at file:line" on failure
//
//   Family C (~4 files) — quiet 2-arg, prints only FAIL to stderr with msg,
//   counters named g_pass / g_fail:
//       #define SINGLET_TEST_HARNESS_QUIET
//       #include "test_harness.h"
//       // CHECK(cond, msg)            — prints "FAIL [line N]: msg" on failure
//
// Header-only, no dependencies beyond <iostream> / <cmath> (already included
// by every test that uses these macros). No framework. No CMake changes:
// this header lives in the same directory as the tests.

#ifndef SINGLET_TEST_HARNESS_H
#define SINGLET_TEST_HARNESS_H

#include <cmath>
#include <iostream>

#if defined(SINGLET_TEST_HARNESS_TERSE)

// ── Family A: terse 1-arg CHECK, g_pass / g_fail counters ──────────────────
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (cond) {                                                     \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::cerr << "FAIL: " << #cond                              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        }                                                               \
    } while (0)

#elif defined(SINGLET_TEST_HARNESS_QUIET)

// ── Family C: quiet 2-arg CHECK, g_pass / g_fail counters ──────────────────
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " << msg << "\n"; \
            ++g_fail;                                                  \
        } else {                                                       \
            ++g_pass;                                                  \
        }                                                              \
    } while (0)

#else

// ── Family B (default): verbose 2-arg CHECK, n_pass / n_fail counters ──────
static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (msg) << "\n";                            \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n";  \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

// Float-compare helper used by a handful of Family B tests. Identical to the
// previously-inlined local copies.
#define CHECK_NEAR(a, b, eps, name) \
    CHECK(std::abs((double)(a) - (double)(b)) <= (eps), name)

#endif

#endif  // SINGLET_TEST_HARNESS_H
