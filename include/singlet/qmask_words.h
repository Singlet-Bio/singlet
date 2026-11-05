#pragma once
// Width of the per-read low-quality mask that STAR's hand-off and the GPU pileup share (P3-ENG-14).
// Bit i of a read's mask: base i of its record (5' clip removed, original orientation) has quality < min_baseq.
// A read takes qmaskWords() 64-bit words. Two words (128 bases) is the layout of every run before this change, so
// reads up to 128 bases keep their files byte for byte. singlet sets SINGLET_HANDOFF_QWORDS from the cDNA read-length
// bound before STAR starts, and every reader takes the width once at setup.
#include <cstdint>
#include <cstdlib>

namespace singlet_qmask {
constexpr uint32_t kMinWords = 2;   // 128 bases
constexpr uint32_t kMaxWords = 8;   // 512 bases: longer cDNA reads keep the CPU pileup
inline uint32_t wordsFor(int readLenBound) {
    uint32_t w = readLenBound > 0 ? (uint32_t)((readLenBound + 63) / 64) : kMinWords;
    return w < kMinWords ? kMinWords : w;
}
// the width this process uses (SINGLET_HANDOFF_QWORDS, clamped to [kMinWords, kMaxWords]); read it once and keep it
inline uint32_t words() {
    const char* e = std::getenv("SINGLET_HANDOFF_QWORDS");
    unsigned long v = e ? std::strtoul(e, nullptr, 10) : kMinWords;
    if (v < kMinWords) v = kMinWords;
    if (v > kMaxWords) v = kMaxWords;
    return (uint32_t)v;
}
inline uint32_t maxBases() { return 64u * kMaxWords; }
}  // namespace singlet_qmask
