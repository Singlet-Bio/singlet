// SPDX-License-Identifier: MIT
// include/singlet/fq/parse_bc1.h
//
// Parse Biosciences Evercode WT v2 round-1 barcodes: each of the 96 round-1 wells primes with an oligo-dT and a random-hexamer
// RT primer that carry different 8 nt barcodes, so one cell's reads arrive under two round-1 barcodes. Rewriting the random-hexamer
// barcode to the oligo-dT barcode of the same well puts both halves in one cell, as Parse's own pipeline reports them. WT v2 Mini
// and the 48-well v2 kit use the first 12 and 48 wells of this table.
//
// Table (random -> dT, wells A1-H12) from pachterlab/SBP_2025, mouse/wells/barcodes/r1_R.txt, r1_T.txt and r1_RT_replace.txt,
// distributed under the BSD 2-Clause License, reproduced here:
//
//   BSD 2-Clause License
//
//   Copyright (c) 2025, Pachter Lab
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//   1. Redistributions of source code must retain the above copyright notice, this
//      list of conditions and the following disclaimer.
//
//   2. Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// The 192 sequences equal the engine's splitseq_bc_rd1.txt list.
#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

namespace singlet::fq {

// Packs an 8 nt byte-numeric barcode (0-3 = A,C,G,T) into 16 bits; returns -1 when a base is N or out of range.
inline int32_t pack_bc8(const uint8_t* s) {
    uint32_t v = 0;
    for (int i = 0; i < 8; ++i) { if (s[i] > 3) return -1; v = (v << 2) | s[i]; }
    return static_cast<int32_t>(v);
}

inline const std::unordered_map<uint32_t, uint32_t>& parse_v2_bc1_random_to_dt() {
    static const std::unordered_map<uint32_t, uint32_t> m = [] {
        static const char* kPairs[96][2] = {
            {"CATCATCC", "CATTCCTA"},  // A1
            {"CTGCTTTG", "CTTCATCA"},  // A2
            {"CTAAGGGA", "CCTATATC"},  // A3
            {"GCTTATAG", "ACATTTAC"},  // A4
            {"TCTGATCC", "ACTTAGCT"},  // A5
            {"TCTCTTGG", "CCAATTCT"},  // A6
            {"CAATTTCC", "GCCTATCT"},  // A7
            {"AGTCTCTT", "ATGCTGCT"},  // A8
            {"TGCTGCTC", "CATTTACA"},  // A9
            {"GTATTTCC", "ACTCGTAA"},  // A10
            {"TTCCTGTG", "CCTTTGCA"},  // A11
            {"GCTGCTTC", "ACTCCTGC"},  // A12
            {"TATGTGTC", "ATTTGGCA"},  // B1
            {"CAATTCTC", "TTATTCTG"},  // B2
            {"TGGTCTCC", "TCATGCTC"},  // B3
            {"GCTCTTTA", "CATACTTC"},  // B4
            {"GCTGCATG", "CCGTTCTA"},  // B5
            {"ACTCATTT", "GCTTCATA"},  // B6
            {"AGTCTTGG", "CTCTGTGC"},  // B7
            {"GGTTCTTC", "CCCTTATA"},  // B8
            {"TCATGTTG", "ACTGCTCT"},  // B9
            {"ATTTTGCC", "CTCTAATC"},  // B10
            {"CTTCTGTA", "ACCCTTGC"},  // B11
            {"GTCCATCT", "ATCTTAGG"},  // B12
            {"GCTATCTC", "CATGTCTC"},  // C1
            {"TAGTTTCC", "TCATTGCA"},  // C2
            {"TCCATTAT", "ACACCTTT"},  // C3
            {"AGGATTAA", "AATTTCTC"},  // C4
            {"AATCTTTC", "ATTCATGG"},  // C5
            {"GTCATATG", "ACTTTACC"},  // C6
            {"GTGCTTCC", "CTTCTAAC"},  // C7
            {"ATGTGTTG", "CTATTTCA"},  // C8
            {"CCATCTTG", "TCTCATGC"},  // C9
            {"TACTGTCT", "ATCCTTAC"},  // C10
            {"TTCATCGC", "TAAATATC"},  // C11
            {"ACTGTGGG", "TTACCTGC"},  // C12
            {"TCTGTGCC", "CACTTTCA"},  // D1
            {"TCAATCTC", "CACCTTTA"},  // D2
            {"GTCCTCTG", "CTGACTTC"},  // D3
            {"TTACATTC", "CATTTGGA"},  // D4
            {"ATTCTGTC", "GCTCTACT"},  // D5
            {"TGTGTATG", "GTTACGTA"},  // D6
            {"TCCATTTG", "CCTGTTGC"},  // D7
            {"TTAGCTTC", "CTATCATC"},  // D8
            {"GTGCTTGA", "GCTATCAT"},  // D9
            {"GTTTGTGA", "ACATTCAT"},  // D10
            {"GAAATTAG", "TTCGCTAC"},  // D11
            {"GCAAATTC", "CATTCTAC"},  // D12
            {"GAGGTTGA", "CACTTATC"},  // E1
            {"CCTGTCTG", "ATAAGCTC"},  // E2
            {"GTGGGTTC", "TCATCCTG"},  // E3
            {"TTTGCATC", "CCTGGTAT"},  // E4
            {"AGGTAATA", "TGGTATAC"},  // E5
            {"GTGCCTTC", "TTGGGAGA"},  // E6
            {"ATGTTTCC", "ACTTCATC"},  // E7
            {"CTTAATTC", "TCTCTAGC"},  // E8
            {"TCTGGCTC", "ATGCCCTT"},  // E9
            {"CATCATTT", "CCCAATTT"},  // E10
            {"GTTGTCTC", "ACTATATA"},  // E11
            {"ATCTTCTG", "CTCTATAC"},  // E12
            {"TGTTTGCC", "CTGTCTCA"},  // F1
            {"TTCTGTCA", "GACCTTTC"},  // F2
            {"ACGGACTC", "GATTTGGC"},  // F3
            {"TTTGGTCA", "CGTCTAGG"},  // F4
            {"TATCCGGG", "TACTCGAA"},  // F5
            {"TGTCATTC", "CAGCCTTT"},  // F6
            {"ATTCTCTG", "CCTCATTA"},  // F7
            {"TGGCTTCC", "CTTATACC"},  // F8
            {"TTGTTGCC", "TCTATTAC"},  // F9
            {"GTCATCTC", "CCTGCATT"},  // F10
            {"TTGCTCAT", "CAATCCTT"},  // F11
            {"CTGTCTGC", "TTGTCTTA"},  // F12
            {"TATATTCC", "TCACTTTA"},  // G1
            {"ATATTGGC", "TGCTTGGG"},  // G2
            {"GTGTCCTC", "CGCTCATT"},  // G3
            {"ATCTTCAT", "GCCTCTAT"},  // G4
            {"CGTGGTTG", "GAGCACAA"},  // G5
            {"TTGCATCC", "CTCTTAAC"},  // G6
            {"TCTTAATC", "TCTAGGCT"},  // G7
            {"TGCATTTC", "AATTCTGC"},  // G8
            {"GATGTTTC", "CATTCTCA"},  // G9
            {"ATCTTGTC", "ACTTGCCT"},  // G10
            {"TCATATTC", "ATCATTGC"},  // G11
            {"TGGCCTCT", "GTTCAACA"},  // G12
            {"CGTTGTCT", "CCATTTGC"},  // H1
            {"TCTTGTCA", "GACTTTGC"},  // H2
            {"TATTCCTG", "ATTGGCTC"},  // H3
            {"TCCATGTC", "GTGCTAGC"},  // H4
            {"TTGTCATC", "CTTTCAAC"},  // H5
            {"ATTTCCTG", "ACTATTGC"},  // H6
            {"GTGTCTCC", "ACTGGCTT"},  // H7
            {"GTGTGTGT", "ATTAGGCT"},  // H8
            {"TATGCTTC", "GCCTTTCA"},  // H9
            {"ATGGTGTT", "ATTCTAGG"},  // H10
            {"GAATAATG", "CCTTACAT"},  // H11
            {"CCTCTGTG", "ACATTTGG"}   // H12
        };
        auto pk = [](const char* a) {
            uint32_t v = 0;
            for (int i = 0; i < 8; ++i) v = (v << 2) | (a[i] == 'A' ? 0u : a[i] == 'C' ? 1u : a[i] == 'G' ? 2u : 3u);
            return v;
        };
        std::unordered_map<uint32_t, uint32_t> r;
        for (const auto& p : kPairs) r.emplace(pk(p[0]), pk(p[1]));
        return r;
    }();
    return m;
}

// SPLiT-seq/Parse barcode read: round 1 sits at 78-85. Rewrites a random-hexamer round-1 barcode in place to its well's oligo-dT
// barcode; true when it rewrote. Reads shorter than 86 nt or with N in the barcode are left alone.
constexpr uint16_t kParseBc1Offset = 78;
inline bool merge_parse_bc1(uint8_t* r1, uint16_t r1_len) {
    if (r1_len < kParseBc1Offset + 8) return false;
    const int32_t k = pack_bc8(r1 + kParseBc1Offset);
    if (k < 0) return false;
    const auto& m = parse_v2_bc1_random_to_dt();
    auto it = m.find(static_cast<uint32_t>(k));
    if (it == m.end()) return false;
    uint32_t v = it->second;
    for (int i = 7; i >= 0; --i) { r1[kParseBc1Offset + i] = static_cast<uint8_t>(v & 3u); v >>= 2; }
    return true;
}

}  // namespace singlet::fq
