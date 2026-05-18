// SPDX-License-Identifier: MIT
// lib1fq/protocol.h — Protocol detection and whitelist utilities
//
// Shared by SraEncoder and FastqEncoder for chemistry auto-detection,
// whitelist loading, and barcode matching.

#pragma once

#include "types.h"
#include "writer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet::fq {

// ── Protocol candidate from auto-detection ──

struct ProtocolCandidate {
    std::string tag;
    uint8_t     protocol_id;
    uint16_t    r1_length;
    uint16_t    bc_offset;
    uint16_t    bc_length;
    uint16_t    umi_offset;
    uint16_t    umi_length;
    double      wl_match_rate;
    double      score;
    Confidence  confidence;
    // True when detection succeeded via inverted R2 barcode scoring
    // (VDB stored R1=cDNA, R2=CB+UMI instead of the standard R1=CB+UMI, R2=cDNA).
    // Encoders must physically swap R1↔R2 before writing the .1fq file.
    bool        reads_swapped = false;
};

// ── Known protocol specifications ──

struct CandidateSpec {
    std::string tag;
    uint8_t protocol_id;
    uint16_t r1_len;           // expected R1 length (0 = no R1, barcode in R2)
    uint16_t bc_offset, bc_len;
    uint16_t umi_offset, umi_len;
    std::string whitelist_file;
    std::string linker;         // linker sequence for detection (empty = none)
    uint16_t linker_offset;     // expected position of linker in R1 (0-based)
    bool barcode_in_r2;         // true if barcode read is R2 (not R1)
    std::string adapter3p = ""; // 3' adapter for STAR --clip3pAdapterSeq (empty = none)
    // Per-segment whitelist filenames for CB_UMI_Complex protocols.
    // For single-whitelist protocols these are empty (whitelist_file is used for all segs).
    // For multi-whitelist protocols (e.g. BD Rhapsody CLS1/CLS2/CLS3) each entry names
    // the file for the corresponding CB segment.  Must match len(soloCBposition) exactly.
    std::vector<std::string> per_seg_whitelist_files = {};
};

inline const std::vector<CandidateSpec>& known_protocols() {
    // Protocol specs: tag, id, r1_len, bc_off, bc_len, umi_off, umi_len,
    //                 whitelist_file, linker, linker_offset, barcode_in_r2
    static const std::vector<CandidateSpec> specs = {
        // ── 10x Chromium ──
        {"10x-3p-v3", 1, 28, 0, 16, 16, 12, "3M-february-2018.txt", "", 0, false},
        {"10x-3p-v2", 2, 26, 0, 16, 16, 10, "737K-august-2016.txt", "", 0, false},
        {"10x-3p-v1", 3, 24, 0, 14, 14, 10, "737K-april-2014.txt", "", 0, false},
        {"10x-5p-v3", 4, 28, 0, 16, 16, 12, "3M-february-2018.txt", "", 0, false,
         "AAGCAGTGGTATCAACGCAGAGTACATGGG"},
        {"10x-5p-v2", 5, 26, 0, 16, 16, 10, "737K-august-2016.txt", "", 0, false,
         "AAGCAGTGGTATCAACGCAGAGTACATGGG"},

        // ── Drop-seq family (no whitelist — random bead synthesis) ──
        {"dropseq",   6, 20, 0, 12, 12, 8, "", "", 0, false},
        {"celseq2",   7, 12, 0, 6,  6,  6, "", "", 0, false},
        {"marsseq2",  8, 15, 0, 7,  7,  8, "", "", 0, false},

        // ── sci-RNA-seq3 ──
        // R1 (34bp): [HP_9-10bp] + CAGAGC + [UMI_8bp] + [RT_10bp]
        // HP barcodes at pos 0 (9-10bp), RT barcode at pos 24 (10bp)
        // Linker CAGAGC at ~position 9-10
        {"sci-rna-seq3", 9, 34, 24, 10, 16, 8,
         "scirna3_rt_bc.txt", "CAGAGC", 9, false},

        // ── BD Rhapsody (V1) ──
        // R1 (~60bp): [CLS1_9bp] + ACTGGCCTGCGA + [CLS2_9bp] + GGTAGCGGTGACA + [CLS3_9bp] + [UMI_8bp]
        // First CLS at pos 0, linker at pos 9
        // Each segment uses a distinct 97-barcode set (bd_cls1.txt, bd_cls2.txt, bd_cls3.txt).
        // Source: teichlab.github.io/scg_lib_structs/data/BD/BD_CLS{1,2,3}.txt
        {"bd-rhapsody", 10, 60, 0, 9, 52, 8,
         "bd_cls1.txt", "ACTGGCCTGCGA", 9, false, "",
         {"bd_cls1.txt", "bd_cls2.txt", "bd_cls3.txt"}},

        // ── SPLiT-seq / Parse ──
        // R1=cDNA (66bp), R2 (94bp): [UMI_10bp]+[Rd3_8bp]+linker15+[Rd2_8bp]+linker15+[Rd1_8bp]
        // Barcode in R2. Rd3 at pos 10, Rd2 at pos 33, Rd1 at pos 56
        // Three distinct barcode sets (Rd1/Rd2/Rd3 are different ligation barcodes).
        {"splitseq", 11, 66, 10, 8, 0, 10,
         "splitseq_bc_rd3.txt", "", 0, true, "",
         {"splitseq_bc_rd3.txt", "splitseq_bc_rd2.txt", "splitseq_bc_rd1.txt"}},

        // ── inDrop v1/v2 ──
        // R1: [BC1_8-11bp] + W1_linker(22bp: GAGTGATTGCTTGTGACGCCTT) + [BC2_8bp] + [UMI_6bp]
        // Variable BC1 length. Canonical: BC1=10bp, total R1≈46bp.
        // W1 linker at ~position 10 (search window 8-12 to handle BC1 length variation)
        // BC1 and BC2 use distinct barcode sets.
        {"indrop", 12, 46, 0, 10, 40, 6,
         "indrop_bc1.txt", "GAGTGATTGCTTGTGACGCCTT", 10, false, "",
         {"indrop_bc1.txt", "indrop_bc2.txt"}},

        // ── 10x Chromium v4 (GEM-X) ──
        // Same geometry as v3: R1(28bp) = BC(16bp) + UMI(12bp), R2 = cDNA
        // Uses same 3M barcode set (compatible with 3M-february-2018.txt)
        {"10x-3p-v4", 13, 28, 0, 16, 16, 12, "3M-february-2018.txt", "", 0, false},
        {"10x-5p-v4", 14, 28, 0, 16, 16, 12, "3M-february-2018.txt", "", 0, false,
         "AAGCAGTGGTATCAACGCAGAGTACATGGG"},

        // ── DNBelab C4 (MGI) ──
        // R1 (56bp): BC1(10bp) + L1(10bp) + BC2(10bp) + L2(6bp) + BC3(10bp) + UMI(10bp)
        // Linker1 at pos 10 ("ATCCACGTGC" in most kit versions)
        // First BC1 at pos 0 is the primary barcode for detection.
        // Total composite barcode = BC1+BC2+BC3 = 30bp, but BC1 alone suffices for detection.
        {"dnbelab-c4", 15, 56, 0, 10, 46, 10,
         "dnbelab_c4_bc.txt", "ATCCACGTGC", 10, false},

        // ── Seq-Well ── (Drop-seq geometry on nanowell arrays)
        // Same as Drop-seq: R1(20bp) = BC(12bp) + UMI(8bp), no whitelist
        {"seqwell", 16, 20, 0, 12, 12, 8, "", "", 0, false},

        // ── ddSEQ (Bio-Rad) ──
        // R1(68bp): [BC1_6bp] + ACG(3bp linker) + [BC2_6bp] + GAC(3bp linker) + [BC3_6bp] + TCA(3bp linker) + [UMI_8bp]
        // Linker "ACG" at position 6
        {"ddseq", 17, 68, 0, 6, 27, 8, "", "ACG", 6, false},

        // ── Quartz-seq2 ──
        // R1(23bp): BC(15bp@0) + UMI(8bp@15), R2 = full-length cDNA
        // No linker sequence in R1. Plate-based with 1536 barcodes.
        {"quartzseq2", 18, 23, 0, 15, 15, 8, "quartzseq2_bc.txt", "", 0, false},

        // ── Microwell-seq ──
        // R1(≥54bp): BC1(6@0) + CGACTCACTACAGGG(15@6) + BC2(6@21) + TCGGTGACACGATCG(15@27) + BC3(6@42) + UMI(6@48)
        // BC1 is the representative barcode for detection; BC2==BC3 (same 96-well set).
        // CB_UMI_Complex in STARsolo.
        {"microwell-seq", 19, 54, 0, 6, 48, 6,
         "microwell_bc1_bare.txt", "CGACTCACTACAGGG", 6, false, "",
         {"microwell_bc1_bare.txt", "microwell_bc1_bare.txt", "microwell_bc1_bare.txt"}},

        // ── SureCell (Bio-Rad ddSEQ predecessor, WTA 3' kit) ──
        // R1(68bp): BC1(6@0) + TAGCCATCGCATTGC(15@6) + BC2(6@21) + TACCTCTGAGCTGAA(15@27) + BC3(6@42) + ACG(3@48) + UMI(8@51)
        // All three BC positions use the same 96-barcode set.
        // CB_UMI_Complex in STARsolo.
        {"surecell", 20, 68, 0, 6, 51, 8,
         "surecell_bc.txt", "TAGCCATCGCATTGC", 6, false, "",
         {"surecell_bc.txt", "surecell_bc.txt", "surecell_bc.txt"}},

        // ── STRT-seq (plate-based 5' cap-switching, original) ──
        // R1: BC(6bp@0) + GGG template-switch + cDNA; plate/well barcodes, no universal WL.
        // Treated as 5' protocol; classified as SC_RNA_5PRIME.
        // soloUMIlen=0 when run without UMI (original STRT-seq).
        {"strtseq", 21, 0, 0, 6, 6, 0, "", "", 0, false,
         "AAGCAGTGGTATCAACGCAGAGTACATGGG"},

        // ── 10x Multiome GEX (ARC) ──
        // R1(28bp): BC(16bp@0) + UMI(12bp@16), identical geometry to v3
        // but uses the ARC GEX barcode whitelist (gex_737K-arc-v1.txt).
        // Classified as SC_MULTIOME_GEX.
        {"10x-arc-gex", 22, 28, 0, 16, 16, 12, "gex_737K-arc-v1.txt", "", 0, false,
         "AAGCAGTGGTATCAACGCAGAGTACATGGG"},

        // ── 10x scATAC ──
        // R1/R2 = genomic DNA fragments; barcode is in I2 index read (not R1/R2).
        // Not STAR-processable. Detected via Tn5 signature + symmetric read lengths.
        // bc_len=0 signals "not detectable from R1 geometry alone".
        {"10x-atac", 23, 50, 0, 0, 0, 0, "737K-cratac-v1.txt", "", 0, false},

        // ── 10x Visium (Spatial Transcriptomics) ──
        // R1(28bp): BC(16bp@0) + UMI(12bp@16), geometrically identical to 10x v3.
        // Spatial barcode WL (~4992 barcodes) is Space Ranger licensed — not bundled.
        // Without WL, detection falls back to 10x-3p-v3. Classified as SPATIAL_RNA.
        {"10x-visium", 24, 28, 0, 16, 16, 12, "", "", 0, false},

        // ── CITE-seq GEX ──
        // R1(28bp): CB(16bp@0) + UMI(12bp@16) — identical geometry to 10x-3p-v3.
        // Uses the same 3M barcode whitelist. Classified as CITE_SEQ_GEX.
        // Bug fix: previously no CandidateSpec entry existed for cite-seq-gex, causing
        // protocol_id=0 and a heuristic CB=19/UMI=19 from R1=38bp SRA data.
        {"cite-seq-gex", 25, 28, 0, 16, 16, 12, "3M-february-2018.txt", "", 0, false},
    };
    return specs;
}

// ── Protocol tag normalization ──
// Lowercases and removes hyphens/underscores so that user-supplied tags like
// "Drop-seq", "drop_seq", "DROPSEQ" all match the canonical registry key "dropseq".
inline std::string normalize_tag(const std::string& tag) {
    std::string out;
    out.reserve(tag.size());
    for (char c : tag) {
        if (c == '-' || c == '_') continue;
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

// Find a CandidateSpec by tag (case-insensitive, ignores hyphens/underscores).
// Also resolves common user-facing aliases (e.g. "10x-v2" → "10x-3p-v2").
// Returns nullptr if not found.
inline const CandidateSpec* find_protocol_spec(const std::string& tag) {
    const std::string norm = normalize_tag(tag);

    // Direct match first.
    for (const auto& spec : known_protocols()) {
        if (normalize_tag(spec.tag) == norm)
            return &spec;
    }

    // Alias table: maps normalized alias → canonical tag string.
    // Handles common user-supplied names from val1_samples.csv and the CLI.
    static const std::unordered_map<std::string, std::string> aliases = {
        // 10x 3' shorthand
        {"10xv2",        "10x-3p-v2"},
        {"10xv3",        "10x-3p-v3"},
        {"10xv4",        "10x-3p-v4"},
        // 10x 5' shorthands
        {"10xv35prime",  "10x-5p-v3"},
        {"10x5primev3",  "10x-5p-v3"},
        {"10xv25prime",  "10x-5p-v2"},
        {"10x5primev2",  "10x-5p-v2"},
        // 10x Multiome / ARC
        {"10xmultiome",  "10x-arc-gex"},
        {"10xarcmultiome", "10x-arc-gex"},
        {"multiome",     "10x-arc-gex"},
        // sci-RNA-seq shorthand
        {"scirna",       "sci-rna-seq3"},
        {"scirnaseq",    "sci-rna-seq3"},
        // MARS-seq shorthand (marsseq → marsseq2)
        {"marsseq",      "marsseq2"},
        // Microwell-seq shorthand
        {"microwell",    "microwell-seq"},
        // DNBelab shorthand (dnbelab → dnbelab-c4)
        {"dnbelab",      "dnbelab-c4"},
        {"dnbelabc",     "dnbelab-c4"},
        // Parse Biosciences (SPLiT-seq chemistry)
        {"parse",        "splitseq"},
        {"parsebio",     "splitseq"},
        // STRT-seq shorthand
        {"strtseq2",     "strtseq"},
        // Seq-Well already normalizes correctly (seqwell == seqwell)
        // CITE-seq GEX shorthands
        {"citeseq",      "cite-seq-gex"},
        {"citeseqgex",   "cite-seq-gex"},
        {"cite",         "cite-seq-gex"},
        // 10x Multiome shorthands (map to 10x-arc-gex)
        {"10xmultiome",     "10x-arc-gex"},
        {"10xmultiomegex",  "10x-arc-gex"},
        {"10xarcgex",       "10x-arc-gex"},
        {"10xarc",          "10x-arc-gex"},
    };

    auto it = aliases.find(norm);
    if (it != aliases.end()) {
        const std::string canonical_norm = normalize_tag(it->second);
        for (const auto& spec : known_protocols()) {
            if (normalize_tag(spec.tag) == canonical_norm)
                return &spec;
        }
    }

    return nullptr;
}

// ── Linker sequence search ──
// Returns the fraction of reads containing the linker at the expected position (±2bp)

inline double linker_match_rate(
        const std::string& linker,
        uint16_t expected_pos,
        const uint8_t* const* reads,
        const uint16_t* read_lens,
        uint32_t n_reads) {
    if (linker.empty() || n_reads == 0) return 0.0;

    // Convert linker to numeric
    std::vector<uint8_t> linker_num(linker.size());
    for (size_t i = 0; i < linker.size(); ++i)
        linker_num[i] = nuc::ascii_to_num(linker[i]);

    uint16_t llen = static_cast<uint16_t>(linker_num.size());
    uint32_t matches = 0;
    // For short linkers (≤4bp), require exact match to avoid false positives.
    // A 3bp motif with 1mm tolerance matches ~75% of random sequences.
    const uint16_t max_mm = (llen <= 4) ? 0 : 1;

    for (uint32_t r = 0; r < n_reads; ++r) {
        uint16_t rlen = read_lens[r];
        if (rlen < llen) continue;

        // Search in window [expected_pos-2, expected_pos+3]
        int16_t lo = static_cast<int16_t>(expected_pos) - 2;
        int16_t hi = static_cast<int16_t>(expected_pos) + 3;
        if (lo < 0) lo = 0;
        if (hi + llen > rlen) hi = rlen - llen;

        for (int16_t pos = lo; pos <= hi; ++pos) {
            uint16_t mm = 0;
            for (uint16_t j = 0; j < llen; ++j) {
                if (reads[r][pos + j] != linker_num[j]) {
                    if (++mm > max_mm) break;
                }
            }
            if (mm <= max_mm) { ++matches; break; }
        }
    }
    return static_cast<double>(matches) / n_reads;
}

// ── Whitelist loading ──

inline std::unordered_set<uint64_t> load_whitelist_hashes(
        const std::string& path, uint16_t bc_len) {
    std::unordered_set<uint64_t> hashes;
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return hashes;

    // Count lines for reserve (avoid repeated rehash for multi-million WL)
    uint32_t line_count = 0;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), fp)) ++line_count;
    std::fseek(fp, 0, SEEK_SET);
    hashes.reserve(line_count * 2);

    while (std::fgets(buf, sizeof(buf), fp)) {
        size_t len = std::strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) --len;
        if (len < bc_len) continue;

        uint64_t h = 0;
        for (uint16_t i = 0; i < bc_len; ++i) {
            h = h * 5 + nuc::ascii_to_num(buf[i]);
        }
        hashes.insert(h);
    }
    std::fclose(fp);
    return hashes;
}

inline std::vector<std::vector<uint8_t>> load_whitelist_sequences(
        const std::string& path, uint16_t bc_len) {
    std::vector<std::vector<uint8_t>> seqs;
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return seqs;

    // Count lines for reserve
    uint32_t line_count = 0;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), fp)) ++line_count;
    std::fseek(fp, 0, SEEK_SET);
    seqs.reserve(line_count);

    while (std::fgets(buf, sizeof(buf), fp)) {
        size_t len = std::strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) --len;
        if (len < bc_len) continue;

        std::vector<uint8_t> seq(bc_len);
        for (uint16_t i = 0; i < bc_len; ++i) {
            seq[i] = nuc::ascii_to_num(buf[i]);
        }
        seqs.push_back(std::move(seq));
    }
    std::fclose(fp);
    return seqs;
}

// ── Barcode hashing and matching ──

inline uint64_t hash_barcode(const uint8_t* bc, uint16_t len) {
    uint64_t h = 0;
    for (uint16_t i = 0; i < len; ++i) {
        h = h * 5 + bc[i];
    }
    return h;
}

inline bool hamming1_match(const uint8_t* bc, uint16_t len,
                           const std::unordered_set<uint64_t>& wl) {
    uint64_t h = hash_barcode(bc, len);
    if (wl.count(h)) return true;

    std::vector<uint8_t> tmp(bc, bc + len);
    for (uint16_t i = 0; i < len; ++i) {
        uint8_t orig = tmp[i];
        for (uint8_t sub = 0; sub < 4; ++sub) {
            if (sub == orig) continue;
            tmp[i] = sub;
            h = hash_barcode(tmp.data(), len);
            if (wl.count(h)) return true;
        }
        tmp[i] = orig;
    }
    return false;
}

// ── Protocol auto-detection from probe reads ──
// Template parameter SpotT must have: r1_seq (indexable uint8_t),
// r1_len (uint16_t), r2_seq (indexable uint8_t), r2_len (uint16_t)

template<typename SpotT>
ProtocolCandidate detect_protocol(
        const std::vector<SpotT>& probe_spots,
        uint16_t r1_len,
        uint16_t r2_len,
        const std::vector<std::string>& whitelist_dirs) {

    const auto& known = known_protocols();

    // ── Determine read layout ──
    // Some datasets have R1=0 with all data in R2 (e.g. celseq2),
    // or have very short R2 with barcodes and long R1 with cDNA (inverted).
    bool concat_mode = (r2_len == 0 && r1_len > 0);
    bool inverted = false;

    // Detect inverted reads: R2 very short (12-34bp), R1 long (>40bp)
    // This means R2 is the barcode read, R1 is cDNA.
    if (r2_len > 0 && r2_len <= 34 && r1_len > 40 && r1_len > r2_len * 2) {
        inverted = true;
    }

    // ── Variable-R2 fallback for inverted detection ──────────────────────
    // When r2_len=0 (VDB reports variable R2 across spots, e.g. due to mixed
    // 2/3-segment spots in older SRA deposits), the inverted check above is
    // skipped.  Sample individual probe spot r2_lens: if a majority share the
    // same short length (≤34bp) while R1 is long (>40bp), treat as inverted.
    if (!inverted && r2_len == 0 && r1_len > 40 && !probe_spots.empty()) {
        const uint32_t n_samp = std::min<uint32_t>(
            2000u, static_cast<uint32_t>(probe_spots.size()));
        std::unordered_map<uint16_t, uint32_t> r2_hist;
        for (uint32_t i = 0; i < n_samp; ++i) {
            const uint16_t rl = probe_spots[i].r2_len;
            if (rl > 0 && rl <= 40) r2_hist[rl]++;
        }
        uint16_t maj_r2 = 0; uint32_t maj_cnt = 0;
        for (const auto& kv : r2_hist)
            if (kv.second > maj_cnt) { maj_cnt = kv.second; maj_r2 = kv.first; }
        // Require ≥50% consistency and invertible geometry
        if (maj_r2 > 0 && maj_cnt >= n_samp / 2 &&
                maj_r2 <= 34 && r1_len > maj_r2 * 2) {
            r2_len = maj_r2;
            inverted = true;
            concat_mode = false;  // Override: we now have an effective R2
        }
    }

    // Filter candidates by R1/barcode-read length
    auto barcode_read_len = inverted ? r2_len : r1_len;
    auto bio_read_len = inverted ? r1_len : r2_len;

    std::vector<const CandidateSpec*> candidates;
    for (const auto& k : known) {
        uint16_t expected = k.r1_len;

        if (barcode_read_len == 0 && bio_read_len > 0) {
            // R1=0, all data in single segment — check if protocol could work in concat mode
            if (expected > 0 && expected <= bio_read_len && bio_read_len - expected >= 10) {
                candidates.push_back(&k);
            }
            continue;
        }

        if (expected == barcode_read_len) {
            candidates.push_back(&k);
        } else if (!concat_mode && barcode_read_len > expected &&
                   barcode_read_len <= expected + 4) {
            // Slightly over-sequenced (1-4bp extra, common in real SRA data)
            candidates.push_back(&k);
        } else if (concat_mode && expected > 0 && expected < r1_len &&
                   r1_len - expected >= 10) {
            candidates.push_back(&k);
        } else if (!concat_mode && barcode_read_len > expected &&
                   barcode_read_len >= expected + 10) {
            // Over-sequenced barcode read (10+ bp extra → cDNA in R1)
            candidates.push_back(&k);
        }
    }

    if (candidates.empty()) {
        return {"UNKNOWN", 0, 0, 0, 0, 0, 0, 0.0, 0.0, Confidence::NONE};
    }

    // Sort candidates: exact R1 match first, then linker-based, then by whitelist
    // presence (with WL > without WL), then by WL file name length.
    // Protocols with whitelist/linker are more discriminating and should be tried
    // first to avoid false positives from generic-only scoring hitting HIGH early.
    std::sort(candidates.begin(), candidates.end(),
        [barcode_read_len](const CandidateSpec* a, const CandidateSpec* b) {
            // Exact R1 match is highest priority
            bool a_exact = (a->r1_len == barcode_read_len);
            bool b_exact = (b->r1_len == barcode_read_len);
            if (a_exact != b_exact) return a_exact;
            // Linker-based detection is fast and discriminating
            bool a_link = !a->linker.empty();
            bool b_link = !b->linker.empty();
            if (a_link != b_link) return a_link;
            // Prefer protocols WITH whitelist over those without
            // (whitelist validation is stronger than generic scoring)
            bool a_wl = !a->whitelist_file.empty();
            bool b_wl = !b->whitelist_file.empty();
            if (a_wl != b_wl) return a_wl;
            // Among protocols with whitelists, prefer lower protocol_id.
            // Lower IDs are more common/standard (v3=1, v2=2, ... arc-gex=22).
            // This prevents rarer specialty protocols (arc-gex, visium) from being
            // evaluated before common ones (v3, v2) purely because of filename length.
            // If v3 scores HIGH first, the early-exit prevents arc-gex from stealing it.
            return a->protocol_id < b->protocol_id;
        });

    ProtocolCandidate best = {"UNKNOWN", 0, 0, 0, 0, 0, 0, 0.0, 0.0, Confidence::NONE};
    size_t best_wl_size = 0;  // whitelist size of current best, for tie-break

    // ── WL-defensive override tracking (AUTOFIX-E2E-A-PROTOCOL-REGRESSION-V1) ──
    // Track the best WL candidate (has_wl=true, rate>0) and best non-WL candidate
    // separately.  After the loop, if a non-WL candidate won over a WL candidate
    // that had confirmed barcode matches (rate > 5%), promote the WL candidate.
    double   wl_def_best_score  = -1.0;
    double   wl_def_best_rate   = 0.0;
    double   nowl_def_best_score = -1.0;
    std::string wl_def_best_tag, nowl_def_best_tag;
    uint8_t  wl_def_best_pid    = 0;
    uint16_t wl_def_best_r1len  = 0;
    uint16_t wl_def_best_bcoff  = 0, wl_def_best_bclen  = 0;
    uint16_t wl_def_best_umioff = 0, wl_def_best_umilen = 0;
    Confidence wl_def_best_conf = Confidence::NONE;
    size_t   wl_def_best_wlsz   = 0;

    // ── Cache loaded whitelists (key = filename + bc_len) ──
    std::unordered_map<std::string, std::unordered_set<uint64_t>> wl_cache;

    // ── Pre-compute geometries covered by WL-based protocols ──
    // Used to suppress no-WL, no-linker protocols (e.g. 10x-visium) that share
    // identical geometry with a WL-based protocol (e.g. 10x-3p-v3).  Without its
    // own whitelist, such a protocol cannot discriminate itself from the WL protocol,
    // so it must never be chosen over one.
    // Enforces the stated intent in known_protocols(): "Without WL, detection falls
    // back to 10x-3p-v3."
    std::unordered_set<uint64_t> wl_geometries;
    auto pack_geom5 = [](uint16_t r1, uint16_t bco, uint16_t bcl,
                         uint16_t umio, uint16_t umil) -> uint64_t {
        // Combine five 16-bit fields into one 64-bit integer for unambiguous keying.
        // Values are small (≤512), so a polynomial hash over them is collision-free
        // for the small number of known protocols (≤32).
        uint64_t h = 17;
        h = h * 31 + r1; h = h * 31 + bco; h = h * 31 + bcl;
        h = h * 31 + umio; h = h * 31 + umil;
        return h;
    };
    for (const auto& k : known_protocols()) {
        if (!k.whitelist_file.empty()) {
            wl_geometries.insert(pack_geom5(k.r1_len, k.bc_offset, k.bc_len,
                                            k.umi_offset, k.umi_len));
        }
    }

    for (const auto* cand : candidates) {
        // ── Skip geometry-only assays from barcode scoring ──
        // Protocols with bc_len=0 && umi_len=0 && no-linker (i.e. 10x-atac) cannot be
        // detected by barcode/UMI matching. Their comment in known_protocols() says:
        // "Not STAR-processable. Detected via Tn5 signature + symmetric read lengths."
        // Including them here causes false positives when R1 geometry coincidentally
        // matches (e.g. R1=50bp from a multi-segment VDB spot vs ATAC's r1_len=50).
        // classify_assay() handles ATAC correctly via Tn5 adapter detection (Rule 2).
        if (cand->bc_len == 0 && cand->umi_len == 0 && cand->linker.empty()) continue;

        // ── WL-geometry suppression ──
        // Skip no-WL, no-linker protocols (e.g. 10x-visium) whose geometry is
        // identical to a known WL-based protocol.  Without its own whitelist the
        // no-WL protocol cannot discriminate itself from the WL protocol no matter
        // how the structural scoring turns out, so it must never win over it.
        // Not applied to protocols with a linker (linker IS an independent signal).
        if (cand->whitelist_file.empty() && cand->linker.empty()) {
            uint64_t g = pack_geom5(cand->r1_len, cand->bc_offset, cand->bc_len,
                                    cand->umi_offset, cand->umi_len);
            if (wl_geometries.count(g)) continue;  // geometry covered; fall back to WL protocol
        }

        uint32_t n_test = std::min<uint32_t>(5000,
            static_cast<uint32_t>(probe_spots.size()));

        // Determine which physical read contains barcodes
        // If inverted: barcodes are in r2_seq; if normal: barcodes in r1_seq
        // If R1=0 (concat_mode from R2): barcodes at start of r2_seq
        bool use_r2_for_bc = inverted || (barcode_read_len == 0 && bio_read_len > 0);

        // ── Prepare pointers for barcode read ──
        // (We use lambda to get the barcode sequence for spot i)
        auto bc_seq = [&](uint32_t i) -> const uint8_t* {
            if (use_r2_for_bc) return probe_spots[i].r2_seq.data();
            return probe_spots[i].r1_seq.data();
        };
        auto bc_read_len = [&](uint32_t i) -> uint16_t {
            if (use_r2_for_bc) return probe_spots[i].r2_len;
            return probe_spots[i].r1_len;
        };

        // ── Linker match rate ──
        double linker_rate = 0.0;
        if (!cand->linker.empty()) {
            // Prepare batch of read pointers for linker search
            std::vector<const uint8_t*> read_ptrs(n_test);
            std::vector<uint16_t> read_lens(n_test);
            for (uint32_t i = 0; i < n_test; ++i) {
                read_ptrs[i] = bc_seq(i);
                read_lens[i] = bc_read_len(i);
            }
            linker_rate = linker_match_rate(
                cand->linker, cand->linker_offset,
                read_ptrs.data(), read_lens.data(), n_test);
        }

        // ── Whitelist match rate ──
        double rate = 0.0;
        bool has_wl = false;
        size_t curr_wl_size = 0;  // for tie-break logic

        if (!cand->whitelist_file.empty()) {
            std::string cache_key = cand->whitelist_file + ":" +
                                    std::to_string(cand->bc_len);
            auto it = wl_cache.find(cache_key);
            if (it == wl_cache.end()) {
                std::unordered_set<uint64_t> wl_hashes;
                for (const auto& dir : whitelist_dirs) {
                    std::string path = dir + "/" + cand->whitelist_file;
                    wl_hashes = load_whitelist_hashes(path, cand->bc_len);
                    if (!wl_hashes.empty()) break;
                }
                it = wl_cache.emplace(cache_key, std::move(wl_hashes)).first;
            }
            const auto& wl_hashes = it->second;
            if (!wl_hashes.empty()) {
                has_wl = true;
                curr_wl_size = wl_hashes.size();
                uint32_t matches = 0;
                // For large whitelists (>=100K entries) use exact match for scoring.
                // hamming1 inflates match rates so much that almost any barcode matches,
                // making large whitelists non-discriminating. Fall back to hamming1 only
                // if exact match rate is poor (< 0.3), indicating a genuine mismatch.
                bool use_exact = (wl_hashes.size() >= 100000);
                for (uint32_t i = 0; i < n_test; ++i) {
                    const uint8_t* read = bc_seq(i);
                    uint16_t rlen = bc_read_len(i);
                    if (rlen < cand->bc_offset + cand->bc_len) continue;
                    const uint8_t* bc = read + cand->bc_offset;
                    if (use_exact) {
                        uint64_t h = hash_barcode(bc, cand->bc_len);
                        if (wl_hashes.count(h)) ++matches;
                    } else {
                        if (hamming1_match(bc, cand->bc_len, wl_hashes)) ++matches;
                    }
                }
                rate = static_cast<double>(matches) / n_test;
                // Fallback: if exact match rate is poor, retry with hamming1
                if (use_exact && rate < 0.3) {
                    matches = 0;
                    for (uint32_t i = 0; i < n_test; ++i) {
                        const uint8_t* read = bc_seq(i);
                        uint16_t rlen = bc_read_len(i);
                        if (rlen < cand->bc_offset + cand->bc_len) continue;
                        const uint8_t* bc = read + cand->bc_offset;
                        if (hamming1_match(bc, cand->bc_len, wl_hashes)) ++matches;
                    }
                    rate = static_cast<double>(matches) / n_test;
                }
            }
        }

        // ── UMI entropy ──
        double umi_entropy = 0.0;
        if (cand->umi_len > 0 && n_test > 100) {
            double total_h = 0.0;
            for (uint16_t pos = 0; pos < cand->umi_len; ++pos) {
                uint32_t counts[4] = {0};
                uint32_t valid = 0;
                for (uint32_t i = 0; i < n_test; ++i) {
                    const uint8_t* read = bc_seq(i);
                    uint16_t rlen = bc_read_len(i);
                    if (rlen > cand->umi_offset + pos) {
                        uint8_t base = read[cand->umi_offset + pos];
                        if (base < 4) { counts[base]++; valid++; }
                    }
                }
                if (valid < 10) continue;
                double h = 0.0;
                for (int b = 0; b < 4; ++b) {
                    if (counts[b] > 0) {
                        double p = static_cast<double>(counts[b]) / valid;
                        h -= p * std::log2(p);
                    }
                }
                total_h += h;
            }
            umi_entropy = total_h / cand->umi_len;
        }

        // ── PolyA fraction ──
        uint32_t polya_count = 0;
        if (concat_mode && cand->r1_len < r1_len) {
            uint16_t r2_start = cand->r1_len;
            for (uint32_t i = 0; i < n_test; ++i) {
                const auto& s = probe_spots[i];
                if (s.r1_len < r2_start + 20) continue;
                uint32_t a_count = 0;
                for (uint16_t j = s.r1_len - 15; j < s.r1_len; ++j) {
                    if (s.r1_seq[j] == 0) ++a_count;
                }
                if (a_count >= 10) ++polya_count;
            }
        } else if (use_r2_for_bc && bio_read_len > 0) {
            // Inverted: cDNA is in R1, check polyA at end of R1
            for (uint32_t i = 0; i < n_test; ++i) {
                const auto& s = probe_spots[i];
                if (s.r1_len < 20) continue;
                uint32_t a_count = 0;
                for (uint16_t j = s.r1_len - 15; j < s.r1_len; ++j) {
                    if (s.r1_seq[j] == 0) ++a_count;
                }
                if (a_count >= 10) ++polya_count;
            }
        } else {
            for (uint32_t i = 0; i < n_test; ++i) {
                const auto& s = probe_spots[i];
                if (s.r2_len < 20) continue;
                uint32_t a_count = 0;
                for (uint16_t j = s.r2_len - 15; j < s.r2_len; ++j) {
                    if (s.r2_seq[j] == 0) ++a_count;
                }
                if (a_count >= 10) ++polya_count;
            }
        }
        double polya_frac = (n_test > 0) ?
            static_cast<double>(polya_count) / n_test : 0.0;

        // ── Scoring ──
        double score = 0.0;

        // ── R1 geometry bonus ──
        // Reward candidates whose expected R1 length matches measured barcode-read length.
        // This is the single most reliable discriminator for protocols with identical CB sizes
        // (e.g. 10x-v2 expected=26, 10x-v3/arc-gex expected=28): even a 2bp difference is
        // detectable and should influence the score.  Penalise when actual R1 is too short to
        // hold the full CB+UMI region (impossible for this protocol to work on these reads).
        // In concat_mode the barcode-read length includes cDNA, so the expected R1 length is
        // always a substring of the combined read — geometry comparison is not useful there.
        double geometry_bonus = 0.0;
        if (cand->r1_len > 0 && !concat_mode) {
            int32_t diff = static_cast<int32_t>(barcode_read_len) -
                           static_cast<int32_t>(cand->r1_len);
            if (diff == 0) {
                geometry_bonus = 0.15;   // exact R1 length match
            } else if (diff >= -2 && diff <= 2) {
                geometry_bonus = 0.10;   // close match (±2bp trim/pad variance)
            } else if (diff > 0 && diff <= 10) {
                geometry_bonus = 0.05;   // slightly oversized (few bp cDNA bleed-in)
            } else if (diff < -2) {
                geometry_bonus = -0.20;  // R1 too short — CB+UMI cannot fit
            }
            // diff > 10: massively oversized — no geometry bonus (WL is the discriminator)
        }

        // ── UMI overflow penalty ──
        // When a protocol declares a UMI that extends beyond the actual read length, the
        // last few UMI positions are unavailable.  Penalise lightly to prefer protocols
        // whose UMI fits entirely within the observed read.
        if (cand->umi_len > 0 && !concat_mode && barcode_read_len > 0) {
            uint16_t umi_end = static_cast<uint16_t>(cand->umi_offset + cand->umi_len);
            if (umi_end > barcode_read_len) {
                // fraction of UMI positions that fall outside the actual read
                uint16_t overflow = umi_end - barcode_read_len;
                double overflow_frac = static_cast<double>(overflow) / cand->umi_len;
                geometry_bonus -= 0.10 * overflow_frac;
            }
        }

        // ── Internal polyT/polyA at BC+UMI boundary (concat mode signal) ──
        // For 3' scRNA-seq concat reads: polyT/polyA run starts right after
        // barcode+UMI region. Very specific discriminator for detecting
        // correct BC/UMI boundary position.
        double internal_polyT = 0.0;
        if (concat_mode && cand->bc_len > 0 && cand->umi_len > 0) {
            uint16_t polyT_start = cand->bc_offset + cand->bc_len + cand->umi_len;
            uint32_t polyT_hits = 0;
            for (uint32_t i = 0; i < n_test; ++i) {
                const uint8_t* read = bc_seq(i);
                uint16_t rlen = bc_read_len(i);
                if (rlen < polyT_start + 8) continue;
                uint16_t window = std::min<uint16_t>(12, rlen - polyT_start);
                uint32_t t_count = 0;
                for (uint16_t j = 0; j < window; ++j) {
                    if (read[polyT_start + j] == 3) ++t_count; // T
                }
                // Also check polyA (reverse strand)
                uint32_t a_count = 0;
                for (uint16_t j = 0; j < window; ++j) {
                    if (read[polyT_start + j] == 0) ++a_count; // A
                }
                if (t_count >= 6 || a_count >= 6) ++polyT_hits;
            }
            internal_polyT = (n_test > 0) ?
                static_cast<double>(polyT_hits) / n_test : 0.0;
        }

        if (has_wl && !cand->linker.empty()) {
            // Protocols with BOTH whitelist AND linker (e.g. sci-RNA-seq3, BD Rhapsody)
            score += 0.35 * rate;
            score += 0.35 * linker_rate;
            score += 0.15 * std::min(umi_entropy / 2.0, 1.0);
            score += 0.15 * (rate > 0.3 ? 1.0 : 0.0);
            score += geometry_bonus;
        } else if (has_wl) {
            // Whitelist-only (10x, etc.)
            score += 0.50 * rate;
            score += 0.20 * std::min(umi_entropy / 2.0, 1.0);
            score += 0.10 * polya_frac;
            score += 0.20 * (rate > 0.5 ? 1.0 : 0.0);
            score += geometry_bonus;
        } else if (!cand->linker.empty()) {
            // Linker-only detection
            score += 0.50 * linker_rate;
            score += 0.25 * std::min(umi_entropy / 2.0, 1.0);
            score += 0.15 * polya_frac;
            score += 0.10 * (linker_rate > 0.5 ? 1.0 : 0.0);
            score += geometry_bonus;
        } else if (!concat_mode && !use_r2_for_bc) {
            // Non-whitelist with proper segments: R1 length is discriminating
            double r1_match = (cand->r1_len == barcode_read_len) ? 1.0 : 0.0;
            double umi_norm = std::min(umi_entropy / 2.0, 1.0);
            double umi_good = (umi_entropy > 1.5) ? 1.0 : 0.0;
            score += 0.35 * r1_match;
            score += 0.30 * umi_norm;
            score += 0.15 * polya_frac;
            score += 0.20 * umi_good;
            score += geometry_bonus;
        } else {
            // Non-whitelist concat/inverted mode: structural analysis
            // Use internal polyT at BC+UMI boundary as primary signal
            double bc_entropy = 0.0;
            if (cand->bc_len > 0 && n_test > 100) {
                double total_h = 0.0;
                for (uint16_t pos = 0; pos < cand->bc_len; ++pos) {
                    uint32_t counts[4] = {0}, valid = 0;
                    for (uint32_t i = 0; i < n_test; ++i) {
                        const uint8_t* read = bc_seq(i);
                        uint16_t rlen = bc_read_len(i);
                        if (rlen > cand->bc_offset + pos) {
                            uint8_t b = read[cand->bc_offset + pos];
                            if (b < 4) { counts[b]++; valid++; }
                        }
                    }
                    if (valid < 10) continue;
                    double h = 0.0;
                    for (int b = 0; b < 4; ++b) {
                        if (counts[b] > 0) {
                            double p = static_cast<double>(counts[b]) / valid;
                            h -= p * std::log2(p);
                        }
                    }
                    total_h += h;
                }
                bc_entropy = total_h / cand->bc_len;
            }

            double umi_norm = std::min(umi_entropy / 2.0, 1.0);
            double umi_good = (umi_entropy > 1.5) ? 1.0 : 0.0;
            double bc_structured = (bc_entropy > 0.5 && bc_entropy < umi_entropy) ?
                                   1.0 : 0.0;

            // Internal polyT is the strongest signal for non-WL concat detection.
            // A high polyT rate at the exact BC+UMI boundary confirms protocol geometry.
            score += 0.40 * internal_polyT;
            score += 0.20 * umi_norm;
            score += 0.15 * umi_good;
            score += 0.10 * polya_frac;
            score += 0.15 * bc_structured;
            score += geometry_bonus;
        }

        Confidence conf = Confidence::NONE;
        if (score >= 0.85)      conf = Confidence::HIGH;
        else if (score >= 0.60) conf = Confidence::MEDIUM;
        else if (score >= 0.40) conf = Confidence::LOW;

        // Non-WL, non-linker protocols can't exceed MEDIUM to avoid
        // preempting more specific WL-based detections via early exit.
        if (!has_wl && cand->linker.empty() && conf > Confidence::MEDIUM)
            conf = Confidence::MEDIUM;

        // ── Update WL-defensive tracking ──
        if (has_wl && rate > 0.0 && score > wl_def_best_score) {
            wl_def_best_score  = score;
            wl_def_best_rate   = rate;
            wl_def_best_tag    = cand->tag;
            wl_def_best_pid    = cand->protocol_id;
            wl_def_best_r1len  = cand->r1_len;
            wl_def_best_bcoff  = cand->bc_offset;
            wl_def_best_bclen  = cand->bc_len;
            wl_def_best_umioff = cand->umi_offset;
            wl_def_best_umilen = cand->umi_len;
            wl_def_best_conf   = conf;
            wl_def_best_wlsz   = curr_wl_size;
        } else if (!has_wl && score > nowl_def_best_score) {
            nowl_def_best_score = score;
            nowl_def_best_tag   = cand->tag;
        }

        // Geometry tie-break: when two candidates have identical CB/UMI layout and
        // similar whitelist match rates (within 0.05), prefer the one with:
        //   (a) the larger whitelist (more general; smaller is usually a subset), OR
        //   (b) the lower protocol_id when whitelist sizes are equal.
        // Condition (b) fixes 10x-arc-gex (id=22) incorrectly beating 10x-3p-v3 (id=1):
        // both share r1_len=28, bc=16+umi=12, and when the bundled arc whitelist happens
        // to be the same file as the v3 whitelist, match rates are identical.  Standard
        // 3p/5p libraries vastly outnumber multiome, so the lower protocol_id (earlier,
        // more common protocol) should win the tie.
        bool tie_break_wins = has_wl && best_wl_size > 0 &&
                              score >= best.score - 0.05 &&
                              (curr_wl_size > best_wl_size ||
                               (curr_wl_size == best_wl_size &&
                                cand->protocol_id < best.protocol_id));

        if (score > best.score || tie_break_wins) {
            best.tag = cand->tag;
            best.protocol_id = cand->protocol_id;
            best.r1_length = cand->r1_len;
            best.bc_offset = cand->bc_offset;
            best.bc_length = cand->bc_len;
            best.umi_offset = cand->umi_offset;
            best.umi_length = cand->umi_len;
            best.wl_match_rate = rate;
            best.score = score;
            best.confidence = conf;
            best_wl_size = curr_wl_size;
        }

        // Early exit: HIGH confidence match found, no need to try more candidates
        if (best.confidence >= Confidence::HIGH) break;
    }

    // ── WL-defensive override ──────────────────────────────────────────────────
    // Barcode evidence (actual WL matching) MUST trump heuristic-only scoring.
    // When a non-WL candidate outscored a WL candidate purely on UMI/polyA/geometry
    // heuristics, but the WL candidate had confirmed barcode matches (rate > 5%),
    // promote the WL candidate's score above the non-WL winner.
    // Reference: AUTOFIX-E2E-A-PROTOCOL-REGRESSION-V1
    if (wl_def_best_score >= 0.0 && nowl_def_best_score >= 0.0 &&
            nowl_def_best_score >= wl_def_best_score &&
            wl_def_best_rate > 0.05) {
        double old_score = wl_def_best_score;
        double new_score = std::max(nowl_def_best_score + 0.01, old_score);
        std::cerr << "[protocol_detect] WL-defensive: promoting WL candidate "
                  << wl_def_best_tag << " score " << old_score << "->" << new_score
                  << " over non-WL " << nowl_def_best_tag << "\n";
        best.tag          = wl_def_best_tag;
        best.protocol_id  = wl_def_best_pid;
        best.r1_length    = wl_def_best_r1len;
        best.bc_offset    = wl_def_best_bcoff;
        best.bc_length    = wl_def_best_bclen;
        best.umi_offset   = wl_def_best_umioff;
        best.umi_length   = wl_def_best_umilen;
        best.wl_match_rate = wl_def_best_rate;
        best.score        = new_score;
        if (new_score >= 0.85)      best.confidence = Confidence::HIGH;
        else if (new_score >= 0.60) best.confidence = Confidence::MEDIUM;
        else if (new_score >= 0.40) best.confidence = Confidence::LOW;
        else                        best.confidence = Confidence::NONE;
    }

    // If detection succeeded through the inverted R2 barcode path, record
    // the swap flag so encoders can physically swap R1↔R2 before writing.
    // The confidence requirement is intentionally absent: the geometry itself
    // (R1>40bp, R2≤34bp, R1>2×R2) is unambiguous — barcode reads are never
    // >34bp in droplet protocols.  A whitelist file that is absent or gives a
    // low match rate should not prevent the swap: AUTOFIX-E2E-A2-READ-SWAP.
    if (inverted)
        best.reads_swapped = true;

    return best;
}

// Check if protocol is a 3' protocol (polyA trimming appropriate)
inline bool is_3prime_protocol(const std::string& tag) {
    return tag.find("3p") != std::string::npos ||
           tag.find("agnostic") != std::string::npos ||
           tag == "dropseq" || tag == "celseq2" || tag == "marsseq2" ||
           tag == "sci-rna-seq3" || tag == "bd-rhapsody" || tag == "splitseq" ||
           tag == "dnbelab-c4" || tag == "seqwell" || tag == "ddseq" ||
           tag == "quartzseq2" || tag == "microwell-seq" || tag == "surecell" ||
           tag == "10x-arc-gex" || tag == "10x-visium" || tag == "cite-seq-gex";
    // strtseq is 5' (not in this list)
    // 10x-atac has no cDNA read — polyA trimming not applicable
}

// ── Geometry-swap predicate ──────────────────────────────────────────────────
// Returns true when the read-length geometry is unambiguously inverted:
// R1 is long cDNA (>50bp) and R2 is short enough to be a barcode+UMI (<=34bp),
// and R1 is more than twice R2.  This pattern is impossible for any known
// droplet protocol's normal orientation (barcode reads are never >34bp).
// AUTOFIX-E2E-A2-READ-SWAP: encoder hard-swap trigger, fires even without WL.
inline bool should_hard_geometry_swap(uint16_t r1_len, uint16_t r2_len) noexcept {
    return r1_len > 50 && r2_len > 0 && r2_len <= 34 &&
           r1_len > 2u * static_cast<uint32_t>(r2_len);
}

// ══════════════════════════════════════════════════════════════════════
// Assay classification — determine modality from read structure signals
// ══════════════════════════════════════════════════════════════════════

// Classify assay type from protocol detection result + read statistics.
// This is called AFTER protocol detection and uses both the detected
// protocol tag and raw read signals to determine the biological assay.
//
// Signals used:
//   1. Known protocol tag (definitive for most sc-RNA protocols)
//   2. Read length asymmetry (short R1 + long R2 = UMI-sc; symmetric = bulk/WGS)
//   3. Base composition (bisulfite: extremely low C%; normal: ~25% each)
//   4. Tn5 adapter signature (ATAC / multiome)
//   5. Poly-T/A prevalence (3' RNA-seq)
//   6. Segment count (3-segment with short index ≈ 10x family)
//   7. Read length ranges (e.g. very long single reads ≈ nanopore/pacbio)

template<typename SpotT>
struct AssaySignals {
    // Base composition across all reads
    double gc_fraction    = 0.0;  // (G+C) / total
    double c_fraction     = 0.0;  // C alone (bisulfite indicator)
    double n_fraction     = 0.0;  // N bases (quality indicator)

    // Adapter signatures (fraction of reads containing)
    double tn5_fraction   = 0.0;  // Contain Tn5 ME (ATAC)
    double polyt_fraction = 0.0;  // Contain polyT run ≥10 at expected position
    double adt_fraction   = 0.0;  // Contain CITE-seq capture sequence

    static AssaySignals compute(const std::vector<SpotT>& probe_spots,
                                uint16_t r1_len, uint16_t r2_len) {
        AssaySignals sig;
        const uint32_t n = std::min<uint32_t>(5000,
            static_cast<uint32_t>(probe_spots.size()));
        if (n < 20) return sig;

        // Tn5 mosaic end: first 10bp of the canonical 19bp ME
        static const uint8_t TN5_ME[] = {0,2,0,3,0,2,0,0,2,0}; // AGATAGAAGA→A=0,G=2
        // Actually: AGATGTGTATAAGAGACAG → recode properly
        static const uint8_t TN5_ME_FULL[] = {0,2,0,3,2,3,2,3,0,3,0,0,2,0,2,0,1,0,2};
        constexpr uint16_t TN5_LEN = 19;
        constexpr uint16_t TN5_CHECK = 12;  // check first 12bp (enough for signal)

        // CITE-seq TotalSeq capture sequence (first 10bp)
        static const uint8_t ADT_CAP[] = {1,1,3,3,2,2,1,0,1,1}; // CCTTGGCACC
        constexpr uint16_t ADT_LEN = 10;

        uint64_t base_counts[5] = {0};  // A, C, G, T, N
        uint32_t tn5_hits = 0, polyt_hits = 0, adt_hits = 0;

        for (uint32_t i = 0; i < n; ++i) {
            const auto& s = probe_spots[i];

            // Count bases in R2 (cDNA read for UMI-sc, or both for bulk)
            const auto& seq = (s.r2_len > 0) ? s.r2_seq : s.r1_seq;
            uint16_t slen = (s.r2_len > 0) ? s.r2_len : s.r1_len;
            for (uint16_t j = 0; j < slen; ++j) {
                uint8_t b = (j < seq.size()) ? seq[j] : 4;
                if (b < 5) base_counts[b]++;
            }

            // Check for Tn5 adapter in R2 (ATAC signal)
            if (slen >= TN5_CHECK) {
                // Search near end of read (Tn5 adapter appears at fragment boundary)
                uint16_t search_start = (slen > 80) ? slen - 80 : 0;
                for (uint16_t p = search_start; p + TN5_CHECK <= slen; ++p) {
                    uint16_t mm = 0;
                    for (uint16_t k = 0; k < TN5_CHECK && mm < 3; ++k) {
                        if (p + k < seq.size() && seq[p + k] != TN5_ME_FULL[k]) mm++;
                    }
                    if (mm < 3) { tn5_hits++; break; }
                }
            }

            // Check for polyT in R2 (3' RNA signal) — first 20bp
            if (s.r2_len >= 15) {
                uint16_t t_run = 0, max_t = 0;
                uint16_t check_len = std::min<uint16_t>(s.r2_len, 20);
                for (uint16_t j = 0; j < check_len && j < s.r2_seq.size(); ++j) {
                    if (s.r2_seq[j] == 3) { t_run++; if (t_run > max_t) max_t = t_run; }
                    else t_run = 0;
                }
                if (max_t >= 8) polyt_hits++;
            }

            // Check for ADT capture sequence in short R2 (CITE-seq ADT)
            if (s.r2_len > 0 && s.r2_len <= 60 && s.r2_len >= ADT_LEN) {
                for (uint16_t p = 0; p + ADT_LEN <= s.r2_len && p < 30; ++p) {
                    uint16_t mm = 0;
                    for (uint16_t k = 0; k < ADT_LEN && mm < 2; ++k) {
                        if (p + k < s.r2_seq.size() && s.r2_seq[p + k] != ADT_CAP[k]) mm++;
                    }
                    if (mm < 2) { adt_hits++; break; }
                }
            }
        }

        uint64_t total_bases = base_counts[0] + base_counts[1] +
                               base_counts[2] + base_counts[3] + base_counts[4];
        if (total_bases > 0) {
            sig.gc_fraction = static_cast<double>(base_counts[1] + base_counts[2]) / total_bases;
            sig.c_fraction  = static_cast<double>(base_counts[1]) / total_bases;
            sig.n_fraction  = static_cast<double>(base_counts[4]) / total_bases;
        }
        sig.tn5_fraction   = static_cast<double>(tn5_hits)   / n;
        sig.polyt_fraction = static_cast<double>(polyt_hits) / n;
        sig.adt_fraction   = static_cast<double>(adt_hits)   / n;

        return sig;
    }
};

// Classify the assay type from detected protocol + read signals.
// Priority: explicit protocol tag > statistical signals > default UNKNOWN.

template<typename SpotT>
inline AssayType classify_assay(
        const std::string& protocol_tag,
        Confidence confidence,
        const std::vector<SpotT>& probe_spots,
        uint16_t r1_len,
        uint16_t r2_len) {

    // ── Rule 1: Known protocol tag → definitive assay type ──
    if (confidence >= Confidence::LOW) {
        // 3' UMI-sc protocols
        if (protocol_tag.find("3p") != std::string::npos ||
            protocol_tag == "dropseq" || protocol_tag == "celseq2" ||
            protocol_tag == "marsseq2" || protocol_tag == "sci-rna-seq3" ||
            protocol_tag == "bd-rhapsody" || protocol_tag == "splitseq" ||
            protocol_tag == "indrop" || protocol_tag == "dnbelab-c4" ||
            protocol_tag == "seqwell" || protocol_tag == "ddseq" ||
            protocol_tag == "quartzseq2" || protocol_tag == "microwell-seq" ||
            protocol_tag == "surecell")
            return AssayType::SC_RNA_3PRIME;

        // 5' UMI-sc protocols
        if (protocol_tag.find("5p") != std::string::npos ||
            protocol_tag == "strtseq")
            return AssayType::SC_RNA_5PRIME;

        // 10x Multiome GEX (ARC)
        if (protocol_tag == "10x-arc-gex")
            return AssayType::SC_MULTIOME_GEX;

        // scATAC — barcode in I2 index, R1/R2 are genomic DNA
        if (protocol_tag == "10x-atac")
            return AssayType::SC_ATAC;

        // Visium spatial transcriptomics
        if (protocol_tag == "10x-visium")
            return AssayType::SPATIAL_RNA;

        // CITE-seq GEX library
        if (protocol_tag == "cite-seq-gex")
            return AssayType::CITE_SEQ_GEX;

        // Agnostic-detected: use polyT signal to distinguish 3'/5'
        if (protocol_tag.find("agnostic") != std::string::npos)
            return AssayType::SC_RNA_3PRIME;  // default; polyT → 3'
    }

    // ── Rule 2: Statistical signals for undetected protocols ──
    auto sig = AssaySignals<SpotT>::compute(probe_spots, r1_len, r2_len);

    // Non-sequence data: >20% N bases
    if (sig.n_fraction > 0.20)
        return AssayType::NOT_SEQUENCE;

    // Bisulfite / methylome: C content < 8% (normal is ~25%)
    // Bisulfite conversion replaces unmethylated C with T, so C drops dramatically
    if (sig.c_fraction < 0.08 && sig.c_fraction > 0.001)
        return AssayType::SC_METHYLOME;

    // ATAC: Tn5 adapter in >15% of reads
    if (sig.tn5_fraction > 0.15) {
        // Could be ATAC or multiome-ATAC — without paired GEX we call it ATAC
        return AssayType::SC_ATAC;
    }

    // CITE-seq ADT: high ADT capture sequence rate + short R2
    if (sig.adt_fraction > 0.30 && r2_len > 0 && r2_len <= 60)
        return AssayType::CITE_SEQ_ADT;

    // Plate-based full-length scRNA: no barcode detected, symmetric long reads
    // (Smart-seq2 typically 100–150bp PE with no barcode structure)
    if (confidence < Confidence::LOW && r1_len > 50 && r2_len > 50) {
        double ratio = static_cast<double>(std::min(r1_len, r2_len)) /
                       std::max(r1_len, r2_len);
        // Symmetric reads (ratio > 0.6) with no detectable barcode
        if (ratio > 0.6) {
            // Check for significant polyT — indicates RNA-seq vs WGS
            if (sig.polyt_fraction > 0.05)
                return AssayType::BULK_RNA;

            // Symmetric long reads with no barcode, no polyT, high GC uniformity
            // Could be WGS, WES, or plate-based scRNA (per-well Smart-seq2).
            // We can't distinguish WGS from Smart-seq2 here — both are
            // long PE reads with no library structure. Return SC_RNA_PLATE
            // if GC looks like transcriptome (40-60%), otherwise BULK_RNA.
            if (sig.gc_fraction > 0.35 && sig.gc_fraction < 0.65)
                return AssayType::SC_RNA_PLATE;
            else
                return AssayType::BULK_RNA;
        }
    }

    // Single-end short reads with no barcode: likely bulk
    if (confidence < Confidence::LOW && r2_len == 0 && r1_len > 50)
        return AssayType::BULK_RNA;

    return AssayType::UNKNOWN;
}

// Map a forced protocol tag to its AssayType (Rule 1 only, no statistical probing).
// Used by FastqEncoder when the protocol is manually specified.
inline AssayType protocol_tag_to_assay_type(const std::string& tag) {
    if (tag.find("3p") != std::string::npos ||
        tag == "dropseq" || tag == "celseq2" || tag == "marsseq2" ||
        tag == "sci-rna-seq3" || tag == "bd-rhapsody" || tag == "splitseq" ||
        tag == "indrop" || tag == "dnbelab-c4" || tag == "seqwell" ||
        tag == "ddseq" || tag == "quartzseq2" || tag == "microwell-seq" ||
        tag == "surecell")
        return AssayType::SC_RNA_3PRIME;
    if (tag.find("5p") != std::string::npos || tag == "strtseq")
        return AssayType::SC_RNA_5PRIME;
    if (tag == "10x-arc-gex")  return AssayType::SC_MULTIOME_GEX;
    if (tag == "10x-atac")     return AssayType::SC_ATAC;
    if (tag == "10x-visium")   return AssayType::SPATIAL_RNA;
    if (tag == "cite-seq-gex") return AssayType::CITE_SEQ_GEX;
    if (tag == "bulk-rna" || tag == "bulk" || tag == "bulk-rnaseq")
        return AssayType::BULK_RNA;
    return AssayType::UNKNOWN;
}

// String representation for metadata / logging
inline const char* assay_type_name(AssayType at) {
    switch (at) {
        case AssayType::SC_RNA_3PRIME:    return "sc-rna-3prime";
        case AssayType::SC_RNA_5PRIME:    return "sc-rna-5prime";
        case AssayType::SC_RNA_FULL:      return "sc-rna-full-length";
        case AssayType::SC_ATAC:          return "sc-atac";
        case AssayType::SC_MULTIOME_GEX:  return "sc-multiome-gex";
        case AssayType::SC_MULTIOME_ATAC: return "sc-multiome-atac";
        case AssayType::SPATIAL_RNA:      return "spatial-rna";
        case AssayType::CITE_SEQ_GEX:     return "cite-seq-gex";
        case AssayType::CITE_SEQ_ADT:     return "cite-seq-adt";
        case AssayType::SC_METHYLOME:     return "sc-methylome";
        case AssayType::BULK_RNA:         return "bulk-rna";
        case AssayType::WGS:             return "wgs";
        case AssayType::WES:             return "wes";
        case AssayType::AMPLICON:        return "amplicon";
        case AssayType::SC_RNA_PLATE:    return "sc-rna-plate";
        case AssayType::NOT_SEQUENCE:    return "not-sequence";
        case AssayType::MULTI:           return "multi";
        default:                         return "unknown";
    }
}

// EFO/OBI ontology term for each assay type.
// Returns {"EFO:XXXXXXX", "human-readable label"} for the broadest matching
// EFO term. Protocol-specific terms (e.g. 10x v3 vs v2) appear in the
// protocol_uri. See https://www.ebi.ac.uk/ols4/ontologies/efo for term IDs.
struct EfoTerm {
    const char* id;     // e.g. "EFO:0008913"
    const char* label;  // e.g. "single-cell RNA sequencing"
};

inline EfoTerm assay_efo_term(AssayType at) {
    switch (at) {
        case AssayType::SC_RNA_3PRIME:
            return {"EFO:0008913", "single-cell RNA sequencing"};
        case AssayType::SC_RNA_5PRIME:
            return {"EFO:0008913", "single-cell RNA sequencing"};
        case AssayType::SC_RNA_FULL:
            return {"EFO:0008931", "Smart-seq2"};
        case AssayType::SC_RNA_PLATE:
            return {"EFO:0008931", "Smart-seq2"};
        case AssayType::SC_ATAC:
            return {"EFO:0010891", "single-cell ATAC sequencing"};
        case AssayType::SC_MULTIOME_GEX:
            return {"EFO:0030059", "10x Multiome"};
        case AssayType::SC_MULTIOME_ATAC:
            return {"EFO:0030059", "10x Multiome"};
        case AssayType::SPATIAL_RNA:
            return {"EFO:0010961", "Visium Spatial Gene Expression"};
        case AssayType::CITE_SEQ_GEX:
            return {"EFO:0009294", "CITE-seq"};
        case AssayType::CITE_SEQ_ADT:
            return {"EFO:0009294", "CITE-seq"};
        case AssayType::SC_METHYLOME:
            return {"EFO:0009640", "single-cell bisulfite sequencing"};
        case AssayType::BULK_RNA:
            return {"EFO:0002770", "RNA-seq"};
        case AssayType::WGS:
            return {"EFO:0002697", "whole genome sequencing"};
        case AssayType::WES:
            return {"EFO:0005396", "whole exome sequencing"};
        case AssayType::AMPLICON:
            return {"OBI:0002117", "amplicon sequencing assay"};
        default:
            return {"", ""};
    }
}

// Canonical protocol URI — resolves to a machine-readable spec document.
// Format: https://protocols.singlet.bio/<tag>
// Enables downstream tools to look up exact barcode positions, WL sources,
// and processing instructions without embedding them in the file itself.
inline std::string protocol_uri(const std::string& tag) {
    if (tag.empty() || tag == "UNKNOWN" || tag.find("agnostic") != std::string::npos)
        return "";
    return "https://protocols.singlet.bio/" + tag;
}

// ══════════════════════════════════════════════════════════════════════
// Data validity guards — detect corrupt, empty, or non-sequence data
// ══════════════════════════════════════════════════════════════════════

// Validation result from probe data sanity checks.
struct DataValidation {
    bool   valid           = true;   // false → abort encoding
    bool   suspect         = false;  // true → encode but flag
    double n_fraction      = 0.0;    // N-base fraction
    double zero_len_rate   = 0.0;    // fraction of spots with zero-length reads
    double mono_rate       = 0.0;    // fraction of reads that are mononucleotide
    std::string reason;              // human-readable rejection/warning reason
};

template<typename SpotT>
inline DataValidation validate_probe_data(
        const std::vector<SpotT>& probe_spots,
        uint64_t total_spots) {

    DataValidation dv;
    const uint32_t n = static_cast<uint32_t>(probe_spots.size());

    // No data at all
    if (n == 0 || total_spots == 0) {
        dv.valid = false;
        dv.reason = "no readable spots in SRA accession";
        return dv;
    }

    // Very few usable spots vs claimed total
    if (n < 20 && total_spots > 1000) {
        dv.valid = false;
        dv.reason = "only " + std::to_string(n) +
                    " usable spots out of " + std::to_string(total_spots);
        return dv;
    }

    // Count problems
    uint32_t zero_reads = 0, mono_reads = 0;
    uint64_t total_bases = 0, n_bases = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const auto& s = probe_spots[i];
        if (s.r1_len == 0 && s.r2_len == 0) { zero_reads++; continue; }

        // Check R2 (or R1 if no R2) for N content and mononucleotide
        const auto& seq = (s.r2_len > 0) ? s.r2_seq : s.r1_seq;
        uint16_t slen = (s.r2_len > 0) ? s.r2_len : s.r1_len;
        uint32_t counts[5] = {0};
        for (uint16_t j = 0; j < slen && j < seq.size(); ++j) {
            uint8_t b = seq[j];
            if (b < 5) counts[b]++;
        }
        total_bases += slen;
        n_bases += counts[4];

        // Mononucleotide: one base > 90% of read
        uint32_t max_count = *std::max_element(counts, counts + 4);
        if (slen >= 20 && max_count > slen * 9 / 10) mono_reads++;
    }

    dv.zero_len_rate = static_cast<double>(zero_reads) / n;
    dv.n_fraction = (total_bases > 0) ? static_cast<double>(n_bases) / total_bases : 1.0;
    dv.mono_rate = static_cast<double>(mono_reads) / n;

    // Hard failures
    if (dv.zero_len_rate > 0.90) {
        dv.valid = false;
        dv.reason = "90%+ of spots have zero-length reads";
        return dv;
    }
    if (dv.n_fraction > 0.50) {
        dv.valid = false;
        dv.reason = "majority N-bases (" +
                    std::to_string(static_cast<int>(dv.n_fraction * 100)) +
                    "%) — likely not sequence data";
        return dv;
    }
    if (dv.mono_rate > 0.80) {
        dv.valid = false;
        dv.reason = "80%+ mononucleotide reads — adapter/spike-in only";
        return dv;
    }

    // Soft warnings (encode but flag as suspect)
    if (dv.n_fraction > 0.10 || dv.mono_rate > 0.30 || dv.zero_len_rate > 0.20) {
        dv.suspect = true;
        dv.reason = "elevated quality issues (N=" +
                    std::to_string(static_cast<int>(dv.n_fraction * 100)) +
                    "%, mono=" +
                    std::to_string(static_cast<int>(dv.mono_rate * 100)) +
                    "%, zero=" +
                    std::to_string(static_cast<int>(dv.zero_len_rate * 100)) + "%)";
    }

    return dv;
}

// Wire detected protocol into WriterConfig (BC dict + polyA)
inline void apply_protocol_to_writer(
        WriterConfig& wcfg,
        const ProtocolCandidate& proto,
        const std::vector<std::string>& whitelist_dirs) {

    // BUGFIX (AUTOFIX-VDB-R2-VARIABLE-EMPTY): allow BC dict loading when
    // confidence is LOW (e.g. from metadata_protocol override) provided the
    // protocol spec has a known bc_length.  NONE (0) still gates loading.
    if (proto.confidence == Confidence::NONE || proto.bc_length == 0)
        return;

    // Multi-segment combinatorial protocols (SPLiT-seq, BD Rhapsody, inDrop, etc.)
    // store barcode segments at non-contiguous positions across the full R1 read.
    // BC dict encoding captures ONLY segment-0 and reconstructs R1 as:
    //   [seg0_BC (bc_length bytes)] [UMI (umi_length bytes)] [zeros]
    // This discards all segment-1..N data, leaving the FIFO rewriter with only
    // zeros at segment-1..N positions → all downstream barcodes are wrong.
    // For these protocols the full R1 must be preserved verbatim; skip BC dict.
    static const std::unordered_set<std::string> multi_seg_no_dict = {
        "bd-rhapsody", "splitseq", "indrop", "ddseq",
        "microwell-seq", "surecell"
    };
    if (multi_seg_no_dict.count(proto.tag) > 0)
        return;

    // Load whitelist as byte-numeric sequences for BC dict
    for (const auto& dir : whitelist_dirs) {
        std::string wl_file;
        for (const auto& k : known_protocols()) {
            if (k.tag == proto.tag) {
                wl_file = k.whitelist_file;
                break;
            }
        }
        if (wl_file.empty()) break;
        std::string path = dir + "/" + wl_file;
        auto seqs = load_whitelist_sequences(path, proto.bc_length);
        if (!seqs.empty()) {
            wcfg.bc_dict = std::move(seqs);
            wcfg.bc_offset = proto.bc_offset;
            wcfg.bc_length = proto.bc_length;
            wcfg.umi_offset = proto.umi_offset;
            wcfg.umi_length = proto.umi_length;
            std::cerr << "[1fq-encode] BC dict: " << wcfg.bc_dict.size()
                      << " barcodes, BC=" << wcfg.bc_length
                      << "bp, UMI=" << wcfg.umi_length << "bp\n";
            break;
        }
    }

    // Enable polyA trimming for 3' protocols
    if (is_3prime_protocol(proto.tag)) {
        wcfg.polya_trim = true;
        wcfg.polya_min_len = 10;
        wcfg.polya_max_mm = 1;
        std::cerr << "[1fq-encode] PolyA trimming enabled (3' protocol)\n";
    }
}

// ── Per-segment whitelist resolution for CB_UMI_Complex protocols ──
//
// Returns a vector of resolved absolute paths, one per CB segment.
// If per_seg_whitelist_files is populated for the protocol, each entry is resolved
// from bin_dir/../whitelists/.  If the per_seg list is empty but whitelist_file is
// set, whitelist_file is repeated n_segs times (uniform whitelist across segments).
// Returns an empty vector if no whitelist can be resolved (caller falls back to
// auto-discovered barcodes or CB_samTagOut).
//
// bin_dir: directory containing the singlet binary (from /proc/self/exe).
// protocol_tag: canonical protocol tag (e.g. "bd-rhapsody").
// n_segs: number of CB segments (must match len(soloCBposition)).
inline std::vector<std::string> resolve_per_seg_whitelists(
    const std::string& bin_dir,
    const std::string& protocol_tag,
    std::size_t n_segs)
{
    const CandidateSpec* spec = find_protocol_spec(protocol_tag);
    if (!spec || n_segs == 0) return {};

    // Collect per-segment filenames: prefer per_seg_whitelist_files, then repeat whitelist_file.
    std::vector<std::string> wl_names;
    if (!spec->per_seg_whitelist_files.empty()) {
        wl_names = spec->per_seg_whitelist_files;
        // Pad or truncate to n_segs (should not happen for correct registry entries).
        while (wl_names.size() < n_segs && !wl_names.empty())
            wl_names.push_back(wl_names[0]);
        wl_names.resize(n_segs);
    } else if (!spec->whitelist_file.empty()) {
        wl_names.assign(n_segs, spec->whitelist_file);
    } else {
        return {}; // protocol has no whitelist
    }

    // Resolve each filename to an absolute path.
    const std::string wl_dir = bin_dir + "/../whitelists/";
    std::vector<std::string> resolved;
    resolved.reserve(n_segs);
    for (std::size_t i = 0; i < n_segs; ++i) {
        if (wl_names[i].empty()) { resolved.clear(); return {}; }
        std::string path = wl_dir + wl_names[i];
        struct stat st{};
        if (stat(path.c_str(), &st) != 0) {
            // File missing — report and bail out (caller uses fallback).
            std::cerr << "[singlet] CB_UMI_Complex whitelist missing: " << path
                      << " (seg " << i << ")\n";
            resolved.clear();
            return {};
        }
        resolved.push_back(std::move(path));
    }
    return resolved;
}

// ══════════════════════════════════════════════════════════════════════
// Protocol-agnostic chemistry detection from statistical properties
// ══════════════════════════════════════════════════════════════════════

// Result of agnostic structure detection
struct AgnosticLayout {
    // Which physical read contains barcodes (1=R1, 2=R2, 0=unknown)
    uint8_t     bc_read = 0;
    bool        is_concat = false;  // single-segment concat (BC+polyT+cDNA)

    // Detected barcode region
    uint16_t    bc_offset = 0;
    uint16_t    bc_length = 0;

    // Detected UMI region
    uint16_t    umi_offset = 0;
    uint16_t    umi_length = 0;

    // polyT/polyA boundary (position in BC read, 0 = not found)
    uint16_t    polyt_pos = 0;
    bool        polyt_is_a = false;  // polyA instead of polyT

    // Internal linker (if found)
    uint16_t    linker_offset = 0;
    uint16_t    linker_length = 0;
    std::string linker_consensus;

    // Confidence and diagnostics
    Confidence  confidence = Confidence::NONE;
    double      score = 0.0;
    std::string description;

    // Convert to ProtocolCandidate for downstream use
    ProtocolCandidate to_candidate() const {
        ProtocolCandidate pc;
        pc.tag = "agnostic";
        pc.protocol_id = 255;
        pc.r1_length = 0;
        pc.bc_offset = bc_offset;
        pc.bc_length = bc_length;
        pc.umi_offset = umi_offset;
        pc.umi_length = umi_length;
        pc.wl_match_rate = 0.0;
        pc.score = score;
        pc.confidence = confidence;
        return pc;
    }
};

namespace agnostic_detail {

// Per-position statistics over N reads
struct PosStats {
    uint32_t counts[4] = {0};  // A, C, G, T
    uint32_t n = 0;

    void add(uint8_t base) {
        if (base < 4) { counts[base]++; n++; }
    }

    double entropy() const {
        if (n < 10) return 2.0;  // insufficient data → assume max
        double h = 0.0;
        for (int b = 0; b < 4; ++b) {
            if (counts[b] > 0) {
                double p = static_cast<double>(counts[b]) / n;
                h -= p * std::log2(p);
            }
        }
        return h;
    }

    double max_base_frac() const {
        if (n == 0) return 0.0;
        uint32_t mx = *std::max_element(counts, counts + 4);
        return static_cast<double>(mx) / n;
    }

    uint8_t dominant_base() const {
        return static_cast<uint8_t>(
            std::max_element(counts, counts + 4) - counts);
    }

    double base_frac(uint8_t b) const {
        return n > 0 ? static_cast<double>(counts[b]) / n : 0.0;
    }
};

// Compute prefix repeat rate: fraction of reads sharing exact prefix of length k
inline double prefix_repeat_rate(
        const std::vector<std::vector<uint8_t>>& reads, uint16_t k) {
    if (reads.empty() || k == 0) return 0.0;

    // Hash each prefix
    std::unordered_map<uint64_t, uint32_t> prefix_counts;
    uint32_t valid = 0;
    for (const auto& r : reads) {
        if (r.size() < k) continue;
        uint64_t h = 0;
        bool ok = true;
        for (uint16_t i = 0; i < k; ++i) {
            if (r[i] >= 4) { ok = false; break; }
            h = h * 5 + r[i];
        }
        if (ok) { prefix_counts[h]++; valid++; }
    }
    if (valid < 10) return 0.0;

    // Count reads that share their prefix with at least one other read
    uint32_t repeated = 0;
    for (const auto& [h, cnt] : prefix_counts) {
        if (cnt > 1) repeated += cnt;
    }
    return static_cast<double>(repeated) / valid;
}

// Compute k-mer diversity at a window position
inline double kmer_diversity(
        const std::vector<std::vector<uint8_t>>& reads,
        uint16_t start, uint16_t k = 8) {
    std::unordered_set<uint64_t> unique_kmers;
    uint32_t total = 0;

    for (const auto& r : reads) {
        if (r.size() < start + k) continue;
        uint64_t h = 0;
        bool ok = true;
        for (uint16_t i = 0; i < k; ++i) {
            if (r[start + i] >= 4) { ok = false; break; }
            h = h * 5 + r[start + i];
        }
        if (ok) { unique_kmers.insert(h); total++; }
    }
    return total > 0 ? static_cast<double>(unique_kmers.size()) / total : 1.0;
}

// Find the "cliff" in prefix repeat rate → indicates barcode length
// Returns the k value where repeat rate drops most sharply
struct RepeatCliff {
    uint16_t bc_length = 0;     // estimated barcode length
    double   max_drop = 0.0;    // magnitude of largest drop
    double   pre_rate = 0.0;    // repeat rate before cliff
    double   post_rate = 0.0;   // repeat rate after cliff
};

inline RepeatCliff find_prefix_cliff(
        const std::vector<std::vector<uint8_t>>& reads,
        uint16_t max_k = 30) {
    RepeatCliff cliff;

    // Compute repeat rates at k = 4, 6, 8, ..., max_k
    std::vector<std::pair<uint16_t, double>> rates;
    for (uint16_t k = 4; k <= max_k; k += 2) {
        double r = prefix_repeat_rate(reads, k);
        rates.push_back({k, r});
    }

    // Use RELATIVE drop: (pre - post) / pre
    // This correctly handles the general combinatorial decay at short k
    // where absolute drops are large but relative drops are modest.
    // True barcode cliffs show >60% relative drop.
    double best_rel_drop = 0.0;
    for (size_t i = 1; i < rates.size(); ++i) {
        double pre = rates[i-1].second;
        double post = rates[i].second;
        if (pre < 0.003) continue;  // skip noise floor
        double rel_drop = (pre - post) / pre;
        if (rel_drop > best_rel_drop) {
            best_rel_drop = rel_drop;
            cliff.bc_length = rates[i-1].first;
            cliff.max_drop = pre - post;
            cliff.pre_rate = pre;
            cliff.post_rate = post;
        }
    }

    // Require meaningful relative drop (>40%) to accept
    if (best_rel_drop < 0.40) {
        cliff.bc_length = 0;
        cliff.max_drop = 0.0;
        return cliff;
    }

    // Refine: check odd k values around the cliff for sharper boundary
    if (cliff.bc_length > 0) {
        uint16_t lo = cliff.bc_length > 2 ? cliff.bc_length - 1 : cliff.bc_length;
        uint16_t hi = cliff.bc_length + 3;
        double best_drop = cliff.max_drop;
        double best_rel = best_rel_drop;
        uint16_t best_k = cliff.bc_length;

        for (uint16_t k = lo; k <= hi; ++k) {
            double r_k = prefix_repeat_rate(reads, k);
            double r_k2 = prefix_repeat_rate(reads, k + 2);
            if (r_k < 0.003) continue;
            double rel = (r_k - r_k2) / r_k;
            if (rel > best_rel) {
                best_rel = rel;
                best_drop = r_k - r_k2;
                best_k = k;
            }
        }
        cliff.bc_length = best_k;
        cliff.max_drop = best_drop;
        cliff.pre_rate = prefix_repeat_rate(reads, best_k);
        cliff.post_rate = prefix_repeat_rate(reads, best_k + 2);
    }

    return cliff;
}

// Detect polyT or polyA run in entropy profile
// Returns start position of the run, or 0xFFFF if not found
struct PolyRun {
    uint16_t start = 0xFFFF;
    uint16_t length = 0;
    bool     is_polyA = false;  // true = polyA, false = polyT
};

inline PolyRun find_poly_run(
        const std::vector<PosStats>& stats, uint16_t from = 0) {
    PolyRun best;
    uint16_t run_start = 0xFFFF;
    uint16_t run_len = 0;
    bool run_is_a = false;

    for (uint16_t i = from; i < stats.size(); ++i) {
        double t_frac = stats[i].base_frac(3);  // T
        double a_frac = stats[i].base_frac(0);  // A

        bool is_poly = false;
        bool is_a = false;
        if (t_frac > 0.75 && stats[i].n >= 50) { is_poly = true; is_a = false; }
        if (a_frac > 0.75 && stats[i].n >= 50) { is_poly = true; is_a = true; }

        if (is_poly) {
            if (run_start == 0xFFFF) {
                run_start = i;
                run_is_a = is_a;
            }
            run_len++;
        } else {
            if (run_len >= 3 && run_len > best.length) {
                best.start = run_start;
                best.length = run_len;
                best.is_polyA = run_is_a;
            }
            run_start = 0xFFFF;
            run_len = 0;
        }
    }
    // Check final run
    if (run_len >= 3 && run_len > best.length) {
        best.start = run_start;
        best.length = run_len;
        best.is_polyA = run_is_a;
    }
    return best;
}

// Detect fixed-sequence (linker) region: consecutive positions with entropy < threshold
struct FixedRegion {
    uint16_t    start = 0xFFFF;
    uint16_t    length = 0;
    std::string consensus;
};

inline std::vector<FixedRegion> find_fixed_regions(
        const std::vector<PosStats>& stats,
        double entropy_thresh = 0.5,
        uint16_t min_len = 2) {
    std::vector<FixedRegion> regions;
    uint16_t run_start = 0xFFFF;
    uint16_t run_len = 0;

    for (uint16_t i = 0; i < stats.size(); ++i) {
        if (stats[i].n < 50) break;  // not enough data
        double ent = stats[i].entropy();
        // Include positions that are near-fixed OR binary (entropy ~1.0 with 2 bases)
        if (ent < entropy_thresh) {
            if (run_start == 0xFFFF) run_start = i;
            run_len++;
        } else {
            if (run_len >= min_len) {
                FixedRegion fr;
                fr.start = run_start;
                fr.length = run_len;
                // Build consensus
                static const char BASES[] = "ACGT";
                for (uint16_t j = run_start; j < run_start + run_len; ++j) {
                    fr.consensus += BASES[stats[j].dominant_base()];
                }
                regions.push_back(std::move(fr));
            }
            run_start = 0xFFFF;
            run_len = 0;
        }
    }
    if (run_len >= min_len) {
        FixedRegion fr;
        fr.start = run_start;
        fr.length = run_len;
        static const char BASES[] = "ACGT";
        for (uint16_t j = run_start; j < run_start + run_len; ++j) {
            fr.consensus += BASES[stats[j].dominant_base()];
        }
        regions.push_back(std::move(fr));
    }
    return regions;
}

// Find BC→UMI boundary using k-mer diversity transition
// Looks for the position where diversity jumps from BC-level to UMI-level
inline uint16_t find_diversity_jump(
        const std::vector<std::vector<uint8_t>>& reads,
        uint16_t search_from = 0,
        uint16_t search_to = 30,
        uint16_t k = 8) {

    // Compute diversity at each position
    std::vector<std::pair<uint16_t, double>> divs;
    for (uint16_t p = search_from; p + k <= search_to && p + k <= 80; ++p) {
        double d = kmer_diversity(reads, p, k);
        divs.push_back({p, d});
    }

    // Find largest upward jump
    double max_jump = 0.0;
    uint16_t jump_pos = 0;
    for (size_t i = 1; i < divs.size(); ++i) {
        double jump = divs[i].second - divs[i-1].second;
        if (jump > max_jump) {
            max_jump = jump;
            jump_pos = divs[i].first;
        }
    }

    // The actual BC→UMI boundary is at (jump_pos + k - 1) since
    // the 8-mer at jump_pos spans [jump_pos, jump_pos+k)
    // The first 8-mer that's mostly UMI starts at ~ BC_length - k + 1
    // So boundary ≈ jump_pos + k - 1
    if (max_jump > 0.05) return jump_pos + k - 1;
    return 0;
}

} // namespace agnostic_detail

// ── Main agnostic detection function ──
// Analyzes raw reads using only statistical properties to identify:
//   - Which read contains barcodes vs cDNA
//   - Barcode position and length
//   - UMI position and length
//   - PolyT/polyA boundaries
//   - Internal linker/adapter sequences
//
// No protocol database or whitelist is consulted.

template<typename SpotT>
AgnosticLayout detect_structure_agnostic(
        const std::vector<SpotT>& probe_spots,
        uint16_t r1_max,
        uint16_t r2_max) {

    using namespace agnostic_detail;
    AgnosticLayout layout;
    const uint32_t n = std::min<uint32_t>(5000,
        static_cast<uint32_t>(probe_spots.size()));
    if (n < 50) {
        layout.description = "insufficient reads";
        return layout;
    }

    // ── Step 0: Gather reads into vectors ──
    const uint16_t MAX_POS = 80;
    std::vector<std::vector<uint8_t>> r1_reads, r2_reads;
    r1_reads.reserve(n);
    r2_reads.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& s = probe_spots[i];
        if (s.r1_len > 0) {
            uint16_t len = std::min<uint16_t>(s.r1_len, MAX_POS);
            r1_reads.emplace_back(s.r1_seq.begin(), s.r1_seq.begin() + len);
        }
        if (s.r2_len > 0) {
            uint16_t len = std::min<uint16_t>(s.r2_len, MAX_POS);
            r2_reads.emplace_back(s.r2_seq.begin(), s.r2_seq.begin() + len);
        }
    }

    bool has_r1 = !r1_reads.empty() && r1_max > 0;
    bool has_r2 = !r2_reads.empty() && r2_max > 0;

    // Single-segment concat mode
    if (has_r1 && !has_r2) {
        layout.is_concat = true;
    } else if (has_r2 && !has_r1) {
        layout.is_concat = true;
        std::swap(r1_reads, r2_reads);
        std::swap(r1_max, r2_max);
        has_r1 = true; has_r2 = false;
    }

    // ── Step 1: Per-position entropy profiles ──
    auto compute_pos_stats = [&](const std::vector<std::vector<uint8_t>>& reads,
                                  uint16_t max_len) {
        uint16_t len = std::min(max_len, MAX_POS);
        std::vector<PosStats> stats(len);
        for (const auto& r : reads) {
            for (uint16_t j = 0; j < len && j < r.size(); ++j) {
                stats[j].add(r[j]);
            }
        }
        return stats;
    };

    std::vector<PosStats> r1_stats, r2_stats;
    if (has_r1) r1_stats = compute_pos_stats(r1_reads, r1_max);
    if (has_r2) r2_stats = compute_pos_stats(r2_reads, r2_max);

    // ── Step 2: Detect polyT/polyA in each read ──
    PolyRun r1_poly, r2_poly;
    if (has_r1) r1_poly = find_poly_run(r1_stats);
    if (has_r2) r2_poly = find_poly_run(r2_stats);

    // ── Step 3: Detect fixed/linker regions ──
    std::vector<FixedRegion> r1_fixed, r2_fixed;
    if (has_r1) r1_fixed = find_fixed_regions(r1_stats);
    if (has_r2) r2_fixed = find_fixed_regions(r2_stats);

    // ── Step 4: Prefix repeat analysis ──
    RepeatCliff r1_cliff, r2_cliff;
    uint16_t prefix_max_k = 30;
    if (has_r1) {
        if (r1_max < prefix_max_k) prefix_max_k = r1_max;
        r1_cliff = find_prefix_cliff(r1_reads, prefix_max_k);
    }
    if (has_r2) {
        prefix_max_k = std::min<uint16_t>(30, r2_max);
        r2_cliff = find_prefix_cliff(r2_reads, prefix_max_k);
    }

    // ── Step 5: Determine which read contains barcodes ──
    if (layout.is_concat) {
        // Single segment: barcodes at the start, polyT/polyA marks end of BC+UMI
        layout.bc_read = 1;
    } else if (has_r1 && has_r2) {
        // Determine BC vs cDNA read using multiple signals:
        // 1. Relative cliff magnitude (barcode reads have sharp repeat-rate cliffs)
        // 2. Read length asymmetry (barcode reads are typically shorter)
        // 3. k-mer diversity asymmetry (barcode regions have lower diversity)
        // 4. Fixed/linker regions (barcode reads may have adapters)

        // Relative cliff strength: how sharp is the prefix repeat cliff
        double r1_rel_cliff = (r1_cliff.pre_rate > 0.03) ?
            r1_cliff.max_drop / r1_cliff.pre_rate : 0.0;
        double r2_rel_cliff = (r2_cliff.pre_rate > 0.03) ?
            r2_cliff.max_drop / r2_cliff.pre_rate : 0.0;

        // Initial k-mer diversity
        double r1_div0 = kmer_diversity(r1_reads, 0);
        double r2_div0 = kmer_diversity(r2_reads, 0);

        // Score each read's likelihood of being the BC read
        double r1_bc_score = 0.0, r2_bc_score = 0.0;

        // Validate cliff regions: barcodes have high entropy (>1.0),
        // adapters/TSO have low entropy (<0.5). Only count barcode-like cliffs.
        auto cliff_region_entropy = [](const RepeatCliff& cliff,
                                        const std::vector<PosStats>& stats) {
            if (cliff.bc_length == 0) return 0.0;
            double sum = 0.0;
            uint16_t cnt = 0;
            for (uint16_t p = 0; p < cliff.bc_length && p < stats.size(); ++p) {
                sum += stats[p].entropy();
                cnt++;
            }
            return cnt > 0 ? sum / cnt : 0.0;
        };
        double r1_cliff_ent = cliff_region_entropy(r1_cliff, r1_stats);
        double r2_cliff_ent = cliff_region_entropy(r2_cliff, r2_stats);
        bool r1_cliff_is_bc = r1_cliff_ent > 1.0;  // high entropy = barcode
        bool r2_cliff_is_bc = r2_cliff_ent > 1.0;

        // Strong relative cliff (>50%) → strong BC signal (only if barcode-like)
        if (r1_cliff_is_bc && r1_rel_cliff > 0.50) r1_bc_score += 2.0;
        else if (r1_cliff_is_bc && r1_rel_cliff > 0.30) r1_bc_score += 1.0;
        if (r2_cliff_is_bc && r2_rel_cliff > 0.50) r2_bc_score += 2.0;
        else if (r2_cliff_is_bc && r2_rel_cliff > 0.30) r2_bc_score += 1.0;

        // Lower initial diversity → more BC-like (barcodes share structure)
        if (r1_div0 < r2_div0 - 0.05) r1_bc_score += 1.0;
        if (r2_div0 < r1_div0 - 0.05) r2_bc_score += 1.0;

        // Shorter read → more likely BC (most protocols)
        if (r1_max < r2_max * 0.7) r1_bc_score += 1.5;
        else if (r2_max < r1_max * 0.7) r2_bc_score += 1.5;

        // Short fixed region at start (<8bp) → adapter/linker in BC read
        // Long fixed regions (>10bp) are likely TSO artifacts in cDNA, penalize
        if (!r1_fixed.empty() && r1_fixed[0].start == 0) {
            if (r1_fixed[0].length < 8) r1_bc_score += 1.0;
            else r1_bc_score -= 0.5;  // long fixed = TSO/artifact
        }
        if (!r2_fixed.empty() && r2_fixed[0].start == 0) {
            if (r2_fixed[0].length < 8) r2_bc_score += 1.0;
            else r2_bc_score -= 0.5;
        }

        // Cliff detected at reasonable barcode length (6-24bp), barcode-like
        if (r1_cliff_is_bc && r1_cliff.bc_length >= 6 &&
            r1_cliff.bc_length <= 24) r1_bc_score += 0.5;
        if (r2_cliff_is_bc && r2_cliff.bc_length >= 6 &&
            r2_cliff.bc_length <= 24) r2_bc_score += 0.5;

        if (r1_bc_score > r2_bc_score) {
            layout.bc_read = 1;
        } else if (r2_bc_score > r1_bc_score) {
            layout.bc_read = 2;
        } else {
            // Tie: use shorter read
            layout.bc_read = (r1_max <= r2_max) ? 1 : 2;
        }
    }

    if (layout.bc_read == 0) {
        layout.description = "cannot determine BC read";
        return layout;
    }

    // Get references to the BC-read data
    auto& bc_reads = (layout.bc_read == 1) ? r1_reads : r2_reads;
    auto& bc_stats = (layout.bc_read == 1) ? r1_stats : r2_stats;
    auto& bc_cliff = (layout.bc_read == 1) ? r1_cliff : r2_cliff;
    auto& bc_poly  = (layout.bc_read == 1) ? r1_poly : r2_poly;
    auto& bc_fixed = (layout.bc_read == 1) ? r1_fixed : r2_fixed;
    uint16_t bc_max = (layout.bc_read == 1) ? r1_max : r2_max;

    // ── Step 6: Detect barcode length from prefix cliff ──
    // Validate that the cliff region has barcode-like entropy (>1.0),
    // not adapter/TSO-like fixed sequence
    uint16_t bc_end = 0;
    if (bc_cliff.max_drop > 0.05 && bc_cliff.bc_length >= 6 &&
        bc_cliff.bc_length <= 30) {
        double cliff_ent = 0.0;
        uint16_t cnt = 0;
        for (uint16_t p = 0; p < bc_cliff.bc_length && p < bc_stats.size(); ++p) {
            cliff_ent += bc_stats[p].entropy();
            cnt++;
        }
        if (cnt > 0) cliff_ent /= cnt;
        if (cliff_ent > 1.0) {
            bc_end = bc_cliff.bc_length;
        }
    }

    // ── Step 7: Identify the start of the barcode ──
    // Check for fixed/adapter region at the beginning of the BC read
    uint16_t bc_start = 0;
    if (!bc_fixed.empty() && bc_fixed[0].start == 0) {
        // Fixed adapter at start — barcode begins after it
        bc_start = bc_fixed[0].start + bc_fixed[0].length;
    }

    // ── Step 8: Refine with polyT/polyA and diversity ──
    if (bc_poly.start != 0xFFFF && bc_poly.start > bc_start) {
        // polyT/polyA found — marks end of BC+UMI region
        // If we have a cliff estimate, the region between cliff and polyT is UMI
        if (bc_end > 0 && bc_end < bc_poly.start) {
            layout.bc_offset = bc_start;
            layout.bc_length = bc_end - bc_start;
            layout.umi_offset = bc_end;
            layout.umi_length = bc_poly.start - bc_end;
            layout.polyt_pos = bc_poly.start;
            layout.polyt_is_a = bc_poly.is_polyA;
        } else {
            // No cliff or cliff past polyT — use diversity to segment
            uint16_t div_boundary = find_diversity_jump(
                bc_reads, bc_start,
                std::min<uint16_t>(bc_poly.start, MAX_POS));
            if (div_boundary > bc_start && div_boundary < bc_poly.start) {
                layout.bc_offset = bc_start;
                layout.bc_length = div_boundary - bc_start;
                layout.umi_offset = div_boundary;
                layout.umi_length = bc_poly.start - div_boundary;
            } else {
                // Can't segment — treat entire pre-polyT region as BC+UMI
                // Use common split: first 60% BC, last 40% UMI
                uint16_t total = bc_poly.start - bc_start;
                layout.bc_offset = bc_start;
                layout.bc_length = (total * 3) / 5;
                layout.umi_offset = bc_start + layout.bc_length;
                layout.umi_length = total - layout.bc_length;
            }
            layout.polyt_pos = bc_poly.start;
            layout.polyt_is_a = bc_poly.is_polyA;
        }
    } else if (bc_end > 0) {
        // No polyT found, but we have a cliff estimate for BC length
        layout.bc_offset = bc_start;
        layout.bc_length = bc_end - bc_start;

        // UMI detection: find diversity jump after barcode
        // The region between BC end and either the end of the read or
        // a diversity plateau change is the UMI
        uint16_t search_to = std::min<uint16_t>(bc_max, MAX_POS);
        uint16_t umi_end = 0;

        // Look for polyT/polyA slightly further in the read
        PolyRun late_poly = find_poly_run(bc_stats, bc_end);
        if (late_poly.start != 0xFFFF && late_poly.start > bc_end) {
            umi_end = late_poly.start;
            layout.polyt_pos = late_poly.start;
            layout.polyt_is_a = late_poly.is_polyA;
        } else {
            // No polyT — UMI extends to the end of the BC read
            // Use diversity: UMI region has high diversity (>0.8)
            // Find where diversity dips back down or read ends
            for (uint16_t p = bc_end; p + 8 <= search_to; ++p) {
                double d = kmer_diversity(bc_reads, p);
                // If diversity drops below 0.4 after being high, UMI ended
                if (d < 0.4 && p > bc_end + 2) { umi_end = p + 4; break; }
            }
            if (umi_end == 0) umi_end = search_to;
        }

        layout.umi_offset = bc_end;
        layout.umi_length = std::min<uint16_t>(umi_end - bc_end, 16);
        // Cap UMI at 16bp — UMIs longer than that are extremely rare
    } else {
        // No cliff, no polyT — fall back to diversity-only segmentation
        // This handles cases like symmetric PE reads where prefix repeat
        // doesn't show a clear cliff

        uint16_t search_to = std::min<uint16_t>(bc_max, MAX_POS);
        // Find the region of lowest diversity (BC) vs highest (UMI/cDNA)
        double min_div = 2.0;
        uint16_t min_pos = 0;
        for (uint16_t p = bc_start; p + 8 <= search_to && p < 30; ++p) {
            double d = kmer_diversity(bc_reads, p);
            if (d < min_div) { min_div = d; min_pos = p; }
        }

        if (min_div < 0.5) {
            // Found a low-diversity region → likely BC
            uint16_t div_boundary = find_diversity_jump(
                bc_reads, min_pos, search_to);
            if (div_boundary > bc_start) {
                layout.bc_offset = bc_start;
                layout.bc_length = div_boundary - bc_start;
                layout.umi_offset = div_boundary;
                // Estimate UMI length from remaining read
                uint16_t remaining = search_to - div_boundary;
                layout.umi_length = std::min<uint16_t>(remaining, 12);
            }
        }
    }

    // ── Step 9: Detect internal linkers ──
    // Look for fixed regions WITHIN the variable part of the BC read
    // (not at the very start, which is an adapter)
    for (const auto& fr : bc_fixed) {
        if (fr.start > bc_start && fr.start < layout.umi_offset + layout.umi_length) {
            layout.linker_offset = fr.start;
            layout.linker_length = fr.length;
            layout.linker_consensus = fr.consensus;
            break;
        }
    }

    // ── Step 10: Compute confidence ──
    double conf_score = 0.0;

    // BC length detected?
    if (layout.bc_length >= 6 && layout.bc_length <= 24) conf_score += 0.30;

    // UMI detected?
    if (layout.umi_length >= 4 && layout.umi_length <= 16) conf_score += 0.20;

    // polyT/polyA found? (strong structural signal)
    if (layout.polyt_pos > 0) conf_score += 0.20;

    // Cliff quality (drop magnitude)
    if (bc_cliff.max_drop > 0.30) conf_score += 0.15;
    else if (bc_cliff.max_drop > 0.10) conf_score += 0.10;

    // BC/UMI entropy validation
    if (layout.bc_length > 0 && layout.umi_length > 0) {
        // Check that BC region has moderate entropy (1.0-2.0)
        double bc_ent = 0.0;
        uint16_t bc_ent_n = 0;
        for (uint16_t p = layout.bc_offset;
             p < layout.bc_offset + layout.bc_length && p < bc_stats.size(); ++p) {
            bc_ent += bc_stats[p].entropy();
            bc_ent_n++;
        }
        if (bc_ent_n > 0) bc_ent /= bc_ent_n;

        double umi_ent = 0.0;
        uint16_t umi_ent_n = 0;
        for (uint16_t p = layout.umi_offset;
             p < layout.umi_offset + layout.umi_length && p < bc_stats.size(); ++p) {
            umi_ent += bc_stats[p].entropy();
            umi_ent_n++;
        }
        if (umi_ent_n > 0) umi_ent /= umi_ent_n;

        // Good: both BC and UMI have high entropy, UMI ≥ BC
        if (bc_ent > 1.0 && umi_ent > 1.0) conf_score += 0.15;
    }

    layout.score = conf_score;
    if (conf_score >= 0.80) layout.confidence = Confidence::HIGH;
    else if (conf_score >= 0.55) layout.confidence = Confidence::MEDIUM;
    else if (conf_score >= 0.30) layout.confidence = Confidence::LOW;

    // ── Build description ──
    layout.description = "R" + std::to_string(layout.bc_read) + ": ";
    if (bc_start > 0) layout.description += "adapter(" + std::to_string(bc_start) + "bp)+";
    layout.description += "BC(" + std::to_string(layout.bc_length) + "bp@" +
                         std::to_string(layout.bc_offset) + ")";
    if (layout.linker_length > 0) {
        layout.description += "+linker(" + layout.linker_consensus + "@" +
                             std::to_string(layout.linker_offset) + ")";
    }
    layout.description += "+UMI(" + std::to_string(layout.umi_length) + "bp@" +
                         std::to_string(layout.umi_offset) + ")";
    if (layout.polyt_pos > 0) {
        layout.description += (layout.polyt_is_a ? "+polyA@" : "+polyT@") +
                             std::to_string(layout.polyt_pos);
    }
    if (layout.is_concat) layout.description += " [concat]";

    return layout;
}

} // namespace singlet::fq
