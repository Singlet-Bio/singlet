// Copied unchanged from PR #61 (branch gpu/w11-prototypes, commit f84aafff), bench/gpu/pileup/pileup_rec.h: see bench/gpu/pileup/README.md there.
// Linked into singlet through ../pileup_gpu.cu (PILEUP_NO_MAIN); identity: docs/gpu-engine/exploration.md 12.13, 12.19.
// GPU input records for the W11 pileup stage (shared by extract.cpp and pileup.cu).
//
// The unit is a SIGNATURE: one distinct (read sequence, STAR transcript set). STAR maps a cDNA read from its sequence alone, so every
// read with the same sequence has the same transcripts (the extractor counts violations of this; E0 in the exploration doc).
// M5 will emit the same table directly: per distinct sequence its final transcripts, with the primary flag and NH.
//   AlnRec  one STAR transcript (BAM alignment): chr id, 0-based start, strand, primary flag, MAPQ, and its CIGAR (BAM encoding
//           len << 4 | op; the gap type N / D / I is what STAR's alignBAM derives from canonSJ and the read/genome gaps between blocks).
//   SigRec  the alignments of one signature (arrival order = BAM coordinate order = (chr, start, STAR tie order)), NH, the read length
//           and the offset of the read sequence (original orientation, one BAM nt16 code per nibble, low nibble first).
//   ReadRec one read: signature, barcode column (-1 = CB missing or not in the barcode set), raw UMI (3 bits per character:
//           A0 C1 G2 T3 N4). Reads are in the order of their primary alignment in the sorted BAM (ties: STAR's sort order); that
//           order decides the first read of a (barcode, SNP, UMI) in the SNP pileup.
#pragma once
#include <cstdint>

struct AlnRec { int32_t tid; int32_t pos; uint32_t ops_off; uint16_t n_ops; uint8_t flags; uint8_t mapq; };   // flags: 1 reverse, 2 primary
struct SigRec { uint32_t aln_off; uint8_t n_aln; uint8_t nh; uint16_t rlen; uint64_t seq_off; };
struct ReadRec { uint32_t sig; int32_t bc; uint64_t umi3; };
static_assert(sizeof(AlnRec) == 16 && sizeof(SigRec) == 16 && sizeof(ReadRec) == 16, "record sizes");
