# Staged read format `.1fg` v2 (draft for agreement with S9, 2026-10-30)

Status: DRAFT. Nothing is built. Author: S8 (consumer side: GPU prefill, STAR per-read pass, CPU engine). Producer (fetch tier, CPU nodes): S9.
Basis: `docs/research-beta/ingest-format.md` (the measured `.1fg` v1 prototype: chunked, struct-of-arrays, zstd per section, CRC32C). v2 changes what a
read carries so that the GPU node does no text parsing, no barcode matching and no sequence dedup.

## Why (measured, pbmc1k_v3, 66.6 M reads, `--engine gpu`, host core-s per M reads; docs/research-beta/gpu-host-offload.md)
`.1fq` decode to temp FASTQ 1.3, STAR FASTQ parse + clip + barcode/UMI handling + Solo records in the per-read pass about 4.1, prefill host work (FASTQ text, CR4 clip,
cache inserts) about 3.6. Together 9 of 14 core-s/M. v1 measured the reader at 0.010-0.014 core-s/M.

## Contents of a v2 file
Header (4 KiB): magic `1FG\0`, version 2, protocol id and geometry (CB length/start, UMI length/start, R2 max length), `chunk_reads` (32768), `n_reads`, `n_unique`,
whitelist id + sha256 + size, flags, section offsets, source provenance (accession, bytes, md5), CRC32C of the header.

Read section, in source order, `chunk_reads` per chunk, struct-of-arrays, each plane a zstd frame (libzstd on the CPU, no nvCOMP), CRC32C of the stored bytes:

| plane | per read | meaning |
|---|---|---|
| `cb` | u32 | whitelist index of the corrected barcode (R11 rule: STAR `1MM_multi_Nbase_pseudocounts` with the sample's exact histogram); `0xFFFFFFFF` = none |
| `cbst` | 3 bits (packed) | STAR's barcode status: exact, one candidate, posterior accepted, posterior rejected, no match, two N, bad UMI (the classes of `cb_status_` and `CellReads.stats` `cbMatch`, `cbPerfect`, `cbMMunique`, `cbMMmultiple`) |
| `umi` | u32 (<= 16 nt, 2 bit/base) + `umin` flag in `exc` | UMI word |
| `seq` | u32 | id of the read's cDNA sequence in the unique-sequence table |
| `len` | u8 | cDNA length actually kept (variable-length trimming) |
| `q` | 2 bits/base, `len` bases | ILLUMINA4 binned quality (`.1fq` qual_mode 4, `docs/format-1fq.md` 4.3: Phred 0-9, 10-19, 20-29, 30+ decode to 2, 11, 25, 37). Per read, own plane and zstd frame: it cannot dedup with the sequence. Required for the atlas (the SNP and mtDNA callers and the per-read quality counters use it); absent only for `qual_mode 0` |
| `exc` | sorted u32 list | N bases in the UMI and the cDNA, reads whose barcode raw bases are needed (uncorrectable) |

Unique-sequence table, in blocks of 32768 sequences (index of block offsets in the footer): `seq2` n x slot bytes (2 bit/base, first base high, slot = 8-byte multiple >= max length),
`slen` u8, `sexc` N list. One entry per distinct cDNA sequence **as STAR's cache keys it**: the clipped sequence after any file-level clip, byte-exact, N included. The
GPU aligns each entry once. A read's `seq` points at it. Expected size: about half the reads on deep samples (seqCache hit rate there is about 46%).

Footer: chunk index, block index, CRC32C of the index. Multipart upload parts end on chunk/block boundaries as in v1.

## Consumers (S8)
- GPU prefill: reads the unique-sequence table straight into pinned buffers and H2D; no hash, sort or dedup stage; the CR4 clip kernel runs on the packed codes. Needs `cbst`/`cb`/`umi`/`seq` only for the hand-off entries.
- STAR per-read pass: replaced for hand-off reads by a block reader that fills the per-read entry (`HandoffReadEntry`: CR, UR, quality counters, clip flags) and the Solo record from `cb`, `cbst`, `umi`, without text. Reads that STAR must map itself (host lane) are rebuilt as numeric reads.
- CPU engine: a `lib1fq`-style block source that fills `DecodedBlock` (v1 measured 0.15-0.18 core-s/M).
- `.1fq` stays an accepted input for one release.

## Identity gate
Run `.1fq` -> `.1fg` v2 -> engine against today's `.1fq` path on pbmc1k_v3 and neuron1k_v3: same 60 files byte-identical (only read order, timings and provenance may differ), for the CPU engine and `--engine gpu`.

## Decisions agreed with S9 (2026-10-30)
- No clipping in the fetch tier: the stored cDNA is what `singlet download` writes into a `.1fq` today (after its own polyA/adapter trimming); max 127 nt. The 5' prefix clip, CR4 TSO clip, 3' poly-tail and adapter clips are applied by the engine on the packed codes; distinct stored sequences can collapse after clipping and the GPU dedups again.
- One pooled file per sample with a run table in the footer (`{accession, first_read, n_reads}`), runs concatenated in the order the screen kept; the R11 histogram is pooled over the kept runs.
- Encoder: standalone CMake target `singlet-stage` in `src/stage/` (libzstd + header-only `include/singlet/pileup/cb_correct.h`, histogram rule of `src/pipeline/singlet.cpp` ~4640-4680 and ~4880-4895). S9 writes the encoder, plane writer, manifest and the `.1fq` -> `.1fg2` runs; S8 writes the reference decoder `dec_1fg2`, the readers and the identity harness.
- GPU pileup and qualities: today it declines to run unless qualities are constant or tied to N; shipping q to the GPU is S8's work item and does not change the format.

## Open questions (historical)
1. Barcode correction needs the sample's whole exact-barcode histogram (STAR pass 1) before any read is corrected: the producer needs two passes over the decoded reads (or a histogram pass over the first decode, then the write). Cost on the CPU node? (v1 transform: 0.24-0.42 core-s/M.)
2. Sequence dedup needs a hash table of up to about 35 M distinct 91-nt sequences (about 1.5-2 GB RAM). OK on the fetch node?
3. (answered) Qualities are required: ILLUMINA4, per read.
4. Reads with N in the barcode, uncorrectable barcodes: `cb = 0xFFFFFFFF` with `cbst`, raw bases not stored (STAR counts them in `CBnotInPasslist` only). Agreed?
5. Unique-sequence table scope: per sample (proposed), not per chunk.
