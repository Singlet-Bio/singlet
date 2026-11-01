# E6: alignment records and per-read expansion on the GPU (hops 6 and 7)

Experiment E6 of `singlet-agents/docs/GPU_ALIGNER_STUDY_2026-10-31.md` (Part II, II.8). Branch `gpu/fable/records` (from
`gpu/fable-engine` = 518e4163). New code only: nothing is wired into the singlet binary's default path. The one change to existing
code is a debug dump in `src/engine_gpu/engine_gpu.cu` (`SINGLET_ENGINE_GPU_DUMP`, off unless the variable is set).

## 0. Answer in brief

On the whole pbmc1k_v3 sample (66,601,887 reads, 31,636,557 distinct sequences with a record), from one `singlet --engine gpu`
run (Slurm 795356, H100 NVL, engine binary at ccfe2c28, driver at 94d1f5b4):

| gate | result |
|---|---|
| hand-off records (seqCacheAlignRecords bytes) | **31,636,557 / 31,636,557 byte-identical**; whole 2,246,261,173-byte data block and its offsets identical |
| pileup arrays vs `Collector::prepend_handoff` | SigRec 31,636,557 / 31,636,557, AlnRec 32,045,370 / 32,045,370, CIGAR ops 41,974,619 and nt16 sequence 1,406,636,366 bytes: **all identical** |
| host port of the same code (records.cuh, 18 threads) | identical to the GPU's and the file's |
| mate-gap transcripts (canonSJ -3, where STAR falls back to alignBAM) | **0** counted (also 0 transcripts without exons, 0 records with trBest = trInit) |
| GPU time, 31.6 M records | **37.2 ms** for the pileup arrays (851 M records/s); 65.1 ms with the hand-off bytes as well |
| GPU time, 2 M records | 2.73 ms (strided subset of the same records) |
| hop 7, ReadRec per read | 61,294,166 / 61,294,166 identical to a transcription of `gpu_handoff_pass_entries` with the engine's R11 correction; the count equals the engine's own (`61294166 reads take their alignments from the GPU prefill`); exact-CR mode 60,060,876 / 60,060,876 identical; low-quality masks, outcome counts and quality sums identical in both |
| hop 7 GPU time | 10.5 ms for 66.6 M reads + 5.7 ms for the 897,422 distinct CRs; the CR table on the host 0.17 s (R11, 18 threads) |

Host work this replaces: STAR's hand-off loop (records + summaries) took 5 s on 18 threads in this run (1-s log granularity; the host
port of the record half alone 3.58 s on 18 threads), the 2.25 GB hand-off file, `prepend_handoff`'s parse (2.8 s serial), and
`gpu_handoff_pass_entries` (4 s on the job's cores after PR #107, host-cost-model.md). Not covered: the hand-off summaries STAR replays
for its per-read output, the pass's statistics counters, and the CR/UR/quality entries (which STAR's per-read loop still writes).


## 1. What was built

| file | what |
|---|---|
| `src/engine_gpu/records.cuh` | host + device logic, one source: blob decode (inverse of `stitch.cuh` encodeTr = `ReadAlign::seqCacheEncode`), `mappedFilter`, `seqCacheLeanAln` (single end), the hand-off record bytes of `seqCacheAlignRecords`, the pileup arrays of `Collector::prepend_handoff`, and the per-read rule of `PileupEngine::gpu_handoff_pass_entries` |
| `src/engine_gpu/records_gpu.cu`, `records_gpu.h` | kernels and API: `buildRecords` (hop 6: thread per distinct sequence, count -> scan -> fill), `distinctBarcodes` + `expandReads` (hop 7: thread per read, eval -> scan -> scatter) |
| `src/engine_gpu/engine_gpu.cu` | `SINGLET_ENGINE_GPU_DUMP=<path>`: at the end of `mapDistinct`, mapDistinct's whole output (sequences, blobs, readRecord, clip lengths) plus the filter parameters and `chrStart` (format in `dumpOutput`) |
| `bench/fable/records/records_test.cu` | driver: dump + the hand-off of the same run -> GPU records, byte-for-byte checks, timings, host port timing, hop 7 checks |
| `bench/fable/records/r11_hist.cpp` | the engine's R11 exact CB histogram from the .1fq (as singlet's decode pass counts it), for the driver's R11 CR table |
| `bench/fable/records/jobs/` | `build_singlet.sbatch` (singlet-gpu with the dump), `compile.sbatch` (driver), `run.sbatch` (engine run + driver, then re-runs the driver on a trigger file without re-running the engine; `FQ`, `TAG`, `TMAX` for other samples), `r11_hist.sbatch` |
| `bench/fable/records/results/` | the final driver log (`.txt`) and JSON of job 795356 |

### 1.1 Hop 6: blob -> records (per distinct sequence)

`recordCount` and `recordFill` follow `seqCacheAlignRecords` line by line for what it does on the hand-off path
(`SINGLET_HANDOFF_LEAN` default, single end):

| step | STAR (ReadAlign_seqCache.cpp) | records.cuh |
|---|---|---|
| decode | `seqCacheRestore`: nW, nTr, iBest, (trInit rLength if nW = 0), per transcript nExons + 29 scalars (`forEachScalar`) + per exon 5 fields and, except after the last exon, canonSJ sjAnnot sjStr (+ 2 shiftSJ when canonSJ >= 0) | `parseHead` (reads nExons, rLength, Chr, Str, nMatch, nMM, maxScore, primaryFlag; skips the rest), `skipRows`, `cigarOps` (reads R, G, L, iFrag, canonSJ, sjAnnot) |
| mapped? | `mappedFilter` on trBest = trMult[iBest]; Lread = L; outFilterMismatchNmaxTotal = min(outFilterMismatchNmax, (uint)(NoverReadLmax * L)) | same expressions, same integer and double conversions |
| alignments | min(outSAMmultNmax, nTr) transcripts, `seqCacheLeanAln` each | same |
| CIGAR | S for trimL1 = 0 + R0 - (R0 < L ? 0 : L + 1); per exon I (gapR > 0), N (canonSJ >= 0 or sjAnnot == 1) else D (gapG > 0), M (L > 0); S for trimR1; all in 64-bit `uint`, stored as uint32 `len << 4 \| op` | same (`u64` arithmetic, `(uint32_t)` store) |
| flag | 0x100 if !primaryFlag; Str * 0x10 (mate 0); `(f & outSAMflagAND) \| outSAMflagOR` | same |
| MAPQ | outSAMmapqUnique; nTr >= 5: 0, >= 3: 1, == 2: 3; `& 0xff` | same |
| tid, pos | `(int32)(uint32)Chr`, `(int32)(uint32)(G0 - chrStart[Chr])` | same |
| NH, nM | min(nTr, 65535), min(nMM, 65535) | same |
| mate gap | canonSJ == -3 -> `seqCacheLeanAln` returns false, alignBAM | counted (status bit), never expected single-end |
| sequence | uint16 L, nt16 codes {A1 C2 G4 T8 N15}, two per byte, low nibble first | same |

Pileup arrays (`pileup_rec.h`), as `prepend_handoff` builds them: record r = signature r; `SigRec{aln_off, n_aln, nh = clamp(NH of the
first alignment, 1, 255) (1 without alignments), rlen = L, seq_off}`; `AlnRec{tid, pos, ops_off, n_ops, flags = (flag & 0x10 ? 1 : 0) |
(flag & 0x100 ? 0 : 2), mapq}`; ops verbatim; sequence bytes = the record's nt16 bytes.

Device scheme: `k_count` (thread per record: alignments, ops, sequence bytes, hand-off bytes) -> one `thrust::exclusive_scan` over a
4-field struct -> `k_fill` (thread per record writes its SigRec, AlnRecs, ops, sequence and, optionally, its hand-off bytes at the scanned
offsets). The hand-off bytes are only for the check; the pileup needs the arrays alone.

### 1.2 Hop 7: per-read expansion

Per read i (read order), `readEval` is `gpu_handoff_pass_entries`' decision for a read, in its order: no record (stream) -> skip; record
without alignments -> skip; no primary alignment -> skip; entry not written -> error count; clip > 255 -> error count; UR (3-bit, codes >= 4 -> N)
empty, with N or a homopolymer -> no barcode; CR -> barcode column from the CR table (-2 no barcode, -1 not in the set); kept: `ReadRec{rec,
column, UR packed 3 bits per base, <= 21 bases}` and the low-quality mask (`.reads.q`, zero when entry flag 64 says there is none).
`k_eval` counts the outcomes and quality sums per block, `thrust::exclusive_scan` of the keep flags gives each kept read its slot, `k_scatter`
writes ReadRec and mask in read order (the order the pass appends them).

CR source: STAR's per-read entries (`.reads`, `HandoffReadEntry`: CR and UR 3 bits per base, written by STAR's mapping threads), not a
decoded R1. The kernel only needs (cr3, ur3, lowQual, qualNeqN, flags) per read; a GPU decode of R1 (hop 2) can produce the same struct.

CR table: `distinctBarcodes` lists the distinct CRs of the reads with a record (GPU sort + unique); the host maps each to a column. Two
tables were run: exact CR (CR with N -> no barcode; BarcodeIndex lookup of the CR) and R11 (`singlet::CbCorrector::resolve` of the CR, then
the lookup of the corrected barcode). R11 is a function of the CR string alone during the pass (singlet's R1 qualities are a constant 'F'
and the pass calls `resolve(cr, nullptr, ...)`; the exact histogram is fixed before STAR starts), so a per-distinct-CR table is exact.

## 2. Method

| item | value |
|---|---|
| data | pbmc1k_v3 (`pipeline-benchmark/data/pbmc1k_v3/pbmc1k_v3.1fq`), whole sample: 66,601,887 reads, 31,741,646 distinct after clipping, 105,089 with N (no record: they go through STAR's stream), 31,636,557 records |
| reference | GRCh38-2024-A, STAR 2.7.11b index (`--genome-dir .../star_2.7.11b`), `genes/genes.gtf` |
| engine run | `singlet-gpu <1fq> --pipeline --output-format 1pz --engine gpu --threads 18` with `SINGLET_GPU_PILEUP=on` (the stream path: STAR's unsorted stream, R11 CB correction, the GPU pileup fed by the hand-off) and `SINGLET_ENGINE_GPU_DUMP`; engine wall 262 s, max RSS 43.6 GB |
| truth | the same run's hand-off: `.gpu_handoff.bin` (SGHOFF02, ReadAlign::seqCacheAlignRecords per record, written by STAR's prefill), `.reads` (HandoffReadEntry per read, written by STAR's mapping threads), `.reads.q` (low-quality masks); hard-linked in job /tmp as soon as they appear, because the pileup unlinks them |
| GPU input | the dump of mapDistinct's output (`engine_gpu.cu` dumpOutput: sequences, blobs, readRecord, clip lengths, filter parameters, chrStart), written at the end of mapDistinct, i.e. the very records the prefill turns into the hand-off, in the same order (readRecord of the dump = readRecord of the hand-off: checked) |
| arrays truth | `singlet::gpupileup::Collector::prepend_handoff` run on (a hard link of) the hand-off by the driver: the bridge's own code, not a copy |
| hop 7 truth | a serial transcription of `gpu_handoff_pass_entries`' per-read loop in the driver, reading the hand-off file's records and the `.reads` entries, in two modes: exact CR, and R11 (`singlet::CbCorrector`, STAR's 3M-february-2018 whitelist, the exact histogram rebuilt from the .1fq by `bench/fable/records/r11_hist.cpp` as singlet's decode pass builds it: 63,169,750 reads, equal to the engine's `[1fq-decode] R11 exact CB histogram: 63169750 reads`); the engine does not dump its PileupInput, so the ReadRec content is checked against the transcription and the ReadRec count against the engine's log |
| barcode set | the run's `auto_barcodes.tsv` (157,719 barcodes), keyed as `BarcodeIndex::build` keys it |
| timing | CUDA events, median of 5 repetitions per kernel set; host timings on the job's 18 threads (OpenMP); one GPU job held the data in job /tmp and re-ran the driver for each new commit (trigger file), so every driver commit saw the same engine run |
| privacy | read sequences and the dump stayed in job /tmp (deleted by the trap); the logs print record fields and sequence hashes only |

## 3. Identity

Hop 6 (job 795356, driver commits bb96789f, 5a7db1b8, b2ebb7f0, 94d1f5b4: four runs, all identical):

| check | compared | identical |
|---|---|---|
| hand-off records, byte for byte (per record: n alignments; per alignment tid, pos, flag, MAPQ, 0, NH, nM, n_cigar, CIGAR; then L and the nt16 sequence) | 31,636,557 records (2,839,242 of them without alignments) | 31,636,557 |
| whole data block + record offsets | 2,246,261,173 bytes, 31,636,558 offsets | identical |
| SigRec | 31,636,557 | 31,636,557 |
| AlnRec | 32,045,370 | 32,045,370 |
| CIGAR ops | 41,974,619 | identical |
| nt16 sequence | 1,406,636,366 bytes | identical |
| host port (records.cuh on 18 threads) vs GPU and file | all of the above | identical |

Records: 28,797,315 mapped (at least one alignment), 32,045,370 alignments (1.11 per mapped record), 41,974,619 CIGAR ops. Status
counters: 0 mate-gap transcripts, 0 transcripts without exons, 0 records whose trBest is trInit.

Hop 7 (66,601,887 reads; outcome = the pass's decision, in its order):

| outcome | exact CR | R11 (engine's rule) |
|---|---|---|
| no record (stream) | 156,931 | 156,931 |
| record without alignments | 3,301,393 | 3,301,393 |
| no primary alignment | 0 | 0 |
| entry not written / clip > 255 (pass errors) | 0 / 0 | 0 / 0 |
| no barcode (UR with N or homopolymer; CR rejected) | 60,549 | 1,496,585 |
| CR (corrected) not in the barcode set | 3,022,138 | 352,812 |
| kept: ReadRec + mask | **60,060,876** | **61,294,166** |
| identical to the transcription | 60,060,876 | 61,294,166 |

Cross-checks against the engine's own log of the same run: `hand-off pass: 63143563 reads without BAM records` = reads with a record,
alignments and a primary (66,601,887 - 156,931 - 3,301,393); `hand-off: 61294166 reads take their alignments from the GPU prefill` = the
R11 kept count; the pileup's input had 61,390,838 reads (`part 0/1: 61390838 reads`) = 61,294,166 + 96,672 stream reads. The engine's
`[cb-correct]` line (hand-off + stream primary records) is above the hand-off part in every class: exact 60,410,417 vs 60,342,859,
one_candidate 450,217 vs 422,359, posterior_accepted 883,619 vs 881,760, posterior_rejected 122,200 vs 122,015, no_match 1,334,012 vs
1,331,654, bad_umi 45,863 vs 42,916. Every kept read had a mask (entry flag 64 clear); quality sums of the kept reads: lowQual 2,286,
qualNeqN 0.

The R11 histogram matters: built from the hand-off entries only (no stream reads, 116,429 fewer exact reads), the table kept 61,294,090
reads, 76 fewer than the engine, all through posterior decisions (driver commit bb96789f). With the engine's histogram (r11_hist,
job 795723) the count is the engine's.

## 4. Timing

H100 NVL (132 SMs), job 795356, medians of 5:

| step | host today | GPU (E6) |
|---|---|---|
| records of 31.6 M sequences (pileup arrays) | STAR's hand-off loop (seqCacheAlignRecords + seqCacheHandoffSummary): 5 s on 18 threads (STAR log, 1-s granularity); records.cuh host port, records only: 3.58 s on 18 threads = 2.04 us per record per thread | **37.2 ms** (851 M records/s); call wall incl. allocations 39.9 ms |
| the same + hand-off record bytes | - | 65.1 ms: count 15.6 + scan 4.6 + fill 44.9 (486 M records/s) |
| 2 M records (every 15th record) | - | 2.73 ms (731 M records/s) |
| hand-off file and its parse | 2.25 GB written and mapped; `prepend_handoff` 2.8 s serial | none (arrays in device memory) |
| distinct CRs of 66.6 M reads | - | 5.7 ms (key + thrust sort + unique) |
| CR table, 897,422 CRs | exact 0.012 s, R11 0.17 s on 18 threads (host, stays) | - |
| per-read expansion, 66.6 M reads | `gpu_handoff_pass_entries` 4 s on the job's cores (47 s serial; host-cost-model.md) | **10.5 ms**: eval 5.4 + scan 0.2 + scatter 4.9 |
| arrays to the host (if PileupInput stays a host struct) | - | 2.59 GB: 1.90 s pageable, 0.18 s pinned |
| blobs + sequences to the device (if they come from the host) | - | 5.01 GB: 1.72 s pageable, 0.41 s pinned |

Bytes produced (31,636,557 records, 61,294,166 kept reads): SigRec 506 MB, AlnRec 513 MB, ops 168 MB, sequence 1,407 MB (arrays 2.59 GB);
ReadRec 981 MB, masks 981 MB; the optional hand-off bytes 2,246 MB. Inputs: blobs 2,232 MB (70.5 B per record), sequences 2,783 MB
(1 byte per base), offsets 506 MB. Device memory of buildRecords: inputs + 1.01 GB scan scratch (32 B per record) + outputs.

The GPU time is about 1 % of the 18-thread host port (3.58 s) and under 1 % of STAR's 5-s loop; copying the blobs to the device costs 11 to 46 times the
kernels: the wiring should start from blobs already on the device.

## 5. Interface

```
// records_gpu.h (namespace singlet_records_gpu); records.cuh (namespace e6) holds the per-record and per-read logic
bool buildRecords(const e6::RecParams& P, const uint8_t* dBlob, const uint64_t* dBlobOff, const char* dSeq, const uint64_t* dSeqOff,
                  uint64_t nRec, const uint64_t* dChrStart, bool wantBytes, DeviceRecords& out, std::string& err);
bool distinctBarcodes(const uint32_t* dReadRecord, const e6::ReadEntry* dE, uint64_t nReads, uint32_t cbL,
                      std::vector<uint64_t>& keys, float& ms, std::string& err);
bool expandReads(const uint32_t* dReadRecord, const e6::ReadEntry* dE, const uint64_t* dQ, uint64_t nReads, const DeviceRecords& R,
                 uint32_t cbL, uint32_t umiL, const uint64_t* dKeys, const int32_t* dVals, uint64_t nKeys, DeviceReads& out, std::string& err);
```

Inputs, all device-resident: per record r (mapDistinct's order) its sequence (STAR numeric codes) and blob with offsets, `chrStart`, the
filter parameters (`RecParams`: outFilterScoreMin/OverLread, MatchNmin/OverLread, MismatchNmax/NoverLmax/NoverReadLmax, MultimapNmax,
outSAMmultNmax, outSAMmapqUnique, outSAMflagAND/OR); per read its record (`readRecord`, 0xFFFFFFFF = none), its entry (`e6::ReadEntry` =
HandoffReadEntry: CR and UR 3 bits per base, lowQual, qualNeqN, flags) and optionally its 16-byte low-quality mask; the CR table (sorted
3-bit CR keys and, per key, column >= 0, -1 not in the set, -2 no barcode).

Outputs (device arrays, `pileup_rec.h` layout, exactly `PileupInput`'s byte vectors for the hand-off part): `DeviceRecords::sigs` (SigRec,
record r = signature r), `alns` (AlnRec), `ops` (uint32 `len << 4 | op`), `seq` (nt16, two bases per byte); `DeviceReads::reads`
(ReadRec{sig, bc, umi3}, hand-off reads in read order), `lowq` (2 x u64 per kept read, `PileupInput::lowq`), plus what `Collector::finish`
needs to set `snpSkipN`/`useQual`: `lowQualSum`, `qualNeqNSum`, `maskAll`; and the outcome counts. Stream records go after the hand-off
ones as in `prepend_handoff`: their `aln_off += nAln`, `ops_off += nOps`, `seq_off += nSeq`, their reads' `sig += nRec`.

## 6. What stays on the host

- The barcode universe (`auto_barcodes.tsv`, `BarcodeIndex`) and the R11 correction (`CbCorrector`: whitelist + the exact histogram of
  pass 1): resolved once per distinct CR (0.17 s on 18 threads for 897,422 CRs). This is exact because during the pass R11 is a function
  of the CR string alone (constant 'F' barcode qualities, `resolve(cr, nullptr, ...)`, histogram fixed before STAR starts).
- The per-read entries: here they come from STAR's per-read loop (`ReadAlign::handoffReadEntry`: CR/UR from Solo's barcode read, quality
  counters and mask from Qual0, clip flags with the poly(A) count and the rRNA hook verdict). Hop 7 consumes only the 24-byte struct and
  the mask, so a GPU decode of R1/R2 (hop 2) plus the clip kernel can fill them instead.
- The pass's statistics, not ported: total/mapped/secondary reads, chrM reads, no_barcode, barcoded, multimapper, low_mapq, no_umi,
  per-cell reads and CellAlignQC, cb_status, the ambient whitelist counts and pending list, and the rRNA sample (1/64 of the reads by
  `hash_qname("r<i>")`, the detector on the read sequence). All but the rRNA sample are reductions over the same per-read decision and the
  first primary alignment (natural additions to `k_eval`); the sample can stay on the host (about 1 M reads).
- The hand-off summaries STAR replays for its own per-read output (Log.final.out, SJ.out.tab, Solo Gene; 1.04 GB here) and the cache
  inserts: built in the same STAR loop as the records, still host work.

## 7. Wiring plan for the host-offload track

1. Device-resident blobs. mapDistinct's lanes already hold each batch's blob on the device; today the host assembles them in record
   order and releases the device. Keep a device copy in record order instead (a gather by record index; upload only the host lane's
   records, 85,570 here) and run `buildRecords` before the device is released. Cost 37 ms; no 5-GB upload.
2. Hand the arrays to the pileup without the file. Either a device-pointer `pileupRun` overload (no copy; the pileup appends the stream's
   records on the device with the offset shifts of section 5), or, as a first step with no API change, copy the arrays to a host
   `PileupInput` through pinned memory (0.18 s) and skip `prepend_handoff`. Either removes the record half of STAR's hand-off loop, the
   2.25 GB file and the 2.8 s parse.
3. Per-read expansion: keep `readRecord` on the device; build the 24-byte entries and masks per read on the GPU from the decoded reads
   (CR/UR from R1 at cbS/cbL/umiS/umiL, lowQual/qualNeqN and the 128-bit mask from R2 qualities and minBaseq, clip lengths from mapDistinct's
   clip5/clip3, the poly(A) count from the clipped tail); `distinctBarcodes` -> host CR table (R11 + barcode set) -> `expandReads`. Until
   then, the `.reads` entries STAR writes work unchanged (that is what was validated here).
4. Move the pass's counters into `k_eval` (block reductions, per-cell atomics) and keep the 1/64 rRNA sample on the host.
5. What remains for the host track: STAR's per-read output from the summaries, Solo, and the stream reads with N.

## 8. Surprises and caveats

- The port was exact at the first run: no field, rule or integer width needed a fix (the 64-bit `uint` arithmetic of trimL1/trimR1 and the
  uint32 store of the CIGAR ops were ported as written).
- Exact CR vs R11 differs by 1,233,290 kept reads (2.0 % more with R11), and the R11 table needs the engine's histogram over every read:
  with a histogram from the hand-off entries alone, 76 reads differed (posterior decisions).
- The hand-off bytes are slow to write thread per record (fill 44.9 ms with them vs about 17 ms without); the pileup does not need them.
- 2,839,242 records without alignments still get a SigRec and sequence bytes, because `prepend_handoff` gives them signatures; their reads
  are skipped. Dropping them would save about 0.17 GB but change the signature numbering.
- STAR's hand-off loop took 5 s here (records and summaries, 18 threads), half the 10 s of host-cost-model.md's measurement on another node.
- The engine's PileupInput is not dumped (a hook would go in `pileup_gpu.cu`, outside this experiment's scope), so ReadRec content is
  validated against the transcription of the pass and only its count against the engine.

## Appendix: jobs, commits, files

| job | what | result |
|---|---|---|
| 795349 | build singlet-gpu with the dump (ccfe2c28), short-cpu | 4:29, ok |
| 795355 | driver compile on debug-cpu | cancelled (QOS CPU limit), resubmitted on short-cpu |
| 795357 | driver compile (c7360f58) | failed: -Wformat flood through `head` killed nvcc; fixed in cd44b64f |
| 795361 | driver compile (cd44b64f) | ok |
| 795552 | driver compile (b2656970) | ok |
| 795356 | engine run + driver (bb96789f), re-runs 5a7db1b8, b2ebb7f0, 94d1f5b4; short-gpu, 21:59 | all PASS |
| 795581 | 2 M-read run on debug-gpu | cancelled before it started (795356 started) |
| 795723 | r11_hist over pbmc1k_v3.1fq (5a7db1b8), short-cpu, 37 s | 63,169,750 exact reads, 302,186 barcodes |

Results on Clipper: `$W/jobs/fable/records/results/run-795356/` (summary.txt, engine.log, driver-<sha>.log, e6-<sha>.json),
`$W/jobs/fable/records/results/r11_exact_hist.bin`. Copies of the final driver log and JSON: `bench/fable/records/results/`.
