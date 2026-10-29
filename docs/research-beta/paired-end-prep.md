# Phase 1d preparation: paired-end alignment on the GPU

Branch `claude/gpu-paired-prep-f54059`, written for W11 (which owns `gpu/engine-gpu-stream` and every Clipper run).
Phase 1d of [align-and-count-design.md](align-and-count-design.md) is **not approved to switch on**. Nothing here changes
`stitch.cuh`, `engine_gpu.cu` or `pileup.cu`; no Clipper job, bridge restart or `sync.sh` was run.

## 1. What is here and how far it was checked

| Piece | File | Status |
|---|---|---|
| STAR's paired-end rules, with file:line | section 3 below | read from `src/star/` (2.7.11b, as vendored) |
| Reference model (Python + numba) | `bench/gpu_paired/paired_model.py` | 25 tests pass (`python -m pytest bench/gpu_paired -q`); expected values were worked out by hand from the rules, before running |
| Handcrafted cases | `bench/gpu_paired/cases.py`, `fixtures/cases.{json,txt}` (about 175 kB each) | overlap, protrusion, mate gap at the limit, junction in a mate, discordant, multimapper, end-to-end |
| Kernel draft | `src/engine_gpu/paired_windows.cuh` | **not compiled with nvcc.** Compiled as plain C++ with g++ 13.2 (`HD` is empty without `__CUDACC__`) and run on the host: all 18 cases equal the model (`bench/gpu_paired/paired_unit_host.cpp`). The `__global__` kernel and everything under `#ifdef __CUDACC__` has never been compiled |
| Microbenchmark harness | `bench/gpu_paired/paired_microbench.cu` (+ `paired_harness_common.h`) | **not compiled with nvcc, not run.** The CUDA-free half is exercised by the host check above |
| Comparison tools | `compare_star_sam.py`, `replay_model.py` | parse and compare correctly on data made from the model itself (`test_tools.py`); never run on real kernel or STAR output |

What the checks do **not** show: that the model equals STAR. The model and the kernel draft are both transliterations of the
single-end port (`stitch.cuh`) and of the STAR source I read; a misreading of a STAR rule would be in both. The only decisive
test is the comparison with a real STAR run (section 7, step 4).

## 2. The main finding: STAR has no separate "paired window join"

The design note says "windows are joined across mates under the mate-gap limit, as STAR's paired windows are". In STAR that is
not a separate step. STAR turns a pair into **one read** and runs the single-end machinery on it:

1. `ReadAlign_oneRead.cpp:40-58`: `Read1[0]` = mate 1, one spacer base (`MARK_FRAG_SPACER_BASE` = 11), then the **reverse
   complement** of mate 2. `Lread = L0 + L1 + 1`. Both mates now read in the same direction on the + strand of the fragment.
2. `qualitySplit` (`SequenceFuns.cpp:425-448`) splits at the spacer and numbers the pieces by mate (`iFrag`, `splitR[2]`).
3. The seed loop (`ReadAlign_mapOneRead.cpp:46-153`) is unchanged; each seed carries its `iFrag`.
4. Seeds of **both mates go into the same windows**. A window is a run of `2^winBinNbits` bins; seeds merge into one window when
   their bins are within `winAnchorDistNbins` (`ReadAlign_createExtendWindowsWithAlign.cpp:32-82`). "Joined under the mate-gap
   limit" therefore means: the window geometry derived from `alignMatesGapMax` (`Genome_genomeLoad.cpp:421-446`) lets the two
   mates' seeds meet in one window, and `stitchAlignToTranscript` then enforces the limit exactly (rule P8).
5. The stitching recursion is unchanged; a seed of the other mate triggers the "different fragments" branch (rules P8-P11).

`stitch.cuh` already ports all of it, including the mate branch of `stitchAlignToTranscript` (`stitch.cuh:889-920`). Only four
places assume one mate (section 5). That is why the kernel draft is small and the work is mostly the driver, the cache and the
capacities, not the alignment core.

**Geometry on a human-size genome.** `winBinNbits` is the larger of two terms: the one from the limits
(`log2(max(alignIntronMax, alignMatesGapMax) / 4)`, 9 for ATAC) and one from genome size (`log2(nGenome/40000 + 1)`, 16 for
3.1 Gb) (`Genome_genomeLoad.cpp:426-427`). For GRCh38 both modes get 16-bit bins (65 kb). ATAC then has flank 1 bin and anchor
distance 2 bins (`:442-443`), spliced default 4 and 9. So in ATAC mode the 2000 limit is enforced by stitching alone; the
windows are wide. (Tested: `test_window_geometry_on_a_human_size_genome`. The harness prints the three values on start.)

## 3. STAR's paired-end rules

Paths are in `src/star/`. "Spliced" = defaults (`alignIntronMax 0`, `alignMatesGapMax 0`: no limit). "Unspliced" = ATAC
(`--alignIntronMax 1 --alignMatesGapMax 2000`). Model = `paired_model.py`; Kernel = `paired_windows.cuh` or the `stitch.cuh`
function it reuses.

| # | Rule | Spliced | Unspliced (ATAC) | STAR source | Model / Kernel |
|---|---|---|---|---|---|
| P1 | Combined read: mate 1, spacer 11, reverse complement of mate 2; `Lread = L0+L1+1`; `Read1[1]` complement, `Read1[2]` reverse of that (complement leaves values > 3 as they are); `Lread > 650` is a fatal error | same | same | `ReadAlign_oneRead.cpp:40-58, 74-77, 43-49` | `build_read` / `combineMates` |
| P2 | Mismatch budget `min(outFilterMismatchNmax, outFilterMismatchNoverReadLmax * (L0+L1))`: the sum of mate lengths, not `Lread` | same | same | `oneRead.cpp:83` | `map_windows` / `mapPhase1PE` (stitch.cuh:1234 uses `Lread`) |
| P3 | Seeds: pieces split at the spacer and at Ns; `iFrag` = mate; `seedSearchStartLmax = min(50, 1.0*(Lread-1))`; seeds never cross the spacer | same | same | `SequenceFuns.cpp:425-448`, `mapOneRead.cpp:18, 46-47, 133` | shared (`qualitySplit`, seed loop) |
| P4 | Window geometry: `winBinNbits`, `winFlankNbins`, `winAnchorDistNbins` from the two limits and genome size; clamped to `genomeChrBinNbits` | defaults 16 / 4 / 9 | 16 / 1 / 2 on a 3.1 Gb genome (9 / 4 / 8 on a toy genome) | `Genome_genomeLoad.cpp:421-449` | `Params.derive` / host `finishParams` |
| P5 | Window creation: only anchors (`Nrep <= winAnchorMultimapNmax`, 50) create windows; a new anchor merges left and right windows of the same strand and chromosome within `winAnchorDistNbins`; flanks of `winFlankNbins` bins are added; windows are per strand, so mates of a concordant pair share one (mate 2 is already reverse complemented) | same | same | `ReadAlign_stitchPieces.cpp:71-123, 129-170`, `createExtendWindowsWithAlign.cpp:32-83` | `Windows.create_extend`, `flanks` / shared |
| P6 | Seed assignment to windows: same-diagonal overlap test includes `iFrag`; long seeds replace short ones; at `seedPerWindowNmax` (50) the shortest non-anchors are dropped | same | same | `ReadAlign_assignAlignToWindow.cpp:10-90` (`:16` iFrag) | `Windows.assign` / shared |
| P7 | Stitching walks the seeds of a window in read order (`rStart`), including or excluding each; the last anchor may not be excluded unless another anchor is in | same | same | `stitchWindowAligns.cpp:310-351`, `stitchPieces.cpp:369-371` | `stitch_window` / `stitchWindowAlignsPE` |
| P8 | **Mate branch**: two seeds of different mates stitch only if `gBstart + exons[0][R] + alignEndsProtrude.nBasesMax >= exons[0][G]` (or `exons[0][G] < exons[0][R]`); else `-1000008`. `alignMatesGapMax > 0` and `gBstart > end of previous exon + alignMatesGapMax` gives `-1000004` | no limit (0) | gap at most 2000: a gap of exactly 2000 stitches, 2001 does not | `stitchAlignToTranscript.cpp:352-356, 406-407` | `stitch_align` / `stitch.cuh:889-920` |
| P9 | Mate branch scoring: add `L` of the new seed; extend the previous mate's end forward with `extendAlign(..., DEF_readSeqLengthMax, ext[iFrag(A)][1])` (stops at the spacer); add the new seed; extend its start backward, to the transcript start unless end-to-end is forced; mark the junction `canonSJ = -3` | same | same | `stitchAlignToTranscript.cpp:360-405`; `extendAlign.cpp:33, 63` | same |
| P10 | Within one mate: gap/overlap/deletion/insertion/junction stitching exactly as single-end. Deletion longer than `alignIntronMax` (when > 0) gives `-1000003`; `Del >= alignIntronMin` (21) is a junction with motif scoring, else a deletion (`scoreDelOpen+Del*scoreDelBase`) | junctions, sjdb, motifs | only 1-base deletions (`Del > 1` refused); no junction can occur; insertions unchanged | `stitchAlignToTranscript.cpp:39-348` (`:100` intron max) | `stitch_align` / shared |
| P11 | Extension of the transcript's ends at finalization: start uses the mate of the first exon, end the mate of the last (`ext[imate][Str != imate]` and `ext[imate][imate == Str]`); order depends on strand | Local (default) soft-clips; EndToEnd forces | same | `stitchWindowAligns.cpp:35-81` | `finalize` / `finalizeTranscriptPE` |
| P12 | Per-mate spliced-length check: a mate with a junction must keep `>= alignSplicedMateMapLmin` and `>= 0.66 * readLength[its mate]` | applies | never (no junctions) | `stitchWindowAligns.cpp:154-167` (`:159`) | **changed vs single-end**: `w.readLength[iFrag]` instead of `Lread` |
| P13 | **Mate overlap rules**, for transcripts whose first and last exon are in different mates: (a) end of last exon `<=` start of first: dropped (negative insert size); (b) if the mates overlap: left start `>` right start + `exons[0][R]` + protrude dropped; left end `>` right end + protrude dropped; junctions of the overlapping stretch must coincide (same donor end and acceptor start) or the transcript is dropped | all of it | (a), (b); no junction check needed | `stitchWindowAligns.cpp:179-219` (`:180, :189, :191, :193, :198-217`) | `mate_overlap_ok` / `mateOverlapOk` (**stitch.cuh:1048 drops every two-mate transcript instead**) |
| P14 | Score: sum of seed and extension scores, plus gap/del/ins/junction penalties, plus `ceil(log2(gLength) * -0.25 - 0.5)` floored at 0, where `gLength` = start of first exon to end of last exon (the whole insert, not per mate) | same | same | `stitchWindowAligns.cpp:221-225`; `gLength` at `:94` | shared (`genomicLengthScore`) |
| P15 | Single-mate transcripts are kept when within `outFilterMultimapScoreRange` of the window best **or of their own mate's best** (`maxScoreMate[0..1]`, two entries); within a window, subsets of a better transcript are dropped (`blocksOverlap`), supersets replace | same | same | `stitchWindowAligns.cpp:232-237, 245-304`; `ReadAlign.cpp:125-127` | `Wn.maxScoreMate` / **changed**: `maxScoreMateP[2]` (stitch.cuh:206 has one entry) |
| P16 | Best transcript `trBest`: highest score over windows' first transcripts, ties to the smaller `gLength` (the shorter insert), then the first window | same | same | `stitchPieces.cpp:420-423` | shared |
| P17 | multMapSelect: every transcript of every window within `outFilterMultimapScoreRange` (1) of the best; `nTr > outFilterMultimapNmax` (10) is unmapped; one transcript = primary; several = primary is `trBest` (default `OneBestScore`, old order) | same | same | `ReadAlign_multMapSelect.cpp:15-60, 82-91` | `map_windows` / shared |
| P18 | mappedFilter thresholds use the combined length: score `< 0.66*(Lread-1)`, matched bases `< 0.66*(Lread-1)`, mismatches `> nMM max` or `> 0.3*rLength`, multimapping | same | same | `ReadAlign_mappedFilter.cpp:8-18` | shared |
| P19 | SAM flag "properly paired" (0x2): if `--alignEndsProtrude N ConcordantPair` with N > 0 always; otherwise only if left start `<=` right start + `exons[0][R]` and left end `<=` right end. Default is not Concordant | default | default | `ReadAlign_outputTranscriptSAM.cpp:70-85`; `Parameters.cpp:1084-1093` | `proper_pair` (not needed by the kernel) |
| P20 | peOverlap (merging overlapping mates before alignment) | off (`peOverlapNbasesMin 0`) | off | `ReadAlign_peOverlapMergeMap.cpp:6`, `parametersDefault:655` | **not modelled, not ported**: host lane if a run sets it |

### Reading the table: which rules the fixtures exercise

| Fixture (`cases.py`) | Rules | Expected, by hand |
|---|---|---|
| `concordant`, `overlap50` | P8, P9, P13 | 198 (200 matches, length term -2) |
| `inside` | P13 (right end before left end) | pair refused |
| `protrude10_0/9/10` | P8 `alignEndsProtrude`, P13 | refused at 0 and 9; accepted at 10 (both comparisons met with equality) |
| `gap_at_limit`, `gap_over_limit` | P8 mate gap (ATAC) | gap 2000 accepted (197), 2001 refused |
| `far_spliced`, `far_atac` | P4, P8 | 100 kb apart: accepted in spliced mode (196), not in ATAC |
| `junction_m1`, `junction_m2` | P10, P12 | GT/AG junction in one mate, `canonSJ [1, -3]` / `[-3, 1]` |
| `junction_consistent`, `junction_inconsistent` | P13 | consistent accepted (208); with different acceptors the 4-exon pair is refused and STAR reports the pair **without** mate 2's first exon (168), not an unmapped pair |
| `discordant_ff`, `discordant_outie`, `discordant_chr` | P5, P8 | no pair transcript, single-mate transcripts only, filtered as too short |
| `multimapper` | P17 | two pair transcripts, one primary |
| end-to-end test | P9, P11 | Local soft-clips 2 bases, EndToEnd extends them |

## 4. The dedup key and a paired cache

**Dedup key.** The design's key is the ordered mate pair. Concretely one record per pair, `mate1 | separator | mate2` with both
mates in their original orientation and the separator outside the base alphabet, so `(A,B) != (B,A)` and the boundary between
mates is part of the key (two pairs with the same concatenation but a different split must not collide). The GPU dedup
(`k_hash`, `k_groups` in `engine_gpu.cu`) hashes a record; it needs the pair record instead of one mate. After dedup,
`combineMates` builds STAR's combined read from the record.

**Why STAR's cache cannot take pairs today.**
- `SeqCache.cpp:73` turns the cache off for `readNmates != 1`.
- Even without that, `ReadAlign_seqCache.cpp:63-70` (`seqCacheable`) refuses any base `> 3`, and the combined read always has
  the spacer (11). The key is also `Read1[0]` of length `Lread` with the mate boundary implicit.
- `SeqCacheGpuPrefill.cpp:222` skips paired-end; `readSequences` reads mate 1 only.

**What a paired cache needs.**
1. Key: hash and stored sequence over the 2-bit bases of both mates **plus `L0`** (the spacer position), no spacer in the packed
   bases; `seqCacheable` rejects an N in either mate.
2. Value: the existing record (`encodeTr` / `seqCacheEncode`, `ReadAlign_seqCache.cpp:72-`) already carries `iFrag`, the exons and
   the `-3` mate breaks; `nTr > 255` stays uncacheable. Not traced: `seqCacheRestore` and `seqCacheAlignRecords` for two-mate BAM
   records (flags 0x1/0x2/0x40/0x80, mate fields) and `seqCacheHandoffSummary` for a pair. Both are single-end today; check them
   before turning the cache on.
3. Size: an entry is about twice as long, so the same memory holds about half as many pairs.

**Expected value.** Pair-level dedup only helps where both mates repeat exactly: PCR duplicates in ATAC, high-copy loci
(chrM). Smart-seq2 reads per well rarely repeat. The GPU's gain on pairs is per read, not per distinct sequence. Measure first
(distinct pairs in the first million):

```bash
paste -d'|' <(awk 'NR%4==2' r1.fq | head -1000000) <(awk 'NR%4==2' r2.fq | head -1000000) | sort -u | wc -l
```

Fragment-level dedup for ATAC (cell, contig, start, end) is the fragment reducer of phase 1e, not this phase.

## 5. What changes, where

New (this branch): `src/engine_gpu/paired_windows.cuh`, `bench/gpu_paired/*`.

`paired_windows.cuh` (namespace `m5::pe`) reuses `stitch.cuh` untouched and re-declares what assumes one mate:
`WorkPE` (adds `readLength[2]`, `maxScoreMateP[2]`), `combineMates` (P1), `mateOverlapOk` (P13), `finalizeTranscriptPE`,
`stitchWindowAlignsPE`, `mapFinishPE` (copies with the changes marked `PE:`), `mapPhase1PE` (P2), `dumpWindows`, and `k_pe_map`
(one pair per thread). The copies exist so this draft needs no edit to `stitch.cuh`.

**Changes the files I did not touch need** (for when 1d is approved):

| File | Change |
|---|---|
| `stitch.cuh:206` | `maxScoreMate[1]` to `[2]`; reset both at `:1241`; use at `:1057`, `:1069` |
| `stitch.cuh:1041` | `Lread` to `readLength[exons[iex][EX_iFrag]]` (needs `readLength[2]` in `Work`) |
| `stitch.cuh:1048` | replace `return true` by the check for mates (`mateOverlapOk`) |
| `stitch.cuh:1234` | mismatch budget from `L0 + L1`, not `Lread` |
| `stitch.cuh` (clean form) | template the above on a `Mates` policy and delete the copies in `paired_windows.cuh` |
| `engine_gpu.cu:38-41` | lane capacities for pairs: `LREAD` = 2 x mate + 1 (304 for 2 x 150; the lanes have 160 and 650), `NWA` 50 (lanes A and M have 16 and 8) |
| `engine_gpu.cu` (`Input`, `k_hash`, `k_groups`, `k_clip`) | second mate stream; pair record for dedup; clipping of mate 2 (1a clips mate 1 only) |
| `SeqCacheGpuPrefill.cpp:222` and `readSequences` | stop skipping pairs; read both FASTQs; paired `fillParams` checks (`peOverlapNbasesMin`, `alignEndsProtrude` accepted, `alignSoftClipAtReferenceEnds`) |
| `SeqCache.cpp:73`, `ReadAlign_seqCache.cpp:63-72` | paired key (section 4) |
| `pileup.cu` | no use of `canonSJ`/`iFrag` found in it; what consumes the alignments of a pair is `seqCacheHandoffSummary` (not traced) |

## 6. Risks to the 99.9% criterion

Ordered by how likely I think they are to cost reads, not by size.

1. **Shared source of error.** Model, kernel and `stitch.cuh` come from the same reading of STAR. The 18 fixtures check the
   rules I know about. Run the real comparison first (section 7).
2. **Clipping.** ATAC and Plate reads usually need adapter trimming (Nextera adapters on both mates). STAR's clipping changes
   `readLength[]` and `Lread`; phase 1a ported 3' clipping of mate 1 only. A paired run without both mates clipped as STAR
   clips them will differ on short inserts. Decide first whether the comparison runs unclipped.
3. **Capacities and cost of the recursion.** A pair window holds the seeds of two mates, so the include/exclude recursion
   (`stitchWindowAligns`, up to 2^n subsets, limited by the anchor rule) is longer than for one mate. The single-end lanes
   have `NWA` 8, 16 and 50; a pair needs 50 and a bigger `Work` (`sizeof(WorkPE<PeLane>)` = 639,368 bytes, measured with g++; 4096 threads per batch,
   the harness default, is 2.6 GB). Expect more reads to overflow to the host lane and fewer threads in flight; measure both
   (status counts and pairs/s in the harness).
4. **Window order and ties.** Order of windows follows suffix-array order and decides primary among equal scores and
   `trBest`'s tie. It matches STAR because the kernel reads the same SA; my model does not have an SA (its loci are sorted by
   position), so the model can disagree with STAR on ties only. Unique reads are not affected.
5. **"Unique" is STAR's `nTr == 1`**, and a single-mate alternative within score range 1 of a pair, in another window, makes
   `nTr == 2` (P15, P17). That quirk is ported but only the real run shows how often it occurs.
6. **End-to-end and protrusion options** used by real recipes (`--alignEndsType EndToEnd` for some Smart-seq2 pipelines,
   `--alignEndsProtrude`): ported, covered by one fixture each; check which options the plate and ATAC recipes actually pass.
7. **Not ported:** `peOverlap`, chimeric detection, `outFilterType BySJout`, variation/WASP. Any recipe that sets one goes to
   the host lane. Check `fillParams` for each (it must refuse them for pairs).
8. **Genome with junctions.** The harness needs a no-sjdb index. The shared code handles sjdb seeds for single-end; the paired
   behaviour with sjdb is untested, the model does not cover it.
9. **The comparison itself.** `compare_star_sam.py` compares per mate chromosome, strand and reference blocks split at
   junctions; it does not compare where an indel sits inside a block, or soft-clip lengths. Reads that differ only there are
   counted identical. Tighten it (CIGAR from exons) before quoting 99.9%.

## 7. W11: build and run

Nothing below was run by me. Every step is yours; it needs a GPU node (and STAR) only where stated.

**1. Branch and the checks that need no GPU** (laptop or login node, g++ and python with numpy/numba/pytest):

```bash
git fetch origin claude/gpu-paired-prep-f54059 && git switch claude/gpu-paired-prep-f54059
python -m pytest bench/gpu_paired -q            # 25 tests
g++ -std=c++17 -O1 -Isrc/engine_gpu bench/gpu_paired/paired_unit_host.cpp -o /tmp/pe_host \
  && /tmp/pe_host bench/gpu_paired/fixtures/cases.txt     # 0 of 18 cases differ from the model
```

**2. Compile the GPU harness** (first time nvcc sees these files; expect fixes):

```bash
CUDA_ARCH=sm_90 bash bench/gpu_paired/build.sh   # sm_80 for A100; also runs --unit on host and device
```
`--unit` prints one line per case: `host ok/DIFF device ok/DIFF`. A `DIFF` on the device alone is a CUDA problem; on both,
a logic problem the host check should already have shown.

**3. Data.** A STAR index **without** `--sjdbGTFfile`, human, the same `--genomeSAindexNbases` as the engine uses; the first
200k pairs of `atac_pbmc_5k_nextgem` (the two genomic mates, R1 and R3 of a 10x ATAC run, as plain FASTQ), then the same for a
Smart-seq2 study from the atlas with `--mode spliced`:

```bash
head -n 800000 r1.fastq > r1.fq ; head -n 800000 r3.fastq > r2.fq
```

**4. STAR reference** (same genome, same pairs, ATAC parameters shown):

```bash
STAR --genomeDir $G --readFilesIn r1.fq r2.fq --outSAMtype SAM --outSAMunmapped Within \
     --outSAMattributes NH HI AS nM --alignIntronMax 1 --alignMatesGapMax 2000 --outFileNamePrefix ref_
```
Record STAR's "Mapping speed" from `ref_Log.final.out` for the timing comparison.

**5. Kernel run and checks:**

```bash
bench/gpu_paired/paired_microbench --genome $G --r1 r1.fq --r2 r2.fq --n 200000 --mode atac \
    --batch 4096 --repeat 3 --out kernel.tsv --dump win.bin
python bench/gpu_paired/compare_star_sam.py kernel.tsv ref_Aligned.out.sam $G      # the 99.9% number
python -P bench/gpu_paired/replay_model.py $G r1.fq r2.fq win.bin kernel.tsv --mode atac   # kernel vs model, real genome
```

Read the output as follows:
- `compare_star_sam.py` prints counts (kernel overflow = host lane, STAR unique vs kernel unmapped, identical, different) and
  the first 20 differences. The criterion is the last line, over STAR-unique pairs.
- `replay_model.py` replays the windows the kernel built through the Python model. Any difference there is in the paired
  rules or their port, independent of seeding. If it is 0 and `compare_star_sam.py` shows differences, they are in seeding or
  window assignment (shared single-end code) or in a rule the model has wrong too.
- The harness prints status counts (ok / overflow / blob full) and kernel pairs per second (kernel only, after the first
  pass). Compare with STAR's speed on the same pairs. Overflow share is the host-lane share.
- For ATAC check the printed geometry is `winBinNbits 16 winAnchorDistNbins 2 winFlankNbins 1`.

**6. Before switching anything on**, the work in section 5's table, the clipping decision (risk 2) and the cache check
(section 4).

## 8. What I did not do or verify

- No Clipper or Anvil job; nothing ran on a GPU; no nvcc compile; no STAR run. The 99.9% figure is unknown.
- I did not read `engine_gpu.cu` in full (lines 38-200 only) or trace the paired path of `seqCacheRestore`,
  `seqCacheAlignRecords` or `seqCacheHandoffSummary`; section 5 lists them as work items, not findings.
- The model finds seeds by brute force on a small genome; it never ran on a real genome. `replay_model.py` is its only
  real-genome use, and it takes seeding from the kernel.
- sjdb seeds, `peOverlap`, clipping, chimeric detection and variation are outside the model and the kernel draft.
