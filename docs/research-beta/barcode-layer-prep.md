# Barcode layer preparation for phases 1c and 1f

Status: preparation; section 12 holds the measured results (the kernels now compile and run on an H100 and were compared with STAR on real reads). Sections 1-11 were written before that, on a laptop. Written on the laptop from the
vendored STAR 2.7.11b sources in `src/star/` and singlet's own tables. Branch `claude/gpu-barcode-layer-484c70`.

Decisions of 2026-10-29 (feature-barcode UMI rule, CRISPR matching, inDrop, gate definitions) are folded in below.

Deliverables:
- `src/engine_gpu/barcode_layer_ref.py`: Python and numba reference, one function per STAR rule.
- `tests/python/test_barcode_layer_ref.py`: 31 tests, handcrafted reads of a few KB, all passing.
- `src/engine_gpu/barcode_layer.cuh`: CUDA drafts, **not compiled**. Not included from `engine_gpu.cu` or `pileup.cu`.
- This note.

Design section 4.1 (`align-and-count-design.md`) says the engine reproduces STAR's correction rules and does not replace
them. This note pins down what those rules are, as **singlet runs them**, which is narrower than what STAR can do.

## 1. What singlet actually asks STAR for

Three facts decide the scope of 1c. All three are in `src/pipeline/singlet.cpp`.

1. **Every complex or no-whitelist protocol runs with `--soloCBmatchWLtype Exact`** (7309-7316). Only `CB_UMI_Simple`
   gets `1MM_multi_Nbase_pseudocounts`. The Complex mismatch rules (-12, multi-match, EditDist_2) never run in
   production, so 1c does not need them for gate (a).
2. **No protocol uses an adapter or a read-end anchor today** (the inDrop W1 route of section 7a is a decided change). `--soloAdapterSequence` is never passed (grep: no hit).
   `soloCBposition` is always absolute from read start (`0_s_0_e`, `sra_encoder.h:2524-2551`; `singlet.cpp:4191` rejects
   anything else for the concatenated form). STAR can anchor on an adapter (inDrop, BD); singlet does not. inDrop's variable
   BC1 length (8-11) is therefore not handled by singlet today: it reads a fixed `0_0_0_9`.
3. **Complex protocols are rewritten before STAR sees them.** `use_complex_concat` (3761-3770, 4175-4215, 6657-6681)
   rewrites R1 as `[seg0][seg1]..[segN][UMI]`, and STAR runs `CB_samTagOut` with a Cartesian-product whitelist and `Exact`,
   to avoid CB_UMI_Complex's 200-400 GB. The cell is then the position in that product list. Native `CB_UMI_Complex` with
   per-segment lists is the fallback (6683-6812). The two forms give the same cell identity (below); the reference models
   the native form because it has the segment structure.

Consequence for the design text: section 4.1 says the no-whitelist case does "knee-based or bead-error correction". STAR does
none, and singlet does none, so gate (a) cannot be passed with it. Either drop that clause or make it a later option.

## 2. Rules table

"Ref" is the function in `barcode_layer_ref.py`. Lines are in `src/star/` unless a path is given.

| # | Rule | Source | Ref |
|---|---|---|---|
| R1 | Simple: `cbSeq = bSeq[cbS-1, +cbL)`, `umiSeq = bSeq[umiS-1, +umiL)`, CB quality likewise. CB at most 31, UMI at most 16 | SoloReadBarcode_getCBandUMI.cpp:249-252; ParametersSolo.cpp:50-62 | `get_cb_umi_simple` |
| R2 | Barcode read shorter than CB+UMI with `--soloBarcodeReadLength 0` (singlet's setting): padded with `N`, quality `H`. With a non-zero length a mismatch is fatal | :222-235 | `_pad_barcode_read` |
| R3 | No whitelist (`None`): any N in the CB gives -2, otherwise the 2-bit value is the match (cbMatch 0). No stats, no correction | :18-27 | `match_cb_to_wl(wl=None)` |
| R4 | With a whitelist: two or more Ns give -2; no N tries an exact hash hit first (cbMatch 0) | :29-40 | `match_cb_to_wl` |
| R5 | Exact type: stop after R4, result -1. `1MM*` types continue | :42-43 | same |
| R6 | One N: try the four bases at the N in order A,C,G,T. Without `Nbase` a second hit gives -3. With `Nbase` all hits are kept. Quality kept is the N base's | :46-63 | same |
| R7 | No N, no exact hit: try every single substitution, last base first (`ii` counts from the least significant bits), substitutions `jj=1..3` (XOR the 2-bit code). Each whitelist hit is kept with the quality of the changed base | :64-75 | same |
| R8 | 0 hits: -1. 1 hit: cbMatch 1. More: -3 and the list is dropped unless `1MM_multi*`, else cbMatch is the number of hits | :76-84 | same |
| R9 | Exact histogram `cbReadCountExact[cb]`: one per read whose **final** cbMatch is 0, counted in `addStats`, which runs after the UMI check. A read with a bad UMI is not counted. Counts are summed over threads | :87-96; SoloReadBarcode.cpp:29-31 | `exact_counts` |
| R10 | Pseudocount types add 1 to every whitelist entry (including unseen ones) after the sum | Solo.cpp:66-71; SoloFeature_sumThreads.cpp:52-57 | `exact_counts` |
| R11 | Posterior (reads with cbMatch greater than 1). Per candidate with count above 0: `q = min(qual - 33, 33)`, `pin = count * 10^(-q/10)` stored as **float**, `ptot` and `pmax` float, first candidate wins ties (`pin > pmax`). Accept when `ptot > 0 && pmax >= cbMinP * ptot`, `cbMinP = 0.975f` (float in this build: `MATCH_CellRanger` is not defined in any build file) | SoloReadFeature_inputRecords.cpp:75-90; ParametersSolo.cpp:287-289; ParametersSolo.h:136-141 | `resolve_cb` |
| R12 | Single hit (cbMatch 0 or 1): taken as is. For types with `oneExact` (Exact, 1MM, 1MM_multi, not the pseudocount types) a 1MM hit with no exact read on that cell is dropped | :35-41; ParametersSolo.cpp:701-715 | `resolve_cb` |
| R13 | A read with cbMatch below 0 is never recorded | SoloReadFeature_record.cpp:66 | `resolve_cb` |
| R14 | UMI check (Simple only): any N gives -23. Value equal to a homopolymer of **`umiL` at construction** gives -24. The read is dropped before CB matching is used | :127-138, :291-303; SoloReadBarcode.cpp:16-20 | `check_umi` |
| R15 | `CB_samTagOut`: same extraction and `matchCBtoWL`, **no UMI check**. `cbSeqCorrected` = whitelist string of the first match, the raw CB with no whitelist, `"-"` otherwise | :306-322 | `get_cb_umi_simple(sam_tag_out)` |
| R16 | Complex: UMI is extracted first (-22 if it cannot be); the UMI check runs but only blocks CB matching | :341-354 | `get_cb_umi_complex` |
| R17 | Complex segment extraction: anchor 0 read start, 1 read end, 2 adapter start, 3 adapter end, plus a signed distance. Fails if `pos[0] < 0`, `pos[1] > size`, or `pos[0] > pos[1]`. `pos[1] == size` **passes** and `substr` clips, so the segment is one base short | SoloBarcode_extractBarcode.cpp:4-33 | `extract_barcode` |
| R18 | Adapter (unused by singlet today; used by the decided inDrop W1 route, section 7a): first position with the fewest mismatches below `adapterMismatchesNmax` (default 1), `N` in the adapter is a wildcard, ties keep the first, -21 on failure. If every base mismatches, the position stays 0 | :332-339; SequenceFuns.cpp:352-371; parametersDefault:828 | `local_align_hamming_dist` |
| R19 | Complex segment match: the extracted length must be at least `minLen`, below the table size, and have a non-empty list for that length (else -11, only the first failure is recorded). Exact or 1MM per segment through the same `matchCBtoWL`. A 1MM hit in a second segment while the first already had one gives -12 | :357-413 | `get_cb_umi_complex` |
| R20 | Complex cell index: `sum over segments of wlFactor * (indexInLengthList + wlAdd[length])`. `wlFactor` is the product of the earlier segments' list sizes, so segment 0 varies fastest. `wlAdd[len]` is the number of entries in shorter lengths | :409; ParametersSolo.cpp:393-395; SoloBarcode.cpp:19-34 | `link_segments`, `build_segment_wl` |
| R21 | Complex UMI homopolymer: `SoloReadBarcode` is built while `umiL` is 0 (set to the first UMI's length later, ParametersSolo.cpp:372, getCBandUMI:347-348), so all four homopolymer values are 0: **only an all-A UMI is rejected**, poly-C/G/T pass | SoloReadBarcode.cpp:16-20 | `check_umi(homopolymer_len=0)` |
| R22 | Whitelist loading: sorted, de-duplicated 2-bit values (index = sorted position). The test `if (convertNuclStrToInt64(...))` skips only an entry whose N is at **position 0**; an N elsewhere is read as A | ParametersSolo.cpp:321, 334-338, 373-396 | `Whitelist` |
| R23 | How the corrected CB reaches the output. `CB_samTagOut`: STAR writes `CB:Z:` = `cbSeqCorrected` into each alignment while mapping. `CB_UMI_Simple`: written only when the sorted BAM is built (`cbWLstr[readInfo[iread].cb]`, `"-"` if none, UB likewise from the collapsed UMI). In singlet's stream path there is no sorted BAM, so the pileup takes CR (raw CB) from the hand-off entry and resolves the cell itself | ReadAlign_alignBAM.cpp:169-170, 468-471; SoloFeature_addBAMtags.cpp:8-23; singlet.cpp:5034 | n/a |
| R24 | The hand-off entry already carries `cr3`, `ur3`: raw CB and UMI, 3 bits per base (A C G T N), first 21 bases | ReadAlign_outputAlignments.cpp:356-380; SeqCache.h:77 | `pack3` |

### Per protocol, as singlet runs it

Geometry is checked against `protocol.h:66-190` and `sra_encoder.h:2524-2551` by two tests, so the table cannot drift
silently. Positions are 0-based [start, end] on the barcode read after singlet's swap.

| Protocol | STAR mode | Correction | Cell identity | Notes |
|---|---|---|---|---|
| 10x 3' v1/v2/v3/v4, CITE-seq GEX, ARC GEX, Quartz-seq2 | `CB_UMI_Simple`, whitelist | `1MM_multi_Nbase_pseudocounts` (R4-R14) | whitelist index | **On today** (phase 1a) |
| 10x 5' v2/v3/v4, ARC | same | same | same | TSO 3' adapter on the cDNA mate (existing `k_clip`) |
| Drop-seq, Seq-Well, CEL-seq2, MARS-seq2 | `CB_samTagOut`, `None` | none (R3, R15) | rank in the discovered barcode list | N-free raw CB; see section 3 |
| Visium | same as Drop-seq, geometry of 10x v3 | none | rank in the discovered list; spot row from `tissue_positions.csv` | list and coordinates are post-hoc metadata (section 4) |
| sci-RNA-seq3 | one segment: `CB_samTagOut`, whitelist | `Exact` | whitelist index | CB at 24..33, UMI 16..23 |
| BD Rhapsody | 3 segments `0_0_0_8`, `0_21_0_29`, `0_43_0_51`, UMI `0_52_0_59` | `Exact` per segment | R20 mixed radix (CLS1, 2, 3 lists) | linkers are not checked |
| SPLiT-seq / Parse | segments `0_10_0_17`, `0_33_0_40`, `0_56_0_63`, UMI `0_0_0_9`, barcode read is R2 | `Exact` | R20 | linkers not checked; Rd3 is segment 0 |
| inDrop v1/v2 | **W1-adapter route (decision)**: segments `0_0_2_-1`, `3_1_3_8`, UMI `3_9_3_14`, adapter `GAGTGATTGCTTGTGACGCCTT`, at most 1 adapter mismatch | `Exact` | R20 with per-length BC1 lists (`wlAdd`) | replaces today's fixed `0_0_0_9`/`0_32_0_39`/`0_40_0_45`; see section 7a |
| Microwell-seq, SureCell | 3 segments, one shared list | `Exact` | R20 | |
| ddSEQ | 3 segments, **no whitelist in the table** | `Exact` on discovered barcodes | concatenated raw segments, then discovery | the weakest-specified row |
| STRT-seq, 10x ATAC, DNBelab C4 | outside this layer | | | no CB/UMI read in R1 (ATAC), or not routed through these paths; not verified |

## 3. The no-whitelist path

STAR itself only checks N (R3). The cell set comes from singlet's Phase 0.9 (`singlet.cpp:5073`, 5160-5255):
barcodes with at least `bc_min_reads` raw reads, **5 on the unsorted stream, 100 on the sorted path**, written to
`auto_barcodes.tsv`; the pileup then keeps only reads whose CR is in that list.

- Stream order is count descending, ties by barcode ascending (5165-5190), so the rank is reproducible on a GPU with a
  sort and a run-length pass (`discover_barcodes`).
- The sorted path orders ties with `std::sort` (unspecified). **The sorted-path cell order cannot be matched.** Gate (a)
  must compare cell *sets* and per-barcode counts, not indices, for these protocols, or 1c must require the stream path.
- The count is over raw reads in the `.1fq` barcode dictionary (including reads with a bad UMI). The stream path says
  "no sorted BAM, so CB from CR" and refuses a missing whitelist file (`singlet.cpp:5041-5047`): that refusal is the
  guard that 1c lifts.

**Gate (a) for these protocols (decision):** cell *sets* plus per-barcode counts. Concretely, at `bc_min_reads` 5 on the
stream path: (i) the set of barcodes that reach the threshold is identical; (ii) for every barcode, the raw read count and the
count of reads that pass the N-free test and the UMI rule are identical; (iii) after the existing reducers, each cell's UMI
count per gene matches when cells are keyed by barcode string, not by index. Cell *index* order is checked only on the
stream path (count descending, ties by barcode ascending), where it is deterministic. The sorted path is not gated on order.

## 4. Visium

- singlet has **no spot whitelist** (`protocol.h:181-183`, licensed list not bundled). The cell set is discovered as in
  section 3; `tissue_positions.csv` is only joined afterwards.
- `VisiumSpatialParser::load_positions` (`visium_spatial.h:52-99`): a header is skipped if the first non-empty line
  contains `barcode` or `array_row`; CR stripped; rows under 6 fields are skipped; `in_tissue` is the field equal to `"1"`;
  a repeated barcode keeps the **last** row; `std::stoi`/`stof` throw on non-numbers (the process aborts). Output rows are in
  engine barcode order (`singlet.cpp:8631-8660`), zeros for a barcode not in the file. Reference: `load_positions`,
  `spot_table`.
- So 1f's cell-key work for Visium is the 1c no-whitelist path plus a join. The join is small (`k_bl_spot_gather` is a
  binary search); the coordinates stay host-side metadata.
- If a spot list is ever supplied as a true whitelist, it becomes a `Simple`/`Exact` case, which is not what singlet does.
  Do not change that inside 1f without a decision.

## 5. Feature barcodes

`singlet.crispr` does not exist as a Python module on `main` or here: the guide and tag matching are C++
(`crispr_guide.h`, `adt_matcher.h`, `adt_counter.h`); `python/singlet/celllabels/guide.py` is unrelated. Rules:

- **ADT / HTO** (`singlet.cpp:8466-8590`, `adt_matcher.h`):
  - the cell barcode is `R1[0:bc_len]` and must be in the called-cell set **exactly** (no 1MM correction) or the read is skipped;
  - the UMI is `R1[bc_len:]`, a raw string;
  - the tag comes from R2. Tag length is the first tag's. For offsets 0..5 (`min(5, len - tag_len)`), exact lookup first, then a
    Hamming-1 table. Hamming-1 neighbours that are themselves tags are not stored; a neighbour shared by two tags is -2 and
    **returns at once**, even if a later offset would match exactly (the -2 is kept distinct from -1 in the reference).
- **Molecule rule (decision): R9.** Per (cell, feature), two UMIs of the same length are joined when exactly one base differs
  and the larger read count is at least `2 * smaller - 1`; a molecule is a connected component; N is a base of its own;
  UMIs over 21 bases match exactly; a read without a UMI is a molecule. Reference: `r9_molecules`, `adt_count`; the CPU
  source is `umi_dedup.h` (`umi3_directional_roots`, `UmiFeatureCollapse`). Boundary to remember: equal counts join only at
  1 vs 1; 3 vs 3 stay two molecules. Today's `AdtCounter` counts distinct UMI strings (kept as `adt_count_exact`); for UMIs one
  base apart with counts 2 and 1 the two rules give 2 and 1. **The CPU counter must be changed to R9 before gate (a) can
  compare ADT counts**; that change is not made here.
- **CRISPR guides (decision; both old behaviours were bugs):**
  - the guide is matched as an exact substring of the upper-cased read **in sequencing orientation**. A BAM record of a
    reverse-strand alignment holds the reverse complement, so it is reverse-complemented back first (any non-ACGT base
    becomes N);
  - a read in which two or more distinct guides occur (a guide nested in another, or one sequence listed twice) is
    **ambiguous (-2) and not counted**. The result depends on the read and the library only, never on hash order.
    Consequence: in a library where one guide is a substring of another, every read of the longer guide is ambiguous;
    that is a library design problem, now visible instead of silently assigned;
  - implemented in C++ on branch `fix/crispr-guide-orientation` (from `main`, no PR): `GuideRef::match` returns
    `kAmbiguous`, `GuideRef::match_bam` does the orientation, both pileup call sites (`pileup_engine.h`, sequential and
    parallel) use it, tests 8 and 9 added to `test_crispr_guide.cpp` (all 9 tests built and passed with Rtools g++;
    `pileup_engine.h` itself needs htslib and was not compiled). Reference: `match_guide`, `match_guide_bam`, `revcomp`.
- Hand-off guard for CRISPR: `singlet.cpp:5060` requires `config.guide_ref_path.empty()`.

## 6. Test results

`python -P -m pytest tests/python/test_barcode_layer_ref.py -p no:cacheprovider --noconftest` gives **31 passed**. Python 3.11.9,
numpy 2.2.4, numba 0.64.0. Fixtures:

| Class | Tests assert |
|---|---|
| Simple (10x) | exact; unique 1MM; no match; two Ns (-2); Exact type not correcting; one-N single hit (cbMatch 1) and two hits with and without `Nbase` (2 vs -3); candidate order and the quality kept; posterior accepted at 100 vs 1 and rejected at 10 vs 1; quality overriding counts (fewer reads, Q10 vs Q33); `oneExact` drop for `1MM_multi` without pseudocounts; N and homopolymer UMIs, and that they stay out of the exact histogram; short read padding and the fatal fixed-length case; whitelist entries with N |
| No whitelist / samTagOut | any N-free CB accepted, N gives -2; discovery order and ties; `CB:Z` for a hit, `-`, and the raw CB; no UMI check |
| Complex (BD, anchors) | mixed-radix index 1+3·2+9·1 = 16; Exact vs 1MM; -12 for two segments; -3 inside a segment; N under Exact (-1) and 1MM; weak UMI check (poly-A only); `pos[1] == size` one-short UMI; -22 and -11; inDrop W1 route built from the protocol row with BC1 lengths 8-11 (today's fixed window failing on them), the shipped 8-mer-only list giving -11 for longer BC1, adapter anchors with BC1 lengths 8/9/10 and `wlAdd`; adapter 1 vs 2 mismatches (-21); all-mismatch adapter position; read-end anchor |
| Visium | header and CR handling, malformed rows, last row wins, zeros, headerless first line, `stoi` throw |
| Feature barcodes | exact, Hamming-1, ambiguous (-2, not -1), offset window 5 vs 6, early ambiguous return, exact-only cell; R9 vs exact-string counting; R9 boundaries (1 vs 1 joins, 3 vs 3 and 4 vs 3 do not, 5 vs 3 does, chain, N, length, over 21 bases, no UMI, order-free); guides: nested and duplicate sequences ambiguous, reverse-strand BAM sequence restored |
| Numba vs Python | 6000 random reads over a 360-entry list with planted neighbours: the numba two-pass version equals the plain-Python path on cell, status class and the exact histogram; both accepted and rejected posteriors occur |
| Tables | protocol geometry equals `protocol.h`; complex positions equal `sra_encoder.h` (inDrop compared on its `segs_current`, since the row now holds the W1 route) |

What the tests do **not** show: that this matches STAR on real data. No STAR binary ran. They assert the outcome that the
cited lines produce when read carefully, nothing more. W11's first job is the real comparison (section 8).

## 7. Guards 1c and 1f would lift

All are in files this task did not edit.

| File:line | Guard today | 1c / 1f change |
|---|---|---|
| `src/pipeline/singlet.cpp:5039` | `detected_soloType != "CB_UMI_Simple"` sends the run to the sorted BAM | allow `CB_samTagOut` and `CB_UMI_Complex` (and the concat form) once the layer is gated for that protocol; keep a per-protocol allow-list rather than dropping the test |
| `singlet.cpp:5041-5043` | no whitelist file means sorted path | allow when Phase 0.9 discovery ran; require `stream_unsorted` so the order is deterministic (section 3) |
| `singlet.cpp:5034` / `config.cb_from_cr` | CB from CR assumes whitelist-exact `.1fq` barcodes | for no-whitelist protocols CR is the raw N-free barcode; the cell comes from the discovered list, not from CR identity. Complex: the cell needs the composite index (R20), not CR |
| `singlet.cpp:5060` | hand-off only if `config.guide_ref_path.empty()` | 1f: lift for CRISPR only after the feature reducer exists; guides need the BAM-orientation sequence |
| `singlet.cpp:6657-6812` and 7309-7316 | solo argument selection, `Exact` for complex | unchanged; the layer must follow this selection exactly, not infer it |
| `SeqCacheGpuPrefill.cpp:337` | layout flag `type == CB_UMI_Simple` | pass the sam-tag/complex flag and the per-segment table in the `lay` block (`HandoffReadEntry` hand-off file, magic `SGHOFF02`; bump the version) |
| `SeqCacheGpuPrefill.cpp:379` | per-read entries only for `CB_UMI_Simple`, `cbL <= 21`, `umiL <= 21` | extend to samTagOut. Complex needs a different `cr3`: `ReadAlign_outputAlignments.cpp:363-368` packs `cbSeq`, and for Complex `cbSeq` is `seg_seg_seg` with `_` (it would pack as N and cut at 21 bases). Carry the composite index instead |
| `SeqCacheGpuPrefill.cpp:222` | `readNmates != 1` skips the whole GPU path | paired-end plates are not part of this task; BD, SPLiT, inDrop are single-end cDNA with a separate barcode read, so this guard stays |
| `ReadAlign_outputAlignments.cpp:356-389` (`handoffReadEntry`) | builds `cr3`/`ur3` from `readBar->cbSeq`/`umiSeq` | works for samTagOut as is; Complex needs the index (above) |
| `include/singlet/pileup/pileup_engine.h:802-815` | guide match on the BAM record | 1f: the GPU feature reducer reads the same sequence |

### 7a. inDrop: cell identity changes

`protocol.h:105-112` says the inDrop row is **v1/v2 with BC1 of 8-11 bases and a W1 linker**, so by the decision the W1-adapter
route is used (STAR's own inDrop recipe: `--soloCBposition 0_0_2_-1 3_1_3_8`, `--soloUMIposition 3_9_3_14`,
`--soloAdapterSequence GAGTGATTGCTTGTGACGCCTT`, `--soloAdapterMismatchesNmax 1`). `include/singlet/fq/indrop_detect.h`
describes other layouts (an 8+8 v1 without linker, a split-read v2, a linker v3); those are separate cases, not in the table
row, and are untouched here. What changes in cell identity:

- BC1 is taken from the read start to the base before the best W1 hit instead of a fixed 10 bases. Reads whose BC1 is 8, 9 or
  11 bases now get a barcode (a fixed 10-base window cuts or overruns them and matches nothing).
- BC2 and the UMI follow the adapter end, not absolute offsets 32 and 40; a read with 2 or more W1 mismatches gets -21.
- The cell index uses per-length BC1 lists: `wlAdd[len]` shifts the index of longer barcodes (R20).
- **Blocking data gap:** `python/singlet/data/whitelists/indrop_bc1.txt` holds 384 entries, **all 8 bases** (`indrop_bc2.txt`
  also 384 8-mers). The real BC1 set is variable length. With the shipped file the W1 route matches only 8-base BC1 reads and
  gives -11 for the rest (tested). The BC1 list with its true lengths has to be installed before 1c gates inDrop. I did not
  write one from memory.
- Singlet code that has to follow: `sra_encoder.h:2543-2544` (positions and adapter in the metadata JSON); the
  `use_complex_concat` test at `singlet.cpp:4191` (it requires anchor type 0, so the W1 route is native `CB_UMI_Complex`, not the
  concat form); `--soloAdapterSequence` and `--soloAdapterMismatchesNmax`, which `singlet.cpp` never passes today.

## 8. What W11 must build and gate

Build, in this order (each is a small step on the existing `k_clip` / hash / sort machinery):

1. Compile `barcode_layer.cuh` with `-fmad=false`; fix what a first nvcc pass finds (expect typos; none of it has been built).
2. **Gate 0, Simple:** run `k_bl_*_simple` on GSM6470208 / pbmc1k and compare, read by read, `cell_key` and `status` with the
   existing `HandoffReadEntry` cr3/ur3 path and with STAR's CB tag. This is an identity check that already has a known answer.
   It validates the float posterior and the candidate order before any new protocol depends on them.
3. No-whitelist: `discover_barcodes` as sort + run-length + threshold; cell key by binary search. Gate on **cell sets and
   per-barcode read counts** against the CPU run (the CPU order is not reproducible on the sorted path).
4. Complex Exact: `k_bl_extract_complex` plus the per-segment length tables (`wl_off`, `wl_add`) built on the host by the same
   sort/unique as `build_segment_wl`. Gate cell index against the concat form's Cartesian-product list position.
5. 1f: ADT/HTO tag tables (host-built with the same algorithm as `build_index`) and the guide kernel.

Gate (a) conditions to state for each protocol class:

- Simple: identical `cell_key` and status for every read; identical exact histogram.
- No-whitelist: identical cell set at `bc_min_reads` 5, identical per-cell UMI counts after the existing reducer.
- Complex: identical cell index per read.
- ADT/HTO: identical (cell, tag) R9 molecule counts, against the CPU counter after it is changed to R9; identical (cell, tag, UMI) triples before the collapse.
- CRISPR: identical assignment for every read, ambiguous reads included (-2), against the fixed C++ matcher; reverse-strand reads compared in sequencing orientation.
- inDrop: identical cell index per read against native `CB_UMI_Complex` with the W1 settings (not against today's run).
- Visium and other no-whitelist protocols: the cell-set definition in section 3.

## 9. Which protocols are hardest to make exact

1. **Visium and any sorted-path no-whitelist run.** The list is discovered and the sorted path leaves tie order to `std::sort`.
   The gate is now defined (cell sets and per-barcode counts, section 3), but cell-index identity is still not achievable there.
2. **ddSEQ.** No whitelist in the table, three segments, and the route (discovery on the concatenated string vs native Complex)
   depends on which files exist at run time (`singlet.cpp:6683-6840`). I could not pin it to one rule. Still the weakest row.
3. **inDrop.** The route is decided, but the data is not there: the shipped BC1 list has only 8-mers, and cell identity changes
   for every read, so there is no CPU baseline until a native `CB_UMI_Complex` run with W1 settings exists.
4. **CRISPR guides.** The rule is now deterministic and implemented on the CPU side; what remains is that matching happens on
   the BAM after alignment, so the GPU needs the mapped read's sequence and strand.
5. **BD Rhapsody and SPLiT/Parse:** easy under Exact with fixed windows; the risks are the product-list size (BD 97^3 = 0.9 M,
   fine) and that linkers are never checked.
6. **The float posterior** is exact only if the kernel reproduces the float operations; a mismatch shows only near the 0.975
   boundary. Gate 0 covers it.
7. **10x on the stream path** (section 10, item 1): the CPU stream path does not apply the 1MM rules at all, so "exact" depends
   on which CPU result is the reference.

## 10. Answers to the open questions, and what remains

1. **Are `.1fq` barcodes whitelist-exact? No.** Traced in `include/singlet/fq/writer.h:285-299` and `protocol.h:1535-1565`. The
   writer builds a dictionary from the protocol whitelist and looks each read's barcode up **exactly**. A hit stores the
   dictionary index; a miss stores the sentinel index plus the **raw barcode** (`block_unknown_bc_`). Nothing is corrected or
   dropped at encoding; the Phase 2a-wl check in `sra_encoder.h:1210-1300` only measures a match rate on up to 10 000 probe
   spots to choose the read orientation. The multi-segment protocols (BD, SPLiT, inDrop, ddSEQ, Microwell, SureCell) skip the
   dictionary and keep R1 verbatim. So the comment "`.1fq` barcodes are whitelist-exact" (`singlet.cpp:5034`) does not describe
   the file. What the stream path does: STAR writes the raw barcode as `CR`; `inline_scan_cb_from_cr` (`pileup_engine.h:318-340`)
   rejects a CR with an N, or a UMI with an N or a homopolymer (R14), then looks the CR up **exactly** in the cell list
   (`bc_index_.lookup`). Consequence: **on the stream path the 1MM rules R4-R11 decide nothing**; a read one base away from a
   called cell is lost, where STAR's own CB (sorted path) would rescue it, and a read with one N in the CB is lost where STAR's
   `Nbase` rule would resolve it. I did not measure how many reads that is.
   **Decision needed:** for 10x, gate the GPU layer against (a) the CPU stream path (exact CR) or (b) STAR's CB. The kernels
   support both (`mm1 = 0` is the exact mode); gating each mode against its own CPU counterpart avoids the question.
2. **Design text:** replace "knee-based or bead-error correction" for the Drop-seq family with "none; discovered list" (an edit to
   `align-and-count-design.md`, not made here).
3. **R9 for feature barcodes: decided, in the reference.** The C++ `AdtCounter` still has to move to R9.
4. **`MATCH_CellRanger`:** defined in no build file I found, so the float posterior applies; confirm on the Clipper binary's flags.
5. **Concat form vs native Complex:** same cell identity by R20 if the product list is built segment 0 fastest; the product
   file's order was not checked. The inDrop W1 route cannot use the concat form (anchor type 2/3).
6. **`singlet.crispr` (Python) does not exist**; the fix went into the C++ matcher (section 5).
7. **New:** the inDrop BC1 whitelist with true lengths (section 7a).
8. **New:** `GuideRef::match` now returns -2; a caller testing `== -1` instead of `>= 0` would treat that as a hit. The two pileup
   call sites test `gid >= 0`; grep found no others.

## 11. Not done (as of the laptop draft; see section 12)

No STAR run, no Clipper job, no compile of the `.cuh`, no change to `pileup.cu`, `engine_gpu.cu`, `singlet.cpp` or `SeqCache*.cpp`.
`pileup_engine.h` on the CRISPR branch was edited but not compiled (no htslib here).


## 12. Measured results (phase 1c preparation run on Clipper, 2026-10-29)

Harness: `bench/barcode_layer/` (`bl_harness.cu` instantiates every kernel in `barcode_layer.cuh`; `run_class.py`, `run_features.py`,
`jobs/*.sbatch`). nvcc 12.4, `-arch=sm_90 -fmad=false -ccbin /usr/bin/g++`, one H100 NVL. Final run: job 792476 (commit e1cf4010; earlier
identical-code runs 792442, 792368). Real reads only; STAR is the pbench conda STAR 2.7.11b on GRCh38-2024-A with singlet's STARsolo settings
(jobs 792395 10x, 792415 the other classes). Whether that build defines `MATCH_CellRanger` was not checked; the float posterior agreed with it
on every read below, and a double-precision posterior changed 0 reads, so these data cannot tell the two apart.

**Compile.** First nvcc pass: no errors, no warnings. No CUDA change was needed. Added `k_bl_discovered_cell` (rank lookup for no-whitelist
protocols); discovery itself (thrust sort, reduce_by_key, threshold, rank) is in the harness.

**Kernel vs reference (every read unless noted).**

| Class (reads) | Kernel vs numba/Python reference | Kernel vs R11 `CbCorrector` (CPU) |
|---|---|---|
| 10x pbmc1k_v3, quality F (12 M) | cell and status identical, 12 M of 12 M; exact histogram identical; plain-Python path 249 197 of 249 197 | cell and status identical, 12 M |
| 10x pbmc1k_v3, sequencer quality (12 M) | same, 12 M; Python 250 144 of 250 144 | same, 12 M |
| Drop-seq (2.85 M) | discovered list and counts identical (91 660 barcodes at 5 reads); numpy 2.85 M; Python 200 000 of 200 000 (cell and status) | n/a |
| BD Rhapsody (1.39 M) | Python 221 972 of 221 972 (cell, status, UMI key and length) | n/a |
| SPLiT-seq, row geometry / STAR-recipe geometry (2.55 M) | Python 224 126 / 224 084, all identical | n/a |
| inDrop W1 route (0.89 M) | Python 258 377 of 258 377 | n/a |
| ADT (1 M real antibody reads, tags from the data + 2 synthetic) | 1 000 000 identical (956 087 matched, 681 ambiguous -2); 200 000 shifted reads identical | n/a |
| Guides (1 M real cDNA reads, 43 guides incl. duplicate, nested, 60-mer, strand-restored) | 1 000 000 identical (155 ambiguous -2) | n/a |
| Visium spot gather (3 500 cells, 4 992 spots) | 3 500 identical (3 000 found) | n/a |

10x status counts (F quality): exact 11 368 597, one 1MM/N hit 96 735, posterior accepted 189 573, posterior rejected 29 902, no match 306 460,
bad UMI 8 733. With sequencer quality: accepted 196 245, rejected 23 230. Kernel time (kernel only, no copies): extract 11 ms, resolve 11 ms,
histogram 15 ms for 12 M reads; complex extraction 10 ms for 0.9-2.5 M reads.

**Kernel vs STAR's own tags (STAR record per read, primary alignments, unmapped reads included).**

| Class | Records | CB identical | Differing reads | UR identical | UB vs raw UMI |
|---|---|---|---|---|---|
| 10x v3, F quality, 1MM_multi_Nbase_pseudocounts | 12 000 000 | 100 % (11 654 905 same cell, 345 095 both none) | 0 | 100 % | 99.393 % equal, 0.586 % one base off, 432 reads (0.004 %) more than one base |
| 10x v3, sequencer quality | 12 000 000 | 100 % (11 661 577 / 338 423) | 0 | 100 % | 99.393 % / 0.587 % / 432 |
| Drop-seq (CB_samTagOut, None) | 2 854 058 | 100 % (2 836 990 same barcode, 17 068 N) | 0 | 100 % | STAR writes no UB in this mode |
| BD Rhapsody (native Complex, Exact) | 1 388 819 | 100 % (701 758 / 687 061) | 0 | 100 % | 99.952 % equal, 335 one base, 2 more |
| SPLiT-seq, singlet's row (33-40, 56-63) | 2 546 590 | 100 %, but only 7 reads have a cell | 0 | 100 % | n/a |
| SPLiT-seq, STAR recipe (10, 48, 78) | 2 546 590 | 100 % (1 665 189 / 881 401) | 0 | 100 % | 99.964 % equal, 595 one base |
| inDrop W1 route | 891 865 | 100 % but only 17 reads have a cell | 0 | 100 % of 397 413 records with a UR | n/a |

Why UB differs from the raw UMI: STAR's `1MM_All` UMI collapse rewrites a UMI to its group's representative; the barcode layer carries the raw
UMI (as designed), so UB is not a layer output. UR, the raw UMI, is identical everywhere. In inDrop, 3 reads have a 5-base UMI in both STAR and the
kernel (R17: `pos[1] == size` passes and `substr` clips); this was first reported as a mismatch by a fixed-length decode in the harness, not a kernel
difference.

**Findings.**
1. Reference bug fixed: `simple_record` for the no-whitelist case returned a cell for a read with a bad UMI (and -2 as cell key for an N); the
   stream drops both (`inline_scan_cb_from_cr`). Test added (`test_nowl_record_drops_bad_umi_and_n_barcodes`), 32 tests pass.
2. **SPLiT-seq geometry in `protocol.h`/`sra_encoder.h` does not fit the real data tested** (GSE281535, SRR31302021): with BC2 at 33-40 and BC1 at
   56-63, 7 of 2 546 590 reads get a cell; with STAR's recipe positions 48-55 and 78-85 (30-base linkers), 65 % do. Both STAR and the kernel agree on
   each geometry, so this is the table, not the layer. Not changed here.
3. **inDrop**: on SRR15409581 only 17 of 891 865 reads get a cell. 494 442 fail the W1 search (the reads carry `GAGTGATTGCTTGAGACGACTT`-like
   linkers, two mismatches from W1) and the shipped BC1 list is 8-mers only (section 7a); STAR and the kernel agree on every read. The W1 route is
   therefore not validated on a positive example; it needs inDrop v1/v2 data with the canonical W1 and the real variable-length BC1 list.
4. Not covered: sorted-path no-whitelist order (not reproducible), ddSEQ, Microwell/SureCell (no lists/data tried), sci-RNA-seq3, native
   `CB_UMI_Complex` 1MM, a real Visium sample, real CRISPR guide libraries (guides here are 20-mers taken from real reads), and ADT tag tables from a
   real feature reference (taken from the data). ENA streams were cut early for several runs; mates were cut to equal, well-formed prefixes
   (jobs `fqcheck`), so Drop-seq is 2.85 M reads, BD 1.39 M, SPLiT 2.55 M.
