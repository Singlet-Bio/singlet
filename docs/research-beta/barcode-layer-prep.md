# Barcode layer preparation for phases 1c and 1f

Status: preparation only. Nothing here is switched on, compiled on a GPU, or run on a cluster. Written on the laptop from the
vendored STAR 2.7.11b sources in `src/star/` and singlet's own tables. Branch `claude/gpu-barcode-layer-484c70`.

Deliverables:
- `src/engine_gpu/barcode_layer_ref.py`: Python and numba reference, one function per STAR rule.
- `tests/python/test_barcode_layer_ref.py`: 28 tests, handcrafted reads of a few KB, all passing.
- `src/engine_gpu/barcode_layer.cuh`: CUDA drafts, **not compiled**. Not included from `engine_gpu.cu` or `pileup.cu`.
- This note.

Design section 4.1 (`align-and-count-design.md`) says the engine reproduces STAR's correction rules and does not replace
them. This note pins down what those rules are, as **singlet runs them**, which is narrower than what STAR can do.

## 1. What singlet actually asks STAR for

Three facts decide the scope of 1c. All three are in `src/pipeline/singlet.cpp`.

1. **Every complex or no-whitelist protocol runs with `--soloCBmatchWLtype Exact`** (7309-7316). Only `CB_UMI_Simple`
   gets `1MM_multi_Nbase_pseudocounts`. The Complex mismatch rules (-12, multi-match, EditDist_2) never run in
   production, so 1c does not need them for gate (a).
2. **No protocol uses an adapter or a read-end anchor.** `--soloAdapterSequence` is never passed (grep: no hit).
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
| R18 | Adapter (not used by singlet): first position with the fewest mismatches below `adapterMismatchesNmax` (default 1), `N` in the adapter is a wildcard, ties keep the first, -21 on failure. If every base mismatches, the position stays 0 | :332-339; SequenceFuns.cpp:352-371; parametersDefault:828 | `local_align_hamming_dist` |
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
| inDrop | segments `0_0_0_9`, `0_32_0_39`, UMI `0_40_0_45` | `Exact` | R20 | fixed BC1 length, no W1 anchor |
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

## 5. Feature barcodes (ADT, hashtag, CRISPR)

There is no `singlet.crispr` Python module; the CRISPR and ADT code is C++ (`python/singlet/` has none). Rules:

- **ADT / HTO** (`singlet.cpp:8466-8590`, `adt_matcher.h`, `adt_counter.h`):
  - the cell barcode is `R1[0:bc_len]` and must be in the called-cell set **exactly** (no 1MM correction) or the read is skipped;
  - the UMI is `R1[bc_len:]`, taken as a raw string;
  - the tag comes from R2. Tag length is the first tag's. For offsets 0..5 (`min(5, len - tag_len)`), exact lookup first, then a
    Hamming-1 table. Hamming-1 neighbours that are themselves tags are not stored; a neighbour shared by two tags is -2 and
    **returns at once**, even if a later offset would match exactly;
  - counts are the number of **distinct UMI strings** per (cell, tag). That is exact-string dedup, not R9's directional rule.
- **CRISPR guides** (`crispr_guide.h:106-116`, `pileup_engine.h:802-815`):
  - the guide is looked up as an exact substring of the BAM query sequence of a mapped record, after upper-casing. It runs in the
    BAM pileup after alignment, not before; the cell is the BAM's CR;
  - the CPU loop walks an `unordered_map`, so with two guides that both occur in a read (one guide inside another) the winner
    depends on hash order. A guide sequence listed twice keeps the last id;
  - a reverse-strand record carries the reverse-complemented sequence in the BAM, so a guide written in read orientation does
    not match it. The GPU kernel must see the same orientation.
- Hand-off guard for CRISPR: `singlet.cpp:5060` requires `config.guide_ref_path.empty()`.

## 6. Test results

`python -P -m pytest tests/python/test_barcode_layer_ref.py -p no:cacheprovider --noconftest` gives **28 passed**. Python 3.11.9,
numpy 2.2.4, numba 0.64.0. Fixtures:

| Class | Tests assert |
|---|---|
| Simple (10x) | exact; unique 1MM; no match; two Ns (-2); Exact type not correcting; one-N single hit (cbMatch 1) and two hits with and without `Nbase` (2 vs -3); candidate order and the quality kept; posterior accepted at 100 vs 1 and rejected at 10 vs 1; quality overriding counts (fewer reads, Q10 vs Q33); `oneExact` drop for `1MM_multi` without pseudocounts; N and homopolymer UMIs, and that they stay out of the exact histogram; short read padding and the fatal fixed-length case; whitelist entries with N |
| No whitelist / samTagOut | any N-free CB accepted, N gives -2; discovery order and ties; `CB:Z` for a hit, `-`, and the raw CB; no UMI check |
| Complex (BD, anchors) | mixed-radix index 1+3·2+9·1 = 16; Exact vs 1MM; -12 for two segments; -3 inside a segment; N under Exact (-1) and 1MM; weak UMI check (poly-A only); `pos[1] == size` one-short UMI; -22 and -11; inDrop-style adapter anchors with BC1 lengths 8/9/10 and `wlAdd`; adapter 1 vs 2 mismatches (-21); all-mismatch adapter position; read-end anchor |
| Visium | header and CR handling, malformed rows, last row wins, zeros, headerless first line, `stoi` throw |
| Feature barcodes | exact, Hamming-1, ambiguous (-2, not -1), offset window 5 vs 6, early ambiguous return, exact-only cell, exact-string UMI dedup; guide substring, two matching guides (`[0, 2]`), repeated sequence |
| Numba vs Python | 6000 random reads over a 360-entry list with planted neighbours: the numba two-pass version equals the plain-Python path on cell, status class and the exact histogram; both accepted and rejected posteriors occur |
| Tables | protocol geometry equals `protocol.h`; complex positions equal `sra_encoder.h` |

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
- ADT/HTO: identical (cell, tag, UMI-string) triples; only then consider R9.
- CRISPR: identical assignment on reads with one matching guide; report the ambiguous reads separately.

## 9. Which protocols are hardest to make exact

1. **Visium and any sorted-path no-whitelist run.** The cell list is discovered, and its order depends on a threshold and a
   tie rule that the sorted path leaves to `std::sort`. Per-read identity is achievable; cell index identity is not. This
   is a gate-definition problem, not a kernel problem.
2. **ddSEQ.** No whitelist in the protocol table, three segments, and the route (discovery of the concatenated string vs
   native Complex) depends on which files exist at run time (`singlet.cpp:6683-6840`). I could not pin it to one rule.
3. **inDrop.** STAR can find BC1 by the W1 adapter, and the barcode length varies (8-11). singlet uses a fixed window. The
   reference supports anchors and variable-length lists and tests them, but any future move to the adapter route changes
   cell identity, so it is a decision, not a drop-in.
4. **CRISPR guides.** Hash-order ties on the CPU; BAM orientation; runs after alignment on the BAM, not on the read.
5. **BD Rhapsody and SPLiT/Parse** are mechanically easy under Exact with fixed windows. Their risk is the product list size
   (BD 97³ is about 0.9 M, fine) and that linkers are never checked, so a read with a shifted linker is read as a valid
   barcode window.
6. The float posterior is exact only if the kernel reproduces the float operations; a mismatch shows up only on reads near
   the 0.975 boundary. Gate 0 covers it.

## 10. Open questions for the team

1. **`.1fq` barcode handling.** `singlet.cpp:5034` (and the comment above it) says `.1fq` barcodes are "whitelist-exact" and the pileup takes CR.
   If so the 1MM posterior never decides a cell on that path today, and Gate 0 compares against STAR's own CB. I did not trace
   where `sra_encoder.h` validates or corrects against the whitelist (only a match rate near 1210 is visible). Someone who knows
   the encoder should say which it is before 1c relies on R7-R11 for 10x.
2. **Design text.** Replace "knee-based or bead-error correction" for the Drop-seq family with "none; discovered list".
3. **R9 and feature barcodes.** The design table makes the feature reducer's molecule key the UMI under R9's directional rule;
   today's ADT counter uses exact-string dedup. These give different counts. Decide before the gate which one the CPU
   reference is (I assumed today's).
4. **`MATCH_CellRanger`.** I found it defined nowhere in the build files, so the float path applies. Check the actual compile
   flags of the binary on Clipper.
5. **Native Complex with per-segment lists** (`singlet.cpp:6683-6812`) versus the concat form: same cell identity by R20 when
   the product list is built in segment order with segment 0 fastest. The concat product-file ordering was not checked
   against R20.

## 11. Not done

No STAR run, no Clipper job, no compile of the `.cuh`, no change to `pileup.cu`, `engine_gpu.cu`, `singlet.cpp` or `SeqCache*.cpp`.
