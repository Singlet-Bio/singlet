# `.singlet` compression study: report

Executive summary and index, 2026-10-22. The full synthesis, with every line of reasoning, the per-file frontier
table and the status of each experiment, is [`notes/frontier_fable.md`](notes/frontier_fable.md). Derivations are in
[`notes/theory_fable.md`](notes/theory_fable.md). The study design is in [`PLAN.md`](PLAN.md).

## Findings

1. **Most of a `.singlet` file is redundant.** Feature names, row permutations and derivable blocks are 82.9% of the
   187 MB study bundle. 77.9% of each sample's bytes regenerate byte-for-byte from the reference and other members.
   Removing them needs no new codec: 14 corpus matrices go from 114.4 MB to 53.5 MB, and the bundle shrinks about 4.5x.
2. **The count payload can shrink 15-20%, and not much beyond that.** The best bit-exact coder is 15.1% below today
   byte-weighted (11-24% per file). Three independent designs agree within about 0.1 bits per nonzero. Conditioning a
   cell on its neighbours or on its own other half saves only 0.04-0.12 bits per nonzero on typical files.
3. **Recursive compression finds nothing.** A second pass over near-entropy outputs returns 0.976-1.004, and choosing
   the best coder per chunk gains at most 0.18%. What remains is sampling noise: which UMIs were captured, where each
   fragment landed in its gene, and which ambient molecules fell into empty droplets.
4. **Decode speed is limited by layout.** A 32-lane AVX-512 rANS decoder reaches 73 Mnnz/s measured on a loaded laptop
   (about 172 at full clock) in its native row order. Sorting back to row-id order halves that. Two sort-free layouts
   cost no size; the fast decoder for them is not yet written.
5. **Layout order is the latency lever.** Names at the end of each file and the chunk table behind the permutation
   force 10 requests before the first cell. Fixing that cuts time to the first 1,024 cells at 100 Mbit/s from about
   1.3 s to about 0.4 s on a typical nuclei sample.

Projected sizes combine separately measured components; no single codec implementing all of them was built.

| object | today | projected |
|---|---|---|
| 14 corpus matrices | 114 MB | about 46 MB (2.5x) |
| GSE145370 bundle | 187 MB | about 37-41 MB (4.5-5x) |
| all called cells, one sample | | 2.3-22x fewer bytes |

## Recommended format changes

1. Container v2: vocabulary id instead of names, per-row counts instead of the permutation, no colsums or pointer
   block, whitelist-index barcodes, chunk index first, derived members recomputed with pinned recipes.
2. Static-table rANS payload (about −15%) with a sort-free row order.
3. Empty droplets as a second member, coded against the ambient profile.
4. Columns in whitelist order, and identity defined by a digest of the decoded matrix rather than container bytes.

## Pipeline issue found

The shipped `splice_psi.1pz` files hold PSI rounded to 0 or 1, while the in-tree writer truncates. Readers who use PSI
as a fraction get a binary indicator. Details are in [`notes/e4.md`](notes/e4.md).

## Status

The study ended when the account's weekly usage limit stopped all agents. Tracks A-C, Ideas 1 and 3, E1 and E4 are
complete. Track D is substantially complete. Idea 2, Idea 4, E2 and the within-cell hub-gene experiment are partial.
E3 and E5 were not run. Every agent's files are on disk; each note says how to resume.

## Engineering status and reproduction (original session)

This section is written by the session that designed the study and ran tracks A-D and ideas 1-4; the synthesis above
and in `notes/frontier_fable.md` is by its fork. It records what exists as code, how each headline number is
reproduced, how much of it is verified, and what is still projection.

### What exists

`bash build.sh` compiles every driver under `harness/` and `kernels/` with the Rtools 4.4 GCC toolchain (no MSVC or
CMake). `bin/bench.exe --list` shows 40 registered codecs; every one is checked by the harness for
`decode(encode(M)) == M` on indptr, indices, data, row names, column names and key-values, and timed single-threaded.

| codec family | header | what it is |
|---|---|---|
| `ref_1pz_v1_L3`, `ref_zstd_csc_L*` | `kernels/ref_zstd_csc.h` | today's writer re-emitted; naive zstd over raw CSC |
| `a_names_as_model`, `a_perm_from_rowcounts`, `a_names_perm_model`, `a_vocsc_*` | `kernels/track_a.h` | container redundancy removal; entropy-coder sweeps on today's transform |
| `b_rans16`, `b_rans16_scalar`, `b_leb_*` | `kernels/track_b*.h` | 32-lane AVX-512 rANS over rank-space gaps, static position x depth tables; scalar twin is bit-identical |
| `c_r1`, `c_r1b`, `c_r2`, `c_r3` | `kernels/track_c*.h` | adaptive binary survival coder, integer-only decode, 1 MB out-of-sample model |
| `d_cm_*` | `kernels/track_d.h` | logistic-mixing context-mixing byte coder used as the strongest second pass |
| `idea1_*`, `idea2_*`, `idea3_*`, `idea4_*`, `w3_coexp*` | own headers and drivers (`harness/idea*_bench.cpp`) | the brainstorm wave; see each note |

### Reproducing the headline numbers

```bash
cd bench/compression && bash build.sh
# container redundancy: 114.4 MB -> 53.5 MB on the 14 matrices (Track A, notes/track_a.md section 7)
./bin/bench.exe --no-file-check --codec ref_1pz_v1_L3,a_names_perm_model --out results/repro_a.tsv corpus/*/*_counts.1pz
# payload frontier: -15.1% and 73 Mnnz/s native order (Track B, notes/track_b.md Part II)
./bin/bench.exe --no-file-check --codec b_rans16,b_rans16_scalar,b_leb_z3 --out results/repro_b.tsv corpus/*/*_counts.1pz
# adaptive model coder, leave-one-study-out (Track C, notes/track_c.md)
./bin/bench.exe --no-file-check --codec c_r2 --out results/repro_c.tsv corpus/*/*_counts.1pz
# entropy floors, recursion table, orchestrator oracle (Track D): python -P tools/track_d_*.py, results/track_d_*.tsv
```

### What is verified, and how far

- **Matrix round trips.** Track C: 98 (file, codec) pairs through its driver plus 42 through `bench.exe`. Track B: 14
  corpus files plus 280 runs over 20 edge-case matrices in `tools/track_b_selftest`. Idea 3: 14 files for the
  gene-then-interval codec and 7 sample pairs for the joint exon+intron codec. Track A: every codec on all 14 files.
  No failure is recorded anywhere in `results/`.
- **Coder overhead.** Coded size equals model cross-entropy within 0.0004 bits/nnz (Track C) and within 0.01 (Idea 1),
  so the frontier numbers are model numbers, not coder artefacts.
- **Container identity** is reproducible only with the zstd build that wrote the file (1.5.7 reproduces 87/87 frames on
  the brain and PBMC studies, 1.5.6 reproduces 72/72 on the T-cell and tumour studies). Identity must be defined on the
  decoded matrix; the harness does that.
- **Timing** was measured on a 4-core laptop shared by up to ten agents and clock-throttled to 0.8-1.4 GHz. Treat
  measured Mnnz/s as lower bounds and the 3 GHz estimates as within about 25%.
- **Not yet one codec.** The 2.5x and 4.5-5x figures in the summary add up separately measured components
  (Track A container removal, Track B payload, Idea 2 ambient empties, E4 derived members). A combined codec that
  measures them as one number is the first item of remaining work and is being built as `w3_total`.
- **Partial experiments.** Idea 2 finished stage 1 (its in-file prior lands on the cross-validated optimum; empties
  within 1-6% of the floor), Idea 4 finished the cross-entropy study (manifold prior +0.13 net on typical PBMC,
  negative on shallow cells), w3_coexp stopped before its stage-1 table, E2 has sizes but no fast decoder. Their notes
  say how to resume; E1's measurement makes the first three unlikely to change the frontier.

### Verdict on recursion, from the architect

The study set out to show that some endpoint is asymptotically incompressible while keeping bit-exact decode. Three
independent coder designs converged within 0.1 bits/nnz, a second pass over their outputs returns 0.99-1.00, and the
split-half bound shows at most 0.0-0.2 bits/nnz of further predictable structure in a cell. The residual is the
entropy of UMI capture, fragment position and ambient sampling: physics, not redundancy. The large factors (4.5x on a
bundle, 2 requests instead of 10 before the first cell) come from removing content that is a function of the
reference or of other members, which is the one place where "the shortest program that generates the data" is a
practical idea rather than a slogan.

### Next engineering steps

1. `w3_total`: one codec = vocabulary-id names + whitelist-index barcodes + per-row counts + `b_rans16` payload (+
   empties as a second member), measured on the 14 files and on every sample of the GSE145370 bundle.
2. Sort-free fast decoder: `b_rans16` in frequency-class row order, or E2's ascending-order streams on the 32-lane
   decoder, timed on an idle machine against the 200 Mnnz/s target.
3. Container v2 specification from items 1-2 plus the E4 derivation recipes, then a writer in `pz_writer.h` and a
   reader that accepts both versions.

## Index

| where | what |
|---|---|
| `notes/frontier_fable.md` | final synthesis |
| `notes/theory_fable.md` | derivations and the predictions-versus-measurements log |
| `notes/brainstorm_fable.md` | the original ideas and hypotheses |
| `notes/track_a.md` … `track_d.md` | accounting, transform kernels, learned model, entropy floor and recursion |
| `notes/idea1.md` … `idea4.md`, `idea4_pp.md`, `w3_coexp.md` | survival coder, in-file model, gene/interval factorisation, manifold prior, hub genes |
| `notes/e1.md`, `e2.md`, `e4.md` | split-half predictability, block-adaptive coder, derivability and latency |
| `plans/E1` … `E5` | investigation plans with context and final status |
| `results/` | every number (TSV); large binaries are git-ignored |
| `harness/`, `kernels/`, `tools/`, `build.sh` | the code; `bash build.sh` builds with the Rtools GCC toolchain |
