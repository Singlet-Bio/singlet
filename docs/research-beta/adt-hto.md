# ADT / HTO modality: research-beta handoff (2026-10-26)

Branch `claude/adt-modality-0637a6`, PR #29 (draft, not merged). Design and QC report: `docs/adt-modality.md`;
format: `docs/format-v2.md` §13. Nothing has been written to R2 or D1.

## What exists

- **Counter** `python/singlet/adt/count.py`: Cell Ranger feature-barcoding rules.
  - Feature barcode exact or unique Hamming-1, with the offset measured per library.
  - Cell barcode corrected by the 1-mismatch posterior ≥ 0.975.
  - Cell Ranger `correct_umis`.
  - 3' v3 TotalSeq-B translation, with its orientation chosen from the reads.
  - Vectorised FASTQ reader (pigz + numpy).
  - Large libraries are sharded: one pass over the reads, collapsed keys spilled, then re-sharded by corrected barcode. The result is identical to the single pass, by test and on real data.
- **Panels** `panel.py`: study references, a TotalSeq catalog of 647 sequences and a GEO-harvested tier of 904.
  - Discovery from the reads when no reference is deposited.
  - Unresolved, ambiguous or reused sequences are flagged, never guessed.
- **QC** `qc.py`:
  - per library: panel mapping, UMIs per cell, empty/isotype background, RNA concordance, saturation, panel completeness, with pass/warn/fail;
  - per cell: totals and isotype ratio.
- **Normalisation** `normalize.py`: DSB and CLR, as derived layers. Raw counts are the stored truth.
- **Hashtags** `hto.py` `call_hashtags`: hashedDrops rule plus a mixture doublet veto, method version `singlet-adt-hto/2`.
- **Storage** `store.py`: members `adt_counts.1pz`, `hto_counts.1pz` and `adt_panel.json`, aligned to the GEX `col`; the HTO sidecar Parquet goes under `v2/cf/`.
- **Clients**: `client.py` and `v2.R` know the modalities `adt` and `hto`.
- **Tests**: `tests/python/test_adt_count.py`, `test_adt_qc_store.py`, `test_hto_methods.py`.
- **Inventory**: 3,795 rows in `bench/adt/inventory/adt_hto_inventory.tsv`.

## Results so far

**Counting vs Cell Ranger 3.0** (pbmc_1k_protein_v3):

| Counter | UMI ratio | Pearson r (log counts) | Identical entries |
|---|---|---|---|
| Singlet | 1.0002 | 0.999995 | 95.5% |
| kite | 1.029 | 0.99956 | — |
| `adt_counter.h` (old C++ counter) | 0 UMIs (broken) | — | — |

- **Public ADT benchmark** (pbmc1k, pbmc10k, pbmc5p): Singlet passes 4 of 4 metrics on pbmc1k against kite. kite and alevin-fry ran on all three datasets.
- **CITE-seq-Count**: it failed on all three, because 1.4.5 crashes on newer pandas. The env now pins pandas 1.3.5, and reruns are queued (below).
- **Singlet runtime** on pbmc10k: 906 s wall, 0.27 CPU-h, 10.7 GB. kite takes 269 s. The counter is effectively single-threaded.
- **HTO on Howitt 2023** (genotype truth): 2 of 8 metrics pass. Mean F-scores: BAL1 0.919, BAL2 0.814, BAL3 0.752, CL 0.742. The best competitors reach 0.935, 0.846, 0.785 and 0.874.
  - The caller was selected leave-one-dataset-out on these same data, so the results are not fully held-out.
  - `bench/public/results/hto.json` is the 1.0.0 version. The 1.1.0 rebuild is queued.
- **Pilot** (12 studies, 32 libraries):
  - 18 ok: 11 pass, 6 warn, 1 fail;
  - 6 `panel_unresolved`;
  - 6 `gex_empty`;
  - 2 `reads_too_short`.
  - 17 objects, 3.39 MB in total.
- **Full atlas run** (as of 2026-10-26): 2,493 library outputs under `$WORK/data/adt/`.
  - Jobs: 2,502 tasks completed, 12 running, 1 timed out (`782519_32`).
  - Still queued: `782521_[692-899]`, `782522` and the retry `785348`.

## Clipper (WORK=/mnt/projects/debruinz_project/debruinz/singlet)

- **Code**: `$WORK/src/singlet-py/claude__adt-modality-0637a6`, synced to `8b864f9`. Envs: `$WORK/envs/adt`, `adt-bench`, `adt-csc` and `adt-afry`.
- **Atlas outputs**: `$WORK/data/adt/<GSE>/<GSM>/`. State and lists are in `$WORK/state/adt/`; logs are `$WORK/logs/adt-run-*`.
- **Benchmark artifacts**: `$WORK/bpub/artifacts/{adt,hto}`. Documents go to `$WORK/state/adt/results/{adt,hto}.json`.
- **Jobs left running on purpose** (they finish on their own):
  - atlas arrays 782521, 782522 and retry 785348 (`%12` each);
  - ADT benchmark rerun 787491 (all tools, now with `cpu_h`);
  - HTO rerun 787492, then document build 787493;
  - CITE-seq-Count env rebuild 787552, then reruns 787553 (`TOOLS=citeseqcount`), then document build 787554.

## How to finish

1. **Read the document builds.** Read `$WORK/logs/adt-py-787554.out`. It holds the validation against bench-cohort's contract 1.1.0, from the bare-repo branch `claude/bench-public-cohort-485cf7` (PR #40). Then copy `state/adt/results/{adt,hto}.json` to `bench/public/results/` on that branch, with `bench/adt/public_{adt,hto}.py` as `bench/public/capabilities/{adt,hto}.py`. Use `_common.contract()` in place of `_contract()`.
2. **Retry the timed-out atlas task.** Retry `782519_32` and any failures in `785348`, from a list built with `bench/adt/hpc/run_array.sbatch` (`LIST=`).
3. **Build objects and QC.**
   - Run `bench/adt/reqc.py`, to apply the current QC rules and hashtag caller to every output.
   - Run `bench/adt/assemble.py --build-objects`, then `bench/adt/report.py` for the figures.
   - Fill the `<!-- RESULTS -->` and `<!-- FULL -->` placeholders in `docs/adt-modality.md`.
4. **Prepare the upload list.** List the exact R2 keys, sizes and sha256 for Zach. Members are added to each sample's `.s2` (a new object version), and the HTO sidecars go to `v2/cf/<GSM>/hto.<digest>.parquet`. Zach does the upload; there are no agent writes.
5. **Open decision (Zach):** keep the HTO results as honest failures, or develop a new hashtag caller that is judged leave-one-dataset-out.

## Lessons

- **Clipper:**
  - `clipper.py sbatch` ignores partition overrides. Submit with `clipper.py run "sbatch ..."` instead.
  - sbatch copies the script at submit time, so sync the worktree before submitting, not after.
  - Never run python on the login node.
- **Downloads:** SRA through fasterq-dump is about 240 MB/s; ENA is about 1 MB/s and resets connections.
- **Memory:** byte-string `np.unique` and `encode` are the memory and time hot spots. Encode column by column, and shard large libraries.
- **Panels:** many GEO ADT studies deposit no feature reference.
  - Read-based discovery recovers most panels. Older TotalSeq-B sequences (the 10x 2018 panels) were reassigned by BioLegend, so catalog names must not be trusted for them.
  - 6 of 32 pilot libraries stay unresolved, and that is reported rather than guessed.
- **Benchmarking:** tools must be pinned to old dependency versions in their own envs. CITE-seq-Count 1.4.5 needs pandas < 1.5.
