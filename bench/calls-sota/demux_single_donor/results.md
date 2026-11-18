# Single-donor samples and the donor demultiplexer: results (2026-11-11)

Gate: bench/calls-sota/gates/donor_single_split.md (committed before scoring). Verdict: **fails the gate on one of eight real single-donor libraries; withhold the donor label until that case is understood.** The in silico single-donor test passes on all eight donors.

## Test 1: real single-donor libraries, stored engine output (design truth, not genotype-confirmed)
Source: pilot/<GSM>/s1cpu_0ea48783/out/donor_assignments.tsv (engine build 0ea48783, 2026-11-01, auto-K). GSM8164511 also has a later rerun (s1cpu_r) with the same K. The four libraries with the most cells were rerun with the standalone harness against the frozen v1.3.7 donor_demux.h on the stored private allele counts (pilot_v137_split.json); results are identical to the pilot build.

| Library | Labeled cells | K found | False-split rate | Split declared |
|---|---|---|---|---|
| GSM5130155 (T cells, subject HD5, 38 y female) | 8,134 | 3 (4,352 / 3,040 / 742) | 46.5% | yes (two non-dominant donors) |
| GSM5130163 (T cells 5p) | 104,091 | 1 | 0 | no |
| GSM5171610 (T cells 5p TCR) | 51,305 | 1 | 0 | no |
| GSM7064234 (retina snMultiome) | 48,716 | 1 | 0 | no |
| GSM7884577 (retina snMultiome) | 59,826 | 1 | 0 | no |
| GSM8183378 (gastric epithelium) | 8,088 | 1 | 0 | no |
| GSM8164511 (HIO organoid) | 8,753 | 1 | 0 | no |
| GSM9038586 (MDA-MB-231 xenograft) | 8,680 | 1 | 0 | no |
| Pooled | 297,593 | | 1.27% | |

GEO lists GSM5130155 as one subject (HD5), so the K=3 call is a false split by design truth. Genotype doublets called: 28 of 8,167 cells. The same K=3 split appears on the CPU and GPU runs of build 0ea48783, on a read-capped run, and in the v1.3.7 rerun, so it is a property of the demultiplexer on these data and not a build or sampling accident. In addition, a read-capped exploratory run of GSM5130163 (s1cap, 108k cells) split into K=2 (28.8%) while its full runs give K=1; capped runs are not part of the gate and are listed here as a sign that K selection is unstable on some T-cell libraries.

Not diagnosed: why HD5 splits. Possible causes are a sample of mixed origin that GEO does not describe, strong clonal or cell-state structure in the allele counts, or an auto-K threshold that is too permissive at low cell numbers (8,134 labeled cells). The single-donor screen in the demultiplexer (screen_single_donor, switched on with SINGLET_DEMUX_SCREEN=1) is off by default and was not tested here; it may or may not fix this case.

## Test 2: in silico single donors with genotype truth (Kang 2018 batch 2 control lane)
Each of the eight donors alone (truth singlets, demuxlet identity), stored allele counts of the pool, auto-K, v1.3.7 header (blob aeef964e), standalone harness. File: kang_single_split.json.

| Donor | Cells | K found | False-split rate |
|---|---|---|---|
| 101 | 858 | 1 | 0 |
| 107 | 517 | 1 | 0 |
| 1015 | 2,738 | 1 | 0 |
| 1016 | 1,723 | 1 | 0 |
| 1039 | 461 | 1 | 0 |
| 1244 | 1,879 | 1 | 0 |
| 1256 | 2,108 | 1 | 0 |
| 1488 | 2,031 | 1 | 0 |
| Pooled | 12,315 | all 1 | 0 (0 of 12,315) |

The earlier committed single-donor subset of donor 1244 (bench/modalities/demux_fix, 1,852 cells) also gave K=1. Caveat: a subset lacks the ambient RNA and doublets of a true single-donor library and the cells are filtered to confident singlets, so this test is easier than a real library.

## Libraries with no stored data
Osorio single lines (GSM3596320 and GSM3596321) and the GSE230239 single lines (HG00362, HG01277) have no stored demultiplexer output or allele counts: the pilot copies of the LCL libraries were cleaned after processing. The cheapest route is the one the call-layer design already names: rerun the engine pileup on a read-capped slice (about 10 CPU-hours per library for the full run, a few for a 30M-pair slice) and run the standalone demultiplexer harness on the private allele counts.

## Runtime
The demultiplexer runs inside the engine export step. In this check it took 15 to 150 s per library on 8 threads (the 150 s case is 8,753 cells including matrix loading). It is therefore an engine cost, outside the post-freeze call budget of 2 core-minutes per ten thousand cells.

## Update 2026-11-16: MHC guard in the auto-K search (D-163)

Cause of the GSM5130155 split: 76 of the 98 well-covered SNPs where the three components disagree lie in the extended MHC (chr6:29-33 Mb); the three components differ at 3 to 5% of typed SNPs, against 20% or more between real donors of the Kang pool. Without the MHC SNPs the held-out likelihood gains 0.0004 nats per cell from a second donor (it gains 1.4 with them). The fix (`select_k_mhc_guarded`, donor_demux.h) repeats the K search without MHC SNPs when the first search finds several donors, and reports one donor when nothing outside the MHC supports a second. Gate files and bars are unchanged; the rerun uses the same stored allele counts and the standalone harness (`run_donorfix_eval.sbatch`).

| | v1.4.0 (6c559c88) | with the guard |
|---|---|---|
| GSM5130155, K found | 3 (46.5% false) | 1 (0%) |
| Other seven libraries | K = 1 | K = 1 |
| Pooled false-split rate, 8 libraries | 1.27% | 0% |
| Libraries with a split declared | 1 of 8 | 0 of 8 |
| Kang b1 lane A, ARI on truth singlets (registered scorer) | 0.9928, K = 4 | 0.9928, K = 4 |
| Multi-donor pilot pools (10-donor, hashed, two cell-line mixes): K and labels | | identical, ARI 1.0 between the two builds |

Counts: the eight gate samples run on the CPU engine with the candidate and with the base build, compared by the harness field check: REFCMP PASS, 0 fields differ (`submit_count_parity_vs_base.sh`). A first comparison against the v1.3.9 references differs only by the all-site files, which those references hold because they ran with the GPU pileup.
