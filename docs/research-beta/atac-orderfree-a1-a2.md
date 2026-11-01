# ATAC order-free rules A1 and A2 on the CPU path: Clipper check

Branch `claude/atac-orderfree-a1-a2` (cut from `origin/claude/atac-spatial-benchmark-2ba7df`, PR #48). Rules and reasoning:
`docs/research-beta/atac-reducers-prep.md` on branch `claude/gpu-atac-reducers-270cf8` (sections 2, 3, 3.1).

- A1: fragment dedup is one global exact-key table (`atac_fragment.h`); the per-start clearing is opt-in (`set_assume_sorted`).
- A2: barcode layout and whitelist are chosen on the records with `qname_hash % 64 == 0`, no prefix cap (`atac_barcodes.h`, `atac_pipeline.h`).

## Build (Clipper, 2026-10-29)

`bench/atac_reducers/build_ordfree.sbatch` (on the GPU-prep branch), no merge of `prod/modality-fixes` (it now conflicts with this branch).
Binaries in `pipeline-benchmark/bin/`: `singlet-atacsp-pr48` (unmodified PR #48 tip) and `singlet-atacsp-ordfree` (this branch, `a7cd1999`).
Two compile errors were found and fixed on the way (newline escapes broken in the new code and tests). `test_atac_fragment` (with the two new tests),
`test_atac_peak_caller` and `test_atac_metrics` build and pass; the other ATAC/Visium test targets were not built in this run.

## Real-data check (job 792135, `short-bigmem`, 23 min)

Sample `pbmc_multiome3k_atac` (73.2 M read pairs), `singlet atac --bam` on its kept `atac_sorted.bam`, and on the same BAM after
`samtools collate` (name-grouped, not coordinate-sorted). Same whitelists, GTF and `--paired-gex` for all four runs.

| run | unique fragments | duplicate read pairs | peaks | cells | fragments.tsv md5 |
|---|---|---|---|---|---|
| old, sorted | 45,349,184 | 23,223,713 | 97,814 | 2,721 | ff0d970e... |
| new, sorted | 45,349,184 | 23,223,713 | 97,814 | 2,721 | ff0d970e... |
| new, collated | 45,349,184 | 23,223,713 | 97,814 | 2,721 | ff0d970e... |
| old, collated | 68,572,896 | 1 | 111,605 | 2,726 | 557cdfc6... |

- `fragments.tsv.gz` (decompressed), `atac_peak_counts.mtx.gz` (decompressed), `peaks.bed`, `atac_barcode_metrics.tsv` and `atac_cells.tsv` are
  **byte-identical** across old-sorted, new-sorted and new-collated.
- The old code on a non-sorted BAM keeps 23.2 M duplicate read pairs as separate fragments (+51%), which also changes the peak set (+14%) and the cell call
  (2,726 of 11,054 candidates). This is finding F1, measured.
- A2: the layout is the same (offset 0, forward, whitelist 1); the exact hit rate on the sample is 0.9539 on 1,144,223 hash-sampled reads vs 0.9549 on the first 200,000
  records. Barcode counts (exact, corrected, ambiguous, unmatched) are identical in all three.
- Cost: wall 3:37 (old) vs 4:16 (new, sorted) and 5:24 (new, collated); peak RSS 2.2 GB vs 4.8 GB (the global dedup table and the raw-barcode map).

Not checked: the 5k and Multiome 10k sets, and the order-free property of A2 in a case where the prefix and the sample disagree (only the port test covers that).
