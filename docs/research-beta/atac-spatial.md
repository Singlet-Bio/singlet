# Research beta: ATAC, Multiome and Visium (hand-over)

Branch `claude/atac-spatial-benchmark-2ba7df`, draft PR Singlet-Bio/singlet#48 (not merged). Full report:
`docs/atac-spatial-benchmark.md`. Storage proposal: `docs/format-store.md` section 14.

## Goal

Process scATAC, 10x Multiome ATAC and 10x Visium (fresh-frozen and FFPE) correctly, and benchmark them against the
competition for singlet.bio/benchmarking. The comparisons use the public results contract 1.1.0 (bench-public), with 10x
published outputs as truth and open-source pipelines run on the same FASTQs.

## What was done

- **ATAC** (`include/singlet/pileup/atac_*.h`, `singlet atac --bam` subcommand):
  - Barcode window and strand are detected, then barcodes are corrected against the whitelist (Hamming-1, count prior).
    Multiome ATAC barcodes are translated to GEX barcodes.
  - Fragments come from the leftmost mate (previously half of all fragments were shifted). They are Tn5-shifted,
    deduplicated, and carry read-pair counts. STAR uses `Extend5pOfReads12` and keeps protruding mates.
  - Peaks use a MACS2-style caller on cut sites, q < 0.05.
  - QC follows Cell Ranger / ArchR / Signac definitions.
  - Cells: an ATAC-only density-valley cut, or a joint GEX+ATAC mixture with `--paired-gex`.
  - Outputs: peak and gene-activity matrices, bgzip+tabix fragments, and `atac_summary.json` with pass/fail QC.
- **Visium** (`visium_tissue.h`, Visium block of `singlet.cpp`):
  - The spot list is the whitelist when the assay is Visium.
  - Tissue is detected from counts: a mixture model plus hexagonal smoothing, no image needed. Tissue spots are the
    called set.
  - Outputs: `spatial_positions.tsv` and `spatial_summary.json` with pass/fail QC.
- **Barcode lists** are bundled with attribution (Zach approved; `python/singlet/data/whitelists/ATTRIBUTION.md`,
  root `NOTICE`):
  - visium-v1 spots and coordinates;
  - ARC ATAC↔GEX pairs;
  - **the real scATAC whitelist.** The old `737K-cratac-v1.txt` was a copy of the 3' v2 list.

  All are rebuilt only from 10x CC BY 4.0 datasets.
- **Store format:** `python/singlet/store/modality_members.py` (section-14 members) and client modalities `atac`, `gene_activity`,
  `spatial`.
- **Contract:** `bench/public/capabilities/{atac,multiome,spatial,_atacsp}.py` and `bench/public/results/*.json`.

## Results (contract 1.1.0; binary = branch 88a41ff merged onto prod/modality-fixes 4f110f2)

| capability | status | headline |
|---|---|---|
| multiome | **pass** | joint cells vs ARC: Jaccard 0.985 (per-modality calls 0.956); ATAC→GEX pairing identical for 11,894 / 11,894 ARC cells |
| atac | **fail** (9/10) | cells vs Cell Ranger: 0.950 (5k), 0.985 (10k); Chromap + SnapATAC2: 0.884 / 0.918. Exact fragments and peaks (5k) win. **10k peak F1 0.830 vs MACS3 0.837** (higher recall, lower precision) |
| spatial | **fail** (2/10) | image-free tissue κ beats Otsu on both slides (0.850 vs 0.776 mouse; 0.840 vs 0.827 FFPE). Spot counts trail STARsolo slightly (UMIs ρ 0.9948 vs 0.9982 mouse; 0.9995 vs 0.9996 FFPE) |

Runtime: Singlet ATAC takes about 1.8× the time of Chromap + MACS3 + SnapATAC2 (resources, not gated).

## Decisions

- 10x comparators are 10x's published outputs; no 10x software is run (Zach).
- Real data only. The development set is Multiome 3k: every post-first-run parameter (density-valley cut,
  Extend5pOfReads12, q 0.05) was chosen on it. The 5k/10k scATAC/Multiome sets and both Visium sets are held out.
- The 5k Cell Ranger ATAC 1.2 outputs are hg19. They are lifted to GRCh38 with UCSC's chain, and both tools are scored on
  the same lifted set.
- Match-or-beat tolerances are the half-width of the bootstrap 95% CI of the paired difference.

## Open problems

1. **Mouse NUMT mito genes (GEX counting, not Visium code).** On GRCm39, Singlet drops multimappers that overlap a single
   gene. As a result mt-Co2, mt-Atp6, mt-Co3 and mt-Nd3 are nearly absent (mt-Co3: 71 vs Space Ranger's 1.7 M), and
   neuron1k shows the same. This drives the spatial count failures. pipeline-benchmark is fixing it on
   `fix/intron-assignment`. **Re-run the Visium (and any mouse) numbers on the binary that has the fix before
   publishing.**
2. **10k peak precision.** Try a minimum-signal filter on the peak calls, choosing it on Multiome 3k only.
3. **Merge into the campaign build.** Results then need re-running on the campaign binary, so that every capability
   carries one Singlet commit.
4. **Visium protocol detection** still cannot tell Visium from 3' v3 automatically. Today Visium mode needs
   `--protocol 10x-visium` at encode, or catalog metadata. A hit-rate test against `visium-v1.txt` was tried in
   `protocol.h` and reverted because it broke two `test_ws2_protocol_hardening` checks. Redo it with those tests.
5. **Not handled:** Visium HD, Slide-seq and CytAssist (visium-v2) spot lists; the client reader for the `fragments`
   modality.

## Clipper state (WORK=/mnt/projects/debruinz_project/debruinz/singlet)

- Data (213 GB): `$WORK/data/atac-spatial/`.
- Singlet runs: `$WORK/pipeline-benchmark/runs/atacsp/<run>/`. The runs are pbmc_multiome3k_{gex,atac},
  pbmc_multiome10k_{gex,atac}, atac_pbmc_5k_nextgem, visium_mouse_brain_ant and visium_ffpe_human_breast. ATAC runs keep
  `atac_sorted.bam`.
- Competitor runs: `chromap_<ds>/` and `starsolo_*`.
- Binary: `$WORK/pipeline-benchmark/bin/singlet-atacsp` (sha256 982f5925…). The runtime lists are in
  `$WORK/pipeline-benchmark/whitelists/` and `whitelists/atacsp/`. The comparator env is `$WORK/envs/atacbench`.
- Artifacts and provenance: `$WORK/bench-atacsp/artifacts/{atac,multiome,spatial,dev}/`, copied to
  `$WORK/bench-public/artifacts/{atac,multiome,spatial}/`. Results are also in `$WORK/bpub/results-incoming/`.
- Final job ids:
  - build 787612;
  - Singlet: mo10k-gex 787613, mo10k-atac 787614, atac5k 787615, mo3k-gex 787616, mo3k-atac 787617,
    visium-mba 787618, visium-ffpe 787623;
  - comparators: chromap 10k 787458 / 5k 786156 / 3k 786149; STARsolo mouse 785797, FFPE 787637.

  All of them have finished. Nothing is pending.

## How to resume

1. Rebuild, after merging fixes: `sbatch -A debruinz bench/atac_spatial/build.sbatch` (BR / BASE env vars choose the
   branches).
2. Re-run everything with dependencies: `bash bench/atac_spatial/submit.sh` on the login node. It also prints the
   `make_provenance.py` command.
3. Copy the artifacts locally. Next to bench-public's `bench/public/{contract.py,cohort.tsv,capabilities/_common.py}`, run
   `compute(artifact_dir)` from each `bench/public/capabilities/<id>.py`. It takes about 15 min on a laptop, most of it in
   the peak bootstrap. Then run `python bench/public/contract.py validate`.
