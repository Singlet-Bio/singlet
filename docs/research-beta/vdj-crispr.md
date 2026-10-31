# Research-beta handoff: V(D)J and CRISPR guide modalities

- **Session:** vdj-crispr, 2026-10-26.
- **Branch:** `claude/vdj-crispr-benchmark-721d3a`, stacked on the ADT branch (PR #29).
- **Draft PR:** Singlet-Bio/singlet#51. Nothing is merged, and nothing has been written to R2 or D1.

## Goal

The library-type screen found 3,907 V(D)J (TCR/BCR) and 616 CRISPR guide libraries in the atlas, all processed as RNA.
The goal is to make both first-class modalities and to benchmark them against the competition: Cell Ranger (10x's and
the authors' published outputs) and TRUST4. MiXCR was dropped because it needs a licence key, by Zach's decision.

## What was done (code)

- **`python/singlet/vdj/`**
  - `reference.py`: the segment reference, built from GENCODE, GENCODE's alt haplotypes and NCBI RefSeq. No IMGT data is
    used, and the built references are packaged as `vdj/data/vdj_{human,mouse}.tsv.gz`.
  - `_kernels.py` + `assemble.py` (numba): Cell Ranger barcode and UMI correction, read-support and one-chain-per-molecule
    filters, and junction assembly with per-UMI consensus. Gene calls go by equivalence class. Contig filters cover error
    variants and ambient junctions (enclone-style "cross").
  - `clonotype.py`, `qc.py` (pass/fail thresholds), `store.py` (v2 members), `library.py` (atlas entry point).
- **`python/singlet/crispr/`**
  - `guides.py`: geometry from the reads, including dual guides.
  - `count.py`: counting on the ADT core. `adt/count.py`'s key packing now allows more than 1,024 features.
  - `assign.py`: the caller, an ambient-calibrated Poisson test whose ambient load comes from empty droplets, plus the
    GMM and Poisson-Gaussian comparators.
  - `qc.py`, `store.py`, `library.py`.
- **Benchmark harness:** `bench/vdj/` (run, compare, artifacts, Clipper jobs in `hpc/`) and `bench/crispr/` (run,
  compare, Replogle and Norman guide-table preparation).
- **Results contract:** `bench/public/capabilities/{vdj,crispr_guides}.py`. The interim JSONs are in
  `bench/public/results/`.
- **Docs:** `docs/vdj-crispr-modalities.md` (method and CRISPR results) and `docs/format-store.md` §14 (storage proposal).

## Results so far

**CRISPR (`crispr-guides.json`): 21 of 21 metrics pass, document still `draft`.** Against the published calls, Singlet
has the best single-call F1 on all four real screens: A549 0.936, A375 0.998, Replogle K562 gem group 41 0.919, and the
Norman held-out half 0.904. It also gives the most guides with significant knockdown and the most usable perturbed
cells.
- On A549, the guides only Cell Ranger adds leave their targets at 1.20x control expression, i.e. they are ambient.
- Counting matches Cell Ranger: a 1.012 UMI ratio, and Cell Ranger's own thresholds on Singlet's counts reproduce 99.4%
  of its calls.

**V(D)J (`vdj.json`) is interim: development libraries only, and the document is invalid until the held-out runs are in.**
On pbmc5p_v1 and pbmc10k_5p_v2, Singlet and TRUST4 compare as follows (TRUST4 in brackets):

| metric | Singlet | TRUST4 |
|---|---|---|
| contig F1 | 0.93-0.97 | 0.68-0.83 |
| clonotype ARI | 0.86-1.0 | 0.0-0.67 |
| wall time | ~10x faster at 8 threads (563 s on PBMC 10k) | 4,468 s |

- Every TCR metric matches or beats TRUST4.
- Failing:
  - BCR on PBMC 10k: CDR3 accuracy 0.985 vs 0.995, correct pairing 0.951 vs 0.974, and J agreement.
  - pbmc5p_v1 BCR J agreement.
  - Wrong-lineage contigs against Cell Ranger: about 0.1-0.6% vs 0-0.04%.
- Known cause of part of the BCR gap: neither GENCODE nor RefSeq annotates IGHV1-8 or IGHV4-30-4 on GRCh38.

## Decisions

- Comparators are 10x's or the authors' published outputs; 10x software is not run. TRUST4 runs on the same FASTQs at 8
  threads.
- Agreement gates on F1 (recall and precision are in the notes). Significant guides are a count, not a share. Ties are
  given a tolerance equal to the half-sample 95% half-width of the Singlet - TRUST4 difference.
- Development data:
  - V(D)J: pbmc5p_v1, pbmc10k_5p_v2, GSE145370 S133A/S158A.
  - CRISPR: A549 and A375.
- Held out:
  - V(D)J: GSE145370 S135A/S149A/S150A/S133B and the 10x Chromium Controller 10k (downloaded after the code was frozen at
    094da510).
  - CRISPR: Replogle and Norman.
- Replogle protospacers were rebuilt from hg19 coordinates in the sgRNA ids. For Norman, the guide-barcode names were
  learnt on half the cells, and scoring uses the other half only.

## Open problems

1. The held-out V(D)J numbers are not computed yet. These TRUST4 reruns were running at the freeze (200 GB, after OOM at
   64 GB): 788170 ctrl10k_b, 788171 ctrl10k_t, 788172 S135A, 788173 S149A, 788174 S150A. Singlet's runs and S133B's
   TRUST4 run are done.
2. The `vdj.json` validity fixes need bench-public's cohort to include `tenx_pbmc10k_5pv2_controller` and
   `replogle2022_k562_gg41`. The results job appends gg41 to its own scratch copy only.
3. PR #51 CI: `lint` and `python-test (3.12)` failed at 34a6f53, and the PR conflicts with `main`. Neither is fixed
   (freeze).
4. BCR gaps:
   - The missing IGHV genes.
   - J calls on SHM'd BCRs.
   - Ambient contigs against Cell Ranger.
5. CRISPR libraries without a guide list cannot be named; no reference-free naming exists yet.

## Next steps (after Alpha)

1. Check that jobs 788170-788174 finished, or rerun them with `--mem=200G`
   (`DS=<lib>,CHAINS=TCR|BCR,METHODS=trust4 run.sbatch`).
2. Run `sbatch $WORK/jobs/vdjc/results.sbatch`. It writes `$WORK/state/vdjc/results/{vdj,crispr-guides}.json`. Copy them
   to `bench/public/results/`, set `status: final`, validate, and deliver to bench-public, or drop them in
   `$WORK/bpub/results-incoming/`.
3. Fill the V(D)J results in `docs/vdj-crispr-modalities.md` (`<!-- VDJ RESULTS -->`), fix CI, merge `main`, and update
   PR #51.
4. Run the atlas campaign with `python -m singlet.vdj.library` / `singlet.crispr.library` per library against its GEX
   sibling.

## Clipper paths (`WORK=/mnt/projects/debruinz_project/debruinz/singlet`)

| what | path |
|---|---|
| data | `$WORK/data/vdjc/<lib>/` (fastqs, `filtered_contig_annotations.csv`, `gex_filtered.h5`); `a549`, `a375`, `replogle_k562_gg41`, `norman2019_gem1`; references in `alts/`, `hg19/`, `trust4/` |
| runs | `$WORK/state/vdjc/runs/<lib>/{singlet,trust4}/`, `metrics.json`; segment references in `$WORK/state/vdjc/ref/` |
| artifacts | `$WORK/bench-public/artifacts/{vdj,crispr-guides}/` |
| jobs | `$WORK/jobs/vdjc/` (`run.sbatch`, `crispr_run.sbatch`, `norman.sbatch`, `results.sbatch`, `build_ref.sbatch`, `src.sh`); logs in `$WORK/logs/vdjc-*` |
| env | `$WORK/envs/vdjcrispr` (TRUST4 1.1.5, numba, dnaio, scanpy) |

## How to resume

1. Check out the branch.
2. On Clipper, jobs copy the pushed commit into `$WORK/cache/vdjc-src/<sha>` (via `src.sh`), so push before you submit.
3. Run the tests: `PYTHONPATH=python pytest tests/python/test_vdj_assemble.py tests/python/test_crispr.py tests/python/test_vdj_crispr_store.py`.
