# Code-path corpus for the GPU engine

Each small input forces one code path. The CPU engine of the same binary is the oracle. A run passes when the output matches the oracle, the target path ran, the invariants hold and no fallback was unplanned.

All results go under `$W/jobs/gpu-paths/` with `W=/mnt/projects/debruinz_project/debruinz/singlet`.

## Files

| File | Use |
|---|---|
| `corpus/manifest.tsv` | One row per input: source, size, synthetic or real, encode options, expected log lines, forbidden lines, fallback policy. |
| `corpus/mkinput.py` | Cuts a FASTQ pair from data on Clipper. With `--synth` it lowers a seeded share of Q11 bases. Output is labeled synthetic. |
| `jobs/build_corpus.sbatch` | Builds the `.1fq` inputs once. Nothing is downloaded. |
| `jobs/paths.sbatch` | Runs the corpus for `BIN` and `COMMIT`. Writes `results/paths-<job>/verdicts.tsv`. |
| `jobs/submit_paths.sh` | The chain: CPU references on `short-cpu` (job array), then one `debug-gpu` job. |
| `check_paths.py` | Identity, path assertions, invariants, BAM digest, verdict table. |
| `jobs/refcmp.sbatch`, `jobs/gate_q.sbatch` | The reference comparison and the gate copy with `FQ_OVERRIDE`, kept from experiment 1. |
| `corpus/earlier/` | The one-off encode and scan scripts of experiment 1. |
| `maskstatus_fixed.py` | `maskstatus.py` that also reads the engine log line without a width. |

## Run

```
bash jobs/build_corpus.sbatch          # once, as: sbatch jobs/build_corpus.sbatch
jobs/submit_paths.sh $W/jobs/gpu-device/bin/c92a494br c92a494b 
```

`BIN` has no default and `COMMIT` is required. The job stops unless `BIN/commit-gpu` starts with `COMMIT`. Scripts come from the tip of branch `bench/gpu-paths`, fetched under `flock`. `EXPECT_SCRIPTS` pins a commit prefix.

## The four checks per input

1. **Identity.** `out_equiv.py` with the run-dependent set, the 10 STAR Solo files, mapping statistics, `SJ.out.tab`, the three read files by sorted hash, `CellReads.stats` by sorted rows, `fieldcheck.py` and the sorted BAM by record hash when both runs leave one. When the CPU run fails, the GPU run must fail with the same exit status.
2. **Path.** Every `must` expression matches the GPU run log. No `must_not` expression matches. Files in `files_must` exist. Examples are `3 words each`, `restarting on the sorted path` and the `.1fq scanned` ingest line.
3. **Invariants.** Pileup reads are at least the uniquely mapped reads. Called cells do not exceed the barcodes of `cell_calls.tsv`. The fraction of reads in cells lies in 0 to 1. No `.gpu_handoff*` file is left in the output. They are checked on the CPU run and the GPU run. A violation fails the input even when the output matches the oracle.
4. **Fallback.** `fallback_strict` inputs need 0 disallowed lines from `fallback.py`. Info inputs report them. The allow-lists are not widened.

## Notes

- The engine mask has a minimum of 2 words (128 bases). A width of 1 word cannot occur. The input `mask2short` has reads of 60 bases and checks that the width clamps to 2.
- The sorted BAM is compared only when a run leaves it in the output. The restart path removes it at the end, so that comparison may report "not written".
- Synthetic inputs: `mask3syn` and `mask2short`. Seed 20261105, share 0.25. Their `stats.json` counts the lowered bases.

## Results at 6c4f1697 (5 Nov, node g052, T=8)

| Binary | Job | Result | Run time |
|---|---|---|---|
| c92a494br (c92a494b) | 800495 | 14 of 16 pass. `strand100` and `strand150` time out at 240 s (exit 124) after the restart on the sorted path. | 979 s with two 240 s hangs |
| s3 (e42c0636) | 800496 | 11 of 16 pass. `strand100` times out. `strand150`, `v5p150`, `mask3syn` and `long_q4` fail the path check by design: s3 declines reads over 128 bases. Their output matches the oracle. | 801 s with one 240 s hang |

The restart hang is not new in c92a494b: s3 shows it on `strand100`. Without the hangs the corpus takes about 8 minutes of GPU time.

## Rules for the shared nodes

Do not run the corpus on a node that holds a timing job of another session (`gf1-*`, `gf2-*`, `gbold-*`). Check `squeue -w <node>` first and pass `EXCLUDE=<nodes>` to `submit_paths.sh`.
