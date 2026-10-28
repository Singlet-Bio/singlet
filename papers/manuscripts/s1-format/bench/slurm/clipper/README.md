# Running the benchmark on GVSU Clipper

These jobs replace `../array.sbatch` and `../serial.sbatch` on Clipper (account `debruinz`). Each job runs one pinned
commit of this repository (`S1_SHA`), cloned under `$S1=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/<sha>`;
`common.sh` gives the layout. Nothing runs on the login node: it only submits.

| Script | What | Where |
|---|---|---|
| `setup.sbatch` | conda env from `environment.yml`, BPCells, the singlet package (Python, R), C++ bench, `test_floor.py`, `smoke.py`, `s1bench env` | short-cpu |
| `stage.sbatch` | the stages in `STAGES` for one dataset of `DS_LIST` per array task | per size class, below |

Hardware (measured 2026-10-28). AVX-512 on every node. Only **c100 and c101** (Xeon 6730P, 64 cores, 512 GB) have a
local NVMe `/tmp`; every other node has a spinning RAID `/tmp`, and `$S1` is NFS. So:

- Untimed stages (`fetch canon write ladder floor second`) run on shared nodes: S and M on `short-cpu` (8 cores, 64 GB),
  L on `short-bigmem` or `bigmem` (16 cores, 256 GB), XL on `bigmem` (16 cores, 600 GB, up to 2 days). Write times from these
  jobs are not headline numbers; `write` is re-timed on c100.
- Timed stages (`decode`, `minibatch`, `eco`, and a re-run of `write --force` for timings) run on c100 alone:
  `--partition=cpu --constraint=granite-rapids --exclusive`, after copying each dataset's files to the node's `/tmp`.

```bash
SHA=<commit>                                   # pushed to GitHub first
sbatch --export=ALL,S1_SHA=$SHA setup.sbatch
sbatch --array=0-9%6 --partition=short-cpu --mem=64G --cpus-per-task=8 \
  --export=ALL,S1_SHA=$SHA,DS_LIST=pbmc3k,pbmc10k_v3,...,STAGES="fetch canon" stage.sbatch
```

Results stay in `$S1/results`; copy the TSVs, `env.json` and `environment.lock.yml` back to `../../results/` in the
repository after each phase. Data and format outputs never leave Clipper.
