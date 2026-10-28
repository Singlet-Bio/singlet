# Running the benchmark on GVSU Clipper

These jobs replace `../array.sbatch` and `../serial.sbatch` on Clipper (account `debruinz`). Each job runs one pinned
commit of this repository (`S1_SHA`), cloned under `$S1=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/<sha>`;
`common.sh` gives the layout. Nothing runs on the login node: it only submits.

| Script | What | Where |
|---|---|---|
| `setup.sbatch` | conda env from `environment.yml`, BPCells, the singlet package (Python, R), C++ bench, `test_floor.py`, `smoke.py`, `s1bench env` | short-cpu |
| `setup.sbatch` with `S1_ENV=$S1/env-gpu S1_ENV_FROM=$S1/env` | the timed jobs' env: a clone of `$S1/env` updated to `environment.yml` (CUDA build of PyTorch) | short-cpu |
| `stage.sbatch` | the stages in `STAGES` for one dataset of `DS_LIST` per array task | per size class, below |
| `netem.sbatch` | installs Toxiproxy into `$S1/bin` and checks `rclone serve s3` behind it (throttled GET, range GET, unsigned listing) | scavenger |
| `timed.sbatch` | `decode`, the thread `sweep`, `minibatch`, `eco` for one dataset per task, on a whole node | exclusive H100 node |

Hardware (measured 2026-10-28). AVX-512 on every node. Only c100 and c101 have a local NVMe `/tmp`; every other node
has a spinning RAID `/tmp`, and `$S1` is NFS. The timings deliberately use an ordinary node, since most users read from
spinning disks, shared filesystems or object stores:

- Untimed stages (`fetch canon write ladder floor second`) run on shared nodes (scavenger): S on 96 GB, M on 192 GB,
  L on 420 GB, XL on 1 TB. Write times from these jobs are not headline numbers (the report does not use them).
- Timed stages run on an H100 node to itself (`timed.sbatch`: `--exclusive --gres=gpu:nvidia_h100_nvl:2`, 4 h on
  short-gpu). Each task copies its dataset's format files to the node's own disk (the `local` tier), reads the NFS copy in
  place (`nfs`), and serves the files from `rclone serve s3` behind Toxiproxy (`s3_*`; MinIO no longer publishes
  community binaries). With a GPU present the first minibatch is also
  copied to the device and timed separately (`to_cuda_s`).
- Cross-check: the same script on a CPU-only node (bigmem, Xeon 6226) with `S1_RESULTS=$S1/results-xcheck`, `--partition=bigmem`
  and no `--gres`.

```bash
SHA=<commit>                                   # pushed to GitHub first
S1_SHA=$SHA sbatch --export=ALL setup.sbatch
# lists go through the environment: sbatch --export splits its argument on commas
DS_LIST=pbmc3k,pbmc10k_v3 STAGES="fetch canon" S1_SHA=$SHA sbatch --array=0-1 --partition=scavenger --mem=64G \
  --cpus-per-task=8 --export=ALL stage.sbatch
DS_LIST=pbmc10k_v3 S1_SHA=$SHA sbatch --array=0 --export=ALL timed.sbatch
```

Results stay in `$S1/results` (cross-checks in `$S1/results-<name>`); copy the TSVs, `env.json` and `environment.lock.yml` back to `../../results/` in the
repository after each phase. Data and format outputs never leave Clipper.
