# Throughput track 2 (from 2026-11-18): rules for coding and cluster agents

Owner request, 2026-11-18: keep optimising the GPU and the CPU-only aligner. Assume the `.1fq` is on local disk; do as much as possible
on the device. Approximate methods are welcome **if** their recall is demonstrated and they save significant time.

## Contract
- **Default behaviour stays byte-identical** to the baseline build (v1.4.0 `b21de2cf` + the two gpu-pool admission fixes, commit
  `54f57754`). Every new behaviour sits behind a switch (environment variable or flag) that is OFF by default.
- An **approximate** mode is never the default. It ships only with a measured price: which outputs change and by how much, on which
  samples. Report counts and rates, never a claim without the number.
- HLA alleles, genotypes and SNP identities are sensitive: report counts and concordance only, never names or calls. Do not open
  `private/` output files except to count or hash them.

## Branches
Integration branch `gpu/throughput-2` (local, in `C:\Users\debruinz\singlet\singlet-gpu`; the orchestrator merges, you never do, and
you never push or open a PR). One worktree per task:
`git -C C:\Users\debruinz\singlet\singlet-gpu worktree add ../wt-t2-<name> -b t2/<name> gpu/throughput-2`, then work and commit only in
`C:\Users\debruinz\singlet\wt-t2-<name>`. Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>` (no model names).
The working trees have CRLF line endings (git autocrlf): edit with the Edit tool, or with scripts that keep CRLF.

## Clipper
Commands run through one shared bridge: `bash "<RC>" '<command>' [timeout_s]` with
RC=/c/Users/debruinz/AppData/Local/Temp/claude/b47c38086aba7e7c/f25663d1-6f16-4dcd-a7c4-2bb7c22e3f45/scratchpad/clipper/rc.sh
- The bridge is ONE serial shell shared by all agents. Keep each command short (< 60 s) and wrap anything that could hang in
  `timeout 30`. Never run recursive grep/find/du over data trees (campaign/, jobs/gpu-campaign, inputs): `ls` named directories, read
  named small files. Never put `sleep` or a wait loop inside a remote command.
- Several agents share the shell, so a call can wait behind another agent's: give every call a timeout of at least 90 s. A call
  that times out locally keeps running remotely and blocks the next one, so after a timeout wait 3 minutes before retrying; stop
  and report after 5 timeouts in a row. Never try plain ssh. Keep at most one poller, polling no more often than every 180 s.
- Send files as base64 (`base64 -w0 file`, then remote `printf %s '<b64>' | base64 -d > path`), at most 150 KB of base64 per command;
  append in chunks for larger files. Strip CRs first (`tr -d '\r'`), or sbatch rejects the script.
- To wait for a job, poll `sacct -X -n -j <id> -o State` every 180 s from a Bash call with `run_in_background: true`; never faster.
- W=/mnt/projects/debruinz_project/debruinz/singlet. Our directory: `$W/jobs/gpu-runtime` (scripts, patches/, bin/, results/); name
  everything you create there `t2-<name>...`. Logs: `$W/logs/<jobname>-<job>.out`. Scripts there are copies of
  `bench/gpu-runtime/jobs/*` and `bench/gpu-runtime/*.{py,sh}` of the repo; put new scripts in the repo too.
- **Never cancel, hold or modify a job you did not submit**, and never write outside `$W/jobs/gpu-runtime` (read-only everywhere
  else; the one exception is submitting the gate from `$W/jobs/gpu-pr2`). Other work runs under the same user: jobs named `pl*`,
  `camp-*`, `s15-node`, `adt-*`, `profile-pull`, `gpu-topo`, `ml-loaders`, `v140-*` and others are not ours.
- The account may hold 3 H100 GPUs, 6 V100S and 3 RTX 8000 at a time, shared with that other work: queueing is expected. A CPU-only
  job on `short-cpu` can wait a day; `-p short-gpu --exclude=g050,g051,g052` (no `--gres`) runs CPU-only on a GPU node at once.

## Build (no push needed)
In the worktree: `git -c core.autocrlf=false diff b21de2cf -- src include CMakeLists.txt bench/gpu/m5/stitch.cuh > <name>.diff`
(all committed and uncommitted changes against v1.4.0), send it to `$W/jobs/gpu-runtime/patches/t2-<name>.diff`, then
`cd $W/jobs/gpu-runtime && BASE=b21de2cf PATCH=$PWD/patches/t2-<name>.diff NAME=t2-<name> TESTS=1 sbatch --parsable -p short-gpu --exclude=g050,g051,g052 rt_build.sbatch`
-> `bin/t2-<name>/singlet-gpu`, BUILD.txt, ctest-full.txt (the build log `$W/logs/rt-build-<job>.out` must say `ctest exit 0`).
About 15 minutes. A name is built once: use `-2`, `-3` for rebuilds, and check first that neither the patch nor the bin directory
exists. There is no local compiler, so check before sending: no string literal broken across lines
(`git diff b21de2cf -- src include bench/gpu/m5 | grep -nE '^\+.*"[^"]*$'`, read every hit).
**Baseline binary:** `$W/jobs/gpu-runtime/bin/multirun/singlet-gpu` (= `54f57754`). The CPU engine is the same binary with
`--engine cpu`.

## Identity gate (every change to src/ or include/, with the new switch OFF, and again with an exact switch ON)
`cd $W/jobs/gpu-pr2 && sbatch --parsable -J t2-gate -p short-gpu --nice=0 --exclude=g050 --gres=gpu:quadro_rtx_8000:1 -c 18 --mem=170G --time=03:59:00 --export="ALL,CUDA_VISIBLE_DEVICES=0,SINGLET_ENGINE_GPU_MIN_CC=70,BINDIR=$W/jobs/gpu-runtime/bin/t2-<name>,BRANCH=gpu/w4-v137,ALLOW_NO_EMBED=1,KEEPOUT=1,SINGLET_HLA=defer,T=18,SAMPLES=pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2,MODES=cpu gpu,PAIRS=cpu:gpu,REFD=$W/jobs/gpu-runtime/ref-t2" pr2-gate.sbatch`
- `ref-t2` holds the baseline build's CPU-engine outputs (made by gate job 818295; if `ref-t2/T18/<sample>` is missing, that job has
  not finished: wait for it, do not make your own references with a modified binary).
- Pass = 8 lines `VERDICT <sample> cpu:gpu IDENTICAL` in `$W/jobs/gpu-pr2/results/gate-<job>/summary.txt` and no "fallback" line in
  the runs' fallbacks.txt (other than the R11 info line).
- For an approximate switch the gate is the measuring tool, not a pass/fail: run it with the switch on (add the variable to
  `--export`) and report per sample which files differ and by how much (`bench/star_accel/out_equiv.py` prints the differing files).

## Timing
- **GPU node, one sample class:** `cd $W/jobs/gpu-runtime && TAG=<tag> PROD=1 PARGS=--no-hla K=3 T=8 NUMA=1 XENV='SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16' BIN=$PWD/bin/<name>/singlet-gpu LIM=2400 STALLSTOP=900 sbatch rt_node.sbatch`
  -> `results/rt-node-<job>-<tag>/results.txt`: the RUN line (rate_overall, item_wall_mean, core_s_per_Mpair), "NODE steady-state
  rate", and the IDENT line (one distinct set hash, equal to the baseline arm's). Then `python3 cpu_ledger.py <dir>`,
  `bash steps.sh <dir>`, `bash lock_ledger.sh <dir>`.
- **GPU node, campaign-like mix:** `rt_mix.sbatch` (mix4, 110 items; read its header for the parameters).
- Compare arms in the order A/B/B/A on the same node. One run has about 4 % run-to-run variation: do not claim a gain under that
  without a repeat. Read each run's "memory: node" line; if MemFree min is far below ~140 GB or memory pressure is above a few
  percent, another tenant disturbed the node and the run is invalid.
- **Kernels only:** the m5 harness (`bench/gpu/m5/`, jobs under `bench/gpu/m5/jobs/`, census in `census.cuh`; see
  `docs/gpu-engine/fable/E1-profile.md`, `E2-census.md`, `E7-seeds.md`).

## Report
Final message: what changed (files, functions), job ids, the exact result lines, anything that failed or was skipped, and the commit
hash on your branch. State facts only; mark every estimate as an estimate. No merge into `gpu/throughput-2`.
