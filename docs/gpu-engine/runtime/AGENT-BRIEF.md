# Runtime track: rules for coding and cluster agents

Goal: more read pairs per second per H100 node for the campaign, outputs unchanged. Integration branch `gpu/runtime-v138` (local only,
never push, never open a PR). Each task has its own worktree `C:\Users\debruinz\singlet\wt-rt-<name>` on branch `rt/<name>`; work and
commit only there (commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`; no model names anywhere). The working
trees have CRLF line endings (git autocrlf): edit with the Edit tool, or with scripts that keep CRLF.

## Clipper
Commands run through one shared bridge: `bash "<RC>" '<command>' [timeout_s]` with
RC=/c/Users/debruinz/AppData/Local/Temp/claude/b47c38086aba7e7c/f25663d1-6f16-4dcd-a7c4-2bb7c22e3f45/scratchpad/clipper/rc.sh
Keep each command short (< 60 s): sbatch, squeue, sacct, cat/grep of logs. Send files as `echo <base64 -w0> | base64 -d > path`
(strip CRs first: `tr -d '\r'`). If the bridge times out twice, stop and report; do not try plain ssh. To wait for a job, poll squeue
every 60 s from a Bash call with run_in_background or a long timeout; never busy-loop faster. The bridge is ONE serial shell shared by all
agents: wrap remote commands in `timeout 30`, and never run recursive grep/find/du over data trees (jobs/gpu-campaign, campaign/, inputs):
one such command blocked every agent for hours on 2026-11-14. Use `ls` of specific directories or read named small files.
W=/mnt/projects/debruinz_project/debruinz/singlet. Our directory: $W/jobs/gpu-runtime (scripts, patches/, bin/, results/); the
scripts there are copies of `bench/gpu-runtime/jobs/*` and `bench/gpu-runtime/*.{py,sh}` of the repo. Logs: $W/logs/<jobname>-<job>.out.
Never cancel, hold or modify a job that is not yours (jobs named camp-pilot, s15-node, adt-*, ... belong to other work), never write
outside $W/jobs/gpu-runtime, never delete others' files. The account may hold 2 H100 GPUs at a time: node-timing jobs queue one after
another; that is expected.

## Build (no push needed)
In the worktree: `git diff c5436ec7 -- src include CMakeLists.txt > /tmp/<name>.diff` (committed and uncommitted changes against v1.3.8), send it to
$W/jobs/gpu-runtime/patches/<name>.diff, then
`cd $W/jobs/gpu-runtime && BASE=e0824b214dd146cf4dffa332a7e73a2b1f33a9bb PATCH=$PWD/patches/<name>.diff NAME=<name> TESTS=1 sbatch rt_build.sbatch`
-> bin/<name>/singlet-gpu, BUILD.txt, ctest-full.txt (must end "100% tests passed"). A name is built once: use <name>-2, -3 for rebuilds.
Compile errors are in $W/logs/rt-build-<job>.out. There is no local compiler, and a build waits in the queue, so check before
sending: no string literal broken across lines (a `\n` typed in an edit can land as a real newline; twice so far), with e.g.
`git diff c5436ec7 -- src include | grep -nE '^\+.*"[^"]*$'` and read every hit.

## Identity gate (every change to src/ or include/)
`cd $W/jobs/gpu-pr2 && sbatch -p short-gpu --nice=0 --exclude=g050 --gres=gpu:quadro_rtx_8000:1 -c 18 --mem=170G --time=03:59:00 --export="ALL,CUDA_VISIBLE_DEVICES=0,SINGLET_ENGINE_GPU_MIN_CC=70,BINDIR=$W/jobs/gpu-runtime/bin/<name>,BRANCH=gpu/w4-v137,ALLOW_NO_EMBED=1,KEEPOUT=1,SINGLET_HLA=defer,T=18,SAMPLES=pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2,MODES=cpu gpu,PAIRS=cpu:gpu,REFD=$W/jobs/gpu-campaign/ref-rel138-e0824b21" pr2-gate.sbatch`
Pass = 8 lines `VERDICT <sample> cpu:gpu IDENTICAL` in $W/jobs/gpu-pr2/results/gate-<job>/summary.txt and no "fallback" line in the
runs' fallbacks.txt. Read-only use of gpu-pr2 and gpu-campaign; this is the only job we submit from there.

## Node timing (the metric)
`cd $W/jobs/gpu-runtime && TAG=<tag> PROD=1 PARGS=--no-hla K=3 T=8 NUMA=1 XENV='SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16' BIN=$PWD/bin/<name>/singlet-gpu LIM=2400 STALLSTOP=900 sbatch rt_node.sbatch`
(HLA is off for optimisation from 2026-11-13 on.) Result: results/rt-node-<job>-<tag>/results.txt: the RUN line (rate_overall,
item_wall_mean, core_s_per_Mpair), "NODE steady-state rate", and the IDENT line (24 outputs, 1 distinct set hash, which must equal the
reference arm's). Then `python3 cpu_ledger.py <dir>`, `bash steps.sh <dir>` (mapping steps and lock holds), `bash lock_ledger.sh <dir>`.
One run per arm has about 4 % run-to-run variation: report numbers, do not claim a gain under that without a repeat.

## Report
Final message: what changed (files, functions), job ids, the exact result lines, anything that failed or was skipped, and the commit
hash on your branch. State facts only; no merge into gpu/runtime-v138 (the orchestrator audits and merges).
