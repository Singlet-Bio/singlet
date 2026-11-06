#!/bin/bash
#SBATCH --job-name=gk2-decide
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-kern2 decider (CPU). If WSHRINK or BSTREAM is IDENTICAL against b0 and at least 10% faster on lane B's phase 3 kernel (ncu) or on the prefill hold
# (timing job), it submits one more exclusive timing job that combines the winners under MPS at 3 and 4 slots (and the base arm under MPS at 3 slots for
# comparison), then a second summarizer after it. Compute nodes may call sbatch (checked by job gk2-subtest).
# Parameters (--export): IDENT, NCU, TP (job ids), BINDIR, EXPECT_COMMIT, SUMMARY_DIR (results/overnight), SPEC (job spec of the summarizer, kinds as in kern2_summary.py).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/gpu-kern2
: "${IDENT:?}" "${NCU:?}" "${TP:?}" "${BINDIR:?}" "${EXPECT_COMMIT:?}" "${SPEC:?}"
R=$J/results/decider-$SLURM_JOB_ID; mkdir -p $R
IR=$(ls $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
in_runtime python3 $J/kern2_summary.py decide $IDENT $NCU $TP > $R/decision.json 2> $R/decide.err
cat $R/decision.json; cat $R/decide.err
ARM=$(sed -n 's/.*"arm": "\([a-z]*\)".*/\1/p' $R/decision.json)
: > $R/next_jobs.txt
if [ -n "$ARM" ]; then
  echo "winner arm: $ARM"
  TPJ=$(sbatch --parsable --job-name=gk2-tp-mps --export=ALL,BINDIR=$BINDIR,EXPECT_COMMIT=$EXPECT_COMMIT,POINTS="k8t7s3r3@mpsbase k8t7s3r3@mps$ARM k8t7s4r3@mps$ARM" $J/tp2.sbatch)
  SUM=$(sbatch --parsable --dependency=afterany:$TPJ --export=ALL,JOBSPEC="$SPEC;decider=$SLURM_JOB_ID=decider",OUT=$SUMMARY_DIR $J/summarize2.sbatch)
  echo "tp=$TPJ summarizer=$SUM" | tee $R/next_jobs.txt
else
  echo "no arm qualifies: no follow-up timing job"
fi
echo done $(date -Is)
