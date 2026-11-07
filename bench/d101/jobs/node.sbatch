#!/bin/bash
#SBATCH --job-name=d101-node
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# All D-101 validation work as one dependency graph on one whole node (Clipper rule 2026-11-07: one node at a time, every core used).
# node_run.py runs the per-item scripts in this folder (dl, bulk_star, run_sc, reditools, hla_sc, geuv_hla, host ctest) as tasks;
# finished tasks are skipped, so resubmit with --dependency=afterany:<this job> until everything is done. Scoring runs at the end.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; J=$W/jobs/d101; LIBS=$J/libs.tsv
BIN=${BIN:-$W/jobs/gpu-pr2/bin2/val-daaf9d80/singlet-gpu}
NC=$(nproc); MEMG=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1048576 - 24 ))
echo "node $(hostname) cpus $NC mem ${MEMG}G $(date -Is)"
URL="$(git -C $W/repos/singlet-py.git config remote.origin.url)"
TF=$J/state/tasks-$SLURM_JOB_ID.tsv; mkdir -p $J/state $J/tasklogs $J/results
run() {  # run SCRIPT ROW [ENV...] -> a command line that runs a job script as one task
  local s=$1 r=$2; shift 2
  printf 'env SLURM_ARRAY_TASK_ID=%s SLURM_CPUS_PER_TASK=$TASK_CPUS SLURM_JOB_ID=%s-$TASK_NAME LIBS=%s BIN=%s %s bash %s/%s' "$r" "$SLURM_JOB_ID" "$LIBS" "$BIN" "$*" "$J" "$s"
}
ct() {  # host ctest of a branch tip, named after the tip
  printf 'BR=%s NAME=%s-$(git ls-remote %s refs/heads/%s | cut -c1-8) SLURM_JOB_ID=%s-$TASK_NAME bash %s/jobs/gpu-pr2/ctest-host2.sbatch' "$1" "$2" "$URL" "$1" "$SLURM_JOB_ID" "$W"
}
{
  for r in 1 2 3 4 5 6 7 8; do printf 'dlsc-%s\tdlsc\t4\t16\t-\t%s\n' $r "$(run dl.sbatch $r KIND=sc)"; done
  for r in 1 2 3 4 5 6 7 8; do printf 'dlbulk-%s\tdlbulk\t6\t24\t-\t%s\n' $r "$(run dl.sbatch $r KIND=bulk)"; done
  printf 'ct-edit\tct\t24\t64\t-\t%s\n' "$(ct feat/rna-editing edit)"
  printf 'ct-hla\tct\t24\t64\t-\t%s\n' "$(ct feat/hla-typing hla)"
  for r in 1 2 3 4 5 6 7 8; do printf 'run-%s\trun\t32\t180\tdlsc-%s\t%s\n' $r $r "$(run run_sc.sbatch $r)"; done
  for r in 1 2 3 4 5 6 7 8; do printf 'star-%s\tstar\t16\t72\tdlbulk-%s\t%s\n' $r $r "$(run bulk_star.sbatch $r)"; done
  for r in 1 2 3 4 5 6 7 8; do printf 'redi-%s\tredi\t16\t48\trun-%s,star-%s\t%s\n' $r $r $r "$(run reditools.sbatch $r)"; done
  for r in $(seq 9 20); do printf 'dlsc-%s\tdlsc\t4\t16\t-\t%s\n' $r "$(run dl.sbatch $r KIND=sc)"; done
  for r in $(seq 9 20); do printf 'hla-%s\thla\t8\t24\tdlsc-%s\t%s\n' $r $r "$(run hla_sc.sbatch $r)"; done
  for r in $(seq 1 60); do printf 'geuv-%s\tgeuv\t8\t24\t-\t%s\n' $r "$(run geuv_hla.sbatch $r)"; done
} > $TF
$ENV/bin/python3 $J/node_run.py $TF --state $J/state --logs $J/tasklogs --cpus $NC --mem $MEMG \
  --limit dlsc=3 --limit dlbulk=2 --limit geuv=3 --limit run=2 --limit ct=1 --stop-after 12000
# Scoring (cheap; reruns on whatever has finished). HLA results are owner-only.
chmod 700 $J/results
$ENV/bin/python3 $J/score_hla.py geuvadis --truth $J/geuv/truth.txt --res $J/geuv/res > $J/results/hla_geuvadis.json 2>&1
ARGS=""
while read -r D SC G B BG; do
  [ "$D" = donor ] && continue
  case $D in C) L=C1;; Sk) L=Sk1;; *) L=$D;; esac
  for f in $J/runs/$SC/out/private/hla/hla_genotype.tsv $J/hla/$SC/private/hla/hla_genotype.tsv; do [ -s $f ] && { ARGS="$ARGS $L=$f"; break; }; done
done < $LIBS
[ -n "$ARGS" ] && $ENV/bin/python3 $J/score_hla.py nih --compiled $J/compiled_nih_3p_genotypes.csv --singlet $ARGS > $J/results/hla_nih.json 2>&1
cat $J/editing/*/score.json > $J/results/editing_scores.jsonl 2>/dev/null
ls -l $J/results; echo "end $(date -Is)"
