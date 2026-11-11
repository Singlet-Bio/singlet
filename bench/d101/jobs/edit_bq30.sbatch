#!/bin/bash
#SBATCH --job-name=d101-bq30
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-109 difference 1 (bench/d101/editing_rules_review.md): REDItools2's base-quality rule at the editing sites. Builds feat/rna-editing
# (076da76d adds --editing-min-baseq; its pileup code is identical to val-daaf9d80, which made the Phred-10 runs), runs donor ROW
# (default 1 = A) through the same --pipeline call with --editing-min-baseq 30, and once the neutral REDItools3 sample of that
# donor exists (editing_neutral/<sc>/), scores both settings on it. Output (owner-only): $W/jobs/d101/bq30/<sc>/
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/d101; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; T=${SLURM_CPUS_PER_TASK:-24}
ROW=${ROW:-1}; read -r DONOR SC SCG BULK BULKG <<< "$(awk -v n=$ROW 'NR==n+1' $J/libs.tsv)"
O=$J/bq30/$SC; mkdir -p $J/bq30 $O; chmod 700 $J/bq30 $O
OUTD=$J/bin; B=$OUTD/singlet-editbq
if [ ! -x $B ]; then
  env BRANCH=feat/rna-editing NAME=editbq OUTD=$OUTD SLURM_CPUS_PER_TASK=$T SLURM_JOB_ID=${SLURM_JOB_ID}-build \
    bash $W/jobs/gpu-pr2/build.sbatch > $O/build.log 2>&1 || { echo "build failed"; tail -20 $O/build.log; exit 2; }
fi
echo "binary $B commit $(cat $OUTD/commit-editbq)"
if [ ! -e $O/DONE ]; then
  SCR=/tmp/$USER-bq30-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
  export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
  ( cd $SCR && /usr/bin/time -v -o $O/time.txt $B $W/data/d101/sc/$SC.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --editing-min-baseq 30 > $O/run.log 2>&1 )
  mkdir -p $O/private; cp $SCR/out/private/editing/editing_sites.tsv.gz $O/private/ && cp $SCR/out/editing_summary.json $O/ 2>/dev/null
  # the SNP panel must be unchanged by the editing threshold
  for f in $(cd $SCR/out && ls private/snp* snp* 2>/dev/null); do cmp -s $SCR/out/$f $J/runs/$SC/out/$f && echo "same $f" || echo "DIFFERS $f"; done | tee $O/snp_check.txt
  [ -s $O/private/editing_sites.tsv.gz ] && date -Is > $O/DONE
fi
N=$J/editing_neutral/$SC
if [ -s $N/reditools_bulk.tsv ] && [ -s $N/reditools_sc.tsv ]; then
  for q in 10 30; do
    SG=$J/runs/$SC/out/private/editing/editing_sites.tsv.gz; [ $q = 30 ] && SG=$O/private/editing_sites.tsv.gz
    $ENV/bin/python3 $J/score_editing.py --singlet $SG --sc $N/reditools_sc.tsv --bulk $N/reditools_bulk.tsv --sites $N/sites.tsv.gz \
      --label "$DONOR-bq$q" --out $O/score_bq$q.json
  done
  $ENV/bin/python3 - $O <<'EOF'
import json, sys
o = sys.argv[1]
for q in (10, 30):
    p = json.load(open(f"{o}/score_bq{q}.json"))["primary_neutral"]
    s, b = p["singlet"], p["both_covered_matched_depth"]
    print(f"bq{q}: neutral {p['sites']} coverage {s['coverage']:.3f} r {s.get('pearson')} mae {s.get('mae')} | matched r {b['singlet'].get('pearson')} vs REDItools3 {b['reditools3_downsampled_to_singlet_umis'].get('pearson')}")
EOF
else
  echo "neutral REDItools3 sample for $SC not ready; score later with this script (it skips the build and the run)"
fi
