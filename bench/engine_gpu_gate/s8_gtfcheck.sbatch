#!/bin/bash
#SBATCH --job-name=s8-gtfcheck
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: the gene model next to an explicit --genome-dir (BINDIR = the build). Runs the same .1fq with --exons (A), without --exons (B, CPU) and without --exons on --engine gpu (C), and without any of
# --exons / --genome-dir (D, must stop with the new message). A, B, C must be identical (out_equiv.py); B and C must log the GTF they found.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; FQ=$W/jobs/s9/fixtures/SRR19421372.1fq; T=16
[ -d $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 PATH=$W/envs/pbench/bin:$PATH
O=$W/s8/gtfcheck_$(cut -c1-8 $BINDIR/commit); rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf $SCR' EXIT
git -C $W/repos/singlet-py.git archive refs/s8/campaign bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
echo "commit=$(cat $BINDIR/commit) node=$(hostname)"
run() { n=$1; shift; mkdir -p $SCR/$n/out; ( cd $SCR/$n && $BIN $FQ --ref-base $REF --out-prefix $SCR/$n/out/ --threads $T --pipeline --output-format 1pz "$@" > $O/$n.log 2>&1 ); echo "$n rc=$?"; }
run A --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf
run B --genome-dir $G/star_2.7.11b
SINGLET_GPU_PILEUP=on run C --genome-dir $G/star_2.7.11b --engine gpu
grep -h "GTF next to" $O/B.log $O/C.log
mkdir -p $SCR/D/out; ( cd $SCR/D && $BIN $FQ --out-prefix $SCR/D/out/ --threads 4 --pipeline --protocol 10x-3p-v3 > $O/D.log 2>&1 ); echo "D rc=$? (expected 1)"; grep -h "ERROR" $O/D.log | head -2
for p in "A B" "A C"; do set -- $p; echo "=== $1 vs $2"; python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$1/out $SCR/$2/out 2>&1 | grep -a '^  DIFF\|^files:\|^RESULT' | head -12; done
