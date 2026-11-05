#!/bin/bash
#SBATCH --job-name=gbold-ident
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Identity of a gpu/bold prototype switch against the same binary without it, on one sample, one run each, T=14, one slot.
# Parameters (--export): BINDIR, EXPECT_COMMIT (8+ hex, prefix of BINDIR/commit-gpu), SAMPLE (default pbmc1k_v3), SWITCH ("NAME=value ..." env of arm B).
# Arm A = no switch. Arm B = switch. Compared as the batch 1 gate does: out_equiv.py with the run-dependent files excused, the ten Solo files,
# Log.final.out statistics, SJ.out.tab, the three read files by sorted hash, CellReads.stats rows sorted.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; SAMPLE=${SAMPLE:-pbmc1k_v3}; SWITCH=${SWITCH:?}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
BSHA=$(sha256sum $BIN | cut -d' ' -f1)
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$BSHA commit=$COMMIT sample=$SAMPLE switch=[$SWITCH] $(date -Is)"
R=$W/jobs/gpu-bold/results/ident-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gbid-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/bench/gpu-batch1:refs/gbold/id-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/gbold/id-$SLURM_JOB_ID bench/star_accel/out_equiv.py | tar -x -C $SCR; git -C $BARE update-ref -d refs/gbold/id-$SLURM_JOB_ID
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1
case $SAMPLE in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; *) G=$REF/GRCh38-2024-A ;; esac
src=$PB/data/$SAMPLE/$SAMPLE.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$SAMPLE/$SAMPLE.1fq
cp $src $SCR/$SAMPLE.1fq
run() {   # arm label env...
  local arm=$1; shift; local D=$SCR/$arm; mkdir -p $D/out
  ( cd $D && env "$@" /usr/bin/time -v -o $R/$arm.time.txt $BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $D/out/ --threads 14 --pipeline --output-format 1pz --engine gpu > $R/$arm.log 2>&1 ); echo "arm $arm exit $?"
  grep -a "NOHOSTCOPY\|\[engine_gpu\] level\|device .* held it\|E12 device -> host" $R/$arm.log $D/out/star_Log.out | cut -c1-260 | tee -a $R/$arm.marks
}
run A X_ARM=A
if [ "${MPS_B:-0}" = 1 ]; then   # arm B under CUDA MPS (all processes of the arm run as this user)
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/gbmps-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=$R/mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "MPS daemon started for arm B"
fi
# shellcheck disable=SC2086
run B X_ARM=B $SWITCH
[ "${MPS_B:-0}" = 1 ] && { echo quit | nvidia-cuda-mps-control; echo "MPS daemon stopped"; ls $R/mps-log 2>/dev/null; tail -5 $R/mps-log/control.log 2>/dev/null; tail -5 $R/mps-log/server.log 2>/dev/null; }
a=$SCR/A/out; b=$SCR/B/out; ok=1
EXPECT_DIFF=" pileup_stats.json read_stats.tsv star_Log.std.out summary.json nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2 "   # the three read files are compared by sorted record hash below (their order differs by design)
python3 $SCR/bench/star_accel/out_equiv.py $a $b > $R/equiv.txt 2>&1
grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $R/equiv.txt | cut -c1-220
for f in $(grep -a '^  DIFF ' $R/equiv.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) echo "UNEXPECTED DIFF $f"; ok=0 ;; esac; done
for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
  cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "star_Solo.out/$f DIFFERS"; ok=0; }; done
cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "Log.final.out statistics DIFFER"; ok=0; }
cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "SJ.out.tab DIFFERS"; ok=0; }
for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
  ha=$([ -f $a/$f ] && paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
  hb=$([ -f $b/$f ] && paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
  [ "$ha" = "$hb" ] || { echo "$f DIFFERS ($ha vs $hb)"; ok=0; }; done
ca=$a/star_Solo.out/Gene/CellReads.stats; cb=$b/star_Solo.out/Gene/CellReads.stats
if [ -f $ca ] && [ -f $cb ]; then { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "CellReads.stats rows DIFFER"; ok=0; }; else echo "CellReads.stats absent (not compared)"; fi
[ $ok = 1 ] && echo "VERDICT IDENTICAL ($SAMPLE, switch [$SWITCH])" || echo "VERDICT NOT IDENTICAL ($SAMPLE, switch [$SWITCH])"
echo "wall A $(grep -a 'Elapsed (wall' $R/A.time.txt)  B $(grep -a 'Elapsed (wall' $R/B.time.txt)"
echo done $(date -Is)
