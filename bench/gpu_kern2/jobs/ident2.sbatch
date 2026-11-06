#!/bin/bash
#SBATCH --job-name=gk2-ident
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-kern2: identity of WSHRINK, BSTREAM and both against the reference binary b0 (no switch), one sample, one run each, T=14, one slot, memory budget MEMGB.
# Never g050. Not exclusive. Bench of bench/gpu_bold/jobs/ident.sbatch (gpu/bold-kern) with the reference arm on another binary and several switch arms.
# Parameters (--export): BINDIR + EXPECT_COMMIT (the kern2 build), REFDIR + REFCOMMIT (b0), SAMPLE (pbmc1k_v3), MEMGB (45).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; REFDIR=${REFDIR:?}; REFBIN=$REFDIR/singlet-gpu; REFCOMMIT=${REFCOMMIT:?}
SAMPLE=${SAMPLE:-pbmc1k_v3}; MEMGB=${MEMGB:-45}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
RC=$(cat $REFDIR/commit-gpu); case $RC in $REFCOMMIT*) ;; *) echo "REFUSED: reference commit $RC"; exit 2 ;; esac
case $(hostname -s) in g050) echo "REFUSED: g050"; exit 2 ;; esac
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT ref=$REFBIN refsha256=$(sha256sum $REFBIN | cut -d' ' -f1) refcommit=$RC sample=$SAMPLE memgb=$MEMGB $(date -Is)"
R=$W/jobs/gpu-kern2/results/ident-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gk2id-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/bench/gpu-batch1:refs/gk2/id-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/gk2/id-$SLURM_JOB_ID bench/star_accel/out_equiv.py | tar -x -C $SCR; git -C $BARE update-ref -d refs/gk2/id-$SLURM_JOB_ID
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
case $SAMPLE in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; *) G=$REF/GRCh38-2024-A ;; esac
src=$PB/data/$SAMPLE/$SAMPLE.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$SAMPLE/$SAMPLE.1fq
cp $src $SCR/$SAMPLE.1fq
run() {   # arm binary env...
  local arm=$1 bin=$2; shift 2; local D=$SCR/$arm; mkdir -p $D/out
  ( cd $D && env "$@" /usr/bin/time -v -o $R/$arm.time.txt $bin $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $D/out/ --threads 14 --pipeline --output-format 1pz --engine gpu > $R/$arm.log 2>&1 ); echo "arm $arm exit $?"
  grep -a "kern2\|E17 lanes\|\[engine_gpu\] level" $R/$arm.log $D/out/star_Log.out | cut -c1-1800 | tee -a $R/$arm.marks
}
EXPECT_DIFF=" pileup_stats.json read_stats.tsv star_Log.std.out summary.json nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2 "
compare() {   # label
  local lab=$1 a=$SCR/ref/out b=$SCR/$1/out ok=1
  python3 $SCR/bench/star_accel/out_equiv.py $a $b > $R/equiv-$lab.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $R/equiv-$lab.txt | cut -c1-220
  for f in $(grep -a '^  DIFF ' $R/equiv-$lab.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) echo "UNEXPECTED DIFF $f"; ok=0 ;; esac; done
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "star_Solo.out/$f DIFFERS"; ok=0; }; done
  cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "Log.final.out statistics DIFFER"; ok=0; }
  cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "SJ.out.tab DIFFERS"; ok=0; }
  for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    ha=$([ -f $a/$f ] && paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
    hb=$([ -f $b/$f ] && paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
    [ "$ha" = "$hb" ] || { echo "$f DIFFERS ($ha vs $hb)"; ok=0; }; done
  local ca=$a/star_Solo.out/Gene/CellReads.stats cb=$b/star_Solo.out/Gene/CellReads.stats
  if [ -f $ca ] && [ -f $cb ]; then { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "CellReads.stats rows DIFFER"; ok=0; }; else echo "CellReads.stats absent (not compared)"; ok=0; fi
  [ $ok = 1 ] && echo "VERDICT IDENTICAL ($SAMPLE, arm $lab, against b0)" || echo "VERDICT NOT IDENTICAL ($SAMPLE, arm $lab, against b0)"
  echo "wall $lab $(grep -a 'Elapsed (wall' $R/$lab.time.txt)"
  rm -rf $SCR/$lab
}
run ref $REFBIN X_ARM=ref
echo "wall ref $(grep -a 'Elapsed (wall' $R/ref.time.txt)"
run base $BIN X_ARM=base;                                                      compare base
run wshrink $BIN SINGLET_BOLD_WSHRINK=1;                                       compare wshrink
run bstream $BIN SINGLET_BOLD_BSTREAM=1;                                       compare bstream
run both $BIN SINGLET_BOLD_WSHRINK=1 SINGLET_BOLD_BSTREAM=1;                   compare both
echo done $(date -Is)
