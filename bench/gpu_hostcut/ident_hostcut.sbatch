#!/bin/bash
#SBATCH --job-name=ghc-ident
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=250G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-hostcut identity of the single-donor demux screen. Per sample (SAMPLES, default "pbmc1k_v3 pbmc1k_v2"), three runs of ONE binary:
#   off  = --engine gpu, SINGLET_DEMUX_SCREEN=0   (the reference)
#   on   = --engine gpu, SINGLET_DEMUX_SCREEN=1
#   cpu  = --engine cpu, SINGLET_DEMUX_SCREEN=1   (the CPU engine; the GPU variables are unset)
# Compared: on against off, on against cpu, with the same checks as ident2.sbatch (out_equiv.py with the by-name timing list, star_Solo files, SJ,
# read files, CellReads.stats) plus a byte comparison of donor_assignments.tsv and the donor/ directory. Not exclusive.
# Parameters (--export): BINDIR + EXPECT_COMMIT (the build), SAMPLES, T (14), MEMGB (45). ALLOW_NO_EMBED=1 is accepted (the commit check reads commit-gpu).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}
SAMPLES=${SAMPLES:-"pbmc1k_v3 pbmc1k_v2"}; T=${T:-14}; MEMGB=${MEMGB:-45}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
EMBED=no; grep -aqF "$COMMIT" $BIN && EMBED=yes
[ $EMBED = yes ] || [ "${ALLOW_NO_EMBED:-0}" = 1 ] || { echo "REFUSED: commit $COMMIT is not inside the binary (ALLOW_NO_EMBED=1 skips this check)"; exit 2; }
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT embedded=$EMBED samples=$SAMPLES T=$T memgb=$MEMGB $(date -Is)"
R=$W/jobs/gpu-hostcut/results/ident-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-ghcid-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/bench/gpu-batch1:refs/ghc/id-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/ghc/id-$SLURM_JOB_ID bench/star_accel/out_equiv.py | tar -x -C $SCR; git -C $BARE update-ref -d refs/ghc/id-$SLURM_JOB_ID
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
GPUENV=(SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB)

EXPECT_DIFF=" pileup_stats.json read_stats.tsv star_Log.std.out summary.json nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2 "
OVERALL=0
run() {   # sample arm engine screen
  local s=$1 arm=$2 eng=$3 scr=$4 G D=$SCR/$1/$2
  case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; *) G=$REF/GRCh38-2024-A ;; esac
  mkdir -p $D/out
  local ENVV=()
  if [ $eng = gpu ]; then ENVV=("${GPUENV[@]}" SINGLET_DEMUX_SCREEN=$scr); else ENVV=(-u SINGLET_GPU_PILEUP -u SINGLET_ENGINE_GPU_E2E -u SINGLET_ENGINE_GPU_INGEST -u SINGLET_ENGINE_GPU_SLOTS -u SINGLET_ENGINE_GPU_MEM_GB SINGLET_DEMUX_SCREEN=$scr); fi
  ( cd $D && env "${ENVV[@]}" /usr/bin/time -v -o $R/$s.$arm.time.txt $BIN $SCR/$s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz --engine $eng > $R/$s.$arm.log 2>&1 ); local rc=$?
  echo "$s arm $arm (engine $eng, screen $scr) exit $rc, $(grep -a 'Elapsed (wall' $R/$s.$arm.time.txt | sed 's/.*: //')"
  grep -a "\[demux\] single-donor screen\|\[demux\] auto-K: selected\|\[demux\] Best K\|\[demux\] K=1 workaround\|\[demux\] Called cells" $R/$s.$arm.log | cut -c1-200 | sed "s/^/  [$arm] /" | tee -a $R/$s.demux-lines.txt
  [ $rc = 0 ] || OVERALL=1
}
compare() {   # sample A_arm B_arm   (B against A)
  local s=$1 la=$2 lb=$3 a=$SCR/$1/$2/out b=$SCR/$1/$3/out ok=1 tag=$1.$3-vs-$2
  python3 $SCR/bench/star_accel/out_equiv.py $a $b > $R/equiv-$tag.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $R/equiv-$tag.txt | cut -c1-220
  for f in $(grep -a '^  DIFF ' $R/equiv-$tag.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) echo "UNEXPECTED DIFF $f"; ok=0 ;; esac; done
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "star_Solo.out/$f DIFFERS"; ok=0; }; done
  cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "Log.final.out statistics DIFFER"; ok=0; }
  cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "SJ.out.tab DIFFERS"; ok=0; }
  for f in donor_assignments.tsv donor/donor_assignments.tsv; do
    cmp -s $a/$f $b/$f || { echo "$f DIFFERS (demux labels)"; ok=0; }; done
  if [ -d $a/private/donor ] && [ -d $b/private/donor ]; then
    diff -rq $a/private/donor $b/private/donor > $R/donor-private-$tag.txt 2>&1 || { echo "private/donor DIFFERS (see donor-private-$tag.txt)"; ok=0; }
  fi
  for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    ha=$([ -f $a/$f ] && paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
    hb=$([ -f $b/$f ] && paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16 || echo absent)
    [ "$ha" = "$hb" ] || { echo "$f DIFFERS ($ha vs $hb)"; ok=0; }; done
  local ca=$a/star_Solo.out/Gene/CellReads.stats cb=$b/star_Solo.out/Gene/CellReads.stats
  if [ -f $ca ] && [ -f $cb ]; then { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "CellReads.stats rows DIFFER"; ok=0; }; else echo "CellReads.stats absent (not compared)"; ok=0; fi
  [ $ok = 1 ] && echo "VERDICT IDENTICAL ($s, $lb against $la)" || { echo "VERDICT NOT IDENTICAL ($s, $lb against $la)"; OVERALL=1; }
}

for s in $SAMPLES; do
  src=$PB/data/$s/$s.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$s/$s.1fq
  [ -f $src ] || { echo "no .1fq for $s"; OVERALL=1; continue; }
  cp $src $SCR/$s.1fq
  run $s off gpu 0
  run $s on  gpu 1
  run $s cpu cpu 1
  compare $s off on  >> $R/verdicts.txt 2>&1
  compare $s cpu on  >> $R/verdicts.txt 2>&1
  cp $SCR/$s/off/out/donor_assignments.tsv $R/$s.donor_assignments.off.tsv 2>/dev/null
  rm -rf $SCR/$s $SCR/$s.1fq
done
cat $R/verdicts.txt
echo "OVERALL_IDENTITY $([ $OVERALL = 0 ] && echo PASS || echo FAIL)" | tee -a $R/verdicts.txt
echo done $(date -Is)
exit 0
