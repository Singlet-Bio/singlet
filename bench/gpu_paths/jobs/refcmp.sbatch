#!/bin/bash
#SBATCH --job-name=gb1-refcmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compare the kept CPU references of two commits, per sample, with the gate's own checks (gate.sbatch cmp2):
# out_equiv.py, the 10 STAR Solo files by cmp, star_Log.final.out, SJ.out.tab, the sorted read-file hashes, CellReads.stats by sorted rows, fieldcheck.py.
# Needs COMMIT_A, COMMIT_B (8+ hex digits). T default 18. SAMPLES default: the gate's ten.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
CA=${COMMIT_A:?}; CB=${COMMIT_B:?}; T=${T:-18}
SAMPLES=(${SAMPLES:-pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc1k_cite_v3_adt pbmc5p_v1 neuron1k_v3 hgmm1k_v3 nuclei2k_v2})
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-batch1}
RES0=$W/jobs/gpu-batch1/results/refcmp-$SLURM_JOB_ID; mkdir -p $RES0
SCR=/tmp/$USER-gb1c-$SLURM_JOB_ID; mkdir -p $SCR/repo; SREF=refs/gb1/refcmp-$SLURM_JOB_ID
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$SBR | cut -f1); [ -n "$WANT" ] || { echo "no scripts branch $SBR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
SFULL=$(git -C $BARE rev-parse $SREF); [ "$SFULL" = "$WANT" ] || { echo "fetched $SFULL, branch tip is $WANT"; exit 3; }
[ -z "${EXPECT_SCRIPTS:-}" ] || [[ $SFULL == $EXPECT_SCRIPTS* ]] || { echo "scripts commit $SFULL, EXPECT_SCRIPTS is $EXPECT_SCRIPTS"; exit 3; }
git -C $BARE archive $SREF bench/star_accel/out_equiv.py bench/gpu_batch1 | tar -x -C $SCR/repo
GB=$SCR/repo/bench/gpu_batch1
export PATH=$W/envs/pbench/bin:$PATH
RA=$W/jobs/gpu-batch1/ref/${CA:0:8}/T$T; RB=$W/jobs/gpu-batch1/ref/${CB:0:8}/T$T
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) refA=$RA refB=$RB scripts=${SFULL:0:8} T=$T $(date -Is)" | tee $RES0/summary.txt
EXPECT_DIFF=" pileup_stats.json read_stats.tsv star_Log.std.out summary.json "
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
printf "sample\tverdict\tout_equiv\tfield_unexplained\n" > $RES0/verdicts.tsv
FAIL=0
for s in "${SAMPLES[@]}"; do
  a=$RA/$s/out; b=$RB/$s/out; RES=$RES0/$s; mkdir -p $RES; ok=1
  if [ ! -f $RA/$s/DONE ] || [ ! -f $RB/$s/DONE ]; then
    echo "== $s: missing reference" | tee -a $RES0/summary.txt
    printf "%s\tMISSING\t-\t-\n" $s >> $RES0/verdicts.tsv; FAIL=1; continue
  fi
  echo "=== $s: $(head -1 $RA/$s/DONE) vs $(head -1 $RB/$s/DONE)" | tee -a $RES0/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $RES/equiv.txt | cut -c1-200 | head -60 | tee -a $RES0/summary.txt
  FL=$(grep -a '^files:' $RES/equiv.txt | sed -E 's/files: ([0-9]+) identical, ([0-9]+) differ only in timings\/paths, ([0-9]+) differ/\1\/\2\/\3/')
  UNX=""
  for f in $(grep -a '^  DIFF ' $RES/equiv.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do
    case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac
  done
  if grep -aq '^  DIFF provenance.json: /references/whitelist/name: [^;]*$' $RES/equiv.txt; then
    UNX=${UNX/ provenance.json/}; echo "  provenance.json: the whitelist path only" | tee -a $RES0/summary.txt
  fi
  for f in $(grep -a '^  only in ' $RES/equiv.txt | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
  [ -n "$FL" ] || UNX="$UNX out_equiv-failed"
  if [ -n "$UNX" ]; then echo "  out_equiv: UNEXPECTED:$UNX" | tee -a $RES0/summary.txt; ok=0
  else echo "  out_equiv: $FL (identical/timing-path/differ), every DIFF in the run-dependent set" | tee -a $RES0/summary.txt; fi
  ns=0
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
           Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    if cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f; then ns=$((ns+1)); else echo "  star_Solo.out/$f: DIFFERS" | tee -a $RES0/summary.txt; ok=0; fi
  done
  echo "  star_Solo.out: $ns of 10 files identical" | tee -a $RES0/summary.txt
  if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then
    echo "  star_Log.final.out: mapping statistics identical (timing lines excluded)" | tee -a $RES0/summary.txt
  else
    echo "  star_Log.final.out: MAPPING STATISTICS DIFFER" | tee -a $RES0/summary.txt; ok=0
    diff <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) | head -20 | tee -a $RES0/summary.txt
  fi
  if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then echo "  star_SJ.out.tab: identical ($(wc -l < $a/star_SJ.out.tab) junctions)" | tee -a $RES0/summary.txt
  else echo "  star_SJ.out.tab: DIFFERS" | tee -a $RES0/summary.txt; ok=0; fi
  ra=$RA/$s/readhash.txt; rb=$RB/$s/readhash.txt
  if [ -f $ra ] && [ -f $rb ]; then
    for f in $READF; do
      ha=$(awk -v f=$f '$1==f{print $2" "$3}' $ra); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rb)
      if [ "$ha" = "$hb" ]; then echo "  $f: same records (sorted sha256 and reads: $ha)" | tee -a $RES0/summary.txt
      else echo "  $f: RECORDS DIFFER ($ha vs $hb)" | tee -a $RES0/summary.txt; ok=0; fi
    done
  else echo "  read files: not compared (no sorted hashes)" | tee -a $RES0/summary.txt; fi
  ca=$RA/$s/CellReads.stats; cb=$RB/$s/CellReads.stats
  if [ -f $ca ] && [ -f $cb ]; then
    if cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then
      echo "  Gene/CellReads.stats: same rows (sorted; $(( $(wc -l < $ca) - 1 )) rows; byte-identical in file order: $(cmp -s $ca $cb && echo yes || echo no))" | tee -a $RES0/summary.txt
    else
      echo "  Gene/CellReads.stats: ROWS DIFFER ($(wc -l < $ca) vs $(wc -l < $cb) lines)" | tee -a $RES0/summary.txt; ok=0
      diff <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort) | head -8 | cut -c1-200 | tee -a $RES0/summary.txt
    fi
  else echo "  Gene/CellReads.stats: not compared (missing copy)" | tee -a $RES0/summary.txt; fi
  if [ $ok = 1 ]; then echo "VERDICT $s ${CA:0:8}:${CB:0:8} IDENTICAL (except the run-dependent files; out_equiv $FL)" | tee -a $RES0/summary.txt
  else echo "VERDICT $s ${CA:0:8}:${CB:0:8} NOT IDENTICAL" | tee -a $RES0/summary.txt; FAIL=1; fi
  echo "  field check $s" | tee -a $RES0/summary.txt
  FC=$(python3 $GB/fieldcheck.py $a $b 2>&1); echo "$FC" | tee -a $RES0/summary.txt $RES/fieldcheck.txt
  FN=$(echo "$FC" | sed -n 's/^FIELD CHECK: \([0-9]*\) unexplained.*/\1/p')
  printf "%s\t%s\t%s\t%s\n" $s "$([ $ok = 1 ] && echo IDENTICAL || echo DIFFERENT)" "$FL" "${FN:-?}" >> $RES0/verdicts.tsv
done
echo "######## verdicts" | tee -a $RES0/summary.txt; column -t -s $'\t' $RES0/verdicts.tsv | tee -a $RES0/summary.txt
[ $FAIL = 0 ] && echo "REFCMP: all identical" | tee -a $RES0/summary.txt || echo "REFCMP: differences or missing references" | tee -a $RES0/summary.txt
echo done $(date -Is) | tee -a $RES0/summary.txt
