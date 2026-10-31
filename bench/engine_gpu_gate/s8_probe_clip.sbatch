#!/bin/bash
#SBATCH --job-name=s8-probeclip
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: probe gate with a constant 5' prefix: the first 2M spots of SRR34789664 (mouse 10x v3, real reads; they carry no natural prefix, the old probe
# gives clip 0) with a fixed 36 nt constant prepended to every cDNA read: a determinism check, not a benchmark. Expected clip = 36.
# Encodes R1/R2 to .1fq and stages it to .1fg2; runs the new probe (BINDIR_N) on .1fq/.1fg2 x FIFO/file decode and the old probe
# (BINDIR_O, before the content sampler) on .1fq FIFO/file decode, each stopped at the probe line. Also compiles and runs test_prefix_probe.
# Gate: the new probe line is identical in the four runs and its clip equals the old probe's. Both binaries must carry the commit given.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCm39-2024-A
BINDIR_N=${BINDIR_N:?}; BINDIR_O=${BINDIR_O:?}; WANT_N=${WANT_N:?}; WANT_O=${WANT_O:?}
[ "$(cat $BINDIR_N/commit)" = "$WANT_N" ] && [ "$(cat $BINDIR_O/commit)" = "$WANT_O" ] || { echo "!! commit mismatch: $(cat $BINDIR_N/commit) $(cat $BINDIR_O/commit)"; exit 3; }
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
D=$W/modality-bench/clip_positive; SCR=/tmp/$USER-s8pc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT
SRC=$SCR/src; mkdir -p $SRC; git -C $W/repos/singlet-py.git archive refs/s8/mt2 include tests/cpp/test_prefix_probe.cpp | tar -x -C $SRC
RES=$W/jobs/s8/results/probeclip-$SLURM_JOB_ID; mkdir -p $RES; S=$RES/summary.txt
for b in $BINDIR_N $BINDIR_O; do [ -e $b/../whitelists ] || ln -s $PB/whitelists $b/../whitelists; done
PFX=ACGTTGCAAGGCTTAACCGGATATCGCGTACGTAGC; Q=$(printf "I%.0s" $(seq 1 36))
zcat $D/R1.fastq.gz | head -n 8000000 | gzip -1 > $SCR/R1.fastq.gz
zcat $D/R2.fastq.gz | head -n 8000000 | awk -v p=$PFX -v q=$Q 'NR%4==2{print p $0; next} NR%4==0{print q $0; next} {print}' | gzip -1 > $SCR/R2.fastq.gz
$BINDIR_N/singlet-gpu encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq --whitelist-dir $PB/whitelists > $RES/encode.log 2>&1 || { echo "!! encode failed"; exit 4; }
$W/jobs/s9/fixtures/singlet-stage --whitelist-dir $PB/whitelists --accessions SRR34789664 --scratch $SCR -o $SCR/s.1fg2 --verify $SCR/s.1fq > $RES/stage.json 2> $RES/stage.err || { echo "!! stage failed"; tail -3 $RES/stage.err; exit 5; }
ls -l $SCR/s.1fq $SCR/s.1fg2 | tee -a $S
( cd $SCR && g++ -O1 -std=c++17 -I$SRC/include -o tpp $SRC/tests/cpp/test_prefix_probe.cpp 2>&1 | tail -3; cd $SRC && $SCR/tpp | tee -a $S ) 2>&1 | tail -3
probe() {  # tag bindir input file_decode
  unset SINGLET_FILE_DECODE; [ "$4" = 1 ] && export SINGLET_FILE_DECODE=1
  mkdir -p $SCR/$1/out; L=$SCR/$1/run.log
  ( cd $SCR/$1 && exec $2/singlet-gpu $3 --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$1/out/ --threads 12 --pipeline --output-format 1pz > $L 2>&1 ) &
  P=$!
  for i in $(seq 1 2400); do grep -aq "5' prefix probe" $L 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 2; done
  pkill -P $P 2>/dev/null; kill $P 2>/dev/null; wait $P 2>/dev/null
  echo "$1: $(grep -a "5' prefix probe" $L | head -1)" | tee -a $S
}
probe n_fq $BINDIR_N $SCR/s.1fq 0; probe n_fg2 $BINDIR_N $SCR/s.1fg2 0; probe n_fq_file $BINDIR_N $SCR/s.1fq 1; probe n_fg2_file $BINDIR_N $SCR/s.1fg2 1
probe o_fq $BINDIR_O $SCR/s.1fq 0; probe o_fq_file $BINDIR_O $SCR/s.1fq 1
nn=$(grep -a "^n_.*probe:" $S | sed 's/^[^:]*: *//' | sort -u | wc -l); clipn=$(grep -a "^n_fq:" $S | grep -o "clip [0-9]* bp")
clipo=$(grep -a "^o_fq:" $S | grep -o "clip [0-9]* bp"); clipof=$(grep -a "^o_fq_file:" $S | grep -o "clip [0-9]* bp")
echo "new distinct lines=$nn new=$clipn old_fifo=$clipo old_file=$clipof" | tee -a $S
[ "$nn" = 1 ] && [ "$clipn" = "clip 36 bp" ] && [ "$clipn" = "$clipo" ] && [ "$clipn" = "$clipof" ] && echo "PROBE CLIP GATE OK" | tee -a $S || echo "!! PROBE CLIP GATE: see lines above" | tee -a $S
