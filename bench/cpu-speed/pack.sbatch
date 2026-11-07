#!/bin/bash
#SBATCH --job-name=cs-pack
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Node packing with one shared genome: `singlet genome load` once, then one run alone and PACK runs at once, each --threads T --genome-shared on.
# Env: BD (binary dir under $W/jobs/cpu-speed/bin), SAMPLE (default pbmc1k_v3), T (default 4), PACK (default cpus/T), SHARED (1 = shared genome, 0 = private).
# Per run: wall, user+sys (time -v), peak RSS, and the anonymous (private) peak of every singlet process sampled every 5 s from smaps_rollup.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BD=${BD:?}; S=${SAMPLE:-pbmc1k_v3}; T=${T:-4}; PACK=${PACK:-$((SLURM_CPUS_PER_TASK / T))}; SH=${SHARED:-1}
BIN=$W/jobs/cpu-speed/bin/$BD/singlet-cpu; G=$REF/GRCh38-2024-A
FQ=$PB/data/$S/$S.1fq; [ -f $FQ ] || FQ=$W/data/atlas-rerun/perf/$S/sample.1fq
RES=$W/jobs/cpu-speed/results/pack-$S-$SLURM_JOB_ID; mkdir -p $RES; SCR=/tmp/$USER-cspk-$SLURM_JOB_ID; mkdir -p $SCR
GS=off; [ $SH = 1 ] && GS=on
cleanup() { kill $(jobs -p) 2>/dev/null; [ $SH = 1 ] && $BIN genome unload $G/star_2.7.11b > /dev/null 2>&1; rm -rf "${SCR:?}"; }
trap cleanup EXIT INT TERM
export SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BD sample=$S T=$T pack=$PACK shared=$SH $(lscpu | grep 'Model name' | sed 's/  */ /g') $(free -g | awk '/Mem/{print "mem_total_gb="$2}') $(date -Is)" | tee $RES/summary.txt
if [ $SH = 1 ]; then L0=$(date +%s.%N); $BIN genome load $G/star_2.7.11b > $RES/genome_load.txt 2>&1; echo "genome load: exit $? in $(echo "$(date +%s.%N) - $L0" | bc) s" | tee -a $RES/summary.txt; fi
( while true; do echo "@ $(date +%s)"; for p in $(pgrep -f "singlet-cpu $FQ"); do [ -r /proc/$p/smaps_rollup ] && echo "$p $(awk '/^Anonymous:|^Rss:/{printf "%s=%s ",$1,$2}' /proc/$p/smaps_rollup)"; done; sleep 5; done ) > $RES/mem.txt 2>/dev/null &
one() {   # $1 = tag
  mkdir -p $SCR/$1/out
  ( cd $SCR/$1 && /usr/bin/time -v -o $RES/time_$1.txt $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$1/out/ --threads $T --genome-shared $GS --pipeline --output-format 1pz --engine cpu > $RES/log_$1.txt 2>&1
    cp $SCR/$1/out/star_Log.final.out $RES/final_$1.txt 2>/dev/null; rm -rf $SCR/$1 )
}
row() { echo "$1 wall=$(awk -F': ' '/Elapsed/{print $2}' $RES/time_$1.txt) cpu_s=$(awk -F': ' '/User time|System time/{s+=$2} END{print s}' $RES/time_$1.txt) maxrss_gb=$(awk -F': ' '/Maximum resident/{printf "%.1f", $2/1048576}' $RES/time_$1.txt) exit=$(awk -F': ' '/Exit status/{print $2}' $RES/time_$1.txt) shared_found=$(grep -ac 'Found genome in shared memory' $RES/log_$1.txt)" | tee -a $RES/summary.txt; }
S0=$(date +%s.%N); one solo; echo "solo: $(echo "$(date +%s.%N) - $S0" | bc) s" | tee -a $RES/summary.txt; row solo
P0=$(date +%s.%N)
for i in $(seq 1 $PACK); do one p$i & done; wait $(jobs -p | tail -n $PACK) 2>/dev/null
for i in $(seq 1 $PACK); do while [ -d $SCR/p$i ]; do sleep 2; done; done
echo "pack of $PACK: $(echo "$(date +%s.%N) - $P0" | bc) s" | tee -a $RES/summary.txt
for i in $(seq 1 $PACK); do row p$i; done
echo "reads: $(awk -F'|' '/Number of input reads/{gsub(/[ \t]/,"",$2); print $2}' $RES/final_solo.txt)" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
