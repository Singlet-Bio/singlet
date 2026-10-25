#!/bin/bash
#SBATCH --job-name=bp-ap-opt
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --exclusive
#SBATCH --constraint=xeon-6248
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch opt_run.sbatch KIND REP
#     KIND star  the STAR-level optimizations of PR #34, one binary per row, 4 threads, paired round-robin on one node
#          pipe  the pipeline-level settings of docs/atlas-rerun-plan.md section 3, on the pinned campaign binary
# Dataset: PBMC 1k v3 (66.6 M read pairs, GRCh38-2024-A), the pre-encoded .1fq, copied to node-local disk before the clock starts.
# Exclusive node of one CPU type (Xeon Gold 6248). Output: $BP/perf_opt/KIND/NAME_rREP/{time.txt,mem.json,star_Log.out,...}.
#
# star rows (name = binary[@ENV=VALUE]); all run `--threads 4 --pipeline --output-format mtx --raw-matrix`:
#   stock-star       singlet-accel-nodedup @ SINGLET_STAR_SEQCACHE_MB=0   stock STAR (adjacent-read dedup compiled out)
#   pr22             singlet-fixed-det                                    the binary of PR #22 (adjacent-read dedup on)
#   lazy-huge-cache1 singlet-accel-lh @ SINGLET_STAR_SEQCACHE_MB=1024     lazy winBin + huge pages + readLoad + 1 GB cache
#   pgo-cache2       singlet-accel-finpgo                                 + PGO, 2 GB cache (the PR #34 default)
#   campaign         singlet-campaign                                     the pinned production binary
#   campaign-nocache singlet-campaign @ SINGLET_STAR_SEQCACHE_MB=0       the same with the sequence cache off
# Note: the PR #22 / #34 binaries clip 11 bp from PBMC R2 reads (the 5' prefix false positive fixed in the campaign binary),
# so those rows map 80 bp reads and the campaign rows map 91 bp reads: compare rows within a group, not across it.
set -eu
source /mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet/bench/public/harness/common.sh
KIND=$1; REP=$2; verify_bin
DS=pbmc1k_v3; REF=$REFBASE/GRCh38-2024-A; GENOME=$REF/star_2.7.11b; GTF=$REF/genes/genes.gtf
O0=$BP/perf_opt/$KIND; mkdir -p $O0
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; SHMBIN=""
trap 'rm -rf $SCR; [ -n "$SHMBIN" ] && $SHMBIN genome unload $GENOME >/dev/null 2>&1; true' EXIT
echo "[opt] $KIND rep $REP node=$(hostname) cpu=$(cpu_model) $(date -Is)"
cp $PB/data/$DS/$DS.1fq $SCR/s.1fq
cat $GENOME/* > /dev/null 2>&1 || true
unset SINGLET_REF_BASE
export SINGLET_STALL_TIMEOUT=10800

one() {  # NAME BIN THREADS ENV(k=v,...) EXTRA...  -> $O0/NAME_rREP
  local name=$1 bin=$2 thr=$3 envs=$4; shift 4
  local O=$O0/${name}_r$REP; rm -rf $O $O.keep; mkdir -p $O
  echo "[opt] $name $(date -Is)" > $O/node.txt
  ( for kv in ${envs//,/ }; do export "$kv"; done
    cd $SCR; timed $O/run $bin $SCR/s.1fq --ref-base $REFBASE --genome-dir $GENOME --exons $GTF --out-prefix $O/out/ \
      --threads $thr --pipeline --output-format mtx --raw-matrix "$@" > $O/run.log 2>&1 ) || echo "exit $?" >> $O/run.log
  mkdir -p $O.keep; cp $O/out/star_Log.out $O/out/star_Log.final.out $O.keep/ 2>/dev/null || true
  rm -rf $O/out
  echo "[opt] $name: $(grep -h Elapsed $O/run/time.txt | sed 's/.*): //') rss $(cat $O/run/mem.json 2>/dev/null)"
}

if [ $KIND = star ]; then
  A=$PB/bin
  ROWS=("stock-star|$A/singlet-accel-nodedup|SINGLET_STAR_SEQCACHE_MB=0" "pr22|$A/singlet-fixed-det|" \
        "lazy-huge-cache1|$A/singlet-accel-lh|SINGLET_STAR_SEQCACHE_MB=1024" "pgo-cache2|$A/singlet-accel-finpgo|" \
        "campaign|$BIN|" "campaign-nocache|$BIN|SINGLET_STAR_SEQCACHE_MB=0")
  if [ $(( REP % 2 )) = 0 ]; then
    R=(); for ((i=${#ROWS[@]}-1;i>=0;i--)); do R+=("${ROWS[$i]}"); done; ROWS=("${R[@]}")
  fi
  for row in "${ROWS[@]}"; do IFS='|' read -r name bin envs <<< "$row"; one $name $bin 4 "$envs"; done
else
  CFGS="t4 t8 t16 t4_shm t4_mem x4_lean t4_nonhost"
  [ $(( REP % 2 )) = 0 ] && CFGS=$(for c in $CFGS; do echo $c; done | tac | tr '\n' ' ')
  for c in $CFGS; do
    case $c in
      t4) one t4 $BIN 4 "" ;;
      t8) one t8 $BIN 8 "" ;;
      t16) one t16 $BIN 16 "" ;;
      t4_mem) one t4_mem $BIN 4 "" --bam-sort-mode memory ;;
      t4_nonhost) one t4_nonhost $BIN 4 "SINGLET_REF_BASE=$REFBASE" ;;
      t4_shm)
        SHMBIN=$BIN; $BIN genome load $GENOME > $O0/genome_load_r$REP.log 2>&1
        one t4_shm $BIN 4 "" --genome-shared on; $BIN genome unload $GENOME > /dev/null 2>&1; SHMBIN="" ;;
      x4_lean)  # 4 concurrent samples, one shared genome, disk sort
        O=$O0/x4_lean_r$REP; rm -rf $O $O.keep; mkdir -p $O; SHMBIN=$BIN
        $BIN genome load $GENOME > $O/genome_load.log 2>&1
        t0=$(date +%s)
        for i in 1 2 3 4; do ( cd $SCR; timed $O/c$i/run $BIN $SCR/s.1fq --ref-base $REFBASE --genome-dir $GENOME --exons $GTF \
            --out-prefix $O/c$i/out/ --threads 4 --pipeline --output-format mtx --raw-matrix --genome-shared on \
            --bam-sort-mode disk > $O/c$i.log 2>&1 ) & done; wait
        echo "wall_all=$(( $(date +%s) - t0 ))" > $O/node.txt
        mkdir -p $O.keep; cp $O/c1/out/star_Log.out $O/c1/out/star_Log.final.out $O.keep/ 2>/dev/null || true
        rm -rf $O/c?/out; $BIN genome unload $GENOME > /dev/null 2>&1; SHMBIN=""
        echo "[opt] x4_lean $(cat $O/node.txt)" ;;
    esac
  done
fi
echo "[opt] done $(date -Is)"
