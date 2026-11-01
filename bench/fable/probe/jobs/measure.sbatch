#!/bin/bash
#SBATCH --job-name=e14-measure
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E14 (docs/gpu-engine/fable/E14-probe.md) step 1: what the "STAR load, CR4 probe, start" stage is made of, at T = 8 on debug-gpu.
#  facts   the node's SysV shared-memory limits, THP settings, /dev/shm, the job's cgroup memory
#  probes  the CR4 probe's STAR run alone (`singlet star`, 200k read pairs of pbmc1k_v3's own FASTQ): STAR's own genome load vs the
#          parallel preload, default vs 2 MB input chunks, 8 vs 4 threads, plain vs cr4 (rates and SJ.out.tab must not change)
#  shm     STAR's SysV genome (singlet genome load = LoadAndExit; LoadAndKeep attach; Remove): time, cgroup charge, and the
#          mapping speed of 2M read pairs from the shared genome vs the private one (transparent hugepages)
#  runs    level-5 pipeline runs (RUNS="new base"), each with the stage script, cr4_probe.json and the probe logs
# A genome this job puts into shared memory is removed on every exit path (trap), and only if it was not there before.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/GRCh38-2024-A/star_2.7.11b; WL=$PB/whitelists/3M-february-2018.txt; T=${T:-8}
BIN=${BIN:-$W/jobs/fable/probe/bin-n1/singlet-gpu}; BASE=${BASE:-$W/jobs/fable/probe/bin-base/singlet-gpu}
BARE=$W/repos/singlet-py.git
RES=$W/jobs/fable/probe/results/measure-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e14m-$SLURM_JOB_ID; mkdir -p $SCR/repo
SHM_OURS=0
cleanup() {
  if [ $SHM_OURS = 1 ]; then $BIN genome unload $G >> $RES/shm.txt 2>&1; SHM_OURS=0; fi
  { echo "exit: $(date -Is)"; ipcs -m; } >> $RES/shm.txt 2>&1
  kill $(jobs -p) 2>/dev/null; rm -rf $SCR
}
trap cleanup EXIT; trap 'exit 1' INT TERM
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
log() { echo "$*" | tee -a $RES/summary.txt; }
ts() { date +%s.%N; }
cgm() { awk '$1=="anon"||$1=="file"||$1=="shmem"{printf "%s %.1f GB, ", $1, $2/2^30}' $CG/memory.stat; printf "current %.1f GB" $(awk '{print $1/2^30}' $CG/memory.current); }
git -C $BARE archive refs/fable/probe-src bench/fable/solowire/s8_stages_e11.py | tar -x -C $SCR/repo
log "node=$(hostname) job=$SLURM_JOB_ID T=$T bin=$(cat $(dirname $BIN)/commit-gpu) base=$(cat $(dirname $BASE)/commit-gpu) $(date -Is)"

# ---- facts
{ hostname; nproc; lscpu | grep -E "Model name|^Socket|NUMA node\(s\)"; free -g
  for k in shmmax shmall shmmni; do echo "kernel.$k=$(cat /proc/sys/kernel/$k)"; done
  for f in enabled defrag shmem_enabled; do echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done
  echo "vm.nr_hugepages=$(cat /proc/sys/vm/nr_hugepages) vm.overcommit_memory=$(cat /proc/sys/vm/overcommit_memory)"
  df -h /dev/shm /tmp; ipcs -m; ipcs -lm; echo "cgroup $CG"; cat $CG/memory.max; cgm; echo
  mount | grep -E "cgroup|/dev/shm"; grep -vE "^#|^$" /etc/slurm/cgroup.conf 2>/dev/null; grep -E "Huge|Shmem" /proc/meminfo
  uname -r; cat /etc/os-release | head -2; } > $RES/facts.txt 2>&1
log "facts: $(grep -E "shmmax|shmall|shmem_enabled|thp enabled|memory.max" $RES/facts.txt | tr '\n' ' ')"
cat $RES/facts.txt

# ---- reads: the first 2M read pairs of pbmc1k_v3's FASTQ (R2 cDNA, R1 barcode), and the page cache of the index
F=$PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001
t0=$(ts); (zcat ${F}_R2_001.fastq.gz | head -n 8000000 > $SCR/R2.fq) & (zcat ${F}_R1_001.fastq.gz | head -n 8000000 > $SCR/R1.fq) & wait
log "reads: $(( $(wc -l < $SCR/R2.fq) / 4 )) pairs in $(echo "$(ts) - $t0" | bc) s"
t0=$(ts); cat $G/Genome $G/SA $G/SAindex > /dev/null; log "index read into the page cache (cat): $(echo "$(ts) - $t0" | bc) s"
t0=$(ts); cat $G/Genome $G/SA $G/SAindex > /dev/null; log "index read again (cached): $(echo "$(ts) - $t0" | bc) s"

pargs() { # profile threads readMapNumber genomeLoad
  local a="--runMode alignReads --genomeDir $G --readFilesIn $SCR/R2.fq $SCR/R1.fq --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloCBstart 1 --soloUMIstart 17 --soloBarcodeReadLength 0 --soloCellReadStats Standard --soloUMIlen 12"
  [ $1 = cr4 ] && a="$a --clipAdapterType CellRanger4 --alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated --outFilterMismatchNoverLmax 0.05"
  echo "$a --outFilterScoreMin 10 --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --runThreadN $2 --genomeLoad $4 --outSAMtype None --readMapNumber $3"
}
star() { # name preloadThreads profile threads readMapNumber genomeLoad [extra STAR args]
  local name=$1 pre=$2 prof=$3 t=$4 n=$5 gl=$6; shift 6
  local O=$SCR/$name/; mkdir -p $O
  local t0=$(ts)
  SINGLET_GENOME_PRELOAD_THREADS=$pre /usr/bin/time -f "%e %U %S %M" -o $O/time.txt $BIN star $(pargs $prof $t $n $gl) --outFileNamePrefix $O "$@" > $O/stdout.txt 2> $O/stderr.txt
  local rc=$? t1=$(ts)
  local u=$(grep -a "Uniquely mapped reads %" $O/Log.final.out | cut -d'|' -f2 | tr -d ' \t')
  local inr=$(grep -a "Number of input reads" $O/Log.final.out | cut -d'|' -f2 | tr -d ' \t')
  local sj=$(md5sum < $O/SJ.out.tab | cut -c1-12)
  local ph=$(grep -a -E "Started job on|Started mapping on|Finished on" $O/Log.final.out | awk -F'|' '{gsub(/\t/,"",$2); print $2}' | awk '{print $3}' | tr '\n' ' ')
  local gs=$(grep -a "Started loading the genome" $O/Log.out | awk '{print $(NF-1)}'); local gf=$(grep -a "Finished loading the genome" $O/Log.out | awk '{print $(NF-1)}')
  local wl=$(grep -a "Finished reading, sorting and deduplicating CB whitelist" $O/Log.out | awk '{print $3}')
  log "$name: rc=$rc wall $(printf %.2f $(echo "$t1-$t0" | bc)) s (time: $(cat $O/time.txt)), unique $u of $inr, SJ $sj | job/map/finish $ph | whitelist done $wl, genome $gs-$gf | $(grep -a -h "Genome preload\|genome preload" $O/Log.out $O/stderr.txt | head -2 | tr '\n' ' ')"
  for f in Log.out Log.final.out stderr.txt; do cp $O/$f $RES/$name.$f 2>/dev/null; done
  rm -rf $O/Solo.out $O/_STARtmp
}

# ---- probes (200k read pairs)
star p8_plain_base 0 plain 8 200000 NoSharedMemory
star p8_plain_pre 8 plain 8 200000 NoSharedMemory
star p8_plain_pre_io4 8 plain 8 200000 NoSharedMemory --limitIObufferSize 4000000 50000000
star p4_plain_pre_io4 8 plain 4 200000 NoSharedMemory --limitIObufferSize 4000000 50000000
star p8_cr4_base 0 cr4 8 200000 NoSharedMemory
star p8_cr4_pre_io4 8 cr4 8 200000 NoSharedMemory --limitIObufferSize 4000000 50000000
star p4_cr4_pre_io4 8 cr4 4 200000 NoSharedMemory --limitIObufferSize 4000000 50000000
star p1_plain_pre_io4 8 plain 1 200000 NoSharedMemory --limitIObufferSize 4000000 50000000

# ---- SysV shared memory
{ echo "== before: $(date -Is)"; ipcs -m; cgm; echo; } >> $RES/shm.txt 2>&1
if $BIN genome status $G >> $RES/shm.txt 2>&1; then
  log "shm: the genome is ALREADY in shared memory on $(hostname) (not ours): shm tests skipped, nothing removed"
else
  c0=$(cat $CG/memory.current); t0=$(ts)
  $BIN genome load $G >> $RES/shm.txt 2>&1; rc=$?
  t1=$(ts)
  if $BIN genome status $G >> $RES/shm.txt 2>&1; then SHM_OURS=1; fi
  c1=$(cat $CG/memory.current)
  log "shm: genome load (LoadAndExit) rc=$rc in $(printf %.2f $(echo "$t1-$t0" | bc)) s, in shm: $SHM_OURS, cgroup memory.current +$(echo "($c1-$c0)/2^30" | bc -l | cut -c1-5) GB ($(cgm)); $(grep -E "^Shmem:|ShmemHugePages" /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  { echo "== after load"; ipcs -m; ipcs -mu; cgm; echo; } >> $RES/shm.txt 2>&1
  if [ $SHM_OURS = 1 ]; then
    star p8_plain_shm_io4 0 plain 8 200000 LoadAndKeep --limitIObufferSize 4000000 50000000
    star p8_cr4_shm_io4 0 cr4 8 200000 LoadAndKeep --limitIObufferSize 4000000 50000000
    star m8_plain_shm 0 plain 8 2000000 LoadAndKeep
    star m8_plain_pre 8 plain 8 2000000 NoSharedMemory
    star m8_plain_shm2 0 plain 8 2000000 LoadAndKeep
    star m8_plain_pre2 8 plain 8 2000000 NoSharedMemory
    c2=$(cat $CG/memory.current); t0=$(ts)
    $BIN genome unload $G >> $RES/shm.txt 2>&1; t1=$(ts)
    $BIN genome status $G >> $RES/shm.txt 2>&1 || SHM_OURS=0
    c3=$(cat $CG/memory.current)
    log "shm: genome unload (Remove) in $(printf %.2f $(echo "$t1-$t0" | bc)) s, still in shm: $SHM_OURS, cgroup memory.current $(echo "($c3-$c2)/2^30" | bc -l | cut -c1-6) GB ($(cgm))"
  fi
fi
star m8_plain_base 0 plain 8 2000000 NoSharedMemory

# ---- level-5 pipeline runs (stage timings at T, cr4_probe.json)
FQ=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5
for r in ${RUNS:-new base}; do
  B=$BIN; [ $r = base ] && B=$BASE
  O=$RES/run-$r; mkdir -p $O $SCR/run-$r/out
  ( echo -e "t\tcpu_s\tgpu_util\tgpu_mem_mb"
    while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')
      printf "%s\t%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
    done ) > $O/monitor.tsv 2>/dev/null &
  MON=$!
  T0=$(ts); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/run-$r && timeout -k 30 1200 $B $FQ --ref-base $REF --genome-dir $G --exons $REF/GRCh38-2024-A/genes/genes.gtf \
      --out-prefix $SCR/run-$r/out/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(ts); kill $MON 2>/dev/null
  log "run-$r: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8)"
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "E10" -e "E11" -e "soloRun" -e "Solo counting" -e "loading the genome" -e "started mapping" -e "finished mapping" -e "Loading SA" -e "Loading Genome" -e "Genome preload" -e "ALL DONE" $SCR/run-$r/out/*Log.out 2>/dev/null | cut -c1-600 | head -160 > $O/starlog_engine.txt
  cp $SCR/run-$r/out/star_Log.out $SCR/run-$r/out/star_Log.final.out $SCR/run-$r/out/cr4_probe.json $O/ 2>/dev/null
  cp -r $SCR/run-$r/out/cr4_probe $O/ 2>/dev/null
  grep -a -e "Genome preload" -e "Genome prefetch" -e "STAR pid" -e "CR4 probe" -e "decoded to temp" -e "References loaded" $O/run.log.ts | tee -a $RES/summary.txt
  log "  cr4_probe.json: $(cat $O/cr4_probe.json)"
  log "  main STAR: $(grep -a -E "Started job on|Started mapping on" $O/star_Log.final.out | awk -F'|' '{print $2}' | tr -d '\t' | tr '\n' ' ')"
  python3 $SCR/repo/bench/fable/solowire/s8_stages_e11.py $O 66.601887 > $O/stages.txt 2>&1; tee -a $RES/summary.txt < $O/stages.txt
  rm -rf $SCR/run-$r
done
log "done $(date -Is)"
