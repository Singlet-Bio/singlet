#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 12 --mem=120G --time=03:59:00 -J t2-cpucache-ab -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-cpucache: CPU engine A/B/B/A on one node, outputs compared. Command line as t2-cpuprof-v2.sbatch (campaign shape: private genome,
# --pipeline --output-format 1pz --no-hla --no-editing, glibc malloc huge pages, pinned to one NUMA node's CPUs of the allocation).
# Parameters: BIN (singlet-gpu), ARMS "label:sample:threads[:VAR=val,VAR=val] ..." run one after the other (sample as t2-cpuprof-v2.sbatch),
#   TAG (result dir suffix), EQ="labelA:labelB ..." pairs whose out/ dirs are compared with out_equiv.py after the runs (read files by
#   sorted-record hash), LIM seconds per run.
# Output: $W/jobs/gpu-runtime/results/t2-cpucache-ab-<job>-<TAG>/results.txt (one RUNLINE per run: wall, user+sys core-s, max RSS; the SEQCACHE
#   lines of star_Log.out; node load before/after, MemAvailable minimum) and equiv-<a>-<b>.txt.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; GD=$REF/GRCh38-2024-A
HLP=$W/jobs/gpu-runtime; TOOLS=$HLP/t2-cpuprof-tools.py; BIN=${BIN:?}; WL=$W/builds/b21de2cf/whitelists; TAG=${TAG:-ab}; ARMS=${ARMS:?}; LIM=${LIM:-6000}
O=$HLP/results/t2-cpucache-ab-$SLURM_JOB_ID-$TAG; mkdir -p $O; R=$O/results.txt
SCR=/tmp/$USER-cpucache-ab-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/out $SCR/keep
export PATH=$W/envs/pbench/bin:$PATH
trap 'rm -rf "${SCR:?}"' EXIT
keypath() { case $1 in d300_3p|d100s_3p|d500s_3p) echo $W/data/gpu-deep/$1/$1.1fq ;; /*) echo $1 ;; *) echo $W/pipeline-benchmark/data/$1/$1.1fq ;; esac; }
{ echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$(nproc) tag=$TAG bin=$BIN arms='$ARMS' $(date -Is)"; lscpu | awk -F: '/Model name/{print $2}'
  echo "other jobs on node: $(squeue -h -w $(hostname) -o '%i/%j/%C' 2>/dev/null | tr '\n' ' ')"; } | tee $R
CPUS=$(python3 $TOOLS cpus 0)
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }
for arm in $ARMS; do
  IFS=: read -r label key T KV <<< "$arm"
  for kv in ${KV//,/ }; do export "$kv"; done
  fq=$(keypath $key); loc=$SCR/data/$(basename $fq)
  [ -s $loc ] || cp $fq $loc || exit 2
  mkdir -p $SCR/out/$label; d=$O/$label; mkdir -p $d
  ( while sleep 5; do awk '/MemAvailable/{print $2}' /proc/meminfo; done ) > $d/memavail.txt 2>/dev/null & MON=$!
  echo "######## run $label sample=$key T=$T env='$KV' load_before=$(cut -d' ' -f1-3 /proc/loadavg) $(date -Is)" | tee -a $R
  t0=$(date +%s.%N)
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 SINGLET_WHITELIST_DIR=$WL SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF \
  taskset -c $CPUS /usr/bin/time -v -o $d/time.txt timeout -k 30 $LIM $BIN $loc --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf \
    --out-prefix $SCR/out/$label/ --threads $T --genome-shared off --ref-base $REF --pipeline --output-format 1pz --no-hla --no-editing --engine cpu > $d/run.log 2>&1
  rc=$?; t1=$(date +%s.%N); kill $MON 2>/dev/null
  awk -v l=$label -v T=$T -v rc=$rc '/User time/{u=$NF} /System time/{s=$NF} /Elapsed/{e=$NF} /Maximum resident/{m=$NF}
    END{n=split(e,a,":"); wall=(n==3? a[1]*3600+a[2]*60+a[3] : a[1]*60+a[2]); printf "RUNLINE %s T=%d wall=%.1f user=%.1f sys=%.1f core_s=%.1f maxrss_gb=%.1f exit=%d\n", l,T,wall,u,s,u+s,m/1048576,rc}' $d/time.txt | tee -a $R
  echo "  load_after=$(cut -d' ' -f1-3 /proc/loadavg) memavail_min_gb=$(sort -n $d/memavail.txt | head -1 | awk '{printf "%.0f",$1/1048576}') other_jobs: $(squeue -h -w $(hostname) -o '%i/%j/%C' | tr '\n' ' ')" | tee -a $R
  grep -a "^SEQCACHE" $SCR/out/$label/star_Log.out | cut -c1-1500 | sed 's/^/  /' | tee -a $R
  grep -a "\.1fq: [0-9]* reads" $d/run.log | head -1 | sed 's/^[0-9.]* /  /' | tee -a $R
  cp $SCR/out/$label/star_Log.out $SCR/out/$label/star_Log.final.out $d/ 2>/dev/null
  rsync -a --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' --exclude='nonhost_reads.fq' $SCR/out/$label/ $SCR/keep/$label/
  readhash $SCR/out/$label | tee $d/readhash.txt >> $R
  rm -rf $SCR/out/$label
  for kv in ${KV//,/ }; do unset "${kv%%=*}"; done
done
for p in ${EQ:-}; do a=${p%%:*}; b=${p##*:}
  python3 $HLP/t2-cpucache-out_equiv.py $SCR/keep/$a $SCR/keep/$b > $O/equiv-$a-$b.txt 2>&1
  echo "=== out_equiv $a vs $b" | tee -a $R; grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $O/equiv-$a-$b.txt | cut -c1-220 | head -40 | tee -a $R
  cmp -s $O/$a/readhash.txt $O/$b/readhash.txt && echo "readhash $a == $b" | tee -a $R || { echo "readhash $a != $b"; diff $O/$a/readhash.txt $O/$b/readhash.txt; } | tee -a $R
done
echo "done $(date -Is)" | tee -a $R
