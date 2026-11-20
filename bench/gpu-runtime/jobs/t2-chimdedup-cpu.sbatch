#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 18 --mem=170G --time=04:00:00 -J t2-chimdedup-cpu -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimdedup-cpu (chimdedup task): the deduplicated --chimeric pass on the CPU engine (--engine cpu), dedup on vs SINGLET_CHIM_DEDUP=0, same binary. CPU only
# (no --gres). The CPU engine writes the pass's input files (star_Unmapped.out.mate1/2, star_Softclip.out.mate1/2) through STAR's own output path, not the
# engine's. One run per arm on one item; the tables' decompressed md5 must be equal.
# Parameters: BIN, ITEM (default pbmc5p_v1), ARMS (default "on:SINGLET_CHIM_DEDUP=1 off:SINGLET_CHIM_DEDUP=0"), T (default 16), TAG
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-chimdedup/singlet-gpu}; TAG=${TAG:-cpu}; ITEM=${ITEM:-pbmc5p_v1}; T=${T:-16}; GD=$REF/GRCh38-2024-A
ARMS=${ARMS:-on:SINGLET_CHIM_DEDUP=1 off:SINGLET_CHIM_DEDUP=0}
O=$JR/results/t2-chimdedup-$SLURM_JOB_ID-$TAG; mkdir -p $O; SCR=/tmp/$USER-chimdedupcpu-$SLURM_JOB_ID; mkdir -p $SCR; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
trap 'rm -rf "${SCR:?}"' EXIT   # own node-local scratch only
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BIN item=$ITEM arms='$ARMS' T=$T $(date -Is)" | tee $R
F=$W/data/gpu-deep/$ITEM/$ITEM.1fq; [ -s $F ] || F=$PB/data/$ITEM/$ITEM.1fq; [ -s $F ] || { echo "MISSING INPUT $ITEM" | tee -a $R; exit 2; }
cp $F $SCR/$ITEM.1fq
for a in $ARMS; do
  arm=${a%%:*}; envs=${a#*:}; L=${ITEM}_cpu_${arm}; dd=$SCR/$L/out; mkdir -p $dd
  echo "RUN $L env '$envs' start $(date -Is)" | tee -a $R
  st=$(date +%s)
  ( IFS=,; for kv in $envs; do export "$kv"; done; unset IFS
    cd $SCR/$L && /usr/bin/time -v -o $O/$L.time.txt timeout -k 30 7200 $BIN $SCR/$ITEM.1fq --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf \
      --out-prefix $dd/ --threads $T --pipeline --output-format 1pz --no-hla --engine cpu --chimeric 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/$L.run.log.ts )
  echo "RUN $L wall $(( $(date +%s) - st )) s" | tee -a $R
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O/$L.run.log.ts | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
  ls $dd | grep -a -e '^chim_' | sed "s/^/RUN $L leftover: /" | tee -a $R
  grep -a 'Maximum resident\|Elapsed (wall' $O/$L.time.txt | sed "s/^/RUN $L time: /" | tee -a $R
done
echo "done $(date -Is)" | tee -a $R
