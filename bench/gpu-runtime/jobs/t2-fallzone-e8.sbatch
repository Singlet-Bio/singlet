#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=00:50:00 --exclude=g050 -J t2-fz-e8 -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-fallzone-e8 (fallzone follow-up): ONE small staged SRA run, encoded twice (ARMS: "auto" = `singlet download <sra>` with no --protocol, as a campaign without the preflight
# hook does; "p3" = with `--protocol 10x-3p-v2`, as the preflight hook passed for the 7 mis-called 10x_5p samples of dry run 2), then both run as items of one pool with today's
# campaign arguments (--no-hla, --engine gpu, and --chimeric for a 10x_5p label as config.sample_output_args gives), the campaign's H100 settings (budget 31, 2 slots).
# Per item: the engine's level lines, the STAR command's BAM mode, and the pipeline's own "[singlet]" lines about the stream / strand / protocol (pool log).
# Parameters: BIN, SRA (path of the staged .sra, read only; copied to node-local disk), ARMS, EXTRA (per-item run flags, default --chimeric), TAG
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SRA=${SRA:?}; ARMS=${ARMS:-"auto p3"}; EXTRA=${EXTRA:---chimeric}; TAG=${TAG:-e8}; T=8; SLOTS=2; MEMGB=31; LIM=2400; GD=$REF/GRCh38-2024-A
O=$W/jobs/gpu-runtime/results/t2-fallzone-$SLURM_JOB_ID-$TAG; mkdir -p $O/pool0; SCR=/tmp/$USER-fze8-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
PARGS="--no-hla"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; stopholders; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
BIN=${BIN:?}
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN sra=$SRA arms='$ARMS' extra='$EXTRA' $(date -Is)" | tee $R
cp $SRA $SCR/data/in.sra || { echo "copy failed" | tee -a $R; exit 2; }
stopholders
: > $SCR/run/queue0; n=0
for a in $ARMS; do
  pf=""; [ "$a" = p3 ] && pf="--protocol 10x-3p-v2"
  ( cd $SCR/data && $BIN download $SCR/data/in.sra $pf -o $SCR/data/$a.1fq > $O/encode-$a.txt 2>&1 ); echo "encode $a rc=$? $(ls -la $SCR/data/$a.1fq 2>&1 | awk '{print $5}') bytes; $(grep -a -i -m3 'protocol\|reads' $O/encode-$a.txt | tr '\n' ' ' | cut -c1-300)" | tee -a $R
  mkdir -p $SCR/run/c0.$n; echo "$SCR/data/$a.1fq $SCR/run/c0.$n/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf $EXTRA" >> $SCR/run/queue0; echo -e "$n\t$a" >> $O/items.tsv; n=$((n+1))
done
st=$(date +%s)
setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers 1 --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1
en=$(date +%s); echo "pool wall $(( en - st )) s" | tee -a $R
cut -f1,3,7,14 $O/pool0/samples.tsv | tee -a $R
while IFS=$'\t' read -r i a; do
  dd=$SCR/run/c0.$i/out; cp $dd/star_Log.out $O/item$i.star_Log.out 2>/dev/null; cp $O/pool0/$i.log $O/item$i.pool.log 2>/dev/null
  echo "ITEM $i arm=$a"  | tee -a $R
  grep -a -h 'E19: level\|E10 (level 5) not used\|E17 budget\|engine_gpu: [0-9]* reads,' $dd/star_Log.out | cut -c1-260 | sed 's/^/  STAR: /' | tee -a $R
  grep -a -m1 -A0 '^outSAMtype' $dd/star_Log.out | sed 's/^/  STAR: /' | tee -a $R
  grep -a -h '\[singlet\] \(STREAM STRAND\|BAM:\|--stream\|WARNING\|chimeric\|Protocol\|protocol\)\|restarting on the sorted\|strand' $O/pool0/$i.log | cut -c1-300 | sed 's/^/  LOG: /' | head -12 | tee -a $R
done < $O/items.tsv
echo "done $(date -Is)" | tee -a $R
