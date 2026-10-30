#!/bin/bash
#SBATCH --job-name=w11-replay
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=470G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 host-time study, step 5 (GSM4006843 only): thread / contention scaling of the post-pileup work.
#  1) one full run (--threads 8, SINGLET_KEEP_STAR_BAM=1) leaves the coordinate-sorted BAM and auto_barcodes.tsv in /tmp
#  2) replay with --bam-file (streaming pileup of the kept BAM + the same export), --threads 1, 4, 8 one after another
#  3) 8 concurrent replays at --threads 4
# The replay is a debug/interop path of the binary: it has no .1fq (protocol comes from --metadata-json, barcodes from the
# first run's refreshed list, STARsolo side files copied from the first run, nonhost screening is skipped for lack of
# unmapped-read files). Each replay's cell / demux sizes are printed for comparison with the full run.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$W/jobs/w11/host/bin/singlet-timers
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
SET=GSM4006843; D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/jobs/w11/host/replay; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-w11r-$SLURM_JOB_ID; mkdir -p $SCR/full; trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
{ echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK jobid=$SLURM_JOB_ID $(date -Is)"; echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "cotenant_cpus=$(squeue -w $(hostname -s) -h -t R -o '%C' | awk -v j=$SLURM_JOB_ID '{s+=$1}END{print s-0}')"; grep -m1 'model name' /proc/cpuinfo; } | tee $O/node.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
( echo -e "t\tmem_mb\tcpu_s"; while true; do printf "%s\t%s\t%s\n" $(date +%s) $(( $(cat $CG/memory.current) / 1048576 )) \
   $(( $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat) / 1000000 )); sleep 5; done ) > $O/monitor.tsv 2>&1 &
MON=$!
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
echo "== full run $(date -Is)"
cd $SCR
SINGLET_KEEP_STAR_BAM=1 $BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/full/ --threads 8 --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" > $O/full.log 2>&1 || echo "full exit $?" | tee -a $O/full.log
grep -E '^\[w11[tn]\]' $O/full.log > $O/full.w11.log
BAM=$SCR/full/star_Aligned.sortedByCoord.out.bam; ls -l $BAM $BAM.bai $SCR/full/auto_barcodes.tsv | tee $O/kept_files.txt
echo '{"protocol": "10x-3p-v2"}' > $SCR/meta.json
WL=$PB/whitelists/737K-august-2016.txt
replay() {  # name threads
  local n=$1 t=$2 P=$SCR/rp_$1; mkdir -p $P
  cp -r $SCR/full/star_Log.final.out $SCR/full/star_SJ.out.tab $SCR/full/star_Solo.out $P/ 2>/dev/null
  /usr/bin/time -v -o $O/$n.time.txt $BIN --bam-file $BAM --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
     --whitelist $WL --barcodes $SCR/full/auto_barcodes.tsv --cell-calling --umi-len 10 --metadata-json $SCR/meta.json \
     --out-prefix $P/ --threads $t --pipeline --output-format 1pz > $O/$n.log 2>&1 || echo "exit $?" >> $O/$n.log
  grep -E '^\[w11[tn]\]' $O/$n.log > $O/$n.w11.log
  grep -E "n_called_cells|estimated_cells|\"n_barcodes\"" $P/summary.json $P/pileup_stats.json > $O/$n.sizes.txt 2>/dev/null
  rm -rf $P
}
for t in 1 4 8; do echo "== replay threads=$t alone $(date -Is)"; replay t$t $t; done
echo "== 8 concurrent replays at threads=4 $(date -Is)"
PIDS=(); for i in 1 2 3 4 5 6 7 8; do replay c$i 4 & PIDS+=($!); done; wait "${PIDS[@]}"
echo "== done $(date -Is)"
mkdir -p $O/small; cp $SCR/full/summary.json $O/small/full.summary.json 2>/dev/null
