#!/bin/bash
#SBATCH --job-name=bb-mem
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch memory.sbatch <dataset> <old-bin> <new-bin> [N]
# What one job costs and what a node can hold. (1) old-bin alone: private genome, private non-host index.
# (2) new-bin alone, then N copies at once (default 4, 4 threads each): the genome loaded once into shared memory
# (singlet genome load, --genome-shared on) and the non-host index memory-mapped. Per run, sample_tree.py reports the
# peak Private (what one more job costs) and Shared (paid once per node) memory of the job's process tree.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
REF=/mnt/projects/debruinz_project/cellarium/reference
DS=$1; OLD=$2; NEW=$3; N=${4:-4}; T=4
case $DS in
  GSM*) IN=$W/data/atlas-rerun/perf/$DS/sample.1fq; SPECIES=GRCh38-2024-A ;;
  neuron*|nuclei*) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCm39-2024-A ;;
  *) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCh38-2024-A ;;
esac
G=$REF/$SPECIES; HERE=$W/jobs/bb-mem
O=$W/bb-mem/$DS; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bbmem-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR; $PB/bin/$NEW genome unload $G/star_2.7.11b > /dev/null 2>&1' EXIT
cp $IN $SCR/s.1fq
echo "node=$(hostname) $(date -Is) ds=$DS old=$OLD new=$NEW N=$N"; free -m | head -2
runone() {  # tag bin genome-shared env
  local tag=$1 bin=$2 gs=$3 envs=$4
  mkdir -p $SCR/$tag/out; cd $SCR/$tag
  env SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF $envs \
    /usr/bin/time -v -o $O/time_$tag.txt $PB/bin/$bin $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $SCR/$tag/out/ --threads $T --pipeline --output-format mtx --raw-matrix \
    --bam-sort-mode disk --genome-shared $gs > $O/run_$tag.log 2>&1 &
  local pid=$!
  $ENV/bin/python $HERE/sample_tree.py $pid $O/mem_$tag.tsv > $O/peak_$tag.txt 2>&1 &
  echo $pid
}
# 1. old: alone
runone old $OLD off "" > /dev/null; wait
echo "[old alone] $(cat $O/peak_old.txt)"
# 2. new: alone, genome shared + mapped index
$PB/bin/$NEW genome load $G/star_2.7.11b > $O/genome_load.log 2>&1; echo "genome load rc=$?"
free -m | head -2 | tee $O/free_after_genome_load.txt
runone new1 $NEW on ""  > /dev/null; wait
echo "[new alone] $(cat $O/peak_new1.txt)"
# 3. new x N at once
for i in $(seq 1 $N); do runone newN$i $NEW on "" > /dev/null; done
sleep 240; free -m | head -2 | tee $O/free_during_N.txt; ipcs -m | tee $O/ipcs.txt | head -8
wait
for i in $(seq 1 $N); do echo "[new x$N job $i] $(cat $O/peak_newN$i.txt)"; done
for t in old new1 newN1; do echo "== $t"; grep -E "Maximum resident|Elapsed|User time|System time|Exit status" $O/time_$t.txt; done
