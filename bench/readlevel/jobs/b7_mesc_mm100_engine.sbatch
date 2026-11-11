#!/bin/bash
#SBATCH --job-name=rl-mm100
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B7 diagnostic on the mESC dev library: the same binary and reads as runs/te/mesc_2c, with the multimap limits of the comparators' STARsolo (100 loci)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
RT=$J/runs/te/mesc_2c/run.txt; BIN=$(grep -o 'bin=[^ ]*' $RT | head -1 | cut -d= -f2); FQ=$(grep -o 'fq=[^ ]*' $RT | head -1 | cut -d= -f2); G=$REF/GRCm39-2024-A; T=$SLURM_CPUS_PER_TASK
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
O=$J/runs/te-mm100/mesc_2c; [ -e $O/out ] && { echo "refusing to overwrite $O"; exit 2; }; mkdir -p $O
SCR=/tmp/$USER-mm100-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) bin=$BIN commit=$(cat $BIN/commit-gpu 2>/dev/null) fq=$FQ $(date -Is)" | tee $O/run.txt
( cd $SCR && /usr/bin/time -v -o $O/time.txt $BIN/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads $T \
    --pipeline --output-format 1pz --star-extra "--outFilterMultimapNmax 100 --winAnchorMultimapNmax 100" > $O/run.log 2>&1 )
ES=$(awk -F': ' '/Exit status/{print $2}' $O/time.txt); echo "exit $ES wall $(awk -F': ' '/Elapsed/{print $2}' $O/time.txt)" | tee -a $O/run.txt
rsync -a --exclude='*.bam' --exclude='*Unmapped*' --exclude='*.fq' --exclude='_STARtmp*' --exclude='.gpu_handoff*' $SCR/out/ $O/out/
grep -E "too many loci|multiple loci" $O/out/star_Log.final.out
[ "$ES" = 0 ]
