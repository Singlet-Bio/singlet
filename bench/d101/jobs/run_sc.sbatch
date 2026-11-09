#!/bin/bash
#SBATCH --job-name=d101-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# D-101 validation: the CPU engine (--pipeline defaults, so the REDIportal panel and T1K come from $REF) on row SLURM_ARRAY_TASK_ID of
# libs.tsv, with --tagged-bam so REDItools3 reads the same alignments. Output: $W/jobs/d101/runs/<sc_run>/ (out/ without the BAM,
# run.log, time.txt); the tagged BAM, sorted and indexed, goes to $W/jobs/d101/bam/<sc_run>.bam (deleted once scored).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
LIBS=${LIBS:?}; BIN=${BIN:?singlet binary}; T=${SLURM_CPUS_PER_TASK:-32}
ROW=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS); read -r DONOR SC SCG BULK BULKG <<< "$ROW"
FQ=$W/data/d101/sc/$SC.1fq; [ -s $FQ ] || { echo "no $FQ"; exit 3; }
RES=$W/jobs/d101/runs/$SC; [ -e $RES/DONE ] && { echo "done already: $RES"; exit 0; }; mkdir -p $RES $W/jobs/d101/bam
SCR=/tmp/$USER-d101-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
G=$REF/GRCh38-2024-A
echo "donor $DONOR $SC node $(hostname) bin $BIN commit $(cat $(dirname $BIN)/commit-gpu 2>/dev/null) $(date -Is)" | tee $RES/run.txt
( cd $SCR && /usr/bin/time -v -o $RES/time.txt $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --tagged-bam $SCR/tagged.bam > $RES/run.log 2>&1 )
ES=$(awk -F': ' '/Exit status/{print $2}' $RES/time.txt); echo "exit $ES" | tee -a $RES/run.txt
grep -aE "\[editing\]|\[hla\]|RNA editing|Default RNA editing|default RNA editing" $RES/run.log | tee -a $RES/run.txt
if [ -s $SCR/tagged.bam ]; then
  samtools sort -@ 16 -m 3G -T $SCR/st -o $W/jobs/d101/bam/$SC.bam $SCR/tagged.bam 2>/dev/null || mv $SCR/tagged.bam $W/jobs/d101/bam/$SC.bam
  samtools index -@ 8 $W/jobs/d101/bam/$SC.bam
fi
rsync -a --exclude='*.bam' --exclude='*.bai' --exclude='*.1fq' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads*' --exclude='.gpu_handoff*' $SCR/out/ $RES/out/
chmod -R go-rwx $RES/out/private 2>/dev/null || true
[ "$ES" = 0 ] && date -Is > $RES/DONE
ls $RES/out | head -80
