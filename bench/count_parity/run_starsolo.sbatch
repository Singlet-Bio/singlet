#!/bin/bash
#SBATCH --job-name=cp-solo
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# STARsolo 2.7.11b with the exact flags of bench/exact_vs_pseudo/run_dataset.sbatch (Cell Ranger 4+ emulation; Gene, GeneFull, Velocyto).
#   sbatch --export=ALL,DS=hgmm5k_v3 run_starsolo.sbatch      (v2 chemistry: 737K-august-2016 whitelist, 10-base UMI, from the name *_v2)
#   PROFILE=beta adds Singlet's Beta aligner profile flags. PROFILE=beta10 also takes Singlet's --outFilterScoreMin 10, which makes
#   the alignment the same as Singlet's, so counting is the only difference. UMI=dir swaps 1MM_CR + MultiGeneUMI_CR for
#   1MM_Directional_UMItools without a multi-gene filter (a dedup-sensitivity control).
# Output: $W/count-parity/runs/<DS>/starsolo[_<PROFILE>][_dir]/
set -euo pipefail
: "${DS:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$W/envs/pbench/bin:$PATH
case $DS in hgmm*) SPECIES=GRCh38-GRCm39-barnyard ;; *) SPECIES=GRCh38-2024-A ;; esac
case $DS in *_v2) WL=$PB/whitelists/737K-august-2016.txt; UMILEN=10 ;; *) WL=$PB/whitelists/3M-february-2018.txt; UMILEN=12 ;; esac
GENOME=$REF/$SPECIES/star_2.7.11b
R1=$(grep "_R1_" $PB/data/$DS/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $PB/data/$DS/gex_fastqs.txt | paste -sd,)
EXTRA=""; SUF=""; SCORE=30; UMIOPT="--soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR"
BETA="--alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated --outFilterMismatchNoverLmax 0.05"
case "${PROFILE:-}" in beta) SUF=_beta; EXTRA=$BETA ;; beta10) SUF=_beta10; EXTRA=$BETA; SCORE=10 ;; "") ;; *) echo "PROFILE?"; exit 2 ;; esac
if [ "${UMI:-}" = dir ]; then SUF=${SUF}_dir; UMIOPT="--soloUMIfiltering - --soloUMIdedup 1MM_Directional_UMItools"; fi
# BAM=1: an unsorted BAM whose GX/GN tags are GeneFull's (the first solo feature), for read-level comparisons. Output starsolo..._bam
FEAT="Gene GeneFull Velocyto"; SAM="--outSAMtype None"
if [ "${BAM:-0}" = 1 ]; then SUF=${SUF}_bam; FEAT="GeneFull Gene Velocyto"; SAM="--outSAMtype BAM Unsorted --outSAMattributes NH HI AS nM CR UR CB UB GX GN sF"; fi
S=$W/count-parity/runs/$DS/starsolo$SUF; rm -rf $S; mkdir -p $S; cd $S
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
STAR --version > $S/version.txt
/usr/bin/time -v -o $S/time.txt STAR --runThreadN $SLURM_CPUS_PER_TASK --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat \
    --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloUMIstart 17 \
    --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand Forward \
    --soloFeatures $FEAT --soloCellFilter EmptyDrops_CR \
    --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts $UMIOPT \
    --clipAdapterType CellRanger4 --outFilterScoreMin $SCORE \
    $SAM --outTmpDir $SCR/startmp $EXTRA > $S/run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" $S/time.txt; tail -3 $S/Log.final.out
