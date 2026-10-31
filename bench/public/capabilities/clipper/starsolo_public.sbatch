#!/bin/bash
#SBATCH --job-name=pub-starsolo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Singlet 1.0.0 public benchmark: STARsolo 2.7.11b on the cohort, two documented configurations, same FASTQs and
# same reference as Singlet. Array index = dataset, MODE = cr | default.
#   sbatch --array=0-7 --export=ALL,MODE=cr  starsolo_public.sbatch
#   sbatch --array=0-7 --export=ALL,MODE=default starsolo_public.sbatch
# MODE=cr       Cell Ranger 4+ emulation from the STARsolo manual; Gene + GeneFull + Velocyto + SJ in one pass. This is the
#               reference for gex-counts, spliced-unspliced and cell-calling (EmptyDrops_CR) and for junctions-psi.
# MODE=reference STARsolo SJ with the STAR options Singlet itself runs for the dataset (the junctions-psi reference since the 2026-10-31 amendment,
#               bench/public/CONTRACT.md "junctions-psi" and docs/research-beta/junctions-psi-reference.md): --outFilterScoreMin 10 and
#               --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts for every dataset, and for the 10x 3' v2/v3 datasets the Beta aligner profile
#               (CellRanger4 clipping, --alignSJoverhangMin 8, --outFilterIntronMotifs RemoveNoncanonicalUnannotated, --outFilterMismatchNoverLmax 0.05).
#               Only --soloFeatures SJ.
# MODE=default  STAR / STARsolo defaults for everything except the chemistry (CB/UMI geometry, whitelist, strand);
#               only --soloFeatures SJ. The second documented configuration of the competitor: its distance to MODE=cr is
#               the tolerance for junctions-psi.
# Output: $PB/runs/<dataset>/starsolo-public-<MODE>/ (Solo.out, SJ.out.tab, Log.final.out, time.txt, command.txt).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$W/envs/pbench/bin:$PATH
MODE=${MODE:-cr}
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}; T=$SLURM_CPUS_PER_TASK
KANG=$W/modality-bench/genotype_pool/fastq
case $ID in
  pbmc1k_v3)      SPECIES=GRCh38-2024-A;           CHEM=10xv3 ;;
  pbmc1k_v2)      SPECIES=GRCh38-2024-A;           CHEM=10xv2 ;;
  neuron1k_v3)    SPECIES=GRCm39-2024-A;           CHEM=10xv3 ;;
  hgmm1k_v3)      SPECIES=GRCh38-GRCm39-barnyard;  CHEM=10xv3 ;;
  pbmc1k_cite_v3) SPECIES=GRCh38-2024-A;           CHEM=10xv3 ;;
  nuclei2k_v2)    SPECIES=GRCm39-2024-A;           CHEM=10xv2 ;;
  pbmc5p_v1)      SPECIES=GRCh38-2024-A;           CHEM=10x5p ;;
  kang2018_ctrl)  SPECIES=GRCh38-2024-A;           CHEM=10xv1 ;;
esac
case $CHEM in
  10xv3) CBLEN=16; UMISTART=17; UMILEN=12; WL=$PB/whitelists/3M-february-2018.txt;  STRAND=Forward ;;
  10xv2) CBLEN=16; UMISTART=17; UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt;  STRAND=Forward ;;
  10x5p) CBLEN=16; UMISTART=17; UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt;  STRAND=Reverse ;;
  10xv1) CBLEN=14; UMISTART=15; UMILEN=10; WL=$PB/whitelists/737K-april-2014.txt;   STRAND=Forward ;;
esac
if [ $ID = kang2018_ctrl ]; then R1=$KANG/R1.fastq.gz; R2=$KANG/R2.fastq.gz
else
  D=$PB/data/$ID
  R1=$(grep "_R1_" $D/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $D/gex_fastqs.txt | paste -sd,)
fi
GENOME=$REF/$SPECIES/star_2.7.11b
O=$PB/runs/$ID/starsolo-public-$MODE; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[pub-starsolo] $ID mode=$MODE species=$SPECIES chem=$CHEM node=$(hostname) $(date -Is)"
echo "node: $(hostname); cpu: $(lscpu | sed -n 's/^Model name: *//p')" > $O/node.txt
STAR --version > $O/star_version.txt
COMMON=(--runThreadN $T --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat
        --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen $CBLEN --soloUMIstart $UMISTART --soloUMIlen $UMILEN
        --soloBarcodeReadLength 0 --soloStrand $STRAND --outSAMtype None --outTmpDir $SCR/startmp)
PROFILE=0; case $ID in pbmc1k_v3|pbmc1k_v2|neuron1k_v3|hgmm1k_v3|pbmc1k_cite_v3|nuclei2k_v2) PROFILE=1 ;; esac   # 10x 3' v2/v3: the Beta aligner profile is on
case $MODE in
  cr)
    CMD=(STAR "${COMMON[@]}" --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR
         --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR
         --clipAdapterType CellRanger4 --outFilterScoreMin 30) ;;
  default)
    CMD=(STAR "${COMMON[@]}" --soloFeatures SJ) ;;
  reference)
    CMD=(STAR "${COMMON[@]}" --soloFeatures SJ --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --outFilterScoreMin 10)
    [ $PROFILE = 1 ] && CMD+=(--clipAdapterType CellRanger4 --alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated
                              --outFilterMismatchNoverLmax 0.05) ;;
esac
echo "${CMD[*]}" > $O/command.txt
cd $O
/usr/bin/time -v -o $O/time.txt "${CMD[@]}" > $O/run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
ls $O/Solo.out; du -sh $O
