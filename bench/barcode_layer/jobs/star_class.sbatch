#!/bin/bash
#SBATCH --job-name=bl-star
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --cpus-per-task=14
#SBATCH --mem=110G
#SBATCH --time=03:30:00
#SBATCH --array=0-6%2
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# STAR 2.7.11b (the pbench conda build) on the same real reads the GPU harness sees, one class per array task, with the
# STARsolo settings singlet passes (singlet.cpp 6600-7316): CB_UMI_Simple + 1MM_multi_Nbase_pseudocounts for 10x, CB_samTagOut + None for Drop-seq,
# native CB_UMI_Complex + Exact for BD / SPLiT-seq / inDrop (W1-adapter route). Output: a BAM with CR UR CB UB, kept under data/barcode-layer/star/<class>.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/data/barcode-layer; WLD=$W/pipeline-benchmark/whitelists
G=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
export PATH=$W/envs/pbench/bin:$PATH
NAMES=(pbmc1k_v3_F pbmc1k_v3_real dropseq bd splitseq indrop splitseq48)
CLS=${NAMES[$SLURM_ARRAY_TASK_ID]}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
OUT=$D/star/$CLS; mkdir -p $OUT
echo "node=$(hostname) class=$CLS $(date -Is)"; STAR --version
# per class: cDNA file, barcode file, barcode-read quality ('F' as singlet writes it, or the sequencer's), solo arguments
case $CLS in
  pbmc1k_v3_F|pbmc1k_v3_real)
    CDNA=$D/pbmc1k_v3/R2.fq; BC=$D/pbmc1k_v3/R1.fq; [ $CLS = pbmc1k_v3_F ] && QF=1 || QF=0; FEAT=Gene; BAMT="BAM SortedByCoordinate"; ATTR="NH HI AS nM CR UR CB UB"
    SOLO="--soloType CB_UMI_Simple --soloCBwhitelist $WLD/3M-february-2018.txt --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloFeatures Gene --soloCellFilter None";;
  dropseq)
    CDNA=$D/dropseq/M2.fq; BC=$D/dropseq/M1.fq; QF=1; BAMT="BAM Unsorted"; ATTR="NH HI AS nM CR UR CB"
    SOLO="--soloType CB_samTagOut --soloCBwhitelist None --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 8 --soloBarcodeReadLength 0 --soloCBmatchWLtype Exact";;
  bd)
    CDNA=$D/bd/M2.fq; BC=$D/bd/M1.fq; QF=1; BAMT="BAM SortedByCoordinate"; ATTR="NH HI AS nM CR UR CB UB"
    SOLO="--soloType CB_UMI_Complex --soloCBwhitelist $WLD/bd_cls1.txt $WLD/bd_cls2.txt $WLD/bd_cls3.txt --soloCBposition 0_0_0_8 0_21_0_29 0_43_0_51 --soloUMIposition 0_52_0_59 --soloBarcodeReadLength 0 --soloCBmatchWLtype Exact --soloFeatures Gene --soloCellFilter None";;
  splitseq)
    CDNA=$D/splitseq_parse/M1.fq; BC=$D/splitseq_parse/M2.fq; QF=1; BAMT="BAM SortedByCoordinate"; ATTR="NH HI AS nM CR UR CB UB"
    SOLO="--soloType CB_UMI_Complex --soloCBwhitelist $WLD/splitseq_bc_rd3.txt $WLD/splitseq_bc_rd2.txt $WLD/splitseq_bc_rd1.txt --soloCBposition 0_10_0_17 0_33_0_40 0_56_0_63 --soloUMIposition 0_0_0_9 --soloBarcodeReadLength 0 --soloCBmatchWLtype Exact --soloFeatures Gene --soloCellFilter None";;
  splitseq48)
    CDNA=$D/splitseq_parse/M1.fq; BC=$D/splitseq_parse/M2.fq; QF=1; BAMT="BAM SortedByCoordinate"; ATTR="NH HI AS nM CR UR CB UB"
    SOLO="--soloType CB_UMI_Complex --soloCBwhitelist $WLD/splitseq_bc_rd3.txt $WLD/splitseq_bc_rd2.txt $WLD/splitseq_bc_rd1.txt --soloCBposition 0_10_0_17 0_48_0_55 0_78_0_85 --soloUMIposition 0_0_0_9 --soloBarcodeReadLength 0 --soloCBmatchWLtype Exact --soloFeatures Gene --soloCellFilter None";;
  indrop)
    CDNA=$D/indrop/M1.fq; BC=$D/indrop/M2.fq; QF=1; BAMT="BAM SortedByCoordinate"; ATTR="NH HI AS nM CR UR CB UB"
    SOLO="--soloType CB_UMI_Complex --soloCBwhitelist $WLD/indrop_bc1.txt $WLD/indrop_bc2.txt --soloCBposition 0_0_2_-1 3_1_3_8 --soloUMIposition 3_9_3_14 --soloAdapterSequence GAGTGATTGCTTGTGACGCCTT --soloAdapterMismatchesNmax 1 --soloBarcodeReadLength 0 --soloCBmatchWLtype Exact --soloFeatures Gene --soloCellFilter None";;
esac
if [ "$QF" = 1 ]; then awk 'NR%4==0{gsub(/./,"F")} {print}' $BC > $SCR/bc.fq; BCF=$SCR/bc.fq; else BCF=$BC; fi
echo "cdna=$CDNA bc=$BC qualF=$QF reads=$(( $(wc -l < $BCF) / 4 ))"
STAR --runThreadN $((SLURM_CPUS_PER_TASK - 1)) --genomeDir $G --genomeLoad NoSharedMemory --readFilesIn $CDNA $BCF $SOLO \
  --outSAMtype $BAMT --outSAMunmapped Within --outSAMattributes $ATTR --outFilterScoreMin 10 \
  --limitBAMsortRAM 30000000000 --outTmpDir $SCR/tmp --outFileNamePrefix $SCR/o/ 2>&1 | tail -20
rc=$?
ls -la $SCR/o/ 2>/dev/null
B=$(ls $SCR/o/Aligned.*bam 2>/dev/null | head -1)
if [ -z "$B" ]; then echo "STAR produced no BAM"; tail -30 $SCR/o/Log.out 2>/dev/null; exit 1; fi
mv $B $OUT/star.bam; cp $SCR/o/Log.final.out $OUT/ 2>/dev/null; cp $SCR/o/Solo.out/Gene/Summary.csv $OUT/ 2>/dev/null
samtools view -F 0x900 $OUT/star.bam | head -3 | cut -f 1-9,12-
echo "done $(date -Is)"; ls -la $OUT
