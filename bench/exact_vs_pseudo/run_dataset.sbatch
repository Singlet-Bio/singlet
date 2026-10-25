#!/bin/bash
#SBATCH --job-name=evp-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Exact vs pseudo-alignment: run alevin-fry (splici / USA), STARsolo and Singlet on ONE real dataset,
# one after the other on the SAME node with the SAME 16 threads, reading the SAME FASTQs and the SAME reference.
#   sbatch run_dataset.sbatch <hgmm1k_v3|pbmc1k_cite_v3|pbmc1k_v3>
# Output: $OUT/<dataset>/<tool>/ with GNU time -v in time.txt (wall, user+sys CPU, max RSS).
set -euo pipefail
ID=$1; T=$SLURM_CPUS_PER_TASK
WORK=/mnt/projects/debruinz_project/debruinz/singlet
PB=$WORK/pipeline-benchmark
OUT=$WORK/exact-vs-pseudo/runs; REF=/mnt/projects/debruinz_project/cellarium/reference
export ALEVIN_FRY_HOME=$PB/af_home
export PATH=$WORK/envs/pbench/bin:$PATH
BIN=$PB/bin/singlet-campaign
echo "sha256 $(sha256sum $BIN | cut -c1-64)"
case $ID in
  hgmm1k_v3)      SPECIES=GRCh38-GRCm39-barnyard ;;
  pbmc1k_v3)      SPECIES=GRCh38-2024-A ;;
  pbmc1k_cite_v3) SPECIES=GRCh38-2024-A ;;
  *) echo "unknown dataset $ID"; exit 2 ;;
esac
UMILEN=12; WL=$PB/whitelists/3M-february-2018.txt; STRAND=Forward; AFCHEM=10xv3; AFORI=fw
D=$PB/data/$ID; O=$OUT/$ID; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R1=$(grep "_R1_" $D/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $D/gex_fastqs.txt | paste -sd,)
GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf
echo "[run] $ID node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)"
hostname > $O/node.txt

# ---- alevin-fry via simpleaf: piscem on the splici index, USA counts, cr-like resolution ----
A=$O/alevinfry; rm -rf $A; mkdir -p $A
IDX=$PB/refs/$SPECIES/af_idx
/usr/bin/time -v -o $A/time.txt simpleaf quant --reads1 $R1 --reads2 $R2 --index $IDX/index \
    --t2g-map $IDX/index/t2g_3col.tsv --chemistry $AFCHEM --expected-ori $AFORI \
    --resolution cr-like --unfiltered-pl $WL --threads $T --output $A/af > $A/run.log 2>&1
# second resolution (EM for multi-gene UMIs), quant step only, timed separately
cp -r $A/af/af_quant $A/af/af_quant_em
/usr/bin/time -v -o $A/time_quant_em.txt alevin-fry quant -i $A/af/af_quant_em -o $A/af/af_quant_em -t $T \
    -m $IDX/index/t2g_3col.tsv -r cr-like-em > $A/run_em.log 2>&1 || echo "cr-like-em failed" >> $A/run_em.log

# ---- STARsolo 2.7.11b, Cell Ranger 4+ emulation, Gene + GeneFull + Velocyto ----
S=$O/starsolo; rm -rf $S; mkdir -p $S; cd $S
/usr/bin/time -v -o $S/time.txt STAR --runThreadN $T --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat \
    --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloUMIstart 17 \
    --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND \
    --soloFeatures Gene GeneFull Velocyto --soloCellFilter EmptyDrops_CR \
    --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR \
    --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
    --outSAMtype None --outTmpDir $SCR/startmp > $S/run.log 2>&1

# ---- Singlet 1.0.0 (singlet-campaign): .1fq encode (counted in the cost) + align + pileup ----
G=$O/singlet; rm -rf $G; mkdir -p $G
export SINGLET_STALL_TIMEOUT=7200
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
/usr/bin/time -v -o $G/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq > $G/encode.log 2>&1
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
cd $SCR
/usr/bin/time -v -o $G/time.txt $BIN $SCR/$ID.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
    --out-prefix $G/ --threads $T --pipeline --output-format mtx --raw-matrix > $G/run.log 2>&1
for t in alevinfry starsolo singlet; do echo == $t; grep -E "Elapsed \(wall|Maximum resident|User time|System time" $O/$t/time.txt; done
grep -E "Elapsed" $G/encode.time.txt
