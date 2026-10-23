#!/bin/bash
#SBATCH --job-name=pb-run
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run ONE tool on ONE dataset from the same FASTQs and the same reference.
#   sbatch run_tool.sbatch <dataset_id> <singlet-baseline|singlet-fixed[-t4]|starsolo|alevinfry>
# Output: $PB/runs/<dataset>/<tool>/ plus time.txt (GNU time -v: wall clock, max RSS).
# Every tool gets 16 threads; runtimes are comparable within this table only.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
ID=$1; TOOL=$2; T=$SLURM_CPUS_PER_TASK
IFS=$'\t' read -r _ SPECIES CHEM _ _ _ < <(dataset_row $ID)
D=$PB/data/$ID; O=$PB/runs/$ID/$TOOL; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R1=$(grep "_R1_" $D/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $D/gex_fastqs.txt | paste -sd,)
GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf
case $CHEM in
  10xv3) UMILEN=12; WL=$PB/whitelists/3M-february-2018.txt; STRAND=Forward; AFCHEM=10xv3; AFORI=fw ;;
  10xv2) UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt; STRAND=Forward; AFCHEM=10xv2; AFORI=fw ;;
  10x5p) UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt; STRAND=Reverse; AFCHEM=10xv2-5p; AFORI=rc ;;
esac
TIME="/usr/bin/time -v -o $O/time.txt"
echo "[run] $ID $TOOL species=$SPECIES chem=$CHEM node=$(hostname) $(date -Is)"
echo "[run] cpu: $(lscpu | sed -n "s/^Model name: *//p")" | tee $O/node.txt

case $TOOL in
singlet-baseline|singlet-fixed|singlet-baseline-t4|singlet-fixed-t4)
  # -t4: the campaign's 4 pileup workers (production ran --threads 4). The atlas engine's
  # counts depend on the worker count (cross-worker multi-mappers), so both are measured.
  BIN=$PB/bin/${TOOL%-t4}; [ "$TOOL" != "${TOOL%-t4}" ] && T=4
  # .1fq encoding is part of Singlet's cost: timed separately (encode.time.txt)
  cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
  cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
  /usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz \
      -o $SCR/$ID.1fq > $O/encode.log 2>&1
  rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
  # Campaign flags (provenance.json of kept results: --pipeline, directional UMI) + mtx and raw matrix
  cd $SCR
  $TIME $BIN $SCR/$ID.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
      --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
  ;;
starsolo)
  # Cell Ranger 4+ emulation per the STARsolo docs; Gene, GeneFull and Velocyto from one pass
  cd $O
  $TIME STAR --runThreadN $T --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat \
      --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloUMIstart 17 \
      --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND \
      --soloFeatures Gene GeneFull Velocyto --soloCellFilter EmptyDrops_CR \
      --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR \
      --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
      --outSAMtype None --outTmpDir $SCR/startmp > $O/run.log 2>&1
  ;;
alevinfry)
  IDX=$PB/refs/$SPECIES/af_idx
  $TIME simpleaf quant --reads1 $R1 --reads2 $R2 --index $IDX/index \
      --t2g-map $IDX/index/t2g_3col.tsv --chemistry $AFCHEM --expected-ori $AFORI \
      --resolution cr-like --unfiltered-pl --threads $T --output $O/af > $O/run.log 2>&1
  ;;
*) echo "unknown tool $TOOL"; exit 2 ;;
esac
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
du -sh $O
