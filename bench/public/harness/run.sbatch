#!/bin/bash
#SBATCH --job-name=bpub-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Run ONE tool on ONE cohort dataset, from the same reads and the same reference as every other tool.
#   sbatch --array=1-N%12 run.sbatch           (line N of $BP/work/tasks.tsv = "<dataset>\t<tool>", written by submit.sh)
#   sbatch run.sbatch <dataset> <tool>
# Output: $BP/runs/<dataset>/<tool>/ with time.txt (GNU time -v), node.txt and the tool's outputs.
# Core tools live here (singlet, starsolo, alevinfry, cellranger). Any other tool runs through its owner's hook,
# bench/public/harness/tools/<tool>.sh, called as: hook <dataset> <outdir> <libs.tsv>, with this file's variables exported.
# Flags follow docs/pipeline-validation.md: Singlet with the campaign flags; STARsolo in Cell Ranger 4+ emulation;
# alevin-fry USA mode, cr-like resolution. 16 threads, 96 GB for every tool so timings compare.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
if [ $# -ge 2 ]; then ID=$1; TOOL=$2; else IFS=$'\t' read -r ID TOOL < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $BP/work/tasks.tsv); fi
cohort_row $ID
T=${SLURM_CPUS_PER_TASK:-16}; D=$BP/data/$ID; O=$BP/runs/$ID/$TOOL
[ -e $O/.done ] && { echo "[run] $ID $TOOL done already"; exit 0; }
rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-${SLURM_JOB_ID:-$$}; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
SPECIES=${C[reference]%%;*}; GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf
CHEM=${C[chemistry]}; chem_settings $CHEM
echo "[run] $ID $TOOL ref=$SPECIES chem=$CHEM node=$(hostname) $(date -Is)"
echo "cpu: $(lscpu | sed -n 's/^Model name: *//p'); threads $T" > $O/node.txt
export SINGLET_STALL_TIMEOUT=7200   # the stall watchdog can fire during STAR's BAM sort on large inputs

# GEX reads: the barcode read (R1, 24-28 nt for 10x) and the cDNA read, whatever the run's file numbering
pick_gex() {
  local lib=${1:-gex} files f len
  R1=(); R2=()
  mapfile -t files < <(awk -F'\t' -v l=$lib '$1==l {print $3}' $D/libs.tsv)
  for f in "${files[@]}"; do
    len=$(zcat -f "$f" 2>/dev/null | sed -n 2p | tr -d '\n' | wc -c)
    if [ "$len" -le 30 ] && [ "$len" -ge 20 ]; then R1+=("$f"); elif [ "$len" -gt 30 ]; then R2+=("$f"); fi   # <20: index reads
  done
  [ ${#R1[@]} -gt 0 ] && [ ${#R1[@]} -eq ${#R2[@]} ] || { echo "cannot pair $lib reads in $D/libs.tsv"; exit 3; }
  R1CSV=$(IFS=,; echo "${R1[*]}"); R2CSV=$(IFS=,; echo "${R2[*]}")
}
# Read cap: the first read_pairs_used_m million pairs, identical for every tool (10x tars are not pre-capped)
cap_reads() {
  local cap=${C[read_pairs_used_m]%.*} tot=${C[read_pairs_m]%.*}
  if [ -n "$cap" ] && [ -n "$tot" ] && [ "$cap" -lt "$tot" ] && [ "${C[source]}" = 10x ]; then
    zcat "${R1[@]}" | head -n $((cap * 4000000)) | pigz -p 4 > $SCR/cap_R1.fastq.gz
    zcat "${R2[@]}" | head -n $((cap * 4000000)) | pigz -p 4 > $SCR/cap_R2.fastq.gz
    R1=($SCR/cap_R1.fastq.gz); R2=($SCR/cap_R2.fastq.gz); R1CSV=${R1[0]}; R2CSV=${R2[0]}
    echo "[run] capped at ${cap}M read pairs"
  fi
}

case $TOOL in
singlet)
  [ $CHEM = smartseq2 ] && { echo "plate protocols: singlet runs one sample per cell; see tools/singlet-plate.sh"; exec bash $SRC/bench/public/harness/tools/singlet-plate.sh $ID $O; }
  pick_gex; cap_reads
  # .1fq encoding is part of Singlet's cost and is timed separately (encode.time.txt); resources sum both
  cat "${R1[@]}" > $SCR/R1.fastq.gz; cat "${R2[@]}" > $SCR/R2.fastq.gz
  timed $O/encode.time.txt $SINGLET_BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq > $O/encode.log 2>&1
  rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
  EXTRA=""
  [[ ${C[capabilities]} == *donor-demux* || ${C[capabilities]} == *genetic-ancestry* ]] && \
    EXTRA="--snps $REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz --tagged-bam --bam-sort-mode disk"
  cd $SCR
  timed $O/time.txt $SINGLET_BIN $SCR/$ID.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
    --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix $EXTRA > $O/run.log 2>&1
  ;;
starsolo)
  pick_gex; cap_reads; cd $O
  case $CHEM in
    dropseq|seqwell) SOLO="--soloType CB_UMI_Simple --soloCBwhitelist None --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 8 --soloStrand Forward" ;;
    indrop) SOLO="--soloType CB_UMI_Complex --soloCBwhitelist $WLDIR/indrop_bc1.txt $WLDIR/indrop_bc2.txt --soloCBposition 0_0_2_-1 3_1_3_8 --soloUMIposition 3_9_3_14 --soloAdapterSequence GAGTGATTGCTTGTGACGCCTT --soloAdapterMismatchesNmax 3 --soloCBmatchWLtype 1MM --soloStrand Forward" ;;
    celseq2) SOLO="--soloType CB_UMI_Simple --soloCBwhitelist None --soloCBstart 7 --soloCBlen 8 --soloUMIstart 1 --soloUMIlen 6 --soloStrand Forward" ;;
    10x-*) SOLO="--soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen $CBLEN --soloUMIstart $UMISTART --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4" ;;
    *) echo "no STARsolo recipe for $CHEM"; exit 4 ;;
  esac
  timed $O/time.txt STAR --runThreadN $T --genomeDir $GENOME --readFilesIn $R2CSV $R1CSV --readFilesCommand zcat \
    $SOLO --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --outFilterScoreMin 30 \
    --outSAMtype None --outTmpDir $SCR/startmp > $O/run.log 2>&1
  ;;
alevinfry)
  [[ $CHEM == 10x-* ]] || { echo "alevin-fry: 10x chemistries only here ($CHEM)"; exit 4; }
  pick_gex; cap_reads
  IDX=$PB/refs/$SPECIES/af_idx
  timed $O/time.txt simpleaf quant --reads1 $R1CSV --reads2 $R2CSV --index $IDX/index \
    --t2g-map $IDX/index/t2g_3col.tsv --chemistry $AFCHEM --expected-ori $AFORI \
    --resolution cr-like --unfiltered-pl $WL --threads $T --output $O/af > $O/run.log 2>&1
  ;;
cellranger)
  if [ -z "$CELLRANGER" ]; then
    # No EULA yet: record 10x's own published matrix as the Cell Ranger reference (resources then unknown)
    echo "CELLRANGER unset: using the matrix published with the dataset" | tee $O/NOTE.txt
    find -L $D -maxdepth 3 -iname "*filtered*matrix*.h5" -o -maxdepth 3 -iname "*filtered*matrices*.tar.gz" | head -1 > $O/published_matrix.txt
  else
    pick_gex; mkdir -p $SCR/fq; i=0
    for f in "${R1[@]}"; do i=$((i+1)); ln -s "$f" $SCR/fq/s_S1_L00${i}_R1_001.fastq.gz; done; i=0
    for f in "${R2[@]}"; do i=$((i+1)); ln -s "$f" $SCR/fq/s_S1_L00${i}_R2_001.fastq.gz; done
    cd $SCR
    timed $O/time.txt $CELLRANGER count --id=cr --transcriptome=$REF/$SPECIES --fastqs=$SCR/fq --sample=s \
      --create-bam=false --localcores=$T --localmem=90 > $O/run.log 2>&1
    cp -r $SCR/cr/outs $O/outs
  fi
  ;;
*)
  H=$SRC/bench/public/harness/tools/$TOOL.sh
  [ -f $H ] || { echo "no runner for $TOOL (owner hook $H missing): skipped" | tee $O/SKIPPED.txt; exit 0; }
  export ID O D SCR T GENOME GTF SPECIES CHEM WL CBLEN UMISTART UMILEN STRAND SINGLET_BIN BP PB REF
  timed $O/time.txt bash $H $ID $O $D/libs.tsv > $O/run.log 2>&1
  ;;
esac
[ -f $O/time.txt ] && grep -E "Elapsed \(wall|Maximum resident" $O/time.txt || true
du -sh $O; touch $O/.done
