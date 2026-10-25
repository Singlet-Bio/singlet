#!/bin/bash
#SBATCH --job-name=bp-ap-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --exclusive
#SBATCH --constraint=xeon-6248
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch ap_run.sbatch DATASET REP "TOOL TOOL ..."
#     DATASET  pbmc1k_v3 | nuclei2k_v2 | gsm5815984
#     TOOL     singlet | singlet-core | starsolo | starsolo-bam | alevinfry
# Timing of one dataset on one node: every tool reads the same two FASTQ files (R1 barcode+UMI, R2 cDNA; lanes concatenated,
# copied to node-local NVMe before the clock starts), the same reference (10x 2024-A, GENCODE v44 / M33), 16 threads and the
# same 128 GB memory limit, on an exclusive node of one CPU type (Xeon Gold 6248, constraint xeon-6248). Tools run one after the
# other; the order alternates between repetitions.
#   singlet        campaign binary, campaign flags (--pipeline: counts, introns, junctions, donor demux, mtDNA, ambient, doublets,
#                  cell calling), non-host screen off; wall = `singlet encode` (FASTQ -> .1fq) + `singlet <1fq>`
#   singlet-core   the same without --pipeline (counts, introns, junctions, cell calling only)
#   starsolo       STARsolo 2.7.11b, Gene GeneFull Velocyto SJ, Cell Ranger 4+ settings, no BAM (the fastest STARsolo run)
#   starsolo-bam   the same plus a coordinate-sorted BAM with CB/UB tags (what Singlet's STAR step also produces)
#   alevinfry      simpleaf quant, spliced+unspliced index, cr-like resolution
# Output: $BP/perf/DATASET/TOOL_rREP/{time.txt,mem.json,node.txt,...}. Cell Ranger is not installed on Clipper and is not run.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
DS=$1; REP=$2; TOOLS=${3:-"singlet singlet-core starsolo starsolo-bam alevinfry"}
verify_bin
export PATH=$ENVDIR/bin:$PATH ALEVIN_FRY_HOME=$PB/af_home
unset SINGLET_REF_BASE
export SINGLET_STALL_TIMEOUT=10800
SPECIES=$(species_of $DS); REF=$REFBASE/$SPECIES; GENOME=$REF/star_2.7.11b; GTF=$REF/genes/genes.gtf
case $DS in
  pbmc1k_v3|nuclei2k_v2) D=$PB/data/$DS; R1S=$(grep "_R1_" $D/gex_fastqs.txt); R2S=$(grep "_R2_" $D/gex_fastqs.txt) ;;
  gsm5815984) R1S=$BP/data/gsm5815984/R1.fastq.gz; R2S=$BP/data/gsm5815984/R2.fastq.gz ;;
  *) echo "unknown dataset $DS"; exit 2 ;;
esac
case $DS in
  nuclei2k_v2) UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt; AFCHEM=10xv2 ;;
  *) UMILEN=12; WL=$PB/whitelists/3M-february-2018.txt; AFCHEM=10xv3 ;;
esac
B=$BP/perf/$DS; mkdir -p $B
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[ap] $DS rep $REP node=$(hostname) cpu=$(cpu_model) cores=$(nproc) $(date -Is)"
echo "[ap] cellranger: $(command -v cellranger || echo 'not on PATH'); $(ls -d /opt/*ellranger* /mnt/projects/*/*ellranger* 2>/dev/null | head -3)"
t0=$(date +%s); cat $R1S > $SCR/R1.fastq.gz; cat $R2S > $SCR/R2.fastq.gz
echo "[ap] inputs copied to node-local disk in $(( $(date +%s) - t0 )) s: $(du -shL $SCR/R1.fastq.gz $SCR/R2.fastq.gz | tr '\n' ' ')"
R1=$SCR/R1.fastq.gz; R2=$SCR/R2.fastq.gz
# warm the page cache with the reference files every tool reads, so the first tool of the run does not pay the cold NFS read
t0=$(date +%s); cat $GENOME/* $PB/refs/$SPECIES/af_idx/index/* > /dev/null 2>&1 || true
echo "[ap] reference files read in $(( $(date +%s) - t0 )) s"

run_tool() {
  local tool=$1 O=$B/${1}_r$REP
  rm -rf $O; mkdir -p $O; cd $SCR
  echo "[ap] $(cpu_model) $(hostname) $(date -Is)" > $O/node.txt
  local rc=0
  case $tool in
    singlet|singlet-core)
      local EXTRA="--pipeline"; [ $tool = singlet-core ] && EXTRA=""
      timed $O/encode $BIN encode --reads $R1 $R2 -o $SCR/s.1fq > $O/encode.log 2>&1 || rc=$?
      [ $rc = 0 ] && { timed $O/run $BIN $SCR/s.1fq --ref-base $REFBASE --genome-dir $GENOME --exons $GTF --out-prefix $O/out/ \
          --threads 16 $EXTRA --output-format mtx --raw-matrix > $O/run.log 2>&1 || rc=$?; }
      rm -f $SCR/s.1fq ;;
    starsolo|starsolo-bam)
      local BAM=(--outSAMtype None)
      [ $tool = starsolo-bam ] && BAM=(--outSAMtype BAM SortedByCoordinate --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM --limitBAMsortRAM 40000000000)
      mkdir -p $O/out
      timed $O/run STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat \
        --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN \
        --soloBarcodeReadLength 0 --soloStrand Forward --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR \
        --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR \
        --clipAdapterType CellRanger4 --outFilterScoreMin 30 "${BAM[@]}" --outFileNamePrefix $O/out/ --outTmpDir $SCR/startmp_$tool \
        > $O/run.log 2>&1 || rc=$?
      rm -rf $SCR/startmp_$tool ;;
    alevinfry)
      timed $O/run simpleaf quant --reads1 $R1 --reads2 $R2 --index $PB/refs/$SPECIES/af_idx/index \
        --t2g-map $PB/refs/$SPECIES/af_idx/index/t2g_3col.tsv --chemistry $AFCHEM --expected-ori fw --resolution cr-like \
        --unfiltered-pl $WL --threads 16 --output $O/af > $O/run.log 2>&1 || rc=$?
      ;;
    *) echo "unknown tool $tool"; return 2 ;;
  esac
  echo "exit $rc" > $O/exit.txt
  # keep the evidence that the alignments are comparable (reads, unique %) and drop the bulky outputs
  for f in $O/out/Log.final.out $O/out/star_Log.final.out; do [ -f $f ] && cp $f $O/Log.final.out; done
  for s in $O/out/Solo.out $O/out/star_Solo.out; do
    [ -f $s/GeneFull/Summary.csv ] && cp $s/GeneFull/Summary.csv $O/GeneFull.Summary.csv
    [ -f $s/Gene/Summary.csv ] && cp $s/Gene/Summary.csv $O/Gene.Summary.csv
  done
  [ -d $O/af ] && { cp $O/af/af_quant/featureDump.txt $O/ 2>/dev/null || true; cp $O/af/af_quant/quant.json $O/ 2>/dev/null || true; }
  du -sh $O/out $O/af 2>/dev/null | tee $O/out_size.txt
  rm -rf $O/out $O/af
  echo "[ap] $tool rep $REP exit $rc: $(grep -h 'Elapsed' $O/run/time.txt $O/encode/time.txt 2>/dev/null | sed 's/.*): //' | tr '\n' ' ') rss $(cat $O/run/mem.json 2>/dev/null)"
}

ORDER="$TOOLS"
if [ $(( REP % 2 )) = 0 ]; then ORDER=$(for t in $TOOLS; do echo $t; done | tac | tr '\n' ' '); fi
for t in $ORDER; do run_tool $t || echo "[ap] $t failed"; done
echo "[ap] done $(date -Is)"
