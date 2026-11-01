#!/bin/bash
# S15 beta pilot: competitor stages, one per sample.   pilot_tools.sh TOOL GSM    (Slurm options come from submit_tools.sh)
# Inputs: $P/fq (first 100 M pairs), $P/starsolo (Solo.out, files.env = R1/R2 lists, chem.txt = whitelist/UMI), $P/s1cpu/out (Singlet CPU run with BAM).
# Tools: rtools (SoupX, decontX, scDblFinder) | edrops (DropletUtils) | cellbender | kraken | mgatk | alevin | kbidx | kb | trust4 | finalclean
set -uo pipefail
T=$1; GSM=${2:-}
W=/mnt/projects/debruinz_project/debruinz/singlet; PL=$W/pilot; P=$PL/$GSM; D=$W/jobs/s15; PB=$W/pipeline-benchmark
REFB=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; WLD=$PB/whitelists
row=$(awk -F'\t' -v g=$GSM '$1==g' $D/samples.tsv); FIVE=$(echo "$row" | cut -f3)
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export HOME=$W/cache/condahome XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl APPTAINER_CACHEDIR=$W/cache/apptainer
command -v apptainer > /dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2> /dev/null; module load apptainer 2> /dev/null; }
echo "== s15 tool=$T $GSM node=$(hostname) $(date -Is)"
O=$P/tools/$T; [ "$T" = kbidx ] || { rm -rf $O; mkdir -p $O; }
[ -n "$GSM" ] && [ -f $P/starsolo/files.env ] && . $P/starsolo/files.env
[ -f $P/starsolo/chem.txt ] && eval "$(sed 's/ /\n/g' $P/starsolo/chem.txt | sed 's/^/CH_/')"   # CH_whitelist, CH_umi, CH_strand
GF=$P/starsolo/Solo.out/GeneFull
tm() { /usr/bin/time -v -o $O/time.txt "$@"; rc=$?; grep -E "Elapsed \(wall|Maximum resident" $O/time.txt; return $rc; }
case $T in
rtools)
  export PATH=$W/envs/ambient/bin:$PATH OMP_NUM_THREADS=8
  tm Rscript $D/pilot_r.R $GF $O > $O/run.log 2>&1; rc=$?; tail -5 $O/run.log; cat $O/summary.tsv 2> /dev/null; exit $rc ;;
edrops)
  tm apptainer exec --bind $W --bind /tmp $W/images/bioc-3.20.sif Rscript $D/pilot_edrops.R $GF/raw $O $W/rlib/bioc-3.20 > $O/run.log 2>&1; rc=$?; tail -5 $O/run.log; cat $O/summary.tsv 2> /dev/null; exit $rc ;;
cellbender)
  NCALL=$(wc -l < $GF/filtered/barcodes.tsv); TOT=$((NCALL * 4 + 20000)); echo "expected $NCALL total-droplets $TOT"
  cd $SCR; tm apptainer exec --nv --bind $W --bind /tmp $W/images/cellbender-0.3.2.sif cellbender remove-background --cuda --input $GF/raw --output $SCR/cb.h5 \
      --expected-cells $NCALL --total-droplets-included $TOT --epochs 150 > $O/run.log 2>&1; rc=$?; tail -5 $O/run.log
  cp $SCR/cb_cell_barcodes.csv $O/ 2> /dev/null; cp $SCR/cb.h5 $O/cb.h5 2> /dev/null; cp $SCR/cb_metrics.csv $O/ 2> /dev/null; exit $rc ;;
kraken)   # viral k2 DB (the one S-nonhost used): reads of the cDNA mate; classified reads are kept, the rest summarised in the report
  export PATH=$W/envs/kraken2/bin:$PATH
  zcat ${R2//,/ } | tm kraken2 --db $W/modality-bench/sj_nonhost/db/k2_viral --threads 16 --confidence 0.1 --report $O/report.txt --output $O/classified_reads.tsv /dev/stdin > $O/run.log 2>&1; rc=$?
  gzip -f $O/classified_reads.tsv; tail -3 $O/run.log; head -12 $O/report.txt; exit $rc ;;
mgatk)
  export PATH=$W/envs/mtl/bin:$PATH PYTHONPATH=$W/envs/mgatk_extra; ulimit -n $(ulimit -Hn) 2> /dev/null
  S=$P/s1cpu/out; B=$(ls $S/*.bam 2> /dev/null | head -1); [ -n "$B" ] || { echo "no Singlet BAM in $S"; exit 6; }
  [ -e $B.bai ] || samtools index -@ 8 $B
  TAGFMT=$(samtools view $B chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1); SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
  awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $S/cell_calls.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt
  samtools view -@ 8 -b $B chrM > $O/chrM.bam && samtools index $O/chrM.bam; samtools view -c $O/chrM.bam
  cd $O; tm mgatk bcall -i chrM.bam -n $GSM -o out -c 8 -bt CB -b barcodes.txt -ub UB -g rCRS > mgatk.log 2>&1; rc=$?; tail -5 mgatk.log; exit $rc ;;
alevin)
  export ALEVIN_FRY_HOME=$PB/af_home PATH=$W/envs/pbench/bin:$PATH; IDX=$PB/refs/GRCh38-2024-A/af_idx
  ORI=fw; [ "$FIVE" = 1 ] && ORI=both
  tm simpleaf quant --reads1 $R1 --reads2 $R2 --index $IDX/index --t2g-map $IDX/index/t2g_3col.tsv --chemistry "1{b[16]u[$CH_umi]x:}2{r:}" --expected-ori $ORI \
      --resolution cr-like --unfiltered-pl $WLD/$CH_whitelist --threads 16 --output $O/af > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
kbidx)   # one shared nac index (kb-python in the mgatk_extra env)
  export PATH=$W/envs/mgatk_extra/bin:$PATH; mkdir -p $W/pilot/kb_index; cd $W/pilot/kb_index
  [ -f index.idx ] && { echo "index exists"; exit 0; }
  kb ref --workflow nac -i index.idx -g t2g.txt -c1 cdna.txt -c2 nascent.txt -t 16 $REFB/fasta/genome.fa $REFB/genes/genes.gtf; exit $? ;;
kb)
  export PATH=$W/envs/mgatk_extra/bin:$PATH; X=10XV3; [ "$CH_umi" = 10 ] && X=10XV2; ST=forward; [ "$FIVE" = 1 ] && ST=unstranded
  PAIRS=$(python3 -c "r1='$R1'.split(','); r2='$R2'.split(','); print(' '.join(x for p in zip(r1, r2) for x in p))")
  tm kb count --workflow nac -i $W/pilot/kb_index/index.idx -g $W/pilot/kb_index/t2g.txt -c1 $W/pilot/kb_index/cdna.txt -c2 $W/pilot/kb_index/nascent.txt -x $X       -w $WLD/$CH_whitelist --strand $ST -t 16 -m 100G -o $O/kb --h5ad $PAIRS > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
trust4)   # TCR/BCR from the 5' GEX reads: barcode+UMI from the short mate, cDNA mate as single-end input
  export PATH=$W/envs/vdjcrispr/bin:$PATH; R4=$W/data/vdjc/trust4; WL=$WLD/$CH_whitelist
  zcat ${R1//,/ } | gzip -1 > $SCR/R1.fq.gz; zcat ${R2//,/ } | gzip -1 > $SCR/R2.fq.gz
  cd $O; tm run-trust4 -f $R4/hg38_bcrtcr.fa --ref $R4/human_IMGT+C.fa -u $SCR/R2.fq.gz --readFormat bc:0:15,um:16:$((CH_umi + 15)) --barcode $SCR/R1.fq.gz --UMI $SCR/R1.fq.gz \
      --barcodeWhitelist $WL -t 16 -o TRUST --od $O > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
finalclean)   # all competitors have ended: drop KEEP; delete the reads unless the GPU lane has not passed this sample yet (its own cleanup then deletes them)
  rm -f $P/KEEP
  if [ -e $P/.gpu_ended ]; then rm -rf $P/sra $P/stage $P/fq $P/cap1fq; find $P \( -name '*.bam' -o -name '*.bai' \) -delete; echo "reads deleted"; else echo "GPU lane not done: lane cleanup will delete"; fi
  du -sh $P ;;
esac
