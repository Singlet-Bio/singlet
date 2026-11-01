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
  S=$P/starsolo_bam; B=$S/Aligned.sortedByCoord.out.bam; [ -s "$B" ] || { echo "no STARsolo BAM in $S"; exit 6; }
  [ -e $B.bai ] || samtools index -@ 8 $B
  TAGFMT=$(samtools view $B chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1); SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
  awk -v s="$SUF" '{print $1 s}' $S/Solo.out/GeneFull/filtered/barcodes.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt
  samtools view -@ 8 -b $B chrM > $O/chrM.bam && samtools index $O/chrM.bam; samtools view -c $O/chrM.bam
  cd $O; tm mgatk bcall -i chrM.bam -n $GSM -o out -c 8 -bt CB -b barcodes.txt -ub UB -g rCRS > mgatk.log 2>&1; rc=$?; tail -5 mgatk.log; exit $rc ;;
vireo|souporcell)   # genotype demux of a known mixture; K = KDON (number of cell lines/donors from the study's GEO text); reads Singlet CPU's tagged BAM and cell calls
  K=${KDON:?KDON}; E=$W/envs/gpool; export APPTAINER_BIND=/mnt/projects:/mnt/projects
  PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
  SAM=$W/envs/pbench/bin/samtools; S=$P/starsolo_bam; B=$S/Aligned.sortedByCoord.out.bam; [ -s "$B" ] || { echo "no STARsolo BAM in $S"; exit 6; }
  [ -e $B.bai ] || $SAM index -@ 8 $B
  TAGFMT=$($SAM view $B chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1); SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
  awk -v s="$SUF" '{print $1 s}' $S/Solo.out/GeneFull/filtered/barcodes.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt; echo "K=$K threads=16" > $O/params.txt; cd $O
  if [ $T = vireo ]; then
    tm apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $B -b barcodes.txt -O cellsnp -R $PANEL -p 16 --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB > cellsnp.log 2>&1 || { tail -5 cellsnp.log; exit 7; }
    $E/venv/bin/vireo -c cellsnp -o vireo_out -N $K -p 16 --randSeed 1 --noPlot > vireo.log 2>&1; rc=$?; tail -4 vireo.log; exit $rc
  else
    FA=$REFB/fasta/genome.fa; PV=$W/bench-public/ref/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf
    [ -s $PV ] || { mkdir -p $W/bench-public/ref; zcat $PANEL > $PV.tmp.$$ && mv $PV.tmp.$$ $PV; }
    $SAM view -@ 8 -b -D CB:barcodes.txt $B > $SCR/cells.bam && $SAM index -@ 8 $SCR/cells.bam
    tm apptainer exec $W/envs/souporcell/souporcell.sif souporcell_pipeline.py -i $SCR/cells.bam -b barcodes.txt -f $FA -t 16 -o $O/out -k $K --common_variants $PV > soup.log 2>&1; rc=$?; tail -4 soup.log; exit $rc
  fi ;;
alevin)
  export ALEVIN_FRY_HOME=$PB/af_home PATH=$W/envs/pbench/bin:$PATH; IDX=$PB/refs/GRCh38-2024-A/af_idx
  ORI=fw; [ "$FIVE" = 1 ] && ORI=both
  tm simpleaf quant --reads1 $R1 --reads2 $R2 --index $IDX/index --t2g-map $IDX/index/t2g_3col.tsv --chemistry "1{b[16]u[$CH_umi]x:}2{r:}" --expected-ori $ORI \
      --resolution cr-like --unfiltered-pl $WLD/$CH_whitelist --threads 16 --output $O/af > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
kbidx)   # one shared nac index (kb-python in the mgatk_extra env)
  export PATH=$W/envs/adt/bin:$PATH; mkdir -p $W/pilot/kb_index; cd $W/pilot/kb_index
  [ -f index.idx ] && { echo "index exists"; exit 0; }
  kb ref --workflow nac -i index.idx -g t2g.txt -f1 cdna.fa -f2 nascent.fa -c1 cdna_t2c.txt -c2 nascent_t2c.txt -t 16 $REFB/fasta/genome.fa $REFB/genes/genes.gtf; exit $? ;;
kb)
  export PATH=$W/envs/adt/bin:$PATH; X=10XV3; [ "$CH_umi" = 10 ] && X=10XV2; ST=forward; [ "$FIVE" = 1 ] && ST=unstranded
  PAIRS=$(python3 -c "r1='$R1'.split(','); r2='$R2'.split(','); print(' '.join(x for p in zip(r1, r2) for x in p))")
  tm kb count --workflow nac -i $W/pilot/kb_index/index.idx -g $W/pilot/kb_index/t2g.txt -c1 $W/pilot/kb_index/cdna_t2c.txt -c2 $W/pilot/kb_index/nascent_t2c.txt -x $X       -w $WLD/$CH_whitelist --strand $ST -t 16 -m 100G -o $O/kb --h5ad $PAIRS > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
trust4)   # TCR/BCR from the 5' GEX reads: barcode+UMI from the short mate, cDNA mate as single-end input
  export PATH=$W/envs/vdjcrispr/bin:$PATH; R4=$W/data/vdjc/trust4; WL=$WLD/$CH_whitelist
  zcat ${R1//,/ } | gzip -1 > $SCR/R1.fq.gz; zcat ${R2//,/ } | gzip -1 > $SCR/R2.fq.gz
  cd $O; tm run-trust4 -f $R4/hg38_bcrtcr.fa --ref $R4/human_IMGT+C.fa -u $SCR/R2.fq.gz --readFormat bc:0:15,um:16:$((CH_umi + 15)) --barcode $SCR/R1.fq.gz --UMI $SCR/R1.fq.gz \
      --barcodeWhitelist $WL -t 16 -o TRUST --od $O > $O/run.log 2>&1; rc=$?; tail -3 $O/run.log; exit $rc ;;
htocount)   # GSM = the HTO library GSM (fetched with NODECODE); GEXGSM = the pilot GEX sample whose called cells define the columns. Singlet HTO (counter + caller) on the pilot cells.
  GP=$PL/${GEXGSM:?GEXGSM}; AT=$W/data/adt/${ATLASGSE:?}/$GSM; export PATH=$W/envs/adt/bin:$PATH PYTHONPATH=$W/src/singlet-py/claude__adt-modality-0637a6/python HOME=$W/cache/condahome
  [ -f $AT/status.json ] || { echo "no atlas panel/offset for $GSM"; exit 6; }
  mkdir -p $P/fq; ls $P/sra/*.sra | xargs -P 8 -I{} $W/envs/pbench/bin/fastq-dump --split-files --gzip -O $P/fq {} || exit 4
  CC=$GP/s1cpu/out/cell_calls.tsv; [ -s $CC ] || { echo "no cell_calls.tsv in $GP/s1cpu/out"; exit 6; }
  python3 -u $D/pilot_hto.py $AT $P/fq $CC $WLD/${HTOWL:-737K-august-2016.txt} $O > $O/run.log 2>&1; rc=$?; tail -25 $O/run.log; exit $rc ;;
htocallers)
  export APPTAINER_BIND=/mnt/projects:/mnt/projects; H=$O; HC=$P/tools/htocount/hto_counts.csv.gz; [ -s $HC ] || { echo "no HTO counts"; exit 6; }
  apptainer exec $W/images/bioc-3.20.sif Rscript $D/pilot_hto_callers.R $HC $O/bioc $W/rlib/bioc-3.20 > $O/bioc.log 2>&1; tail -3 $O/bioc.log
  apptainer exec $W/images/seurat-5.sif Rscript $D/pilot_hto_callers.R $HC $O/seurat $W/rlib/seurat-5 > $O/seurat.log 2>&1; tail -3 $O/seurat.log
  # GMM-Demux (pip into scratch)
  python3 -m venv $SCR/v && $SCR/v/bin/pip install -q GMM-Demux > $O/gmm_install.log 2>&1 && { zcat $HC | head -1 | tr ',' '
' | tail -n +2 | paste -sd, > $SCR/tags.txt; zcat $HC > $SCR/hto.csv
    (cd $O && $SCR/v/bin/GMM-demux -f $O/gmm -c $SCR/hto.csv $(cat $SCR/tags.txt) > $O/gmm.log 2>&1; echo "GMM-Demux rc=$?"; tail -3 $O/gmm.log); } || echo "GMM-Demux install failed: $(tail -2 $O/gmm_install.log)"
  ls $O; exit 0 ;;
vdjlib)   # GSM = the V(D)J library's GSM (fetched with NODECODE by the submitter); paired GEX sample in GEXGSM. Singlet V(D)J (S11's run_vdj.py) and TRUST4 on the same reads.
  export PATH=$W/envs/vdjcrispr/bin:$PATH PYTHONPATH= HOME=$W/cache/condahome NUMBA_CACHE_DIR=$W/cache/numba-vdjc; R4=$W/data/vdjc/trust4
  python3 - <<PY || exit 4
import json, subprocess, os
r = json.load(open("$P/runs.json"))["runs"]; left = 100000000; os.makedirs("$P/fq", exist_ok=True)
for acc, n in r:
    if left <= 0: break
    k = min(n, left); left -= k
    subprocess.run(["$W/envs/pbench/bin/fastq-dump", "-X", str(k), "--split-files", "--gzip", "-O", "$P/fq", "$P/sra/%s.sra" % acc], check=True)
PY
  FD=$P/fastqs; mkdir -p $FD; python3 - <<PY
import glob, gzip, os, collections
runs = collections.defaultdict(dict)
for f in sorted(glob.glob("$P/fq/*.fastq.gz")):
    acc, n = os.path.basename(f)[:-9].rsplit("_", 1)
    with gzip.open(f, "rt") as h: h.readline(); L = len(h.readline().strip())
    runs[acc][n] = (f, L)
for i, acc in enumerate(sorted(runs)):
    fl = sorted(runs[acc].values(), key=lambda x: x[1]); bc = [x for x in fl if 24 <= x[1] <= 30][0]; cd = fl[-1]
    os.symlink(bc[0], "$FD/%s_S1_L00%d_R1_001.fastq.gz" % (acc, i + 1)); os.symlink(cd[0], "$FD/%s_S1_L00%d_R2_001.fastq.gz" % (acc, i + 1))
PY
  R1F=$(ls $FD/*_R1_*.fastq.gz | head -1); best=""; bestf=0
  for w in 737K-august-2016.txt 3M-february-2018.txt; do f=$(zcat $R1F | awk 'NR%4==2{print substr($0,1,16)}' | head -200000 | grep -cxFf $WLD/$w); echo "whitelist $w: $f"; [ $f -gt $bestf ] && { bestf=$f; best=$w; }; done
  [ $bestf -gt 20000 ] || { echo "no whitelist matches the V(D)J library barcodes"; exit 5; }
  source $W/jobs/vdjc/src.sh; echo "vdj src $SRC sha $SHA"
  tm python -P $SRC/bench/vdj/run_vdj.py --fastq-dir $FD --whitelist $WLD/$best --ref $SRC/python/singlet/vdj/data/vdj_human.tsv.gz --out $O/singlet --threads 16 > $O/singlet.log 2>&1; echo "singlet vdj rc=$?"; tail -3 $O/singlet.log
  cat $FD/*_R1_*.fastq.gz > $SCR/R1.fq.gz; cat $FD/*_R2_*.fastq.gz > $SCR/R2.fq.gz; mkdir -p $O/trust4
  cd $O/trust4; /usr/bin/time -v -o $O/trust4.time run-trust4 -f $R4/hg38_bcrtcr.fa --ref $R4/human_IMGT+C.fa -u $SCR/R2.fq.gz --readFormat bc:0:15,um:16:25 --barcode $SCR/R1.fq.gz --UMI $SCR/R1.fq.gz      --barcodeWhitelist $WLD/$best -t 16 -o TRUST --od $O/trust4 > $O/trust4.log 2>&1; rc=$?; echo "trust4 rc=$rc"; grep -E "Elapsed \(wall|Maximum resident" $O/trust4.time; exit $rc ;;
vdjclean) rm -rf $P/sra $P/stage $P/fq $P/fastqs; du -sh $P ;;
finalclean)   # all competitors have ended: drop KEEP; delete the reads unless the GPU lane has not passed this sample yet (its own cleanup then deletes them)
  rm -f $P/KEEP
  [ -d $P/s1cpu/out ] && [ -d $P/s1gpu/out ] && [ ! -s $P/equiv.txt ] && python3 $D/out_equiv.py $P/s1cpu/out $P/s1gpu/out > $P/equiv.txt 2>&1
  if [ -e $P/.gpu_ended ]; then rm -rf $P/stage $P/fq $P/cap1fq; find $P \( -name '*.bam' -o -name '*.bai' \) -delete; echo "reads deleted (.sra kept for the engine reruns)"; else echo "GPU lane not done: lane cleanup will delete"; fi
  du -sh $P ;;
esac
