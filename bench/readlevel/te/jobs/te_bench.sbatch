#!/bin/bash
#SBATCH --job-name=rl-te
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=320G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# TE benchmark for one dataset, all on one allocation: the engine (pipeline default TE annotation) side by side with
# STARsolo -> scTE and SoloTE, and for A375 the nanopore subfamily truth (minimap2 -> te_longread_truth.py).
# Needs NAME (a375_10x | mesc_2c) and BIN (engine binary dir). Comparator outputs: $W/jobs/readlevel/te/$NAME/.
set -uo pipefail
: "${NAME:?}" "${BIN:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; ENV=$W/envs/tebench; T=$W/tools/te
REF=/mnt/projects/debruinz_project/cellarium/reference; WL=$W/pipeline-benchmark/whitelists; D=$W/data/readlevel/te/$NAME
case $NAME in
  a375_10x) GENOME=GRCh38-2024-A; TEG=$REF/pileup_indices/te/GRCh38_GENCODE_rmsk_TE.gtf.gz; R1=$D/SRR21492169_2.fastq.gz; R2=$D/SRR21492169_3.fastq.gz ;;
  mesc_2c)  GENOME=GRCm39-2024-A; TEG=$REF/pileup_indices/te/GRCm39_GENCODE_rmsk_TE.gtf.gz; R1=$D/ERR3981386_1.fastq.gz; R2=$D/ERR3981386_2.fastq.gz ;;
  *) echo "unknown NAME"; exit 2 ;;
esac
G=$REF/$GENOME; O=$J/te/$NAME; mkdir -p $O; N=${SLURM_CPUS_PER_TASK:-28}
SCR=/tmp/$USER-te-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH
echo "node=$(hostname) name=$NAME genome=$GENOME bin=$BIN cpus=$N scTE=$(cat $T/scTE.commit) SoloTE=$(cat $T/SoloTE.commit) $(date -Is)"

[ -x $ENV/bin/Rscript ] || MAMBA_ROOT_PREFIX=$W/tools/mm/root CONDA_PKGS_DIRS=$W/cache/conda_pkgs HOME=$W/cache/condahome \
  $W/tools/mm/bin/micromamba install -y -p $ENV -c conda-forge --strict-channel-priority r-base > $O/rbase.log 2>&1   # SoloTE's matrix step
# the barcode read is the short one (ENA file order is not guaranteed)
# swap when the first file is clearly longer; for two long reads, decide by poly-T (see below)
L1=$(zcat $R1 | head -2 | tail -1 | awk '{print length($0)}'); L2=$(zcat $R2 | head -2 | tail -1 | awk '{print length($0)}')
# when both reads are long (E-MTAB-8869: 100 and 101 bp, varying per read) the barcode read is the one with poly-T after base 28
polyt() { zcat $1 | head -8000 | awk 'NR%4==2{s=substr($0,29,30); n=gsub(/T/,"",s); if(n>=20)c++; t++} END{print int(1000*c/t)}'; }
if [ "$L1" -gt $((L2 + 20)) ]; then SWAP=1
elif [ "$L2" -gt $((L1 + 20)) ]; then SWAP=0
else P1=$(polyt $R1); P2=$(polyt $R2); echo "poly-T per mille: $(basename $R1) $P1, $(basename $R2) $P2"; [ "$P2" -gt "$P1" ] && SWAP=1 || SWAP=0; fi
if [ $SWAP = 1 ]; then t=$R1; R1=$R2; R2=$t; L1=$L2; fi
# 10x names for the engine
F=$D/10x; mkdir -p $F; ln -sfn $R1 $F/${NAME}_S1_L001_R1_001.fastq.gz; ln -sfn $R2 $F/${NAME}_S1_L001_R2_001.fastq.gz
if [ "$L1" -ge 28 ]; then CBWL=$WL/3M-february-2018.txt; UMIL=12; else CBWL=$WL/737K-august-2016.txt; UMIL=10; fi
echo "R1 length $L1 -> whitelist $(basename $CBWL), UMI $UMIL"

# engine (SKIP_ENGINE=1: the run happens elsewhere)
( [ "${SKIP_ENGINE:-0}" = 1 ] && exit 0
  if [ ! -x $BIN/singlet-gpu ]; then   # build the branch tip first (BIN = $J/bin/<name>, see build_arch.sbatch)
    SLURM_JOB_ID=${SLURM_JOB_ID}b SLURM_CPUS_PER_TASK=$((N/2)) BR=${BR:-feat/te-counts} NAME=$(basename $BIN) bash $J/build_arch.sbatch || exit 1
  fi
  SLURM_JOB_ID=${SLURM_JOB_ID}e SLURM_CPUS_PER_TASK=$((N/2)) BIN=$BIN TAG=te SAMPLE=$NAME FQDIR=$F GENOME=$GENOME bash $J/run_engine.sbatch
) > $O/engine.log 2>&1 & PE=$!

# STARsolo BAM for the comparators (multimappers kept as one primary, CB/UB tags), then scTE and SoloTE
comparators() {
  set -uo pipefail
  local C=$((N/2))
  if [ ! -s $O/starsolo/Aligned.sortedByCoord.out.bam ]; then
    mkdir -p $O/starsolo
    STAR --runThreadN $C --genomeDir $G/star_2.7.11b --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple \
      --soloCBwhitelist $CBWL --soloUMIlen $UMIL --soloBarcodeReadLength 0 --soloFeatures Gene --soloCellFilter EmptyDrops_CR \
      --soloUMIdedup 1MM_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --clipAdapterType CellRanger4 \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMmultNmax 1 --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 60000000000 \
      --outSAMattributes NH HI AS nM CR CY UR UY CB UB GX GN sS sQ sM --outFileNamePrefix $O/starsolo/ > $O/starsolo.log 2>&1 || return 1
    samtools index -@ $C $O/starsolo/Aligned.sortedByCoord.out.bam || return 1
  fi
  local BAM=$O/starsolo/Aligned.sortedByCoord.out.bam
  # annotations from the engine's TE GTF: scTE BED6 (name = subfamily), SoloTE BED (id = chr|start|end|sub:family:class|score|strand)
  zcat $TEG | awk -F'\t' 'BEGIN{OFS="\t"} !/^#/{match($9,/gene_id "[^"]+"/); s=substr($9,RSTART+9,RLENGTH-10);
      match($9,/family_id "[^"]+"/); f=substr($9,RSTART+11,RLENGTH-12); match($9,/class_id "[^"]+"/); c=substr($9,RSTART+10,RLENGTH-11);
      sub(/\?$/,"",f); sub(/\?$/,"",c); st=$4-1;
      print $1,st,$5,s,0,$7 > "'$SCR'/scte.bed"; print $1,st,$5,$1"|"st"|"$5"|"s":"f":"c"|0|"$7,0,$7 > "'$SCR'/solote.bed"}'
  sort -k1,1 -k2,2n $SCR/solote.bed > $O/solote_te.bed
  ( [ -s $O/scte.h5ad ] && exit 0   # finished earlier
    cd $SCR && mkdir -p scte && cd scte
    [ -s $O/scte_idx.exclusive.idx ] || { scTE_build -te $SCR/scte.bed -gene $G/genes/genes.gtf -o $O/scte_idx > $O/scte_build.log 2>&1; }
    /usr/bin/time -v -o $O/scte_time.txt scTE -i $BAM -o scte -x $O/scte_idx.exclusive.idx --hdf5 True -CB CB -UMI UB -p $((C/2)) \
      --min_genes 100 --expect-cells 20000 > $O/scte.log 2>&1 && mv -f scte.h5ad $O/scte.h5ad ) &   # -o takes a bare name: scTE builds its temp paths from it
  local P1=$!
  ( ls $O/solote/*/*_MATRIX/matrix.mtx* >/dev/null 2>&1 && exit 0   # finished earlier
    /usr/bin/time -v -o $O/solote_time.txt python $T/SoloTE/SoloTE_pipeline.py --threads $((C/2)) --bam $BAM --teannotation $O/solote_te.bed \
      --outputprefix $NAME --outputdir $O/solote > $O/solote.log 2>&1 ) &
  local P2=$!
  wait $P1; local r1=$?; wait $P2; local r2=$?; echo "scTE exit $r1, SoloTE exit $r2"; return $((r1 + r2))
}
comparators > $O/comparators.log 2>&1; RC=$?; echo "comparators exit $RC"; tail -3 $O/comparators.log

# nanopore truth (A375), after the comparators so its threads do not crowd them
if [ "$NAME" = a375_10x ] && [ ! -s $O/longread_truth.tsv ]; then
  ONT=$(ls $W/data/readlevel/te/a375_ont/*.fastq.gz | head -1)
  zcat $ONT | head -n ${ONT_READS:-20000000} > $SCR/ont.fq   # first 5 M reads
  minimap2 -ax splice -uf -k14 -t $((N/2)) $G/fasta/genome.fa $SCR/ont.fq 2> $O/minimap2.log | samtools view -b -F 4 - > $SCR/ont.bam
  python $J/te_longread_truth.py --bam $SCR/ont.bam --te-gtf $TEG --gene-gtf $G/genes/genes.gtf --out $O/longread_truth.tsv > $O/longread_truth.log 2>&1
  tail -2 $O/longread_truth.log
fi
wait $PE; echo "engine exit $?"; tail -5 $O/engine.log
