#!/bin/bash
#SBATCH --job-name=vdj2-dl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Inputs of the vdj2 study -> $WORK/data/vdjc/<lib>/{fastqs/,filtered_contig_annotations.csv[,gex_filtered.h5]}.
#   sbatch --export=ALL,SET=dev download.sbatch       GSE145370 development libraries (not in the burned held-out set)
#   sbatch --export=ALL,SET=heldout download.sbatch   the fresh held-out set of PREREG.md (downloaded only, nothing is run)
# The held-out download prints only file sizes and, per library, the number of Cell Ranger is_cell barcodes and of
# reference-paired cells (the denominators of the primary metric), never contig contents.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$WORK/envs/vdjcrispr/bin:$PATH HOME=$WORK/cache/condahome
D=$WORK/data/vdjc
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; } || { rm -f "$2.part"; echo "MISSING $1"; return 1; }; }
gse() {  # name gsm run
  local smp=$1 gsm=$2 run=$3 o=$D/gse145370_$1
  mkdir -p $o/fastqs
  local f=${gsm}_${smp}_filtered_contig_annotations.csv.gz
  for u in "https://www.ncbi.nlm.nih.gov/geo/download/?acc=$gsm&format=file&file=$f" \
           https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4317nnn/$gsm/suppl/$f ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4317nnn/$gsm/suppl/$f; do
    [ -s $o/filtered_contig_annotations.csv ] && break
    get "$u" $o/fca.csv.gz && gunzip -t $o/fca.csv.gz && gunzip -c $o/fca.csv.gz > $o/filtered_contig_annotations.csv || rm -f $o/fca.csv.gz $o/filtered_contig_annotations.csv
  done
  rm -f $o/fca.csv.gz
  if [ ! -s $o/fastqs/${run}_S1_L001_R2_001.fastq.gz ]; then
    T=/tmp/$USER-$SLURM_JOB_ID/$run; mkdir -p $T
    prefetch -q -O $T $run && fasterq-dump -q --split-files --include-technical -e 8 -O $T $T/$run
    ls -la $T
    # 10x V(D)J on SRA: _1 = barcode+UMI read (R1), _2 = insert (R2); a 3-file run carries the index read as _3
    pigz -p 8 -c $T/${run}_1.fastq > $o/fastqs/${run}_S1_L001_R1_001.fastq.gz
    pigz -p 8 -c $T/${run}_2.fastq > $o/fastqs/${run}_S1_L001_R2_001.fastq.gz
    rm -rf $T
  fi
  echo "lib gse145370_$smp $gsm $run"
}
tenx() {  # name ver dataset lib(t|b) [gex h5 name]
  local name=$1 ver=$2 ds=$3 lib=$4 o=$D/$1 X=https://cf.10xgenomics.com/samples/cell-vdj
  mkdir -p $o
  get $X/$ver/$ds/${ds}_${lib}_filtered_contig_annotations.csv $o/filtered_contig_annotations.csv
  get $X/$ver/$ds/${ds}_${lib}_metrics_summary.csv $o/metrics_summary.csv || true
  get $X/$ver/$ds/${ds}_filtered_feature_bc_matrix.h5 $o/gex_filtered.h5
  if [ ! -d $o/fastqs ]; then
    get $X/$ver/$ds/${ds}_${lib}_fastqs.tar $o/fastqs.tar && mkdir -p $o/fastqs && tar -xf $o/fastqs.tar -C $o/fastqs && rm $o/fastqs.tar
  fi
  echo "lib $name $ver/$ds $lib"
}
tenxu() {  # name dir/prefix-of-fca (rel. to cell-vdj) fastq-tar (rel.) gex-h5 (rel. or -) [tar member wildcard]
  local name=$1 o=$D/$1 X=https://cf.10xgenomics.com/samples/cell-vdj
  mkdir -p $o
  get $X/$2 $o/filtered_contig_annotations.csv
  [ "$4" = - ] || get $X/$4 $o/gex_filtered.h5
  if [ ! -d $o/fastqs ]; then
    mkdir -p $o/fastqs.part
    # streamed: multi-library tars (GEX + V(D)J, up to 43 GB) are never stored, only the V(D)J-B member is extracted
    curl -fsSL --retry 5 $X/$3 | tar -x -C $o/fastqs.part --wildcards "${5:-*}" && mv $o/fastqs.part $o/fastqs || { echo "MISSING fastqs $3"; rm -rf $o/fastqs.part; }
  fi
  # some 10x tars hold plain FASTQ named .fastq.gz: compress those in place
  for f in $(find $o/fastqs -name "*.fastq.gz" 2>/dev/null); do
    # (stdin: pigz skips a file argument that ends in .gz)
    gzip -t $f 2>/dev/null || { pigz -p 8 < $f > $f.tmp && [ -s $f.tmp ] && mv $f.tmp $f && echo "recompressed $(basename $f)" || rm -f $f.tmp; }
  done
  echo "lib $name"
}
if [ "${SET:-dev}" = bcrdev ] || [ "${SET:-dev}" = bcrho ]; then
  # BCR study (PREREG.md Addendum 3): fresh human BCR libraries, data fetch only
  if [ $SET = bcrdev ]; then
    tenxu melv11_b 4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_b_filtered_contig_annotations.csv \
      4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_b_fastqs.tar \
      4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_filtered_feature_bc_matrix.h5
    tenxu nsclc22_b 2.2.0/vdj_v1_hs_nsclc_b/vdj_v1_hs_nsclc_b_filtered_contig_annotations.csv \
      2.2.0/vdj_v1_hs_nsclc_b/vdj_v1_hs_nsclc_b_fastqs.tar \
      2.2.0/vdj_v1_hs_nsclc_5gex/vdj_v1_hs_nsclc_5gex_filtered_gene_bc_matrices_h5.h5
    C=10k_PBMC_5pv2_nextgem_Chromium_X
    tenxu chx10k_b 6.1.0/${C}_$C/${C}_${C}_vdj_b_filtered_contig_annotations.csv \
      6.1.0/${C}_Multiplex/${C}_Multiplex_fastqs.tar 6.1.0/${C}_$C/${C}_${C}_count_sample_feature_bc_matrix.h5 "*vdj_b*"
    L="melv11_b nsclc22_b chx10k_b"
  else
    tenxu pbmc22_b 2.2.0/vdj_v1_hs_pbmc_b/vdj_v1_hs_pbmc_b_filtered_contig_annotations.csv \
      2.2.0/vdj_v1_hs_pbmc_b/vdj_v1_hs_pbmc_b_fastqs.tar \
      2.2.0/vdj_v1_hs_pbmc_5gex/vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices_h5.h5
    tenxu cd19_b 2.2.0/vdj_v1_hs_cd19_b/vdj_v1_hs_cd19_b_filtered_contig_annotations.csv \
      2.2.0/vdj_v1_hs_cd19_b/vdj_v1_hs_cd19_b_fastqs.tar -
    B=sc5p_v2_hs_B_1k_multi_5gex_b
    tenxu b1k_b 5.0.0/$B/${B}_vdj_b_filtered_contig_annotations.csv 5.0.0/$B/${B}_fastqs.tar \
      5.0.0/$B/${B}_count_filtered_feature_bc_matrix.h5 "*B_1k_b_fastqs*"
    L="pbmc22_b cd19_b b1k_b"
  fi
elif [ "${SET:-dev}" = dev ]; then
  gse S149B GSM4317433 SRR11094270
  gse S150B GSM4317434 SRR11094271
  gse S135B GSM4317432 SRR11094269
  L="gse145370_S149B gse145370_S150B gse145370_S135B"
else
  gse S134A GSM4317424 SRR11094261
  gse S134B GSM4317431 SRR11094268
  gse S159A GSM4317429 SRR11094266
  gse S159B GSM4317436 SRR11094273
  tenx pbmc3_t 3.1.0 vdj_v1_hs_pbmc3 t
  tenx pbmc3_b 3.1.0 vdj_v1_hs_pbmc3 b
  tenx ngpbmc3_t 3.1.0 vdj_nextgem_hs_pbmc3 t
  tenx ngpbmc3_b 3.1.0 vdj_nextgem_hs_pbmc3 b
  tenx pbmc1k_t 5.0.0 sc5p_v2_hs_PBMC_1k t
  tenx pbmc1k_b 5.0.0 sc5p_v2_hs_PBMC_1k b
  tenx melanoma_b 4.0.0 sc5p_v2_hs_melanoma_10k b
  L="gse145370_S134A gse145370_S134B gse145370_S159A gse145370_S159B pbmc3_t pbmc3_b ngpbmc3_t ngpbmc3_b pbmc1k_t pbmc1k_b melanoma_b"
fi
for l in $L; do
  echo "== $l"; du -sh $D/$l/fastqs; ls $D/$l/fastqs | head -12
  python -I - $D/$l/filtered_contig_annotations.csv <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
t = lambda x: str(x).strip().lower() in ("true", "1")
d = d[d.is_cell.map(t)] if "is_cell" in d else d
cells = d.barcode.nunique()
d = d[d.productive.map(t) & d.high_confidence.map(t)]
ch = d.groupby("barcode").chain.agg(set)
paired = sum(1 for s in ch if {"TRA", "TRB"} <= s or {"TRG", "TRD"} <= s or ("IGH" in s and ("IGK" in s or "IGL" in s)))
print(f"is_cell barcodes {cells}  reference-paired cells {paired}")
EOF
done
