#!/bin/bash
#SBATCH --job-name=as-download
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Download the ATAC / Multiome / Visium benchmark datasets (10x Genomics public datasets, CC BY 4.0): FASTQs and the
# published Cell Ranger ARC / Cell Ranger ATAC / Space Ranger outputs. Idempotent (curl -C -, skips finished files).
set -euo pipefail
D=/mnt/projects/debruinz_project/debruinz/singlet/data/atac-spatial; mkdir -p $D; cd $D
get() { # url dir
  mkdir -p $2; local f=$2/${1##*/}
  [ -s $f.done ] && return 0
  curl -sSfL --retry 5 -C - -o $f "$1" && touch $f.done && echo "ok $(du -h $f | cut -f1) $f"
}
ARC=https://cf.10xgenomics.com/samples/cell-arc/2.0.0
for s in pbmc_granulocyte_sorted_3k pbmc_granulocyte_sorted_10k; do
  for x in summary.csv per_barcode_metrics.csv atac_fragments.tsv.gz atac_fragments.tsv.gz.tbi atac_peaks.bed \
           raw_feature_bc_matrix.h5 filtered_feature_bc_matrix.h5 atac_peak_annotation.tsv; do get $ARC/$s/${s}_$x $s; done
done
# ATAC->GEX barcode pairs (translation table) from other published ARC runs: per_barcode_metrics only
for s in pbmc_granulocyte_sorted_10k pbmc_unsorted_3k pbmc_unsorted_10k human_brain_3k e18_mouse_brain_fresh_5k lymph_node_lymphoma_14k; do
  get $ARC/$s/${s}_per_barcode_metrics.csv arc_pbm || echo "missing $s"
done
CRA=https://cf.10xgenomics.com/samples/cell-atac/1.2.0
for s in atac_pbmc_5k_nextgem; do
  for x in summary.csv singlecell.csv fragments.tsv.gz fragments.tsv.gz.tbi peaks.bed; do get $CRA/$s/${s}_$x $s; done
done
SR=https://cf.10xgenomics.com/samples/spatial-exp/1.1.0
for s in V1_Mouse_Brain_Sagittal_Anterior; do
  for x in metrics_summary.csv spatial.tar.gz raw_feature_bc_matrix.h5 filtered_feature_bc_matrix.h5; do get $SR/$s/${s}_$x $s; done
done
for s in V1_Human_Lymph_Node V1_Breast_Cancer_Block_A_Section_1; do
  for x in metrics_summary.csv spatial.tar.gz raw_feature_bc_matrix.h5 filtered_feature_bc_matrix.h5; do get $SR/$s/${s}_$x $s; done
done
# FASTQs last (large)
get $ARC/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_fastqs.tar pbmc_granulocyte_sorted_3k
get https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_fastqs.tar pbmc_granulocyte_sorted_10k
get $SR/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_fastqs.tar V1_Mouse_Brain_Sagittal_Anterior
get $CRA/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs.tar atac_pbmc_5k_nextgem
for t in */*_fastqs.tar; do
  [ -e ${t%.tar}.extracted ] && continue
  tar -xf $t -C $(dirname $t) && touch ${t%.tar}.extracted && rm -f $t && echo "extracted $t"
done
find $D -name "*.fastq.gz" | sort
du -sh $D/*
