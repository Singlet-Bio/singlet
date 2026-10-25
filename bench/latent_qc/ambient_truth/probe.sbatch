#!/bin/bash
#SBATCH --job-name=at-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# HEAD-probe candidate 10x barnyard (human+mouse) matrices; record which exist and their sizes.
B=https://cf.10xgenomics.com/samples/cell-exp
for u in \
 3.0.0/hgmm_1k_v3/hgmm_1k_v3_raw_feature_bc_matrix.h5 3.0.0/hgmm_1k_v3/hgmm_1k_v3_filtered_feature_bc_matrix.h5 \
 3.0.0/hgmm_5k_v3/hgmm_5k_v3_raw_feature_bc_matrix.h5 3.0.0/hgmm_5k_v3/hgmm_5k_v3_filtered_feature_bc_matrix.h5 \
 3.0.0/hgmm_10k_v3/hgmm_10k_v3_raw_feature_bc_matrix.h5 3.0.0/hgmm_10k_v3/hgmm_10k_v3_filtered_feature_bc_matrix.h5 \
 3.0.0/hgmm_1k_v2/hgmm_1k_v2_raw_feature_bc_matrix.h5 3.0.0/hgmm_1k_v2/hgmm_1k_v2_filtered_feature_bc_matrix.h5 \
 2.1.0/hgmm_12k/hgmm_12k_raw_gene_bc_matrices.tar.gz 2.1.0/hgmm_12k/hgmm_12k_filtered_gene_bc_matrices.tar.gz \
 2.1.0/hgmm_6k/hgmm_6k_raw_gene_bc_matrices.tar.gz 2.1.0/hgmm_6k/hgmm_6k_filtered_gene_bc_matrices.tar.gz \
 2.1.0/hgmm_1k/hgmm_1k_raw_gene_bc_matrices.tar.gz 2.1.0/hgmm_1k/hgmm_1k_filtered_gene_bc_matrices.tar.gz \
 3.1.0/5k_hgmm_v3_nextgem/5k_hgmm_v3_nextgem_raw_feature_bc_matrix.h5 3.1.0/5k_hgmm_v3_nextgem/5k_hgmm_v3_nextgem_filtered_feature_bc_matrix.h5 \
 3.1.0/hgmm_5k_v3_nextgem/hgmm_5k_v3_nextgem_raw_feature_bc_matrix.h5 \
 3.0.2/hgmm_1k_v3/hgmm_1k_v3_raw_feature_bc_matrix.h5 \
 1.1.0/hgmm_12k/hgmm_12k_raw_gene_bc_matrices.tar.gz 1.1.0/hgmm_6k/hgmm_6k_raw_gene_bc_matrices.tar.gz 1.1.0/hgmm_1k/hgmm_1k_raw_gene_bc_matrices.tar.gz \
 3.0.0/hgmm_1k_v2/hgmm_1k_v2_raw_gene_bc_matrices.tar.gz 3.0.0/hgmm_5k_v2/hgmm_5k_v2_raw_feature_bc_matrix.h5 \
 2.1.0/hgmm_1k_v2/hgmm_1k_v2_raw_gene_bc_matrices.tar.gz ; do
  code=$(curl -sIL -m 30 -o /tmp/h.$$ -w '%{http_code}' $B/$u)
  echo "$code $(grep -i '^content-length' /tmp/h.$$ | tail -1 | tr -d '\r') $u"
done
