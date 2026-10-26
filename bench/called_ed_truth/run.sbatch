#!/bin/bash
# called_ed ground-truth study (docs/called-ed-truth.md). Scripts are put into $WORK/jobs/called_ed_truth (with cellcall.py from
# branch claude/called-v2-migration); results go to $WORK/state/called_ed_truth. Reads only existing run outputs.
#
#   sbatch --array=0-5 --export=ALL,MODE=datasets run.sbatch
#   sbatch --array=0-17 --export=ALL,MODE=revalidate run.sbatch      (then MODE=aggregate, no array)
#SBATCH --job-name=cv2t
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
J=$W/jobs/called_ed_truth
OUT=$W/state/called_ed_truth
PB=$W/pipeline-benchmark
MB=$W/modality-bench
PY=$W/envs/gpool/venv/bin/python
export CELLCALL_DIR=$J CV2_BENCH=$W/src/singlet-py/claude__festive-vaughan-4a4a8f/bench
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} NUMBA_CACHE_DIR=$W/cache/numba-cv2t OMP_NUM_THREADS=4
TASK=${SLURM_ARRAY_TASK_ID:-0}
SCRATCH=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-$TASK; mkdir -p "$SCRATCH" "$OUT"; trap 'rm -rf "$SCRATCH"' EXIT
MODE=${MODE:-datasets}
echo "mode=$MODE task=$TASK host=$(hostname)"

case "$MODE" in
  datasets)
    case $TASK in
      0) $PY -P $J/datasets.py kang $MB/genotype_pool/singlet_modfix --out $OUT/datasets \
           --truth kang:$MB/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz \
           --donor $MB/genotype_pool/singlet_modfix/donor_assignments.tsv ;;
      1) $PY -P $J/datasets.py hgmm1k_v3 $PB/runs/hgmm1k_v3/singlet-modfix-t8 --out $OUT/datasets \
           --truth cellranger:$PB/data/hgmm1k_v3/cellranger_hgmm_1k_v3_filtered_feature_bc_matrix.h5 \
           --truth cellbender:$MB/ambient/cellbender/cb_cell_barcodes.csv ;;
      2) $PY -P $J/datasets.py pbmc1k_v3 $PB/runs/pbmc1k_v3/singlet-modfix-t8 --out $OUT/datasets \
           --truth cellranger:$PB/data/pbmc1k_v3/cellranger_pbmc_1k_v3_filtered_feature_bc_matrix.h5 ;;
      3) $PY -P $J/datasets.py neuron1k_v3 $PB/runs/neuron1k_v3/singlet-modfix-t8 --out $OUT/datasets \
           --truth cellranger:$PB/data/neuron1k_v3/cellranger_neuron_1k_v3_filtered_feature_bc_matrix.h5 ;;
      4) tar -xzf $PB/data/nuclei2k_v2/cellranger_nuclei_2k_filtered_gene_bc_matrices.tar.gz -C $SCRATCH
         BC=$(find $SCRATCH -name barcodes.tsv | head -1)
         $PY -P $J/datasets.py nuclei2k_v2 $PB/runs/nuclei2k_v2/singlet-modfix-t8 --out $OUT/datasets --truth cellranger:$BC ;;
      5) $PY -P $J/datasets.py pbmc5p_v1 $PB/runs/pbmc5p_v1/singlet-modfix-t8 --out $OUT/datasets \
           --truth cellranger:$PB/data/pbmc5p_v1/cellranger_vdj_v1_hs_pbmc2_5gex_protein_filtered_feature_bc_matrix.h5 ;;
    esac ;;
  revalidate)
    RES=$W/state/called_ed/results
    GSE=$(ls -d $RES/GSE* | sed -n "$((TASK+1))p" | xargs -r basename)
    [ -n "$GSE" ] || { echo "no study $TASK"; exit 0; }
    $PY -P $J/revalidate.py score $RES/$GSE --data $W/data/emptydrop/$GSE --truth-cache $W/data/called_ed/truth --out $OUT/revalidate/$GSE ;;
  aggregate)
    $PY -P $J/revalidate.py aggregate $OUT/revalidate --out $OUT/report ;;
  kang_lower)
    $PY -P $J/kang_lower.py $MB/genotype_pool/singlet_modfix $MB/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz       $OUT/kang_genotype $OUT/kang_lower.json ;;
esac
# kang_lower: sbatch --export=ALL,MODE=kang_lower run.sbatch (after kang_genotype.sbatch)
