#!/bin/bash
#SBATCH --job-name=singlet_cycle69b_robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69b_robust_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69b_robust_%j.log

set +e  # NEVER abort on failure

echo "=== cycle69b robust P1 ctest starting at $(date) ==="
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo ""
echo "=== cmake configure ==="
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

echo ""
echo "=== building P1 test targets (missing only) ==="
for t in de_wilcoxon_correctness de_ttest_correctness de_donor_pseudobulk_correctness \
         gsea_fgsea_correctness gsea_aucell_correctness \
         anno_marker_score_correctness anno_reference_map_correctness \
         integrate_harmony_correctness integrate_bbknn_correctness; do
    if [ -f "tests/$t" ]; then
        echo "  [skip] $t already built"
    else
        echo "  [build] $t ..."
        make -j8 $t 2>&1 | tail -5
        echo "  [build exit $?] $t"
    fi
done

echo ""
echo "=== running P1 test groups independently ==="

echo "=== 1/9 DE Wilcoxon ==="
ctest -R "WilcoxonTest" -V --timeout 120 2>&1 || true

echo ""
echo "=== 2/9 DE t-test ==="
ctest -R "TtestDE" -V --timeout 120 2>&1 || true

echo ""
echo "=== 3/9 DE Donor Pseudobulk ==="
ctest -R "DonorPseudo" -V --timeout 120 2>&1 || true

echo ""
echo "=== 4/9 GSEA fgsea ==="
ctest -R "^Fgsea\." -V --timeout 120 2>&1 || true

echo ""
echo "=== 5/9 GSEA AUCell ==="
ctest -R "^AUCell\." -V --timeout 120 2>&1 || true

echo ""
echo "=== 6/9 Anno marker ==="
ctest -R "MarkerScore" -V --timeout 120 2>&1 || true

echo ""
echo "=== 7/9 Anno reference ==="
ctest -R "ReferenceMap" -V --timeout 120 2>&1 || true

echo ""
echo "=== 8/9 Integration Harmony ==="
ctest -R "Harmony" -V --timeout 120 2>&1 || true

echo ""
echo "=== 9/9 Integration BBKNN ==="
ctest -R "Bbknn" -V --timeout 120 2>&1 || true

echo ""
echo "=== ALL DONE at $(date) ==="
