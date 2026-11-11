#!/bin/bash
#SBATCH -J ah-cmp
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# vs the pin: cell libraries identical except ambient_contamination.json; nuclei: corrected == raw. SHA8 of the built commit.
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code; PY=$W/envs/pbench/bin/python3; N=$W/release-gate/runs/ec/${SHA8}-cpu
for d in pbmc1k_v3 hgmm1k_v3 nuclei2k_v2; do echo "== $d vs pin"; $PY -I $X/cmp_gate.py $N/$d $W/pipeline-benchmark/runs/$d/singlet-public 2>&1 | head -2 | cut -c1-500; done
for d in nuclei2k_v2; do echo "== $d corrected vs raw"; $PY -I $X/corr_vs_raw.py $N/$d 2>&1 | tail -1; grep -E 'skipped_reason|estimated_contamination|median_intronic|"method"' $N/$d/ambient_contamination.json; head -3 $N/$d/ambient_contamination.tsv; done
echo "== gaublomme corrected vs raw"; $PY -I $X/corr_vs_raw.py $W/expression-sota/nucleus_hash/singlet_ah 2>&1 | tail -1; grep -E 'skipped_reason|estimated_contamination|median_intronic' $W/expression-sota/nucleus_hash/singlet_ah/ambient_contamination.json
for d in pbmc1k_v3 hgmm1k_v3; do echo "== $d json"; grep -E 'estimated_contamination|skipped_reason|n_doublets|median_intronic' $N/$d/ambient_contamination.json; done
echo "== gpu vs cpu pbmc1k_v3"; $PY -I $X/cmp_gate.py $W/release-gate/runs/ec/${SHA8}-gpu/pbmc1k_v3 $N/pbmc1k_v3 2>&1 | head -1 | cut -c1-500
