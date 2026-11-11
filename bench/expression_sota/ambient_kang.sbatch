#!/bin/bash
#SBATCH --job-name=amb-kang
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Kang b2 ctrl ambient: pin (d9834b8a) Singlet run vs SoupX / DecontX / CellBender (tools already run on the same raw matrix in bench-public/work)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; H=$W/jobs/bp/harness; X=$W/expression-sota; O=$X/ambient; mkdir -p $O
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
python -I $H/kang_ambient_artifacts.py kang2018_ctrl $W/pipeline-benchmark/runs/kang2018_ctrl/singlet-public $BP/work/kang2018_ctrl $O \
  $W/modality-bench/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz stim=ctrl
python -I $X/code/ambient_kang_eval.py $O/ambient-rna/kang2018_ctrl_tissue.tsv.gz $O/kang_ctrl_pin.json
