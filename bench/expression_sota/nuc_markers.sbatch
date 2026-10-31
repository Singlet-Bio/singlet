#!/bin/bash
#SBATCH -J nuc-mark
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota; PY="$W/envs/pbench/bin/python3 -I"
$PY $X/code/nuc_markers.py $X/nucleus_hash/singlet $X/nuclei/markers_gaublomme.json 2>&1 | grep -v -i -E "warn|spmatrix|sparse|^ +(That|Unless|on 2D|See|https)|^$"
echo == nuclei2k_v2
$PY $X/code/nuc_markers.py $W/release-gate/runs/ec/92095e29-cpu/nuclei2k_v2 $X/nuclei/markers_nuclei2k.json 2>&1 | grep -v -i -E "warn|spmatrix|sparse|^ +(That|Unless|on 2D|See|https)|^$"
