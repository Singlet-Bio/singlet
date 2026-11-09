#!/bin/bash
#SBATCH --job-name=d101-rt3
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
IR=$W/src/singlet-agents/main/hpc/clipper/in_runtime.sh; [ -f $IR ] || IR=$(ls -d $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
mkdir -p $W/envs/reditools3
in_runtime bash -c "pip install -q --target $W/envs/reditools3 REDItools3==3.7 2>&1 | tail -3; PYTHONPATH=$W/envs/reditools3 python3 -m reditools analyze -h; PYTHONPATH=$W/envs/reditools3 python3 -c 'import reditools,pysam; print(pysam.__version__)'"
$ENV/bin/STAR --version; $ENV/bin/samtools --version | head -1
