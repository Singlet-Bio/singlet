#!/bin/bash
#SBATCH --job-name=rg-prep
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Put the barcode whitelists where the binary looks (<bin dir>/../whitelists), plus the shared job helpers.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate
mkdir -p $RG/bin/whitelists $RG/jobs
rsync -aL --exclude=whitelists $PB/whitelists/ $RG/bin/whitelists/
rsync -aL --exclude=whitelists $W/bpub/whitelists/ $RG/bin/whitelists/
ls $RG/bin/whitelists | wc -l; du -shL $RG/bin/whitelists
