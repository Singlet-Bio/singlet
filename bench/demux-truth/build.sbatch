#!/bin/bash
#SBATCH --job-name=dmx2-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-10
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One array task = one mixture (row of mixtures.tsv): build it with planted doublets (read keep 0.6-0.8) and export the
# mixed AD/DP counts to $J/private/<id> (per-SNP counts: private, never leaves Clipper). Public side: results/<id>/.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
CAP=${CAP:-2000}
row=$(grep -v '^#' $J/repo/bench/demux-truth/mixtures.tsv | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
IFS=$'\t' read -r id donors rate mseed <<< "$row"
OUT=$J/results/$id; rm -rf $OUT $J/private/$id; mkdir -p $OUT $J/private/$id
echo "mix=$id donors=$donors rate=$rate $(date -Is)"
$J/bin/demux_mix --samples $J/samples.tsv --donors $donors --rate $rate --cap $CAP --mix-seed $mseed \
  --keep-lo 0.6 --keep-hi 0.8 --out $OUT --export $J/private/$id 2>&1 | grep -E '^\[mix\]|rror|mismatch'
cat $OUT/mix.json; ls -la $J/private/$id
