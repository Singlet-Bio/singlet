#!/bin/bash
#SBATCH --job-name=cpuperf-ab
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Paired same-node A/B of two binaries (cpu-perf noise between nodes is larger than a 2% change). Env: SET, A and B as <commitdir>:<binary>, REPS (default 2).
# Runs A,B,B,A,A,B,... in one allocation through profile.sbatch (PACK=1, T=4) and prints wall, user+sys CPU and a hash line per run.
# Outputs of every run are kept under jobs/cpu-perf/results/<TAG>-<jobid> for the byte-identity diff.
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/cpu-perf; S=${SET:?}; REPS=${REPS:-2}
echo "node=$(hostname) $(lscpu | grep "Model name" | sed "s/  */ /g") $(date -Is)"
run() { local tag=$1 spec=$2; SET=$S T=4 CM=${spec%%:*} BINV=${spec##*:} TAG=ab-$S-$tag SLURM_JOB_ID=${SLURM_JOB_ID}$tag bash $R/profile.sbatch > /dev/null 2>&1; d=$(ls -d $W/jobs/cpu-perf/results/ab-$S-$tag-* | tail -1); echo "$tag $spec wall=$(awk "{print \$3}" $d/pack_wall.txt) cpu=$(grep -aE "User time|System time" $d/time_1.txt | awk "{s+=\$NF} END {print s}") $(cat $d/stream_path.txt) dir=$d"; }
for i in $(seq 1 $REPS); do if [ $((i % 2)) = 1 ]; then run A$i "$A"; run B$i "$B"; else run B$i "$B"; run A$i "$A"; fi; done
