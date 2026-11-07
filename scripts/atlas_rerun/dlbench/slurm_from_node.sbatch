#!/bin/bash
#SBATCH --job-name=s9-slurmcheck
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: may a compute-node job call sbatch / squeue / scancel, and is scrontab available?
echo "node $(hostname)"; which sbatch squeue scancel scrontab sinfo
echo "-- sbatch --test-only from a compute node"
sbatch --test-only --account=debruinz --partition=short-cpu --time=00:10:00 --wrap="true" 2>&1
echo "-- real tiny submit (dependency never satisfied -> cancelled right away)"
id=$(sbatch --parsable --account=debruinz --partition=short-cpu --time=00:05:00 --hold --job-name=s9-selftest --wrap="true" 2>&1); echo "submitted: $id"
squeue -j ${id%%;*} -o "%i %T %r" 2>&1; scancel ${id%%;*} 2>&1; echo "scancel rc=$?"
echo "-- squeue by user"; squeue -u $USER -h | wc -l
echo "-- scrontab"; scrontab -l 2>&1 | head -5
echo "-- limits"; sacctmgr -nP show assoc user=$USER format=maxsubmitjobs,maxjobs,grptres 2>&1 | head -3; scontrol show config 2>/dev/null | grep -E "MaxJobCount|MaxArraySize"
