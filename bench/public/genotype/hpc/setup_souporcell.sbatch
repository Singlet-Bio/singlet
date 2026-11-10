#!/bin/bash
#SBATCH --job-name=bpg-soup-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# souporcell (Heaton 2020) as an apptainer image; also record the vireo / cellsnp-lite versions and vireo's K options.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/souporcell; mkdir -p $E
export APPTAINER_CACHEDIR=$W/cache/apptainer; export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR $APPTAINER_CACHEDIR
cd $E
[ -s souporcell.sif ] || apptainer pull souporcell.sif docker://quay.io/biocontainers/souporcell:2.5--hc1c3326_0
ls -la $E
export APPTAINER_BIND=/mnt/projects:/mnt/projects
apptainer exec souporcell.sif souporcell_pipeline.py -h 2>&1 | head -40
apptainer exec souporcell.sif bash -c "which souporcell_pipeline.py souporcell vartrix freebayes minimap2 samtools; souporcell --version"
echo --- vireo
G=$W/envs/gpool
$G/venv/bin/pip list 2>/dev/null | grep -i -E "vireo|numpy|pysam"
$G/venv/bin/vireo -h 2>&1 | head -80
apptainer exec $G/cellsnp-lite.sif cellsnp-lite -V 2>&1 | head -2
