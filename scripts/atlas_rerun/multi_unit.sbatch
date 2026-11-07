#!/bin/bash
#SBATCH --job-name=s14-multiunit
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The Linux-only tests of the multi-sample worker (end-to-end drain with a fake engine needs GNU time) and the neighbouring launcher / queue tests. Env: BRANCH.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT; cd $T
git clone -q --depth 1 -b ${BRANCH:-fix/pinned-binaries} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1; cd repo && git log --oneline -1
python3 -m pip install -q --target $T/pk pytest > /dev/null 2>&1; PYTHONPATH=$T/pk python3 -m pytest -q -p no:cacheprovider tests/python/test_multi_drain.py tests/python/test_pinned_binary.py tests/python/test_atlas_rerun_orchestrator.py tests/python/test_organism_reference.py tests/python/test_atlas_rerun_memory.py tests/python/test_stage_pipeline.py 2>&1 | tail -15
