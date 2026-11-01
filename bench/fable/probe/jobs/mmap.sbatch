#!/bin/bash
#SBATCH --job-name=e14-mmap
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E14 (docs/gpu-engine/fable/E14-probe.md, section 5): the file-backed and shmem mappings of the genome vs a private THP copy
# (bench/fable/probe/mmap_e14.py). Nothing is left behind: the /dev/shm file is unlinked once mapped, no SysV segment is made.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/${GENOME:-GRCh38-2024-A}/star_2.7.11b; BARE=$W/repos/singlet-py.git
RES=$W/jobs/fable/probe/results/mmap-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e14x-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'rm -rf $SCR /dev/shm/e14-mmap-*' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/probe:refs/fable/probe-src
git -C $BARE archive refs/fable/probe-src bench/fable/probe/mmap_e14.py | tar -x -C $SCR/repo
{ echo "node=$(hostname) job=$SLURM_JOB_ID genome=$G $(date -Is)"; df -hT $G | tail -1; uname -r
  cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shmem_enabled
  t0=$(date +%s.%N); cat $G/Genome $G/SA $G/SAindex > /dev/null; echo "index into the page cache: $(echo "$(date +%s.%N) - $t0" | bc) s"
  $W/envs/pbench/bin/python3 $SCR/repo/bench/fable/probe/mmap_e14.py $G ${NRAND:-20000000}
  echo "ipcs -m: $(ipcs -m | grep -c '^0x') segments; /dev/shm: $(ls /dev/shm | grep -c e14-mmap) e14 files; done $(date -Is)"; } 2>&1 | tee $RES/mmap.txt
