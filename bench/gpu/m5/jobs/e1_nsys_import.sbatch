#!/bin/bash
#SBATCH --job-name=e1-nsysimp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1: import an nsys .qdstrm (written by /usr/local/bin/nsys in cuda-13-devel.sif, whose QdstrmImporter needs libdw.so.1,
# missing in the image) on the host (Ubuntu 24.04 has libdw), then nsys stats -> CSV. Usage: sbatch e1_nsys_import.sbatch DIR
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; PY=$W/images/python-3.12-bookworm.sif; D=$1
SCR=/tmp/$USER-e1i-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
apptainer exec -B $SCR $IMG cp -r /opt/nvidia/nsight-compute/2026.2.0/host $SCR/
H=$SCR/host/linux-desktop-glibc_2_11_3-x64; T=$SCR/host/target-linux-x64
echo "missing libs:"; ldd $H/QdstrmImporter | grep "not found"
cp $D/nsys_base.qdstrm $SCR/
$H/QdstrmImporter --help 2>&1 | head -12
$H/QdstrmImporter -i $SCR/nsys_base.qdstrm 2>&1 | tail -5
[ -f $SCR/nsys_base.nsys-rep ] || $H/QdstrmImporter --input-file $SCR/nsys_base.qdstrm 2>&1 | tail -5
ls -la $SCR | grep nsys_base
[ -f $SCR/nsys_base.nsys-rep ] || exit 1
cp $SCR/nsys_base.nsys-rep $D/
for NS in "$T/nsys" "$H/nsys" "apptainer exec -B $W -B $SCR $IMG /usr/local/bin/nsys"; do
  echo "== stats with $NS"
  $NS stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace -f csv -o $D/nsys_base $D/nsys_base.nsys-rep > $D/nsys_stats.log 2>&1
  X=$?; tail -3 $D/nsys_stats.log; [ $X -eq 0 ] && ls $D/nsys_base_*.csv >/dev/null 2>&1 && break
done
ls -la $D | grep nsys
BARE=$W/repos/singlet-py.git; mkdir -p $SCR/py
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/profile:refs/fable/profile && git -C $BARE archive refs/fable/profile bench/fable/profile | tar -x -C $SCR/py
apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/e1_summarize.py $D > $D/summary_time.txt 2>&1; tail -30 $D/summary_time.txt
