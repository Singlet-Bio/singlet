#!/bin/bash
#SBATCH --job-name=bs-prodcls
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,A=runs_r8,B=runs_prod prod_classify.sbatch   (~0.5 node-h)
# Gate 3: decode every .1pz of run A and run B (pz2mtx, built here from the engine headers) and compare them by
# feature/barcode name with prod_classify.py. Results: bam-stream/prod_classify/<SET>.json (decoded files stay in /tmp).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; J=$W/jobs/bam-stream
INC=$W/src/singlet-py/perf__bam-stream/include
SCR=/tmp/$USER-bsprod-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
# Tool-only copy of the headers: pad missing rownames instead of rejecting the file, so the junction
# matrices of runs made before 87c6550 (up to 999 unnamed empty rows) still decode.
mkdir -p $SCR/inc && cp -r $INC/singlet $SCR/inc/
sed -i 's/throw PZReadError("rowname count does not match the number of rows");/rownames.resize(hdr.m);/' $SCR/inc/singlet/pileup/pz_reader.h
INC=$SCR/inc
$ENV/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$INC -I$ENV/include $J/pz2mtx.cpp -L$ENV/lib -Wl,-rpath,$ENV/lib -lzstd -o $SCR/pz2mtx
O=$W/bam-stream/prod_classify; mkdir -p $O
for SET in ${SETS:-GSM4006843 GSM7901264}; do
  for side in A B; do
    R=$W/bam-stream/${!side}/$SET/t4_r1/out; D=$SCR/$SET/$side; mkdir -p $D
    (cd $R && find . -name "*.1pz" | sed 's#^\./##') | while read -r f; do
      $SCR/pz2mtx $R/$f $D/$(echo ${f%.1pz} | tr / _) || echo "decode failed: $side $f"
    done
  done
  in_runtime python -I $J/prod_classify.py $SCR/$SET/A $SCR/$SET/B $W/bam-stream/$A/$SET/t4_r1/out $W/bam-stream/$B/$SET/t4_r1/out $O/$SET.json
  rm -rf $SCR/$SET
done
