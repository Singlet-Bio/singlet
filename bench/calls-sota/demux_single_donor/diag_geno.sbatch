#!/bin/bash
#SBATCH --job-name=donorfix_geno
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=190G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Genotype-distance statistics of the auto-K components (SINGLET_DEMUX_GENODBG=1) on single-donor and pooled libraries.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH; P=$W/envs/pbench
BRANCH=${BRANCH:-fix/donor-single-split}; TAG=${TAG:-geno}
O=$W/jobs/donorfix/$TAG; mkdir -p $O
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch $BRANCH https://github.com/Singlet-Bio/singlet.git $S/e || exit 1
(cd $S/e && git rev-parse --short HEAD)
x86_64-conda-linux-gnu-g++ -std=gnu++17 -O3 -DNDEBUG -march=x86-64-v3 -pthread -I$S/e/include -I$P/include $S/e/bench/modalities/demux_fix/demux_harness.cpp -o $S/h -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz || exit 1
export SINGLET_DEMUX_GENODBG=1
for g in ${LIBS:-GSM5130155 GSM4987144 GSM4987148 GSM5682562 GSM8085472 kang_b1A}; do
  mkdir -p $O/$g
  if [ $g = kang_b1A ]; then d=$W/bench/rescore-v135/kang_b1A/out
    CALLS_COL=is_cell $S/h $d/private/snp_ad.mtx.gz $d/private/snp_dp.mtx.gz $d/private/snp_ad_barcodes.tsv.gz $d/cell_calls.tsv $O/$g 16 auto -1 ${EXTRA:-} > $O/$g/log.txt 2>&1
  else $S/h --1pz $W/pilot/$g/s1cpu_0ea48783/out $O/$g 16 auto -1 ${EXTRA:-} > $O/$g/log.txt 2>&1; fi
  echo "== $g $(grep -E 'selected K' $O/$g/log.txt | sed -E 's/^\[ *[0-9.]+\] \[demux\] //')"
  grep -E 'genodbg|genotype distance' $O/$g/log.txt | sed -E 's/^\[ *[0-9.]+\] \[demux\] //' | sort -u | cut -c1-200
done
