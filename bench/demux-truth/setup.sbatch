#!/bin/bash
#SBATCH --job-name=dmx-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch branch bench/demux-truth, build demux_mix against the engine's donor_demux.h, write samples.tsv.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth
mkdir -p $J/bin $J/results
rm -rf $J/repo
git clone -q --depth 1 --branch "${BRANCH:-bench/demux-truth}" git@github.com:Singlet-Bio/singlet.git $J/repo \
  || git clone -q --depth 1 --branch "${BRANCH:-bench/demux-truth}" https://github.com/Singlet-Bio/singlet.git $J/repo
git -C $J/repo log --oneline -1
export PATH=$W/envs/pbench/bin:$PATH
H=$J/repo/include; S=$J/repo/bench/demux-truth
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$H $S/demux_mix.cpp -o $J/bin/demux_mix \
  || x86_64-conda-linux-gnu-g++ -O2 -std=c++20 -pthread -I$H $S/demux_mix.cpp -o $J/bin/demux_mix
# single-donor samples: 10x-3p-v2, same 1000G panel (7,352,548 SNPs). private/ holds the SNP matrices and stays on Clipper.
F=$W/jobs/fq-read-source/runs
printf 'S1\t%s\t%s\n' $F/GSM4006843/camp_none/private $F/GSM4006843/camp_none/out/cell_calls.tsv  > $J/samples.tsv
printf 'S2\t%s\t%s\n' $F/GSM7474576/camp_none/private $F/GSM7474576/camp_none/out/cell_calls.tsv >> $J/samples.tsv
printf 'S3\t%s\t%s\n' $F/GSM7901264/chunk_none_nmiss/private $F/GSM7901264/chunk_none_nmiss/out/cell_calls.tsv >> $J/samples.tsv
P=$W/pipeline-benchmark/runs/pbmc1k_v2/singlet-public
printf 'S4\t%s\t%s\n' $P/private $P/cell_calls.tsv >> $J/samples.tsv
ls -l $J/bin/demux_mix
cut -f1,2 $J/samples.tsv
