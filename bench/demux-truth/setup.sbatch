#!/bin/bash
#SBATCH --job-name=dmx2-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Clone branch bench/demux-truth2, build demux_mix against the engine's October donor_demux.h, install the ceiling
# tools (Vireo in a pip venv, souporcell and troublet from the biocontainers image) and write samples.tsv.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
BR=${BRANCH:-bench/demux-truth2}
mkdir -p $J/bin $J/results $J/private
rm -rf $J/repo
git clone -q --depth 1 --branch "$BR" git@github.com:Singlet-Bio/singlet.git $J/repo \
  || git clone -q --depth 1 --branch "$BR" https://github.com/Singlet-Bio/singlet.git $J/repo || exit 1
git -C $J/repo log --oneline -1
export PATH=$W/envs/pbench/bin:$PATH
H=$J/repo/include; S=$J/repo/bench/demux-truth
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$H $S/demux_mix.cpp -o $J/bin/demux_mix \
  || x86_64-conda-linux-gnu-g++ -O2 -std=c++20 -pthread -I$H $S/demux_mix.cpp -o $J/bin/demux_mix || exit 1
ls -l $J/bin/demux_mix

# Vireo: pip venv (falls back to a pip --target directory on the pinned base python)
V=$W/venvs/demux-ceiling; BASE=$W/envs/gpool/venv/bin/python
if [ ! -x $V/bin/python ]; then
  $BASE -m venv $V && $V/bin/python -m pip install -q --upgrade pip \
    && $V/bin/python -m pip install -q vireoSNP numpy scipy pandas scikit-learn threadpoolctl \
    || { rm -rf $V; mkdir -p $V; $BASE -m pip install -q --target $V/site vireoSNP threadpoolctl; echo "fallback: pip --target"; }
fi
if [ -x $V/bin/python ]; then PYC=$V/bin/python; else PYC="env PYTHONPATH=$V/site $BASE"; fi
$PYC -c "import vireoSNP, numpy, scipy, sklearn; print('vireoSNP', vireoSNP.__version__, 'numpy', numpy.__version__)" \
  || echo "WARNING: vireoSNP import failed"
echo "$PYC" > $J/python_ceiling.txt

# souporcell clustering and troublet (MIT): released biocontainers build, cached under envs/souporcell
E=$W/envs/souporcell; mkdir -p $E
export APPTAINER_CACHEDIR=$W/cache/apptainer TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR $APPTAINER_CACHEDIR
[ -s $E/souporcell.sif ] || apptainer pull $E/souporcell.sif docker://quay.io/biocontainers/souporcell:2.5--hc1c3326_0
apptainer exec $E/souporcell.sif bash -c "which souporcell troublet" || echo "WARNING: souporcell or troublet missing in image"
apptainer exec $E/souporcell.sif souporcell --help 2>&1 | head -30
apptainer exec $E/souporcell.sif troublet --help 2>&1 | head -30
rm -rf $TMPDIR

# Single-donor runs. private/ holds per-SNP counts and stays on Clipper.
R=$J/runs; F=$W/jobs/fq-read-source/runs
{ for pair in S1:GSM7873657 S2:GSM7873658 S5:GSM5102900 S6:GSM5102901; do
    id=${pair%%:*}; g=${pair##*:}
    printf '%s\t%s\t%s\n' $id $R/$g/private $R/$g/out/cell_calls.tsv
  done
  printf 'S3\t%s\t%s\n' $F/GSM7901264/camp_none/private $F/GSM7901264/camp_none/out/cell_calls.tsv
  P=$W/pipeline-benchmark/runs/pbmc1k_v2/singlet-public
  printf 'S4\t%s\t%s\n' $P/private $P/cell_calls.tsv; } > $J/samples.tsv
cut -f1,2 $J/samples.tsv
