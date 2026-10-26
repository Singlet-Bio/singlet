#!/bin/bash
#SBATCH --job-name=vdjc-env
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Env for the V(D)J and CRISPR modalities and their benchmarks: SRA/ENA download, the Singlet code (numba), TRUST4,
# scanpy for the knockdown checks. Also fetches this branch into its own worktree (never sync.sh) and builds the
# V(D)J segment references from the GENCODE genomes the pipeline uses.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=claude/vdj-crispr-benchmark-721d3a
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs HOME=$WORK/cache/condahome
mkdir -p $HOME $CONDA_PKGS_DIRS
source /etc/profile.d/lmod.sh; module load miniforge3
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
if [ -d $SRC ]; then git -C $SRC checkout -q --detach "$BRANCH"; git -C $SRC reset -q --hard "$BRANCH";
else git -C $WORK/repos/singlet-py.git worktree add --detach $SRC "$BRANCH"; fi
git -C $SRC log --oneline -1
if [ ! -f $ENV/.ok ]; then rm -rf $ENV; mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 numpy scipy pandas pyarrow anndata h5py scikit-learn numba matplotlib zstandard requests dnaio isa-l \
  sra-tools pigz trust4 scanpy pytest openpyxl && touch $ENV/.ok; fi
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
$ENV/bin/python -c "import openpyxl" 2>/dev/null || $ENV/bin/pip install -q openpyxl
python -c "import numba, dnaio, sklearn, scanpy; print('py ok', numba.__version__, dnaio.__version__)"
run-trust4 2>&1 | head -3 || true
# TRUST4's own references (IMGT-derived), for the comparison runs only
mkdir -p $WORK/data/vdjc/trust4
for f in human_IMGT+C.fa hg38_bcrtcr.fa mouse/GRCm38_bcrtcr.fa mouse/mouse_IMGT+C.fa; do
  [ -s $WORK/data/vdjc/trust4/$(basename $f) ] || curl -fsSL -o $WORK/data/vdjc/trust4/$(basename $f)     https://raw.githubusercontent.com/liulab-dfci/TRUST4/master/$f || echo "missing $f"; done
ls -la $WORK/data/vdjc/trust4
R=$WORK/data/references/junctions
mkdir -p $WORK/state/vdjc/ref
python -P -m singlet.vdj.reference $R/GRCh38-2024-A/GRCh38.primary_assembly.genome.fa.gz \
  $R/GRCh38-2024-A/gencode.v44.primary_assembly.annotation.gtf.gz $WORK/state/vdjc/ref/vdj_human.tsv.gz
python -P -m singlet.vdj.reference $R/GRCm39-2024-A/GRCm39.primary_assembly.genome.fa.gz \
  $R/GRCm39-2024-A/gencode.vM33.primary_assembly.annotation.gtf.gz $WORK/state/vdjc/ref/vdj_mouse.tsv.gz
cd $SRC && python -m pytest -q tests/python/test_vdj_assemble.py 2>&1 | tail -3
