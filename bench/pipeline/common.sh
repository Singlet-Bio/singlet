# Shared paths for the pipeline benchmark jobs on Clipper (sourced by the sbatch scripts).
WORK=/mnt/projects/debruinz_project/debruinz/singlet
PB=$WORK/pipeline-benchmark            # data, references, outputs of this study
ENV=$WORK/envs/pbench                  # conda env: STAR, samtools, salmon/alevin-fry, python
REPO=$WORK/repos/singlet-py.git        # bare clone of Singlet-Bio/singlet
SRC_NEW=$WORK/src/singlet-py/fix__molecule-labels   # this branch (fixed engine)
SRC_OLD=$WORK/src/singlet-py/pb-baseline             # b1335a8 (main before the fix)
BIN_NEW=$PB/bin/singlet-fixed
BIN_OLD=$PB/bin/singlet-baseline
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs
export PATH=$ENV/bin:$PATH
mkdir -p $PB/{refs,data,runs,results,bin}
