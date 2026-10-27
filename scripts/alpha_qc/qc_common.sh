# SPDX-License-Identifier: MIT
# Shared setup for the Research Alpha QC jobs (sourced by the .sbatch files). Read-only on R2: HTTPS reads from the public host,
# S3 list calls only. The code is a job-local clone of Singlet-Bio/singlet at QC_COMMIT, built once under $QC/build-<c10>.
WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}
QC=${QC:-$WORK/qc/alpha_qc}
QC_COMMIT=${QC_COMMIT:?set QC_COMMIT (commit of claude/alpha-qc)}
C10=${QC_COMMIT:0:10}
SRC=$QC/build-$C10
SIF=${SIF:-$WORK/images/python-3.12-bookworm.sif}
VENV=${VENV:-$WORK/venvs/v2tools}          # existing venv with numpy/scipy/anndata/zstandard/boto3/pybind11; used read-only
ZSTD=${ZSTD:-$WORK/cache/zstd-1.5.6-static}
mkdir -p "$QC" "$QC/out" "$WORK/logs"
export XDG_CACHE_HOME=$QC/cache PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }

qc_build() {
  (
    flock -w 3600 9 || exit 75
    [ -e "$SRC/.qcbuild.ok" ] && exit 0
    rt bash -c "cd '$SRC' && CPATH='$ZSTD/include' LIBRARY_PATH='$ZSTD/lib' CFLAGS=-pthread LDFLAGS=-pthread '$VENV/bin/python' setup.py -q build_ext --inplace -j 8" >/dev/null
    rt "$VENV/bin/python" -c "import sys; sys.path.insert(0, '$SRC/python'); from singlet import _pz, _pz2; print('[qc build] _pz and _pz2 import')"
    touch "$SRC/.qcbuild.ok"
  ) 9>"$QC/build.lock"
  [ -e "$SRC/.qcbuild.ok" ] || { echo "qc build failed"; exit 6; }
}
export QC_SRC=$SRC
PY="$VENV/bin/python -I"   # -I: isolated; scripts add $QC_SRC/python to sys.path themselves
