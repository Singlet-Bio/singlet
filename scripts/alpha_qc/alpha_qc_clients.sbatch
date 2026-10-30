#!/bin/bash
# Research Alpha QC check 7: Python client (and R client, if it can be installed) end to end from data.singlet.bio.
#   sbatch --export=ALL,QC_COMMIT=<sha>,GSES="GSE.. GSE.." alpha_qc_clients.sbatch
#SBATCH --job-name=aqc-clients
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
QCB=/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/build-${QC_COMMIT:0:10}
( flock -w 3600 9; [ -e "$QCB/.git" ] || { git clone -q --no-checkout git@github.com:Singlet-Bio/singlet.git "$QCB" && git -C "$QCB" fetch -q origin claude/alpha-qc && git -C "$QCB" checkout -q --detach "$QC_COMMIT"; } ) 9>/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/clone.lock
source "$QCB/scripts/alpha_qc/qc_common.sh"
qc_build
SCR=/tmp/$USER-aqcc-$SLURM_JOB_ID; mkdir -p "$SCR"; trap 'rm -rf "$SCR"' EXIT
cd "$SCR"
echo "=== Python client"
rt env SINGLET_S1_CACHE="$SCR/pycache" $PY "$SRC/scripts/alpha_qc/qc_clients.py" --out "$QC/out/clients_py.json" --scratch "$SCR/py" ${GSES:?}
echo "=== R client"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load r/4.5.3 2>/dev/null || module load r 2>/dev/null || { echo "no R module"; exit 0; }
export R_LIBS_USER=$QC/Rlib; mkdir -p "$R_LIBS_USER"
MV=$SCR/Makevars; printf 'CPPFLAGS += -I%s/include\nLDFLAGS += -L%s/lib\n' "$ZSTD" "$ZSTD" > "$MV"; export R_MAKEVARS_USER=$MV
Rscript -e 'r <- "https://cloud.r-project.org"; need <- c("Rcpp","jsonlite","curl","BiocManager","Matrix"); miss <- need[!vapply(need, requireNamespace, TRUE, quietly=TRUE)]; if (length(miss)) install.packages(miss, repos=r, Ncpus=16, lib=Sys.getenv("R_LIBS_USER")); if (!requireNamespace("SingleCellExperiment", quietly=TRUE)) BiocManager::install("SingleCellExperiment", update=FALSE, ask=FALSE, Ncpus=16, lib=Sys.getenv("R_LIBS_USER"))' > "$QC/out/r_install.log" 2>&1 || { echo "R dependency install failed (see r_install.log)"; tail -20 "$QC/out/r_install.log"; exit 0; }
cp -r "$SRC/r" "$SCR/rpkg"
R CMD INSTALL --no-test-load -l "$R_LIBS_USER" "$SCR/rpkg" > "$QC/out/r_pkg_install.log" 2>&1 || { echo "R package install failed"; tail -30 "$QC/out/r_pkg_install.log"; exit 0; }
SINGLET_CACHE_DIR=$SCR/rcache Rscript "$SRC/scripts/alpha_qc/qc_clients.R" "$QC/out/clients_r.json" ${GSES}
