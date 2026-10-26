#!/bin/bash
#SBATCH --job-name=bp-versions
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Tool versions for the ambient-rna / doublets manifests (competitor environments used by the public benchmark).
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
echo "== envs/ambient python"; python -I - <<'EOF'
import importlib
for m in ["scanpy","anndata","scrublet","numpy","scipy","pandas","sklearn","igraph","leidenalg","louvain","h5py"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
EOF
echo "== R"; Rscript -e 'for (p in c("SoupX","celda","scDblFinder","SingleCellExperiment","Seurat","igraph","jsonlite","scran","bluster")) cat(p, tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING"), "\n"); cat(R.version.string, "\n")'
echo "== gpool"; ls $W/envs/gpool; $W/envs/gpool/venv/bin/pip list 2>/dev/null | grep -iE "vireo|scrublet|scikit|numpy|pandas|scipy|cellsnp|souporcell|h5py"
$W/envs/gpool/venv/bin/vireo --version 2>&1 | head -2
export APPTAINER_BIND=/mnt/projects:/mnt/projects
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
apptainer exec $W/envs/gpool/cellsnp-lite.sif cellsnp-lite --version 2>&1 | head -2
ls -la $W/images/
apptainer exec $W/images/cellbender-0.3.2.sif cellbender --version 2>&1 | tail -2
ls $W/envs/pbench/bin | grep -E "^(python|samtools|STAR)$"; $W/envs/pbench/bin/samtools --version | head -1
nproc; lscpu | sed -n 's/^Model name: *//p'
