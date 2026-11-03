#!/bin/bash
#SBATCH --job-name=s15-toolver
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S15 benchmark store: tool versions as installed (read by pilot_tools.sh's reg()). Output: $STORE/tool_versions.tsv  (tool<TAB>version string)
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/benchmark-store; mkdir -p $S; command -v apptainer > /dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2> /dev/null; module load apptainer 2> /dev/null; }
v() { printf "%s\t%s\n" "$1" "$(shift; "$@" 2>&1 | head -2 | tr '\n' ' ' | cut -c1-160)"; }
{ v rtools env PATH=$W/envs/ambient/bin:$PATH Rscript -e 'cat("SoupX", as.character(packageVersion("SoupX")), "celda", as.character(packageVersion("celda")), "scDblFinder", as.character(packageVersion("scDblFinder")), "R", R.version.string)'
  v edrops apptainer exec --bind $W $W/images/bioc-3.20.sif Rscript -e 'cat("DropletUtils", as.character(packageVersion("DropletUtils")), R.version.string)'
  v cellbender apptainer exec $W/images/cellbender-0.3.2.sif cellbender --version
  v mgatk env PATH=$W/envs/mtl/bin:$PATH PYTHONPATH=$W/envs/mgatk_extra mgatk --version
  v kb env PATH=$W/envs/adt/bin:$PATH kb --version
  v alevin env PATH=$W/envs/adt-afry/bin:$PATH simpleaf --version
  v vireo $W/envs/gpool/venv/bin/vireo --version
  v souporcell apptainer exec $W/envs/souporcell/souporcell.sif souporcell_pipeline.py --help
  v trust4 env PATH=$W/envs/vdjcrispr/bin:$PATH run-trust4 --version
  v STARsolo $W/pipeline-benchmark/bin/STAR-velodump --version; } > $S/tool_versions.tsv; cat $S/tool_versions.tsv
