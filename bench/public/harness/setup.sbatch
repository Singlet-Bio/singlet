#!/bin/bash
#SBATCH --job-name=bpub-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One-off setup for the public benchmark harness. Safe to re-run.
#  1. fetch THIS branch only into the shared bare clone and (re)point its worktree;
#  2. build Singlet's Python readers in place inside the runtime (the compute modules read bundles);
#  3. whitelists: Singlet's bundled lists plus 10x's GEM-X 3' v4 list;
#  4. R library with cellXY (GitHub, not Bioconductor) on top of the ambient env's R 4.4;
#  5. somalier 0.3.6 binary, its 1000 Genomes labelled background and hg38 sites; env bpub (bcftools, peddy);
#  6. record the Singlet binary under test (path, sha256, source commit) in $BP/singlet_binary.json.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BR=claude/bench-public-cohort-485cf7; BARE=$WORK/repos/singlet-py.git; SRC=$WORK/src/singlet-py/${BR//\//__}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; git -C $SRC log --oneline -1
source $SRC/bench/public/harness/common.sh
# the pinned venv has no setuptools; a side --target dir provides it for the in-place build of Singlet's readers
BLD=$WORK/venvs/bpub-build; [ -d $BLD/setuptools ] || rt python -m pip install -q --target $BLD setuptools
PYTHONPATH=$BLD rt bash -c "cd $SRC && python setup.py build_ext --inplace -j 8 > $BP/work/build_ext.log 2>&1" && echo readers built

mkdir -p $BP/whitelists && cp --update=none $SRC/python/singlet/data/whitelists/*.txt $BP/whitelists/
# GEM-X 3' v4 permit list (Cell Ranger's 3M-3pgex-may-2023) from simpleaf's chemistry registry, checked by its sha256
[ -s $BP/whitelists/3M-3pgex-may-2023.txt ] || curl -sSL --retry 5 -o $BP/whitelists/3M-3pgex-may-2023.txt \
  https://umd.box.com/shared/static/4a2q2lxngogaq4vkniptpu2gts5a38fb.txt
echo "0bfa4a0bea1d636e7ec1908aabb94307c53778185c12fc164da32dd085848131  $BP/whitelists/3M-3pgex-may-2023.txt" | sha256sum -c - \
  || echo "WARNING: v4 permit list checksum differs from simpleaf's registry"
wc -l $BP/whitelists/3M-3pgex-may-2023.txt

RLIB=$WORK/envs/rlib-benchpublic; mkdir -p $RLIB
PATH=$WORK/envs/ambient/bin:$PATH R_LIBS=$RLIB $WORK/envs/ambient/bin/Rscript -e '
  r <- "https://cloud.r-project.org"; lib <- Sys.getenv("R_LIBS")
  for (p in c("remotes", "BiocManager", "caret")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, repos = r)
  for (p in c("AnnotationDbi", "org.Hs.eg.db", "org.Mm.eg.db")) if (!requireNamespace(p, quietly = TRUE))
    BiocManager::install(p, lib = lib, update = FALSE, ask = FALSE)
  if (!requireNamespace("cellXY", quietly = TRUE)) remotes::install_github("phipsonlab/cellXY", lib = lib, upgrade = "never")
  cat("cellXY", as.character(packageVersion("cellXY")), "\n")'

T=$WORK/tools/bpub; mkdir -p $T/somalier
[ -x $T/somalier/somalier ] || { curl -sSL -o $T/somalier/somalier https://github.com/brentp/somalier/releases/download/v0.3.6/somalier && chmod +x $T/somalier/somalier; }
[ -s $T/somalier/sites.hg38.vcf.gz ] || curl -sSL -o $T/somalier/sites.hg38.vcf.gz https://github.com/brentp/somalier/files/3412456/sites.hg38.vcf.gz
[ -s $T/somalier/ancestry-labels-1kg.tsv ] || curl -sSL -o $T/somalier/ancestry-labels-1kg.tsv https://raw.githubusercontent.com/brentp/somalier/master/scripts/ancestry-labels-1kg.tsv
[ -d $T/somalier/1kg-somalier ] || { curl -sSL -o $T/somalier/1kg.somalier.tar.gz "https://zenodo.org/record/3479773/files/1kg.somalier.tar.gz?download=1" && tar -xzf $T/somalier/1kg.somalier.tar.gz -C $T/somalier && rm $T/somalier/1kg.somalier.tar.gz; }
$T/somalier/somalier 2>&1 | head -3 || true
ls $T/somalier | head
# bcftools / peddy / pysam for the ancestry and sex pipelines (pbench has no bcftools)
BPE=$WORK/envs/bpub
[ -x $BPE/bin/bcftools ] || MAMBA_ROOT_PREFIX=$WORK/tools/mm/root $WORK/tools/mm/bin/micromamba create -y -q -p $BPE   -c conda-forge -c bioconda --strict-channel-priority python=3.11 bcftools=1.21 samtools htslib peddy=0.4.8 pysam pandas pyarrow
$BPE/bin/bcftools --version | head -1; $BPE/bin/python -c "import peddy; print('peddy', peddy.__version__)"

python3 - <<EOF
import hashlib, json, subprocess
b = "$SINGLET_BIN"
h = hashlib.sha256(open(b, "rb").read()).hexdigest()
c = subprocess.run(["git", "-C", "$SINGLET_SRC", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
br = subprocess.run(["git", "-C", "$SINGLET_SRC", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
v = subprocess.run([b, "--version"], capture_output=True, text=True)
json.dump({"binary": b, "binary_sha256": h, "commit": c, "branch": br, "version_output": (v.stdout or v.stderr).strip()[:200]},
          open("$BP/singlet_binary.json", "w"), indent=1)
print(open("$BP/singlet_binary.json").read())
EOF
