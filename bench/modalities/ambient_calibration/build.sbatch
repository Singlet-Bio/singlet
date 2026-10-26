#!/bin/bash
#SBATCH --job-name=af-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# PGO build of fix/ambient-calibration -> pipeline-benchmark/bin/singlet-ambfix (same recipe as jobs/campaign/build_campaign.sbatch),
# then the ambient C++ test, then an end-to-end run on hgmm1k_v3 whose ambient output is compared with the harness on its own counts.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=fix/ambient-calibration; SRC=$W/src/singlet-py/fix__ambient-calibration-build
git --git-dir=$BARE fetch -q --refmap= origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git --git-dir=$BARE worktree add -q --detach $SRC refs/heads/$BR
git -C $SRC checkout -q --detach refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
REF=/mnt/projects/debruinz_project/cellarium/reference
cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
export TMPDIR=$SCR CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT"
bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
  $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-ambfix.new && mv $PB/bin/singlet-ambfix.new $PB/bin/singlet-ambfix
sha256sum $PB/bin/singlet-ambfix
echo "[build] ambient test"
export CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
$CXX -std=c++20 -O2 -pthread -I$SCR/src/include -I$SCR/src/tests/cpp $SCR/src/tests/cpp/test_ambient_decontx.cpp -o $SCR/t_amb && (cd $SCR && ./t_amb 2>/dev/null | tail -3)
echo "[build] e2e hgmm1k_v3"
O=$W/ambfix/e2e/hgmm1k_v3; rm -rf $O; mkdir -p $O
G=$REF/GRCh38-GRCm39-barnyard
cp $PB/data/hgmm1k_v3/hgmm1k_v3.1fq $SCR/hgmm1k_v3.1fq
cd $SCR && /usr/bin/time -v -o $O/time.txt $PB/bin/singlet-ambfix $SCR/hgmm1k_v3.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
  --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "\[ambient\]|\[cell_calling\] n_cells" $O/run.log
# the binary's ambient output must equal the harness (same header, default config) on the run's own stored counts
source $W/jobs/ambfix/common.sh
D=$O/dump; py $SRC/bench/modalities/ambient_calibration/dump_run.py $O $D
$CXX -std=c++20 -O2 -pthread -I$SRC/include $SRC/bench/modalities/ambient_calibration/harness.cpp -o $SCR/h
$SCR/h $D $D/v2 threads=16
py - <<EOF
import numpy as np, pandas as pd
P = pd.read_csv("$O/ambient_contamination.tsv", sep="\t")
h = pd.Series(np.fromfile("$D/v2.rho.f64"), index=pd.read_csv("$D/cells.txt", header=None)[0].astype(str))
d = np.abs(P.set_index("barcode").rho - h.reindex(P.barcode.astype(str)).values)
print("binary vs harness: cells", len(P), "max |rho diff|", float(d.max()), "(tsv rounding 1e-6)")
EOF
