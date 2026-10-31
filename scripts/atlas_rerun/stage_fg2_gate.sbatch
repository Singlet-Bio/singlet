#!/bin/bash
#SBATCH --job-name=s9-fg2gate
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: gate of Fg2RunSet. Rebuilds singlet-stage + test_fg2_runset from the branch, regenerates the fixtures' .1fg2 in $WORK/jobs/s9/fixtures from their .1fq (same
# order), then per fixture: test_fg2_runset --ordered --threads 4 (decoded blocks equal the .1fq RunSet's, read for read).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; F=$W/jobs/s9/fixtures
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
B=/tmp/$USER-fg2gate-$SLURM_JOB_ID; mkdir -p $B; trap 'rm -rf $B' EXIT
cd $B && git clone -q --depth 1 -b ${BRANCH:-claude/stage-1fg2} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1
cd $B/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
CXXF="-O2 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI"
g++ $CXXF src/stage/singlet_stage.cpp -o $F/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
g++ $CXXF src/stage/test_fg2_runset.cpp -o $B/test_fg2_runset -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
RC=0
for R in SRR19421372 SRR19573735 SRR30477995; do
  echo "== $R"
  $F/singlet-stage --whitelist-dir $SINGLET_WHITELIST_DIR --accessions $R --scratch $B --verify --json $F/$R.stage.json -o $F/$R.1fg2 $F/$R.1fq | tail -c 160
  $B/test_fg2_runset --whitelist-dir $SINGLET_WHITELIST_DIR --fg2 $F/$R.1fg2 --ordered --threads 4 $F/$R.1fq || RC=1
done
echo "== gate rc=$RC"
# the combined path (metadata through the writer's sink): singlet download --stage-out vs the .1fq of the same run, unordered (SRA order vs barcode-sorted)
if [ -x $W/jobs/s9/bin/singlet-cpu ]; then
  R=SRR19421372
  python3 -c "
import sys
sys.path.insert(0, '$W/jobs/s9/src')
import fetch_worker as F
F.fetch_run('$R', '$B', 16, lambda m: None)
"
  $W/jobs/s9/bin/singlet-cpu download $B/$R.sra --quality binned --vdb-threads 2 --no-profile --whitelist-dir $SINGLET_WHITELIST_DIR -o $B/a.1fq > /dev/null 2>&1
  $W/jobs/s9/bin/singlet-cpu download $B/$R.sra --quality binned --vdb-threads 2 --no-profile --whitelist-dir $SINGLET_WHITELIST_DIR --stage-out $B/c.1fg2 -o $B/unused.1fq > /dev/null 2>&1
  echo "== combined path vs its .1fq"
  $B/test_fg2_runset --whitelist-dir $SINGLET_WHITELIST_DIR --fg2 $B/c.1fg2 --threads 4 $B/a.1fq || RC=1
fi
echo "== gate rc=$RC (final)"
