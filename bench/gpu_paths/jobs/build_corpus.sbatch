#!/bin/bash
#SBATCH --job-name=gpup-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Code-path corpus (bench/gpu_paths/README.md): builds the small .1fq inputs listed in corpus/manifest.tsv from data already on Clipper.
# Nothing is downloaded. Inputs marked synthetic have seeded low qualities (corpus/mkinput.py) and are labeled so in the manifest and in manifest_built.tsv.
# Output: $W/jobs/gpu-paths/corpus/<id>/<id>.1fq, <id>/stats.json, <id>/encode.log and corpus/manifest_built.tsv.
# Parameters: ONLY=id1,id2 (default all), EXPECT_SCRIPTS=<commit prefix of bench/gpu-paths>.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export W
WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
ENCBIN=${ENCBIN:-$W/jobs/s15/bin/0ea48783/singlet-cpu}
CORPUS=$W/jobs/gpu-paths/corpus; mkdir -p $CORPUS
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-gpupb-$SLURM_JOB_ID; mkdir -p $SCR/repo; SREF=refs/gpup/build-$SLURM_JOB_ID; BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-paths}
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$SBR | cut -f1); [ -n "$WANT" ] || { echo "no scripts branch $SBR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
SFULL=$(git -C $BARE rev-parse $SREF); [ "$SFULL" = "$WANT" ] || { echo "fetched $SFULL, branch tip is $WANT"; exit 3; }
[ -z "${EXPECT_SCRIPTS:-}" ] || [[ $SFULL == $EXPECT_SCRIPTS* ]] || { echo "scripts commit $SFULL, EXPECT_SCRIPTS is $EXPECT_SCRIPTS"; exit 3; }
git -C $BARE archive $SREF bench/gpu_paths | tar -x -C $SCR/repo
GP=$SCR/repo/bench/gpu_paths
echo "BUILD job=$SLURM_JOB_ID node=$(hostname -s) scripts=${SFULL:0:8} encoder=$ENCBIN $(date -Is)"
printf "id\tkind\ttarget\tpairs\tbytes\tqual_mode\tprotocol_id\tstream_lengths\tsynthetic_stats\tsha256_16\tbuild_s\n" > $CORPUS/manifest_built.tsv.new
for id in $(python3 $GP/check_paths.py ids "${ONLY:-}"); do
  T0=$(date +%s); D=$CORPUS/$id; mkdir -p $D; S=$SCR/$id; mkdir -p $S
  gp() { python3 $GP/check_paths.py get $id "$1"; }
  N=$(gp n); R1=$(gp r1); R2=$(gp r2); R1C=$(gp r1cut); R2C=$(gp r2cut); SY=$(gp synth); Q=$(gp qual); PR=$(gp protocol); KIND=$(gp kind)
  ARGS=(--n $N --out1 $S/R1.fastq --out2 $S/R2.fastq --stats $D/stats.json --seed 20261105)
  [ "$R1C" != 0 ] && ARGS+=(--r1cut $R1C); [ "$R2C" != 0 ] && ARGS+=(--r2cut $R2C); [ "$SY" != 0 ] && ARGS+=(--synth $SY)
  case $R1 in
    sra:*) SRA=${R1#sra:}; mkdir -p $S/sra
           fastq-dump -X $N --split-files -O $S/sra $SRA > $D/fastq-dump.log 2>&1
           python3 $GP/corpus/mkinput.py --sra-pair $S/sra/*_2.fastq $S/sra/*_3.fastq --wl $WL/3M-february-2018.txt "${ARGS[@]}" > $D/mkinput.log 2>&1 ;;
    *)     python3 $GP/corpus/mkinput.py --r1 "$R1" --r2 "$R2" "${ARGS[@]}" > $D/mkinput.log 2>&1 ;;
  esac
  # a manifest path starting with @W@ is a path under $W
  EA=(--reads $S/R1.fastq $S/R2.fastq -o $D/$id.1fq --quality $Q --r2-maxlen 512 --no-dedup --no-trim --whitelist-dir $WL)
  [ -n "$PR" ] && EA+=(--protocol $PR)
  rm -f $D/$id.1fq
  $ENCBIN encode "${EA[@]}" > $D/encode.log 2>&1; rc=$?
  if [ $rc != 0 ] || [ ! -f $D/$id.1fq ]; then
    echo "!! $id: encode failed (exit $rc): $(tail -2 $D/encode.log | tr '\n' ' ' | cut -c1-200)"
    printf "%s\t%s\t%s\t%s\tENCODE-FAILED\t-\t-\t-\t-\t-\t%s\n" $id $KIND "$(gp target)" $N $(( $(date +%s) - T0 )) >> $CORPUS/manifest_built.tsv.new
    rm -rf $S; continue
  fi
  python3 - $D/$id.1fq $id $KIND "$(gp target)" $D/stats.json $(( $(date +%s) - T0 )) >> $CORPUS/manifest_built.tsv.new <<'PY'
import struct, sys, hashlib, os, json
p, i, kind, tgt, st, secs = sys.argv[1:7]
b = open(p, "rb").read(96)
sl = struct.unpack_from("<4H", b, 12); nu, no = struct.unpack_from("<QQ", b, 20)
h = hashlib.sha256(open(p, "rb").read(1 << 24)).hexdigest()[:16]
syn = ""
try:
    s = json.load(open(st)); syn = ("seed %s share %s lowered %s reads_ge128 %s" % (s["seed"], s["share"], s["lowered_random"], s["reads_with_lowered_ge128"])) if s.get("synthetic") else "real"
except Exception:
    syn = "?"
print("\t".join([i, kind, tgt, str(nu), str(os.path.getsize(p)), str(b[55]), str(b[7]), str(sl), syn, h, secs]))
PY
  echo "built $id: $(tail -1 $CORPUS/manifest_built.tsv.new | cut -f4,5,6,7,8) in $(( $(date +%s) - T0 )) s"
  rm -rf $S
done
mv $CORPUS/manifest_built.tsv.new $CORPUS/manifest_built.tsv
column -t -s $'\t' $CORPUS/manifest_built.tsv | cut -c1-230
echo "done $(date -Is)"
