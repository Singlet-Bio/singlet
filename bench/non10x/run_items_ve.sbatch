#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=vn-run
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# D-150 validation and bulk check. One array task = one line of $LIST (| separated):
#   NAME|ACCESSION (GSM, SRS, ERS, SRR or ERR)|SPOTS|GENOME dir name|SCORE kind (barnyard dge umifm bdcsv or -)|truth URL(s) (space separated)
# Fetches a prefix of the first run (never a full download), encodes it with AUTO protocol detection (the case that matters),
# runs the patched engine, scores against the authors' matrix (bench/non10x/score.py, bars in PASS_BARS.md) and appends one JSON line to $OUTROOT/results.jsonl.
# SPOTS=0: full depth, from $W/data/non10x/full/<ACC>/R_[12].fastq (dl_full.sbatch)
# env: BIN, SCORE (path of score.py) (patched singlet-gpu, used on CPU), LIST, OUTROOT, WLD (whitelist directory; default: the commit lists of
#      $W/jobs/validation-engine/non10x/whitelists, because pipeline-benchmark/whitelists holds a stale inDrop BC1 list).
# validation-engine variant (D-152): the list has one more field, FORCE, before the truth URLs:
#   NAME|ACCESSION|SPOTS|GENOME|SCORE kind|FORCE protocol tag (- = auto-detect)|truth URL(s)
# Score kinds added: barcodes (Drop-seq-style author matrix with barcode columns "LIB_BARCODE"; recall and precision of barcodes).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REFB=/mnt/projects/debruinz_project/cellarium/reference
: "${BIN:?}" "${LIST:?}" "${OUTROOT:?}"
export PATH=$W/envs/sratools/bin:$W/envs/pbench/bin:$PATH
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
IFS='|' read NAME ACC SPOTS GENOME KIND FORCE TRUTH <<< "$LINE"
WLD=${WLD:-$W/jobs/validation-engine/non10x/whitelists}
FARG=""; [ "$FORCE" != "-" ] && [ -n "$FORCE" ] && FARG="--protocol $FORCE"
D=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $D; trap 'rm -rf $D' EXIT
OUT=$OUTROOT/$NAME; rm -rf $OUT; mkdir -p $OUT; exec > >(tee $OUT/job.log) 2>&1
echo "=== $NAME $ACC spots=$SPOTS genome=$GENOME $(date -Is) bin=$(sha256sum $BIN | cut -c1-12)"
emit() { D2=$(printf %s "$2" | tr -d "'\"\\\\"); python3 -I -c "import json;print(json.dumps(dict(name='$NAME',acc='$ACC',outcome='$1',detail='$D2')))" >> $OUTROOT/results.jsonl; }
case $ACC in
  SRR*|ERR*) RUN=$ACC ;;
  GSM*) RUN=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=experiment_title=%22*$ACC*%22&fields=run_accession&format=tsv" | sed 1d | head -1 | cut -f1) ;;
  *) RUN=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$ACC&result=read_run&fields=run_accession&format=tsv" | sed 1d | head -1 | cut -f1) ;;
esac
echo "run=$RUN"; [ -n "$RUN" ] || { emit unresolved "no run for $ACC"; exit 0; }
if [ "$SPOTS" = 0 ]; then   # full depth: reads staged by dl_full.sbatch under $W/data/non10x/full/$ACC
  F1=$W/data/non10x/full/$ACC/R_1.fastq; F2=$W/data/non10x/full/$ACC/R_2.fastq
else
fastq-dump -X $SPOTS --split-files -O $D $RUN 2>&1 | tail -2
F1=$D/${RUN}_1.fastq; F2=$D/${RUN}_2.fastq
fi
if [ ! -s $F1 ] || [ ! -s $F2 ]; then
  # fall back to the ENA copy (first SPOTS reads)
  for i in 1 2; do U=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$RUN&result=read_run&fields=fastq_ftp&format=tsv" | sed 1d | cut -f1 | tr ';' '\n' | sed -n "${i}p")
    [ -n "$U" ] && curl -s "https://$U" | zcat 2>/dev/null | head -n $((SPOTS * 4)) > $D/${RUN}_$i.fastq; done
fi
[ -s $F1 ] && [ -s $F2 ] || { emit fetch_failed "no fastq pair for $RUN"; exit 0; }
echo "R1 length $(sed -n 2p $F1 | tr -d '\n' | wc -c), R2 length $(sed -n 2p $F2 | tr -d '\n' | wc -c)"
$BIN download --r1 $F1 --r2 $F2 --whitelist-dir $WLD $FARG -o $OUT/x.1fq --no-profile > $OUT/encode.log 2>&1; ERC=$?
grep -a -i -E "protocol|gate|REJECT|floor|whitelist" $OUT/encode.log | head -8
TAG=$(grep -a -o -i -E "protocol[^ ]*[:= ]+[A-Za-z0-9._-]+" $OUT/encode.log | head -1)
if [ $ERC -ne 0 ] || [ ! -s $OUT/x.1fq ]; then
  REASON=$(grep -a -E "REJECT|ERROR|floor" $OUT/encode.log | head -1 | cut -c1-200)
  emit "encode_abort_rc$ERC" "$TAG | $REASON"; exit 0
fi
G=$REFB/$GENOME
export SINGLET_WHITELIST_DIR=$WLD SINGLET_REF_BASE=$REFB SINGLET_STALL_TIMEOUT=3600 TMPDIR=$D
mkdir -p $OUT/res
$BIN $OUT/x.1fq --ref-base $REFB --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $OUT/res/ \
  --threads $SLURM_CPUS_PER_TASK --pipeline --output-format mtx --raw-matrix > $OUT/run.log 2>&1; RRC=$?
echo "engine rc=$RRC"
[ -s $OUT/res/summary.json ] || { emit "engine_failed_rc$RRC" "$TAG"; exit 0; }
STATUS=$(python3 -I -c "import json;d=json.load(open('$OUT/res/summary.json'));print(d.get('status'),d.get('protocol_name'),d.get('estimated_cells'),d.get('median_genes_per_cell'),d.get('mapping_rate'),'|'.join(d.get('warnings',[])))")
echo "summary: $STATUS"
SC=""
if [ "$KIND" != "-" ]; then
  TRD2=$W/data/validation-engine/non10x/truth; mkdir -p $TRD2; T=""
  for u in $TRUTH; do f=$W/data/non10x/truth/$(basename $u); [ -s $f ] || { f=$TRD2/$(basename $u); [ -s $f ] || curl -sL -o $f "$u"; }; T="$T $f"; done
  KB=${KIND%%:*}; KO=""; case $KIND in *:*) KO=${KIND#*:} ;; esac   # KIND may carry scorer options after a colon (score.py dgebc)
  SC=$(python3 -I ${SCORE:?} $KB $OUT/res ${KO:+"$KO"} $T 2>&1 | tail -1 | tr -d "'")
fi
echo "$SC"
python3 -I - <<PY >> $OUTROOT/results.jsonl
import json
print(json.dumps(dict(name='$NAME', acc='$ACC', run='$RUN', outcome='ran', tag='$TAG', summary='$STATUS', score='''$SC'''[6:] if '''$SC'''.startswith('SCORE ') else '''$SC''')))
PY
rm -rf $OUT/x.1fq
