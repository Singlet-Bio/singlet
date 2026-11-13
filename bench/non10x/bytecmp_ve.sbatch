#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --job-name=vn-bytecmp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Two KEEPOUT=1 gates' kept outputs of one mode, file by file: which files are byte-identical, which differ, which exist on one side
# only, then the field check (fieldcheck.py) of each pair. A=<gate id> B=<gate id> [MODE=gpu]. Expected differences are named, never implied:
#   NEWOK=<regex of relative paths expected only in B>   OLDOK=<regex of relative paths expected only in A>
#   XALLOW="<file>:<json pointer> ..."  extra fieldcheck allowances for this comparison only (printed in the output)
# Exits 1 on a file on one side that neither regex names, or on any field difference outside the allowlist and XALLOW.
# Writes <B results>/bytecmp-vs-<A>.txt.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; G=$W/jobs/gpu-campaign; RA=${RA:-$W/jobs/gpu-pr2/results}; R=$W/results/validation-engine/non10x   # A (the v1.3.8 gate) under RA, B (this build) under R
A=${A:?}; B=${B:?}; MODE=${MODE:-gpu}
F=$R/gate-$B/bytecmp-vs-$A.txt; : > $F; ns=0; bad=0
NEWOK=${NEWOK:-<none>}; OLDOK=${OLDOK:-<none>}
read -ra XA <<< "${XALLOW:-}"; AL=(); for a in ${XA[@]+"${XA[@]}"}; do AL+=(--allow "$a"); done
echo "expected only in B: $NEWOK" >> $F; echo "expected only in A: $OLDOK" >> $F
for d in $R/gate-$B/*/; do
  s=$(basename $d); PA=$RA/gate-$A/$s/$MODE/out; PB=${d%/}/$MODE/out
  [ -d $PA ] && [ -d $PB ] || continue
  same=0; diff=0; only=0
  echo "=== $s $MODE: gate $A vs gate $B" >> $F
  while read -r f; do
    f=${f#./}
    if [ ! -f "$PB/$f" ]; then echo "  ONLY-A $f" >> $F; only=$((only+1)); echo "$f" | grep -qxE "$OLDOK" || { echo "  UNEXPLAINED $s only in A: $f" >> $F; bad=$((bad+1)); }
    elif [ ! -f "$PA/$f" ]; then echo "  ONLY-B $f" >> $F; only=$((only+1)); echo "$f" | grep -qxE "$NEWOK" || { echo "  UNEXPLAINED $s only in B: $f" >> $F; bad=$((bad+1)); }
    elif cmp -s "$PA/$f" "$PB/$f"; then same=$((same+1))
    else echo "  DIFF $f" >> $F; diff=$((diff+1)); fi
  done < <( { (cd $PA && find . -type f); (cd $PB && find . -type f); } | sort -u )
  echo "  BYTES $s: $same identical, $diff differ, $only on one side" >> $F
  python3 $G/items/fieldcheck.py $PA $PB ${AL[@]+"${AL[@]}"} > $F.fc 2>&1; grep -E '^(FIELD|builds|extra allowances|FIELDCHECK)' $F.fc >> $F
  ns=$((ns+1)); grep -q '^FIELDCHECK: ' $F.fc || { echo "  UNEXPLAINED $s: no field check result" >> $F; bad=$((bad+1)); }
  while read -r l; do echo "  UNEXPLAINED $s $l" >> $F; bad=$((bad+1)); done < <(grep -E '^FIELD ' $F.fc | grep -vE "^FIELD (($NEWOK): only in B|($OLDOK): only in A)\$")
done
rm -f $F.fc
grep -m1 '^extra allowances' $F; grep -E '^expected only|^  BYTES|^FIELDCHECK|^  UNEXPLAINED' $F
grep -hE '^FIELD |^  ONLY-' $F | sed 's/^ *//' | sort | uniq -c | sed -E 's/^ *([0-9]+) /SEEN x\1: /'
# exit code: every sample needs a field check; every one-side file and every FIELD line must be one the regexes name
if [ $ns -gt 0 ] && [ $bad = 0 ]; then echo "BYTECMP PASS samples=$ns" | tee -a $F; exit 0; fi
echo "BYTECMP FAIL samples=$ns unexplained=$bad" | tee -a $F; exit 1
