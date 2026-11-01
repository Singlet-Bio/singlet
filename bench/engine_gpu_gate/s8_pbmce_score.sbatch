#!/bin/bash
#SBATCH --job-name=s8-pbmce
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: PBMC-E (pbmc1, full depth) long-read USA score of BINDIR's singlet-gpu (CPU engine, shipped intron rules) with the count-matrix layout of PR #165 (spliced_counts / unspliced_counts /
# ambiguous_counts hold the USA labels). Same steps as the expression-sota run_singlet_labels.sbatch + run_score_labels.sbatch (R1 padded to 28 nt, --protocol 10x-3p-v3), scoring copy patched to read S/U/A from the USA matrices.
# Output: $W/s8/pbmce_<sha8>/{singlet,X/score_usa.json}.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; TAG=$(cut -c1-8 $BINDIR/commit)
[ -d $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
source $W/expression-sota/longread_usa/code/common.sh
O0=$W/s8/pbmce_$TAG; O=$O0/singlet; rm -rf $O0; mkdir -p $O; export SINGLET_STALL_TIMEOUT=10800 SINGLET_WHITELIST_DIR=$PB/whitelists
echo "commit=$(cat $BINDIR/commit) node=$(hostname) $(date -Is)"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR
pad() { zcat "$1" | awk 'NR%4==2{print $0 "A"; next} NR%4==0{print $0 "F"; next} {print}' | gzip -1 > "$2"; }
i=0; for f in ${R1//,/ }; do pad $f $SCR/R1_$i.fastq.gz & i=$((i+1)); done; wait
cat $SCR/R1_*.fastq.gz > $SCR/R1.fastq.gz; rm -f $SCR/R1_*.fastq.gz; cat ${R2//,/ } > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz --protocol 10x-3p-v3 -o $SCR/x.1fq > $O/encode.log 2>&1 || { tail -20 $O/encode.log; exit 1; }
rm -f $SCR/R*.fastq.gz; cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/x.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
echo "run exit=$?"; grep -q "BAM: unsorted STAR stream into the pileup" $O/run.log && echo "STREAM PATH OK" || echo "STREAM PATH FAIL"; rm -rf $O/private
# scoring copy: S/U/A from the USA matrices
XS=$W/expression-sota/longread_usa; X=$O0/X; mkdir -p $X; for d in truth starsolo alevinfry; do ln -s $XS/$d $X/$d; done; ln -s $O $X/singlet
cp -r $XS/code $X/code; cp -r $XS/extract $X/extract
PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python
$PY -P - <<PYEOF
p = "$X/extract/runio.py"; s = open(p).read()
a = '("E", "exon_counts"), ("U", "intron_counts"), ("A", "ambiguous_counts")'
assert a in s; open(p, "w").write(s.replace(a, a + ', ("Su", "spliced_counts"), ("Uu", "unspliced_counts")'))
p = "$X/code/score_usa.py"; s = open(p).read()
a = s.index('        E, U, A = (s[k].pad(universe) for k in ("E", "U", "A"))')
b = s.index('        return {"S": S, "U": U.X, "A": A.X}, E.present') + len('        return {"S": S, "U": U.X, "A": A.X}, E.present')
s = s[:a] + '        S, U, A = (s[k].pad(universe) for k in ("Su", "Uu", "A"))\n        return {"S": S.X, "U": U.X, "A": A.X}, S.present' + s[b:]
open(p, "w").write(s)
PYEOF
$PY -P $X/code/score_usa.py $X $X/score_usa.json
ls -l $X/score_usa.json
