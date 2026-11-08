#!/bin/bash
#SBATCH --job-name=b20-5pv1
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B20 follow-up: GSM4987144 (10x 5' v1, human + mouse cell lines; pilot touch class), run SRR13296299. The detector calls it 10x-3p-v2
# (same 26-nt R1 and whitelist as 3' v2). Encode it in auto mode and with --protocol 10x-5p-v2 (the registry's 26-nt 5' chemistry),
# run the engine on both against GRCh38 and compare the gene count matrices and cell calls. Identical outputs mean the engine's
# wrong-strand restart already corrects the call, and only the recorded chemistry needs fixing.
# Needs BIN (singlet-gpu). Output: $W/jobs/protocol_b20/5pv1-<job>/ (logs, compare.txt).
set -uo pipefail
: "${BIN:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists SINGLET_REF_BASE=$REF SINGLET_STALL_TIMEOUT=3600
D=$W/data/b20/SRR13296299; O=$W/jobs/protocol_b20/5pv1-$SLURM_JOB_ID; mkdir -p $D $O; T=$SLURM_CPUS_PER_TASK
SCR=/tmp/$USER-5pv1-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
part() {   # url out s e: append to out until it holds bytes s..e (EBI closes long transfers early)
  local url=$1 out=$2 s=$3 e=$4 t have
  for t in $(seq 1 30); do have=$(stat -c %s $out 2>/dev/null || echo 0)
    [ $have -ge $((e - s + 1)) ] && return 0
    curl -sS --retry 5 -r $((s + have))-$e "$url" >> $out; done
  return 1
}
fetch() {   # url out bytes: 8 resumable byte ranges
  local url=$1 out=$2 n=$3 k s e
  [ -s $out ] && [ $(stat -c %s $out) = $n ] && return 0
  for k in 0 1 2 3 4 5 6 7; do s=$((n * k / 8)); e=$((n * (k + 1) / 8 - 1))
    part "$url" $out.p$k $s $e & done; wait
  cat $out.p? > $out && rm -f $out.p?; [ $(stat -c %s $out) = $n ]
}
B=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR132/099/SRR13296299
fetch $B/SRR13296299_1.fastq.gz $D/R1.fastq.gz 1425242552 || { echo "R1 size mismatch"; exit 2; }
fetch $B/SRR13296299_2.fastq.gz $D/R2.fastq.gz 4645771477 || { echo "R2 size mismatch"; exit 2; }
echo "bin $BIN commit $(cat $(dirname $BIN)/commit-gpu 2>/dev/null) $(date -Is)" | tee $O/provenance.txt
for mode in auto forced; do
  P=""; [ $mode = forced ] && P="--protocol 10x-5p-v2"
  ( cd $SCR && $BIN download --r1 $D/R1.fastq.gz --r2 $D/R2.fastq.gz $P --staging-dir $SCR/stage-$mode -o $SCR/$mode.1fq ) > $O/download.$mode.log 2>&1
  echo "download $mode exit $? $(grep -a -h 'Protocol:' $O/download.$mode.log | tail -1)"
  ( cd $SCR && /usr/bin/time -v -o $O/engine.$mode.time $BIN $SCR/$mode.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/out-$mode/ --threads $T --pipeline --output-format mtx ) > $O/engine.$mode.log 2>&1
  echo "engine $mode exit $?"
  grep -a -E "strand|Strand|restart|protocol_id|soloStrand|BAM:" $O/engine.$mode.log | cut -c1-300 | head -20 | sed "s/^/  [$mode] /"
  rm -f $SCR/$mode.1fq
done
{
  echo "file auto forced same"
  for f in $(cd $SCR/out-auto && ls *.mtx.gz *_barcodes.tsv.gz *_features.tsv.gz cell_calls.tsv 2>/dev/null); do
    a=$(zcat -f $SCR/out-auto/$f | md5sum | cut -c1-12); b=$(zcat -f $SCR/out-forced/$f 2>/dev/null | md5sum | cut -c1-12)
    echo "$f $a $b $([ "$a" = "$b" ] && echo yes || echo NO)"
  done
  for m in auto forced; do echo "$m cells $(($(wc -l < $SCR/out-$m/cell_calls.tsv) - 1))"; grep -a -o -E '"protocol[a-z_]*": *"[^"]*"' $SCR/out-$m/provenance.json | sed "s/^/$m /"; done
} | tee $O/compare.txt
# a numeric comparison when the files differ: total counts and called-cell overlap
$W/envs/pbench/bin/python3 - $SCR/out-auto $SCR/out-forced <<'EOF' | tee -a $O/compare.txt
import sys, gzip, scipy.io as io
a, b = sys.argv[1], sys.argv[2]
def cells(d):
    return set(l.split("\t")[0] for l in open(d + "/cell_calls.tsv").read().splitlines()[1:])
ca, cb = cells(a), cells(b)
print("called cells: auto %d forced %d shared %d" % (len(ca), len(cb), len(ca & cb)))
for f in ("exon_counts.mtx.gz", "intron_counts.mtx.gz"):
    try:
        ma, mb = io.mmread(gzip.open(a + "/" + f)), io.mmread(gzip.open(b + "/" + f))
        print("%s: auto sum %d nnz %d; forced sum %d nnz %d" % (f, ma.sum(), ma.nnz, mb.sum(), mb.nnz))
    except Exception as e:
        print(f, "not compared:", e)
EOF
echo done $(date -Is)
