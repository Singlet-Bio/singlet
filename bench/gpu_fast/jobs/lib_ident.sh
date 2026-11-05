# Identity helpers of the fast lane. Source this file. It needs SCR (scratch with repo/ from the scripts branch) and the variable `s` (sample) set by the caller.
# The block below is a verbatim copy of bench/gpu_batch1/jobs/tp.sbatch at 0fdec57f, lines 136 to 177: EXPECT_DIFF, KNOWN_nuclei2k_v2, READF,
# prune, readhash and compare_out. tests/test_sync.py checks that it still equals the batch 1 text. Edit the batch 1 copy first.
EXPECT_DIFF=" ${EXPECT_DIFF:-pileup_stats.json read_stats.tsv star_Log.std.out summary.json} "
KNOWN_nuclei2k_v2=${KNOWN_nuclei2k_v2-nonhost_reads.fq}   # the known non-host writer bug of the gate (E15 section 5.7)
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
prune() { find $1 \( -name '*.bam' -o -name '*.sam' -o -name '*.cram' -o -name '*.fq*' -o -name '*.fastq*' -o -name '*.1fq' -o -name '.gpu_handoff*' -o -name '*Unmapped*' \
                     -o -name '_STARtmp*' -o -name '*.reads' -o -name '*.reads.q' \) -prune -exec rm -rf {} + 2>/dev/null; }
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }

# compare_out LABEL A_OUT A_CRS A_RH B_OUT B_CRS B_RH DETAILFILE   -> 0 when identical, 1 when not (same checks as the gate's pair comparison)
compare_out() {
  local lab=$1 a=$2 ca=$3 rha=$4 b=$5 cb=$6 rhb=$7 d=$8 ok=1 f
  echo "=== $lab: $b vs $a" >> $d
  if [ ! -d "$a" ]; then echo "  no reference ($a)" >> $d; return 2; fi
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $d.equiv 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $d.equiv | cut -c1-220 >> $d
  local UNX=""
  for f in $(grep -a '^  DIFF ' $d.equiv | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac; done
  grep -aq '^  DIFF provenance.json: /references/whitelist/name: [^;]*$' $d.equiv && UNX=${UNX/ provenance.json/}
  for f in $(grep -a '^  only in ' $d.equiv | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
  grep -aq '^files:' $d.equiv || UNX="$UNX out_equiv-failed"
  [ -n "$UNX" ] && { echo "  out_equiv UNEXPECTED:$UNX" >> $d; ok=0; }
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
           Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "  star_Solo.out/$f DIFFERS" >> $d; ok=0; }
  done
  cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "  star_Log.final.out STATISTICS DIFFER" >> $d; ok=0; }
  cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "  star_SJ.out.tab DIFFERS" >> $d; ok=0; }
  if [ -f "$rha" ] && [ -f "$rhb" ]; then
    local kv=KNOWN_$s; local KN=" ${!kv:-} "
    for f in $READF; do
      local ha hb; ha=$(awk -v f=$f '$1==f{print $2" "$3}' $rha); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rhb)
      if [ "$ha" = "$hb" ]; then :
      elif [[ "$KN" == *" $f "* ]] && [ -n "$ha" ] && [ "${ha##* }" = "${hb##* }" ]; then echo "  $f: other records, same count (known non-host writer bug)" >> $d
      else echo "  $f: RECORDS DIFFER ($ha vs $hb)" >> $d; ok=0; fi
    done
  else echo "  read files: not compared (no hashes)" >> $d; ok=0; fi
  if [ -f "$ca" ] && [ -f "$cb" ]; then
    { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "  CellReads.stats ROWS DIFFER" >> $d; ok=0; }
  else echo "  CellReads.stats: not compared" >> $d; ok=0; fi
  [ $ok = 1 ] && { echo "  IDENTICAL" >> $d; return 0; } || { echo "  NOT IDENTICAL" >> $d; return 1; }
}
