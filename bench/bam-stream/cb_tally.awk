# Tallies CB vs CR (and UR) over `samtools view` records. Usage: awk -F'\t' -v out=DIR -f cb_tally.awk
function bad_umi(u) { return (u ~ /N/) || (u ~ /^A+$/) || (u ~ /^C+$/) || (u ~ /^G+$/) || (u ~ /^T+$/) }
{ n++; cr=""; cb=""; cy=""; ur="";
  for (i = 12; i <= NF; i++) { t = substr($i, 1, 2); if (t == "CR") cr = substr($i, 6); else if (t == "CB") cb = substr($i, 6); else if (t == "CY") cy = substr($i, 6); else if (t == "UR") ur = substr($i, 6) }
  if (cr == "") nocr++;
  else if (cb == "") { if (cr ~ /N/) cb_none_N++; else cb_none_noN++ }
  else if (cb == cr) same++;
  else if (cb == "-" && cr ~ /^N+$/) allN_dash++;
  else if (cb == "-" && cr !~ /N/ && bad_umi(ur)) umi_bad_dash++;
  else { other++; if (other <= 50) print "DIFF", cr, cy, ur, cb > (out "/diff_examples.txt") } }
END { printf "reads=%d no_CR=%d CB_absent_CRhasN=%d CB_absent_CRnoN=%d CB_equals_CR=%d CB_allN_dash=%d CB_UMIbad_dash=%d CB_other_differs=%d\n", n, nocr, cb_none_N, cb_none_noN, same, allN_dash, umi_bad_dash, other }
