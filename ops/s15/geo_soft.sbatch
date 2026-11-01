#!/bin/bash
#SBATCH --job-name=s15-geosoft
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GEO family SOFT files (public FTP/HTTPS data, not the web UI): sample characteristics, descriptions and series summary for the pilot's HTO / paired-library checks.
D=/mnt/projects/debruinz_project/debruinz/singlet/jobs/s15/geo; mkdir -p $D; cd $D
for g in GSE256089 GSE298047 GSE168859; do
  p=${g:0:${#g}-3}nnn
  for u in https://ftp.ncbi.nlm.nih.gov/geo/series/$p/$g/soft/${g}_family.soft.gz ftp://ftp.ncbi.nlm.nih.gov/geo/series/$p/$g/soft/${g}_family.soft.gz "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=$g\&targ=self\&form=text\&view=full"; do
    for t in 1 2 3 4 5 6; do curl -sSL -m 120 -C - -o $g.soft.gz "$u" 2> curl.err && break; done; if [ -s $g.soft.gz ] && gzip -t $g.soft.gz 2> /dev/null; then echo "$g ok via $u $(du -h $g.soft.gz | cut -f1)"; break; else echo "$g failed via $u: $(head -c 200 curl.err) $(head -c 80 $g.soft.gz | tr -d "
")"; rm -f $g.soft.gz; fi; done
done
python3 - <<'PY'
import gzip, re
want = {"GSE256089": ["GSM8085472", "GSM8085473"], "GSE298047": ["GSM9005866", "GSM9005867", "GSM9005868"], "GSE168859": ["GSM5171610", "GSM5171642"]}
for g, gs in want.items():
    cur = None; out = []
    for l in gzip.open(g + ".soft.gz", "rt", errors="replace"):
        if l.startswith("^SAMPLE"): cur = l.split("=")[1].strip()
        if l.startswith("^SERIES"): cur = "SERIES"
        if cur in gs or (cur == "SERIES" and re.match(r"!Series_(summary|overall_design)", l)):
            if re.match(r"!(Sample_(title|characteristics_ch1|description|data_processing|library_selection|library_strategy|source_name_ch1)|Series_(summary|overall_design))", l):
                out.append("%s %s" % (cur, l.strip()[:420]))
    print("=====", g); print("\n".join(out))
PY
