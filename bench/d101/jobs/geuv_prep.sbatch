#!/bin/bash
#SBATCH --job-name=d101-geuvprep
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# HLA benchmark set of the T1K paper (Song et al. 2023): Geuvadis RNA-seq (E-GEUV-1, 2 x 75) of 1000 Genomes samples typed by
# Abi-Rached et al. 2018 (20181129_HLA_types_full_1000_Genomes_Project_panel.txt). Writes $W/jobs/d101/geuv/list.tsv with N samples
# (sample, run, fastq_1 URL, fastq_2 URL, population): every sample of the truth file that Geuvadis sequenced, ordered by sample id,
# then every k-th so N spreads over the list. Truth and the SDRF stay next to it.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/d101/geuv; mkdir -p $D; N=${N:-60}
cd $D
curl -fsS --retry 5 -o truth.txt https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HLA_types/20181129_HLA_types_full_1000_Genomes_Project_panel.txt
curl -fsSL --retry 5 -o sdrf.txt https://www.ebi.ac.uk/biostudies/files/E-GEUV-1/E-GEUV-1.sdrf.txt || \
  curl -fsSL --retry 5 -o sdrf.txt https://www.ebi.ac.uk/arrayexpress/files/E-GEUV-1/E-GEUV-1.sdrf.txt
head -2 truth.txt | cut -c1-300; head -1 sdrf.txt | tr '\t' '\n' | cat -n | head -40
python3 - "$N" <<'EOF'
import csv, sys
N = int(sys.argv[1])
truth = {}
with open("truth.txt") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    si = next(i for i, h in enumerate(hdr) if h.lower().replace(" ", "") in ("sampleid", "sample"))
    pi = next((i for i, h in enumerate(hdr) if h.lower() == "population"), None)
    for line in f:
        v = line.rstrip("\n").split("\t")
        truth[v[si]] = v[pi] if pi is not None else "."
rows = {}
with open("sdrf.txt") as f:
    r = csv.reader(f, delimiter="\t")
    h = next(r)
    src = h.index("Source Name")
    uri = [i for i, x in enumerate(h) if "FASTQ_URI" in x]
    run = [i for i, x in enumerate(h) if "ENA_RUN" in x]
    for v in r:
        s = v[src]
        if s not in truth:
            continue
        rows.setdefault(s, {"run": v[run[0]] if run else ".", "uris": set()})
        for i in uri:
            if v[i]:
                rows[s]["uris"].add(v[i])
ok = sorted(s for s, d in rows.items() if len(d["uris"]) == 2)
print(f"{len(truth)} typed samples, {len(rows)} with Geuvadis RNA-seq, {len(ok)} with a FASTQ pair", file=sys.stderr)
step = max(1, len(ok) // N)
pick = ok[::step][:N]
with open("list.tsv", "w") as o:
    for s in pick:
        u = sorted(rows[s]["uris"])
        u = [x if x.startswith("http") or x.startswith("ftp") else "ftp://" + x for x in u]
        o.write(f"{s}\t{rows[s]['run']}\t{u[0]}\t{u[1]}\t{truth[s]}\n")
print(f"picked {len(pick)}", file=sys.stderr)
EOF
wc -l list.tsv; head -3 list.tsv
