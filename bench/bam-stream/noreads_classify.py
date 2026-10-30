"""Classify the "No reads processed" logs found by noreads_census.sbatch (read-only).

Usage: python -I noreads_classify.py <hits.txt> <out.tsv>
Per log: date, sample ids, .1fq read total and block count, FIFO pass-2 count, STAR input/mapped reads,
singlet version/commit line if any, and a class:
  fifo_race_signature  pass 2 stopped after at most two blocks while the .1fq holds more (the 37e7cbd race)
  fifo_short_other     pass 2 stopped short, after more than two blocks
  fifo_full            pass 2 wrote every read, STAR still mapped none (input or reference problem)
  no_fifo              the run did not use the 1fq FIFO path
"""
import os, re, sys, time

R = {
    "tot": re.compile(r"\.1fq: (\d+) reads, (\d+) blocks"),
    "p2": re.compile(r"Pass 2 done: (\d+)"),
    "star_in": re.compile(r"Number of input reads \|\s*(\d+)"),
    "ver": re.compile(r"(singlet(?:ify)? (?:v|version )[\w.\-+]+|commit[= ][0-9a-f]{7,40})", re.I),
    "ids": re.compile(r"\b(GSM\d+|SRR\d+|ERR\d+|DRR\d+|GSE\d+)\b"),
}


def scan(path):
    tot = blocks = p2 = star_in = ver = None
    ids = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            if tot is None and (m := R["tot"].search(line)):
                tot, blocks = int(m.group(1)), int(m.group(2))
            if p2 is None and (m := R["p2"].search(line)):
                p2 = int(m.group(1))
            if star_in is None and (m := R["star_in"].search(line)):
                star_in = int(m.group(1))
            if ver is None and (m := R["ver"].search(line)):
                ver = m.group(1)
            if len(ids) < 12:
                ids.update(R["ids"].findall(line))
    if p2 is None:
        cls = "no_fifo"
    elif tot is not None and p2 >= tot:
        cls = "fifo_full"
    elif tot is not None and blocks:
        per_block = tot / blocks
        cls = "fifo_race_signature" if p2 <= 2 * per_block + 1 else "fifo_short_other"
    else:
        cls = "fifo_unknown_total"
    return cls, tot, blocks, p2, star_in, ver, sorted(ids)


def main():
    hits, out = sys.argv[1], sys.argv[2]
    files = [l.split("\t", 1)[0] for l in open(hits) if l.strip() and not l.startswith("##")]
    counts = {}
    with open(out, "w") as fo:
        fo.write("class\tdate\tgsm\tsrr\tgse\tfq_reads\tfq_blocks\tpass2\tstar_input\tversion\tlog\n")
        for f in files:
            cls, tot, blocks, p2, star_in, ver, ids = scan(f)
            counts[cls] = counts.get(cls, 0) + 1
            d = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
            pick = lambda p: ",".join(i for i in ids if i.startswith(p))
            fo.write("\t".join(str(x) for x in (cls, d, pick("GSM"), pick("SRR") or pick("ERR") or pick("DRR"),
                                                  pick("GSE"), tot, blocks, p2, star_in, ver, f)) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
