#!/usr/bin/env python
"""Extract the exon/intron matrices and cell_calls.tsv of every sample of the GSE145370 bundle into
corpus_w3_total/GSE145370/<gsm>/ (data only: parsed, never executed).

    python -I tools/w3_total_extract.py [bundle] [outdir]

The archive is untrusted: only members whose full name matches samples/GSM<digits>/<known file> are written, to a path
built from the matched groups (never from the member name), and every member is CRC-checked by zipfile while reading.
"""
import os, re, sys, zipfile, hashlib

WANT = {"exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv"}
PAT = re.compile(r"^samples/(GSM[0-9]+)/([a-z_]+\.[0-9a-z]+)$")


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "corpus/bundles/GSE145370.singlet"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "corpus_w3_total/GSE145370"
    z = zipfile.ZipFile(bundle)
    n = 0
    tot = 0
    for info in z.infolist():
        m = PAT.match(info.filename)
        if not m or m.group(2) not in WANT:
            continue
        gsm, name = m.group(1), m.group(2)
        d = os.path.join(outdir, gsm)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name)
        h = hashlib.sha256()
        with z.open(info) as src, open(dst, "wb") as out:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                h.update(b)
                out.write(b)
        n += 1
        tot += info.file_size
        print(f"{gsm}/{name}\t{info.file_size}\t{h.hexdigest()[:16]}")
    print(f"extracted {n} members, {tot} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
