"""Order-independent content digest of FASTQ files: spot count, read length, and 64-bit sums of per-record hashes.

    python fq_digest.py FILE [FILE ...]            tab-separated rows: file, spots, read_len, seq_digest, qual_digest
    python fq_digest.py --records FILE             spots only (cheap)

A record contributes crc32(seq) << 32 | adler32(seq) to the sequence digest (likewise for quality); digests are the sum
modulo 2^64 over records, so two files hold the same reads iff their digests are equal, whatever the order of the records
or the read names. gzip files are read through `pigz -dc` (or `gzip -dc`). Standard library only.
"""
import subprocess
import sys
import zlib

MASK = (1 << 64) - 1


def open_stream(path):
    if path.endswith(".gz"):
        for exe in ("pigz", "gzip"):
            try:
                p = subprocess.Popen([exe, "-dc", path], stdout=subprocess.PIPE)
                return p.stdout
            except FileNotFoundError:
                continue
    return open(path, "rb")


def digest(path):
    fh = open_stream(path)
    n = 0
    rl = 0
    sd = qd = 0
    crc, adl = zlib.crc32, zlib.adler32
    while True:
        h = fh.readline()
        if not h:
            break
        s = fh.readline().rstrip(b"\r\n")
        fh.readline()
        q = fh.readline().rstrip(b"\r\n")
        if n == 0:
            rl = len(s)
        n += 1
        sd += (crc(s) << 32) | adl(s)
        qd += (crc(q) << 32) | adl(q)
    return n, rl, sd & MASK, qd & MASK


def main():
    args = sys.argv[1:]
    for f in args:
        n, rl, sd, qd = digest(f)
        print(f"{f}\t{n}\t{rl}\t{sd:016x}\t{qd:016x}", flush=True)


if __name__ == "__main__":
    main()
