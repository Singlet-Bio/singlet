"""Laptop check for the in-memory cDNA reads of the GPU prefill (singlet.cpp process_block -> prefillReadsSet -> readSequences/appendPieces).

Run: python bench/host_cost/model_prefill_mem.py

file_path(): what the prefill did: the file-mode decode writes the R2 FASTQ (records skipped when R1 or R2 has length 0, each record
repeated dup_count times, bases written as ACGTN from the .1fq's codes 0-4), then readSequences parses the sequence lines with
the nucl() table (A,C,G,T -> 0-3, anything else 4) and builds seq / off.
mem_path(): what process_block now keeps per block (the same reads, the codes as stored, in the same loop and under the same skip rule)
and appendPieces concatenates (base = off.back(), running offsets, pieces in block order, any number of pool workers).
They must be equal. Random blocks: variable lengths, duplicate counts, empty reads (skipped), all five base codes.
"""
import random

N2A = "ACGTN"
LUT = {ord('A'): 0, ord('C'): 1, ord('G'): 2, ord('T'): 3}


def file_path(blocks):
    lines = []
    for blk in blocks:
        for r1len, r2, dup in blk:
            if r1len == 0 or len(r2) == 0:
                continue
            for _ in range(dup):
                lines.append("".join(N2A[c] for c in r2))
    seq = bytearray(); off = [0]
    for ln in lines:                      # readSequences: lut per byte, off = running end
        seq.extend(LUT.get(ord(c), 4) for c in ln)
        off.append(off[-1] + len(ln))
    return bytes(seq), off


def mem_path(blocks, workers):
    pieces = []
    for blk in blocks:                    # process_block: r2num / r2lens per block
        num = bytearray(); lens = []
        for r1len, r2, dup in blk:
            if r1len == 0 or len(r2) == 0:
                continue
            for _ in range(dup):
                num.extend(r2); lens.append(len(r2))
        pieces.append((bytes(num), lens))
    # appendPieces
    seq_at = [0]; off_at = [1]
    for s, l in pieces:
        seq_at.append(seq_at[-1] + len(s)); off_at.append(off_at[-1] + len(l))
    seq = bytearray(seq_at[-1]); off = [0] * off_at[-1]
    base = 0
    order = list(range(len(pieces))); random.shuffle(order)   # pool workers take pieces in any order
    for t in order:
        s, l = pieces[t]
        seq[seq_at[t]:seq_at[t] + len(s)] = s
        o = base + (seq_at[t] - seq_at[0])
        for k, ln in enumerate(l):
            o += ln; off[off_at[t] + k] = o
    return bytes(seq), off


def main():
    rng = random.Random(3)
    for trial in range(2000):
        blocks = []
        for _ in range(rng.randint(0, 9)):
            blk = []
            for _ in range(rng.randint(0, 40)):
                r1len = rng.choice([0, 28, 28, 28, 16])
                r2 = [rng.choice([0, 1, 2, 3, 3, 4]) for _ in range(rng.choice([0, 1, 5, 91, 91, 150]))]
                blk.append((r1len, r2, rng.choice([1, 1, 1, 2, 5])))
            blocks.append(blk)
        a = file_path(blocks); b = mem_path(blocks, rng.randint(1, 8))
        assert a == b, (trial, blocks)
    print("OK: 2000 random block sets, in-memory reads identical to the parsed FASTQ")


if __name__ == "__main__":
    main()
