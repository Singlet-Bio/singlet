"""The comparison tools on data made from the model itself: a fake STAR genome directory, the kernel's TSV and window dump written
in the harness's formats, and a SAM written from the model's alignments. Checks that the formats parse and that identical data
compares as identical (the numbers of the real run need the GPU and STAR)."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compare_star_sam  # noqa: E402
import replay_model  # noqa: E402
from cases import build_genome, case_list, params  # noqa: E402
from paired_model import U64, map_pair  # noqa: E402

KEEP = ("concordant", "overlap50", "junction_m1", "junction_consistent", "protrude10_0", "discordant_ff")


def make_run(tmp):
    G = build_genome()
    d = tmp / "genome"
    d.mkdir()
    (d / "genomeParameters.txt").write_text("versionGenome\t2.7.4a\ngenomeChrBinNbits\t%d\n" % G.gChrBinNbits)
    (d / "chrName.txt").write_text("\n".join(G.names) + "\n")
    (d / "chrStart.txt").write_text("\n".join(str(x) for x in G.chrStart + [G.nGenome]) + "\n")
    (d / "chrLength.txt").write_text("\n".join(str(x) for x in G.chrLength) + "\n")
    G.G.astype(np.int8).tofile(d / "Genome")
    cases = [c for c in case_list(G) if c[0] in KEEP]
    r1, r2, tsv, dump, sam = [], [], ["idx\tstatus\tunmap\tnW\tnTr\tiBest\tChr\tStr\tscore\tnMM\tnMatch\tgLength\tiFrag\tnExons\texons\tcanonSJ"], bytearray(), []
    for i, (name, kind, kw, m1, m2) in enumerate(cases):
        P = params(kind, **kw)
        G.prepare(P)
        res = map_pair(G, P, m1, m2)
        r1.append(f"@r{i}\n{m1}\n+\n{'I' * len(m1)}\n")
        r2.append(f"@r{i}\n{m2}\n+\n{'I' * len(m2)}\n")
        words = [len(res.windows)]
        for w in res.windows:
            words += [w["Str"], w["Chr"], len(w["WA"]), w["WlastAnchor"]] + [v for a in w["WA"] for v in a]
        dump += struct.pack("<Q", len(words)) + struct.pack(f"<{len(words)}Q", *[v & U64 for v in words])
        line = [str(i), "0", str(res.unmapType), str(res.nW), str(len(res.transcripts)), str(res.primary)]
        if res.primary >= 0:
            t = res.transcripts[res.primary]
            exs = ";".join(":".join(str(int(v)) for v in t.exons[e][:4]) for e in range(t.nExons))
            sj = ",".join(str(int(v)) for v in t.canonSJ[:t.nExons - 1])
            line += [str(t.Chr), str(t.Str), str(t.maxScore), str(t.nMM), str(t.nMatch), str(t.gLength), str(t.iFrag), str(t.nExons), exs, sj]
        else:
            line += ["."] * 10
        tsv.append("\t".join(line))
        # SAM: the primary of a mapped pair, two records, CIGAR from the blocks
        if res.unmapType == -1 and res.primary >= 0 and len(res.transcripts) == 1 and t.iFrag == -1:
            mates = {}
            cur = []
            for e in range(t.nExons):
                cur.append((int(t.exons[e][1]), int(t.exons[e][2]), int(t.exons[e][3])))
            by = {}
            for g, ln, m in cur:
                by.setdefault(m, []).append((g, ln))
            for m in (0, 1):
                blocks = by[m]
                cig = ""
                for k, (g, ln) in enumerate(blocks):
                    if k:
                        cig += f"{g - (blocks[k - 1][0] + blocks[k - 1][1])}N"
                    cig += f"{ln}M"
                rev = (t.Str == 1) if m == 0 else (t.Str == 0)
                flag = 1 | 2 | (0x40 if m == 0 else 0x80) | (0x10 if rev else 0) | (0x20 if not rev else 0)
                sam.append(f"r{i}\t{flag}\t{G.names[t.Chr]}\t{blocks[0][0] - G.chrStart[t.Chr] + 1}\t255\t{cig}\t=\t1\t0\t*\t*\tNH:i:1")
        else:
            for m in (0, 1):
                sam.append(f"r{i}\t{77 if m == 0 else 141}\t*\t0\t0\t*\t*\t0\t0\t*\t*\tNH:i:0")
    (tmp / "r1.fq").write_text("".join(r1))
    (tmp / "r2.fq").write_text("".join(r2))
    (tmp / "kernel.tsv").write_text("\n".join(tsv) + "\n")
    (tmp / "win.bin").write_bytes(bytes(dump))
    (tmp / "ref.sam").write_text("\n".join(sam) + "\n")
    return d, len(cases)


def test_replay_and_compare_on_model_data(tmp_path, capsys):
    gdir, n = make_run(tmp_path)
    # the replay model sees one chromosome offset: chrStart.txt written above has nChr + 1 entries like STAR's
    rc = replay_model.main(str(gdir), str(tmp_path / "r1.fq"), str(tmp_path / "r2.fq"), str(tmp_path / "win.bin"), str(tmp_path / "kernel.tsv"),
                           "spliced", n)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"replayed {n} pairs: {n} identical" in out
    compare_star_sam.main(str(tmp_path / "kernel.tsv"), str(tmp_path / "ref.sam"), str(gdir), 5)
    out = capsys.readouterr().out
    assert "different alignment" not in out
    assert "= 100.0000%" in out
