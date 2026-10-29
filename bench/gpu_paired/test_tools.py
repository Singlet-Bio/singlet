"""The comparison tools on data made from the model itself: a fake STAR genome directory, the kernel's TSV and window dump written in
the harness's formats, and a SAM written from the model's alignments with star_sam.py (the transcription of STAR's SAM code).
Checks that the formats parse, that identical data compares as identical, and that every kind of difference the strict comparison is
meant to catch (indel position, soft-clip length, strand, mate position, NH, tie set, primary) is reported as one.
The numbers of the real run need the GPU and STAR."""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import compare_star_sam  # noqa: E402
import replay_model  # noqa: E402
from cases import NEXTERA, build_genome, case_list, params, rc  # noqa: E402
from paired_model import U64, MateClip, map_pair  # noqa: E402
from star_sam import sam_records  # noqa: E402


def extra_cases(G):
    t = G.text["chr1"]
    frag = t[2000:2040]
    return [
        # (name, kind, overrides, mate1, mate2, clip)
        ("deletion_in_m1", "spliced", {}, t[1000:1050] + t[1051:1101], rc(t[1300:1400]), None),
        ("insertion_in_m2", "spliced", {}, t[1000:1100], rc(t[1300:1350] + "A" + t[1350:1400]), None),
        ("adapter_clipped", "spliced", {}, frag + NEXTERA, rc(frag) + NEXTERA, (MateClip(adapter3=NEXTERA), MateClip(adapter3=NEXTERA))),
        ("n5_both", "spliced", {}, "TT" + t[1000:1100], rc(t[1300:1400]) + "GG", (MateClip(n5=2), MateClip(n3=2))),
    ]


def make_run(tmp, drop=()):
    G = build_genome()
    d = tmp / "genome"
    d.mkdir()
    (d / "genomeParameters.txt").write_text("versionGenome\t2.7.4a\ngenomeChrBinNbits\t%d\n" % G.gChrBinNbits)
    (d / "chrName.txt").write_text("\n".join(G.names) + "\n")
    (d / "chrStart.txt").write_text("\n".join(str(x) for x in G.chrStart + [G.nGenome]) + "\n")
    (d / "chrLength.txt").write_text("\n".join(str(x) for x in G.chrLength) + "\n")
    G.G.astype(np.int8).tofile(d / "Genome")
    keep = ("concordant", "overlap50", "junction_m1", "junction_consistent", "protrude10_0", "discordant_ff", "multimapper")
    cases = [(n, k, kw, m1, m2, None) for (n, k, kw, m1, m2) in case_list(G) if n in keep] + extra_cases(G)
    r1, r2, tsv, dump, sam = [], [], ["idx\tstatus\tunmap\tnW\tnTr\tiBest\ttranscripts\tlorig1\tlorig2\tclip5_1\tclip3_1\tclip5_2\tclip3_2"], bytearray(), []
    names = []
    for i, (name, kind, kw, m1, m2, clip) in enumerate(cases):
        names.append(name)
        P = params(kind, **kw)
        G.prepare(P)
        res = map_pair(G, P, m1, m2, clip=clip)
        rd = res.read
        r1.append(f"@r{i}\n{m1}\n+\n{'I' * len(m1)}\n")
        r2.append(f"@r{i}\n{m2}\n+\n{'I' * len(m2)}\n")
        words = [len(res.windows)]
        for w in res.windows:
            words += [w["Str"], w["Chr"], len(w["WA"]), w["WlastAnchor"]] + [v for a in w["WA"] for v in a]
        dump += struct.pack("<Q", len(words)) + struct.pack(f"<{len(words)}Q", *[v & U64 for v in words])
        tsv.append("\t".join([str(i), "0", str(res.unmapType), str(res.nW), str(len(res.transcripts)), str(res.primary), res.tsv(),
                              str(rd.readLengthOriginal[0]), str(rd.readLengthOriginal[1]), str(rd.clip5[0]), str(rd.clip3[0]),
                              str(rd.clip5[1]), str(rd.clip3[1])]))
        if res.unmapType == -1:
            nTr = len(res.transcripts)
            for hi, t in enumerate(res.transcripts):
                recs = sam_records([tuple(map(int, t.exons[e][:4])) for e in range(t.nExons)],
                                   [int(x) for x in t.canonSJ[:t.nExons - 1]], t.Str, G.chrStart[t.Chr], rd.readLength,
                                   rd.readLengthOriginal, rd.clip5, rd.clip3, rd.Lread, primary=(hi == res.primary))
                for r in sorted(recs, key=lambda r: r["mate"]):
                    sam.append(f"r{i}\t{r['flag']}\t{G.names[t.Chr]}\t{r['pos']}\t255\t{r['cigar']}\t=\t{r.get('pnext', 0)}\t0\t*\t*\tNH:i:{nTr}\tHI:i:{hi + 1}")
        else:
            for m in (0, 1):
                sam.append(f"r{i}\t{77 if m == 0 else 141}\t*\t0\t0\t*\t*\t0\t0\t*\t*\tNH:i:0")
    (tmp / "r1.fq").write_text("".join(r1))
    (tmp / "r2.fq").write_text("".join(r2))
    (tmp / "kernel.tsv").write_text("\n".join(tsv) + "\n")
    (tmp / "win.bin").write_bytes(bytes(dump))
    (tmp / "ref.sam").write_text("\n".join(sam) + "\n")
    return d, len(cases), names


def test_replay_and_compare_on_model_data(tmp_path, capsys):
    gdir, n, _ = make_run(tmp_path)
    rc_ = replay_model.main(str(gdir), str(tmp_path / "r1.fq"), str(tmp_path / "r2.fq"), str(tmp_path / "win.bin"), str(tmp_path / "kernel.tsv"),
                            "spliced", n)
    out = capsys.readouterr().out
    assert rc_ == 0, out
    assert f"replayed {n} pairs: {n} identical" in out
    compare_star_sam.main(str(tmp_path / "kernel.tsv"), str(tmp_path / "ref.sam"), str(gdir), 5)
    out = capsys.readouterr().out
    assert "= 100.0000%" in out and "differs" not in out, out


def _mutate(tmp_path, name, fn):
    gdir, n, names = make_run(tmp_path)
    i = names.index(name)
    lines = (tmp_path / "ref.sam").read_text().splitlines()
    new = []
    for ln in lines:
        c = ln.split("\t")
        new.append("\t".join(fn(c)) if c[0] == f"r{i}" else ln)
    (tmp_path / "ref.sam").write_text("\n".join(new) + "\n")
    compare_star_sam.main(str(tmp_path / "kernel.tsv"), str(tmp_path / "ref.sam"), str(gdir), 5)


def test_cigar_of_indel_and_soft_clip_cases(tmp_path):
    make_run(tmp_path)
    sam = (tmp_path / "ref.sam").read_text().splitlines()
    by = {}
    for ln in sam:
        c = ln.split("\t")
        by.setdefault(c[0], []).append((int(c[1]), c[3], c[5], c[7]))
    G = build_genome()
    names = [c[0] for c in case_list(G) if c[0] in ("concordant", "overlap50", "junction_m1", "junction_consistent", "protrude10_0", "discordant_ff", "multimapper")]
    i = len(names)   # first extra case: deletion_in_m1
    import re
    m = re.fullmatch(r"(\d+)M1D(\d+)M", by[f"r{i}"][0][2])        # the deletion is left-aligned where the bases repeat: 49M1D51M or 50M1D50M
    assert m and int(m.group(1)) + int(m.group(2)) == 100
    assert by[f"r{i + 1}"][1][2] == "50M1I50M"
    assert by[f"r{i + 2}"][0][2] == "40M19S" and by[f"r{i + 2}"][1][2] == "19S40M"
    assert by[f"r{i + 3}"][0][2] == "2S100M" and by[f"r{i + 3}"][1][2] == "2S100M"   # 5' clip of mate 1; mate 2's 3' clip is on its left


def _shift_deletion(cigar):
    import re
    m = re.fullmatch(r"(\d+)M1D(\d+)M", cigar)
    return f"{int(m.group(1)) + 1}M1D{int(m.group(2)) - 1}M" if m else cigar


@pytest.mark.parametrize("name, mutation, expect", [
    ("deletion_in_m1", lambda c: [*c[:5], _shift_deletion(c[5]), *c[6:]], "alignment set"),
    ("adapter_clipped", lambda c: [*c[:5], c[5].replace("40M19S", "41M18S"), *c[6:]], "alignment set"),
    ("concordant", lambda c: [c[0], str(int(c[1]) ^ 0x10 ^ 0x20), *c[2:]], "alignment set"),
    ("concordant", lambda c: [*c[:7], str(int(c[7]) + 1), *c[8:]], "alignment set"),
    ("concordant", lambda c: [*c[:3], str(int(c[3]) + 1), *c[4:]], "alignment set"),
    ("concordant", lambda c: [*c[:11], "NH:i:2", *c[12:]], "NH"),
    ("concordant", lambda c: [c[0], str(int(c[1]) ^ 0x2), *c[2:]], "alignment set"),
])
def test_strict_comparison_catches(tmp_path, capsys, name, mutation, expect):
    _mutate(tmp_path, name, mutation)
    out = capsys.readouterr().out
    assert "= 100.0000%" not in out.split("criterion")[0].splitlines()[-1] or "differs" in out, out
    assert "differs" in out and expect in out, out


def test_tie_set_and_primary(tmp_path, capsys):
    gdir, n, names = make_run(tmp_path)
    i = names.index("multimapper")
    lines = (tmp_path / "ref.sam").read_text().splitlines()
    # (a) STAR reports the second locus's primary flag instead: same set, different primary
    swap = []
    for ln in lines:
        c = ln.split("\t")
        if c[0] == f"r{i}":
            c[1] = str(int(c[1]) ^ 0x100)
        swap.append("\t".join(c))
    (tmp_path / "ref.sam").write_text("\n".join(swap) + "\n")
    compare_star_sam.main(str(tmp_path / "kernel.tsv"), str(tmp_path / "ref.sam"), str(gdir), 5)
    assert "primary only (tie-break)" in capsys.readouterr().out
    # (b) STAR reports one of the two tied loci only (NH 1): a different tie set
    one = [ln.replace("NH:i:2", "NH:i:1") for ln in lines if not (ln.split("\t")[0] == f"r{i}" and ln.endswith("HI:i:2"))]
    (tmp_path / "ref.sam").write_text("\n".join(one) + "\n")
    compare_star_sam.main(str(tmp_path / "kernel.tsv"), str(tmp_path / "ref.sam"), str(gdir), 5)
    out = capsys.readouterr().out
    assert "NH (kernel 2, STAR 1)" in out or "differs: NH" in out
