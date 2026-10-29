"""Write fixtures/cases.json: the synthetic genome (as text), the handcrafted cases, and what the model returns for each.
The CUDA harness's unit mode (paired_microbench --unit fixtures/cases.json) reads it. Run from bench/gpu_paired."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import build_genome, case_list, clip_cases, params  # noqa: E402
from paired_model import MateClip, clip_mate, map_pair, nucl  # noqa: E402


def main(out="fixtures/cases.json"):
    G = build_genome()
    cases = []
    windows = []
    for name, kind, kw, m1, m2 in case_list(G):
        P = params(kind, **kw)
        G.prepare(P)
        r = map_pair(G, P, m1, m2)
        windows.append(r.windows)
        cases.append(dict(
            name=name, params=kind, overrides=kw, mate1=m1, mate2=m2,
            expect=dict(unmapType=r.unmapType, nW=r.nW, primary=r.primary, transcripts=r.summaries()),
        ))
    doc = dict(
        genome=dict(gChrBinNbits=G.gChrBinNbits, chroms=G.text),
        note="chromosomes start at multiples of 2^gChrBinNbits with padding 5 between them; params: STAR defaults, 'atac' = "
             "--alignIntronMax 1 --alignMatesGapMax 2000; exons are [rStart, gStart, length, iFrag] in the combined read",
        cases=cases)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(out, os.path.getsize(out), "bytes,", len(cases), "cases")
    txt = out.replace(".json", ".txt")
    clips = []
    for name, sq, kw in clip_cases(G):
        c = MateClip(**kw)
        out_seq, c5, c3 = clip_mate(nucl(sq), c)
        clips.append((name, sq, c, len(out_seq), c5 if len(out_seq) else 0, c5, c3))
    write_txt(txt, G, doc, windows, clips)
    print(txt, os.path.getsize(txt), "bytes")


def _s(x):  # unsigned 64-bit -1 sentinels as -1
    return -1 if x >= (1 << 63) else int(x)


def write_txt(path, G, doc, windows, clips):
    """Line-oriented copy for the CUDA harness's unit mode (no JSON parser needed): the genome, and per case the mates, the windows
    the model built (the kernel stage under test starts from them), and the expected result."""
    L = [f"GENOME {G.gChrBinNbits} {len(G.names)}"]
    for n in G.names:
        L.append(f"CHR {n} {G.text[n]}")
    for c, wins in zip(doc["cases"], windows):
        kv = " ".join(f"{k} {v}" for k, v in c["overrides"].items())
        L.append(f"CASE {c['name']} {c['params']} {len(c['overrides'])} {kv}".rstrip())
        L.append(f"READ {c['mate1']} {c['mate2']}")
        L.append(f"NWIN {len(wins)}")
        for w in wins:
            rows = " ".join(" ".join(str(_s(v)) for v in a) for a in w["WA"])
            L.append(f"WIN {w['Str']} {w['Chr']} {_s(w['WlastAnchor'])} {len(w['WA'])} {rows}".rstrip())
        e = c["expect"]
        L.append(f"EXPECT {e['unmapType']} {e['nW']} {e['primary']} {len(e['transcripts'])}")
        for t in e["transcripts"]:
            ex = " ".join(" ".join(str(v) for v in x) for x in t["exons"])
            sj = " ".join(str(v) for v in t["canonSJ"])
            L.append(f"TR {len(t['exons'])} {t['maxScore']} {t['nMatch']} {t['nMM']} {t['gLength']} {t['iFrag']} {t['Str']} {t['Chr']} "
                     f"{int(t['primary'])} {ex} {sj}".rstrip())
        L.append("END")
    for name, sq, c, ln, start, c5, c3 in clips:
        L.append(f"CLIPCASE {name} {c.n5} {c.after5} {c.n3} {c.after3} {c.adMMp} {c.adapter3} {sq} {ln} {start} {c5} {c3}")
    with open(path, "w") as f:
        f.write(chr(10).join(L) + chr(10))


if __name__ == "__main__":
    main(*sys.argv[1:])
