"""Write fixtures/cases.json: the synthetic genome (as text), the handcrafted cases, and what the model returns for each.
The CUDA harness's unit mode (paired_microbench --unit fixtures/cases.json) reads it. Run from bench/gpu_paired."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import build_genome, case_list, params  # noqa: E402
from paired_model import map_pair  # noqa: E402


def main(out="fixtures/cases.json"):
    G = build_genome()
    cases = []
    for name, kind, kw, m1, m2 in case_list(G):
        P = params(kind, **kw)
        G.prepare(P)
        r = map_pair(G, P, m1, m2)
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


if __name__ == "__main__":
    main(*sys.argv[1:])
