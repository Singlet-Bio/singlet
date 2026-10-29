"""Strip germline (genotype-like) site lists from result JSONs before they are committed: the line-defining homoplasmic
sites, privacy-rule examples and MQuad's top calls (in a pool these are the donors' haplotype variants). Counts stay.
  python sanitize.py results/*.json      (in place; the unredacted files stay on Clipper)"""
import json
import sys

DROP = {"line_defining_examples", "leaked_examples", "examples", "top", "homoplasmic_calls_written"}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


for f in sys.argv[1:]:
    d = json.load(open(f))
    json.dump(clean(d), open(f, "w"), indent=1)
    print("sanitized", f)
