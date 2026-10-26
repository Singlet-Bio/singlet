#!/usr/bin/env python3
"""Run peddy with the benchmark's test individuals removed from its 1000 Genomes ancestry background.

peddy 0.4.8 trains its ancestry classifier (PCA + SVM) on a built-in genotype matrix of the 2,504 phase 3 individuals
(GRCH38.sites.bin.gz) with a label vector in the module string `peddy.pca._str`. The individuals the benchmark scores are
in that background, which would let peddy recognise them; Singlet's allele frequencies and somalier's labelled background
already exclude them. This wrapper rewrites `peddy.pca.pca` at import time so the classifier is fit without those rows.

The matrix rows follow the 1000 Genomes phase 3 panel order. That is checked, not assumed: the panel's super-population
sequence must equal peddy's own label vector, or the wrapper refuses to run.

    python peddy_leaveout.py PANEL EXCLUDED.txt -- <peddy arguments>
"""
import inspect
import sys

import numpy as np


def main():
    panel, excluded = sys.argv[1], sys.argv[2]
    assert sys.argv[3] == "--", "usage: peddy_leaveout.py PANEL EXCLUDED -- peddy args"
    import peddy.pca as P

    codes = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}
    rows = [line.split("\t") for line in open(panel).read().splitlines()[1:] if line.strip()]
    ids = [r[0] for r in rows]
    labels = np.array([codes[r[2]] for r in rows])
    peddy_labels = np.array([int(x) for x in P._str.split("|")])
    if len(ids) != 2504 or not np.array_equal(labels, peddy_labels):
        sys.exit("peddy_leaveout: the panel order does not match peddy's background labels; refusing to guess")
    drop = set(open(excluded).read().split())
    keep = np.array([i not in drop for i in ids])
    print(f"peddy_leaveout: background {len(ids)} -> {int(keep.sum())} individuals", file=sys.stderr)

    src = inspect.getsource(P.pca)
    anchor = 'background_target = np.array([int(x) for x in _str.split("|")])'
    assert anchor in src, "peddy.pca.pca changed; update peddy_leaveout.py"
    src = src.replace(anchor, anchor + "\n    genos1kg = genos1kg[_KEEP]\n    background_target = background_target[_KEEP]")
    ns = dict(P.__dict__)
    ns["_KEEP"] = keep
    exec(compile(src, P.__file__, "exec"), ns)
    P.pca = ns["pca"]
    import peddy.peddy as PP
    if hasattr(PP, "pca"):
        PP.pca = ns["pca"]

    from peddy.cli import cli
    sys.argv = ["peddy"] + sys.argv[4:]
    sys.exit(cli())


if __name__ == "__main__":
    main()
