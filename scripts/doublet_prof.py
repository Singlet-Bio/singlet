#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""cProfile of the current sdfl port on one dataset (stage timing before optimisation). python doublet_prof.py NAME --out OUT"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import doublet_v2 as S  # noqa: E402
from singlet.features import doublet_sdfl as DS  # noqa: E402

X, y, genes, sp_ = S.load_counts(sys.argv[1])
t = time.perf_counter()
pr = cProfile.Profile()
pr.enable()
DS.score(X, ("a", "c"))
pr.disable()
print("wall", round(time.perf_counter() - t, 1), "s for both variants,", X.shape[0], "cells")
st = io.StringIO()
pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(28)
print("\n".join(l[:170] for l in st.getvalue().splitlines() if l.strip()))
