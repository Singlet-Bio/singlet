# SPDX-License-Identifier: MIT
"""Scrublet at its documented defaults (expected_doublet_rate 0.06, scrub_doublets() with no arguments) on the called cells.

    python -I scrublet_run.py WORK_DIR        reads WORK_DIR/cells.mtx (genes x cells) and cells.csv, writes tools/scrublet.csv

Kang pools use the same entry point on the Singlet filtered matrix (see kang_prep.py), which writes the same two files.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scrublet as scr

work = sys.argv[1]
meta = pd.read_csv(f"{work}/cells.csv")
X = scipy.io.mmread(f"{work}/cells.mtx").T.tocsr().astype(np.float32)  # cells x genes
os.makedirs(f"{work}/tools", exist_ok=True)
s = scr.Scrublet(X, expected_doublet_rate=0.06, random_state=0)  # random_state only fixes the seed; no parameter is tuned
score, call = s.scrub_doublets(verbose=False)
pd.DataFrame({"barcode": meta.barcode, "score": score, "call": np.asarray(call, dtype=bool)}).to_csv(f"{work}/tools/scrublet.csv", index=False)
print({"threshold": float(s.threshold_), "n_called": int(np.sum(call)), "n_cells": X.shape[0]})
