# SPDX-License-Identifier: MIT
"""Assemble longread_usa.json, the long-read artifact of the spliced-unspliced capability, from the two scoring outputs.

    python longread_usa.py SCORE_USA.json SCORE_MUSC.json OUT_DIR

SCORE_USA.json   expression-sota/longread_usa/code/score_usa.py on the PBMC-E truth (Singlet, STARsolo Velocyto, alevin-fry USA).
SCORE_MUSC.json  s8 score_musc.py on the MuSC dev cells (Singlet's molecule dump against the long-read molecule truth);
                 the rule set under test is named by --rules (default r397, the shipped rule set).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("score_usa")
ap.add_argument("score_musc")
ap.add_argument("out_dir")
ap.add_argument("--rules", default="r397")
a = ap.parse_args()
usa = json.loads(Path(a.score_usa).read_text())
musc = json.loads(Path(a.score_musc).read_text())
out = {"pbmc_e": usa, "musc_dev": {"truth_cells": musc["truth_cells"], "rules": a.rules, "dev": musc[a.rules]["dev"],
                                   "truth_classes": musc.get("truth_classes", {})}}
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
(Path(a.out_dir) / "longread_usa.json").write_text(json.dumps(out))
print("wrote", Path(a.out_dir) / "longread_usa.json")
