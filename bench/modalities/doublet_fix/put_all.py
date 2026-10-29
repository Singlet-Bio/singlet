"""Upload the harness + the doublet_detect.h under test to Clipper ($W/jobs/modfix-doublet). Usage: python put_all.py"""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
inc = os.path.abspath(os.path.join(here, "..", "..", "..", "include", "singlet", "pileup"))
clip = os.path.expanduser("~/.claude/skills/clipper/clipper.py")
J = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modfix-doublet"
def run(*a):
    r = subprocess.run([sys.executable, clip, *a], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
run("run", f"mkdir -p {J}/inc")
import glob
for f in ["dbl_run.cpp", "kang_prep.py", "kang_eval.py", "hgmm_prep.py", "hgmm_eval.py", "dbl_fix.sbatch", "dbl_sweep.sbatch"] + \
         [os.path.basename(p) for p in glob.glob(os.path.join(here, "sweep*.txt"))]:
    run("put", os.path.join(here, f), f"{J}/{f}")
for f in ["doublet_detect.h", "sparse_accumulator.h"]:
    run("put", os.path.join(inc, f), f"{J}/inc/{f}")
