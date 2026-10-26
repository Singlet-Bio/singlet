# SPDX-License-Identifier: MIT
"""Write and validate ``<cap>.json`` (cap = adt or hto) with the contract and cohort of the bench-public branch.

    CAP=adt python -P bench/adt/contract_doc.py      (in a job; see hpc/pyrun.sbatch)

The contract and cohort are read from the bare clone's ``refs/heads/claude/bench-public-cohort-485cf7`` (that branch is
checked out in its owner's worktree, so it is not fetched here). Artifacts: ``$WORK/bpub/artifacts/<cap>``; the document
goes to ``$WORK/state/adt/results/<cap>.json``.
"""

import os
import subprocess
import sys

W = "/mnt/projects/debruinz_project/debruinz/singlet"
SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CB = "claude/bench-public-cohort-485cf7"
R = W + "/repos/singlet-py.git"
st = W + "/state/adt/contract"
out = W + "/state/adt/results"
os.makedirs(st, exist_ok=True)
os.makedirs(out, exist_ok=True)
for f in ("contract.py", "cohort.tsv"):
    txt = subprocess.run(["git", "-C", R, "show", f"refs/heads/{CB}:bench/public/{f}"], capture_output=True, text=True,
                         check=True).stdout
    with open(f"{st}/{f}", "w") as fh:
        fh.write(txt)
print("contract at", subprocess.run(["git", "-C", R, "rev-parse", f"refs/heads/{CB}"], capture_output=True, text=True).stdout)
os.environ["SINGLET_CONTRACT_PY"] = f"{st}/contract.py"
for cap in os.environ.get("CAP", "adt hto").split():
    doc = f"{out}/{cap}.json"
    subprocess.run([sys.executable, "-P", f"{SRC}/bench/adt/public_{cap}.py", f"{W}/bpub/artifacts/{cap}", doc], check=True,
                   env=os.environ)
    r = subprocess.run([sys.executable, "-P", f"{st}/contract.py", "validate", doc, "--cohort", f"{st}/cohort.tsv"],
                       capture_output=True, text=True)
    print(cap, r.returncode, r.stdout[-3000:], r.stderr[-2000:])
