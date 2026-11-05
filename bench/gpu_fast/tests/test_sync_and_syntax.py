import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "..", "jobs")
TP = os.path.join(HERE, "..", "..", "gpu_batch1", "jobs", "tp.sbatch")


def test_identity_block_is_verbatim_from_batch1():
    lib = open(os.path.join(JOBS, "lib_ident.sh"), newline="").read().replace("\r\n", "\n").split("\n")
    tp = open(TP, newline="").read().replace("\r\n", "\n").split("\n")
    block = [l for l in lib if not l.startswith("# Identity helpers") and not l.startswith("# The block below") and not l.startswith("# prune, readhash")]
    start = tp.index(block[0])
    assert tp[start:start + len(block)] == block


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", ["t1.sbatch", "t2.sbatch", "lib_run.sh", "lib_ident.sh", "fast_build.sbatch"])
def test_bash_syntax(name):
    p = os.path.join(JOBS, name)
    if not os.path.exists(p):
        pytest.skip("missing")
    r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
