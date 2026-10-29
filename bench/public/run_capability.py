#!/usr/bin/env python3
"""Run a capability's compute function on an artifact directory, validate the result, and optionally write it.

    python bench/public/run_capability.py chromosomal-sex ARTIFACT_DIR --out bench/public/results/chromosomal-sex.json
    python bench/public/run_capability.py chromosomal-sex ARTIFACT_DIR --check bench/public/results/chromosomal-sex.json

``--check`` recomputes and diffs against a published file (what the notebook does). The module is
``bench/public/capabilities/<id with - as _>.py``, or the document's own ``reproduce.module`` when checking.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PUB = Path(__file__).resolve().parent
REPO = PUB.parent.parent


def load_compute(module_path: str | Path, function: str = "compute"):
    p = Path(module_path)
    p = p if p.is_absolute() else REPO / p
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(f"cap_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, function)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capability")
    ap.add_argument("artifact_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=Path, help="published document to diff against")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(PUB))
    import contract as C

    if a.check:
        pub = json.loads(a.check.read_text(encoding="utf-8"))
        fn = load_compute(pub["reproduce"]["module"], pub["reproduce"]["function"])
    else:
        fn = load_compute(PUB / "capabilities" / f"{a.capability.replace('-', '_')}.py")
    doc = fn(a.artifact_dir)
    errs = C.validate(doc, C.load_cohort(PUB / "cohort.tsv"))
    st = C.capability_status(doc)
    print(f"{a.capability}: {'valid' if not errs else 'INVALID'}, {st}")
    for e in errs:
        print("  -", e)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    if a.check:
        d = C.diff(pub, doc)
        print("reproduced exactly" if not d else "\n".join(["DIFFERENCES:"] + d))
        return 1 if d or errs else 0
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
