#!/usr/bin/env python3
"""Keep the R package's copy of the .1pz codec headers identical to the C++ library's.

The R package must be self-contained, so the codec headers are staged into
``r/inst/include/singlet/``. They are generated, never edited by hand:

    python scripts/sync_r_headers.py          # rewrite the copies
    python scripts/sync_r_headers.py --check  # exit 1 if they differ (CI)

``pz_reader.h`` is copied verbatim. ``pz_writer.h`` is copied without the
``SINGLET_SPARSE_ACCUMULATOR_DEFINED`` guard, because R does not ship the
sparse accumulator (the template under the guard is never instantiated there).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "include" / "singlet" / "pileup"
DST = ROOT / "r" / "inst" / "include" / "singlet"
GUARD_LINES = [
    re.compile(r"^// Only available when sparse_accumulator\.h[^\n]*\n", re.M),
    re.compile(r"^#ifdef SINGLET_SPARSE_ACCUMULATOR_DEFINED\n", re.M),
    re.compile(r"^\n#endif  // SINGLET_SPARSE_ACCUMULATOR_DEFINED\n", re.M),
]


def render(name: str) -> str:
    text = (SRC / name).read_text(encoding="utf-8")
    if name == "pz_writer.h":
        for pat in GUARD_LINES:
            text, n = pat.subn("", text)
            if n != 1:
                sys.exit("pz_writer.h: sparse-accumulator guard not found as expected")
    return text


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for name in ("pz_reader.h", "pz_writer.h"):
        want = render(name)
        path = DST / name
        have = path.read_text(encoding="utf-8") if path.exists() else None
        if have != want:
            stale.append(name)
            if not check:
                path.write_text(want, encoding="utf-8", newline="\n")
    if check and stale:
        print("R header copies are stale: " + ", ".join(stale) +
              "\nrun: python scripts/sync_r_headers.py", file=sys.stderr)
        return 1
    print("R headers in sync" if check else "updated: " + (", ".join(stale) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
