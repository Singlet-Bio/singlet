#!/usr/bin/env python3
"""Keep the R package's copies of the codec headers identical to the C++ library's.

The R package must be self-contained, so the codec headers are staged into
``r/inst/include/singlet/``. They are generated, never edited by hand:

    python scripts/sync_r_headers.py          # rewrite the copies
    python scripts/sync_r_headers.py --check  # exit 1 if they differ (CI)

``pz_reader.h`` is copied verbatim. ``pz_writer.h`` is copied without the
``SINGLET_SPARSE_ACCUMULATOR_DEFINED`` guard, because R does not ship the
sparse accumulator (the template under the guard is never instantiated there).

The v2 matrix codec is staged verbatim too: ``include/singlet/pz2/pz2.h`` and
``include/singlet/pz2/detail/*.h`` to ``r/inst/include/singlet/pz2/``, and the
binding helpers ``src/bindings/common/singlet/pz2_client.h`` to
``r/inst/include/singlet/pz2_client.h``. A file that no longer exists in the
source tree is removed from the staged directory (reported by ``--check``).
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


PZ2_SRC = ROOT / "include" / "singlet" / "pz2"
PZ2_DST = DST / "pz2"
CLIENT_SRC = ROOT / "src" / "bindings" / "common" / "singlet" / "pz2_client.h"


def pz2_files() -> dict:
    """{relative path under r/inst/include/singlet: source path} for the v2 codec headers and the binding helpers."""
    files = {}
    if PZ2_SRC.is_dir():
        for p in sorted(PZ2_SRC.rglob("*.h")):
            files["pz2/" + p.relative_to(PZ2_SRC).as_posix()] = p
    if CLIENT_SRC.is_file():
        files["pz2_client.h"] = CLIENT_SRC
    return files


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
                path.write_bytes(want.encode("utf-8"))  # LF on every OS (write_text has no newline= before 3.10)
    wanted = pz2_files()
    for rel, src in wanted.items():
        want = src.read_text(encoding="utf-8").replace("\r\n", "\n")
        path = DST / rel
        have = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
        if have != want:
            stale.append(rel)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(want.encode("utf-8"))  # LF on every OS (write_text has no newline= before 3.10)
    if wanted and PZ2_DST.is_dir():
        for p in sorted(PZ2_DST.rglob("*.h")):
            rel = "pz2/" + p.relative_to(PZ2_DST).as_posix()
            if rel not in wanted:
                stale.append(rel + " (no longer in include/singlet/pz2)")
                if not check:
                    p.unlink()
    if check and stale:
        print("R header copies are stale: " + ", ".join(stale) +
              "\nrun: python scripts/sync_r_headers.py", file=sys.stderr)
        return 1
    print("R headers in sync" if check else "updated: " + (", ".join(stale) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
