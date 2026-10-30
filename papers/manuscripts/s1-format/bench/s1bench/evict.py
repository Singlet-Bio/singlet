"""`python -m s1bench.evict PATH...`: drop files from the page cache (used by the R timings)."""

import sys
from pathlib import Path

from .util import evict

for a in sys.argv[1:]:
    evict(Path(a))
