# SPDX-License-Identifier: MIT
"""python -m singlet — print live atlas stats and usage hints.

Stats come from https://singlet.bio/api/stats. When that cannot be reached
(or ``$SINGLET_OFFLINE`` is set) the offline catalog snapshot bundled with the
package is summarised instead, and labelled as such.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

import singlet


def _live_stats(timeout: float = 5.0) -> Optional[dict]:
    """Corpus statistics from the live ``/api/stats`` endpoint, or None."""
    from singlet.find import _api_base, _offline, _request_headers

    if _offline():
        return None
    req = urllib.request.Request(f"{_api_base()}/stats", headers=_request_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except Exception:  # noqa: BLE001 — any failure means "show the snapshot"
        return None
    return payload if isinstance(payload, dict) else None


def _format_live(stats: dict) -> str:
    """One line from a ``/api/stats`` payload (downloadable data only)."""

    def _n(key: str) -> str:
        value = stats.get(key)
        return f"{int(value):,}" if isinstance(value, (int, float)) else "?"

    return (
        f"singlet atlas (live, singlet.bio): {_n('studies_with_files')} studies with files • "
        f"{_n('samples_in_files')} usable samples • {_n('cells_in_files')} cells"
    )


def main() -> None:
    print(f"singlet v{singlet.__version__}")
    stats = _live_stats()
    if stats is not None:
        print(_format_live(stats))
    else:
        print("(offline: summarising the catalog snapshot bundled with this package)")
        print(singlet.summary())
    print()
    print("Quick start:")
    print("  import singlet")
    print('  singlet.find("human PBMC 10x")                  # search studies')
    print('  adata = singlet.load("GSE138867")               # load a study → AnnData')
    print('  adata = singlet.load("GSM4120733")              # one sample of it')
    print('  adata = singlet.load(["GSE138867", "GSE146974"])  # several studies, one AnnData')
    print('  singlet.info("GSE138867")                       # metadata (snapshot, then live)')
    print()
    print("Local MCP server:    singlet-mcp   (pip install 'singlet[mcp]')")
    print("Debug info:          singlet.show_versions()")
    print("Full docs:           https://singlet.bio")


if __name__ == "__main__":
    main()
