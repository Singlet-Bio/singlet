# SPDX-License-Identifier: MIT
"""Allow running with: python -m singlet.mcp"""

import asyncio
import sys

from singlet.mcp.server import main as _server_main


def main():
    """Entry point for the singlet-mcp console script."""
    asyncio.run(_server_main())


def deprecated_main():
    """Entry point for the deprecated ``singlet`` console script.

    ``singlet`` is the name of the C++ pipeline binary, so the MCP server now
    installs as ``singlet-mcp``. The old name keeps working for existing MCP
    client configs; the notice goes to stderr because stdout carries the MCP
    protocol.
    """
    print(
        "warning: the 'singlet' command for the MCP server is deprecated; "
        "use 'singlet-mcp' (or 'python -m singlet.mcp').",
        file=sys.stderr,
    )
    main()


if __name__ == "__main__":
    main()
