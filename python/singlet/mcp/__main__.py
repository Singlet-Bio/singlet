"""Allow running with: python -m singlet.mcp"""

import asyncio

from singlet.mcp.server import main as _server_main


def main():
    """Entry point for singlet-mcp console script."""
    asyncio.run(_server_main())


if __name__ == "__main__":
    main()
