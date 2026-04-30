"""Allow running with: python -m singlet.mcp"""
from singlet.mcp.server import main as _server_main
import asyncio


def main():
    """Entry point for singlet-mcp console script."""
    asyncio.run(_server_main())


if __name__ == "__main__":
    main()
