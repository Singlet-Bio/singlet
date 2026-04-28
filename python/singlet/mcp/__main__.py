"""Allow running with: python -m singlet.mcp"""
from singlet.mcp.server import main
import asyncio

asyncio.run(main())
