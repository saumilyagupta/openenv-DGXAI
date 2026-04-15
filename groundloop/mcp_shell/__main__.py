from __future__ import annotations

import asyncio

from groundloop.mcp_shell.server import _run_server


def main() -> int:
    asyncio.run(_run_server())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
