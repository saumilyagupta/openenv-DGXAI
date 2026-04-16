from __future__ import annotations

"""OpenEnv server entry point."""

import uvicorn

from codeforge.app import app

__all__ = ["app"]


def main() -> None:
    """Start the CodeForge server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
