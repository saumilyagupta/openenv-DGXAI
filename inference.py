from __future__ import annotations

import json
import logging
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

_LOG = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MAX_ITERS_PER_TASK = 3
TIMEOUT_S = 120.0

_STUB_SOLUTIONS: dict[str, dict[str, str]] = {
    "easy": {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    },
    "medium": {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str | None) -> str:\n"
            "    if name is None:\n"
            "        msg = \"name must not be None\"\n"
            "        raise ValueError(msg)\n"
            "    return f\"Hello, {name}!\"\n"
        ),
        "test_main.py": (
            "from __future__ import annotations\n\n"
            "import pytest\n\n"
            "from main import greet\n\n\n"
            "def test_greet_hello() -> None:\n"
            "    assert greet(\"Alice\") == \"Hello, Alice!\"\n\n\n"
            "def test_greet_none_raises() -> None:\n"
            "    with pytest.raises(ValueError):\n"
            "        greet(None)\n"
        ),
    },
    "hard": {
        "main.py": (
            "from __future__ import annotations\n\nfrom core import greet\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    print(greet(\"World\"))\n"
        ),
        "core.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
        "test_core.py": (
            "from __future__ import annotations\n\n"
            "from core import greet\n\n\n"
            "def test_greet() -> None:\n"
            "    assert greet(\"World\") == \"Hello, World!\"\n"
        ),
    },
}


def _run_task(client: httpx.Client, level: str) -> dict[str, float | str | int | bool]:
    reset_resp = client.post(f"{API_BASE_URL}/reset", json={"task_level": level})
    reset_resp.raise_for_status()
    reset_body = reset_resp.json()
    obs = reset_body.get("observation", reset_body)

    query = {"action_type": "query_kb", "claim": obs.get("task_brief", level), "top_k": 3}
    q_resp = client.post(f"{API_BASE_URL}/step", json={"action": query})
    q_resp.raise_for_status()

    submit = {"action_type": "submit", "files": _STUB_SOLUTIONS[level]}
    s_resp = client.post(f"{API_BASE_URL}/step", json={"action": submit})
    s_resp.raise_for_status()
    final_body = s_resp.json()
    final = final_body.get("observation", final_body)

    return {
        "task_level": level,
        "reward": final.get("last_reward", 0.0),
        "done": final.get("is_done", False),
        "citations_seen": len(final.get("last_citations", ())),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    results: list[dict] = []
    t0 = time.monotonic()
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for level in ("easy", "medium", "hard"):
            result = _run_task(client, level)
            results.append(result)
            _LOG.info("%s -> reward=%.3f done=%s", level, result["reward"], result["done"])
    _LOG.info("total_time=%.2fs", time.monotonic() - t0)
    print(json.dumps({"baseline": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
