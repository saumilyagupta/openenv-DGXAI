#!/usr/bin/env python3
"""Pre-submission validation script for EpistemicNav environment.

Runs 5 checks against the server to verify correctness:
  1. Server starts and responds to health check
  2. reset() returns valid observation
  3. step(QUERY) decrements budget
  4. step(COMMIT) returns reward in [0,1] and done=True
  5. Full episode (DummyAgent) completes successfully
"""

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET_ANSI = "\033[0m"


def _pass(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {GREEN}PASS{RESET_ANSI}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {RED}FAIL{RESET_ANSI}  {label}{suffix}")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_proc: Optional[subprocess.Popen] = None


def _cleanup_server() -> None:
    global _server_proc
    if _server_proc is not None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        _server_proc = None


atexit.register(_cleanup_server)


def _handle_signal(signum: int, _frame: object) -> None:
    _cleanup_server()
    sys.exit(1)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_port(preferred: int = 7860) -> int:
    if not _port_in_use(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> subprocess.Popen:
    global _server_proc
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    _server_proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return _server_proc


def _wait_for_server(base_url: str, timeout: float = 15.0) -> bool:
    """Retry GET / with exponential backoff up to *timeout* seconds."""
    deadline = time.monotonic() + timeout
    delay = 0.25
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/", timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 2.0)
    return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_server_starts(base_url: str) -> bool:
    """Check 1: GET / returns 200 with expected JSON."""
    label = "Check 1: Server starts"
    try:
        r = httpx.get(f"{base_url}/", timeout=5.0)
        data = r.json()
        if r.status_code == 200 and data.get("status") == "ok":
            _pass(label, f"status={data['status']}")
            return True
        _fail(label, f"status_code={r.status_code} body={data}")
        return False
    except Exception as exc:
        _fail(label, str(exc))
        return False


def check_reset(base_url: str) -> bool:
    """Check 2: POST /reset returns valid observation."""
    label = "Check 2: reset() works"
    try:
        r = httpx.post(f"{base_url}/reset", json={"task_level": "easy"}, timeout=10.0)
        data = r.json()
        obs = data.get("observation", data)

        claim = obs.get("claim", "")
        budget = obs.get("budget_remaining")
        evidence = obs.get("evidence_gathered")

        errors = []
        if not isinstance(claim, str) or len(claim) == 0:
            errors.append(f"claim empty or not string: {claim!r}")
        if budget != 8:
            errors.append(f"budget_remaining={budget}, expected 8")
        if not isinstance(evidence, list) or len(evidence) != 0:
            errors.append(f"evidence_gathered={evidence}, expected []")

        if errors:
            _fail(label, "; ".join(errors))
            return False
        _pass(label, f"claim={claim!r:.60s} budget={budget}")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


def check_step_query(base_url: str) -> bool:
    """Check 3: POST /step with QUERY decrements budget."""
    label = "Check 3: step(QUERY) works"
    try:
        # Reset first to ensure clean state
        httpx.post(f"{base_url}/reset", json={"task_level": "easy"}, timeout=10.0)

        r = httpx.post(
            f"{base_url}/step",
            json={"action": {"action_type": "query", "query_text": "test evidence search"}},
            timeout=10.0,
        )
        data = r.json()
        obs = data.get("observation", data)

        budget = obs.get("budget_remaining")
        evidence = obs.get("evidence_gathered")

        errors = []
        if budget != 7:
            errors.append(f"budget_remaining={budget}, expected 7")
        if not isinstance(evidence, list):
            errors.append(f"evidence_gathered is not a list: {type(evidence)}")

        if errors:
            _fail(label, "; ".join(errors))
            return False
        _pass(label, f"budget={budget} evidence_count={len(evidence)}")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


def check_step_commit(base_url: str) -> bool:
    """Check 4: POST /step with COMMIT returns reward in [0,1] and done=True."""
    label = "Check 4: step(COMMIT) reward in [0,1]"
    try:
        # Reset first to ensure clean state
        httpx.post(f"{base_url}/reset", json={"task_level": "easy"}, timeout=10.0)

        r = httpx.post(
            f"{base_url}/step",
            json={"action": {"action_type": "commit", "verdict": "true", "confidence": 0.8}},
            timeout=10.0,
        )
        data = r.json()

        reward = data.get("reward")
        done = data.get("done")

        errors = []
        if not isinstance(reward, (int, float)):
            errors.append(f"reward is not a number: {reward!r}")
        elif not (0.0 <= reward <= 1.0):
            errors.append(f"reward={reward} not in [0.0, 1.0]")
        if done is not True:
            errors.append(f"done={done}, expected True")

        if errors:
            _fail(label, "; ".join(errors))
            return False
        _pass(label, f"reward={reward:.4f} done={done}")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


def check_full_episode(base_url: str) -> bool:
    """Check 5: Full episode (DummyAgent) completes successfully."""
    label = "Check 5: Full episode (DummyAgent)"
    try:
        # Step 0: reset with medium difficulty
        r = httpx.post(f"{base_url}/reset", json={"task_level": "medium"}, timeout=10.0)
        reset_data = r.json()
        obs = reset_data.get("observation", reset_data)
        claim = obs.get("claim", "unknown claim")

        # Step 1: QUERY using the claim text
        r = httpx.post(
            f"{base_url}/step",
            json={"action": {"action_type": "query", "query_text": claim}},
            timeout=10.0,
        )
        step1 = r.json()
        step1_done = step1.get("done", False)
        if step1_done:
            _fail(label, "episode ended after QUERY step (unexpected)")
            return False

        # Step 2: COMMIT with verdict=uncertain, confidence=0.5
        r = httpx.post(
            f"{base_url}/step",
            json={"action": {"action_type": "commit", "verdict": "uncertain", "confidence": 0.5}},
            timeout=10.0,
        )
        step2 = r.json()
        reward = step2.get("reward")
        done = step2.get("done")

        errors = []
        if done is not True:
            errors.append(f"done={done}, expected True")
        if not isinstance(reward, (int, float)):
            errors.append(f"reward is not a number: {reward!r}")
        elif not (0.0 <= reward <= 1.0):
            errors.append(f"reward={reward} not in [0.0, 1.0]")

        if errors:
            _fail(label, "; ".join(errors))
            return False
        _pass(label, f"reward={reward:.4f} done={done}")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EpistemicNav environment")
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Assume the server is already running on port 7860",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to use (default: 7860)",
    )
    args = parser.parse_args()

    port = args.port
    base_url = f"http://localhost:{port}"

    print(f"\n{BOLD}EpistemicNav Validation{RESET_ANSI}")
    print("=" * 40)

    if args.skip_server:
        print(f"Using existing server at {base_url}")
        if not _wait_for_server(base_url, timeout=5.0):
            print(f"{RED}ERROR: No server responding at {base_url}{RESET_ANSI}")
            sys.exit(1)
    else:
        port = _pick_port(port)
        base_url = f"http://localhost:{port}"
        print(f"Starting server on port {port} ...")
        _start_server(port)
        if not _wait_for_server(base_url, timeout=15.0):
            print(f"{RED}ERROR: Server did not start within 15 seconds{RESET_ANSI}")
            _cleanup_server()
            sys.exit(1)
        print(f"Server ready at {base_url}\n")

    results = [
        check_server_starts(base_url),
        check_reset(base_url),
        check_step_query(base_url),
        check_step_commit(base_url),
        check_full_episode(base_url),
    ]

    _cleanup_server()

    passed = sum(results)
    total = len(results)
    print()
    print("=" * 40)
    if passed == total:
        print(f"{GREEN}{BOLD}All {total}/{total} checks passed.{RESET_ANSI}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}{passed}/{total} checks passed.{RESET_ANSI}")
        sys.exit(1)


if __name__ == "__main__":
    main()
