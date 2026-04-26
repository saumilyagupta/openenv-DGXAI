"""Online GRPO trainer for the unified DriftCall Space.

Spawns the canonical training loop (`scripts/train_driftcall_grpo.py`) as a
subprocess at Space startup, parses its stdout for `[train] step=… reward=…
loss=… kl=… grad=…` lines, and exposes the latest values via a thread-safe
status dict that `/training` reads.

Why subprocess: the canonical training script is the verified GRPO loop —
the same code that produced the LoRA on HF Hub. We don't reimplement RL
inside the FastAPI process; we let the trained-and-tested loop run, scrape
its output, and surface metrics over REST. The subprocess has its own VRAM
budget so it cannot interfere with the Gradio inference path.

Resource notes:
  * Subprocess loads Gemma-3n-E2B + LoRA in bf16 → ~7 GB VRAM
  * /demo Gradio loads its own copy (lazy) → another ~7 GB VRAM
  * On A10G-small (24 GB) both fit comfortably with ~10 GB headroom
  * Hardware: cpu-basic — trainer aborts at startup, status reports CPU mode

Cap: trainer auto-stops after MAX_STEPS so we don't burn GPU hours
forever. Hitting GET /training/start re-launches it.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("online_trainer")

# Default budget for one launched training run. Re-startable via /training/start.
DEFAULT_MAX_STEPS: int = int(os.environ.get("DRIFTCALL_ONLINE_MAX_STEPS", "150"))
DEFAULT_NUM_GENERATIONS: int = int(os.environ.get("DRIFTCALL_ONLINE_G", "2"))
DEFAULT_STAGE: int = int(os.environ.get("DRIFTCALL_ONLINE_STAGE", "2"))
DEFAULT_HARDWARE: str = os.environ.get("DRIFTCALL_HARDWARE", "h100")

OUTPUT_DIR = Path("/tmp/online_lora")
APP_DIR = Path("/app")
TRAIN_SCRIPT = APP_DIR / "scripts" / "train_driftcall_grpo.py"

# Regex matching the step lines emitted by scripts/train_driftcall_grpo.py:
#   [train] step=  17 reward=0.275±0.025 loss=+6.3729 kl=+6.3729 beta=1.0000 lr=...
_STEP_RE = re.compile(
    r"\[train\] step=\s*(?P<step>\d+)\s+"
    r"reward=(?P<reward>[\d.]+)±(?P<reward_std>[\d.]+)\s+"
    r"loss=(?P<loss>[+\-\d.]+)\s+"
    r"kl=(?P<kl>[+\-\d.]+)\s+"
    r"beta=(?P<beta>[\d.]+)\s+"
    r"lr=(?P<lr>[\d.eE+\-]+)\s+"
    r"grad=(?P<grad>[\d.]+)\s+"
    r"ep_s=(?P<ep_s>[\d.]+)"
)


class OnlineTrainer:
    """Lifecycle wrapper around the GRPO subprocess."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._gpu_available: bool | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "boot_complete": False,
            "started_at": None,
            "stopped_at": None,
            "max_steps": DEFAULT_MAX_STEPS,
            "stage": DEFAULT_STAGE,
            "num_generations": DEFAULT_NUM_GENERATIONS,
            "hardware": DEFAULT_HARDWARE,
            # Latest metrics
            "step": 0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "loss": 0.0,
            "kl": 0.0,
            "beta": 0.0,
            "lr": 0.0,
            "grad_norm": 0.0,
            "ep_seconds": 0.0,
            "last_step_at": None,
            # Boot / error state
            "boot_lines": [],   # last 25 stdout lines during boot
            "error": None,
        }

    # ── GPU probe ──────────────────────────────────────────────────────

    def _has_gpu(self) -> bool:
        if self._gpu_available is not None:
            return self._gpu_available
        # Heuristic: nvidia-smi present on the PATH and exits 0.
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            self._gpu_available = False
            return False
        try:
            r = subprocess.run(
                [nvidia_smi, "-L"], capture_output=True, timeout=5, check=False,
            )
            self._gpu_available = r.returncode == 0 and b"GPU" in r.stdout
        except Exception:
            self._gpu_available = False
        return self._gpu_available

    # ── Subprocess management ─────────────────────────────────────────

    def start(self) -> dict[str, Any]:
        """Launch the GRPO subprocess. Idempotent — returns immediately if
        a run is already active."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._status_snapshot()

            if not self._has_gpu():
                self._status["error"] = (
                    "No GPU available on this Space. Upgrade hardware to a10g-small "
                    "in Settings → Hardware (or set hardware: a10g-small in README)."
                )
                self._status["running"] = False
                return self._status_snapshot()

            if not TRAIN_SCRIPT.exists():
                self._status["error"] = f"training script not found at {TRAIN_SCRIPT}"
                return self._status_snapshot()

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3", str(TRAIN_SCRIPT),
                "--stage", str(self._status["stage"]),
                "--num-steps", str(self._status["max_steps"]),
                "--hardware", str(self._status["hardware"]),
                "--num-generations", str(self._status["num_generations"]),
                "--output-dir", str(OUTPUT_DIR),
            ]
            env = os.environ.copy()
            # The training script defaults wandb on; for the live demo we'd
            # rather have a fast subprocess that doesn't depend on a wandb
            # token at startup. Override:
            env.setdefault("WANDB_MODE", "disabled")
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(APP_DIR),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                self._status["error"] = f"failed to spawn trainer: {exc}"
                self._proc = None
                return self._status_snapshot()

            self._status.update(
                running=True,
                boot_complete=False,
                started_at=time.time(),
                stopped_at=None,
                error=None,
                step=0,
                reward_mean=0.0,
                last_step_at=None,
                boot_lines=[],
            )
            self._reader = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader.start()
            logger.info("OnlineTrainer launched: %s", " ".join(cmd))
            return self._status_snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGTERM)
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._status["running"] = False
            self._status["stopped_at"] = time.time()
            return self._status_snapshot()

    def _read_stdout(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for raw in iter(self._proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue
            self._handle_line(line)
        # Subprocess exited.
        with self._lock:
            self._status["running"] = False
            self._status["stopped_at"] = time.time()
            rc = self._proc.poll() if self._proc else None
            if rc is not None and rc != 0 and not self._status.get("error"):
                self._status["error"] = f"trainer exited with rc={rc}"

    def _handle_line(self, line: str) -> None:
        m = _STEP_RE.search(line)
        if m:
            with self._lock:
                gd = m.groupdict()
                self._status.update(
                    boot_complete=True,
                    step=int(gd["step"]),
                    reward_mean=float(gd["reward"]),
                    reward_std=float(gd["reward_std"]),
                    loss=float(gd["loss"]),
                    kl=float(gd["kl"]),
                    beta=float(gd["beta"]),
                    lr=float(gd["lr"]),
                    grad_norm=float(gd["grad"]),
                    ep_seconds=float(gd["ep_s"]),
                    last_step_at=time.time(),
                )
            logger.debug("step=%s reward=%s", gd["step"], gd["reward"])
            return

        # Pre-step boot lines (Unsloth, model load, wandb etc.) — keep last 25.
        with self._lock:
            self._status["boot_lines"].append(line)
            if len(self._status["boot_lines"]) > 25:
                self._status["boot_lines"] = self._status["boot_lines"][-25:]

    # ── Status read ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_snapshot()

    def _status_snapshot(self) -> dict[str, Any]:
        snap = dict(self._status)
        snap["gpu_available"] = bool(self._has_gpu())
        if snap.get("started_at"):
            snap["wall_seconds"] = round(time.time() - snap["started_at"], 1)
        else:
            snap["wall_seconds"] = 0.0
        if snap.get("max_steps"):
            snap["progress"] = round(snap["step"] / snap["max_steps"], 4)
        else:
            snap["progress"] = 0.0
        return snap


# Module-level singleton so /training/* handlers share state across requests.
_singleton: OnlineTrainer | None = None
_singleton_lock = threading.Lock()


def get_online_trainer() -> OnlineTrainer:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = OnlineTrainer()
        return _singleton
