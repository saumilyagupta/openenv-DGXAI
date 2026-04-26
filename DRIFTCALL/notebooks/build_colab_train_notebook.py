"""Builder for the Colab clone-and-train notebook.

Produces ``notebooks/colab_clone_and_train.ipynb`` — a self-contained Colab
notebook that:

  1. Clones the public DriftCall repo from GitHub.
  2. Installs pinned deps (Unsloth + TRL + torch + audio + HF Hub).
  3. Authenticates with Hugging Face Hub.
  4. Runs ``scripts/train_driftcall_grpo.py`` for one curriculum stage.
  5. Pushes the trained LoRA to ``DGXAI/gemma-3n-e2b-driftcall-lora``.

The notebook is meant to be opened directly from the deployed Space at
``https://huggingface.co/spaces/saumilyajj/driftcall/blob/main/notebooks/
colab_clone_and_train.ipynb`` and "Open in Colab".

Re-runs are byte-identical given the same inputs; metadata is fixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import nbformat

_nb_v4: Any = nbformat.v4

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH: Final[Path] = (
    _REPO_ROOT / "notebooks" / "colab_clone_and_train.ipynb"
)

GITHUB_HTTPS: Final[str] = "https://github.com/saumilyagupta/openenv-DGXAI.git"
GITHUB_BRANCH: Final[str] = "main"
LORA_REPO: Final[str] = "DGXAI/gemma-3n-e2b-driftcall-lora"
SPACE_HF: Final[str] = "saumilyajj/driftcall"

_NOTEBOOK_METADATA: Final[dict[str, object]] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}


_INTRO_MD: Final[str] = f"""\
# DriftCall — Clone & Train (Colab)

> One-cell-per-step Colab notebook that clones the DriftCall repo, installs
> dependencies, runs a real GRPO training stage on the Gemma-3n-E2B base
> model, and pushes the trained LoRA back to the Hugging Face Hub.

| | |
|---|---|
| Repo | [saumilyagupta/openenv-DGXAI]({GITHUB_HTTPS.removesuffix(".git")}) |
| Branch | `{GITHUB_BRANCH}` |
| Trained adapter target | [`{LORA_REPO}`](https://huggingface.co/{LORA_REPO}) |
| Live Space | [`{SPACE_HF}`](https://huggingface.co/spaces/{SPACE_HF}) |
| Recommended hardware | Colab `T4` (free) — works; `A100` (Pro+) — fast |

**Before you run cell §03**, make sure your Hugging Face token has
`write` access to `{LORA_REPO.split("/")[0]}/*` or change `LORA_REPO`
in §05 to a namespace you own.
"""


_CLONE_CODE: Final[str] = f"""\
# §01 — Clone the DriftCall repo
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "{GITHUB_HTTPS}"
REPO_BRANCH = "{GITHUB_BRANCH}"
WORKDIR = Path("/content/openenv-DGXAI")
DRIFTCALL_DIR = WORKDIR / "DRIFTCALL"

if WORKDIR.exists():
    print(f"[clone] {{WORKDIR}} already exists — pulling latest")
    subprocess.run(["git", "-C", str(WORKDIR), "fetch", "--all"], check=True)
    subprocess.run(
        ["git", "-C", str(WORKDIR), "checkout", REPO_BRANCH], check=True
    )
    subprocess.run(
        ["git", "-C", str(WORKDIR), "reset", "--hard", f"origin/{{REPO_BRANCH}}"],
        check=True,
    )
else:
    subprocess.run(
        ["git", "clone", "--branch", REPO_BRANCH, "--depth", "1",
         REPO_URL, str(WORKDIR)],
        check=True,
    )

assert DRIFTCALL_DIR.exists(), f"DRIFTCALL/ not found under {{WORKDIR}}"
os.chdir(DRIFTCALL_DIR)
sys.path.insert(0, str(DRIFTCALL_DIR))
print(f"[clone] cwd = {{Path.cwd()}}")
print(f"[clone] head = " + subprocess.check_output(
    ["git", "-C", str(WORKDIR), "rev-parse", "--short", "HEAD"], text=True
).strip())
"""


_INSTALL_CODE: Final[str] = """\
# §02 — Install pinned dependencies
# Heavy: torch + unsloth + trl + transformers + faster-whisper + kokoro.
# Expect ~5 min on Colab T4.
import subprocess
import sys

PIP_PINS = [
    "torch>=2.5,<3.0",
    "transformers>=4.46,<5.0",
    "trl>=0.23,<0.25",
    "unsloth==2026.4.8",
    "unsloth-zoo>=2026.4.5",
    "datasets>=3.0",
    "accelerate>=1.1",
    "peft>=0.13",
    "bitsandbytes>=0.45",
    "huggingface-hub>=0.27",
    "soundfile>=0.12",
    "librosa>=0.10",
    "rapidfuzz>=3.10",
]

# Quiet install — uncomment "-v" if a wheel breaks.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--upgrade", "--no-cache-dir", *PIP_PINS],
    check=True,
)

# Verify GPU is wired.
import torch
print(f"[install] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[install] device 0 = {torch.cuda.get_device_name(0)}")
else:
    print("[install] WARNING: no GPU — switch Runtime → GPU before §04")
"""


_AUTH_CODE: Final[str] = """\
# §03 — Hugging Face authentication
# Paste a HF token with write access to the org you want to push the LoRA to.
import os
from huggingface_hub import login, whoami

# Colab way (interactive):
try:
    from google.colab import userdata  # type: ignore[import-not-found]
    HF_TOKEN = userdata.get("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
except Exception:
    pass

if "HF_TOKEN" not in os.environ:
    from getpass import getpass
    os.environ["HF_TOKEN"] = getpass("HF_TOKEN (write-scope): ").strip()

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print(f"[auth] logged in as {whoami()['name']}")

# Wandb is optional and disabled by default to avoid extra prompts.
os.environ["WANDB_MODE"] = "disabled"
"""


_TRAIN_CODE: Final[str] = """\
# §04 — Train one GRPO stage
# Defaults to Stage 2 (single-drift, mixed languages) for a balanced demo.
# Increase NUM_STEPS / change STAGE for a fuller run.
import subprocess
import sys
from pathlib import Path

STAGE = 2          # 1 = warmup, 2 = single drift, 3 = compound drift
NUM_STEPS = 30     # bump to 150–200 for a real curriculum stage
NUM_GENERATIONS = 2  # G in GRPO; 8 is canonical, 2 keeps Colab T4 happy
HARDWARE = "h100"   # the script reads this for its own dtype/precision picks
OUTPUT_DIR = Path("/content/openenv-DGXAI/DRIFTCALL/checkpoints/colab/final")
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "scripts/train_driftcall_grpo.py",
    "--stage", str(STAGE),
    "--num-steps", str(NUM_STEPS),
    "--num-generations", str(NUM_GENERATIONS),
    "--hardware", HARDWARE,
    "--output-dir", str(OUTPUT_DIR),
]
print("[train] running:", " ".join(cmd))

# Stream stdout/stderr live so the Colab user sees [train] step= lines tick.
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
assert proc.stdout is not None
try:
    for line in proc.stdout:
        print(line, end="")
finally:
    proc.wait()
print(f"[train] exit_code={proc.returncode}")
assert proc.returncode == 0, f"trainer failed with exit code {proc.returncode}"
"""


_PUSH_CODE: Final[str] = f"""\
# §05 — Push the trained LoRA back to the Hub
# This force-uploads the contents of OUTPUT_DIR to LORA_REPO.
# Change LORA_REPO if you don't have write access to the default org.
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

LORA_REPO = "{LORA_REPO}"
OUTPUT_DIR = Path("/content/openenv-DGXAI/DRIFTCALL/checkpoints/colab/final")

assert OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()), (
    f"no checkpoint at {{OUTPUT_DIR}} — did §04 finish cleanly?"
)

api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(LORA_REPO, repo_type="model", exist_ok=True, private=False,
            token=os.environ["HF_TOKEN"])

print(f"[push] uploading {{OUTPUT_DIR}} → https://huggingface.co/{{LORA_REPO}}")
api.upload_folder(
    folder_path=str(OUTPUT_DIR),
    repo_id=LORA_REPO,
    repo_type="model",
    commit_message="colab: clone-and-train run via build_colab_train_notebook.py",
)
print(f"[push] done. browse → https://huggingface.co/{{LORA_REPO}}")
"""


_FOOTER_MD: Final[str] = f"""\
---

### Done.

You just trained a real GRPO stage end-to-end and pushed the resulting
adapter to the Hub. The trained LoRA is live at
[`{LORA_REPO}`](https://huggingface.co/{LORA_REPO}) and any Space that
loads `unsloth/gemma-3n-E2B-it` + this LoRA will pick it up.

> **Want to feel the impact?** Open
> [`{SPACE_HF}`](https://huggingface.co/spaces/{SPACE_HF}), hit `/demo/`,
> and toggle between *base* and *trained* in the checkpoint radio.
"""


def build_notebook(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    nb: Any = _nb_v4.new_notebook()
    nb.cells = [
        _nb_v4.new_markdown_cell(_INTRO_MD),
        _nb_v4.new_code_cell(_CLONE_CODE),
        _nb_v4.new_code_cell(_INSTALL_CODE),
        _nb_v4.new_code_cell(_AUTH_CODE),
        _nb_v4.new_code_cell(_TRAIN_CODE),
        _nb_v4.new_code_cell(_PUSH_CODE),
        _nb_v4.new_markdown_cell(_FOOTER_MD),
    ]
    nb.metadata = _NOTEBOOK_METADATA  # type: ignore[assignment]

    # Strip volatile per-cell metadata so successive builds are byte-identical.
    for cell in nb.cells:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return output_path


if __name__ == "__main__":
    out = build_notebook()
    print(f"wrote {out}")
