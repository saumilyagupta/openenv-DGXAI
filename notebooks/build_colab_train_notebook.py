"""Builder for the public Colab clone-and-train notebook.

Produces ``notebooks/colab_clone_and_train.ipynb`` — a self-contained Colab
notebook that anyone (with their own HF account) can run end-to-end:

  1. Clone the public DriftCall repo from GitHub.
  2. Install pinned deps (Unsloth + TRL + torch + audio + HF Hub).
  3. Authenticate with Hugging Face Hub via the standard widget
     (``huggingface_hub.notebook_login``).
  4. Run ``scripts/train_driftcall_grpo.py`` for one curriculum stage.
  5. Push the trained LoRA back to **the user's own** namespace
     (``{user}/gemma-3n-e2b-driftcall-lora``), not DGXAI's.

Every code cell is preceded by an explanatory markdown cell so the
notebook reads top-to-bottom for someone who has never seen the project.

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
LORA_NAME: Final[str] = "gemma-3n-e2b-driftcall-lora"
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


# ─────────────────────────────────────────────────────────────────────────
# Markdown cells — one per step, written for someone who has never seen
# the project. Plain English, no jargon, every dependency named.
# ─────────────────────────────────────────────────────────────────────────


_INTRO_MD: Final[str] = f"""\
# DriftCall — Train Your Own Indic Voice Agent (Colab)

> A reproducible, end-to-end training run for the **DriftCall** OpenEnv
> environment. Clone the public repo, install pinned dependencies, run
> a real GRPO stage on Gemma-3n-E2B, and push the resulting LoRA adapter
> to **your own** Hugging Face namespace.

| | |
|---|---|
| Repo | [saumilyagupta/openenv-DGXAI]({GITHUB_HTTPS.removesuffix(".git")}) |
| Branch | `{GITHUB_BRANCH}` |
| Live demo | [`{SPACE_HF}`](https://huggingface.co/spaces/{SPACE_HF}) |
| Recommended hardware | Colab `T4` (free) — works · Colab `A100` (Pro+) — fast |
| Wall time | ~6 min install · ~15 min for the default 30-step run on T4 |

### What you'll do (5 cells, top to bottom)

1. **Clone** the public DriftCall repo into the Colab VM.
2. **Install** torch + Unsloth 2026.4.8 + TRL + the rest of the training stack.
3. **Authenticate** with your Hugging Face account (token never leaves the VM).
4. **Train** one GRPO curriculum stage on the env's mock Indic vendor APIs.
5. **Push** the trained LoRA to `<your-username>/{LORA_NAME}` so anyone
   can load it via `peft.PeftModel.from_pretrained(...)`.

> **You'll need:** a free [Hugging Face account](https://huggingface.co/join)
> and a personal access token with **write** scope (create one at
> [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
> Colab's free T4 GPU is enough — no Pro subscription required.
"""


_CLONE_MD: Final[str] = f"""\
## §01 — Clone the public DriftCall repo

This cell pulls the latest `{GITHUB_BRANCH}` of the public repo into
`/content/openenv-DGXAI` (a fresh shallow clone) and `cd`'s into the
project root. If you re-run the cell after a kernel restart it'll
detect the existing clone and `git pull` instead of re-cloning.

The clone includes everything the trainer needs: the env code under
`cells/`, the training script under `scripts/`, the data fixtures under
`data/`, and the pinned `pyproject.toml`.
"""


_CLONE_CODE: Final[str] = f"""\
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
print(f"[clone] cwd  = {{Path.cwd()}}")
print(f"[clone] head = " + subprocess.check_output(
    ["git", "-C", str(WORKDIR), "rev-parse", "--short", "HEAD"], text=True
).strip())
"""


_INSTALL_MD: Final[str] = """\
## §02 — Install pinned dependencies

Heavy install: `torch`, `transformers`, `trl`, `unsloth==2026.4.8`,
`peft`, `bitsandbytes`, `accelerate`, plus `soundfile`/`librosa` for the
audio boundary. These are the exact pins we used for the published
LoRA — different versions of Unsloth/TRL/transformers will silently
break Gemma-3n training.

Expect **~5 minutes** on a Colab T4. After install the cell verifies
that CUDA is wired and prints the GPU name. If `torch.cuda.is_available()`
returns `False`, switch the runtime to GPU before running §04
(`Runtime → Change runtime type → GPU`).
"""


_INSTALL_CODE: Final[str] = """\
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
print(f"[install] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[install] device 0 = {torch.cuda.get_device_name(0)}")
else:
    print("[install] WARNING: no GPU detected.")
    print("            Runtime → Change runtime type → GPU, then re-run.")
"""


_AUTH_MD: Final[str] = """\
## §03 — Sign in to Hugging Face

Run the cell below and a login widget appears. Paste a personal access
token with **write** scope (create one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
if you don't have one).

The token stays inside this Colab VM — it's never written to the
notebook output, never logged to W&B (we disable W&B by default), and
is wiped when the VM is recycled.

After you log in, the cell echoes back the username it captured. That
username decides where your trained LoRA gets pushed in §05
(`<your-username>/gemma-3n-e2b-driftcall-lora`).
"""


_AUTH_CODE: Final[str] = """\
import os
from huggingface_hub import notebook_login, whoami

# Standard HF widget — interactive paste of a write-scope token.
notebook_login()

# Capture the username so §05 can push to <username>/<lora_name>.
HF_USERNAME = whoami()["name"]
os.environ["HF_USERNAME"] = HF_USERNAME
print(f"[auth] logged in as: {HF_USERNAME}")

# Disable W&B by default so the trainer doesn't prompt for another token.
# Set WANDB_MODE=online + WANDB_API_KEY here if you want metrics dashboards.
os.environ["WANDB_MODE"] = "disabled"
"""


_TRAIN_MD: Final[str] = """\
## §04 — Train one GRPO stage

This is the actual training run. The cell shells out to
`scripts/train_driftcall_grpo.py` — the same self-contained native-PyTorch
GRPO loop the project's training docs (`docs/modules/training.md` §3.2)
describe — and streams its stdout back into the cell so you can watch
`[train] step= …` lines tick live.

**Defaults** (Colab-T4 friendly):
- `STAGE = 2` — single-drift episodes, mixed Indic languages
- `NUM_STEPS = 30` — short enough to fit in T4's free runtime quota
- `NUM_GENERATIONS = 2` — group-relative advantage with G=2 (the paper
  uses G=8 — bump it on A100 for more stable advantages)

Total wall time on T4: **~15 minutes** (model load ~3 min + 30 GRPO
steps ~10–12 min). Bump `NUM_STEPS` to 150–200 for a real
curriculum stage; multiply VRAM headroom on A100/H100 to bump `G`.

The trained adapter is written to `/content/.../checkpoints/colab/final/`
and §05 picks it up from there.
"""


_TRAIN_CODE: Final[str] = """\
import subprocess
import sys
from pathlib import Path

# ── Tweak these to control your run ──────────────────────────────────
STAGE = 2            # 1=warmup (no drift), 2=single drift, 3=compound
NUM_STEPS = 30       # bump to 150–200 for a real stage; 30 = quick demo
NUM_GENERATIONS = 2  # G in GRPO; 2 keeps T4 happy, 8 is canonical
HARDWARE = "h100"    # script reads this for dtype/precision picks
# ────────────────────────────────────────────────────────────────────

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
print("[train] launching:", " ".join(cmd))

# Stream stdout/stderr live so you see [train] step= lines tick.
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


_PUSH_MD: Final[str] = f"""\
## §05 — Push your trained LoRA to the Hub

This uploads the contents of `OUTPUT_DIR` (the LoRA adapter weights +
tokenizer + config) to **your own** Hugging Face namespace at:

```text
<your-username>/{LORA_NAME}
```

If the repo doesn't exist yet, it's created (public, Apache-2.0). If it
already exists, this push **overwrites** its contents — the upload is
the new HEAD revision.

Once it's up, anyone (including you, in another notebook) can load it::

    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3n-E2B-it")
    model = PeftModel.from_pretrained(base, "<your-username>/{LORA_NAME}")

The DriftCall demo Space lets you swap any LoRA in via the *trained*
checkpoint radio — useful for A/B-ing your run against the published
adapter.
"""


_PUSH_CODE: Final[str] = f"""\
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

LORA_NAME = "{LORA_NAME}"
LORA_REPO = f"{{os.environ['HF_USERNAME']}}/{{LORA_NAME}}"
OUTPUT_DIR = Path("/content/openenv-DGXAI/DRIFTCALL/checkpoints/colab/final")

assert OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()), (
    f"no checkpoint at {{OUTPUT_DIR}} — did §04 finish cleanly?"
)

api = HfApi()
create_repo(LORA_REPO, repo_type="model", exist_ok=True, private=False)

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

## Done.

You just trained a real GRPO stage end-to-end and pushed the resulting
adapter to **your** namespace on the Hub. Recap of what happened:

| Step | What you did |
|---|---|
| §01 | Cloned the public DriftCall repo into the Colab VM |
| §02 | Installed the exact pins used for the published model |
| §03 | Authenticated as `{{your-username}}` via the HF widget |
| §04 | Ran `{{NUM_STEPS}}` GRPO steps on Stage `{{STAGE}}` of the curriculum |
| §05 | Uploaded the LoRA to `{{your-username}}/{LORA_NAME}` |

### Want to feel the impact?

Open the live demo Space — [`{SPACE_HF}`](https://huggingface.co/spaces/{SPACE_HF})
— hit `/demo/`, and try the same Indic concierge prompts against
**base** Gemma-3n-E2B vs your **trained** adapter. The difference is
dramatic on schema-drift episodes (renamed fields, hidden fees, MFA
thresholds) — see [`BLOG.md`](https://github.com/saumilyagupta/openenv-DGXAI/blob/main/DRIFTCALL/BLOG.md)
for full before/after transcripts in five languages.

### Going further

- **Bump the curriculum.** Re-run §04 with `STAGE=3` and `NUM_STEPS=150`
  for a full Stage-3 (compound-drift) run on A100.
- **Crank G.** On A100/H100, set `NUM_GENERATIONS=8` for more stable
  group-relative advantages.
- **Log to W&B.** Replace `os.environ["WANDB_MODE"] = "disabled"` in
  §03 with `os.environ["WANDB_API_KEY"] = "<your-key>"` and
  `os.environ["WANDB_MODE"] = "online"` to get the 20-column training
  dashboard described in `docs/modules/training.md` §3.4.
- **Read the spec.** Everything is in
  [`DRIFTCALL/DESIGN.md`](https://github.com/saumilyagupta/openenv-DGXAI/blob/main/DRIFTCALL/DESIGN.md)
  (master spec, locked v1.0).

Apache 2.0. Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon (April 2026).
"""


def build_notebook(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    nb: Any = _nb_v4.new_notebook()
    nb.cells = [
        _nb_v4.new_markdown_cell(_INTRO_MD),
        _nb_v4.new_markdown_cell(_CLONE_MD),
        _nb_v4.new_code_cell(_CLONE_CODE),
        _nb_v4.new_markdown_cell(_INSTALL_MD),
        _nb_v4.new_code_cell(_INSTALL_CODE),
        _nb_v4.new_markdown_cell(_AUTH_MD),
        _nb_v4.new_code_cell(_AUTH_CODE),
        _nb_v4.new_markdown_cell(_TRAIN_MD),
        _nb_v4.new_code_cell(_TRAIN_CODE),
        _nb_v4.new_markdown_cell(_PUSH_MD),
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
