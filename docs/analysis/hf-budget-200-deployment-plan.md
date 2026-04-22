# HF Spaces Deployment + $200 Credit Budget — CodeForge Training Plan

**Date:** 2026-04-22
**Hard constraint:** total HuggingFace credit budget = **$200 USD** (covers env hosting + all fine-tuning runs).
**Deployment target:** HF Spaces (env + MCP server + training worker).
**Note:** Hackathon onsite compute credits (25–26th) are **separate** per `docs/hackthondocs/` — this $200 is for pre-onsite prep + prod hosting.
**Sibling docs:** `theme2-rl-training-design.md`, `theme2-rl-training-design-critique.md`, `research-paper-alignment-analysis.md`, `hackathon-alignment-analysis.md`.

---

## 0. TL;DR

- Env hosting: **$0** (CPU Basic Space — sandbox is CPU-bound).
- Training: **$135** across SFT warmup + GRPO smoke + GRPO main + ablations + 7B scale-up.
- Reserve: **$65** (retrain buffer, PRM stretch, Mercor bonus exp).
- Stack: Qwen-Coder-1.5B on A10G Small for 90% of runs, A100 Large only for final 7B scale.
- vLLM rollout mandatory → 3–5× hour compression → fits budget with margin.
- HF training happens in **dedicated Training Space** separate from env Space (env must stay up for demo).

---

## 1. HF Pricing Reference (confirm before launch)

**Confirm current pricing at** `https://huggingface.co/pricing` — numbers below are my estimate as of 2026-01 and may have drifted.

| Instance | Rate (est.) | VRAM | Use case |
|---|---|---|---|
| CPU Basic | $0.00 / hr | — | Env Space (sandbox, BGE embed, MCP) |
| CPU Upgrade | $0.03 / hr | — | Env if CPU Basic OOM |
| Nvidia T4 small | $0.40 / hr | 16 GB | Training 1.5B QLoRA (tight) |
| Nvidia T4 medium | $0.60 / hr | 16 GB | Slight headroom |
| **Nvidia A10G small** | **$1.05 / hr** | **24 GB** | **Primary training instance — 1.5B GRPO + vLLM** |
| Nvidia A10G large | $1.50 / hr | 48 GB | 7B QLoRA GRPO (no vLLM) |
| Nvidia A100 Large | $4.13 / hr | 80 GB | 7B + vLLM + G=8 rollouts |
| Nvidia H100 | ~$8 / hr | 80 GB | Overkill for hackathon scale |
| ZeroGPU | free, throttled | shared A100 | Demos only, 15 s–2 min burst |

**Uncertainty flag:** Confirm prices on HF billing dashboard before running. Plan below has 33% slack for drift.

---

## 2. Architecture on HF Spaces

Two Spaces, separate billing lines.

### 2.1 Env Space — `krrishchoudhary109/code-forge`

| Component | Instance | Cost |
|---|---|---|
| FastAPI on 7860 | CPU Basic | $0.00 |
| MCP SSE on 7861 | same process | — |
| Sandbox: ruff + mypy + pytest | CPU Basic | — |
| BGE-small embedder (33M, CPU) | CPU Basic | — |
| Corpus baked into Docker image | — | — |
| Auto-sleep after 48h idle | HF default | — |

**30-day hosting cost: $0.** Verify sandbox latency on CPU Basic is acceptable (<5 s per `/step`). If not, upgrade to CPU Upgrade $0.03 × 720h = **$21.60/mo** — still within budget.

### 2.2 Training Space — `krrishchoudhary109/code-forge-training`

Separate Space. Persistent GPU with `app.py` that runs TRL GRPOTrainer. Started only when training is active, stopped after.

| Run | Instance | Hours (est.) | Cost |
|---|---|---|---|
| SFT warmup (200 traj, 1 epoch) | A10G small | 2 | $2.10 |
| GRPO smoke (100 steps, 1.5B) | A10G small | 4 | $4.20 |
| GRPO main (500 steps, 1.5B) | A10G small | 12 | $12.60 |
| Ablation suite × 4 (env-only, +format, +cite, +PRM) | A10G small | 24 | $25.20 |
| 7B scale-up (final run, vLLM) | A100 Large | 10 | $41.30 |
| Retrain buffer (expected ≥1 failed run) | A100 Large | 10 | $41.30 |
| Eval + held-out sweep | A10G small | 8 | $8.40 |
| **Subtotal** | | **70 h** | **$135.10** |
| Reserve (PRM stretch, Mercor exp, debug) | flex | — | **$64.90** |
| **Total** | | | **$200.00** |

---

## 3. Revised Training Strategy Under $200

Forces simplification vs prior doc.

| Decision | Prior doc said | Revised (budget-driven) | Why |
|---|---|---|---|
| Base model | Qwen-Coder-1.5B primary, 7B stretch | **1.5B for all ablations, 7B for 1 final scale-up run only** | A10G small fits 1.5B comfortably. 7B needs A100 ($4.13/h). One A100 run = $41, four = $165, budget dead |
| Rollout engine | vLLM recommended | **vLLM mandatory — Colab-free path dropped** | Hackathon rubric needs Unsloth/TRL in **Colab** — keep Colab as demo notebook, but real training on HF paid GPU. vLLM cuts 12h → 4h → saves ~$8/run |
| PRM training (Phase 4) | 8 h A100 budgeted | **Deferred to post-hackathon** | PRM labeling alone 10+ h A100 = $41. Drop for $200 plan, mention in pitch as future work |
| DPO warmup (Phase 2) | Optional | **Kept, cheap** — 1.5B DPO = 2 h A10G = $2.10 | Valuable offline warmup from audit ledger |
| Ablation count | 4 runs | **Kept — 4 runs × 6 h A10G = $25.20** | Directly serves rubric 20% (reward curves) |
| Scatter 300-instr task | 1-day ship | **Synth dataset built offline, no training cost** | Design task, generate via Claude API (existing key, outside HF budget) |
| Env on GPU | never proposed | **Confirmed stays CPU** | $0 hosting |

---

## 4. Implementation — Training Space Setup

### 4.1 Directory layout

```
CODEFORGE/training-space/
├── Dockerfile                  # GPU base, vllm + trl + unsloth + wandb
├── app.py                      # Minimal Gradio UI to start/stop training, show logs
├── train_grpo.py               # Actual trainer entry
├── train_sft.py                # SFT warmup entry
├── reward_client.py            # asyncio + keep-alive CodeForge /step_batch client
├── config/
│   ├── grpo_smoke.yaml
│   ├── grpo_main.yaml
│   ├── grpo_ablation_{env,format,cite,prm}.yaml
│   └── grpo_7b_scaleup.yaml
└── requirements.txt
```

### 4.2 `Dockerfile` (Training Space)

```dockerfile
FROM huggingface/transformers-pytorch-gpu:4.47.0

# vLLM for fast rollout
RUN pip install --no-cache-dir \
    vllm==0.6.3 \
    trl==0.12.1 \
    unsloth==2025.1.5 \
    peft==0.13.0 \
    datasets==3.0.0 \
    accelerate==1.0.0 \
    bitsandbytes==0.44.1 \
    httpx==0.27.0 \
    wandb==0.18.0 \
    pydantic==2.9.0

WORKDIR /app
COPY . .

# Expose Gradio control panel
EXPOSE 7860
CMD ["python", "app.py"]
```

### 4.3 `reward_client.py` (fixes V2 / H2 / V4 / H4 from critique doc)

```python
from __future__ import annotations
import asyncio
import httpx
from typing import Any

class CodeForgeClient:
    def __init__(self, url: str, max_concurrent: int = 10, timeout: float = 180.0) -> None:
        self.url = url.rstrip("/")
        self.sem = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self._episode_cache: dict[str, str] = {}  # task_id -> episode_id

    async def reset(self, task_id: str, client: httpx.AsyncClient) -> str:
        r = await client.post(f"{self.url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        eid = r.json()["episode_id"]
        self._episode_cache[task_id] = eid
        return eid

    async def step_with_retry(
        self, task_id: str, action: dict[str, Any], client: httpx.AsyncClient
    ) -> float:
        """Reset-on-stale-session guard (fixes H4)."""
        eid = self._episode_cache.get(task_id) or await self.reset(task_id, client)
        async with self.sem:
            try:
                r = await client.post(
                    f"{self.url}/step",
                    json={"episode_id": eid, "action": action},
                    timeout=self.timeout,
                )
                if r.status_code == 404:
                    eid = await self.reset(task_id, client)
                    r = await client.post(
                        f"{self.url}/step",
                        json={"episode_id": eid, "action": action},
                        timeout=self.timeout,
                    )
                r.raise_for_status()
                return float(r.json()["reward"])
            except Exception as e:
                print(f"[reward] {task_id} failed: {e!r}")
                return 0.0

    async def step_batch(
        self, submissions: list[tuple[str, dict[str, Any]]]
    ) -> list[float]:
        """asyncio.gather — fixes V2 / H2."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self.step_with_retry(tid, act, client) for tid, act in submissions]
            return await asyncio.gather(*tasks, return_exceptions=False)
```

### 4.4 `train_grpo.py` (fixes V3 / H3 / V6 / V7 / H1 from critique doc)

```python
from __future__ import annotations
import asyncio
import os
import re
from typing import Any

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from reward_client import CodeForgeClient

CODEFORGE_URL = os.environ["CODEFORGE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FILE_RE = re.compile(r'<file path="(?P<path>[^"]+)">(?P<body>.*?)</file>', re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def parse_submit_action(completion: str) -> tuple[dict[str, Any], bool]:
    """Returns (action, format_ok)."""
    files: dict[str, str] = {}
    m = _ANSWER_RE.search(completion)
    if m:
        for fm in _FILE_RE.finditer(m.group(1)):
            files[fm.group("path")] = fm.group("body")
    if not files:
        fences = _FENCE_RE.findall(completion)
        if fences:
            files["solution.py"] = fences[0]
    return (
        {"type": "submit", "files": files, "confidence": 0.7},
        bool(files) and m is not None,
    )


def make_reward_fn(client: CodeForgeClient, format_bonus: float = 0.05):
    def reward_fn(prompts, completions, task_ids, **_) -> list[float]:
        parsed = [parse_submit_action(c) for c in completions]
        submissions = [(tid, act) for tid, (act, _) in zip(task_ids, parsed)]
        rewards = asyncio.run(client.step_batch(submissions))
        return [
            r + (format_bonus if ok else 0.0)
            for r, (_, ok) in zip(rewards, parsed)
        ]
    return reward_fn


def build_curriculum(client_sync_reset) -> Dataset:
    """Fixes H1 — weighted difficulty curriculum."""
    rows: list[dict[str, Any]] = []
    task_weights = {
        "multi_file_module": 50,   # hard — anchor variance
        "greet_with_tests": 30,    # medium
        "greet_single_file": 20,   # easy
    }
    for tid, n in task_weights.items():
        for _ in range(n):
            rows.append({"prompt": build_prompt(tid), "task_ids": tid})
    return Dataset.from_list(rows)


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    client = CodeForgeClient(CODEFORGE_URL, max_concurrent=10)
    train_ds = build_curriculum(client)

    cfg = GRPOConfig(
        output_dir="codeforge-grpo",
        learning_rate=5e-6,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        beta=0.04,
        temperature=0.9,
        use_vllm=USE_VLLM,
        vllm_gpu_memory_utilization=0.6,  # leave room for policy
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        reward_funcs=[make_reward_fn(client)],
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model("codeforge-grpo-final")


def build_prompt(task_id: str) -> str:
    return (
        f"You are a Python engineer. Task: {task_id}.\n"
        "Output exactly:\n"
        '<answer>\n<file path="PATH">CODE</file>\n... more files ...\n</answer>'
    )


if __name__ == "__main__":
    main()
```

### 4.5 `train_sft.py` (fixes V3 / H3 via cold-start)

```python
from __future__ import annotations
import os
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
SFT_DATA_JSONL = os.environ["SFT_DATA_JSONL"]   # trajectories filtered quality >= 0.8


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    rows = Dataset.from_json(SFT_DATA_JSONL)  # {"prompt": ..., "completion": ...}
    cfg = SFTConfig(
        output_dir="codeforge-sft-warmup",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_length=4096,
        logging_steps=1,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=rows)
    trainer.train()
    trainer.save_model("codeforge-sft-final")


if __name__ == "__main__":
    main()
```

---

## 5. Cost-Saving Playbook

| Move | Saves | How |
|---|---|---|
| **vLLM rollout** | ~60% of GRPO hours | `use_vllm=True` in `GRPOConfig`. 3–5× throughput |
| **Stop Training Space when idle** | 100% of non-running time | HF dashboard "Pause Space". Resume = ~2 min |
| **CPU Basic for env Space** | $21.60/mo vs CPU Upgrade | Only upgrade if latency profile fails |
| **1.5B for ablations, 7B for final** | ~$130 vs all-7B | A10G small = 1/4 cost of A100 large |
| **QLoRA 4-bit** | ~50% VRAM → smaller instance | `load_in_4bit=True` in Unsloth |
| **Gradient accumulation 4** | batch memory → fits A10G | `gradient_accumulation_steps=4` |
| **Reuse env Space**, not per-run | fixed cost | Single persistent env |
| **Claude-Haiku-4.5 for SFT data gen** | $0 of HF budget | Existing Anthropic key, ~100 trajectories × $0.50 = ~$50 off-budget |
| **W&B for logging** (free tier) | — | `report_to="wandb"` |
| **Skip PRM MVP** | $41+ | Drop Phase 4, keep as future work |

---

## 6. Pitch-Deck Reward Curves — Under $200

From ablation suite ($25.20), produce 4 curves on one chart:

| Run | Instance | Hours | Cost | What it shows |
|---|---|---|---|---|
| Ablation A — env-only reward | A10G | 6 | $6.30 | Baseline RLVR signal |
| Ablation B — +format reward (§4.4) | A10G | 6 | $6.30 | Parser robustness lift |
| Ablation C — +citation bonus | A10G | 6 | $6.30 | Shaping reward effect (SYSTEM_DESIGN §4.8.4) |
| Ablation D — +hybrid retrieval (BGE) | A10G | 6 | $6.30 | Retrieval signal lift |

**Single pitch slide:** all 4 curves on `x=step, y=reward_mean`. Each curve separates contribution. Directly serves rubric 20% (Showing Improvement).

---

## 7. Timeline Against Budget

30-day pre-onsite timeline. Hours = training hours, not calendar hours.

| Week | Activity | Cost |
|---|---|---|
| W1 | Env Space deploy + P0-1 / P0-5 server patches (see critique doc) | $0 |
| W1 | Claude-Haiku SFT data gen (off-budget, Anthropic key) | $0 HF |
| W2 | SFT warmup run + smoke GRPO run | $6.30 |
| W2 | Main 1.5B GRPO 500-step run | $12.60 |
| W3 | Ablation suite × 4 | $25.20 |
| W3 | Eval on LiveCodeBench held-out + CodeHalu replay | $8.40 |
| W4 | 7B scale-up (final pitch run, A100 + vLLM) | $41.30 |
| W4 | Retrain buffer (probability ≈ 50% needed) | $41.30 |
| Slack | — | $64.90 reserve |

---

## 8. Risks Specific to $200 Budget

| Risk | Probability | Mitigation |
|---|---|---|
| HF price drift (A10G +20%, A100 +30%) | medium | 33% slack built in; re-price at launch |
| 7B run fails at step ~200, needs re-run | medium | Retrain buffer in budget |
| vLLM memory conflict with Unsloth → fallback to HF generate | low | +$20 on final run; reserve covers |
| Env Space sandbox latency → upgrade needed | low | $21.60/mo upgrade fits reserve |
| HF Training Jobs GA'd with different pricing | unknown | If cheaper, use it; if more expensive, stick to persistent Space |
| Spot / preemption on cheaper A10G tier | not applicable | HF Spaces persistent = non-preempt |

---

## 9. What to Cut If Budget Tightens to $150

| Priority | Cut | Savings |
|---|---|---|
| Drop 7B scale-up | $41 | Keep 1.5B only — pitch framed as "1.5B CodeForge baseline" |
| Drop retrain buffer | $41 | Risk: if main run fails, no recovery |
| Drop 2 of 4 ablations | $12 | Reduces rubric 20% story depth |

**$150-plan total:** $93 if cut 7B + 2 ablations. Leaves $57 reserve.

---

## 10. What Stays Off-Budget (free or billed elsewhere)

- **Anthropic API** (Claude-Haiku-4.5 for SFT data gen, ~$50 off-budget, existing key).
- **Onsite hackathon credits** (25–26th) — described in `docs/hackthondocs/` as separate allocation.
- **Colab free tier** — used as backup demo notebook only, not primary training.
- **W&B free tier** — metrics logging.
- **GitHub Actions / HF webhooks** — CI for env Space build.

---

## 11. Decisions Needed From User

Before launching any paid run, confirm:

1. Is **Anthropic API off-budget confirmed**? (SFT data gen costs ~$50 Anthropic credits.)
2. **Which HF org owns the Training Space** — `krrishchoudhary109` or a new hackathon org?
3. **Is vLLM + Unsloth known-good** in your setup, or test-first-on-smoke-run?
4. Should the **env Space auto-pause** (free) or stay always-on (CPU upgrade $21.60/mo) during the 30-day window?

---

## 12. Cross-References

- `theme2-rl-training-design.md` §4 — original Colab skeleton (now revised in §4.3–4.5 above).
- `theme2-rl-training-design-critique.md` P0-1..P0-6 — server + client patches encoded here.
- `research-paper-alignment-analysis.md` §5 — env-side fixes (uncertain-floor, zero-symbol ground) precondition for training.
- `hackathon-alignment-analysis.md` §2 — rubric hole = Pipeline 10% + Improvement 20%, directly addressed by §6 ablations above.
- `CODEFORGE/SYSTEM_DESIGN.md §13` — existing Docker / HF Space deployment config.
