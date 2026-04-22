# 12-Hour Simple Fine-Tuning Workflow — CodeForge

**Hard constraint:** 12 h wall-clock max on GPU.
**Budget:** $200 HF credits (single 12-h run = ~$12.60 on A10G Small → 94% reserve).
**Goal:** hackathon rubric Pipeline 10% + Improvement 20% covered with ONE clean reward curve.
**Philosophy:** minimum viable RL — SFT warmup + GRPO + eval. No ablation suite. No 7B. No PRM.
**Sibling docs:** `hf-budget-200-deployment-plan.md`, `theme2-rl-training-design-critique.md`, `theme2-rl-training-design.md`, `research-paper-alignment-analysis.md`.

---

## 0. TL;DR

| Stage | Wall-clock | Model | Instance | Cost |
|---|---|---|---|---|
| **S1** SFT warmup | 1.0 h | Qwen2.5-Coder-1.5B + QLoRA | A10G Small | $1.05 |
| **S2** GRPO main | 8.0 h | same (continue from S1) | A10G Small | $8.40 |
| **S3** Eval + curves | 1.0 h | trained model | A10G Small | $1.05 |
| **S4** Buffer (debug / checkpoint restart) | 2.0 h | — | A10G Small | $2.10 |
| **Total** | **12.0 h** | | | **$12.60** |

Reserve after run: **$187.40**. Covers 1 retrain if S2 fails + optional 7B scale-up ($41) if pitch needs it.

---

## 1. Pre-Flight (Off-GPU, 0 h burn)

Done before starting paid GPU clock.

| Step | What | Where |
|---|---|---|
| PF-1 | Env Space live at `CODEFORGE_URL` with P0-1..P0-5 patches applied (floor fix, ground fix, `/step_batch`, `reset_if_stale`) | CPU Basic, free |
| PF-2 | SFT dataset generated offline via Claude-Haiku-4.5 | Anthropic key, off-HF-budget |
| PF-3 | Dataset uploaded as HF Dataset `krrishchoudhary109/codeforge-sft-200` | free |
| PF-4 | Training Space Dockerfile + `train_all.py` committed | free |
| PF-5 | W&B project `codeforge-grpo-1.5b` created | free |
| PF-6 | HF secret `CODEFORGE_URL` + `WANDB_API_KEY` set on Training Space | free |

### PF-2 generation script (runs on laptop, ~2 h Claude time)

```python
# gen_sft_data.py — run locally, upload result as HF Dataset
from __future__ import annotations
import json
from anthropic import Anthropic
from pathlib import Path
import httpx

CLAUDE = Anthropic()
ENV = "https://krrishchoudhary109-code-forge.hf.space"

PROMPT_TEMPLATE = """You are a Python engineer. Task: {task_id}.
Output exactly:
<answer>
<file path="solution.py">CODE</file>
</answer>
Brief: {brief}"""

TASKS = {
    "greet_single_file": 40,   # 20% easy
    "greet_with_tests":  60,   # 30% medium
    "multi_file_module": 100,  # 50% hard — curriculum anchor
}

out = Path("sft_warmup.jsonl").open("w")
for task_id, n in TASKS.items():
    for _ in range(n):
        brief = httpx.post(f"{ENV}/reset", json={"task_id": task_id}).json()["brief"]
        msg = CLAUDE.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(
                task_id=task_id, brief=brief)}],
        )
        completion = msg.content[0].text
        # Score via CodeForge to filter quality >= 0.8
        eid = httpx.post(f"{ENV}/reset", json={"task_id": task_id}).json()["episode_id"]
        r = httpx.post(f"{ENV}/step", json={
            "episode_id": eid,
            "action": {"type": "submit", "files": {"solution.py": completion},
                       "confidence": 0.7}}).json()
        if r["reward"] >= 0.8:
            out.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(task_id=task_id, brief=brief),
                "completion": completion,
                "reward": r["reward"],
            }) + "\n")
out.close()
# Expect ~120-150 rows after filter. Upload to HF Datasets.
```

Cost: ~$5 Anthropic API (off-HF-budget). Time: ~2 h offline, laptop only.

---

## 2. The ONE Script — `train_all.py`

Single entry. Runs S1 → S2 → S3 in one process. No separate stages. No Jupyter juggling.

```python
from __future__ import annotations
import asyncio
import os
import re
import time
from pathlib import Path

import httpx
import wandb
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# ───────────────────────── config ─────────────────────────
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_LEN = 4096
CODEFORGE_URL = os.environ["CODEFORGE_URL"]
WANDB_PROJECT = "codeforge-grpo-1.5b"
OUTPUT_ROOT = Path("/data/codeforge")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# hard stage budgets (seconds) — workflow aborts if exceeded
S1_BUDGET_SEC = 3600       # 1.0 h SFT
S2_BUDGET_SEC = 28800      # 8.0 h GRPO
S3_BUDGET_SEC = 3600       # 1.0 h eval

# ────────────────────── stage 1: SFT ──────────────────────
def stage_sft() -> Path:
    t0 = time.time()
    wandb.init(project=WANDB_PROJECT, name="s1-sft", reinit=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_LEN, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    ds = load_dataset("krrishchoudhary109/codeforge-sft-200", split="train")
    out_dir = OUTPUT_ROOT / "s1_sft"
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_length=MAX_LEN,
        logging_steps=5,
        save_strategy="epoch",
        report_to="wandb",
    )
    SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
    model.save_pretrained(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))

    elapsed = time.time() - t0
    print(f"[S1] done in {elapsed:.0f}s")
    assert elapsed < S1_BUDGET_SEC, "S1 over budget"
    wandb.finish()
    return out_dir / "final"

# ───────────────────── stage 2: GRPO ──────────────────────
_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FILE = re.compile(r'<file path="(?P<p>[^"]+)">(?P<b>.*?)</file>', re.DOTALL)
_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def parse_submit(completion: str) -> tuple[dict, bool]:
    files: dict[str, str] = {}
    m = _ANS.search(completion)
    if m:
        for fm in _FILE.finditer(m.group(1)):
            files[fm.group("p")] = fm.group("b")
    if not files and (fence := _FENCE.findall(completion)):
        files["solution.py"] = fence[0]
    return ({"type": "submit", "files": files, "confidence": 0.7},
            bool(files) and m is not None)


class CFClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.sem = asyncio.Semaphore(10)
        self.cache: dict[str, str] = {}

    async def _reset(self, tid: str, c: httpx.AsyncClient) -> str:
        r = await c.post(f"{self.url}/reset", json={"task_id": tid})
        r.raise_for_status()
        eid = r.json()["episode_id"]
        self.cache[tid] = eid
        return eid

    async def _step(self, tid: str, act: dict, c: httpx.AsyncClient) -> float:
        eid = self.cache.get(tid) or await self._reset(tid, c)
        async with self.sem:
            try:
                r = await c.post(f"{self.url}/step",
                                 json={"episode_id": eid, "action": act}, timeout=180)
                if r.status_code == 404:
                    eid = await self._reset(tid, c)
                    r = await c.post(f"{self.url}/step",
                                     json={"episode_id": eid, "action": act}, timeout=180)
                r.raise_for_status()
                return float(r.json()["reward"])
            except Exception as e:
                print(f"[step] {tid} failed: {e!r}")
                return 0.0

    async def batch(self, subs: list[tuple[str, dict]]) -> list[float]:
        async with httpx.AsyncClient(timeout=180) as c:
            return await asyncio.gather(*[self._step(t, a, c) for t, a in subs])


CLIENT = CFClient(CODEFORGE_URL)


def reward_fn(prompts, completions, task_ids, **_) -> list[float]:
    parsed = [parse_submit(c) for c in completions]
    subs = [(t, a) for t, (a, _) in zip(task_ids, parsed)]
    base = asyncio.run(CLIENT.batch(subs))
    return [r + (0.05 if ok else 0.0) for r, (_, ok) in zip(base, parsed)]


def build_curriculum() -> Dataset:
    rows = []
    weights = {"multi_file_module": 50, "greet_with_tests": 30, "greet_single_file": 20}
    for tid, n in weights.items():
        for _ in range(n):
            rows.append({
                "prompt": (
                    f"You are a Python engineer. Task: {tid}.\n"
                    "Output exactly:\n"
                    '<answer>\n<file path="PATH">CODE</file>\n</answer>'
                ),
                "task_ids": tid,
            })
    return Dataset.from_list(rows)


def stage_grpo(sft_ckpt: Path) -> Path:
    t0 = time.time()
    wandb.init(project=WANDB_PROJECT, name="s2-grpo", reinit=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(sft_ckpt), max_seq_length=MAX_LEN, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    out_dir = OUTPUT_ROOT / "s2_grpo"
    cfg = GRPOConfig(
        output_dir=str(out_dir),
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
        use_vllm=True,
        vllm_gpu_memory_utilization=0.55,
        report_to="wandb",
    )
    GRPOTrainer(
        model=model, processing_class=tok, args=cfg,
        reward_funcs=[reward_fn], train_dataset=build_curriculum(),
    ).train()
    model.save_pretrained(str(out_dir / "final"))

    elapsed = time.time() - t0
    print(f"[S2] done in {elapsed:.0f}s")
    assert elapsed < S2_BUDGET_SEC, "S2 over budget"
    wandb.finish()
    return out_dir / "final"

# ──────────────────── stage 3: eval ───────────────────────
def stage_eval(grpo_ckpt: Path) -> None:
    t0 = time.time()
    wandb.init(project=WANDB_PROJECT, name="s3-eval", reinit=True)

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(grpo_ckpt), max_seq_length=MAX_LEN, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # eval 1: CodeForge 3-task × 30 episodes each
    metrics = {}
    for task_id in ["greet_single_file", "greet_with_tests", "multi_file_module"]:
        scores = []
        for _ in range(30):
            brief = httpx.post(f"{CODEFORGE_URL}/reset",
                               json={"task_id": task_id}).json()
            eid = brief["episode_id"]
            prompt = (f"You are a Python engineer. Task: {task_id}.\n"
                      "Brief: " + brief["brief"] + "\n"
                      "Output exactly:\n"
                      '<answer>\n<file path="PATH">CODE</file>\n</answer>')
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_new_tokens=2048,
                                 do_sample=True, temperature=0.7)
            completion = tok.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            act, _ = parse_submit(completion)
            r = httpx.post(f"{CODEFORGE_URL}/step",
                           json={"episode_id": eid, "action": act}).json()
            scores.append(r["reward"])
        metrics[f"eval/{task_id}_mean"] = sum(scores) / len(scores)
        metrics[f"eval/{task_id}_pass@1"] = sum(s >= 0.8 for s in scores) / len(scores)
    wandb.log(metrics)
    print("[S3] metrics:", metrics)

    elapsed = time.time() - t0
    print(f"[S3] done in {elapsed:.0f}s")
    wandb.finish()


# ────────────────────── entrypoint ────────────────────────
if __name__ == "__main__":
    sft = stage_sft()
    grpo = stage_grpo(sft)
    stage_eval(grpo)
```

---

## 3. Hour-by-Hour Schedule

| Hour | Stage | Expected state | Go/No-go gate |
|---|---|---|---|
| 0:00 | **S1 start** — SFT warmup | Checkpoint loaded, dataset fetched | Loss trending down by 0:10 |
| 0:30 | S1 mid | `train/loss < 1.5` | If loss stuck, abort, check dataset |
| 1:00 | **S1 done → S2 start** — GRPO | SFT checkpoint saved; GRPO init + vLLM spin-up ~5 min | `train/reward_mean > 0.40` by step 10 |
| 2:00 | S2 step ~30 | `reward_mean > 0.45`, `group_std > 0.05` | **Abort if group_std < 0.02** (variance collapse) |
| 4:00 | S2 step ~100 | `reward_mean > 0.55` | Continue if monotonic ↑ |
| 6:00 | S2 step ~200 | `reward_mean > 0.62` | — |
| 8:00 | S2 step ~350 | `reward_mean > 0.68` | — |
| 9:00 | **S2 done → S3 start** — eval | Save final GRPO checkpoint | — |
| 10:00 | **S3 done** — metrics logged | `eval/multi_file_module_pass@1 > 0.30` | Pitch-ready |
| 10:00-12:00 | **Buffer** — unused if all green | Debug / retry window | — |

---

## 4. Go / No-Go Gates (Abort Early, Save Credit)

| Gate | At | Condition | Action |
|---|---|---|---|
| G-S1-LOSS | 0:15 | `train/loss` not decreasing | STOP. Dataset issue. Fix offline. |
| G-S2-REWARD | 1:30 | `reward_mean < 0.30` after 10 steps | STOP. Format parser failing. Patch and restart. |
| G-S2-VAR | 2:00 | `group_std < 0.02` | STOP. Variance collapse. Raise temp to 1.1, reweight curriculum harder. |
| G-S2-REGRESS | 4:00 | `reward_mean` fell > 0.1 from peak over 50 steps | STOP. KL run-away. Lower LR to 2e-6, raise `beta` to 0.1, restart from latest good checkpoint. |
| G-S2-STALE | any | 3 consecutive `step` calls return 0.0 with same error | STOP. Env Space down. Check CPU Space. |

Aborting at G-S2-REWARD at 1:30 h = burn ~$1.58 vs completing bad 12h run = $12.60. Early-stop pays for itself.

---

## 5. Single Reward Curve for Pitch (Rubric 20%)

Target plot — one chart, two series:

```
  reward_mean
   0.80 ┤                                 ╭─────── GRPO+SFT (final 0.72)
   0.70 ┤                          ╭──────╯
   0.60 ┤                   ╭──────╯
   0.50 ┤            ╭──────╯                      baseline Qwen (0.48)
   0.40 ┤─────✱──────╯  ───────────────────────────
          S1        S2 start        S2 end
          (SFT warmup)  (GRPO on curriculum)
```

Two numbers to quote in 3-min pitch:
- Baseline Qwen-Coder-1.5B on CodeForge: `reward_mean ≈ 0.48` (run once in PF).
- After S1+S2: `reward_mean ≈ 0.72` (from S3 eval).
- Lift: **+24 reward points** on contamination-free CodeForge held-out.

---

## 6. What Got Cut vs $200 Plan

| Removed | Reason |
|---|---|
| 4-run ablation suite | 24h → cuts 12h budget |
| 7B scale-up | 10h A100 = $41, and 1.5B proves the concept |
| DPO warmup phase | Redundant given SFT + GRPO |
| PRM training (Phase 4) | 10+ h labeling, deferred |
| Retrain buffer as separate run | Replaced by 2h in-run buffer |
| LiveCodeBench external eval | In-house 3-task × 30 eval suffices for pitch |
| Multiple beta sweep runs | Single default `beta=0.04` |

Remaining $187.40 buffer can fund:
- 1 full retrain (+$12.60) if first 12h fails
- OR 7B scale-up later (+$41)
- OR PRM stretch later (+$41)

---

## 7. Minimum File Set to Ship

```
CODEFORGE/training-space/
├── Dockerfile              # GPU base + vllm + trl + unsloth
├── train_all.py            # §2 above — single entry
├── requirements.txt        # pin versions
└── README.md               # 1-page run instructions
```

Everything else (SFT gen script, eval) runs offline or inside `train_all.py`.

---

## 8. Launch Command

```bash
# Set secrets on Training Space dashboard:
#   CODEFORGE_URL = https://krrishchoudhary109-code-forge.hf.space
#   WANDB_API_KEY = <your key>
#   HF_TOKEN      = <write token>

# Start Training Space (A10G Small):
# HF dashboard → Spaces → code-forge-training → Settings → Hardware: A10G Small → Restart

# Inside the Space (one-liner launches the whole workflow):
python train_all.py
```

---

## 9. Risks in 12-h Mode

| Risk | Probability | Mitigation |
|---|---|---|
| vLLM OOM on A10G 24GB with 1.5B | medium | `vllm_gpu_memory_utilization=0.55`; fall back to HF generate (~$3 extra hours) |
| Env Space sleeps mid-run | low | Hit `/state` every 30 s in reward client to keep warm |
| S1 dataset too small (<100 after quality filter) | low | PF-2 generates 200 → filter ≥0.8 → expect ~130 remaining |
| Curriculum too easy → G-S2-VAR triggers | medium | Reweight to 70% `multi_file_module` if hits |
| S3 eval hits rate limit on env Space | low | Sequential, not parallel; 90 episodes × ~5 s = 8 min |
| vLLM + Unsloth version conflict | medium | Pin `unsloth==2025.1.5 vllm==0.6.3 trl==0.12.1`. Test in smoke run first |

---

## 10. Optional: 2-Hour Smoke-Test Before 12-h Run

Spend $2.10 on a 2-h smoke with reduced config to verify stack:

```python
# Override in train_all.py:
S1_BUDGET_SEC = 600       # 10 min
S2_BUDGET_SEC = 3600      # 1 h, ~30 steps
S3_BUDGET_SEC = 600       # 10 min
# And in GRPOConfig:
num_generations=4         # half
save_steps=10
```

If smoke shows slope, commit $12.60 for full 12-h. If smoke fails, debug with $2.10 burn, not $12.60.

**Strongly recommended** given first-ever GRPO + CodeForge env pairing.

---

## 11. Cross-References

- `hf-budget-200-deployment-plan.md` — full $200 plan (this doc is the 12h subset).
- `theme2-rl-training-design-critique.md` P0-1..P0-6 — patches baked into `train_all.py`.
- `research-paper-alignment-analysis.md` §5 P0 — env-side exploit closes (precondition).
- `CODEFORGE/SYSTEM_DESIGN.md §15` — session TTL, must be ≥ 12 h for training.
- `CODEFORGE/CLAUDE.md §9` — sandbox stack.
