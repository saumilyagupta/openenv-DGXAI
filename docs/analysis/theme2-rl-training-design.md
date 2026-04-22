# Theme #2 — RL-Based LLM Training Design for CodeForge

**Date:** 2026-04-22
**Scope:** OpenEnv Apr '26 Hackathon — Theme #2 (Super Long-Horizon Planning & Instruction Following). Design full RL training pipeline for CodeForge that satisfies hackathon rubric weight 10% (Reward + Training Pipeline) + 20% (Showing Improvement in Rewards).
**Constraint:** Unsloth or HF TRL in Colab (explicit rubric requirement). OpenEnv latest. HF Spaces host.
**Sibling docs:** `research-paper-alignment-analysis.md`, `hackathon-alignment-analysis.md`, `codeforge-critical-analysis.md`.

---

## 0. Theme #2 Recap

> **Expected outcome:** environment that captures and improves LLM behaviour on challenging long-horizon tasks needing long running sessions **beyond context memory limits**.
> **Example:** large-scale codebase refactoring, long-horizon logistics, **300 instructions scattered around**.
> **Mercor bonus:** capped/uncapped rewards scaling with token output.

Key RL demands:

- **Sparse + delayed reward** — reward after N steps, not per-token.
- **Credit assignment across trajectory** — which iter caused gain?
- **Verifiable outcome reward** (RLVR) — no LLM judge in loop.
- **Beyond-context state tracking** — audit ledger or summarizer memory.
- **Robust to early mistakes** — agent must recover.

CodeForge already supplies the *environment half*. This doc specifies the *training half*.

---

## 1. RL Training Stack — Decision Table


| Layer                 | Choice                                                                        | Alt rejected           | Reason                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Base policy           | Qwen2.5-Coder-7B-Instruct (or 1.5B for Colab)                                 | Llama-3-8B             | Code-pretrained, Apache-2.0, fits 16 GB VRAM w/ 4-bit QLoRA                                                         |
| Trainer               | **TRL GRPOTrainer** (primary) + Unsloth LoRA                                  | PPOTrainer, DPOTrainer | GRPO = DeepSeek-R1 lineage, works with verifiable rule-based rewards, group-relative advantage needs no value model |
| Reward source         | **CodeForge `/step` endpoint** (sandbox + grounder + Brier)                   | Hand-written reward    | Entire point of CodeForge                                                                                           |
| Process reward        | **Math-Shepherd-style PRM** on Ralph iters                                    | End-reward only        | Theme #2 long-horizon credit                                                                                        |
| Verifier (optional)   | **V-STaR** small DistilBERT verifier                                          | None                   | Orthogonal signal, enables rejection sampling                                                                       |
| Memory / long-horizon | Audit ledger as external state                                                | In-context             | Spec literally says "beyond context memory limits"                                                                  |
| Eval                  | **LiveCodeBench** held-out + CodeHalu ablation + in-house `multi_file_module` | MBPP only              | Contamination-free                                                                                                  |


---

## 2. Paper Detail — All 16 Papers Expanded (+ 6 RL-specific)

Each entry: **what it actually does**, **what to copy**, **where in CodeForge pipeline**, **concrete hyper-parameters or API signatures where known**.

### 2.1 Search / Orchestration Pattern Papers

---

#### [1] FunSearch — Romera-Paredes et al., *Nature* 2024

**Title:** "Mathematical discoveries from program search with large language models."
**Loop:**

```
program_db := {seed_programs}
for t in 1..T:
    parents := sample_islands(program_db, k=2)
    child := LLM.mutate(parents, temperature=0.9)
    fitness := evaluator(child)        # deterministic Python call
    program_db.insert(child, fitness)
```

**Key design:** *island model* (multiple sub-populations to prevent premature convergence) + *best-shot sampling* (parent selection biased toward high-fitness + diversity). LLM is a *frozen mutation operator*, never retrained.
**Copy for CodeForge Ralph:** replace `synthesize→score→keep-if-better` with `sample_parents→mutate→score→island-insert`. K=4 islands, 2 parents/iter.  
**Why underrated\**

---

#### [2] AlphaEvolve — DeepMind, 2025 (tech report + blog)

**Extends FunSearch** with: (a) multi-file program repr, (b) evaluator ensemble (run fitness on k eval cases, median), (c) prompt meta-evolution (prompts also mutated).
**Headline result:** discovered faster matrix-multiplication algorithm for 4×4 complex matrices (beat Strassen, first improvement in 56 years).
**Copy:** evaluator ensemble = run `composite_score` under multiple sandbox configs (ruff-only, ruff+mypy, ruff+mypy+pytest) and take median → robustness to reward gaming.

---

#### [10] LATS — Zhou et al., *ICML* 2024

**Title:** "Language Agent Tree Search Unifies Reasoning, Acting, and Planning."
**Loop:** MCTS over agent trajectories. Each node = (state, action). Selection via UCB1. Expansion by LLM sampling. Evaluation by environment reward + LLM self-reflection. Backprop values to ancestors.
**HumanEval Pass@1:** GPT-4 + LATS = 94.4 vs ReAct 67.0.
**Copy:** wrap Ralph in MCTS loop; node value = CodeForge `quality`, expansion = Ralph synthesizer call, reflection = `interrogate` action.

---

#### [13] PlanSearch — Wang et al. 2024 (Scale AI)

**Title:** "Planning in Natural Language Improves LLM Search for Code Generation."
**Finding:** diverse natural-language plans → more diverse code → higher Pass@k than temperature-sampling alone. At equal budget, PlanSearch > ToT > CoT > N-sampling on HumanEval+, MBPP+, LiveCodeBench.
**Mechanism:** generate k "observations" about problem → combine into plans → each plan → code.
**Copy:** in M13 planner, generate k=5 *diverse observations* about the spec before decomposition.

---

#### [11] Agentless — Xia et al. 2024

**Title:** "Agentless: Demystifying LLM-based Software Engineering Agents."
**Result:** 27.3% on SWE-Bench Lite **without agent scaffold** using only 3 phases: (1) hierarchical file localization, (2) line-level edit, (3) patch validation via regression tests.
**Lesson:** simple pipelines beat complex agent scaffolds at the same budget.
**Copy:** validates CodeForge's 3-layer design. Pitch slide: "CodeForge = Agentless + verifier triple."

---

#### [9] CodePlan — Bairi et al. (Microsoft Research), *ICSE* 2024

**Title:** "CodePlan: Repository-level Coding using LLMs and Planning."
**Approach:** multi-step code edits across a repository via *planning over dependency DAG*. Build call/type-dep graph → topologically order edits → edit → propagate.
**Benchmark:** package migration, temporal-edits. +3–5× success rate vs ReAct.
**Copy:** M13 planner should emit topo-ordered `(target_file, edit_spec, tools)` tuples over KB2 dep graph.

---

#### [12] AutoCodeRover — Zhang et al., *ISSTA* 2024

**Title:** "AutoCodeRover: Autonomous Program Improvement."
**Approach:** AST-aware stratified retrieval + spectrum-based fault localization (SBFL). Beat SWE-Bench Lite with ~20% resolve rate.
**Copy:** merge into KB2 — expose `search_class`, `search_method_in_file`, `get_failing_tests_spectrum` as MCP tools.

---

### 2.2 Reward / Credit Assignment Papers

---

#### [3] RLEF — Gehring et al., Meta 2024

**Title:** "RLEF: Grounding Code LLMs in Execution Feedback."
**Approach:** multi-turn RL with execution feedback. Model emits code → interpreter runs it → stderr/stdout + test results fed back → model revises. Trained with PPO on execution-pass reward.
**Result:** Llama-3.1-70B + RLEF = CodeContests SOTA, 40% → 54% Pass@1 (`n=1`).
**Copy:** CodeForge `/step` = RLEF's execution feedback channel. Train with GRPO instead of PPO (no value model). Multi-turn = Ralph iters.

---

#### [4] SWE-RL — Wei et al., Meta 2025

**Title:** "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution."
**Approach:** RL on real GitHub PR data. Reward = *rule-based similarity* between generated patch and human patch (sequence ratio). No LLM judge. Llama-3.1-70B → +10 pp on SWE-Bench Verified, +1.5 pp on HumanEval+ (**transfer** to unrelated code tasks).
**Copy:** CodeForge uses sandbox-determined reward (stronger signal than patch-similarity). SWE-RL's transfer result = evidence that rule-based rewards generalize. Cite in pitch: "SWE-RL showed rule-based rewards transfer; CodeForge pushes further with triple-grounded rule-based rewards."

---

#### [7] Math-Shepherd / PRM — Wang et al., *ACL* 2024

**Title:** "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations."
**Approach:** auto-label each reasoning step as correct/incorrect by estimating `P(final_correct | step)` via Monte Carlo rollouts from that step. Train PRM on labels. Use PRM as reward per step.
**Result:** +9–15 pp on GSM8K, MATH with same base model.
**Copy for Ralph:** for each Ralph iter state `s_i`, do `K=4` rollouts of remaining iters, measure fraction reaching `quality ≥ 0.8`. Label = fraction. Train small PRM (DistilBERT, ~66M) on `(state_tokens, label)` pairs. Use PRM score as per-iter reward.

---

#### [6] V-STaR — Hosseini et al. 2024

**Title:** "V-STaR: Training Verifiers for Self-Taught Reasoners."
**Approach:** co-train generator + verifier. Generator samples k completions. Verifier scores them. Verifier trained on `(completion, final_correct_label)` via DPO (prefer correct over incorrect).
**Result:** +4–17 pp over self-consistency on MATH / GSM8K / MBPP.
**Copy:** train verifier on CodeForge audit-ledger trajectories. Use at inference as rejection-sampling filter: sample k patches from policy, rank by verifier, submit top-1.

---

#### [5] Absolute Zero Reasoner (AZR) — Zhao et al. 2025

**Title:** "Absolute Zero: Reinforced Self-play Reasoning with Zero Data."
**Approach:** *no human data.* Model proposes tasks (deduction, abduction, induction) in code → solves them → Python executor verifies. Reward = verification pass rate. Fully self-play.
**Result:** Qwen2.5-Coder-7B → +10+ pp on multiple benchmarks *starting from zero labeled data*.
**Copy:** Theme #4 pivot OR Theme #2 bolt-on. In CodeForge: add "propose task" action where agent generates a `Task(brief, budget, tools)` + a reference solution, then solves its own task. Corpus grows with verified self-generated tasks.

---

### 2.3 Retrieval / Knowledge / Library Papers

---

#### [8] LILO — Grand et al., *ICLR* 2024

**Title:** "LILO: Learning Interpretable Libraries by Compressing and Documenting Code."
**Approach:** three-stage loop: (1) LLM solves tasks, (2) *Stitch* (refactoring tool) extracts shared subroutines, (3) LLM auto-names + docs the extracted subroutines. Library grows with each round.
**Result:** +10–22 pp on string-edit, LOGO-drawing, scene-synthesis domains.
**Copy:** run offline over accepted CodeForge submissions → Stitch-extracted reusable helpers → LLM-documented → inserted into skill corpus. Replaces static ECC corpus with dynamic learned corpus.

---

#### [14] CodeHalu — Tian et al. 2024

**Title:** "CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification."
**Approach:** taxonomy of 5 code-hallucination types: *Fabrication* (nonexistent funcs), *Mapping Halluc* (wrong signature), *Naming Halluc*, *Resource Halluc* (missing imports), *Logical Halluc*. 699 labeled cases across 8 LLMs.
**Copy:** replay 699 cases against CodeForge's `grounder.py` → report false-negative rate per category. Pitch slide: "CodeForge grounder catches N/699 hallucinations CodeHalu labeled."

---

#### [15] LiveCodeBench — Jain et al. 2024

**Title:** "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code."
**Approach:** code tasks from LeetCode / AtCoder / CodeForces released after a cutoff, so model training data cannot have seen them.
**Copy:** pull 50 tasks released post-2025-07 into `codeforge://tasks/livecodebench_holdout`. Use for training-curve eval.

---

#### [16] ScienceAgentBench — Chen et al. 2024

**Title:** "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery."
**Approach:** 102 tasks across bioinformatics, chemistry, neuroscience; reference to 44 peer-reviewed papers. Input = natural-language spec, output = executable Python + data artifact.
**Copy:** Theme #3.1 pivot. Import task definitions into CodeForge `tasks.py` as hard-level tasks. Direct rubric match ("papers → code → experiments").

---

### 2.4 RL-Specific Papers (added for Theme #2 training stack)

---

#### [17] DeepSeek-R1 / **GRPO** — DeepSeek-AI 2025

**Title:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
**Algorithm — Group Relative Policy Optimization (GRPO):** for each prompt, sample group of G outputs `{o_1..o_G}`, compute rewards `{r_1..r_G}`, advantage `A_i = (r_i - mean(r)) / std(r)`. Update policy with PPO-style clipped objective but using A_i. **No value network needed.**
**Result:** DeepSeek-R1-Zero used *only rule-based rewards* (exact-match for math, compile+test for code) and reached o1-level. Template:

```
<think> ... </think>
<answer> ... </answer>
```

**Copy:** GRPO is our training algorithm. CodeForge reward = rule-based. Zero value-model overhead → fits Colab.

**TRL API:**

```python
from trl import GRPOTrainer, GRPOConfig

cfg = GRPOConfig(
    output_dir="codeforge-grpo",
    num_generations=8,          # G
    max_prompt_length=1024,
    max_completion_length=2048,
    temperature=0.9,
    beta=0.04,                  # KL penalty to ref model
    learning_rate=5e-6,
)

def codeforge_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        r = codeforge_client.step(prompt, completion)["reward"]
        rewards.append(r)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    args=cfg,
    reward_funcs=[codeforge_reward_fn],
    train_dataset=codeforge_task_dataset,
)
trainer.train()
```

---

#### [18] RLVR — RL with Verifiable Rewards (OpenAI o1-lineage, adopted by DeepSeek-R1, Tulu-3)

**Pattern:** reward only when answer is *programmatically verifiable*. For code: compile + run tests. For math: exact match. Rejects LLM-judge.
**Copy:** CodeForge `quality = 0.6*sandbox + 0.4*ground` IS an RLVR reward (no LLM judge anywhere).
**Pitch framing:** "CodeForge is RLVR with a triple-grounded reward, stronger than test-suite-only verifiers."

---

#### [19] PPO — Schulman et al. 2017, adapted by InstructGPT (Ouyang 2022)

**Baseline RL algorithm.** Clip objective + value model. Still default in many code-RL stacks (RLEF used PPO).
**When to use over GRPO:** longer horizons where per-token value matters, or when rewards are dense.
**Trade-off:** value model doubles VRAM — GRPO often preferred on Colab.

---

#### [20] DPO — Rafailov et al. 2024, *NeurIPS*

**Title:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."
**Approach:** skip reward model and RL entirely. Given pairs `(chosen, rejected)`, directly optimize policy via implicit reward. Closed-form objective.
**When to use for CodeForge:** offline phase — mine audit ledger for `(higher_quality, lower_quality)` pairs, train with DPO first, then GRPO online.
**TRL:** `DPOTrainer` with `DPOConfig`.

---

#### [21] KTO — Ethayarajh et al. 2024

**Title:** "KTO: Model Alignment as Prospect Theoretic Optimization."
**Approach:** DPO variant that needs only binary `good/bad` labels, not pairs.
**When to use:** when CodeForge audit has `quality ≥ 0.8` (good) vs `< 0.5` (bad) but no pairwise structure.

---

#### [22] Reflexion — Shinn et al., *NeurIPS* 2023

**Title:** "Reflexion: Language Agents with Verbal Reinforcement Learning."
**Approach:** after failed episode, LLM *verbally* reflects on failure, stores reflection in episodic memory, retries with memory prepended. No weight update.
**Copy:** already implicit in Ralph (keep-if-better). Strengthen by logging reflection strings in audit ledger for Theme #2 "beyond context memory" compliance.

---

## 3. Full Training Pipeline Design

### 3.1 Phase-structured pipeline

```
                             ┌────────────────────────────┐
                             │  CodeForge Environment      │
                             │  (FastAPI /step endpoint)   │
                             └──────────────┬──────────────┘
                                            │ reward + obs + audit
                                            ▼
Phase 1 — Cold-start SFT                           Phase 2 — Offline DPO                         Phase 3 — Online GRPO                        Phase 4 — PRM-augmented GRPO
┌──────────────────────┐                ┌──────────────────────────┐                 ┌──────────────────────────┐                ┌────────────────────────────┐
│ collect 500 trajec-  │  quality≥0.8   │ mine audit: pairs         │  chosen/       │ rollout Qwen-Coder-1.5B   │  reward        │ train PRM on rollout        │
│ tories from Claude-  │ ─────────────▶ │ (high_quality vs          │  rejected ─▶   │ policy on CodeForge env,  │ from step  ──▶ │ completion-rate labels,     │
│ Haiku 4.5 self-play  │                │  low_quality, same brief) │                │ group G=8, GRPO update    │                │ GRPO with r = PRM + sandbox │
│ on 3 CodeForge tasks │                │                           │                │                           │                │                             │
└──────────────────────┘                └──────────────────────────┘                 └──────────────────────────┘                └────────────────────────────┘
      1×A10 2h                                   1×A10 3h                                      1×A10 6h Colab Pro                          1×A100 8h (stretch)
```

### 3.2 Reward composition (per step)

```python
def shaped_reward(obs, action, env_step_result, prm=None) -> float:
    # Primary: CodeForge triple reward (RLVR)
    r_env = env_step_result["reward"]          # from /step, range [0, 1]

    # Secondary: PRM step-level (Phase 4 only)
    r_prm = prm.score(obs, action) if prm else 0.0   # [0, 1]

    # Shaping: citation bonus (SYSTEM_DESIGN §4.8.4)
    r_cite = 0.01 * cited_skills_appearing_in_code(action)  # cap at 0.05

    # Penalty: exploit markers (from P0-1, P0-2 of sibling doc)
    penalty = 0.20 if env_step_result.get("exploit_flagged") else 0.0

    return max(0.0, min(1.0, 0.6 * r_env + 0.3 * r_prm + r_cite - penalty))
```

### 3.3 Long-horizon memory (beyond-context)

Theme #2 literal requirement: "long running sessions beyond context memory limits."


| Memory type | Source                        | Written by                | Read by                        |
| ----------- | ----------------------------- | ------------------------- | ------------------------------ |
| Episodic    | audit ledger (M6)             | env on every action       | agent via `get_audit` MCP tool |
| Skill       | KB1 corpus (M3)               | LILO compaction (offline) | agent via `query_kb`           |
| Code graph  | KB2 (M10)                     | env on file write         | agent via `query_code_graph`   |
| Reflection  | `reflection_trace[]` in audit | agent on Ralph iter fail  | prepended to next iter prompt  |


Context stays bounded (<16K). Trajectory can be arbitrary length via audit replay.

### 3.4 Dataset shape (for TRL)

```python
# train_dataset rows
{
  "prompt": str,       # CodeForge brief + retrieved skill context + audit summary
  "task_id": str,      # one of greet_single_file / greet_with_tests / multi_file_module / scattered_300
  "episode_id": str,   # for stable env reset
}

# GRPO generates G=8 completions per prompt, each scored via codeforge_reward_fn.
```

---

## 4. Minimal Unsloth + TRL Colab Skeleton

Satisfies hackathon minimum requirement **"show a minimal training script for your environment using Unsloth or HF TRL in Colab."**

```python
# Cell 1 — install
!pip install -q unsloth trl==0.12.* vllm==0.6.* datasets httpx

# Cell 2 — CodeForge client (pointing at HF Space URL)
import httpx, os
CODEFORGE_URL = os.getenv("CODEFORGE_URL", "https://krrishchoudhary109-code-forge.hf.space")

class CodeForgeClient:
    def __init__(self, url): self.url = url
    def reset(self, task_id):
        return httpx.post(f"{self.url}/reset", json={"task_id": task_id}, timeout=60).json()
    def step(self, episode_id, action):
        return httpx.post(f"{self.url}/step",
                          json={"episode_id": episode_id, "action": action},
                          timeout=120).json()

client = CodeForgeClient(CODEFORGE_URL)

# Cell 3 — base model with Unsloth 4-bit QLoRA
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Cell 4 — reward function that calls CodeForge env
def codeforge_reward(prompts, completions, task_ids, episode_ids, **_):
    rewards = []
    for prompt, completion, tid, eid in zip(prompts, completions, task_ids, episode_ids):
        # Parse completion as a CodeForge `submit` action
        action = {"type": "submit",
                  "files": parse_files_from_completion(completion),
                  "confidence": parse_confidence(completion) or 0.7}
        result = client.step(eid, action)
        rewards.append(float(result["reward"]))
    return rewards

# Cell 5 — GRPO trainer
from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir="codeforge-grpo-qwen-1.5b",
    learning_rate=5e-6,
    num_generations=8,                  # G
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=50,
    beta=0.04,
    temperature=0.9,
    report_to="wandb",
)

from datasets import Dataset
task_briefs = [
    {"prompt": build_prompt(tid), "task_ids": tid,
     "episode_ids": client.reset(tid)["episode_id"]}
    for tid in ["greet_single_file"] * 200 + ["greet_with_tests"] * 200 + ["multi_file_module"] * 100
]
train_ds = Dataset.from_list(task_briefs)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=cfg,
    reward_funcs=[codeforge_reward],
    train_dataset=train_ds,
)
trainer.train()
trainer.save_model("codeforge-grpo-final")
```

Colab A100 40GB: ~6 h for 500 steps × G=8. T4 16GB feasible with 1.5B + batch=1 + accum=8.

---

## 5. Metrics & Reward-Curve Demo (Rubric 20% hole)

For hackathon "Showing Improvement in Rewards" 20% criterion, log:


| Metric                      | Source                                   | Plot type                        |
| --------------------------- | ---------------------------------------- | -------------------------------- |
| `train/reward_mean`         | GRPO trainer                             | line, per-step                   |
| `train/group_std`           | GRPO trainer                             | line (variance collapse signal)  |
| `eval/pass@1_livecodebench` | held-out eval every 50 steps             | line                             |
| `eval/exploit_rate`         | audit-ledger empty-submit count          | line (should stay 0 post-P0 fix) |
| `eval/groundedness_mean`    | grounder on eval submissions             | line                             |
| `eval/codehalu_fn_rate`     | CodeHalu 699-case replay every 100 steps | bar, 5 categories                |


Pitch deck:

- Slide: "SFT baseline vs DPO-warmup vs GRPO-online vs GRPO+PRM" — 4 lines on same axes.
- Slide: "before weights / after weights" on 1 sampled `multi_file_module` episode, side-by-side Ralph trace.

---

## 6. Mercor Bonus Alignment (capped/uncapped token-scaled)

Mercor sub-theme: **"environment with capped/uncapped rewards where frontier model rewards scale with token output."**

CodeForge already caps reward at 1.0 (Brier bound). To also reward token-output-scaled for frontier models:

```python
def token_scaled_bonus(completion_tokens: int, quality: float) -> float:
    # Uncapped lane: reward ∝ log(tokens) × quality. Rewards long, correct reasoning.
    if quality >= 0.8:
        return min(0.10, 0.02 * math.log(max(1, completion_tokens / 128)))
    # Capped lane: zero bonus unless quality threshold met.
    return 0.0
```

Add `reward_mode: Literal["capped", "uncapped"]` to task config. Switch at env reset. Log in audit for ablation.

---

## 7. What to Ship — Ordered


| #   | Deliverable                                            | Effort | Rubric hit                       |
| --- | ------------------------------------------------------ | ------ | -------------------------------- |
| 1   | Close P0-1, P0-2 exploits (`grader.py`, `grounder.py`) | 0.5 d  | Pipeline 10%, Innovation 40%     |
| 2   | Hybrid BM25+BGE retrieval (P1-3 from sibling doc)      | 1 d    | Innovation 40%                   |
| 3   | **GRPO Colab notebook using skeleton in §4**           | 1 d    | **Pipeline 10% (currently 0/5)** |
| 4   | LiveCodeBench held-out split (50 tasks)                | 0.5 d  | Improvement 20%                  |
| 5   | Reward curves + exploit-rate curve for pitch           | 0.5 d  | Improvement 20%                  |
| 6   | PRM training on audit ledger (Phase 4)                 | 2 d    | Innovation 40% (stretch)         |
| 7   | Scatter-brief 300-instr task variant                   | 1 d    | Theme #2 literal spec match      |
| 8   | Mercor capped/uncapped reward mode                     | 0.5 d  | Bonus prize eligibility          |


**Minimum-viable-pitch set (if 3-day sprint):** 1 + 3 + 4 + 5. Closes both exploits, adds Unsloth/TRL training with CodeForge as reward env, produces reward curves on contamination-free eval.

---

## 8. Uncertainty Flags

- GRPO stability on <2B code models with variance-based advantage can collapse when all G completions score the same (common on trivial tasks). Mitigation: temperature=1.0 + task-difficulty curriculum.
- TRL's `GRPOTrainer` reward-function signature evolved in v0.12; exact kwargs may shift. Pin `trl==0.12.1`.
- CodeForge `/step` latency (sandbox run ~2–5 s) × G=8 × batch=4 → 32–64 s per training step. Plan Colab session length accordingly.
- PRM label propagation threshold (Math-Shepherd used `label ≥ 0.5`) may need tuning for sparse-reward Theme #2 tasks.
- Scatter-brief 300-instr task has no existing reference implementation for our env; design novel, expect iteration.

---

## 9. Cross-References

- `research-paper-alignment-analysis.md` — full 16-paper table + deconstruction + P0–P3 plan.
- `hackathon-alignment-analysis.md` — 57/100 current score, biggest hole = Pipeline + Improvement.
- `codeforge-critical-analysis.md` — architectural audit.
- `CODEFORGE/SYSTEM_DESIGN.md §4.8.4` — reward shaping details (citation bonus etc.).
- `CODEFORGE/CLAUDE.md §4` — reward formula.

