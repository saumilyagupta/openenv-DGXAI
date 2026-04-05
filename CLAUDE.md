---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

This project is **EpistemicNav**, an OpenEnv-compliant RL environment for the Meta PyTorch OpenEnv Hackathon x Scaler SST 2026. It trains LLM agents to reason accurately under uncertainty, rewarding calibrated confidence (Brier score) rather than just correct answers.

### Core Components

*   **`server/environment.py` (The Engine):** `EpistemicNavEnvironment(Environment)` — manages claims, evidence gathering via BM25, budget tracking, and the step/reset/state loop.
*   **`server/app.py` (The API):** FastAPI app created via `openenv.create_app()` exposing `/step`, `/reset`, `/state` endpoints. Binds to `0.0.0.0:7860`.
*   **`server/grader.py` (The Reward):** Brier score reward function. Penalises overconfidence on wrong answers AND underconfidence on right ones. Special case: `verdict="uncertain"` on genuinely uncertain claims gets minimum 0.70 reward.
*   **`server/retriever.py` (The Search):** BM25Okapi wrapper over `data/evidence.json`. Pure Python, <10ms per query, no GPU.
*   **`models.py` (The Schema):** Pydantic v2 models — `EpistemicAction` (QUERY/COMMIT), `EpistemicObservation`, `EvidenceSnippet`.
*   **`client.py` (The Client):** `EpistemicEnv(GenericEnvClient)` — HTTP client for connecting to the env server.
*   **`inference.py` (The Baseline Agent):** LLM-powered agent using OpenAI-compatible API. Must run in <20 minutes on 2vCPU/8GB.

### Data

*   **`data/claims.json`:** Target 400 claims (200 easy, 150 medium, 50 hard). Currently 10.
*   **`data/evidence.json`:** Target 2000 snippets across 15+ domains. Currently 50.

### Three Tasks

1. **easy (single_hop):** Single-hop factual claim. One query sufficient. Reward ceiling ~0.98.
2. **medium (multi_hop):** Multi-hop claim requiring 3-4 evidence pieces. Reward ceiling ~0.88.
3. **hard (contradictory):** Contradictory evidence. Correct answer is `"uncertain"`. Reward floor 0.70.

## Development Guidelines

*   **OpenEnv Spec Strictness:** All API endpoints must strictly adhere to the OpenEnv specification and `problem_statement_and_guidelines.md`.
*   **Anti-Reward Hacking:** Rewards tied to Brier score calibration. Query steps return 0.0 reward. No negative rewards (breaks [0,1] range).
*   **Determinism:** Grader must return reproducible scores in [0.0, 1.0].
*   **No External Calls from Env:** All evidence is pre-cached in `evidence.json`. No Wikipedia API, no web requests from the server.
*   **Containerization:** Docker on HF Spaces. FastAPI binds `0.0.0.0:7860`. Image target ~280MB (python:3.11-slim).

## Common Commands

*   **Local Server:** `uvicorn server.app:app --host 0.0.0.0 --port 7860`
*   **Docker Build:** `docker build -t epistemic-nav .` then `docker run -p 7860:7860 epistemic-nav`
*   **Run Baseline Agent:** `python inference.py` (Requires API keys in `.env`)
*   **Deploy to HF Spaces:** `git push space main` (Requires `space` remote with HF write token)

## Skill Usage Guide

Use the following skills at the appropriate phase of development. Invoke via `/skill-name`.

### Phase 1: Planning

| Skill | When to Use |
|---|---|
| `/plan` | Before starting any implementation chunk. Create step-by-step plan, identify risks. |
| `/superpowers:brainstorming` | Before creative decisions: data domain selection, claim design strategy, evidence corpus structure. |
| `/superpowers:writing-plans` | When you have a spec or requirements for a multi-step task, before touching code. |

### Phase 2: Data Generation (claims.json + evidence.json)

| Skill | When to Use |
|---|---|
| `/deep-research` | Research factual claims across 15+ domains. Find claims with known ground truths. Generate evidence snippets with citations. |
| `/huggingface-skills:hugging-face-datasets` | When publishing or managing the dataset on HF Hub. |

### Phase 3: Python Development

| Skill | When to Use |
|---|---|
| `/python-patterns` | When writing Python code — idiomatic patterns, type hints, Pydantic v2, dataclasses. |
| `/coding-standards` | General code quality, naming, file organization. |
| `/backend-patterns` | FastAPI endpoint design, API response format, error handling. |
| `/tdd` | Before implementing any new feature. Write tests first (RED), implement (GREEN), refactor (IMPROVE). Target 80%+ coverage. |
| `/python-testing` | pytest strategies, fixtures, mocking, parametrization for grader/retriever/environment tests. |
| `/superpowers:test-driven-development` | Before writing implementation code for any feature or bugfix. |

### Phase 4: Code Review & Quality

| Skill | When to Use |
|---|---|
| `/python-review` | After writing Python code. Review for PEP 8, type hints, security, idioms. |
| `/code-review` | After each chunk of implementation. Check quality, security, maintainability. |
| `/simplify` | After implementation — review changed code for reuse, quality, efficiency. |
| `/security-scan` | Before commits. Check for leaked API keys, config issues, injection risks. |
| `/security-review` | When handling user input, API endpoints, secrets, or authentication. |

### Phase 5: Build & Verification

| Skill | When to Use |
|---|---|
| `/build-fix` | When Docker build or server startup fails. |
| `/superpowers:verification-before-completion` | Before claiming any task is done. Run verification commands, confirm output. |
| `/superpowers:systematic-debugging` | When encountering bugs, test failures, or unexpected behavior. |
| `/verify` | Run full verification loop before submission. |

### Phase 6: Deployment (HF Spaces)

| Skill | When to Use |
|---|---|
| `/huggingface-skills:hf-cli` | Push to HF Spaces, manage repo, upload files. |
| `/huggingface-skills:huggingface-gradio` | Only if adding a web UI (currently disabled via `ENABLE_WEB_INTERFACE=false`). |
| `/huggingface-skills:hugging-face-jobs` | If running workloads on HF infrastructure. |

### Phase 7: Documentation & Finish

| Skill | When to Use |
|---|---|
| `/update-docs` | Update README with action/obs spaces, setup instructions, task descriptions. |
| `/superpowers:finishing-a-development-branch` | When implementation is complete and tests pass. Guides merge/PR/cleanup. |
| `/superpowers:requesting-code-review` | Before merging. Verify work meets all spec requirements. |

### Utility Skills (Use Anytime)

| Skill | When to Use |
|---|---|
| `/aside` | Answer a quick side question without losing context on current task. |
| `/superpowers:dispatching-parallel-agents` | When facing 2+ independent tasks (e.g., generate claims AND write tests simultaneously). |
| `/exa-search` | Web research when deep-research or GitHub search is insufficient. |
| `/save-session` | Save session state before ending work so next session can resume. |
| `/resume-session` | Start of new session — load prior context. |

### Skills NOT Relevant (Skip These)

- Vercel skills (deploying to HF Spaces, not Vercel)
- Frontend/React/Next.js skills (no frontend)
- Django/Spring Boot/Go/Kotlin skills (pure Python project)
- Database skills (no database — flat JSON files)
- Figma/design skills (no UI design)

## Submission Deadline

**April 8, 2026, 11:59 PM IST**

## Submission Checklist

- [ ] `claims.json` — 400 claims, balanced distribution (200 easy, 150 medium, 50 hard)
- [ ] `evidence.json` — 2000 snippets, 15+ domains
- [ ] `models.py` — typed Pydantic v2 Action + Observation
- [ ] `server/environment.py` — step(), reset(), state() implemented
- [ ] `server/grader.py` — Brier score, scores in [0.0, 1.0]
- [ ] `server/retriever.py` — BM25, top-k, <10ms per query
- [ ] `Dockerfile` builds locally
- [ ] `openenv.yaml` valid — name, tasks, reward_range
- [ ] HF Space deployed — returns 200, responds to reset()
- [ ] `inference.py` in root — uses API_BASE_URL, MODEL_NAME, HF_TOKEN
- [ ] `inference.py` runtime <20 min on 2vCPU / 8GB
- [ ] `README.md` — action space, observation space, setup, task descriptions
- [ ] Pre-submission validation script passes
