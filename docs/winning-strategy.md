# EpistemicNav — Winning Strategy

## Psychometric Analysis of Hackathon Judging

**Deadline:** April 8, 2026, 11:59 PM IST (3 days remaining)

---

## 1. Judge Psychology — Who Evaluates Us and What They Value

### Phase 1: Automated Validation (Pass/Fail Gate)

Automated scripts check:
- HF Space returns 200 and responds to `reset()`
- `openenv validate` passes
- `docker build && docker run` succeeds
- `inference.py` completes without error and produces scores
- 3+ tasks with graders, all scores in [0.0, 1.0]

**Psychology:** Binary. No partial credit. If any check fails, we're disqualified before a human ever sees our work. This is pure compliance — every requirement must be met exactly.

**Our status:** Passing, but **missing three required endpoints** (`/baseline`, `/grader`, `/tasks`) specified in the problem statement. This is a disqualification risk.

### Phase 2: Agentic Evaluation (Scored)

- Our baseline agent is re-run
- A **standard Open LLM agent (Nemotron 3 Super)** is run against our environment
- **Score variance check** is performed

**Psychology:** The judges want to see:
1. **Difficulty progression in the numbers** — easy scores > medium scores > hard scores. If Nemotron gets similar scores across levels, the difficulty design looks fake.
2. **Score variance** — if every episode returns nearly the same score, the grader looks deterministic/rigged. They want genuine variability showing the environment responds differently to different agent strategies.
3. **Agent learnability** — can a generic agent (not our tailored one) interact meaningfully? If the observations are confusing or the action space is unclear, the agent will flail and scores will be random noise.

**Our risk:** Query steps return 0.0 reward. A generic agent gets zero learning signal during exploration — only one final score when it commits. This looks like sparse reward, which the rubric explicitly penalizes.

### Phase 3: Human Review (Top Submissions Only)

**Meta and HuggingFace engineers** review for:
- Real-world utility
- Creativity
- Exploit checks (reward hacking, gaming the grader)

**Psychology:** These are senior ML/AI researchers who:
- See hundreds of submissions — they're pattern-matching for "yet another toy env" vs "this is actually useful"
- Value research utility — "would I use this to evaluate my own agents?"
- Can spot shallow implementations in seconds — surface-level domain wrapping won't impress them
- Appreciate clever engineering that solves real problems
- Are allergic to bullshit — overclaiming without evidence backfires

**What wins at this phase:** A compelling narrative backed by solid execution. The README is your pitch deck.

---

## 2. Scoring Rubric Deep Dive

### Real-World Utility — 30% (Heaviest Weight)

| Score Range | Description |
|-------------|-------------|
| 0-5 | Toy/artificial problem with no practical application |
| 6-15 | Valid domain but shallow modeling of the real task |
| 16-25 | Good domain modeling, would be useful for agent evaluation |
| **26-30** | **Excellent — fills a real gap, immediate value for the RL/agent community** |

**What 26-30 requires:**
- The environment addresses a genuine, unsolved problem
- Someone in the RL/agent community would actually use this
- It produces insights you can't get from existing benchmarks

**Our position (estimated: ~20/30):**
- LLM calibration IS a real unsolved problem — models are dangerously overconfident when wrong, hedging when right
- The concept is strong, but the README undersells it
- **No baseline scores documented** — judges can't see the difficulty progression working. They see claims about "reward ceiling ~0.98" but no proof.

**Path to 27/30:**
- Add real baseline scores to README showing difficulty separation
- Frame the problem in research terms: cite Brier scores, calibration literature
- Position as "the first benchmark specifically for epistemic calibration in LLMs"

### Task & Grader Quality — 25%

**Explicit checklist judges use:**
- 3+ tasks with difficulty range? **Yes**
- Graders produce scores between 0.0-1.0? **Yes**
- Graders deterministic and reproducible? **Yes**
- Hard task genuinely challenges frontier models? **Needs verification**

**Our position (estimated: ~18/25):**
- Three well-defined tiers with clear design rationale
- "Uncertain" verdict on contradictory evidence is genuinely hard
- Brier score is a well-researched, defensible metric

**Gap:** We haven't verified that a strong generic model (like the ones judges use) actually struggles on hard tasks. If Nemotron 3 Super aces all three levels, the difficulty design looks fake.

**Path to 22/25:**
- Run a generic agent and confirm score separation across levels
- If hard tasks are too easy, add more adversarial contradictory claims
- Document expected score ranges per difficulty in README

### Environment Design — 20%

**Explicit checklist:**
- `reset()` produces clean state? **Yes**
- Action/observation types well-designed and documented? **Yes**
- Reward function provides useful varying signal (not just sparse)? **NO — this is our biggest weakness**
- Episode boundaries sensible? **Yes**

**Our position (estimated: ~13/20):**

**The problem:** Query steps always return `reward = 0.0`. The agent explores blindly for up to 8 steps, then gets one final Brier score on commit. This is textbook **sparse reward** — exactly what the rubric penalizes.

The rubric says: *"Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion."*

We provide signal at end-of-episode only. We fail this criterion.

**Path to 17/20:**
Add intermediate query rewards — a small signal (0.01-0.05) based on:
- Evidence relevance (BM25 score above threshold)
- Information gain (new evidence vs. duplicate)
- Budget efficiency (not wasting queries on low-relevance searches)

This transforms sparse reward into dense reward across the full trajectory, directly addressing the rubric's concern.

### Code Quality & Spec Compliance — 15%

**Explicit checklist:**
- `openenv validate` passes? **Yes**
- `docker build && docker run` works? **Yes**
- HF Space deploys and responds? **Needs redeployment with latest changes**
- Baseline script runs and reproduces scores? **Yes (after stdout format fix)**

**CRITICAL FINDING — Missing Required Endpoints:**

The problem statement specifies three additional endpoints:

> **`/baseline`**: Trigger inference script and returns baseline score for all 3 tasks
> **`/grader`**: Returns grader score after an episode is completed
> **`/tasks`**: Returns list of tasks and the action schema (fields required for an action in a step)

**We don't have any of these.** This is a direct spec violation that costs compliance points.

**Other gaps:**
- No tests at all — rubric says "tested"
- No docstrings on key functions

**Our position (estimated: ~10/15):**

**Path to 14/15:**
- Add the three missing endpoints
- Add basic pytest tests (grader, retriever, episode flow)
- Ensure baseline scores are reproducible

### Creativity & Novelty — 10%

**Checklist:**
- Domain not seen in OpenEnv before? **Yes** — epistemic reasoning is novel
- Reward design has interesting properties? **Yes** — Brier score penalizes both overconfidence AND underconfidence
- Clever mechanics? **Yes** — "uncertain" as a first-class verdict with guaranteed minimum reward

**Our position (estimated: ~8/10):** This is our strongest category. Just needs better storytelling.

---

## 3. Current Score vs. Winner Projection

| Category | Weight | Current Est. | With Fixes | Delta |
|----------|--------|-------------|------------|-------|
| Real-world utility | 30% | 20 | 27 | **+7** |
| Task & grader quality | 25% | 18 | 22 | **+4** |
| Environment design | 20% | 13 | 17 | **+4** |
| Code quality & spec | 15% | 10 | 14 | **+4** |
| Creativity & novelty | 10% | 8 | 9 | **+1** |
| **TOTAL** | **100%** | **69** | **89** | **+20** |

A score of 89/100 would place us in strong contention for first place.

---

## 4. Priority-Ranked Improvements

### Priority 1: Missing Endpoints (CRITICAL — Spec Compliance)
**Impact:** 15% category, potential disqualification risk
**Effort:** ~1 hour

Add three required endpoints to `server/app.py`:
- `GET /tasks` — return task list with action schema
- `GET /grader` — return grader score for completed episode
- `POST /baseline` — trigger inference and return scores

### Priority 2: Intermediate Query Rewards (HIGH — Reward Design)
**Impact:** 20% category + 25% category
**Effort:** ~1 hour

Modify `environment.py` step() to return small rewards (0.01-0.05) on QUERY steps:
- Base reward on evidence relevance score from BM25
- Bonus for novel evidence (not duplicating what's already gathered)
- Keep total query rewards small so COMMIT reward remains dominant

This directly addresses the rubric's "useful varying signal" criterion.

### Priority 3: Baseline Scores in README (HIGH — Proof of Value)
**Impact:** 30% category
**Effort:** ~30 minutes

Run inference.py, capture actual scores, and add to README:
- Mean scores per difficulty level
- Score variance showing genuine separation
- Concrete numbers the judges can see immediately

### Priority 4: Basic Test Suite (MEDIUM — Quality Signal)
**Impact:** 15% category
**Effort:** ~1 hour

Add `tests/` with pytest:
- `test_grader.py` — verify Brier score math, edge cases, [0,1] range
- `test_retriever.py` — verify BM25 search returns results, handles empty queries
- `test_environment.py` — verify reset/step/state cycle, budget tracking, forced commit

### Priority 5: README Storytelling (MEDIUM — Persuasion)
**Impact:** 30% category + 10% category
**Effort:** ~30 minutes

Rewrite README to lead with the research problem:
- "LLMs are miscalibrated — overconfident when wrong, underconfident when right"
- Cite Brier score as a principled metric from forecasting literature
- Position as "the first OpenEnv benchmark specifically for epistemic calibration"
- Add a "Why This Matters" section before the technical details
- Include the baseline scores showing difficulty progression

### Priority 6: Phase 2 Verification (MEDIUM — Score Separation)
**Impact:** 25% category
**Effort:** ~30 minutes

Run a generic agent (not our tailored one) against all three difficulty levels:
- Verify easy > medium > hard score separation
- Check score variance is non-trivial
- If hard tasks are too easy for a strong model, add more adversarial claims

---

## 5. Competitive Moat Analysis

### What Other Submissions Likely Look Like

Most hackathon entries will be:
- **Email triage / content moderation / customer support** — the examples listed in the problem statement. Judges will see dozens of these.
- **Surface-level wrapping** — slap an RL interface on a classification task, call it a day.
- **Binary rewards** — correct/incorrect with no calibration signal.

### Our Differentiators

1. **Novel domain** — epistemic reasoning under uncertainty isn't in the example list. Judges explicitly reward "domain we haven't seen before."
2. **Principled reward function** — Brier score is from forecasting/statistics, not invented for the hackathon. It has mathematical properties (proper scoring rule) that judges will recognize.
3. **The "uncertain" verdict** — most environments force binary choices. Ours rewards knowing when you don't know. This is philosophically interesting and practically important.
4. **Budget mechanics** — the explore/exploit tradeoff (spend budget gathering evidence vs. commit early) creates genuine strategic depth.

### Anti-Exploit Properties

Judges check for reward hacking. Our design is robust:
- Always committing "uncertain" with 0.5 confidence gives ~0.65 reward — decent but not winning
- Always committing "true" with 1.0 confidence gives ~0.45 average (wrong on false/uncertain claims)
- Optimal strategy requires actually reading evidence and calibrating confidence
- Budget efficiency bonus rewards decisive agents, preventing "always query everything" exploits

---

## 6. Execution Timeline

### Day 1 (Today — April 5)
- [ ] Add `/baseline`, `/grader`, `/tasks` endpoints
- [ ] Add intermediate query rewards
- [ ] Add basic test suite

### Day 2 (April 6)
- [ ] Run baseline inference, capture scores
- [ ] Rewrite README with scores and research framing
- [ ] Verify Phase 2 readiness (generic agent test)

### Day 3 (April 7)
- [ ] Final Docker build verification
- [ ] Deploy to HF Spaces
- [ ] Run full validation pipeline
- [ ] Buffer for fixes

### Day 4 (April 8 — Deadline)
- [ ] Final push to HF Spaces
- [ ] Verify Space is live and responding
- [ ] Submit before 11:59 PM IST
