"""
Generate scaled epistemic claims and evidence using Gemini API.

Targets: 400 claims (200 easy, 150 medium, 50 hard) + 2000 evidence snippets.
Saves intermediate results for crash recovery.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Please set GEMINI_API_KEY in .env file.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"},
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLAIMS_PARTIAL = DATA_DIR / "claims_partial.json"
EVIDENCE_PARTIAL = DATA_DIR / "evidence_partial.json"
CLAIMS_FINAL = DATA_DIR / "claims.json"
EVIDENCE_FINAL = DATA_DIR / "evidence.json"

DOMAINS = [
    "science", "geography", "history", "technology", "economics",
    "medicine", "law", "mathematics", "linguistics", "astronomy",
    "biology", "engineering", "nutrition", "sports", "politics",
]

# Target counts
EASY_COUNT = 200
MEDIUM_COUNT = 150
HARD_COUNT = 50
TOTAL_CLAIMS = EASY_COUNT + MEDIUM_COUNT + HARD_COUNT  # 400
TOTAL_EVIDENCE = 2000

# Batch sizes
CLAIMS_BATCH_SIZE = 10
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5  # seconds


def _hex4() -> str:
    return uuid.uuid4().hex[:4]


def _call_gemini(prompt: str, retries: int = MAX_RETRIES) -> Any:
    """Call Gemini with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FATAL: Failed after {retries} attempts: {e}")
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{retries} after error: {e}. Waiting {delay}s...")
            time.sleep(delay)


def _load_partial(path: Path) -> list[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    return []


def _save_partial(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _assign_domains_to_batches(
    total_items: int, batch_size: int, domains: list[str]
) -> list[list[str]]:
    """Distribute domains across batches so every domain gets covered."""
    num_batches = (total_items + batch_size - 1) // batch_size
    batches: list[list[str]] = []
    for i in range(num_batches):
        # Each batch gets 2-3 domains, cycling through all of them
        start = (i * 2) % len(domains)
        batch_domains = []
        for j in range(3):
            batch_domains.append(domains[(start + j) % len(domains)])
        batches.append(batch_domains)
    return batches


def generate_claims() -> list[dict]:
    """Generate 400 claims in batches, with crash recovery."""
    existing = _load_partial(CLAIMS_PARTIAL)
    if existing:
        # Count what we already have by level
        easy_done = sum(1 for c in existing if c["task_level"] == "easy")
        med_done = sum(1 for c in existing if c["task_level"] == "medium")
        hard_done = sum(1 for c in existing if c["task_level"] == "hard")
        print(f"Resuming: found {len(existing)} partial claims "
              f"(easy={easy_done}, medium={med_done}, hard={hard_done})")
    else:
        easy_done = med_done = hard_done = 0

    all_claims = list(existing)
    global_idx = len(all_claims)

    # Build generation plan: list of (level, count_needed, ground_truth_note)
    plan: list[tuple[str, int, str]] = []
    if easy_done < EASY_COUNT:
        plan.append(("easy", EASY_COUNT - easy_done,
                      "ground_truth must be 'true' or 'false' (roughly 50/50 split)"))
    if med_done < MEDIUM_COUNT:
        plan.append(("medium", MEDIUM_COUNT - med_done,
                      "ground_truth must be 'true' or 'false' (roughly 50/50 split)"))
    if hard_done < HARD_COUNT:
        plan.append(("hard", HARD_COUNT - hard_done,
                      "ground_truth must ALWAYS be 'uncertain'"))

    for level, needed, gt_note in plan:
        level_desc = {
            "easy": "Simple single-hop factual claims verifiable with 1 query",
            "medium": "Multi-hop claims requiring synthesis of 3-4 evidence pieces",
            "hard": "Claims where evidence genuinely conflicts and the answer is debatable",
        }[level]

        domain_batches = _assign_domains_to_batches(needed, CLAIMS_BATCH_SIZE, DOMAINS)
        generated = 0

        for batch_i, batch_domains in enumerate(domain_batches):
            remaining = needed - generated
            if remaining <= 0:
                break
            batch_count = min(CLAIMS_BATCH_SIZE, remaining)
            domains_str = ", ".join(batch_domains)

            prompt = f"""Generate a JSON array of exactly {batch_count} epistemic claims.

Level: {level} — {level_desc}
{gt_note}

Domains to draw from (use at least 2 of these): {domains_str}
You may also use related sub-domains.

Each claim MUST be a specific, verifiable factual statement (not an opinion).
Vary topics widely. Do NOT repeat topics from common knowledge trivia.

Schema for each item:
{{
  "text": "The specific factual claim text",
  "ground_truth": "true" or "false" or "uncertain",
  "task_level": "{level}",
  "evidence_tags": ["domain1", "domain2"]
}}

Return ONLY the JSON array, no other text."""

            print(f"  [{level}] Batch {batch_i + 1}: generating {batch_count} claims "
                  f"(domains: {domains_str})...")

            result = _call_gemini(prompt)
            if not isinstance(result, list):
                print(f"  WARNING: Expected list, got {type(result)}. Skipping batch.")
                continue

            for item in result:
                global_idx += 1
                item["id"] = f"claim_{global_idx}_{_hex4()}"
                item["task_level"] = level
                if level == "hard":
                    item["ground_truth"] = "uncertain"
                if "evidence_tags" not in item or not item["evidence_tags"]:
                    item["evidence_tags"] = batch_domains[:2]

            all_claims.extend(result)
            generated += len(result)
            _save_partial(CLAIMS_PARTIAL, all_claims)
            print(f"  [{level}] Progress: {generated}/{needed}")

            # Brief pause to respect rate limits
            time.sleep(1)

    print(f"Total claims generated: {len(all_claims)}")
    return all_claims


def generate_evidence(claims: list[dict]) -> list[dict]:
    """Generate ~2000 evidence snippets in batches, with crash recovery."""
    existing = _load_partial(EVIDENCE_PARTIAL)
    if existing:
        print(f"Resuming: found {len(existing)} partial evidence snippets")

    all_evidence = list(existing)
    global_idx = len(all_evidence)

    # Group claims by level
    easy_claims = [c for c in claims if c["task_level"] == "easy"]
    medium_claims = [c for c in claims if c["task_level"] == "medium"]
    hard_claims = [c for c in claims if c["task_level"] == "hard"]

    # Calculate evidence budget
    # easy: ~4-5 per claim => target 900
    # medium: ~5-6 per claim => target 825
    # hard: ~6 per claim => target 300
    # distractors: ~10% => ~200 (but we don't allocate separately, we fold into generation)
    # Subtotal target per level (including ~10% distractors folded in):
    easy_ev_target = 900
    medium_ev_target = 825
    hard_ev_target = 275

    # Estimate how many evidence we already have per "phase" based on count
    already = len(existing)
    # We generate in order: easy -> medium -> hard -> distractors
    # Track completed phases based on partial data

    evidence_plan: list[tuple[str, list[dict], int, str]] = [
        ("easy", easy_claims, easy_ev_target,
         "Generate 4-5 short factual evidence snippets per claim. "
         "Include ~10% distractor snippets (topically related but not directly relevant)."),
        ("medium", medium_claims, medium_ev_target,
         "Generate 5-6 evidence snippets per claim. Each claim requires synthesis of "
         "multiple pieces. Include ~10% distractor snippets."),
        ("hard", hard_claims, hard_ev_target,
         "Generate 6 evidence snippets per claim with CONTRADICTORY pairs: "
         "for each claim produce some snippets that SUPPORT and some that REFUTE the claim. "
         "Include ~10% distractor snippets."),
    ]

    for level, level_claims, target_count, instruction in evidence_plan:
        if not level_claims:
            continue

        # Process claims in batches of 5 (each generating multiple evidence)
        claim_batch_size = 5
        generated_for_level = 0
        ev_per_claim = target_count // len(level_claims)

        for i in range(0, len(level_claims), claim_batch_size):
            if generated_for_level >= target_count:
                break

            batch_claims = level_claims[i:i + claim_batch_size]
            claims_context = json.dumps(
                [{"id": c["id"], "text": c["text"], "ground_truth": c["ground_truth"],
                  "evidence_tags": c["evidence_tags"]}
                 for c in batch_claims],
                indent=2,
            )
            remaining = target_count - generated_for_level
            ev_count = min(ev_per_claim * len(batch_claims), remaining)

            prompt = f"""Generate a JSON array of exactly {ev_count} evidence snippets for the following {level}-level claims.

{instruction}

Claims:
{claims_context}

Each evidence snippet should be 1-3 sentences of factual information.
Use varied relevance_tags drawn from the claims' evidence_tags and related terms.

Schema for each item:
{{
  "text": "The factual evidence text (1-3 sentences)",
  "relevance_tags": ["tag1", "tag2", "tag3"]
}}

Return ONLY the JSON array."""

            print(f"  [{level}] Evidence batch {i // claim_batch_size + 1}: "
                  f"generating ~{ev_count} snippets for {len(batch_claims)} claims...")

            result = _call_gemini(prompt)
            if not isinstance(result, list):
                print(f"  WARNING: Expected list, got {type(result)}. Skipping batch.")
                continue

            for item in result:
                global_idx += 1
                item["id"] = f"ev_{global_idx}_{_hex4()}"
                if "relevance_tags" not in item or not item["relevance_tags"]:
                    item["relevance_tags"] = ["general"]

            all_evidence.extend(result)
            generated_for_level += len(result)
            _save_partial(EVIDENCE_PARTIAL, all_evidence)
            print(f"  [{level}] Evidence progress: {generated_for_level}/{target_count}")

            time.sleep(1)

    print(f"Total evidence generated: {len(all_evidence)}")
    return all_evidence


def validate_data(
    claims_path: Path = CLAIMS_FINAL,
    evidence_path: Path = EVIDENCE_FINAL,
) -> bool:
    """Validate generated data meets all requirements. Returns True if valid."""
    print("\n=== Data Validation ===\n")
    errors: list[str] = []

    # Load data
    if not claims_path.exists():
        print(f"ERROR: {claims_path} not found.")
        return False
    if not evidence_path.exists():
        print(f"ERROR: {evidence_path} not found.")
        return False

    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.loads(f.read())
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence = json.loads(f.read())

    # --- Claims checks ---
    total_claims = len(claims)
    easy = [c for c in claims if c.get("task_level") == "easy"]
    medium = [c for c in claims if c.get("task_level") == "medium"]
    hard = [c for c in claims if c.get("task_level") == "hard"]

    print(f"Claims total:   {total_claims} (target: {TOTAL_CLAIMS})")
    print(f"  Easy:         {len(easy)} (target: {EASY_COUNT})")
    print(f"  Medium:       {len(medium)} (target: {MEDIUM_COUNT})")
    print(f"  Hard:         {len(hard)} (target: {HARD_COUNT})")

    if total_claims != TOTAL_CLAIMS:
        errors.append(f"Expected {TOTAL_CLAIMS} claims, got {total_claims}")
    if len(easy) != EASY_COUNT:
        errors.append(f"Expected {EASY_COUNT} easy claims, got {len(easy)}")
    if len(medium) != MEDIUM_COUNT:
        errors.append(f"Expected {MEDIUM_COUNT} medium claims, got {len(medium)}")
    if len(hard) != HARD_COUNT:
        errors.append(f"Expected {HARD_COUNT} hard claims, got {len(hard)}")

    # Check ground_truth distribution
    easy_true = sum(1 for c in easy if c.get("ground_truth") == "true")
    easy_false = sum(1 for c in easy if c.get("ground_truth") == "false")
    med_true = sum(1 for c in medium if c.get("ground_truth") == "true")
    med_false = sum(1 for c in medium if c.get("ground_truth") == "false")
    hard_uncertain = sum(1 for c in hard if c.get("ground_truth") == "uncertain")

    print(f"\n  Easy true/false:    {easy_true}/{easy_false}")
    print(f"  Medium true/false:  {med_true}/{med_false}")
    print(f"  Hard uncertain:     {hard_uncertain}/{len(hard)}")

    if hard_uncertain != len(hard):
        errors.append(f"All hard claims must have ground_truth='uncertain', "
                      f"but only {hard_uncertain}/{len(hard)} do")

    # Check for invalid ground_truth values
    for c in easy + medium:
        if c.get("ground_truth") not in ("true", "false"):
            errors.append(f"Claim {c.get('id')}: easy/medium must be true/false, "
                          f"got '{c.get('ground_truth')}'")
            break  # Only report first

    # Duplicate claim IDs
    claim_ids = [c.get("id") for c in claims]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("Duplicate claim IDs found")

    # Empty text
    empty_claims = [c for c in claims if not c.get("text", "").strip()]
    if empty_claims:
        errors.append(f"{len(empty_claims)} claims have empty text")

    # Domain coverage
    all_tags: set[str] = set()
    for c in claims:
        for tag in c.get("evidence_tags", []):
            all_tags.add(tag.lower())
    print(f"\n  Unique claim domains: {len(all_tags)}")
    if len(all_tags) < 15:
        errors.append(f"Need 15+ unique domains, got {len(all_tags)}: {sorted(all_tags)}")

    # --- Evidence checks ---
    total_ev = len(evidence)
    print(f"\nEvidence total: {total_ev} (target: {TOTAL_EVIDENCE})")

    if total_ev != TOTAL_EVIDENCE:
        errors.append(f"Expected {TOTAL_EVIDENCE} evidence, got {total_ev}")

    # Duplicate evidence IDs
    ev_ids = [e.get("id") for e in evidence]
    if len(ev_ids) != len(set(ev_ids)):
        errors.append("Duplicate evidence IDs found")

    # Empty evidence text
    empty_ev = [e for e in evidence if not e.get("text", "").strip()]
    if empty_ev:
        errors.append(f"{len(empty_ev)} evidence snippets have empty text")

    # Evidence domain coverage
    ev_tags: set[str] = set()
    for e in evidence:
        for tag in e.get("relevance_tags", []):
            ev_tags.add(tag.lower())
    print(f"  Unique evidence tags: {len(ev_tags)}")

    # --- Summary ---
    print(f"\n{'=' * 40}")
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("VALIDATION PASSED")
        return True


def finalize(claims: list[dict], evidence: list[dict]) -> None:
    """Trim to exact targets and write final files."""
    # Trim claims to exact counts if over-generated
    easy = [c for c in claims if c["task_level"] == "easy"][:EASY_COUNT]
    medium = [c for c in claims if c["task_level"] == "medium"][:MEDIUM_COUNT]
    hard = [c for c in claims if c["task_level"] == "hard"][:HARD_COUNT]
    final_claims = easy + medium + hard

    # Re-index claim IDs for clean numbering
    for i, c in enumerate(final_claims):
        c["id"] = f"claim_{i + 1}_{_hex4()}"

    # Trim evidence
    final_evidence = evidence[:TOTAL_EVIDENCE]
    for i, e in enumerate(final_evidence):
        e["id"] = f"ev_{i + 1}_{_hex4()}"

    _save_partial(CLAIMS_FINAL, final_claims)
    _save_partial(EVIDENCE_FINAL, final_evidence)

    print(f"\nFinal output: {len(final_claims)} claims -> {CLAIMS_FINAL}")
    print(f"Final output: {len(final_evidence)} evidence -> {EVIDENCE_FINAL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate epistemic claims and evidence")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing data files")
    parser.add_argument("--clean", action="store_true",
                        help="Remove partial files and start fresh")
    args = parser.parse_args()

    if args.validate_only:
        valid = validate_data()
        sys.exit(0 if valid else 1)

    if args.clean:
        for p in [CLAIMS_PARTIAL, EVIDENCE_PARTIAL]:
            if p.exists():
                p.unlink()
                print(f"Removed {p}")

    print("=" * 50)
    print("Epistemic Data Generator")
    print(f"Targets: {TOTAL_CLAIMS} claims, {TOTAL_EVIDENCE} evidence")
    print("=" * 50)

    # Phase 1: Generate claims
    print("\n--- Phase 1: Generating Claims ---")
    claims = generate_claims()

    # Phase 2: Generate evidence
    print("\n--- Phase 2: Generating Evidence ---")
    evidence = generate_evidence(claims)

    # Phase 3: Finalize
    print("\n--- Phase 3: Finalizing ---")
    finalize(claims, evidence)

    # Phase 4: Validate
    valid = validate_data()

    # Cleanup partial files on success
    if valid:
        for p in [CLAIMS_PARTIAL, EVIDENCE_PARTIAL]:
            if p.exists():
                p.unlink()
        print("\nPartial files cleaned up. Done!")
    else:
        print("\nPartial files kept for debugging.")

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
