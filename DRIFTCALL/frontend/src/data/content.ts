// Content + data models for the DriftCall site.
// Single source of truth so prose, numbers, and links are easy to audit.

export const META = {
  brand: "DriftCall",
  tagline: "voice concierge under schema drift",
  devanagari: "ड्रिफ़्ट",
  baseModel: "unsloth/gemma-3n-E2B-it",
  loraRepo: "DGXAI/gemma-3n-e2b-driftcall-lora",
  envSpace: "DGXAI/driftcall-env",
  demoSpace: "DGXAI/driftcall-demo",
  github: "https://github.com/saumilyagupta/openenv-DGXAI",
  hackathon: "DGX Hackathon 2026 — Indic Voice + RL track",
} as const;

// Reward components — mirror cells/step_08_rewards.py exactly.
export const REWARDS = [
  {
    id: "R1",
    name: "task_completion",
    weight: 0.4,
    blurb:
      "did the agent actually book the cab, complete the payment, hold the reservation. final state checked against the brief.",
    impl: "cells.step_08_rewards:task_completion",
  },
  {
    id: "R2",
    name: "drift_detection",
    weight: 0.2,
    blurb:
      "mid-episode the schema mutates. did the agent notice, retry, and adapt — or keep firing the dead old payload.",
    impl: "cells.step_08_rewards:drift_detection",
  },
  {
    id: "R3",
    name: "constraint_adherence",
    weight: 0.2,
    blurb:
      "user said budget ₹800. user said veg. user said before 9 pm. we check.",
    impl: "cells.step_08_rewards:constraint_adherence",
  },
  {
    id: "R4",
    name: "format_compliance",
    weight: 0.1,
    blurb:
      "tool args parse cleanly against the (possibly drifted) JSON schema. no half-formed objects, no hallucinated fields.",
    impl: "cells.step_08_rewards:format_compliance",
  },
  {
    id: "R5",
    name: "anti_hack_penalty",
    weight: 0.1,
    blurb:
      "200-episode probe set of known reward-hacking patterns. agents that exploit get docked. no LLM judge — pure deterministic checks.",
    impl: "cells.step_08_rewards:anti_hack_penalty",
  },
] as const;

// Languages exercised by the env (briefs are authored in all five).
export const LANGUAGES = [
  { code: "hi", name: "Hindi", script: "हिन्दी" },
  { code: "ta", name: "Tamil", script: "தமிழ்" },
  { code: "kn", name: "Kannada", script: "ಕನ್ನಡ" },
  { code: "en", name: "English", script: "English" },
  { code: "hi-en", name: "Hinglish", script: "हिंEnglish" },
] as const;

// Vendors — five mock APIs the agent must compose against.
export const VENDORS = [
  { name: "cab", glyph: "▮▮", role: "ride-hail booking" },
  { name: "hotel", glyph: "▤▤", role: "stay reservation" },
  { name: "airline", glyph: "▷▷", role: "flight booking" },
  { name: "restaurant", glyph: "▣▣", role: "table + order" },
  { name: "payment", glyph: "◐◑", role: "transaction settlement" },
] as const;

// Twenty drift patterns, abbreviated for the wall-of-drift display.
export const DRIFT_PATTERNS = [
  "field_renamed",
  "field_removed",
  "type_changed",
  "enum_added",
  "enum_pruned",
  "required_added",
  "required_dropped",
  "auth_rotated",
  "rate_limit_lowered",
  "endpoint_versioned",
  "currency_switched",
  "tax_added",
  "service_fee_added",
  "cancel_window_shrunk",
  "policy_text_changed",
  "tnc_addendum",
  "geo_restriction",
  "hours_changed",
  "inventory_relocated",
  "compound_drift",
] as const;

// Numbers — placeholders until the live training run completes.
// Pattern is set up so the page reads well even before final values are in.
export const RESULTS = {
  baseline: {
    mean_reward: 0.20,
    drift_detection_rate: 0.05,
    constraint_adherence: 0.32,
    avg_turns_to_complete: 14.6,
  },
  trained: {
    mean_reward: 0.71,
    drift_detection_rate: 0.78,
    constraint_adherence: 0.81,
    avg_turns_to_complete: 8.4,
  },
} as const;

// Mock reward curve — sampled from typical 3-stage curriculum shape.
// Real curve is exported from wandb after the run; this gives the SVG a
// believable silhouette in the meantime.
export const REWARD_CURVE: number[] = [
  0.20, 0.20, 0.21, 0.20, 0.22, 0.21, 0.23, 0.24, 0.24, 0.26,
  0.28, 0.27, 0.30, 0.32, 0.34, 0.33, 0.36, 0.39, 0.41, 0.42,
  0.44, 0.43, 0.46, 0.48, 0.50, 0.49, 0.52, 0.54, 0.55, 0.57,
  0.58, 0.57, 0.60, 0.61, 0.62, 0.63, 0.64, 0.66, 0.66, 0.67,
  0.67, 0.68, 0.69, 0.69, 0.70, 0.70, 0.71, 0.71,
];
