---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

This project is the **OpenRange Cyber Gym**, an OpenEnv-compliant environment built for the DGXAI RL Hackathon. It replaces the old "Autonomous SaaS Factory" concept.

The environment simulates a "Red vs. Blue" cybersecurity exercise where an AI agent acts as a Red Team operator attempting to exploit a 3-node corporate network, while a hardcoded, rule-based Blue Team monitors for noisy behavior and blocks brute-force attacks. The environment is specifically designed to train and evaluate Reinforcement Learning (RL) agents.

### Core Components

*   **`src/simulation.py` (The Engine):** Contains the `CyberSimulation` class which manages the network state (nodes, open ports, vulnerabilities like `SQL_INJECTION` and `sudo_misconfig`), the Blue Team alert logic, and strict anti-reward-hacking checks.
*   **`src/main.py` (The API):** A FastAPI application that exposes the required OpenEnv specification endpoints (`/step`, `/reset`, `/state`) alongside hackathon-specific grading endpoints (`/tasks`, `/grader`, `/baseline`).
*   **`src/models.py` (The Schema):** Defines the Pydantic models for the Red Team `Action`, `Observation`, `State`, and `Reward`.
*   **`inference.py` (The Baseline Agent):** A standalone script that uses the OpenAI API (or Nvidia NIMs) to simulate an RL agent interacting with the environment to solve the 3 graded tasks.

### Development Workflow & Guidelines

*   **OpenEnv Spec Strictness:** Any changes to the API endpoints must strictly adhere to the OpenEnv specification and the hackathon's `problem_statement_and_guidelines.md`.
*   **Anti-Reward Hacking:** When modifying the reward logic in `simulation.py`, always ensure that agents cannot "farm" points by repeating the same action (e.g., repeatedly scanning known nodes or escalating privileges on already-rooted machines). Rewards must be tied to *new* discoveries or *successful state changes*.
*   **Determinism:** The `/grader` endpoint must return perfectly objective, reproducible scores between 0.0 and 1.0 based on the simulation state.
*   **Containerization:** The project runs via Docker on Hugging Face Spaces. The FastAPI server must bind to `0.0.0.0` on port `7860`.

## Common Commands

*   **Local Server (Uvicorn):** `uvicorn src.main:app --host 0.0.0.0 --port 7860`
*   **Local Server (Docker):** `docker build -t openrange .` then `docker run -p 7860:7860 openrange`
*   **Run Baseline Agent:** `python inference.py` (Requires `OPENAI_API_KEY` in `.env`)
*   **Run Nvidia Nemotron Agent:** `python nemotron_test/nemo_agent.py` (Requires `NVIDIA_API_KEY` in `.env`)
*   **Deploy to Hugging Face:** `git push space main` (Requires setting up the `space` remote with an HF write token)
