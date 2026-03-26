# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

This project, the "Autonomous SaaS Factory," is an OpenEnv environment where a team of AI agents collaborates to build, deploy, and manage a full-fledged SaaS application. The system is designed around a Reinforcement Learning (RL) framework where a CEO agent learns the optimal policy for building a startup.

### Agent Team Roles

The core of the system is a team of specialized AI agents:

*   **CEO Agent (Orchestrator)**: The primary learning agent that functions as the "brain" of the operation. It analyzes the state, decides on the next task, and delegates it.
*   **CTO Agent (Architect)**: Responsible for producing technical designs, including database schemas and API contracts.
*   **Product Manager Agent (Planner)**: Translates business goals into a backlog of well-defined tasks for the engineering team.
*   **Software Engineer Agent (Builder)**: Implements code for specific tasks, including writing unit and integration tests.
*   **DevOps Agent (Deployer)**: Manages cloud infrastructure and CI/CD pipelines, primarily using Supabase and Vercel CLI tools.

### Development Workflow

The development process is a sequence of interactions between these agents, managed by the CEO agent. The workflow includes task delegation, code implementation, committing, and deployment. The system includes error handling, where build failures are reported back to the CEO agent to create recovery tasks.

The project uses Git for version control, Supabase for the infrastructure platform, and Vercel for hosting.

## Common Commands

The available documentation does not specify any commands for building, linting, or running tests for this project.
