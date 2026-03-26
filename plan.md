# Round 1 Submission Plan

This document outlines the plan for building an OpenEnv environment that meets the requirements of the Round 1 problem statement. The goal is to create a focused, high-quality environment that simplifies the broader "Autonomous SaaS Factory" vision into a tangible and competitive submission.

## 1. Core Concept: "SaaS Factory as a Service"

The environment will simulate a core real-world task for a backend developer: taking a feature request and turning it into a deployed API endpoint. An AI agent interacting with this environment will need to learn the correct sequence of development operations.

This approach is:

- **Real-world**: It models a genuine and valuable software development workflow.
- **Focused**: It avoids the complexity of a multi-agent learning system, which is out of scope for Round 1.
- **Compliant**: It allows us to cleanly implement the `step()`, `reset()`, `state()` API, tasks, graders, and a meaningful reward function.

## 2. Graded Tasks (Easy → Medium → Hard)

The environment will feature three distinct tasks with a clear difficulty progression.

### Task 1 (Easy): `generate-schema`

- **Objective**: Given a natural language feature description (e.g., "a blog post with a title and content"), generate a valid SQL DDL file.
- **Grader**: Checks if the output file is created and contains syntactically valid SQL. Scores 1.0 for success, 0.0 for failure.

### Task 2 (Medium): `generate-and-implement`

- **Objective**: Given a feature description, first generate the SQL schema, then generate the corresponding Supabase migration file and a basic API Edge Function.
- **Grader**: Checks for a valid schema, a correctly named migration file, and a syntactically correct Edge Function file.

### Task 3 (Hard): `generate-and-deploy`

- **Objective**: The full end-to-end task. Given a feature description, generate the schema, implement the backend code, and deploy it using mocked Supabase CLI commands within the environment.
- **Grader**: Checks for successful completion of all steps. For the final submission, this would be adapted to call a real deployed URL.

## 3. Meaningful Reward Function

The reward function will provide partial progress signals to guide the agent.

- `**+0.3`**: For successfully creating a valid SQL schema.
- `**+0.3**`: For successfully creating the migration and API implementation files.
- `**+0.4**`: For a successful final deployment (in the 'hard' task).
- `**-0.5**`: For any action that results in a simulated error (e.g., failed CLI command, invalid code generation).

## 4. Implementation Roadmap

1. **Project Setup**:
  - Create `openenv.yaml` with environment metadata.
  - Create a working `Dockerfile` for containerized execution.
  - Set up a basic Python project structure (`src`, `tests`).
2. **Core Environment Logic**:
  - Implement the main `SaaSFactoryEnv` class.
  - Define the Pydantic models for `Observation`, `Action`, and `Reward`.
  - Implement the `step()`, `reset()`, and `state()` methods.
3. **Task and Grader Implementation**:
  - Implement the logic for the three tasks (`generate-schema`, `generate-and-implement`, `generate-and-deploy`).
  - Implement the corresponding programmatic graders for each task.
4. **Baseline Script**:
  - Create a baseline `inference.py` script that uses the OpenAI API to interact with the environment and solve the tasks.
5. **Documentation**:
  - Write a comprehensive `README.md` that details the environment, setup instructions, and baseline scores.
6. **Deployment**:
  - Package the environment and deploy it as a Hugging Face Space.

