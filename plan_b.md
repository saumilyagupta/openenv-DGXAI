# Round 1 Submission Plan (v2 with Security Review): OpenRange Cyber Gym

This document outlines an enhanced plan for building a top-tier OpenEnv environment based on a multi-agent cybersecurity simulation. This version incorporates security review principles to increase realism and align more closely with the competition's evaluation criteria for real-world utility and task quality.

## 1. Core Concept: "OpenRange" - A Red vs. Blue Cyber Gym

The environment will simulate a small corporate network. An AI agent will take on the role of an attacker (Red Team) to learn and achieve its objectives against a rule-based defender (Blue Team).

This approach is:
- **Novel & Creative**: A unique problem domain that stands out.
- **Real-world**: Models fundamental concepts of cybersecurity and network defense, enhanced with realistic threat modeling.
- **Perfect for RL**: The competitive nature of attack/defense is a classic fit for reinforcement learning.

## 2. Graded Tasks (Easy → Medium → Hard)

The tasks are from the perspective of the **Red Team agent**. The security enhancements focus on making the exploit process more nuanced than a single action.

### Task 1 (Easy): `Vulnerability Scan`
- **Objective**: Given the network topology, scan all machines and identify a specific, predefined vulnerability (e.g., an open port running an outdated web server version).
- **Grader**: Scores 1.0 if the agent correctly identifies the vulnerable machine, port, and service.

### Task 2 (Medium): `Scan and Exploit`
- **Objective**: First, complete the `Vulnerability Scan`. Then, use a specific `exploit` action on the identified vulnerability, providing the correct payload type (e.g., `SQL_INJECTION`) to gain initial access.
- **Grader**: Scores based on successful identification (partial credit) and successful exploitation with the correct payload (full credit).

### Task 3 (Hard): `Exploit, Escalate, and Persist`
- **Objective**: The full "kill chain." Scan for the vulnerability, exploit it to gain access, use a `privilege_escalation` action based on a *second* vulnerability found post-access (e.g., misconfigured `sudo`), and finally, establish persistence.
- **Grader**: Provides partial credit for scanning and initial exploit, with full credit (1.0) only upon successful privilege escalation.

## 3. The Simulated Environment & Agent Actions

- **Network**: A simple 3-node network (e.g., Web Server, Database, User Workstation). The Web Server will have a known, realistic vulnerability (e.g., outdated Apache Struts).
- **Red Team Actions**:
    - `scan(ip)`: Returns open ports, services, and versions.
    - `exploit(ip, port, payload_type)`: Attempts to exploit a vulnerability with a specified payload.
    - `escalate(ip, method)`: Attempts to gain root privileges using a specific method.
- **Blue Team (Enhanced Logic)**: A rule-based Blue Team agent will monitor the network and react to suspicious activity. Its rules will be more deterministic:
    - **Trigger**: "Block IP after 3 failed `exploit` attempts from the same source."
    - **Trigger**: "Alert and temporarily lock account if `scan` traffic exceeds a certain threshold in a short period."

## 4. Meaningful Reward Function (for the Red Team Agent)

-   **`+0.3`**: For each new machine successfully scanned and its vulnerabilities mapped.
-   **`+0.4`**: For successfully gaining initial access via an exploit.
-   **`+0.3`**: For successfully escalating privileges to root.
-   **`-0.1`**: For each failed or noisy action (to discourage brute-forcing).
-   **`-0.5`**: If the Blue Team agent detects and blocks the action based on its rules.

## 5. Implementation Roadmap

1.  **Threat Modeling & Vulnerability Design**:
    -   Define network assets (e.g., "customer PII" on the DB).
    -   Model a realistic attack path from initial scan to final objective.
    -   Choose specific, well-understood vulnerabilities (e.g., from OWASP Top 10) to implement in the simulation.

2.  **Project Setup**:
    -   Update `openenv.yaml` with the new environment name and description.
    -   Ensure `Dockerfile` is ready for a Python/FastAPI environment.
    -   Confirm Python project structure (`src`, `tests`).

3.  **Core Environment Logic**:
    -   Implement the main `OpenRangeEnv` class.
    -   Define Pydantic models for `Action` (scan, exploit, escalate) and `Observation` (scan results, access level).
    -   Implement the `step()`, `reset()`, and `state()` methods.

4.  **Simulation & Task Logic**:
    -   Build the simulated network graph based on the threat model.
    -   Implement the logic for the Red Team actions and the rule-based Blue Team agent.
    -   Implement the three graded tasks and their programmatic graders.

5.  **Sandbox Validation**:
    -   Review the implementation of agent actions to prevent any "escape" from the simulation (e.g., ensure no command injection vulnerabilities in the environment code itself).
    -   Add unit tests to confirm agent actions are properly constrained.

6.  **Baseline Script**:
    -   Create a baseline `inference.py` that demonstrates an agent solving the tasks.

7.  **Documentation & Deployment**:
    -   Write a new `README.md` explaining the cybersecurity environment and the enhanced, more realistic tasks.
    -   Deploy to Hugging Face Spaces.
