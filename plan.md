# Round 1 Submission Plan: OpenRange Cybersecurity Simulation

This document outlines the plan for building a top-tier OpenEnv environment based on a multi-agent cybersecurity simulation. This concept, inspired by projects like Microsoft's CyberBattleSim, directly addresses the competition's requirements for a novel, real-world task.

## 1. Core Concept: "OpenRange" - A Red vs. Blue Cyber Gym

The environment will simulate a small corporate network. An AI agent will take on the role of either an attacker (Red Team) or a defender (Blue Team) and learn to achieve its objectives. This provides a dynamic and engaging "game" for AI agents to play.

This approach is:
- **Novel & Creative**: It's a unique and exciting problem domain that stands out.
- **Real-world**: It models fundamental concepts of cybersecurity and network defense.
- **Perfect for RL**: The competitive nature of attack/defense is a classic fit for reinforcement learning.

## 2. Graded Tasks (Easy → Medium → Hard)

The tasks will be from the perspective of the **Red Team agent**, as this is easier to grade objectively.

### Task 1 (Easy): `Vulnerability Scan`
- **Objective**: Given the network topology, scan all machines and identify a specific, predefined vulnerability (e.g., an open port with a vulnerable service).
- **Grader**: Scores 1.0 if the agent correctly identifies the vulnerable machine and port.

### Task 2 (Medium): `Scan and Exploit`
- **Objective**: First, complete the `Vulnerability Scan`. Then, use a specific "exploit" action on the identified vulnerability to gain initial access to the machine.
- **Grader**: Scores based on successful identification (partial credit) and successful exploitation (full credit).

### Task 3 (Hard): `Exploit and Escalate`
- **Objective**: The full "kill chain." Scan for the vulnerability, exploit it to gain access, and then use a "privilege escalation" action to gain root access on the target machine.
- **Grader**: Provides partial credit for scanning and initial exploit, and full credit (1.0) only upon successful privilege escalation.

## 3. The Simulated Environment & Agent Actions

- **Network**: A simple 3-node network (e.g., Web Server, Database, User Workstation). The Web Server will have a known vulnerability.
- **Red Team Actions**:
    - `scan(ip)`: Returns open ports and services on a machine.
    - `exploit(ip, port)`: Attempts to exploit a vulnerability.
    - `escalate(ip)`: Attempts to gain root privileges.
- **Blue Team (for future development)**: A simple, rule-based Blue Team will be part of the environment (e.g., has a chance to detect and block the Red Team's actions), but the graded tasks will focus on the Red Team agent's performance.

## 4. Meaningful Reward Function (for the Red Team Agent)

-   **`+0.3`**: For each new machine successfully scanned and its vulnerabilities mapped.
-   **`+0.4`**: For successfully gaining initial access via an exploit.
-   **`+0.3`**: For successfully escalating privileges to root.
-   **`-0.1`**: For each failed action (to discourage brute-forcing).
-   **`-0.5`**: If the Blue Team agent "detects" and "blocks" the action.

## 5. Implementation Roadmap

1.  **Project Setup**:
    -   Update `openenv.yaml` with the new environment name and description.
    -   Ensure `Dockerfile` is ready for a Python/FastAPI environment.
    -   Confirm Python project structure (`src`, `tests`).

2.  **Core Environment Logic**:
    -   Implement the main `OpenRangeEnv` class.
    -   Define Pydantic models for cyber-specific `Action` (scan, exploit, escalate) and `Observation` (scan results, access level).
    -   Implement the `step()`, `reset()`, and `state()` methods to manage the simulation.

3.  **Simulation & Task Logic**:
    -   Build the simulated network graph.
    -   Implement the logic for the Red Team actions and the rule-based Blue Team agent.
    -   Implement the three graded tasks and their programmatic graders.

4.  **Baseline Script**:
    -   Create a baseline `inference.py` that demonstrates an agent solving the tasks.

5.  **Documentation & Deployment**:
    -   Write a new `README.md` explaining the cybersecurity environment.
    -   Deploy to Hugging Face Spaces.
