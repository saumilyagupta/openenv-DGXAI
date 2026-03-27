# Local Setup Guide for Teammates

Welcome to the **OpenRange Cyber Gym**! This guide will walk you through cloning the repository, setting up your local environment, and testing the OpenEnv endpoints.

---

## 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+**
- **Git**
- **Docker** (Optional, but recommended for testing the Hugging Face Space build)

---

## 2. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/saumilyagupta/openenv-DGXAI.git
cd openenv-DGXAI
```

---

## 3. Environment Setup (Python)

It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. API Keys (.env file)
To run the automated `inference.py` baseline script, you need an OpenAI API key.

1. Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
2. Open the `.env` file and add your key:
   ```env
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

---

## 5. Running the Application Locally

You can run the environment locally using `uvicorn` or Docker.

### Option A: Using Uvicorn (Fastest for Development)
Start the FastAPI server on port 7860 (this matches our Hugging Face Space configuration):
```bash
uvicorn src.main:app --host 0.0.0.0 --port 7860
```
- Open **http://localhost:7860/docs** in your browser to interact with the API via the Swagger UI.

### Option B: Using Docker (Best for testing production builds)
```bash
# Build the image
docker build -t openrange-cyber .

# Run the container
docker run -p 7860:7860 openrange-cyber
```
- The API will be available at **http://localhost:7860**.

---

## 6. Testing the Environment

### Test the Endpoints Manually
With the server running (either via Uvicorn or Docker), open a new terminal window and run these `curl` commands:

```bash
# Check if the server is alive
curl http://localhost:7860/

# See the available tasks and the Action JSON schema
curl http://localhost:7860/tasks

# Start Task 1 and reset the environment
curl -X POST http://localhost:7860/set_task/1
curl -X POST http://localhost:7860/reset

# Take an action as the Red Team AI
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type": "scan", "target_ip": "192.168.1.10"}'
```

### Run the Baseline Inference Agent
The `inference.py` script automatically talks to the OpenAI API, fetches decisions, and sends them to your local endpoints to solve the tasks.

*(Ensure your `.env` file is set up and the FastAPI server is running on port `7860` before executing).*

```bash
python inference.py
```
**Expected Output:** You will see the AI agent's step-by-step logic printed to the console, concluding with perfect `1.0` scores for all three tasks.

---

## 7. Hugging Face Deployment

If you make changes to the code and want to push them to the live Hugging Face Space:

1. Add the Hugging Face remote (you only need to do this once):
   ```bash
   git remote add space https://huggingface.co/spaces/DGXAI/openenv-rl-hackathon
   ```
2. Push the code to Hugging Face:
   ```bash
   git push space main
   ```
*(You will need your Hugging Face username and a Write Access Token as the password).*