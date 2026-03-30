import os
import json
import uuid
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Please set GEMINI_API_KEY in .env file.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

def generate_claims():
    print("Generating claims...")
    prompt = """
    Generate a JSON array of 10 claims.
    Include 4 "easy" claims (single hop), 4 "medium" claims (multi-hop), and 2 "hard" claims (contradictory/uncertain).
    The ground_truth should be "true", "false", or "uncertain".
    Schema per item:
    {
      "id": "claim_001",
      "text": "The boiling point of water at standard atmospheric pressure is 100°C",
      "ground_truth": "true",
      "task_level": "easy",
      "evidence_tags": ["physics", "water"]
    }
    """
    response = model.generate_content(prompt)
    try:
        claims = json.loads(response.text)
        # ensure ids are unique
        for i, c in enumerate(claims):
            c["id"] = f"claim_{i+1}_{uuid.uuid4().hex[:4]}"

        with open("data/claims.json", "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2)
        print(f"Generated {len(claims)} claims.")
        return claims
    except Exception as e:
        print("Failed to generate claims", e)
        return []

def generate_evidence(claims):
    print("Generating evidence...")
    prompt = """
    Generate a JSON array of 50 evidence snippets.
    These snippets should provide the necessary facts to verify the claims below.
    For the "hard" (uncertain) claims, provide contradictory snippets.
    Schema per item:
    {
      "id": "ev_001",
      "text": "Water boils at 100 degrees Celsius (212°F) at sea level (1 atm pressure).",
      "relevance_tags": ["physics", "water"]
    }
    """
    # Just sending the claims as context
    prompt += "\nClaims context:\n" + json.dumps([c["text"] for c in claims], indent=2)

    response = model.generate_content(prompt)
    try:
        evidence = json.loads(response.text)
        # ensure ids are unique
        for i, e in enumerate(evidence):
            e["id"] = f"ev_{i+1}_{uuid.uuid4().hex[:4]}"

        with open("data/evidence.json", "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        print(f"Generated {len(evidence)} evidence snippets.")
    except Exception as e:
        print("Failed to generate evidence", e)

if __name__ == "__main__":
    claims = generate_claims()
    if claims:
        generate_evidence(claims)
