import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"


# ---------------- PROMPT ---------------- #
def build_prompt(topic, num_questions, difficulty):
    return f"""
Generate {num_questions} {difficulty} multiple-choice questions about {topic}.

Return ONLY raw JSON (no markdown, no ```).

Format:
[
  {{
    "question": "string",
    "options": {{
      "A": "string",
      "B": "string",
      "C": "string",
      "D": "string"
    }},
    "correct_answer": "A",
    "explanation": "short explanation"
  }}
]

Rules:
- Do NOT wrap in ```json
- Do NOT add extra text
- Ensure correct_answer matches correct option
"""


# ---------------- API ---------------- #
def make_api_request(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
    except requests.exceptions.RequestException as error:
        print("❌ Network error:", error)
        return None

    if response.status_code != 200:
        print("❌ API Error:", response.status_code)
        print(response.text)
        return None

    try:
        return response.json()
    except ValueError:
        print("❌ Invalid JSON response")
        return None


# ---------------- CLEAN JSON ---------------- #
def clean_json(raw_text):
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "")

    return raw_text.strip()


# ---------------- EXTRACT ---------------- #
def extract_text_from_response(response_json):
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("❌ Unexpected API format")
        return None


# ---------------- VALIDATION ---------------- #
def validate_question(q):
    try:
        if "question" not in q:
            return False

        if "options" not in q or len(q["options"]) != 4:
            return False

        if "correct_answer" not in q:
            return False

        if q["correct_answer"] not in q["options"]:
            return False

        if "explanation" not in q:
            return False

        return True
    except:
        return False


# ---------------- PARSE ---------------- #
def parse_questions(raw_text):
    cleaned = clean_json(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON")
        print(cleaned)
        return []

    valid_questions = []

    for q in data:
        if validate_question(q):
            valid_questions.append(q)
        else:
            print("⚠️ Skipped invalid question:", q)

    return valid_questions


# ---------------- MAIN ---------------- #
def generate_questions(topic, num_questions, difficulty):
    prompt = build_prompt(topic, num_questions, difficulty)

    response_json = make_api_request(prompt)
    if not response_json:
        return []

    raw_text = extract_text_from_response(response_json)
    if not raw_text:
        return []

    print("\nRAW RESPONSE:\n", raw_text)

    questions = parse_questions(raw_text)

    print("\nVALID QUESTIONS:", len(questions))

    return questions
