import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"
REQUEST_TIMEOUT_SECONDS = 30
MAX_GENERATION_ATTEMPTS = 3


def build_prompt(topic, num_questions, difficulty):
    return f"""
Generate EXACTLY {num_questions} {difficulty} multiple-choice questions about {topic}.

Return ONLY valid JSON.

Rules:
- MUST return exactly {num_questions} questions
- Each question must have 4 options
- Use simple clear language
- Explanation max 2 lines

Format:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "A",
    "explanation": "..."
  }}
]
"""


def make_api_request(prompt):
    if not API_KEY:
        print("Missing OPENROUTER_KEY in environment.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"API request failed: {error}")
        return None

    try:
        return response.json()
    except ValueError:
        print("Invalid JSON response from API.")
        return None


def clean_json(raw_text):
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "")

    return cleaned.strip()


def extract_json_array(raw_text):
    start_index = raw_text.find("[")
    end_index = raw_text.rfind("]")

    if start_index == -1 or end_index == -1 or end_index < start_index:
        return raw_text

    return raw_text[start_index : end_index + 1]


def extract_text_from_response(response_json):
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("Unexpected API response format.")
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts).strip() or None

    return None


def validate_question(question):
    try:
        prompt_text = question.get("question", "").strip()
        options = question.get("options", {})
        correct_answer = question.get("correct_answer")
        explanation = question.get("explanation", "").strip()

        if not prompt_text:
            return False

        if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C", "D"}:
            return False

        if not all(isinstance(value, str) and value.strip() for value in options.values()):
            return False

        if correct_answer not in options:
            return False

        if not explanation:
            return False

        return True
    except Exception:
        return False


def parse_questions(raw_text):
    cleaned = clean_json(raw_text)
    extracted = extract_json_array(cleaned)

    try:
        data = json.loads(extracted)
    except json.JSONDecodeError:
        print("Failed to parse question JSON.")
        print(extracted)
        return []

    if not isinstance(data, list):
        return []

    valid_questions = []
    for question in data:
        if validate_question(question):
            valid_questions.append(question)

    return valid_questions


def merge_unique_questions(existing_questions, new_questions):
    seen_questions = {question["question"].strip().lower() for question in existing_questions}

    for question in new_questions:
        normalized_question = question["question"].strip().lower()
        if normalized_question in seen_questions:
            continue

        existing_questions.append(question)
        seen_questions.add(normalized_question)

    return existing_questions


def generate_questions(topic, num_questions, difficulty):
    collected_questions = []
    attempts = 0

    while len(collected_questions) < num_questions and attempts < MAX_GENERATION_ATTEMPTS:
        remaining_questions = num_questions - len(collected_questions)
        prompt = build_prompt(topic, remaining_questions, difficulty)

        if collected_questions:
            existing_question_text = "\n".join(
                f'- "{question["question"]}"' for question in collected_questions
            )
            prompt += f"""

Do not repeat any of these existing questions:
{existing_question_text}
"""

        response_json = make_api_request(prompt)
        if not response_json:
            break

        raw_text = extract_text_from_response(response_json)
        if not raw_text:
            break

        parsed_questions = parse_questions(raw_text)
        if not parsed_questions:
            attempts += 1
            continue

        collected_questions = merge_unique_questions(collected_questions, parsed_questions)
        attempts += 1

    return collected_questions[:num_questions]
