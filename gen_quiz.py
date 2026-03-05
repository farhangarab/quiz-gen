import os
import json
import time
import re
from typing import Callable, Dict, Any, List, Optional
import requests

# ---- Configuration ----
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://api.example.com/v1/generate")  # replace with your endpoint
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
DEFAULT_MODEL = "gemini-1.5-flash"

# ---- Exceptions ----
class QuizGenerationError(Exception):
    pass

# ---- Prompt builder ----
def _build_prompt(topic: str, num_questions: int, level: str) -> str:
    """
    Build a strict instruction prompt that asks the model to return only valid JSON.
    """
    return (
        f"You are an expert quiz writer. Create exactly {num_questions} multiple-choice questions "
        f"about **{topic}** at **{level}** difficulty. Output must be a single valid JSON object "
        f"and nothing else. The JSON object must have a top-level key named \"quiz\" whose value "
        f"is an array of question objects. Each question object must include the following keys:\n\n"
        f"  - \"question\": a concise question string.\n"
        f"  - \"options\": an object with keys \"A\",\"B\",\"C\",\"D\" and string values for each option.\n"
        f"  - \"correct_answer\": one of the letters \"A\",\"B\",\"C\",\"D\" (exactly one correct).\n"
        f"  - \"feedback\": a short explanation (1-2 sentences) explaining why the correct answer is correct.\n\n"
        f"Constraints:\n"
        f"  - Do not include any additional keys at the top level.\n"
        f"  - Do not include commentary, markdown, or code fences—only the JSON object.\n"
        f"  - Ensure options are plausible distractors and only one option is correct.\n\n"
        f"Return the JSON object now."
    )

# ---- Model call wrapper (default uses requests) ----
def _default_call_model(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 30) -> str:
    """
    Default HTTP POST to the LLM endpoint. Adjust headers/payload to match your Gemini 1.5 Flash API.
    This function expects the endpoint to return a plain text or JSON response containing the model text.
    Replace GEMINI_API_URL and header format as required by your provider.
    """
    if not GEMINI_API_KEY:
        raise QuizGenerationError("GEMINI_API_KEY environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1200,
        "temperature": 0.2,
        "top_p": 0.95,
        # Add other provider-specific params here
    }

    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise QuizGenerationError(f"Model API returned HTTP {resp.status_code}: {resp.text}") from e

    # Try to extract text from common response shapes
    try:
        data = resp.json()
    except ValueError:
        # If not JSON, return raw text
        return resp.text

    # Common shapes: {"text": "..."} or {"output": "..."} or provider-specific
    if isinstance(data, dict):
        for key in ("text", "output", "content", "response"):
            if key in data and isinstance(data[key], str):
                return data[key]
        # Some APIs return nested choices: {"choices":[{"text": "..."}]}
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                return first["message"].get("content", "")
    # Fallback: stringify
    return json.dumps(data)

# ---- JSON extraction helpers ----
def _extract_json(text: str) -> Optional[str]:
    """
    Try to extract the first JSON object from text. Returns JSON string or None.
    """
    # Quick attempt: if text starts with { or [, assume it's JSON
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    # Otherwise search for first balanced JSON object using regex heuristics
    # Find first '{' and attempt to find matching '}' by counting braces
    start = text.find("{")
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                candidate = text[start:i+1]
                # quick sanity check
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return None
    return None

def _validate_quiz_structure(obj: Any, expected_n: int) -> bool:
    """
    Validate the parsed JSON structure minimally.
    """
    if not isinstance(obj, dict):
        return False
    if "quiz" not in obj or not isinstance(obj["quiz"], list):
        return False
    if len(obj["quiz"]) != expected_n:
        return False
    for q in obj["quiz"]:
        if not isinstance(q, dict):
            return False
        if "question" not in q or "options" not in q or "correct_answer" not in q or "feedback" not in q:
            return False
        if not isinstance(q["options"], dict):
            return False
        if set(q["options"].keys()) != {"A", "B", "C", "D"}:
            return False
        if q["correct_answer"] not in {"A", "B", "C", "D"}:
            return False
    return True

# ---- Public function ----
def generate_quiz(
    topic: str,
    num_questions: int,
    level: str,
    call_model: Callable[[str], str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate a quiz by sending a strict prompt to an LLM and returning parsed JSON.
    - topic: subject of the quiz (e.g., "Python decorators")
    - num_questions: integer > 0
    - level: difficulty label (e.g., "beginner", "intermediate", "advanced")
    - call_model: optional callable(prompt) -> str. If None, uses the default HTTP caller.
    Returns a dict with key "quiz" -> list of question dicts.
    Raises QuizGenerationError on failure.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string.")
    if not isinstance(num_questions, int) or num_questions <= 0:
        raise ValueError("num_questions must be a positive integer.")
    if not isinstance(level, str) or not level.strip():
        raise ValueError("level must be a non-empty string.")

    prompt = _build_prompt(topic.strip(), num_questions, level.strip())
    caller = call_model or (lambda p: _default_call_model(p, model=DEFAULT_MODEL))

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = caller(prompt)
            # Try direct parse
            parsed = None
            # If the model returned JSON-like text, extract it
            json_text = _extract_json(raw)
            if json_text:
                parsed = json.loads(json_text)
            else:
                # If raw is JSON string already
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None

            if parsed and _validate_quiz_structure(parsed, num_questions):
                return parsed

            # If validation failed, attempt to be forgiving: try to find array of questions
            # e.g., model returned a top-level array
            if parsed and isinstance(parsed, list):
                candidate = {"quiz": parsed}
                if _validate_quiz_structure(candidate, num_questions):
                    return candidate

            # If we reach here, treat as failure and retry
            last_err = QuizGenerationError(
                f"Model returned invalid structure on attempt {attempt}. Raw output: {raw[:1000]}"
            )
        except Exception as e:
            last_err = e

        # backoff before retry
        if attempt < max_retries:
            time.sleep(retry_delay * attempt)

    # All retries exhausted
    raise QuizGenerationError(f"Failed to generate valid quiz: {last_err}")

# ---- Example FastAPI integration snippet (minimal) ----
# Place this in your FastAPI app file and import generate_quiz from this module.
#
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
#
# app = FastAPI()
#
# class QuizRequest(BaseModel):
#     topic: str
#     num_questions: int
#     level: str
#
# @app.post("/generate-quiz")
# def api_generate_quiz(req: QuizRequest):
#     try:
#         quiz = generate_quiz(req.topic, req.num_questions, req.level)
#         return quiz
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# ---- Example usage (synchronous) ----
if __name__ == "__main__":
    # Example: replace GEMINI_API_URL and GEMINI_API_KEY in environment before running.
    try:
        sample = generate_quiz("Python list comprehensions", 3, "intermediate")
        print(json.dumps(sample, indent=2))
    except Exception as e:
        print("Error:", e)
