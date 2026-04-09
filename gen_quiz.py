import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"

REQUIRED_FIELDS = ["A)", "B)", "C)", "D)", "Answer:"]


def build_prompt(topic, num_questions, difficulty):
    return f"""
Generate {num_questions} {difficulty} multiple choice questions about {topic}.

FORMAT EXACTLY LIKE THIS:

Question 1:
A)
B)
C)
D)
Answer:

Question 2:
A)
B)
C)
D)
Answer:

Do not write explanation.
Do not change format.
"""


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


def extract_text_from_response(response_json):
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("❌ Unexpected API response format")
        return None


def parse_questions(raw_text):
    parts = raw_text.split("Question ")
    questions = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        question = "Question " + part

        # Validate format
        if all(field in question for field in REQUIRED_FIELDS):
            questions.append(question)
        else:
            print("⚠️ Skipped malformed question:\n", question)

    return questions


def save_questions_to_file(questions, filename="questions.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            for question in questions:
                file.write(question + "\n\n")
        print(f"\n✅ Questions saved to {filename}")
    except Exception as error:
        print("❌ Error saving file:", error)


def generate_questions(topic, num_questions, difficulty):
    prompt = build_prompt(topic, num_questions, difficulty)

    response_json = make_api_request(prompt)
    if not response_json:
        return []

    raw_text = extract_text_from_response(response_json)
    if not raw_text:
        return []

    print("\nRAW TEXT:\n")
    print(raw_text)

    questions = parse_questions(raw_text)

    print("\nPARSED QUESTIONS:\n")
    for q in questions:
        print("------------------")
        print(q)

    return questions


def get_user_input():
    topic = input("Topic: ").strip()

    try:
        num_questions = int(input("Number of questions: "))
    except ValueError:
        print("❌ Invalid number. Please enter an integer.")
        exit()

    difficulty = input("Difficulty (easy/medium/hard): ").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        print("⚠️ Invalid difficulty, defaulting to 'medium'")
        difficulty = "medium"

    return topic, num_questions, difficulty


def main():
    if not API_KEY:
        print("❌ API key not found. Check your .env file.")
        return

    topic, num_questions, difficulty = get_user_input()

    questions = generate_questions(topic, num_questions, difficulty)

    print("\nTotal questions:", len(questions))

    save = input("Save to file? (y/n): ").strip().lower()
    if save == "y":
        save_questions_to_file(questions)


if __name__ == "__main__":
    main()
