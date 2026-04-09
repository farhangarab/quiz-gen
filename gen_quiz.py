import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate_questions(topic, n, difficulty):

    prompt = f"""
Generate {n} {difficulty} multiple choice questions about {topic}.

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

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "openrouter/auto",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
    except requests.exceptions.RequestException as e:
        print("❌ Network error:", e)
        return []

    # check status code
    if response.status_code != 200:
        print("❌ API Error:", response.status_code)
        print(response.text)
        return []

    # parse JSON safely
    try:
        result = response.json()
    except ValueError:
        print("❌ Invalid JSON response")
        return []

    # extract text safely
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("❌ Unexpected API response format")
        return []

    print("\nRAW TEXT:\n")
    print(text)

    # safer splitting
    parts = text.split("Question ")

    questions = []

    for p in parts:
        p = p.strip()
        if p:
            q = "Question " + p

            # basic validation
            if all(x in q for x in ["A)", "B)", "C)", "D)", "Answer:"]):
                questions.append(q)
            else:
                print("⚠️ Skipped malformed question:\n", q)

    print("\nPARSED QUESTIONS:\n")

    for q in questions:
        print("------------------")
        print(q)

    return questions


def save_to_file(questions, filename="questions.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q + "\n\n")
        print(f"\n✅ Questions saved to {filename}")
    except Exception as e:
        print("❌ Error saving file:", e)


if __name__ == "__main__":

    topic = input("Topic: ").strip()

    # validate number input
    try:
        n = int(input("Number of questions: "))
    except ValueError:
        print("❌ Please enter a valid number")
        exit()

    # difficulty input
    difficulty = input("Difficulty (easy/medium/hard): ").strip().lower()

    if difficulty not in ["easy", "medium", "hard"]:
        print("⚠️ Invalid difficulty, defaulting to 'medium'")
        difficulty = "medium"

    # check API key
    if not API_KEY:
        print("❌ API key not found. Check your .env file.")
        exit()

    qs = generate_questions(topic, n, difficulty)

    print("\nTotal questions:", len(qs))

    # ask to save
    save = input("Save to file? (y/n): ").strip().lower()

    if save == "y":
        save_to_file(qs)
