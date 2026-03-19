import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate_questions(topic, n):

    prompt = f"""
Generate {n} multiple choice questions about {topic}.

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

    response = requests.post(API_URL, headers=headers, json=data)

    result = response.json()

    text = result["choices"][0]["message"]["content"]

    print("\nRAW TEXT:\n")
    print(text)

    # split questions
    parts = text.split("Question ")

    questions = []

    for p in parts:
        p = p.strip()
        if p:
            questions.append("Question " + p)

    print("\nPARSED QUESTIONS:\n")

    for q in questions:
        print("------------------")
        print(q)

    return questions


if __name__ == "__main__":

    topic = input("Topic: ")
    n = input("Number: ")

    qs = generate_questions(topic, n)

    print("\nTotal questions:", len(qs))
