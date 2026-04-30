# AI Quiz Generator

## Overview

This Streamlit application generates multiple-choice quiz questions with the OpenRouter AI API. A user enters a topic, chooses a difficulty, selects the number of questions, and completes an interactive quiz with scoring and explanations.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`.
4. Add your OpenRouter API key:

   ```text
   OPENROUTER_KEY=your_openrouter_api_key_here
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Security Review

The project was reviewed for the following risks:

- Missing input validation
- Hardcoded secrets
- Overly permissive AI prompt logic
- Weak validation of AI-generated output
- Lack of graceful error handling
- Unnecessary third-party script injection

## Security Improvements Made

- Added centralized validation for topic, question count, and difficulty.
- Limited topic length to reduce prompt-injection and abuse risk.
- Treated the topic as data in the AI prompt instead of as trusted instructions.
- Added strict validation and normalization for AI-generated questions, answer options, correct answers, and explanations.
- Replaced `print()` error output with Python logging so errors are handled without exposing unnecessary details in the UI.
- Removed the custom CDN-loaded confetti script and replaced it with Streamlit's built-in `st.balloons()`.
- Added `.env.example` so API key setup is documented without committing real secrets.
- Confirmed `.env` is ignored by Git.

## Testing

The following checks were run:

```bash
.\venv\Scripts\python.exe -m py_compile app.py gen_quiz.py
```

A validation smoke test also confirmed that:

- Valid quiz data parses correctly.
- Overly long generated questions are rejected.
- Empty topics are rejected.

## Ethical AI Reflection

AI-generated code can be useful, but it still needs human review. The most important lesson from this security pass is that AI output should not be trusted automatically. User input must be validated, generated content must be checked before display or use, and secrets must be protected. Ethical use of AI means being responsible for the code after it is generated, especially when the app depends on user input and external AI services.
