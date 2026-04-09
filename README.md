# AI-Assisted Debugging and Feature Development

## Overview

This project is a Python-based application that generates multiple-choice questions using an AI API (OpenRouter). The program takes a topic, number of questions, and difficulty level as input and returns formatted multiple-choice questions.

---

## AI Tools Used

* ChatGPT (for debugging, improving code, and identifying edge cases)
* GitHub Copilot (used in Visual Studio Code for suggestions and fixes)

---

## Debugging Process

Using AI tools, I analyzed the original code and identified several issues:

* Missing error handling for API requests
* Possible crashes when parsing invalid JSON responses
* Unsafe access to API response fields
* No validation for user input (number of questions)
* Weak parsing logic that could break if AI output format changes

### Fixes Applied:

* Added try/except blocks for network errors and JSON parsing
* Validated API response structure before accessing data
* Handled incorrect or missing API responses safely
* Converted user input into correct data types
* Added validation to skip malformed questions

---

## Feature Improvements

The following features were added:

* Difficulty level selection (easy, medium, hard)
* Input validation for user entries
* Ability to save generated questions to a text file
* Improved prompt design for more consistent AI output
* Basic format validation for generated questions

---

## Testing

The program was tested with different inputs:

### Test Cases:

* Topic: Math, Difficulty: Easy
* Topic: Physics, Difficulty: Medium
* Topic: History, Difficulty: Hard

### Results:

* The program handled API errors without crashing
* Output format remained mostly consistent
* Invalid or malformed questions were skipped safely
* File saving functionality worked correctly

---

## Challenges

* Ensuring consistent output format from AI responses
* Handling unexpected API response structures
* Designing reliable parsing logic

---

## Conclusion

Using AI tools significantly improved the debugging process and code quality. AI assistance helped identify hidden issues, suggest improvements, and speed up development. The final program is more stable, user-friendly, and robust.

---

## How to Run the Project

1. Install dependencies:
   pip install requests python-dotenv

2. Create a `.env` file and add your API key:
   OPENROUTER_KEY=your_api_key_here

3. Run the program:
   python your_script_name.py
