# AI-Assisted Optimization and Feature Expansion

## Overview

This project is a Python application that generates multiple-choice questions using an AI API (OpenRouter). The goal of this assignment was to optimize the codebase for better readability, maintainability, and overall structure using AI tools.

---

## AI Tools Used

* ChatGPT (used to suggest optimizations, refactor code, and explain improvements)
* GitHub Copilot (used in Visual Studio Code for inline suggestions)

---

## Optimizations Made

### ✅ Accepted Optimizations

1. Modular Code Structure

* Refactored the program into smaller, focused functions:

  * `build_prompt()`
  * `make_api_request()`
  * `extract_text_from_response()`
  * `parse_questions()`
  * `save_questions_to_file()`
* This improved readability and made the code easier to maintain and debug.

2. Use of Constants

* Introduced constants such as:

  * `API_URL`
  * `MODEL`
  * `REQUIRED_FIELDS`
* This avoids repetition and makes future updates easier.

3. Improved Variable Naming

* Replaced unclear variable names with meaningful ones:

  * `p` → `part`
  * `qs` → `questions`
* This significantly improved code clarity.

4. Separation of Concerns

* Divided responsibilities across functions:

  * API handling
  * Data extraction
  * Parsing
  * User input
* This makes the code more organized and scalable.

5. Enhanced Error Handling

* Added try/except blocks for:

  * Network errors
  * JSON parsing errors
  * API response structure issues
* Prevents crashes and improves robustness.

6. Input Validation

* Ensured user inputs are valid:

  * Number must be an integer
  * Difficulty must be one of: easy, medium, hard

7. Centralized Validation Logic

* Introduced `REQUIRED_FIELDS` to validate question format
* Avoids repeating validation logic and improves maintainability

---

### ❌ Rejected Optimizations

1. Regex-Based Parsing

* AI suggested using regular expressions for parsing questions
* Rejected because:

  * Adds unnecessary complexity
  * Current approach is simpler and sufficient for the expected format

2. Asynchronous API Requests

* AI suggested using async/await for API calls
* Rejected because:

  * Only a single request is made at a time
  * Would make the code more complex without significant benefit

---

## CLEAR Evaluation

Each AI suggestion was evaluated using the CLEAR checklist:

* **Correct**
  Verified that all changes worked correctly and did not introduce new bugs.

* **Logical**
  Ensured the optimizations made sense within the program structure and flow.

* **Efficient**
  Evaluated whether the changes improved performance or reduced unnecessary operations.

* **Appropriate**
  Avoided over-engineering or adding complexity that is not needed for this project.

* **Readable**
  Prioritized clean, understandable code with clear structure and naming.

---

## Testing

The optimized code was tested using multiple inputs:

* Topic: Math (easy)
* Topic: Physics (medium)
* Topic: History (hard)

### Results:

* Program handled API and input errors without crashing
* Output format remained consistent
* Invalid questions were safely skipped
* Code became easier to read and extend

---

## Conclusion

Using AI tools significantly improved the quality of the codebase. The optimizations made the program more modular, readable, and maintainable. Not all AI suggestions were accepted, and the CLEAR checklist helped ensure that only meaningful and appropriate improvements were applied.
