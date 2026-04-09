import streamlit as st
from gen_quiz import generate_questions

st.set_page_config(page_title="AI Quiz Generator", layout="centered")

st.title("🧠 AI Quiz Generator")
st.write("Generate multiple-choice questions using AI")

# ---------------- SESSION ---------------- #
if "questions" not in st.session_state:
    st.session_state.questions = []

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "score" not in st.session_state:
    st.session_state.score = 0


# ---------------- INPUT ---------------- #
topic = st.text_input("Enter Topic")

num_questions = st.number_input(
    "Number of Questions", min_value=1, max_value=20, value=5
)

difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])


# ---------------- GENERATE ---------------- #
if st.button("Generate Questions"):

    if not topic.strip():
        st.warning("⚠️ Please enter a topic")
    else:
        st.session_state.submitted = False
        st.session_state.score = 0

        with st.spinner("Generating..."):
            st.session_state.questions = generate_questions(
                topic, num_questions, difficulty
            )


# ---------------- DISPLAY ---------------- #
if st.session_state.questions:

    st.success(f"✅ Generated {len(st.session_state.questions)} questions")

    user_answers = []

    for i, q in enumerate(st.session_state.questions):

        st.markdown(f"### Question {i+1}")
        st.write(q["question"])

        options = q["options"]
        correct = q["correct_answer"]

        option_list = [f"{k}) {v}" for k, v in options.items()]

        selected = st.radio("Select an answer:", option_list, key=f"q{i}")

        user_answers.append((selected, correct, q["explanation"]))

    # ---------------- SUBMIT ---------------- #
    if st.button("Submit Quiz"):

        score = 0
        st.session_state.submitted = True

        st.markdown("## Results")

        for i, (selected, correct, explanation) in enumerate(user_answers):

            selected_letter = selected[0]

            if selected_letter == correct:
                st.success(f"Question {i+1}: Correct ✅")
                score += 1
            else:
                st.error(f"Question {i+1}: Wrong ❌ (Correct: {correct})")

            st.info(f"💡 Explanation: {explanation}")

        st.session_state.score = score

        st.markdown(f"## 🎯 Score: {score} / {len(user_answers)}")


# ---------------- SIDEBAR ---------------- #
if st.session_state.submitted:
    st.sidebar.markdown("## 📊 Score")
    st.sidebar.write(f"{st.session_state.score} / {len(st.session_state.questions)}")
