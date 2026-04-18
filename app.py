import random
import streamlit as st
import streamlit.components.v1 as components
from gen_quiz import generate_questions

st.set_page_config(page_title="AI Quiz Generator", layout="centered")

st.title("AI Quiz Generator")
st.write("Generate multiple-choice questions using AI.")


# ---------------- STATE HELPERS ---------------- #
def clear_answer_state():
    for key in list(st.session_state.keys()):
        if key.startswith("q"):
            del st.session_state[key]


def reset_quiz_state():
    clear_answer_state()
    st.session_state.questions = []
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.shuffled_options = {}


def prepare_questions(raw_questions):
    prepared = []
    shuffled = {}
    for i, q in enumerate(raw_questions):
        prepared.append(q)
        opts = list(q["options"].items())
        random.shuffle(opts)
        shuffled[i] = opts
    return prepared, shuffled


# ---------------- PROGRESS BAR ---------------- #
def render_sticky_progress(answered, total):
    percent = int((answered / total) * 100) if total else 0

    dots_html = "".join(
        f'<div style="width:9px;height:9px;border-radius:50%;background:{"#2dd4bf" if st.session_state.get(f"q{i}") is not None else "#1e293b"};flex-shrink:0;transition:background 0.3s;"></div>'
        for i in range(total)
    )

    # Bar starts VISIBLE. JS only hides it when at the very top.
    # If JS fails -> bar stays visible (safe fallback).
    st.markdown(
        f"""
        <style>
        #sp-bar {{
            position: fixed;
            top: 3.5rem;
            left: 50%;
            transform: translateX(-50%);
            width: min(730px, calc(100vw - 2rem));
            z-index: 9999;
            background: rgba(13, 20, 38, 0.97);
            border-radius: 14px;
            padding: 0.75rem 1.2rem 0.7rem 1.2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            opacity: 1;
            transition: opacity 0.25s ease, transform 0.25s ease;
        }}

        .sp-top {{
            display: flex;
            justify-content: space-between;
            color: #e2e8f0;
            font-size: 0.83rem;
            font-weight: 600;
            margin-bottom: 7px;
            font-family: sans-serif;
        }}
        .sp-track {{
            width: 100%;
            height: 8px;
            background: #1e293b;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        .sp-fill {{
            height: 100%;
            width: {percent}%;
            background: linear-gradient(90deg, #0ea5e9, #2dd4bf);
            border-radius: 999px;
        }}
        .sp-dots {{
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }}
        </style>

        <div id="sp-bar">
            <div class="sp-top">
                <span>Progress</span>
                <span>{answered} / {total} &nbsp;·&nbsp; {percent}%</span>
            </div>
            <div class="sp-track"><div class="sp-fill"></div></div>
            <div class="sp-dots">{dots_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- CONFETTI ---------------- #
def render_confetti():
    components.html(
        """
        <script>
        (function tryConfetti(tries) {
            const p = window.parent;
            if (!p.confetti) {
                if (!p.document.getElementById('confetti-script')) {
                    const s = p.document.createElement('script');
                    s.id = 'confetti-script';
                    s.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js';
                    s.onload = () => fireConfetti();
                    p.document.head.appendChild(s);
                } else if (tries < 20) {
                    setTimeout(() => tryConfetti(tries + 1), 100);
                }
                return;
            }
            fireConfetti();
        })(0);

        function fireConfetti() {
            const c = window.parent.confetti;
            const end = Date.now() + 3000;
            const colors = ['#2dd4bf', '#0ea5e9', '#818cf8', '#f472b6', '#facc15'];
            (function frame() {
                c({ particleCount: 6, angle: 60,  spread: 55, origin: { x: 0 }, colors });
                c({ particleCount: 6, angle: 120, spread: 55, origin: { x: 1 }, colors });
                if (Date.now() < end) requestAnimationFrame(frame);
            })();
        }
        </script>
        """,
        height=1,
    )


# ---------------- SESSION INIT ---------------- #
for key, default in [
    ("questions", []),
    ("submitted", False),
    ("score", 0),
    ("shuffled_options", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- INPUT ---------------- #
topic = st.text_input("Enter topic")
num_questions = st.number_input("Number of questions", 1, 20, 5)
difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])


# ---------------- BUTTON CONTROL ---------------- #
quiz_generated = len(st.session_state.questions) > 0

if not quiz_generated:
    if st.button("Generate Questions", type="primary"):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            clear_answer_state()
            st.session_state.submitted = False
            st.session_state.score = 0

            with st.spinner("Generating questions..."):
                questions = generate_questions(topic, int(num_questions), difficulty)

            if not questions:
                st.error("No questions generated. Try again.")
            else:
                questions = questions[:num_questions]
                random.shuffle(questions)
                prepared, shuffled = prepare_questions(questions)
                st.session_state.questions = prepared
                st.session_state.shuffled_options = shuffled
else:
    if st.button("↺ Restart"):
        reset_quiz_state()
        st.rerun()


# ---------------- QUIZ ---------------- #
if st.session_state.questions:
    total = len(st.session_state.questions)
    answered = sum(1 for i in range(total) if st.session_state.get(f"q{i}") is not None)

    render_sticky_progress(answered, total)

    # Spacer so content doesn't sit under the fixed bar
    st.markdown("<div style='height:5rem'></div>", unsafe_allow_html=True)

    user_answers = []

    for i, q in enumerate(st.session_state.questions):
        is_answered = st.session_state.get(f"q{i}") is not None
        indicator = "✅" if is_answered else "⬜"
        st.markdown(f"### {indicator} Question {i + 1}")
        st.write(q["question"])

        options = st.session_state.shuffled_options.get(i, [])
        option_list = [f"{k}) {v}" for k, v in options]
        selected = st.radio("Select an answer:", option_list, key=f"q{i}", index=None)
        user_answers.append((selected, q))
        st.markdown("---")

    # ---------------- SUBMIT ---------------- #
    unanswered = total - answered
    if unanswered > 0:
        st.caption(f"{unanswered} question{'s' if unanswered > 1 else ''} remaining.")

    if st.button("Submit Quiz", type="primary", disabled=(answered < total)):
        score = 0
        st.session_state.submitted = True

        st.markdown("## 📊 Results")

        for i, (selected, q) in enumerate(user_answers):
            selected_key = selected.split(")")[0]
            correct = q["correct_answer"]

            if selected_key == correct:
                st.success(f"**Q{i+1}: Correct ✓**")
                score += 1
            else:
                correct_text = q["options"][correct]
                st.error(
                    f"**Q{i+1}: Incorrect ✗** — Correct: {correct}) {correct_text}"
                )

            st.info(f"💡 {q['explanation']}")

        st.session_state.score = score
        pct = (score / total) * 100

        if pct == 100:
            grade = "🏆 Perfect score!"
        elif pct >= 80:
            grade = "🎉 Great job!"
        elif pct >= 60:
            grade = "👍 Not bad!"
        else:
            grade = "📚 Keep practicing!"

        st.markdown(f"## Score: {score}/{total} ({pct:.0f}%) — {grade}")
        render_confetti()


# ---------------- SIDEBAR ---------------- #
st.sidebar.markdown("## Quiz Info")
st.sidebar.write(f"**Topic:** {topic or '—'}")
st.sidebar.write(f"**Difficulty:** {difficulty}")
st.sidebar.write(f"**Questions:** {len(st.session_state.questions)}")

if st.session_state.questions:
    total = len(st.session_state.questions)
    answered = sum(1 for i in range(total) if st.session_state.get(f"q{i}") is not None)
    st.sidebar.write(
        f"**Progress:** {answered}/{total} ({int(answered / total * 100)}%)"
    )

if st.session_state.submitted:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Result")
    total = len(st.session_state.questions)
    st.sidebar.write(f"**Score:** {st.session_state.score}/{total}")
