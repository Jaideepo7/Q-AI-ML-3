import streamlit as st
import requests

st.set_page_config(page_title="Q-AI Quiz Generator", page_icon="🎓", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# --- Session State Init ---
for key, default in [
    ("logged_in", False),
    ("user_id", None),
    ("quiz_id", None),
    ("quiz", None),
    ("session_id", None),
    ("current_q", None),
    ("last_result", None),
    ("session_summary", None),
    ("quiz_history", None),
    ("session_results", None),
    ("submitted_answers", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- AUTH ---
if not st.session_state.logged_in:
    st.title("Q-AI: Video to Quiz Generator")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Welcome Back!")
        email = st.text_input("Email", key="login_email")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Log In"):
            if not email:
                st.error("Email is required.")
            elif not pwd:
                st.error("Password is required.")
            else:
                res = requests.post(f"{BACKEND_URL}/login", json={"email": email, "password": pwd})
                if res.status_code == 200:
                    data = res.json()
                    if data.get("status") == "success" and "session" in data:
                        st.session_state.logged_in = True
                        st.session_state.user_id = data["session"]["user_id"]
                        st.rerun()
                    else:
                        st.error(data.get("message", "Login failed."))
                elif res.status_code == 401:
                    st.error(res.json().get("detail", "Invalid email or password."))
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")

    with tab2:
        st.subheader("Create Account")
        new_email = st.text_input("Email", key="signup_email")
        new_pwd = st.text_input("Password", type="password", key="signup_pwd")
        if st.button("Register"):
            res = requests.post(f"{BACKEND_URL}/signup", json={"email": new_email, "password": new_pwd})
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    st.success("Account created! Now please log in.")
                else:
                    st.error(f"Registration failed: {data.get('message', 'Unknown error')}")
            else:
                st.error("Registration failed. Email might already be in use.")

# --- MAIN APP ---
else:
    st.sidebar.success("Logged in!")
    if st.sidebar.button("Logout"):
        for k in ["logged_in", "user_id", "quiz_id", "quiz", "session_id",
                  "current_q", "last_result", "session_summary", "quiz_history",
                  "session_results"]:
            st.session_state[k] = None
        st.session_state.submitted_answers = {}
        st.session_state.logged_in = False
        st.rerun()

    # ── Quiz History ──────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Quiz History")

    if st.session_state.quiz_history is None:
        hist_res = requests.get(
            f"{BACKEND_URL}/quizzes",
            params={"user_id": st.session_state.user_id},
        )
        st.session_state.quiz_history = (
            hist_res.json().get("quizzes", []) if hist_res.status_code == 200 else []
        )

    history = st.session_state.quiz_history
    if not history:
        st.sidebar.caption("No quizzes yet.")
    else:
        for q in history:
            title = q.get("title") or q["video_url"][-20:]
            date = q["created_at"][:10]
            if st.sidebar.button(f"{title}  ·  {date}", key=f"hist_{q['quiz_id']}"):
                with st.spinner("Loading quiz..."):
                    qres = requests.get(f"{BACKEND_URL}/quiz/{q['quiz_id']}")
                if qres.status_code == 200:
                    st.session_state.quiz = qres.json()
                    st.session_state.quiz_id = q["quiz_id"]
                    st.session_state.session_id = None
                    st.session_state.current_q = None
                    st.session_state.last_result = None
                    st.session_state.session_summary = None
                    st.rerun()
                else:
                    st.sidebar.error("Failed to load quiz.")

    if st.sidebar.button("Refresh", key="refresh_history"):
        st.session_state.quiz_history = None
        st.rerun()

    # ── Generate ──────────────────────────────────────────────────────────────
    st.title("Generate Your Quiz")
    video_url = st.text_input("Enter YouTube or Video URL:", placeholder="https://youtube.com/...")

    if st.button("Generate Quiz"):
        if not video_url:
            st.warning("Please enter a YouTube URL.")
        else:
            with st.spinner("Downloading, transcribing, and generating quiz... (~2-5 min)"):
                payload = {"url": video_url, "user_id": st.session_state.user_id}
                response = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=600)
            if response.status_code == 200:
                data = response.json()
                st.session_state.quiz_id = data.get("quiz_id")
                st.session_state.session_id = None
                st.session_state.session_summary = None
                st.session_state.quiz_history = None  # force sidebar re-fetch
                st.success(data.get("message", "Quiz generated!"))
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    st.divider()

    # ── Load existing quiz ────────────────────────────────────────────────────
    if st.button("Load Latest Quiz"):
        with st.spinner("Fetching quiz..."):
            res = requests.get(f"{BACKEND_URL}/quiz")
        if res.status_code == 200:
            quiz_data = res.json()
            st.session_state.quiz = quiz_data
            st.session_state.quiz_id = quiz_data.get("quiz_id")
            st.session_state.session_id = None
            st.session_state.current_q = None
            st.session_state.last_result = None
            st.session_state.session_summary = None

    # ── Results Page ──────────────────────────────────────────────────────────
    if st.session_state.session_summary:
        s = st.session_state.session_summary
        results = st.session_state.session_results or []
        user_answers = st.session_state.submitted_answers or {}
        quiz = st.session_state.quiz

        st.title("Session Results")

        # Score metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{s['correct']} / {s['questions_answered']}")
        c2.metric("Accuracy", f"{s['accuracy']}%")
        c3.metric("Duration", f"{s['duration_minutes']} min")

        # Topic mastery
        st.divider()
        st.subheader("Topic Mastery")
        for t in s["topics"]:
            nrm = t["next_review_minutes"]
            if nrm < 60:
                review_str = f"{nrm:.0f} min"
            elif nrm < 1440:
                review_str = f"{nrm / 60:.1f} hr"
            else:
                review_str = f"{nrm / 1440:.1f} days"
            mastered_tag = "  ✓ Mastered" if t["mastered"] else ""
            st.markdown(f"**{t['topic']}**{mastered_tag}")
            st.progress(int(t["mastery"]))
            st.caption(f"{t['mastery']}% mastery · optimal next review in **{review_str}**")

        # Per-question breakdown
        if results and quiz:
            st.divider()
            st.subheader("Question Breakdown")
            q_map = {q["id"]: q for q in quiz["questions"]}
            for i, r in enumerate(results):
                q = q_map.get(r["question_id"])
                if not q:
                    continue
                icon = "✓" if r["is_correct"] else "✗"
                label = f"{icon}  Q{i + 1}: {q['question']}"
                with st.expander(label, expanded=not r["is_correct"]):
                    user_sel = user_answers.get(r["question_id"], "—")
                    correct_full = next(
                        (opt for opt in q["options"] if opt.startswith(r["correct_answer"] + ":")),
                        r["correct_answer"],
                    )
                    if r["is_correct"]:
                        st.success(f"Correct — **{correct_full}**")
                    else:
                        st.error(f"Your answer: **{user_sel or '—'}**")
                        st.info(f"Correct answer: **{correct_full}**")

        # Actions
        st.divider()
        col1, _, col2 = st.columns([2, 6, 2])
        with col1:
            if st.button("Start New Session", type="primary", use_container_width=True):
                st.session_state.session_summary = None
                st.session_state.session_id = None
                st.session_state.current_q = None
                st.session_state.session_results = None
                st.session_state.submitted_answers = {}
                st.rerun()
        with col2:
            if st.button("Load Different Quiz", use_container_width=True):
                st.session_state.session_summary = None
                st.session_state.session_id = None
                st.session_state.current_q = None
                st.session_state.quiz_id = None
                st.session_state.quiz = None
                st.session_state.session_results = None
                st.session_state.submitted_answers = {}
                st.rerun()

    # ── Active Session ────────────────────────────────────────────────────────
    elif st.session_state.session_id:
        session_id = st.session_state.session_id

        # Feedback from last answer
        if st.session_state.last_result:
            r = st.session_state.last_result
            if r["is_correct"]:
                st.success(f"Correct!  Mastery: {r['mastery_before']}% → {r['mastery_after']}%")
            else:
                st.error(
                    f"Incorrect.  Correct answer: **{r['correct_answer']}**  "
                    f"Mastery: {r['mastery_before']}% → {r['mastery_after']}%"
                )
            st.caption(f"Next review in {r['next_review_minutes']} min")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next Question"):
                    st.session_state.last_result = None
                    with st.spinner("Loading next question..."):
                        nxt = requests.get(
                            f"{BACKEND_URL}/session/next",
                            params={"session_id": session_id},
                        )
                    nxt_data = nxt.json()
                    if nxt_data.get("done"):
                        end_res = requests.post(
                            f"{BACKEND_URL}/session/end",
                            json={"session_id": session_id},
                        )
                        st.session_state.session_summary = end_res.json()
                        st.session_state.session_id = None
                    else:
                        st.session_state.current_q = nxt_data
                    st.rerun()
            with col2:
                if st.button("End Session"):
                    end_res = requests.post(
                        f"{BACKEND_URL}/session/end",
                        json={"session_id": session_id},
                    )
                    st.session_state.session_summary = end_res.json()
                    st.session_state.session_id = None
                    st.session_state.current_q = None
                    st.rerun()

        # ── All 10 questions form ─────────────────────────────────────────────
        else:
            quiz = st.session_state.quiz
            if not quiz:
                st.warning("No quiz loaded.")
            else:
                st.subheader(f"{len(quiz['questions'])} Questions")
                with st.form("full_quiz_form"):
                    for i, q in enumerate(quiz["questions"]):
                        st.markdown(f"**Q{i+1}. {q['question']}**")
                        st.radio(
                            label="",
                            options=q["options"],
                            key=f"batch_q_{q['id']}",
                            index=None,
                            label_visibility="collapsed",
                        )
                        st.divider()
                    submitted = st.form_submit_button("Submit All Answers")

                if submitted:
                    answers = []
                    unanswered = []
                    for q in quiz["questions"]:
                        sel = st.session_state.get(f"batch_q_{q['id']}")
                        if sel:
                            answers.append({
                                "question_id": q["id"],
                                "selected_option": sel.split(":")[0].strip(),
                            })
                        else:
                            unanswered.append(q["id"])

                    if unanswered:
                        st.warning(f"Please answer all questions ({len(unanswered)} remaining).")
                    else:
                        # Capture selections before rerun clears form state
                        st.session_state.submitted_answers = {
                            q["id"]: st.session_state.get(f"batch_q_{q['id']}")
                            for q in quiz["questions"]
                        }
                        with st.spinner("Submitting answers..."):
                            sub_res = requests.post(
                                f"{BACKEND_URL}/session/submit_all",
                                json={"session_id": session_id, "answers": answers},
                            )
                        if sub_res.status_code == 200:
                            data = sub_res.json()
                            st.session_state.session_summary = data
                            st.session_state.session_results = data.get("results", [])
                            st.session_state.session_id = None
                            st.rerun()
                        else:
                            st.error(f"Submission failed: {sub_res.json().get('detail')}")

    # ── Start Session ─────────────────────────────────────────────────────────
    elif st.session_state.quiz_id:
        quiz = st.session_state.get("quiz")
        if not quiz:
            st.info("Quiz loaded. Click below to start.")
        else:
            st.info(f"Quiz ready — {len(quiz['questions'])} questions. Click below to start.")

        if st.button("Start Study Session"):
            # Load questions if not already loaded
            if not st.session_state.quiz:
                with st.spinner("Loading questions..."):
                    res = requests.get(f"{BACKEND_URL}/quiz")
                if res.status_code == 200:
                    st.session_state.quiz = res.json()
                else:
                    st.error("Failed to load questions.")
                    st.stop()

            with st.spinner("Initializing session..."):
                start_res = requests.post(
                    f"{BACKEND_URL}/session/start",
                    json={
                        "user_id": st.session_state.user_id,
                        "quiz_id": st.session_state.quiz_id,
                    },
                )
            if start_res.status_code == 200:
                st.session_state.session_id = start_res.json()["session_id"]
                st.rerun()
            else:
                st.error(f"Failed to start session: {start_res.json().get('detail')}")

