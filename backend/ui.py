import streamlit as st
import requests

st.set_page_config(page_title="Q-AI Quiz Generator", page_icon="🎓")

st.title("🎓 Q-AI: Video to Quiz Generator")

# --- SIDEBAR: Progress & User Info ---
st.sidebar.header("User Dashboard")
if st.sidebar.button("Show My Progress"):
    # Call your FastAPI /progress endpoint
    try:
        res = requests.get("http://127.0.0.1:8000/progress?user_id=Gopika")
        if res.status_code == 200:
            prog = res.json()
            st.sidebar.metric("Learning Progress", prog["progress"])
            st.sidebar.info(prog["message"])
    except Exception as e:
        st.sidebar.error("Backend not reachable")

# --- MAIN UI: URL Input ---
video_url = st.text_input("Enter YouTube or Video URL:", placeholder="https://youtube.com/...")

if st.button("Generate Quiz"):
    if video_url:
        with st.spinner("Processing video and generating quiz..."):
            try:
                response = requests.post("http://127.0.0.1:8000/generate", json={"url": video_url})
                if response.status_code == 200:
                    st.success("Video received! Records initialized in Supabase.")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error("Could not connect to FastAPI server.")
    else:
        st.warning("Please enter a valid URL.")

st.divider()

# --- MAIN UI: Quiz Display ---
if st.button("Load Latest Quiz"):
    try:
        response = requests.get("http://127.0.0.1:8000/quiz")
        if response.status_code == 200:
            data = response.json()
            st.subheader(f"Quiz for: {data['video_name']}")
            
            # This is the 'Display Logic' you were asking about
            with st.form("quiz_form"):
                user_answers = {}
                
                for q in data['questions']:
                    st.write(f"### {q['question']}")
                    # If you have 'type' in your DB, you could use an if-statement here
                    # for st.text_input vs st.radio
                    user_answers[q['id']] = st.radio(
                        "Choose your answer:", 
                        q['options'], 
                        key=f"radio_{q['id']}"
                    )
                
                submitted = st.form_submit_button("Submit All Answers")
                
                if submitted:
                    st.info("Grading your quiz...")
                    for q_id, selected in user_answers.items():
                        ans_payload = {
                            "user_id": "00000000-0000-0000-0000-000000000000", # Match your Backend UUID
                            "question_id": q_id,
                            "selected_option": selected
                        }
                        ans_res = requests.post("http://127.0.0.1:8000/answer", json=ans_payload)
                        
                        if ans_res.status_code == 200:
                            result = ans_res.json()
                            if result["is_correct"]:
                                st.success(f"Question {q_id}: Correct! ✅")
                            else:
                                st.error(f"Question {q_id}: Incorrect. ❌")
        else:
            st.info("No quiz found yet. Generate one first!")
    except Exception as e:
        st.error(f"Backend Error: {e}")