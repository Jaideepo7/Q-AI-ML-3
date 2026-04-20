import streamlit as st
import requests

# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="Q-AI Quiz Generator", page_icon="🎓", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# 2. Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# --- AUTHENTICATION UI ---
if not st.session_state.logged_in:
    st.title("🎓 Q-AI: Video to Quiz Generator")
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
                        st.error(data.get("message", "Login failed: Invalid credentials."))
                elif res.status_code == 401:
                    error_text = res.json().get("detail", "Invalid email or password.")
                    st.error(error_text)
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error occurred')}")

    with tab2:
        st.subheader("Create Account")
        new_email = st.text_input("Email", key="signup_email")
        new_pwd = st.text_input("Password", type="password", key="signup_pwd")
        if st.button("Register"):
            if not new_email:
                st.error("Email is required.")
            elif len(new_pwd) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                res = requests.post(f"{BACKEND_URL}/signup", json={"email": new_email, "password": new_pwd})
                if res.status_code == 200:
                    data = res.json()
                    if data.get("status") == "success":
                        st.success("Account created!")
                    else:
                        st.error(data.get("message", "Registration failed."))
                else:
                    st.error(res.json().get("detail", "Registration failed. Email might already be in use."))

# --- MAIN APP UI (Only shows if logged_in is True) ---
else:
    # Sidebar
    st.sidebar.success(f"Logged in!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.rerun()

    st.title("🎥 Generate Your Quiz")

    # URL Input Section
    video_url = st.text_input("Enter YouTube or Video URL:", placeholder="https://youtube.com/...")

    if st.button("Generate Quiz"):
        if not video_url:
            st.warning("Please enter a YouTube URL.")
        else:
            with st.spinner("Downloading audio and transcribing... this may take a few minutes."):
                payload = {"url": video_url, "user_id": st.session_state.user_id}
                response = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=600)
                if response.status_code == 200:
                    data = response.json()
                    st.success(data.get("message", "Transcript complete! Ready to generate quiz."))
                    import json as _json
                    transcript = _json.loads(data.get("transcript", "{}")).get("transcript", [])
                    if transcript:
                        with st.expander("View Transcript"):
                            for line in transcript:
                                st.markdown(f"**Speaker {line['speaker']}:** {line['text']}")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    st.divider()

    # Quiz Display Section
    if st.button("Load Latest Quiz"):
        # Logic to fetch and display the quiz...
        # (Your existing quiz display code goes here)
        st.info("Loading questions...")