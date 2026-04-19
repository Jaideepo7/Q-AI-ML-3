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
            # Connect to your actual Backend /login endpoint
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
        new_email = st.text_input("Email", key="sigup_email")
        new_pwd = st.text_input("Password", type="password", key="signup_pwd")
        if st.button("Register"):
            res = requests.post(f"{BACKEND_URL}/signup", json={"email": new_email, "password": new_pwd})
            if res.status_code == 200:
                st.success("Account created! Now please log in.")
            else:
                st.error("Registration failed. Email might already be in use.")

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
        if video_url:
            with st.spinner("Processing..."):
                # Use the REAL user_id from session state
                payload = {"url": video_url, "user_id": st.session_state.user_id}
                response = requests.post(f"{BACKEND_URL}/generate", json=payload)
                if response.status_code == 200:
                    st.success("Video received! Records initialized in Supabase.")
                else:
                    st.error("Error generating quiz.")

    st.divider()

    # Quiz Display Section
    if st.button("Load Latest Quiz"):
        # Logic to fetch and display the quiz...
        # (Your existing quiz display code goes here)
        st.info("Loading questions...")