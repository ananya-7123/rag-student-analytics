# pyrefly: ignore [missing-import]
import streamlit as st
import requests
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# 1. ALWAYS FIRST: Page config must happen before any other st. command
st.set_page_config(page_title="Student Success Analytics", layout="wide")

# --- CONFIGURATION ---
BACKEND_URL = "https://rag-student-analytics.onrender.com/api/v1"

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

# Function to handle the login/signup form
def show_login_page():
    st.subheader("Welcome to Student Success Analytics")
    
    auth_mode = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)
    
    username = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")
    
    if auth_mode == "Login":
        if st.button("Login"):
            if username and password:
                try:
                    res = requests.post(f"{BACKEND_URL}/auth/login", json={"username": username, "password": password})
                    if res.status_code == 200:
                        token = res.json().get("access_token")
                        if token:
                            st.session_state.auth_token = token
                            st.session_state.logged_in = True
                            st.rerun()
                        else:
                            st.error("Login failed: No access token received.")
                    else:
                        st.error("Invalid username or password")
                except Exception as e:
                    st.error(f"Cannot connect to backend: {e}")
            else:
                st.error("Please provide both username and password.")
    else:
        if st.button("Sign Up"):
            if username and password:
                try:
                    res = requests.post(f"{BACKEND_URL}/auth/signup", json={"username": username, "password": password})
                    if res.status_code == 200:
                        st.success("Sign up successful! You can now switch to Login.")
                    elif res.status_code == 400:
                        error_detail = res.json().get("detail", "Sign up failed.")
                        st.error(error_detail)
                    else:
                        st.error(f"Sign up failed. Server returned {res.status_code}")
                except Exception as e:
                    st.error(f"Cannot connect to backend: {e}")
            else:
                st.error("Please provide both username and password.")

# Core Logic: Show login page OR show the dashboard
if not st.session_state.logged_in:
    show_login_page()
else:
    # --- EVERYTHING BELOW IS NOW INDENTED INSIDE THE ELSE BLOCK ---
    st.title("Student Success Analytics Platform")
    st.caption("Transforming static academic documents into interactive exam preparation insights.")

    # --- SIDEBAR: CONTROL HUB ---
    st.sidebar.header("Control Hub")

    # Setup authenticated headers
    headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

    # Fetch dynamic workspaces from backend
    if "subjects_list" not in st.session_state:
        try:
            res = requests.get(f"{BACKEND_URL}/workspaces", headers=headers)
            if res.status_code == 200:
                fetched_workspaces = res.json()
                if fetched_workspaces:
                    st.session_state.subjects_list = fetched_workspaces
                else:
                    st.session_state.subjects_list = ["Biology"]
            else:
                st.session_state.subjects_list = ["Biology"]
        except Exception as e:
            st.sidebar.error(f"Failed to fetch workspaces: {e}")
            st.session_state.subjects_list = ["Biology"]

    # Input block to add extra subjects
    new_subject = st.sidebar.text_input("Create New Subject Workspace")
    if st.sidebar.button("Add Subject to Dashboard"):
        if new_subject and new_subject not in st.session_state.subjects_list:
            try:
                res = requests.post(
                    f"{BACKEND_URL}/workspaces", 
                    json={"subject_name": new_subject},
                    headers=headers
                )
                if res.status_code == 200:
                    st.session_state.subjects_list.append(new_subject)
                    st.rerun()
                else:
                    st.sidebar.error("Failed to add workspace to backend.")
            except Exception as e:
                st.sidebar.error(f"Backend connection failed: {e}")

    st.sidebar.markdown("---")

    # Dropdown selector to jump between saved workspaces
    subject = st.sidebar.selectbox("Active Subject Workspace", st.session_state.subjects_list)
    
    # Clear old data instantly if the student switches subjects
    if "prev_subject" not in st.session_state:
        st.session_state.prev_subject = subject

    if st.session_state.prev_subject != subject:
        st.session_state.analytics_data = None
        st.session_state.prev_subject = subject

    st.sidebar.subheader("Document Ingestion")
    doc_type = st.sidebar.selectbox("Document Type", ["syllabus", "notes", "pyq"])
    uploaded_files = st.sidebar.file_uploader("Upload PDF sources", type=["pdf"], accept_multiple_files=True)

    if st.sidebar.button("Process & Index"):
        if uploaded_files:    
            with st.spinner("Running ETL Pipeline..."):
                files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
                data_payload = {"subject": subject, "doc_type": doc_type}
                
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", data=data_payload, files=files_payload, headers=headers)
                    if response.status_code == 200:
                        st.sidebar.success("Successfully processed chunks!")
                    else:
                        st.sidebar.error("Upload failed.")
                except Exception as e:
                    st.sidebar.error(f"Cannot connect to backend: {e}")
        else:
            st.sidebar.warning("Please select files first.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out"):
        st.session_state.clear()
        st.session_state.logged_in = False
        st.session_state.auth_token = None
        st.rerun()

    # --- MAIN INTERFACE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Coverage Dashboard & Topics", "Revision Workspace", "Tactical Study Planner", "Career & Project Hub"])

    # Create persistent state to store data across button clicks
    if "analytics_data" not in st.session_state:
        st.session_state.analytics_data = None

    with tab1:
        st.subheader("Syllabus Coverage Matrix & High-Yield Topics")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Fetch Latest Analytics Matrix", key="fetch_matrix"):
                with st.spinner("Compiling database logs..."):
                    res = requests.post(f"{BACKEND_URL}/analytics/topics", json={"subject": subject}, headers=headers)
                    if res.status_code == 200:
                        st.session_state.analytics_data = res.json()
                    else:
                        st.error("Failed to load metrics from server.")
                        
            # Render the gauge if data exists in state
            if st.session_state.analytics_data:
                coverage = st.session_state.analytics_data.get("overall_coverage_percentage", 0.0)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = coverage,
                    title = {'text': "Overall Progress Match %"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#10b981"}}
                ))
                st.plotly_chart(fig, use_container_width=True)
                        
        with col2:
            st.write("### Extracted High-Yield Exam Topics")
            if st.session_state.analytics_data:
                topics = st.session_state.analytics_data.get("analyzed_topics", [])
                if isinstance(topics, list) and len(topics) > 0:
                    for idx, t in enumerate(topics):
                        st.markdown(f"**{idx+1}. {t.get('topic_name', 'Unknown Topic')}**")
                        st.caption(f"Priority Score: `{t.get('importance_score', 'N/A')}%` | Status: {t.get('status', 'Pending')}")
                else:
                    st.warning("No documents indexed yet. Use the sidebar Hub to ingest files into Pinecone first.")
            else:
                st.info("Click 'Fetch Latest Analytics Matrix' to pull trends directly out of your uploaded PYQs.")

    with tab2:
        st.subheader("AI Revision Notes Workspace")
        target_topic = st.text_input("Enter topic name to generate summaries", value="Normalization (1NF, 2NF, 3NF)")
        
        if st.button("Generate Scannable Summary Notes"):
            with st.spinner("Extracting context and synthesizing..."):
                res = requests.post(f"{BACKEND_URL}/generation/notes", json={"subject": subject, "topic": target_topic})
                if res.status_code == 200:
                    st.markdown("---")
                    st.markdown(res.json().get("markdown_content", ""))
                else:
                    st.error("Could not fetch notes context.")

    with tab3:
        st.subheader("Custom High-Speed Study Planner")
        col3, col4 = st.columns(2)
        with col3:
            days = st.number_input("Days remaining until exam", min_value=1, max_value=60, value=7)
        with col4:
            hours = st.number_input("Study availability hours per day", min_value=1.0, max_value=12.0, value=3.0, step=0.25)
            
        if st.button("Compile Preparation Schedule"):
            with st.spinner("Optimizing schedule via Gemini Flash..."):
                res = requests.post(f"{BACKEND_URL}/generation/study-plan", json={"subject": subject, "days_remaining": days, "daily_study_hours": hours})
                if res.status_code == 200:
                    schedule_data = res.json().get("schedule", "")
                    
                    # Check if backend returned an empty/grounded fallback string
                    if "know" in str(schedule_data).lower() or not schedule_data:
                        st.warning("Gemini couldn't map a plan because your Pinecone knowledge base is empty for this subject. Ingest a syllabus PDF using the sidebar first.")
                    else:
                        st.success("Custom timeline generated successfully.")
                        st.write(schedule_data)
                else:
                    st.error("Failed to generate plan.")

    with tab4:
        st.subheader("AI Career & Project Advisor")
        field_major = st.text_input("Enter your Field or Major (e.g., Biology, CSE, Finance)")
        skills_interest = st.text_area("Enter your Skills and Interests (e.g., Lab work, Python, Designing)")
        student_goal = st.text_input("What is your immediate goal? (e.g., Need a portfolio project, Need a career path roadmap)")
        
        if "current_advice" not in st.session_state:
            st.session_state.current_advice = None
            
        if st.button("Generate Personalized Guidance"):
            if not field_major.strip() or not student_goal.strip():
                st.warning("Please fill out your major and immediate goal!")
            else:
                with st.spinner("Consulting Gemini Career Advisor..."):
                    payload = {
                        "field_major": field_major,
                        "skills_interest": skills_interest,
                        "student_goal": student_goal
                    }
                    try:
                        res = requests.post(f"{BACKEND_URL}/career/advise", json=payload)
                        if res.status_code == 200:
                            st.session_state.current_advice = res.json().get("advice", "No advice returned.")
                        else:
                            st.error(f"Failed to fetch advice (Status {res.status_code}).")
                    except Exception as e:
                        st.error(f"Could not connect to backend: {e}")
                        
        if st.session_state.current_advice:
            st.markdown(st.session_state.current_advice)
            if st.button("Save this Roadmap to Profile"):
                try:
                    save_res = requests.post(
                        f"{BACKEND_URL}/career/save",
                        json={"subject_name": subject, "advice_text": st.session_state.current_advice},
                        headers=headers
                    )
                    if save_res.status_code == 200:
                        st.success("Roadmap saved successfully!")
                    else:
                        st.error("Failed to save roadmap.")
                except Exception as e:
                    st.error(f"Backend connection failed: {e}")
                    
        with st.expander("View Saved Roadmaps History"):
            try:
                hist_res = requests.get(f"{BACKEND_URL}/career/history", headers=headers)
                if hist_res.status_code == 200:
                    history = hist_res.json()
                    if history:
                        for idx, item in enumerate(history):
                            st.markdown(f"**Subject:** {item.get('subject_name', 'Unknown')} | **Saved:** {item.get('timestamp', 'Unknown')}")
                            st.markdown(item.get("advice_text", ""))
                            st.markdown("---")
                    else:
                        st.info("No saved roadmaps yet.")
                else:
                    st.error("Failed to fetch history.")
            except Exception as e:
                st.error(f"Backend connection failed: {e}")