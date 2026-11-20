import streamlit as st
from sws import store
from views import dashboard, predict, results, upload, model_report, subjects as subjects_view, components as X

st.set_page_config(page_title="Student Performance Warning System", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="First start: training models on the seed data (about 20 seconds)...")
def _boot():
    store.bootstrap_seed()
    return True


_boot()
subjects = store.list_subjects()

st.sidebar.title("🎓 Student Performance\nWarning System")
PAGES = ["Dashboard", "Predict a student", "Upload data", "Student results", "Model report", "Subjects"]
page = st.sidebar.radio("Navigate", PAGES)

if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"))

if not subjects:
    st.sidebar.info("No subjects yet.")
    upload.render([])
    st.stop()

if st.session_state.get("subject") not in subjects:
    st.session_state["subject"] = subjects[0]
subject = st.sidebar.selectbox("Subject", subjects, key="subject")

if page == "Dashboard":
    dashboard.render(subject)
elif page == "Predict a student":
    predict.render(subject)
elif page == "Upload data":
    upload.render(subjects)
elif page == "Student results":
    results.render(subject)
elif page == "Model report":
    model_report.render(subject, X.get_bundle(subject))
else:
    subjects_view.render(subjects)
