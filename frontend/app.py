import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ATS AI Evaluation", page_icon="📊", layout="wide")

st.title("📄 AI-Powered ATS & Algorithmic Fairness Analysis")
st.write("Master's Project — Semantic CV-Job Matching via Zero-Shot NER (GLiNER) and Vector Embeddings.")

menu = st.sidebar.selectbox("Navigation", ["Manage Jobs", "Evaluate Candidates"])

if menu == "Manage Jobs":
    st.header("📌 Add New Job Post")
    with st.form("job_form"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description (Requirements & Tech Stack)")
        if st.form_submit_button("Create Job"):
            if title and description:
                res = requests.post(f"{API_URL}/jobs/", json={"title": title, "description": description})
                if res.status_code == 200:
                    st.success("Job created successfully!")
            else:
                st.warning("All fields are required.")

    st.subheader("📋 Active Jobs")
    res = requests.get(f"{API_URL}/jobs/")
    if res.status_code == 200 and res.json():
        for job in res.json():
            with st.expander(f"🔹 {job['title']} (ID: {job['id']})"):
                st.write(job['description'])

elif menu == "Evaluate Candidates":
    st.header("📂 Candidate CV Compatibility")
    
    res = requests.get(f"{API_URL}/jobs/")
    if res.status_code == 200 and res.json():
        jobs = {f"{j['title']} (ID: {j['id']})": j['id'] for j in res.json()}
        selected_job = st.selectbox("Select Target Job", list(jobs.keys()))
        job_id = jobs[selected_job]
        
        st.divider()
        uploaded_file = st.file_uploader("Upload CV (.pdf)", type=["pdf"])
        
        if uploaded_file and st.button("Process & Evaluate CV"):
            with st.spinner("Extracting entities and computing semantic overlap..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                r = requests.post(f"{API_URL}/jobs/{job_id}/upload-cv/", files=files)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"Processed: {data['full_name']} | Compatibility: {data['match_score']}%")
        
        st.divider()
        st.subheader("🏆 Live Leaderboard & AI Justification")
        
        lb_res = requests.get(f"{API_URL}/jobs/{job_id}/leaderboard/")
        if lb_res.status_code == 200 and lb_res.json():
            for idx, cand in enumerate(lb_res.json(), 1):
                color = "🟢" if cand['match_score'] >= 75 else "🟡" if cand['match_score'] >= 50 else "🔴"
                with st.expander(f"#{idx} - {cand['full_name']} {color} ({cand['match_score']}%)"):
                    st.write(f"**Contact:** {cand['email']} | {cand['phone']}")
                    
                    st.markdown("#### 🔍 Structural Matching Evidence")
                    match_details = cand['bias_report'].get("match_details", {})
                    st.write(match_details.get("explanation", ""))
                    
                    results = match_details.get("match_results", [])
                    if results:
                        df = pd.DataFrame(results)
                        def highlight(val):
                            if val == 'Full Match': return 'background-color: #c6efce; color: #006100'
                            if val == 'Partial Match': return 'background-color: #ffeb9c; color: #9c5700'
                            if val == 'Missing': return 'background-color: #ffc7ce; color: #9c0006'
                            return ''
                        st.dataframe(df.style.map(highlight, subset=['status']), use_container_width=True)
                    
                    st.markdown("#### ⚖️ Algorithmic Fairness Check")
                    st.json({k: v for k, v in cand['bias_report'].items() if k != 'match_details'})