import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ATS AI Evaluation", page_icon="📊", layout="wide")
st.title("📄 AI-Powered ATS & Algorithmic Fairness Analysis")
st.write("Master's Project — GLiNER extraction, contextual semantic retrieval, and explainable CV-job evaluation.")

menu = st.sidebar.selectbox("Navigation", ["Manage Jobs", "Evaluate Candidates"])


def show_evaluation_details(bias_report, weighted_score=None):
    details = (bias_report or {}).get("match_details", {})
    st.markdown("#### 📊 Evaluation overview")
    if weighted_score is not None:
        st.metric("Weighted compatibility", f"{weighted_score:.2f}%")

    saved_results = details.get("match_results", [])
    if saved_results:
        # Derive this count from stored statuses so older database evaluations
        # are displayed correctly after the coverage definition changed.
        context_count = sum(
            1 for item in saved_results
            if item.get("status") in ("Full Match", "Partial Match")
        )
        criteria_count = len(saved_results)
    else:
        context_count = details.get("evidence_covered_count", 0)
        criteria_count = details.get("requirements_count", 0)
    cols = st.columns(5)
    cols[0].metric(
        "Criteria with direct/partial evidence",
        f"{context_count}/{criteria_count}",
        help="Counts only Full Match and Partial Match criteria. Transferable Context is shown separately because it is not direct evidence."
    )
    cols[1].metric("Fully matched", details.get("fully_matched", 0))
    cols[2].metric("Partially matched", details.get("partial_matches", 0))
    cols[3].metric("Transferable context", details.get("transferable_contexts", 0))
    cols[4].metric("Missing", details.get("missing", 0))

    required_cols = st.columns(2)
    required_cols[0].metric(
        "Required without direct evidence",
        details.get("required_without_evidence", 0)
    )
    required_score = details.get("required_only_score")
    required_cols[1].metric(
        "Required-only score",
        f"{required_score}%" if required_score is not None else "N/A"
    )
    st.caption(
        "Weighted compatibility uses 100 points for Full Match, 50 for Partial Match, "
        "25 for Transferable Context, and 0 for Missing, divided by the total criterion "
        "weights. Criteria with direct/partial evidence is only a count; it does not measure evidence strength."
    )

    st.caption(details.get("explanation", "No evaluation explanation available."))
    results = details.get("match_results", [])
    if results:
        st.markdown("#### 🔍 Requirement-by-requirement evidence")
        for index, item in enumerate(results, 1):
            status = item.get("status", "Not assessed")
            icon = "🟢" if status == "Full Match" else "🟡" if status in ("Partial Match", "Transferable Context") else "🔴"
            priority = item.get("priority", "unspecified").title()
            with st.expander(f"{icon} {index}. {item.get('requirement', 'Criterion')} — {status} ({priority})"):
                st.write(f"**Category:** {item.get('category', 'Not classified')}")
                st.write(f"**Criterion type:** {item.get('criterion_type', 'Not classified')}")
                st.write(f"**Priority source:** {item.get('priority_source', 'Not stated')}")
                st.write(f"**Semantic retrieval similarity:** {item.get('confidence', 0)}%")
                st.write(f"**Evidence assessment:** {item.get('evidence_strength', 'Not available')}")
                excerpt = item.get("cv_evidence", "")
                if excerpt:
                    st.markdown("**Potentially transferable CV context (not direct evidence)**" if status == "Transferable Context" else "**Retrieved supporting CV excerpt**")
                    st.info(excerpt)
                else:
                    closest = item.get("closest_evidence", {})
                    if closest.get("text"):
                        st.markdown("**Closest retrieved CV passage (not counted as evidence)**")
                        st.info(
                            f"{closest.get('text', '')}\n\n"
                            f"Passage type: {closest.get('evidence_type', 'context')} | "
                            f"Retrieval similarity: {round(closest.get('similarity', 0) * 100, 1)}%"
                        )
                    else:
                        st.warning("No substantive CV passage was available for comparison.")
                st.caption(item.get("evidence_note", ""))
                alternatives = item.get("alternative_evidence", [])
                if alternatives:
                    st.markdown("**Other retrieved excerpts (not automatically treated as proof)**")
                    for alternative in alternatives:
                        st.write(f"• {alternative.get('text', '')} — similarity {alternative.get('similarity', 0)}")

    requirements = details.get("requirements", [])
    if requirements:
        with st.expander("📌 Extracted job criteria"):
            display_requirements = [{
                **item,
                "entities": ", ".join(entity.get("text", "") for entity in item.get("entities", []))
            } for item in requirements]
            st.dataframe(pd.DataFrame(display_requirements), use_container_width=True, hide_index=True)

    excluded = details.get("excluded_job_information", [])
    if excluded:
        with st.expander("🚫 Excluded job-post information (not scored)"):
            st.caption("These statements describe employer-provided benefits or compensation, not candidate qualifications or responsibilities.")
            st.dataframe(pd.DataFrame(excluded), use_container_width=True, hide_index=True)

    cv_evidence = details.get("cv_evidence", [])
    if cv_evidence:
        with st.expander("📄 Extracted CV context"):
            st.dataframe(
                pd.DataFrame([{"CV excerpt": item.get("text", ""),
                               "Evidence type": item.get("evidence_type", "context"),
                               "GLiNER entities": ", ".join(
                                   entity.get("text", "") for entity in item.get("entities", [])
                               )} for item in cv_evidence]),
                use_container_width=True, hide_index=True
            )

    st.markdown("#### ⚖️ Algorithmic fairness screening")
    st.json({k: v for k, v in (bias_report or {}).items() if k != "match_details"})
    st.caption("The fairness check is heuristic and limited to configured textual indicators. A 'Low' result does not establish that the model or recruitment process is fair.")


if menu == "Manage Jobs":
    st.header("📌 Add New Job Post")
    with st.form("job_form"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description (responsibilities and requirements)", height=220)
        submitted = st.form_submit_button("Create Job")
        if submitted:
            if title and description:
                try:
                    response = requests.post(
                        f"{API_URL}/jobs/",
                        json={"title": title, "description": description},
                        timeout=30
                    )
                    if response.status_code == 200:
                        st.success("Job created successfully!")
                    else:
                        st.error(f"Could not create job: {response.text}")
                except requests.RequestException as exc:
                    st.error(f"API connection error: {exc}")
            else:
                st.warning("All fields are required.")

    st.subheader("📋 Active Jobs")
    try:
        response = requests.get(f"{API_URL}/jobs/", timeout=15)
        if response.status_code == 200 and response.json():
            for job in response.json():
                with st.expander(f"🔹 {job['title']} (ID: {job['id']})"):
                    st.write(job["description"])
        elif response.status_code == 200:
            st.info("No jobs have been created yet.")
    except requests.RequestException as exc:
        st.error(f"API connection error: {exc}")

elif menu == "Evaluate Candidates":
    st.header("📂 Candidate CV Compatibility")
    try:
        response = requests.get(f"{API_URL}/jobs/", timeout=15)
        response.raise_for_status()
        available_jobs = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not load jobs: {exc}")
        available_jobs = []

    if available_jobs:
        jobs = {f"{job['title']} (ID: {job['id']})": job["id"] for job in available_jobs}
        selected_job = st.selectbox("Select Target Job", list(jobs.keys()))
        job_id = jobs[selected_job]

        st.divider()
        uploaded_file = st.file_uploader("Upload CV (.pdf)", type=["pdf"])
        if uploaded_file and st.button("Process & Evaluate CV"):
            with st.spinner("Extracting job criteria and CV context, then retrieving evidence..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    result = requests.post(
                        f"{API_URL}/jobs/{job_id}/upload-cv/",
                        files=files,
                        timeout=900
                    )
                    if result.status_code == 200:
                        data = result.json()
                        st.success(
                            f"Processed: {data['full_name']} | Weighted compatibility: {data['match_score']}%"
                        )
                        summary = data.get("evaluation_summary", {})
                        context_count = summary.get("evidence_covered_count", 0)
                        criteria_count = summary.get("requirements_count", 0)
                        cols = st.columns(5)
                        cols[0].metric(
                            "Criteria with direct/partial evidence",
                            f"{context_count}/{criteria_count}",
                            help="Counts only Full Match and Partial Match criteria. Transferable Context is reported separately."
                        )
                        cols[1].metric("Fully matched", summary.get("fully_matched", 0))
                        cols[2].metric("Partially matched", summary.get("partial_matches", 0))
                        cols[3].metric("Transferable context", summary.get("transferable_contexts", 0))
                        cols[4].metric("Missing", summary.get("missing", 0))
                        required_cols = st.columns(2)
                        required_cols[0].metric(
                            "Required without direct evidence",
                            summary.get("required_without_evidence", 0)
                        )
                        required_score = summary.get("required_only_score")
                        required_cols[1].metric(
                            "Required-only score",
                            f"{required_score}%" if required_score is not None else "N/A"
                        )
                        st.caption(
                            "Weighted compatibility uses evidence points (100/50/25/0). Direct/partial "
                            "evidence coverage counts only Full and Partial matches; Transferable Context "
                            "is shown separately."
                        )
                    else:
                        st.error(f"Evaluation failed: {result.text}")
                except requests.RequestException as exc:
                    st.error(f"API request failed: {exc}")

        st.divider()
        st.subheader("🏆 Live Leaderboard & Explainable Evidence")
        try:
            leaderboard_response = requests.get(
                f"{API_URL}/jobs/{job_id}/leaderboard/", timeout=30
            )
            leaderboard_response.raise_for_status()
            leaderboard = leaderboard_response.json()
            if leaderboard:
                for index, candidate in enumerate(leaderboard, 1):
                    score = candidate.get("match_score", 0)
                    color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
                    with st.expander(f"#{index} — {candidate.get('full_name', 'Unknown')} {color} ({score}%)"):
                        st.caption(f"Last evaluated: {candidate.get('submitted_at', 'Unknown')}")
                        st.write(f"**Contact:** {candidate.get('email', 'Unspecified')} | {candidate.get('phone', 'Unspecified')}")
                        show_evaluation_details(candidate.get("bias_report", {}), candidate.get("match_score"))
            else:
                st.info("No candidates have been evaluated for this job yet.")
        except requests.RequestException as exc:
            st.error(f"Could not load the leaderboard: {exc}")
    else:
        st.info("Create a job first in the Manage Jobs section.")
