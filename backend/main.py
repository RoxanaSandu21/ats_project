import os
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models
import schemas
from services.nlp_service import (
    extract_text_from_pdf, calculate_hybrid_match_score,
    evaluate_bias, extract_candidate_info
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ATS & Evidence-Based Fairness Evaluation API", version="2.1")
UPLOAD_DIR = "uploaded_cvs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/jobs/", response_model=schemas.JobResponse)
def create_job(job: schemas.JobCreate, db: Session = Depends(get_db)):
    db_job = models.Job(title=job.title, description=job.description)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.get("/jobs/", response_model=list[schemas.JobResponse])
def get_jobs(db: Session = Depends(get_db)):
    return db.query(models.Job).all()


@app.post("/jobs/{job_id}/upload-cv/")
def upload_cv(job_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    stored_name = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(UPLOAD_DIR, stored_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    cv_text = extract_text_from_pdf(file_path)
    if not cv_text.strip():
        raise HTTPException(status_code=422, detail="No readable text was extracted from this PDF.")

    info = extract_candidate_info(cv_text)
    db_candidate = None
    if info["email"] != "Unspecified":
        db_candidate = db.query(models.Candidate).filter(
            models.Candidate.email == info["email"]
        ).first()

    if db_candidate:
        db_candidate.full_name = info["full_name"]
        db_candidate.phone = info["phone"]
        db_candidate.file_path = file_path
        db_candidate.parsed_text = cv_text
    else:
        db_candidate = models.Candidate(
            full_name=info["full_name"], email=info["email"],
            phone=info["phone"], file_path=file_path, parsed_text=cv_text
        )
        db.add(db_candidate)

    db.commit()
    db.refresh(db_candidate)

    result = calculate_hybrid_match_score(job.description, cv_text)
    bias_report = evaluate_bias(cv_text)
    bias_report["match_details"] = {
        key: result.get(key) for key in (
            "explanation", "match_results", "requirements", "excluded_job_information", "cv_evidence",
            "evidence_coverage", "evidence_covered_count", "requirements_count",
            "fully_matched", "partial_matches", "missing", "source_statement_count",
            "required_without_evidence", "required_only_score", "required_count"
        )
    }

    db.add(models.Evaluation(
        job_id=job.id, candidate_id=db_candidate.id,
        match_score=result["score"], bias_report=bias_report
    ))
    db.commit()

    return {
        "message": "CV processed successfully!",
        "candidate_id": db_candidate.id,
        "full_name": db_candidate.full_name,
        "match_score": result["score"],
        "evaluation_summary": {
            "evidence_coverage": result.get("evidence_coverage", 0),
            "evidence_covered_count": result.get("evidence_covered_count", 0),
            "requirements_count": result.get("requirements_count", 0),
            "source_statement_count": result.get("source_statement_count", 0),
            "fully_matched": result.get("fully_matched", 0),
            "partial_matches": result.get("partial_matches", 0),
            "missing": result.get("missing", 0),
            "required_without_evidence": result.get("required_without_evidence", 0),
            "required_only_score": result.get("required_only_score"),
            "required_count": result.get("required_count", 0)
        }
    }


@app.get("/jobs/{job_id}/leaderboard/")
def get_leaderboard(job_id: int, db: Session = Depends(get_db)):
    # Keep only the most recent evaluation per candidate for this job.
    evaluations = (
        db.query(models.Evaluation, models.Candidate)
        .join(models.Candidate, models.Evaluation.candidate_id == models.Candidate.id)
        .filter(models.Evaluation.job_id == job_id)
        .order_by(models.Evaluation.created_at.desc())
        .all()
    )

    latest_by_candidate = {}
    for ev, cand in evaluations:
        if cand.id not in latest_by_candidate:
            latest_by_candidate[cand.id] = {
                "candidate_id": cand.id,
                "full_name": cand.full_name,
                "email": cand.email,
                "phone": cand.phone,
                "match_score": ev.match_score,
                "bias_report": ev.bias_report,
                "submitted_at": ev.created_at
            }

    return sorted(
        latest_by_candidate.values(),
        key=lambda candidate: candidate["match_score"],
        reverse=True
    )
