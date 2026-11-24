import os
import shutil
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models
import schemas
from services.nlp_service import extract_text_from_pdf, calculate_hybrid_match_score, evaluate_bias, extract_candidate_info

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SOTA ATS & Bias Evaluation API", version="2.0")

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
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    cv_text = extract_text_from_pdf(file_path)
    info = extract_candidate_info(cv_text)
    
    db_candidate = models.Candidate(
        full_name=info["full_name"],
        email=info["email"],
        phone=info["phone"],
        file_path=file_path,
        parsed_text=cv_text
    )
    db.add(db_candidate)
    db.commit()
    db.refresh(db_candidate)
    
    hybrid_result = calculate_hybrid_match_score(job.description, cv_text)
    
    bias_report = evaluate_bias(cv_text)
    bias_report["match_details"] = {
        "explanation": hybrid_result["explanation"],
        "match_results": hybrid_result.get("match_results", [])
    }
    
    db_evaluation = models.Evaluation(
        job_id=job.id,
        candidate_id=db_candidate.id,
        match_score=hybrid_result["score"],
        bias_report=bias_report
    )
    db.add(db_evaluation)
    db.commit()
    
    return {
        "message": "CV processed successfully!",
        "candidate_id": db_candidate.id,
        "full_name": db_candidate.full_name,
        "match_score": hybrid_result["score"]
    }

@app.get("/jobs/{job_id}/leaderboard/")
def get_leaderboard(job_id: int, db: Session = Depends(get_db)):
    evaluations = db.query(models.Evaluation, models.Candidate)\
        .join(models.Candidate, models.Evaluation.candidate_id == models.Candidate.id)\
        .filter(models.Evaluation.job_id == job_id)\
        .order_by(models.Evaluation.match_score.desc())\
        .all()
        
    results = []
    for ev, cand in evaluations:
        results.append({
            "candidate_id": cand.id,
            "full_name": cand.full_name,
            "email": cand.email,
            "phone": cand.phone,
            "match_score": ev.match_score,
            "bias_report": ev.bias_report,
            "submitted_at": ev.created_at
        })
    return results