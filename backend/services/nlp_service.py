import fitz
import re
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER

print("[NLP] Loading Semantic Similarity Model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("[NLP] Loading GLiNER Zero-Shot Extraction Model...")
ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

ENTITY_LABELS = ["technical skill", "soft skill", "software", "tool", "language", "certification", "methodology"]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF document."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    return text

def split_into_sentences(text: str) -> list:
    """Splits text into independent sentences for contextual evidence."""
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def extract_job_requirements(job_description: str) -> list:
    """Extracts required skills from the job description using GLiNER."""
    if not job_description.strip():
        return []
    
    entities = ner_model.predict_entities(job_description, ENTITY_LABELS, threshold=0.45)
    
    requirements = []
    seen = set()
    for ent in entities:
        skill = ent["text"].strip().title()
        if len(skill) > 2 and skill.lower() not in seen:
            seen.add(skill.lower())
            requirements.append({
                "skill": skill,
                "type": ent["label"]
            })
    return requirements

def extract_cv_evidence(cv_text: str) -> list:
    """Extracts skills from the CV and maps them to their exact source sentence."""
    sentences = split_into_sentences(cv_text)
    evidence_list = []
    
    for sentence in sentences:
        entities = ner_model.predict_entities(sentence, ENTITY_LABELS, threshold=0.45)
        for ent in entities:
            skill = ent["text"].strip().title()
            if len(skill) > 2:
                evidence_list.append({
                    "skill": skill,
                    "type": ent["label"],
                    "context": sentence
                })
    return evidence_list

def calculate_hybrid_match_score(job_description: str, cv_text: str) -> dict:
    """
    SOTA ATS Logic: Extracted Requirements vs Extracted CV Evidence via Semantic Cosine Similarity.
    """
    if not cv_text.strip():
        return {"score": 0.0, "match_results": [], "explanation": "Empty CV text."}

    job_reqs = extract_job_requirements(job_description)
    cv_evidence = extract_cv_evidence(cv_text)
    
    if not job_reqs:
        return {"score": 0.0, "match_results": [], "explanation": "No requirements identified."}

    cv_skills = [item["skill"] for item in cv_evidence]
    
    if not cv_skills:
        return {"score": 0.0, "match_results": [], "explanation": "No relevant skills identified in CV."}

    cv_embs = embedding_model.encode(cv_skills, convert_to_tensor=True)
    
    requirement_details = []
    matched_skills = []
    partial_requirements = []
    missing_skills = []
    
    total_score = 0.0

    for req in job_reqs:
        req_skill = req["skill"]
        req_emb = embedding_model.encode(req_skill, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(req_emb, cv_embs)[0]
        max_idx = cosine_scores.argmax().item()
        best_score = float(cosine_scores[max_idx].item())
        
        best_cv_match = cv_evidence[max_idx]
        
        if best_score >= 0.70:
            status = "Full Match"
            req_score = 100.0
            matched_skills.append(req_skill)
        elif best_score >= 0.50:
            status = "Partial Match"
            req_score = 50.0
            partial_requirements.append(req_skill)
        else:
            status = "Missing"
            req_score = 0.0
            missing_skills.append(req_skill)

        total_score += req_score

        requirement_details.append({
            "requirement": req_skill,
            "category": req["type"],
            "status": status,
            "confidence": round(best_score * 100, 1),
            "cv_evidence": best_cv_match["context"] if status != "Missing" else "No contextual evidence found."
        })

    final_score = round(total_score / len(job_reqs), 2) if job_reqs else 0.0
    
    explanation = f"Evaluated {len(job_reqs)} extracted requirements: {len(matched_skills)} matched, {len(partial_requirements)} partially matched, and {len(missing_skills)} missing. Verified via GLiNER and Semantic Similarity."

    return {
        "score": final_score,
        "match_results": requirement_details,
        "explanation": explanation
    }

def evaluate_bias(cv_text: str) -> dict:
    """Heuristic bias screening for demographic indicators."""
    text_lower = cv_text.lower()
    has_photo = bool(re.search(r'\b(photo|picture|fotografie|poza)\b', text_lower))
    has_marital = bool(re.search(r'\b(married|single|children|family|casatorit|necasatorit|copii)\b', text_lower))
    
    risk_factors = 0
    warnings = []
    
    if has_photo:
        risk_factors += 1
        warnings.append("CV contains photo references (risk of visual bias).")
    if has_marital:
        risk_factors += 1
        warnings.append("CV contains marital status or family details.")

    return {
        "bias_level": "Low" if risk_factors == 0 else ("Medium" if risk_factors == 1 else "High"),
        "risk_score_factors": risk_factors,
        "warnings": warnings,
        "recommendation": "Anonymization of personal data is recommended." if risk_factors > 0 else "No standard bias indicators detected."
    }

def extract_candidate_info(cv_text: str) -> dict:
    """Extracts basic contact info via regex."""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text)
    phone_match = re.search(r'(?:\+40|0040|0)?\s*([732][0-9]{8})\b', cv_text)
    
    lines = [line.strip() for line in cv_text.split('\n') if line.strip()]
    
    return {
        "full_name": lines[0] if lines else "Unknown Candidate",
        "email": email_match.group(0) if email_match else "Unspecified",
        "phone": phone_match.group(0) if phone_match else "Unspecified"
    }