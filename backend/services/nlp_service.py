import fitz
import re
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER

print("[NLP] Loading semantic similarity model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("[NLP] Loading GLiNER zero-shot extraction model...")
ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

ENTITY_LABELS = [
    "technical skill", "soft skill", "software", "tool",
    "language", "certification", "methodology", "job responsibility",
    "work experience", "qualification"
]
MIN_SENTENCE_LENGTH = 18


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF while preserving line breaks."""
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        print(f"[NLP] PDF parsing error: {exc}")
        return ""


def split_into_sentences(text: str) -> List[str]:
    """Split CV/job text into useful contextual units (bullets and sentences)."""
    chunks = []
    for line in re.split(r"\n+", text):
        line = re.sub(r"^\s*(?:[•●▪◦*\-–—]+|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
        chunks.extend(part.strip() for part in parts if len(part.strip()) >= MIN_SENTENCE_LENGTH)
    return chunks


def _unique_entities(text: str) -> List[dict]:
    if not text.strip():
        return []
    try:
        raw = ner_model.predict_entities(text, ENTITY_LABELS, threshold=0.40)
    except Exception as exc:
        print(f"[NLP] GLiNER extraction warning: {exc}")
        return []
    results, seen = [], set()
    for item in raw:
        value = item.get("text", "").strip()
        label = item.get("label", "other")
        key = (value.casefold(), label)
        if value and key not in seen:
            seen.add(key)
            results.append({
                "text": value,
                "type": label,
                "start": item.get("start"),
                "end": item.get("end")
            })
    return results


def _priority(text: str) -> str:
    lowered = text.casefold()
    preferred = ("nice to have", "preferred", "desirable", "bonus", "advantage",
                 "would be a plus", "ideally", "optional")
    required = ("must", "required", "requirement", "essential", "mandatory",
                "you have", "we require", "minimum")
    if any(term in lowered for term in preferred):
        return "preferred"
    if any(term in lowered for term in required):
        return "required"
    return "unspecified"


def _category(text: str, entities: List[dict]) -> str:
    labels = [e["type"] for e in entities]
    if "language" in labels:
        return "language"
    if "certification" in labels or "qualification" in labels:
        return "qualification"
    if "soft skill" in labels:
        return "soft skill"
    if any(x in labels for x in ("technical skill", "software", "tool", "methodology")):
        return "technical skill"
    lowered = text.casefold()
    if any(x in lowered for x in ("years of experience", "degree", "bachelor", "master")):
        return "experience / education"
    return "job criterion"


def extract_job_requirements(job_description: str) -> List[dict]:
    """
    Extract complete requirement statements, not just isolated entity names.
    GLiNER supplies entity/category hints; the original sentence is retained as
    the assessable requirement and priority is inferred from its wording.
    """
    requirements = []
    seen = set()
    for sentence in split_into_sentences(job_description):
        entities = _unique_entities(sentence)
        lowered = sentence.casefold()
        has_requirement_signal = bool(entities) or any(term in lowered for term in (
            "experience", "knowledge", "ability", "ability to", "proficient",
            "familiar", "understanding", "communication", "troubleshoot",
            "responsible", "support", "degree", "certification", "english"
        ))
        if not has_requirement_signal:
            continue
        normalized = re.sub(r"\s+", " ", sentence).strip()
        key = normalized.casefold().strip(" .;:-")
        if key in seen:
            continue
        seen.add(key)
        requirements.append({
            "requirement": normalized,
            "category": _category(normalized, entities),
            "priority": _priority(normalized),
            "entities": [{"text": e["text"], "type": e["type"]} for e in entities],
            "source_text": sentence
        })
    return requirements


def extract_cv_evidence(cv_text: str) -> List[dict]:
    """
    Keep complete CV sentences/bullets as evidence. GLiNER entities are attached
    as hints; matching never discards the surrounding responsibility/context.
    """
    evidence = []
    for sentence in split_into_sentences(cv_text):
        entities = _unique_entities(sentence)
        evidence.append({
            "text": sentence,
            "evidence_type": "responsibility_or_context",
            "entities": [{"text": e["text"], "type": e["type"]} for e in entities]
        })
    return evidence


def _entity_overlap(requirement: dict, evidence: dict) -> float:
    req_terms = {e["text"].casefold() for e in requirement.get("entities", [])}
    cv_terms = {e["text"].casefold() for e in evidence.get("entities", [])}
    if not req_terms:
        return 0.0
    return len(req_terms & cv_terms) / len(req_terms)


def _assess_evidence(requirement: dict, evidence: dict, similarity: float) -> tuple:
    overlap = _entity_overlap(requirement, evidence)
    # Similarity retrieves candidate evidence; it is not treated as proof by itself.
    if similarity >= 0.62 and (overlap > 0 or similarity >= 0.76):
        return "Full Match", "direct_or_strong", 100.0
    if similarity >= 0.40:
        return "Partial Match", "indirect_or_partial", 50.0
    return "Missing", "insufficient", 0.0


def calculate_hybrid_match_score(job_description: str, cv_text: str) -> dict:
    """Evaluate full job statements against contextual CV excerpts."""
    if not cv_text.strip():
        return {
            "score": 0.0, "match_results": [], "requirements": [],
            "cv_evidence": [], "explanation": "Empty CV text."
        }

    requirements = extract_job_requirements(job_description)
    evidence = extract_cv_evidence(cv_text)
    if not requirements:
        return {
            "score": 0.0, "match_results": [], "requirements": [],
            "cv_evidence": evidence, "explanation": "No job requirements could be extracted."
        }
    if not evidence:
        return {
            "score": 0.0, "match_results": [], "requirements": requirements,
            "cv_evidence": [], "explanation": "No usable CV text was extracted."
        }

    evidence_texts = [item["text"] for item in evidence]
    evidence_embeddings = embedding_model.encode(
        evidence_texts, convert_to_tensor=True, normalize_embeddings=True
    )
    requirement_details = []
    weighted_total = 0.0
    total_weight = 0.0
    required_total = 0.0
    required_weight = 0.0
    fully_matched = partial = missing = required_without_evidence = 0
    evidence_covered = 0

    for req in requirements:
        req_embedding = embedding_model.encode(
            req["requirement"], convert_to_tensor=True, normalize_embeddings=True
        )
        similarities = util.cos_sim(req_embedding, evidence_embeddings)[0]
        ranked = similarities.argsort(descending=True)[:3].tolist()
        best_idx = ranked[0]
        best_similarity = float(similarities[best_idx].item())
        best_evidence = evidence[best_idx]
        status, evidence_strength, points = _assess_evidence(req, best_evidence, best_similarity)

        weight = 1.0 if req["priority"] == "required" else (
            0.5 if req["priority"] == "preferred" else 1.0
        )
        weighted_total += points * weight
        total_weight += weight
        if req["priority"] == "required":
            required_total += points
            required_weight += 1
            if status == "Missing":
                required_without_evidence += 1
        if status == "Full Match":
            fully_matched += 1
        elif status == "Partial Match":
            partial += 1
        else:
            missing += 1
        if status != "Missing":
            evidence_covered += 1

        alternatives = []
        for idx in ranked[1:]:
            alternatives.append({
                "text": evidence[idx]["text"],
                "similarity": round(float(similarities[idx].item()), 3)
            })

        requirement_details.append({
            "requirement": req["requirement"],
            "category": req["category"],
            "priority": req["priority"],
            "status": status,
            "confidence": round(best_similarity * 100, 1),
            "evidence_strength": evidence_strength,
            "cv_evidence": best_evidence["text"] if status != "Missing" else "",
            "evidence_note": (
                "Relevant CV context retrieved; review the excerpt to confirm the claim."
                if status != "Missing" else
                "No sufficiently relevant contextual evidence was found. This does not prove the candidate lacks the skill."
            ),
            "alternative_evidence": alternatives
        })

    score = round(weighted_total / total_weight, 2) if total_weight else 0.0
    required_score = round(required_total / required_weight, 2) if required_weight else None
    coverage = round(evidence_covered / len(requirements) * 100, 2)
    required_count = sum(1 for r in requirements if r["priority"] == "required")
    explanation = (
        f"Evaluated {len(requirements)} complete job criteria: {fully_matched} fully matched, "
        f"{partial} partially supported, and {missing} without sufficient evidence. "
        "Similarity is used to retrieve evidence, not as proof of competence."
    )
    return {
        "score": score,
        "evidence_coverage": coverage,
        "evidence_covered_count": evidence_covered,
        "requirements_count": len(requirements),
        "fully_matched": fully_matched,
        "partial_matches": partial,
        "missing": missing,
        "required_without_evidence": required_without_evidence,
        "required_only_score": required_score,
        "required_count": required_count,
        "match_results": requirement_details,
        "requirements": requirements,
        "cv_evidence": evidence,
        "explanation": explanation
    }


def evaluate_bias(cv_text: str) -> dict:
    """Heuristic screening only; a low result is not proof of fairness."""
    text_lower = cv_text.lower()
    has_photo = bool(re.search(r"\b(photo|picture|fotografie|poza)\b", text_lower))
    has_marital = bool(re.search(r"\b(married|single|children|family|casatorit|necasatorit|copii)\b", text_lower))
    warnings = []
    if has_photo:
        warnings.append("CV contains photo references (potential visual-bias exposure).")
    if has_marital:
        warnings.append("CV contains marital-status or family details.")
    return {
        "bias_level": "Low" if not warnings else ("Medium" if len(warnings) == 1 else "High"),
        "risk_score_factors": len(warnings),
        "warnings": warnings,
        "recommendation": "Review and minimize unnecessary personal data." if warnings else
            "No configured indicators detected. This heuristic check does not establish that the process is fair."
    }


def extract_candidate_info(cv_text: str) -> dict:
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", cv_text)
    phone_match = re.search(r"(?:\+40|0040|0)?\s*([732][0-9]{8})\b", cv_text)
    lines = [line.strip() for line in cv_text.split("\n") if line.strip()]
    name = lines[0] if lines else "Unknown Candidate"
    return {
        "full_name": name[:255],
        "email": email_match.group(0) if email_match else "Unspecified",
        "phone": phone_match.group(0) if phone_match else "Unspecified"
    }
