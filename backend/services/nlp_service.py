import fitz
import re
from typing import Dict, List, Tuple
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

# These identify employer-provided compensation/perks, not candidate criteria.
# This is deliberately separate from the skill/entity vocabulary.
EMPLOYER_BENEFIT_CUES = (
    "retail voucher", "voucher", "reimburse", "reimbursement",
    "tuition fee", "dedicated training department", "benefits package",
    "employee benefits", "health insurance", "meal ticket", "meal voucher",
    "salary", "annual leave", "paid leave", "pension contribution",
    "staff discount", "employee discount"
)

QUALIFICATION_CUES = (
    "experience", "knowledge", "ability", "understanding", "proficient",
    "proficiency", "familiar", "degree", "certification", "qualification",
    "spoken and written", "language skills"
)

RESPONSIBILITY_CUES = (
    "provide", "provides", "provided", "providing", "act as", "record",
    "records", "recording", "log", "logging", "track", "tracking", "troubleshoot", "troubleshooting", "manage",
    "manages", "maintain", "maintains", "assist", "assists", "monitor",
    "monitors", "support", "supports", "ensure", "ensures", "resolve",
    "resolves", "handle", "handles", "coordinate", "coordinates",
    "deliver", "delivers", "administer", "administers"
)

ACTION_CUES = (
    "developed", "built", "created", "designed", "implemented", "provided",
    "supported", "managed", "maintained", "troubleshot", "resolved",
    "assisted", "coordinated", "recorded", "monitored", "tested",
    "documented", "handled", "worked", "collaborated", "contributed",
    "delivered", "configured", "installed", "diagnosed", "helped",
    "improved", "deployed", "analysed", "analyzed"
)

PREFERRED_PATTERN = re.compile(
    r"\b(?:preferred|desirable|bonus|advantage|ideally|optional)\b|"
    r"nice to have|would be a plus",
    re.IGNORECASE
)
REQUIRED_PATTERN = re.compile(
    r"\b(?:must|required|essential|mandatory|minimum)\b|"
    r"we require|you are required",
    re.IGNORECASE
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF while preserving page-level line breaks."""
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        print(f"[NLP] PDF parsing error: {exc}")
        return ""


def _strip_bullet(line: str) -> Tuple[str, bool]:
    match = re.match(r"^\s*([•●▪◦*\-–—]+|\d+[.)])\s*", line)
    if match:
        return line[match.end():].strip(), True
    return line.strip(), False


def _looks_like_heading(line: str) -> bool:
    """Identify short title-like lines so they are not merged into body text."""
    words = line.split()
    if not words or len(words) > 8 or line.endswith((".", "?", "!", ":", ";", ",")):
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", line):
        return True
    # Short title-case / uppercase lines are usually CV headings or role titles.
    letters = [char for char in line if char.isalpha()]
    if letters and (all(char.isupper() for char in letters) or
                    sum(word[:1].isupper() for word in words if word) >= max(2, len(words) - 2)):
        return True
    return False


def _join_wrapped_lines(text: str) -> List[str]:
    """
    Rejoin PDF visual line wraps without merging distinct bullets or headings.
    A wrapped continuation commonly starts in lowercase or closes punctuation
    opened on the previous physical line.
    """
    logical_lines = []
    current = ""

    for raw_line in text.replace("\u200b", "").splitlines():
        line, is_bullet = _strip_bullet(raw_line)
        if not line:
            if current:
                logical_lines.append(current.strip())
                current = ""
            continue

        if not current:
            current = line
            continue

        current_ends_sentence = current.rstrip().endswith((".", "?", "!", ";"))
        continuation = (
            line[:1].islower()
            or line.startswith((")", "]", ",", ";", ":"))
            or current.rstrip().endswith((",", "(", "/", "-", "–", "and", "or"))
        )
        if is_bullet or current_ends_sentence or _looks_like_heading(line):
            logical_lines.append(current.strip())
            current = line
        elif continuation:
            current = f"{current.rstrip()} {line.lstrip()}"
        else:
            logical_lines.append(current.strip())
            current = line

    if current:
        logical_lines.append(current.strip())
    return logical_lines


def split_into_sentences(text: str) -> List[str]:
    """Split text into contextual units while repairing common PDF line wraps."""
    chunks = []
    for logical_line in _join_wrapped_lines(text):
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", logical_line)
        for part in parts:
            # Semicolon-separated job duties often contain separate criteria.
            clauses = re.split(r"\s*;\s*(?=[A-Z])", part) if ";" in part else [part]
            chunks.extend(
                re.sub(r"\s+", " ", clause).strip()
                for clause in clauses
                if len(clause.strip()) >= 3
            )
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


def _is_employer_benefit(text: str) -> bool:
    lowered = text.casefold()
    return any(cue in lowered for cue in EMPLOYER_BENEFIT_CUES)


def _category(text: str, entities: List[dict]) -> str:
    lowered = text.casefold()
    labels = [entity["type"] for entity in entities]

    if "language" in labels or re.search(r"\benglish\b|\bspoken and written\b", lowered):
        return "language"
    if "soft skill" in labels or any(
        cue in lowered for cue in ("work under pressure", "communication skills", "interpersonal skills")
    ):
        return "soft skill"
    if "certification" in labels or "qualification" in labels:
        return "qualification"
    if any(label in labels for label in ("technical skill", "software", "tool", "methodology")):
        return "technical skill"
    if any(cue in lowered for cue in QUALIFICATION_CUES):
        return "qualification"
    return "job criterion"


def _criterion_type(text: str, category: str) -> str:
    lowered = text.casefold()
    # Explicit qualification wording takes precedence over action-like words.
    # For example, "troubleshooting experience" is a qualification, while
    # "you will troubleshoot..." is a job responsibility.
    if any(cue in lowered for cue in QUALIFICATION_CUES):
        return "qualification"
    if any(cue in lowered for cue in RESPONSIBILITY_CUES):
        return "responsibility"
    if category in ("language", "soft skill", "qualification", "technical skill"):
        return "qualification"
    return "job criterion"


def _priority(text: str, criterion_type: str) -> Tuple[str, str]:
    if PREFERRED_PATTERN.search(text):
        return "preferred", "explicit wording"
    if REQUIRED_PATTERN.search(text):
        return "required", "explicit wording"
    # A stated qualification (e.g. English proficiency or troubleshooting
    # experience) is treated as required unless the advert marks it preferred.
    # Generic plural wording such as "support requirements" is not an explicit
    # priority signal and must not accidentally mark a criterion as required.
    if criterion_type == "qualification":
        return "required", "inferred from qualification wording"
    return "unspecified", "not stated in the advert"


def extract_job_criteria(job_description: str) -> Tuple[List[dict], List[dict]]:
    """
    Extract assessable candidate criteria and separately retain employer benefits.
    GLiNER provides entity hints; the complete source statement remains visible.
    """
    requirements, excluded, seen = [], [], set()
    for sentence in split_into_sentences(job_description):
        entities = _unique_entities(sentence)
        normalized = re.sub(r"\s+", " ", sentence).strip()
        key = normalized.casefold().strip(" .;:-")
        if not key or key in seen:
            continue
        seen.add(key)

        if _is_employer_benefit(normalized):
            excluded.append({
                "statement": normalized,
                "category": "employer benefit",
                "reason": "Employer-provided benefit or compensation; not a candidate qualification or responsibility."
            })
            continue

        lowered = normalized.casefold()
        has_requirement_signal = bool(entities) or any(
            cue in lowered for cue in (
                "experience", "knowledge", "ability", "proficient",
                "familiar", "understanding", "communication", "troubleshoot",
                "responsible", "support", "degree", "certification", "english",
                "provide", "record", "log", "logging", "track", "tracking", "incident", "manage", "maintain", "assist", "monitor"
            )
        )
        if not has_requirement_signal:
            continue

        category = _category(normalized, entities)
        criterion_type = _criterion_type(normalized, category)
        priority, priority_source = _priority(normalized, criterion_type)
        requirements.append({
            "requirement": normalized,
            "category": category,
            "criterion_type": criterion_type,
            "priority": priority,
            "priority_source": priority_source,
            "entities": [{"text": entity["text"], "type": entity["type"]} for entity in entities],
            "source_text": sentence
        })
    return requirements, excluded


def extract_job_requirements(job_description: str) -> List[dict]:
    """Backward-compatible helper returning only assessable criteria."""
    requirements, _ = extract_job_criteria(job_description)
    return requirements


def _looks_like_cv_heading(text: str) -> bool:
    words = text.split()
    if len(words) <= 8 and not re.search(r"[.!?]", text):
        letters = [char for char in text if char.isalpha()]
        if letters and sum(word[:1].isupper() for word in words if word) >= max(2, len(words) - 1):
            return True
    return False


def _is_skills_list(text: str) -> bool:
    lowered = text.casefold()
    return any(label in lowered for label in (
        "tools used:", "languages:", "frameworks", "databases:", "tools:"
    ))


def extract_cv_evidence(cv_text: str) -> List[dict]:
    """
    Retain complete CV passages and distinguish action statements, skills lists,
    generic context, and short headings. Headings are displayed but not used as
    evidence of competence.
    """
    evidence = []
    for sentence in split_into_sentences(cv_text):
        entities = _unique_entities(sentence)
        lowered = sentence.casefold()
        if _looks_like_cv_heading(sentence):
            evidence_type = "heading_or_title"
        elif _is_skills_list(sentence):
            evidence_type = "skills_list"
        elif any(cue in lowered for cue in ACTION_CUES) or any(
            entity["type"] in ("job responsibility", "work experience") for entity in entities
        ):
            evidence_type = "action_statement"
        else:
            evidence_type = "context"

        evidence.append({
            "text": sentence,
            "evidence_type": evidence_type,
            "entities": [{"text": entity["text"], "type": entity["type"]} for entity in entities]
        })
    return evidence


def _entity_overlap(requirement: dict, evidence: dict) -> float:
    """Return exact phrase overlap between GLiNER entity spans."""
    req_terms = {
        re.sub(r"\s+", " ", entity["text"].casefold()).strip(" .,:;")
        for entity in requirement.get("entities", [])
    }
    cv_terms = {
        re.sub(r"\s+", " ", entity["text"].casefold()).strip(" .,:;")
        for entity in evidence.get("entities", [])
    }
    req_terms.discard("")
    cv_terms.discard("")
    if not req_terms:
        return 0.0
    return len(req_terms & cv_terms) / len(req_terms)


def _semantic_entity_overlap(
    requirement: dict,
    evidence: dict,
    entity_embeddings=None,
    entity_indices=None
) -> float:
    """
    Estimate whether extracted entity phrases are semantically related.
    Entity vectors are precomputed once per evaluation to avoid repeated
    transformer calls for every requirement/evidence pair.
    """
    req_terms = [entity["text"].casefold() for entity in requirement.get("entities", []) if entity.get("text")]
    cv_terms = [entity["text"].casefold() for entity in evidence.get("entities", []) if entity.get("text")]
    if not req_terms or not cv_terms:
        return 0.0

    if entity_embeddings is None or entity_indices is None:
        return _entity_overlap(requirement, evidence)

    req_positions = [entity_indices[term] for term in req_terms if term in entity_indices]
    cv_positions = [entity_indices[term] for term in cv_terms if term in entity_indices]
    if not req_positions or not cv_positions:
        return _entity_overlap(requirement, evidence)

    similarities = util.cos_sim(
        entity_embeddings[req_positions],
        entity_embeddings[cv_positions]
    )
    best_per_requirement = similarities.max(dim=1).values
    return float(best_per_requirement.mean().item())

def _assess_evidence(requirement: dict, evidence: dict, similarity: float, entity_embeddings=None, entity_indices=None) -> Tuple[str, str, float]:
    """
    Semantic similarity retrieves passages. Evidence is assessed using the
    passage type plus exact or semantic entity alignment. A strong sentence
    similarity alone can only produce a review-needed partial result.
    """
    evidence_type = evidence.get("evidence_type", "context")
    criterion_type = requirement.get("criterion_type", "job criterion")

    if evidence_type == "heading_or_title":
        return "Missing", "title_only_not_evidence", 0.0

    if criterion_type == "responsibility" and evidence_type == "skills_list":
        return "Missing", "skills_list_does_not_demonstrate_responsibility", 0.0

    exact_overlap = _entity_overlap(requirement, evidence)
    semantic_overlap = _semantic_entity_overlap(requirement, evidence, entity_embeddings, entity_indices)
    entity_support = max(exact_overlap, semantic_overlap)

    # A responsibility needs a substantive action passage. A skill list may
    # support a skill/qualification, but cannot prove that the candidate has
    # performed the responsibility.
    if entity_support >= 0.55 and similarity >= 0.38:
        if (
            similarity >= 0.66
            and entity_support >= 0.70
            and evidence_type == "action_statement"
        ):
            return "Full Match", "strong_semantic_entity_and_action_evidence", 100.0
        if (
            similarity >= 0.68
            and entity_support >= 0.72
            and evidence_type == "skills_list"
            and criterion_type == "qualification"
        ):
            return "Full Match", "strong_semantic_entity_and_skills_evidence", 100.0
        return "Partial Match", "semantically_aligned_entity_and_context", 50.0

    # A transferable-context tier captures related experience without claiming
    # that it proves the exact job requirement. It is deliberately worth less
    # than a partial match and remains visibly distinct in the UI.
    #
    # For example, a software project may show related technical problem-solving,
    # but it does not prove service-desk, room-booking, or AV support experience.
    if criterion_type == "responsibility" and evidence_type == "action_statement":
        if similarity >= 0.44:
            return "Transferable Context", "related_action_context_manual_review_required", 25.0

    if criterion_type == "qualification":
        if evidence_type == "skills_list" and similarity >= 0.44:
            return "Transferable Context", "related_skills_context_manual_review_required", 25.0
        if evidence_type == "action_statement" and similarity >= 0.44:
            return "Transferable Context", "related_action_context_manual_review_required", 25.0

    if evidence_type == "action_statement" and similarity >= 0.48:
        return "Transferable Context", "related_context_manual_review_required", 25.0

    return "Missing", "retrieved_passage_lacks_sufficient_support", 0.0

def calculate_hybrid_match_score(job_description: str, cv_text: str) -> dict:
    """Evaluate complete job criteria against contextual CV passages."""
    if not cv_text.strip():
        return {
            "score": 0.0, "match_results": [], "requirements": [],
            "excluded_job_information": [], "cv_evidence": [],
            "explanation": "Empty CV text."
        }

    requirements, excluded = extract_job_criteria(job_description)
    evidence = extract_cv_evidence(cv_text)
    if not requirements:
        return {
            "score": 0.0, "match_results": [], "requirements": [],
            "excluded_job_information": excluded, "cv_evidence": evidence,
            "explanation": "No assessable job criteria could be extracted."
        }
    if not evidence:
        return {
            "score": 0.0, "match_results": [], "requirements": requirements,
            "excluded_job_information": excluded, "cv_evidence": [],
            "explanation": "No usable CV text was extracted."
        }

    evidence_texts = [item["text"] for item in evidence]
    evidence_embeddings = embedding_model.encode(
        evidence_texts, convert_to_tensor=True, normalize_embeddings=True
    )

    entity_terms = list(dict.fromkeys(
        entity["text"].casefold()
        for record in requirements + evidence
        for entity in record.get("entities", [])
        if entity.get("text")
    ))
    entity_indices = {term: index for index, term in enumerate(entity_terms)}
    entity_embeddings = (
        embedding_model.encode(
            entity_terms, convert_to_tensor=True, normalize_embeddings=True
        )
        if entity_terms else None
    )

    requirement_details = []
    weighted_total = total_weight = 0.0
    required_total = required_weight = 0.0
    fully_matched = partial = transferable = missing = required_without_evidence = 0
    evidence_covered = 0

    for req in requirements:
        req_embedding = embedding_model.encode(
            req["requirement"], convert_to_tensor=True, normalize_embeddings=True
        )
        similarities = util.cos_sim(req_embedding, evidence_embeddings)[0]

        # Headings and titles can be useful context, but they are not competence evidence.
        candidate_indices = [
            index for index, item in enumerate(evidence)
            if item.get("evidence_type") != "heading_or_title"
        ]
        if not candidate_indices:
            requirement_details.append({
                "requirement": req["requirement"],
                "category": req["category"],
                "criterion_type": req["criterion_type"],
                "priority": req["priority"],
                "priority_source": req["priority_source"],
                "status": "Missing",
                "confidence": 0.0,
                "evidence_strength": "no_substantive_cv_passage",
                "cv_evidence": "",
                "evidence_note": "Only headings or titles were available; they are not treated as evidence of competence.",
                "alternative_evidence": []
            })
            missing += 1
            if req["priority"] == "required":
                required_without_evidence += 1
                required_weight += 1
            weight = 0.5 if req["priority"] == "preferred" else 1.0
            total_weight += weight
            continue

        # Assess every substantive passage before selecting evidence. A weaker
        # similarity result with direct entity support should outrank an unrelated
        # passage with a slightly higher embedding score.
        assessed = {}
        for candidate_idx in candidate_indices:
            candidate_similarity = float(similarities[candidate_idx].item())
            assessed[candidate_idx] = _assess_evidence(
                req, evidence[candidate_idx], candidate_similarity,
                entity_embeddings, entity_indices
            )

        status_rank = {"Full Match": 3, "Partial Match": 2, "Transferable Context": 1, "Missing": 0}
        ranked = sorted(
            candidate_indices,
            key=lambda index: (
                status_rank[assessed[index][0]],
                float(similarities[index].item())
            ),
            reverse=True
        )[:3]
        best_idx = ranked[0]
        best_similarity = float(similarities[best_idx].item())
        best_evidence = evidence[best_idx]
        status, evidence_strength, points = assessed[best_idx]

        closest_idx = max(
            candidate_indices,
            key=lambda index: float(similarities[index].item())
        )
        closest_evidence = evidence[closest_idx]
        closest_similarity = float(similarities[closest_idx].item())

        weight = 1.0 if req["priority"] == "required" else (
            0.5 if req["priority"] == "preferred" else 1.0
        )
        weighted_total += points * weight
        total_weight += weight
        if req["priority"] == "required":
            required_total += points
            required_weight += 1
            if status in ("Missing", "Transferable Context"):
                # Transferable context is not direct evidence of a required criterion.
                required_without_evidence += 1
        if status == "Full Match":
            fully_matched += 1
        elif status == "Partial Match":
            partial += 1
        elif status == "Transferable Context":
            transferable += 1
        else:
            missing += 1
        # Direct/partial evidence coverage excludes Transferable Context.
        # Transferable passages are counted separately and still contribute
        # only their reduced 25-point value to the compatibility score.
        if status in ("Full Match", "Partial Match"):
            evidence_covered += 1

        alternatives = []
        for index in ranked[1:]:
            alternatives.append({
                "text": evidence[index]["text"],
                "evidence_type": evidence[index]["evidence_type"],
                "similarity": round(float(similarities[index].item()), 3)
            })

        requirement_details.append({
            "requirement": req["requirement"],
            "category": req["category"],
            "criterion_type": req["criterion_type"],
            "priority": req["priority"],
            "priority_source": req["priority_source"],
            "status": status,
            "confidence": round(best_similarity * 100, 1),
            "evidence_strength": evidence_strength,
            "cv_evidence": best_evidence["text"] if status != "Missing" else "",
            "closest_evidence": {
                "text": closest_evidence["text"],
                "evidence_type": closest_evidence.get("evidence_type", "context"),
                "similarity": round(closest_similarity, 3)
            },
            "evidence_note": (
                "A relevant entity and contextual passage were retrieved; review the excerpt to confirm the claim."
                if status in ("Full Match", "Partial Match") else
                "This passage describes related experience only. It does not establish that the candidate has performed this exact responsibility or possesses the exact qualification; manual review is required."
                if status == "Transferable Context" else
                "The closest passage did not contain sufficient supporting evidence. This does not prove the candidate lacks the skill."
            ),
            "alternative_evidence": alternatives
        })

    score = round(weighted_total / total_weight, 2) if total_weight else 0.0
    required_score = round(required_total / required_weight, 2) if required_weight else None
    # Coverage means criteria with direct or partial evidence, not merely
    # a semantically related/transferable passage.
    coverage = round(evidence_covered / len(requirements) * 100, 2)
    explanation = (
        f"Evaluated {len(requirements)} assessable job criteria: {fully_matched} fully supported, "
        f"{partial} partially supported, {transferable} with transferable context only, and "
        f"{missing} without sufficient evidence. Excluded {len(excluded)} employer-benefit "
        "statement(s). Transferable context is a weak indication for human review, not proof "
        "that the candidate has performed the stated responsibility."
    )
    return {
        "score": score,
        "evidence_coverage": coverage,
        "evidence_covered_count": evidence_covered,
        "requirements_count": len(requirements),
        "excluded_information_count": len(excluded),
        "fully_matched": fully_matched,
        "partial_matches": partial,
        "transferable_contexts": transferable,
        "missing": missing,
        "required_without_evidence": required_without_evidence,
        "required_only_score": required_score,
        "required_count": required_weight,
        "match_results": requirement_details,
        "requirements": requirements,
        "excluded_job_information": excluded,
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
