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
    "improved", "deployed", "analysed", "analyzed", "executed", "performed",
    "conducted"
)

NICE_TO_HAVE_PATTERN = re.compile(
    r"nice to have|\bbonus\b|\boptional\b|would be a plus",
    re.IGNORECASE
)
PREFERRED_PATTERN = re.compile(
    r"\b(?:preferred|desirable|advantage|ideally)\b",
    re.IGNORECASE
)
REQUIRED_PATTERN = re.compile(
    r"\b(?:must|required|essential|mandatory|minimum)\b|"
    r"we require|you are required",
    re.IGNORECASE
)

# Keep the same priority model as the Ollama prototype so the two approaches
# can be compared on the same weighted scale.
PRIORITY_WEIGHTS = {
    "required": 3.0,
    "preferred": 1.5,
    "nice_to_have": 1.0,
    "unspecified": 1.5,
}


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
    if NICE_TO_HAVE_PATTERN.search(text):
        return "nice_to_have", "explicit wording"
    if PREFERRED_PATTERN.search(text):
        return "preferred", "explicit wording"
    if REQUIRED_PATTERN.search(text):
        return "required", "explicit wording"

    # Core imperative duties are treated as expected responsibilities, not
    # because a skill keyword appears in the advert, but because the posting
    # directly instructs the holder of the role to perform the task.
    if re.match(
        r"^\s*(?:you will|act as|provide|record|attempt to resolve)\b",
        text,
        re.IGNORECASE,
    ):
        return "required", "imperative responsibility wording"

    if criterion_type == "qualification":
        return "required", "inferred from qualification wording"
    return "unspecified", "not stated in the advert"


def _requirement_group_key(requirement: dict) -> str:
    """Group overlapping duties into a small set of assessable job capabilities."""
    text = " ".join((
        requirement.get("requirement", ""),
        requirement.get("source_text", ""),
    )).casefold()
    category = requirement.get("category", "")

    if category == "language" or re.search(r"\benglish\b|\bspoken and written\b", text):
        return "language"
    if re.search(r"\b(work under pressure|pressure|fast[- ]paced)\b", text):
        return "working_under_pressure"
    if re.search(r"\b(mailbox|ola|slas?|service level agreement)\b", text):
        return "service_level_monitoring"

    # Explicit incident/request logging stays with service-desk incident work,
    # even when the logged request happens to concern a video conference.
    if (
        re.search(r"\b(record|recording|log|logging|tracking|track)\b", text)
        and re.search(r"\b(incidents?|service requests?|tickets?|requests?)\b", text)
    ):
        return "incident_and_troubleshooting"

    if re.search(r"\b(audio[- ]visual|\bav support\b|video conference|video conferencing)\b", text) or (
        "video" in text and "audio" in text
    ):
        return "audiovisual_support"

    # First-contact resolution and room-booking support remain end-user duties.
    if re.search(
        r"\b(end[- ]?users?|point of contact|first[- ]line support|room bookings?|"
        r"users experiencing|user support|communication|first point of contact)\b",
        text,
    ):
        return "end_user_support"

    if re.search(
        r"\b(troubleshoot|incident|hardware|software|service requests?|"
        r"resolve|documentation|support environment)\b",
        text,
    ):
        return "incident_and_troubleshooting"

    return "standalone:" + re.sub(r"\W+", " ", requirement.get("requirement", "").casefold()).strip()

_GROUP_LABELS = {
    "end_user_support": "End-user and first-line support",
    "incident_and_troubleshooting": "Troubleshooting and incident management",
    "audiovisual_support": "Video, audio and audiovisual support",
    "service_level_monitoring": "Mailbox and service-level monitoring",
    "language": "Language proficiency",
    "working_under_pressure": "Working under pressure",
}


def _group_job_requirements(requirements: List[dict]) -> List[dict]:
    """Collapse overlapping job duties while retaining every original statement."""
    groups: Dict[str, List[dict]] = {}
    order: List[str] = []
    for requirement in requirements:
        key = _requirement_group_key(requirement)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(requirement)

    grouped = []
    for key in order:
        members = groups[key]
        priorities = [item.get("priority", "unspecified") for item in members]
        if "required" in priorities:
            priority = "required"
        elif "preferred" in priorities:
            priority = "preferred"
        elif "nice_to_have" in priorities:
            priority = "nice_to_have"
        else:
            priority = "unspecified"

        source_details = []
        entities = []
        seen_entities = set()
        for item in members:
            source_details.append({
                "requirement": item["requirement"],
                "source_text": item["source_text"],
                "category": item["category"],
                "criterion_type": item["criterion_type"],
                "priority": item["priority"],
                "entities": item.get("entities", []),
            })
            for entity in item.get("entities", []):
                entity_key = (entity.get("text", "").casefold(), entity.get("type", ""))
                if entity_key not in seen_entities:
                    entities.append(entity)
                    seen_entities.add(entity_key)

        first = members[0]
        if key.startswith("standalone:"):
            label = first["requirement"]
        else:
            label = _GROUP_LABELS[key]

        priority_sources = {item.get("priority_source", "") for item in members}
        grouped.append({
            "requirement": label,
            "category": first.get("category", "job criterion"),
            "criterion_type": (
                "qualification" if all(
                    item.get("criterion_type") == "qualification" for item in members
                ) else "responsibility"
            ),
            "priority": priority,
            "priority_source": (
                next(iter(priority_sources)) if len(priority_sources) == 1
                else "combined from grouped source statements"
            ),
            "entities": entities,
            "source_text": " | ".join(item["source_text"] for item in members),
            "source_statements": [item["source_text"] for item in members],
            "source_details": source_details,
            "group_key": key,
        })
    return grouped


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
    return _group_job_requirements(requirements), excluded


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
    """Recognize common CV skill-section labels without classifying prose as a list."""
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    if re.search(
        r"^(?:technical )?skills?\s*:?$|^(?:technical )?skills?\s*:",
        lowered,
    ):
        return True
    if re.search(
        r"^(?:technologies|technical skills|core competencies|programming languages)\s*:?$|"
        r"^(?:technologies|technical skills|core competencies|programming languages)\s*:",
        lowered,
    ):
        return True
    return any(label in lowered for label in (
        "tools used:", "languages:", "frameworks", "databases:", "tools:",
        "cloud & databases:", "devops & tools:", "frameworks & libraries:",
    ))


_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "and", "of", "to", "for", "with", "from", "that", "this", "which",
    "who", "where", "when", "while", "although", "because", "is", "are", "was",
    "were", "be", "been", "have", "has", "had", "will", "can", "do", "does",
    "did", "not", "into", "through", "during", "between", "within", "without",
    "a", "i", "in", "my", "our", "their", "they", "we", "you", "it", "its",
    "these", "those", "also", "as", "on", "at", "by", "or", "an", "all",
})
_ENGLISH_DISTINCTIVE_WORDS = frozenset({
    "developed", "implemented", "designed", "created", "managed", "supported",
    "customers", "users", "application", "project", "experience", "responsibilities",
    "education", "university", "skills", "team", "worked", "built", "provided",
})
_CV_DATE_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\.?\s+\d{4}\b|\b(?:19|20)\d{2}\b|\b(?:present|current|now)\b",
    re.IGNORECASE,
)


def _chunk_cv_context(text: str, max_chars: int = 420) -> List[str]:
    """Create exact source lines and overlapping windows retaining CV context.

    Windows keep a role/title close to the organization and dates that follow it.
    This is important for customer-facing roles whose CVs list the title but do
    not include a separate responsibilities paragraph.
    """
    lines = []
    for raw_line in text.replace("\u200b", "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        while len(line) > max_chars:
            split_at = line.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            piece = line[:split_at].strip()
            if piece:
                lines.append(piece)
            line = line[split_at:].strip()
        if line:
            lines.append(line)

    chunks = list(lines)
    start = 0
    while start < len(lines):
        window = []
        end = start
        while end < len(lines):
            candidate = "\n".join(window + [lines[end]])
            if window and len(candidate) > max_chars:
                break
            window.append(lines[end])
            end += 1
        if len(window) > 1:
            chunks.append("\n".join(window))
        if end >= len(lines):
            break
        start = max(start + 1, end - 2)

    unique = []
    seen = set()
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def _looks_like_english_resume(text: str) -> bool:
    """Detect an English-language CV; this supports written English only."""
    words = re.findall(r"\b[a-z]+\b", text.casefold())
    if len(words) < 35:
        return False
    common = [word for word in words if word in _ENGLISH_FUNCTION_WORDS]
    distinctive = {word for word in words if word in _ENGLISH_DISTINCTIVE_WORDS}
    return (
        len(common) >= 6
        and len(common) / len(words) >= 0.04
        and len(distinctive) >= 2
    )


def _find_english_writing_sample(text: str) -> str:
    """Return one exact, readable CV excerpt demonstrating written English."""
    candidates = []
    for chunk in _chunk_cv_context(text):
        words = re.findall(r"\b[a-z]+\b", chunk.casefold())
        common = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
        distinctive = {word for word in words if word in _ENGLISH_DISTINCTIVE_WORDS}
        if len(words) >= 8 and common >= 2 and common / len(words) >= 0.04:
            candidates.append((common, len(words), len(distinctive), len(chunk), chunk))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[4]
    fallback = [
        (len(re.findall(r"\b[a-z]+\b", chunk.casefold())), len(chunk), chunk)
        for chunk in _chunk_cv_context(text)
    ]
    fallback = [item for item in fallback if item[0] >= 8]
    return max(fallback, key=lambda item: (item[0], item[1]))[2] if fallback else ""


def extract_cv_evidence(cv_text: str) -> List[dict]:
    """
    Build source-grounded evidence from sentences and overlapping CV context.
    In contrast to isolated sentence-only extraction, windows preserve role,
    employer, dates, project headings, and the text immediately following them.
    """
    raw_items = []
    for sentence in split_into_sentences(cv_text):
        if _looks_like_cv_heading(sentence):
            evidence_type = "heading_or_title"
        elif _is_skills_list(sentence):
            evidence_type = "skills_list"
        elif any(cue in sentence.casefold() for cue in ACTION_CUES):
            evidence_type = "action_statement"
        else:
            evidence_type = "context"
        raw_items.append((sentence, evidence_type))

    for chunk in _chunk_cv_context(cv_text):
        if len(chunk.strip()) < 18:
            continue
        lowered = chunk.casefold()
        if _is_skills_list(chunk):
            evidence_type = "skills_list"
        elif any(cue in lowered for cue in ACTION_CUES):
            evidence_type = "action_statement"
        elif _CV_DATE_PATTERN.search(chunk) and len(chunk.splitlines()) > 1:
            evidence_type = "experience_context"
        elif _looks_like_cv_heading(chunk):
            evidence_type = "heading_or_title"
        else:
            evidence_type = "context_window"
        raw_items.append((chunk, evidence_type))

    # A CV written in English is a source-grounded writing sample. It can support
    # written use of English, but never spoken proficiency by itself.
    if _looks_like_english_resume(cv_text):
        writing_sample = _find_english_writing_sample(cv_text)
        if writing_sample:
            raw_items.append((writing_sample, "language_context"))

    evidence = []
    by_normalized_text = {}
    for text, evidence_type in raw_items:
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        if not normalized:
            continue

        # The chosen English writing sample often duplicates a sentence/context
        # window already added above. Merge its language marker instead of
        # dropping it during deduplication.
        existing = by_normalized_text.get(normalized)
        if existing is not None:
            if evidence_type == "language_context":
                existing["is_english_writing_sample"] = True
                existing["language_context_label"] = "English-language resume excerpt"
            continue

        entities = _unique_entities(text)
        item = {
            "text": text,
            "quote": text,
            "label": (
                "English-language resume excerpt" if evidence_type == "language_context"
                else "CV context excerpt"
            ),
            "type": evidence_type,
            "evidence_type": evidence_type,
            "entities": [{"text": entity["text"], "type": entity["type"]} for entity in entities],
            "is_english_writing_sample": evidence_type == "language_context",
        }
        if evidence_type == "language_context":
            item["language_context_label"] = "English-language resume excerpt"
        evidence.append(item)
        by_normalized_text[normalized] = item

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

def _evidence_support_level(requirement_text: str, evidence: dict) -> str:
    """Return direct, contextual, or none using source-grounded task evidence.

    Contextual evidence can support a partial match, never a full match by itself.
    The rules mirror the evidence distinctions used by the Ollama prototype.
    """
    requirement = requirement_text.casefold()
    quote = (evidence.get("quote") or evidence.get("text") or "").casefold()
    label = (evidence.get("label") or "").casefold()
    text = f"{label} {quote}"

    def has(patterns) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    if re.search(r"\benglish\b|language proficiency", requirement):
        explicitly_named = bool(re.search(r"\benglish\b", quote))
        explicit_level = has((
            r"\b(native|fluent|proficient|advanced|intermediate|basic)\b",
            r"\b(a[12]|b[12]|c[12])\b",
            r"\b(speak|spoken|speaking|write|written|writing)\b",
            r"\b(level|proficiency)\b",
        ))
        if explicitly_named and explicit_level:
            return "direct"
        if evidence.get("evidence_type") == "language_context" or evidence.get("is_english_writing_sample"):
            return "contextual"
        return "none"

    if "pressure" in requirement or "fast-paced" in requirement:
        return "direct" if has((
            r"\b(under pressure|high[- ]pressure|fast[- ]paced|tight deadlines?)\b",
            r"\b(manag(ed|ing) multiple priorities|multiple competing priorities)\b",
            r"\b(urgent requests|time[- ]critical|high[- ]volume environment)\b",
        )) else "none"

    if (
        "audio visual" in requirement
        or "audiovisual" in requirement
        or "video conference" in requirement
        or "video conferencing" in requirement
        or "video and audio" in requirement
    ):
        av_mention = has((
            r"\b(audio[- ]visual|a/v|av equipment|video conferencing|video conference)\b",
            r"\b(projector|microphone|display setup|sound system|conference room "
            r"(?:av )?(?:equipment|system|setup|hardware))\b",
        ))
        # Keep the activity and AV object close together. A CV paragraph that
        # separately mentions "managed a team" and "conference room" should
        # not become proof of AV support.
        av_action = (
            r"\b(set ?up|configured|programmed|tested|monitored|operated|supported|"
            r"troubleshot|installed|maintained|prepared|provided|coordinated|managed)\b"
        )
        av_object = (
            r"\b(audio[- ]visual|a/v|av equipment|video[- ]conferenc(?:e|ing)s?|"
            r"projectors?|microphones?|speakers?|display setup|sound system|"
            r"conference room (?:av )?(?:equipment|system|setup|hardware))\b"
        )
        av_task = bool(
            re.search(
                rf"(?:{av_action})[^.!?;]{{0,60}}(?:{av_object})|"
                rf"(?:{av_object})[^.!?;]{{0,60}}(?:{av_action})",
                text,
                re.IGNORECASE,
            )
        )
        if av_mention and av_task:
            return "direct"
        if av_mention and evidence.get("evidence_type") in (
            "skills_list", "experience_context", "context_window"
        ):
            return "contextual"
        return "none"

    if "network" in requirement:
        network_mention = has((
            r"\b(networking|computer network|tcp/?ip|dns|dhcp|router|switch|lan|wan)\b",
            r"\b(firewall|vpn|subnet|ethernet|wi[ -]?fi|wireless|osi model|network connectivity)\b",
        ))
        network_action = (
            r"\b(configur(?:e|es|ed|ing)|administer(?:s|ed|ing)|troubleshoot(?:s|ed|ing)?|"
            r"diagnos(?:e|es|ed|ing)|monitor(?:s|ed|ing)?|manag(?:e|es|ed|ing)|"
            r"maintain(?:s|ed|ing)?|install(?:s|ed|ing)?|secur(?:e|es|ed|ing)|"
            r"design(?:s|ed|ing)|deploy(?:s|ed|ing)|set ?up)\b"
        )
        network_object = (
            r"\b(tcp/?ip|dns|dhcp|routers?|switches|lan|wan|firewalls?|vpns?|"
            r"subnets?|ethernet|wi[ -]?fi|wireless|osi model|network connectivity)\b"
        )
        network_task = bool(
            re.search(
                rf"(?:{network_action})[^.!?;]{{0,80}}(?:{network_object})|"
                rf"(?:{network_object})[^.!?;]{{0,80}}(?:{network_action})",
                text,
                re.IGNORECASE,
            )
        )
        if network_task:
            return "direct"
        if network_mention and evidence.get("evidence_type") in (
            "skills_list", "experience_context", "context_window"
        ):
            return "contextual"
        return "none"

    if "mailbox" in requirement or "ola" in requirement or "sla" in requirement:
        service_level_mention = has((
            r"\b(shared )?mailboxes?\b|\b(?:ola|sla)s?\b|"
            r"\bservice level agreements?\b|\bservice levels?\b",
        ))
        service_level_action = (
            r"\b(manag(?:e|es|ed|ing)|monitor(?:s|ed|ing)?|track(?:s|ed|ing)?|"
            r"follow(?:s|ed|ing)?|maintain(?:s|ed|ing)?|review(?:s|ed|ing)?|"
            r"administer(?:s|ed|ing)?|measur(?:e|es|ed|ing))\b"
        )
        service_level_object = (
            r"\b(shared )?mailboxes?\b|\b(?:ola|sla)s?\b|"
            r"\bservice level agreements?\b|\bservice levels?\b"
        )
        # Group each alternative explicitly. Without these non-capturing
        # groups, the regex alternation can match an object mention alone.
        service_level_activity = bool(
            re.search(
                rf"(?:{service_level_action})[^.!?;]{{0,60}}(?:{service_level_object})|"
                rf"(?:{service_level_object})[^.!?;]{{0,60}}(?:{service_level_action})",
                text,
                re.IGNORECASE,
            )
        )
        if service_level_mention and service_level_activity:
            return "direct"
        if service_level_mention and evidence.get("evidence_type") in (
            "skills_list", "experience_context", "context_window"
        ):
            return "contextual"
        return "none"

    incident_logging_present = has((
        r"\b(record(?:ed|ing)?|log(?:ged|ging)?|track(?:ed|ing)?)\b.{0,40}\bincidents?\b",
        r"\b(incident log(?:ging)?|logged incidents?|tracked incidents?|service tickets?|"
        r"support tickets?|ticketing system|service requests?)\b",
    ))

    if any(term in requirement for term in ("incident", "service request", "logging", "tracking")):
        if incident_logging_present:
            if "troubleshoot" in requirement or "support environment" in requirement:
                support_environment = has((
                    r"\b(service desk|help desk|technical support|it support|user support|customer support)\b",
                    r"\b(incident management|support tickets?|support environment|helpdesk)\b",
                ))
                troubleshooting_task = has((
                    r"\b(troubleshoot(?:ing|ed)?|diagnos(?:ed|ing|is)|automated system diagnostics)\b",
                    r"\b(resolved|investigated)\b.{0,60}\b(technical )?(?:issues|incidents|tickets)\b",
                ))
                return "direct" if support_environment and troubleshooting_task else "contextual"
            return "direct"

    if "troubleshoot" in requirement or "support environment" in requirement:
        support_environment = has((
            r"\b(service desk|help desk|technical support|it support|user support|customer support)\b",
            r"\b(incident management|support tickets?|support environment|helpdesk)\b",
        ))
        troubleshooting_task = has((
            r"\b(troubleshoot(?:ing|ed)?|diagnos(?:ed|ing|is)|automated system diagnostics)\b",
            r"\b(resolved|investigated)\b.{0,60}\b(technical )?(?:issues|incidents|tickets)\b",
        ))
        if support_environment and troubleshooting_task:
            # A combined incident/troubleshooting criterion is complete only
            # when the CV also demonstrates incident or ticket logging.
            if "incident" in requirement and not incident_logging_present:
                return "contextual"
            return "direct"
        if support_environment:
            return "contextual"
        if has((
            r"\b(troubleshoot(?:ing|ed)?|diagnos(?:ed|ing|is)|debug(?:ged|ging)?)\b",
            r"\b(fixed (?:software )?bugs?|resolved (?:software )?(?:issues?|bugs?|errors?)|"
            r"investigated (?:software )?(?:issues?|bugs?|errors?))\b",
        )):
            return "contextual"
        return "none"

    if (
        "speak" in requirement
        or "communication" in requirement
        or "communicat" in requirement
        or "end-user" in requirement
        or "end user" in requirement
        or "point of contact" in requirement
        or "first-line support" in requirement
    ):
        # Generic "communication" or "presented" wording is not proof of
        # user-facing support. Require a named audience or a concrete user
        # assistance interaction.
        if has((
            r"\bcommunicat(?:ed|ing)\s+(?:directly\s+)?with\s+(?:end[- ]?users?|users?|customers?|clients?)\b",
            r"\binteract(?:ed|ing)\s+with\s+(?:end[- ]?users?|users?|customers?|clients?)\b",
            r"\b(taught|teaching|advised|assisted|supported|helped|served)\s+"
            r"(?:students|customers|clients|users|end[- ]?users)\b",
            r"\b(handled|answered|resolved|processed)\s+(?:customer|client|user|end[- ]?user)\s+"
            r"(?:questions|inquiries|calls|requests|issues|tickets)\b",
            r"\bprovided\s+(?:technical\s+)?(?:support|assistance|guidance)\s+to\s+"
            r"(?:users|customers|clients|end[- ]?users)\b",
        )):
            return "direct"
        if has((
            r"\b(business )?sales advisor\b",
            r"\bsales (?:representative|consultant|associate)\b",
            r"\bbrand ambassador\b",
            r"\bcustomer service (?:advisor|representative|agent|associate)\b",
            r"\b(client advisor|account manager|account executive)\b",
            r"\b(teacher|tutor|instructor|trainer|lecturer|educator)\b",
            r"\bretail associate\b",
            r"\b(service desk|help desk|helpdesk) analyst\b",
            r"\bit support (?:analyst|specialist|assistant|technician)\b",
            r"\btechnical support (?:analyst|specialist|assistant|technician)\b",
        )):
            return "contextual"
        # Generic technical support is direct only when the excerpt describes
        # the support activity, not merely a role title containing "support".
        if has((
            r"\b(technical support|it support|service desk|help desk|helpdesk)\b",
        )) and has((
            r"\b(assisted|supported|resolved|handled|responded to|helped)\b",
        )):
            return "direct"
        return "none"

    return "none"


def _assess_evidence(
    requirement: dict,
    evidence: dict,
    similarity: float,
    entity_embeddings=None,
    entity_indices=None,
) -> Tuple[str, str, float]:
    """Assess evidence with direct/contextual rules before generic semantic similarity."""
    evidence_type = evidence.get("evidence_type", "context")
    if evidence_type == "heading_or_title":
        return "Missing", "title_only_not_evidence", 0.0

    support_level = _evidence_support_level(requirement.get("requirement", ""), evidence)
    if support_level == "contextual":
        # Customer-facing roles, software debugging, and English CV text are
        # useful transferable context, not proof of the exact job criterion.
        return "Partial Match", "contextual_transferable_evidence", 50.0

    if support_level == "direct":
        quote = (evidence.get("quote") or evidence.get("text") or "").casefold()
        requirement_text = requirement.get("requirement", "").casefold()
        # Grouped labels (e.g. "Language proficiency") can hide qualifiers
        # from the original job post (e.g. "spoken and written"). Preserve and
        # inspect those source statements when applying criterion-specific rules.
        original_requirement_text = " ".join(
            [requirement_text]
            + list(requirement.get("source_statements", []))
            + [
                f"{item.get('requirement', '')} {item.get('source_text', '')}"
                for item in requirement.get("source_details", [])
            ]
        ).casefold()

        if "english" in original_requirement_text or "language proficiency" in requirement_text:
            speaks = bool(re.search(r"\b(spoken|speaking)\b", quote))
            writes = bool(re.search(r"\b(written|writing)\b", quote))
            asks_both = (
                re.search(r"\bspoken\b", original_requirement_text)
                and re.search(r"\bwritten\b", original_requirement_text)
            )
            if asks_both and not (speaks and writes):
                return "Partial Match", "explicit_english_level_but_modes_not_both_proven", 60.0
            if asks_both and speaks and writes:
                return "Full Match", "explicit_spoken_and_written_english_evidence", 80.0

        if evidence_type in ("skills_list", "experience_context", "context_window"):
            return "Partial Match", "explicit_but_limited_or_context_only_evidence", 50.0
        return "Full Match", "direct_source_grounded_task_evidence", 80.0

    if requirement.get("criterion_type") == "responsibility" and evidence_type == "skills_list":
        return "Missing", "skills_list_does_not_demonstrate_responsibility", 0.0

    # Generic entity alignment is used only for criteria without a specialized
    # evidence rule. Similarity retrieves passages; it does not establish a match.
    exact_overlap = _entity_overlap(requirement, evidence)
    semantic_overlap = _semantic_entity_overlap(
        requirement, evidence, entity_embeddings, entity_indices
    )
    entity_support = max(exact_overlap, semantic_overlap)
    if (
        entity_support >= 0.68
        and similarity >= 0.42
        and evidence_type in ("action_statement", "skills_list")
    ):
        if evidence_type == "skills_list":
            return "Partial Match", "named_skill_list_without_proficiency_depth", 50.0
        return "Partial Match", "semantically_aligned_action_evidence", 50.0

    return "Missing", "retrieved_passage_lacks_sufficient_support", 0.0


def _calculate_weighted_match_score(requirement_details: List[dict]) -> float:
    """Use the same priority weights as the Ollama prototype."""
    weighted_total = 0.0
    total_weight = 0.0
    for item in requirement_details:
        priority = item.get("priority", "unspecified")
        weight = PRIORITY_WEIGHTS.get(priority, PRIORITY_WEIGHTS["unspecified"])
        weighted_total += float(item.get("score", 0) or 0) * weight
        total_weight += weight
    return round(weighted_total / total_weight, 2) if total_weight else 0.0

def calculate_hybrid_match_score(job_description: str, cv_text: str) -> dict:
    """Group related job duties, retrieve CV evidence, and score with Ollama-compatible weights."""
    if not cv_text or not cv_text.strip():
        return {
            "score": 0.0,
            "match_results": [],
            "requirements": [],
            "excluded_job_information": [],
            "cv_evidence": [],
            "explanation": "Empty CV text.",
        }

    requirements, excluded = extract_job_criteria(job_description)
    evidence = extract_cv_evidence(cv_text)
    if not requirements:
        return {
            "score": 0.0,
            "match_results": [],
            "requirements": [],
            "excluded_job_information": excluded,
            "cv_evidence": evidence,
            "explanation": "No assessable job criteria could be extracted.",
        }
    if not evidence:
        return {
            "score": 0.0,
            "match_results": [],
            "requirements": requirements,
            "excluded_job_information": excluded,
            "cv_evidence": [],
            "explanation": "No usable CV text was extracted.",
        }

    evidence_texts = [item["text"] for item in evidence]
    evidence_embeddings = embedding_model.encode(
        evidence_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
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
            entity_terms,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if entity_terms else None
    )

    requirement_details = []
    required_total = 0.0
    required_count = 0
    fully_matched = partial = missing = required_without_evidence = 0
    evidence_covered = 0

    for req in requirements:
        source_details = req.get("source_details") or [{
            "requirement": req["requirement"],
            "source_text": req.get("source_text", req["requirement"]),
            "category": req.get("category", "job criterion"),
            "criterion_type": req.get("criterion_type", "job criterion"),
            "priority": req.get("priority", "unspecified"),
            "entities": req.get("entities", []),
        }]
        source_texts = [
            item.get("requirement") or item.get("source_text") or req["requirement"]
            for item in source_details
        ]
        source_embeddings = embedding_model.encode(
            source_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity_matrix = util.cos_sim(source_embeddings, evidence_embeddings)
        best_similarity_per_evidence = similarity_matrix.max(dim=0).values
        # Evidence rules must inspect the canonical grouped capability, not a
        # concatenation of every original sentence. Mixing unrelated phrases
        # (e.g. "video conference" in an incident-logging sentence) can cause
        # an earlier keyword rule to mask the actual capability being assessed.
        grouped_assessment_requirement = dict(req)
        grouped_assessment_requirement["requirement"] = req.get("requirement", "")

        candidate_indices = [
            index for index, item in enumerate(evidence)
            if item.get("evidence_type") != "heading_or_title"
        ]
        if not candidate_indices:
            status, strength, points = "Missing", "no_substantive_cv_passage", 0.0
            best_idx = None
            best_similarity = 0.0
            best_evidence = None
            closest_evidence = None
            ranked = []
        else:
            assessed = {}
            for candidate_idx in candidate_indices:
                candidate_similarity = float(best_similarity_per_evidence[candidate_idx].item())
                assessed[candidate_idx] = _assess_evidence(
                    grouped_assessment_requirement,
                    evidence[candidate_idx],
                    candidate_similarity,
                    entity_embeddings,
                    entity_indices,
                )

            status_rank = {"Full Match": 2, "Partial Match": 1, "Missing": 0}
            ranked = sorted(
                candidate_indices,
                key=lambda index: (
                    status_rank[assessed[index][0]],
                    float(best_similarity_per_evidence[index].item()),
                ),
                reverse=True,
            )
            best_idx = ranked[0]
            best_similarity = float(best_similarity_per_evidence[best_idx].item())
            best_evidence = evidence[best_idx]
            status, strength, points = assessed[best_idx]

            closest_idx = max(
                candidate_indices,
                key=lambda index: float(best_similarity_per_evidence[index].item()),
            )
            closest_evidence = evidence[closest_idx]

        if req.get("priority") == "required":
            required_total += points
            required_count += 1
            if status == "Missing":
                required_without_evidence += 1

        if status == "Full Match":
            fully_matched += 1
        elif status == "Partial Match":
            partial += 1
        else:
            missing += 1

        if status in ("Full Match", "Partial Match"):
            evidence_covered += 1

        alternatives = []
        for index in ranked[1:4]:
            alternatives.append({
                "text": evidence[index]["text"],
                "evidence_type": evidence[index].get("evidence_type", "context"),
                "similarity": round(float(best_similarity_per_evidence[index].item()), 3),
            })

        if status == "Full Match":
            note = "Direct, source-grounded evidence supports this criterion. Review the excerpt to confirm its scope."
        elif strength == "contextual_transferable_evidence":
            note = (
                "This is transferable context, not direct proof of the exact job task. "
                "For example, customer-facing work supports communication, while an English CV supports written English only."
            )
        elif strength == "explicit_english_level_but_modes_not_both_proven":
            note = "The CV states an English level, but does not separately establish both spoken and written proficiency."
        elif status == "Partial Match":
            note = "The CV supports part of this criterion, but does not establish the full requested depth or responsibility."
        else:
            note = "No sufficiently relevant source-grounded evidence was found. This does not prove the candidate lacks the skill."

        closest_payload = None
        if closest_evidence is not None:
            closest_idx = max(
                candidate_indices,
                key=lambda index: float(best_similarity_per_evidence[index].item()),
            )
            closest_payload = {
                "text": closest_evidence["text"],
                "evidence_type": closest_evidence.get("evidence_type", "context"),
                "similarity": round(float(best_similarity_per_evidence[closest_idx].item()), 3),
            }

        requirement_details.append({
            "requirement": req["requirement"],
            "source_statements": req.get("source_statements", [req["source_text"]]),
            "category": req.get("category", "job criterion"),
            "criterion_type": req.get("criterion_type", "job criterion"),
            "priority": req.get("priority", "unspecified"),
            "priority_source": req.get("priority_source", "not stated"),
            "status": status,
            "score": points,
            "confidence": round(best_similarity * 100, 1),
            "evidence_strength": strength,
            "cv_evidence": best_evidence["text"] if best_evidence is not None and status != "Missing" else "",
            "closest_evidence": closest_payload,
            "evidence_note": note,
            "alternative_evidence": alternatives,
        })

    # Preserve the Ollama-compatible 0–80 weighted points for comparison,
    # but also expose a conventional 0–100 percentage. A Full Match is 80
    # points, so displaying the raw value with a '%' sign understates the
    # normalized compatibility and makes a perfect result look like 80%.
    score = _calculate_weighted_match_score(requirement_details)
    compatibility_percent = round(score / 80.0 * 100.0, 2)
    required_score = round(required_total / required_count, 2) if required_count else None
    required_only_percent = (
        round(required_score / 80.0 * 100.0, 2)
        if required_score is not None else None
    )
    coverage = round(evidence_covered / len(requirements) * 100, 2)
    source_statement_count = sum(len(item.get("source_statements", [])) for item in requirements)

    explanation = (
        f"Evaluated {len(requirements)} grouped job criteria from {source_statement_count} source statements: "
        f"{fully_matched} fully matched, {partial} partially matched, and {missing} without sufficient evidence. "
        "Related duties are grouped to avoid counting several descriptions of the same capability as separate requirements. "
        "Partial matches can include transferable evidence, which must be reviewed by a human."
    )
    return {
        # "score" remains the historical Ollama-compatible point value (0–80).
        "score": score,
        "ollama_compatible_score": score,
        "score_scale_max": 80.0,
        "compatibility_percent": compatibility_percent,
        "evidence_coverage": coverage,
        "evidence_covered_count": evidence_covered,
        "requirements_count": len(requirements),
        "source_statement_count": source_statement_count,
        "excluded_information_count": len(excluded),
        "fully_matched": fully_matched,
        "partial_matches": partial,
        "missing": missing,
        "required_without_evidence": required_without_evidence,
        "required_only_score": required_score,
        "required_only_percent": required_only_percent,
        "required_count": required_count,
        "match_results": requirement_details,
        "requirements": requirements,
        "excluded_job_information": excluded,
        "cv_evidence": evidence,
        "explanation": explanation,
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


def _extract_phone_number(text: str) -> str:
    """Extract Romanian mobile numbers even when digits are separated by spaces."""
    pattern = re.compile(
        r"(?<!\d)(?:\+40|0040)?[\s().-]*0?[237](?:[\s().-]*\d){8}(?!\d)"
    )
    match = pattern.search(text)
    if not match:
        return "Unspecified"

    raw = match.group(0).strip()
    compact = re.sub(r"[^\d+]", "", raw)
    if compact.startswith("0040"):
        return "+40" + compact[4:]
    if compact.startswith("+40"):
        return compact
    digits = compact
    if digits.startswith("40") and len(digits) == 11:
        return "+" + digits
    if len(digits) == 9 and digits[0] in "237":
        return "0" + digits
    return digits


def _extract_candidate_name(lines: List[str]) -> str:
    """Choose a plausible name from the opening CV lines, skipping headers/contact lines."""
    ignored_exact = {
        "resume", "curriculum vitae", "cv", "personal details", "contact",
        "contact details", "profile", "about me",
    }
    for line in lines[:8]:
        candidate = re.sub(r"\s+", " ", line).strip(" |•·")
        lowered = candidate.casefold()
        if not candidate or lowered in ignored_exact:
            continue
        if re.search(r"@|https?://|www\.|linkedin|github|portfolio", lowered):
            continue
        if re.search(r"\+?\d[\d\s().-]{7,}", candidate):
            continue
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*", candidate)
        if 2 <= len(words) <= 4 and not re.search(r"[:/]", candidate):
            return candidate[:255]
    return lines[0][:255] if lines else "Unknown Candidate"


def extract_candidate_info(cv_text: str) -> dict:
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", cv_text)
    lines = [line.strip() for line in cv_text.split("\n") if line.strip()]
    return {
        "full_name": _extract_candidate_name(lines),
        "email": email_match.group(0).casefold() if email_match else "Unspecified",
        "phone": _extract_phone_number(cv_text),
    }
