import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (
    _category,
    _assess_evidence,
    _criterion_type,
    _explicit_technical_entity_matches,
    _group_job_requirements,
    _job_exclusion_reason,
    _normalize_technical_entity,
    _priority,
    extract_job_criteria,
)


class JavaJobRequirementExtractionTests(unittest.TestCase):
    def test_java_programming_language_entity_is_not_a_spoken_language_requirement(self):
        text = "Strong experience with Java and Spring-based application development."
        entities = [
            {"text": "Java", "type": "language"},
            {"text": "Spring", "type": "software"},
        ]

        category = _category(text, entities)

        self.assertEqual(category, "technical skill")
        requirement = {
            "requirement": text,
            "source_text": text,
            "category": category,
            "criterion_type": _criterion_type(text, category),
            "priority": "required",
            "priority_source": "inferred from qualification wording",
            "entities": entities,
        }
        self.assertTrue(_group_job_requirements([requirement])[0]["group_key"].startswith("standalone:"))

    def test_english_spoken_and_written_requirement_is_still_a_language_criterion(self):
        text = "Strong written and verbal English communication skills."
        entities = [{"text": "English", "type": "language"}]

        self.assertEqual(_category(text, entities), "language")

    def test_non_english_spoken_language_with_cefr_level_is_a_language_criterion(self):
        text = "German C1 proficiency is required."
        entities = [{"text": "German", "type": "language"}]

        self.assertEqual(_category(text, entities), "language")

    def test_job_ad_headings_are_excluded(self):
        headings = [
            "Qualifications",
            "Experience:",
            "Ideal Candidate Profiles, Talents, and Desired Qualifications:",
            "Here are the core functions, responsibilities, and expectations for this role:",
        ]

        for heading in headings:
            with self.subTest(heading=heading):
                self.assertIsNotNone(_job_exclusion_reason(heading))

    def test_compensation_paragraphs_are_excluded_from_candidate_scoring(self):
        statements = [
            "The targeted compensation package for this role may vary based upon internal equity and years of experience.",
            "We may make further adjustments through an approval process if the targeted compensation range needs to be modified based on business needs and market trends.",
        ]

        for statement in statements:
            with self.subTest(statement=statement):
                self.assertIsNotNone(_job_exclusion_reason(statement))

    def test_role_intro_and_remote_metadata_are_not_scored(self):
        self.assertIsNotNone(
            _job_exclusion_reason(
                "We are looking for a talented Software Engineer (Java) to join our growing global team at Sectigo."
            )
        )
        self.assertIsNotNone(
            _job_exclusion_reason(
                "This is a full-time and remote position, with ideal candidates based in Romania or surrounding countries with proper work authorizations."
            )
        )

    def test_strongly_recommended_degree_is_preferred_not_required(self):
        text = (
            "Bachelor's degree in Computer Science, Engineering, Mathematics, "
            "or a related field, or equivalent practical experience is strongly recommended."
        )
        self.assertEqual(_priority(text, "qualification")[0], "preferred")

    def test_restful_apis_normalize_to_rest_api(self):
        self.assertEqual(_normalize_technical_entity("REST APIs"), "restapi")
        self.assertEqual(_normalize_technical_entity("RESTful API"), "restapi")
        self.assertEqual(_normalize_technical_entity("RESTful APIs"), "restapi")
        self.assertNotEqual(_normalize_technical_entity("Java"), _normalize_technical_entity("JavaScript"))

    def test_existing_employer_benefits_remain_excluded(self):
        benefits = [
            "The company provides retail vouchers and health insurance.",
            "Employees can receive tuition fee reimbursement.",
        ]

        for statement in benefits:
            with self.subTest(statement=statement):
                self.assertIsNotNone(_job_exclusion_reason(statement))

    def test_build_and_contribute_responsibilities_are_required(self):
        self.assertEqual(
            _priority("Build and enhance RESTful APIs.", "responsibility")[0],
            "required",
        )
        self.assertEqual(
            _priority("The engineer will contribute to platform maintenance.", "responsibility")[0],
            "required",
        )

    def test_restful_api_action_evidence_matches_rest_api_requirement(self):
        requirement = {
            "requirement": "Experience building and consuming REST APIs.",
            "category": "technical skill",
            "criterion_type": "qualification",
            "entities": [{"text": "REST APIs", "type": "technical skill"}],
        }
        evidence = {
            "label": "API development responsibility",
            "quote": "Implemented and maintained RESTful APIs for enterprise applications.",
            "text": "Implemented and maintained RESTful APIs for enterprise applications.",
            "evidence_type": "action_statement",
        }

        matched, required = _explicit_technical_entity_matches(requirement, evidence)
        self.assertEqual(matched, ["REST APIs"])
        self.assertEqual(required, ["REST APIs"])

        status, reason, points = _assess_evidence(requirement, evidence, 0.20)
        self.assertEqual(status, "Full Match")
        self.assertEqual(reason, "explicit_named_technical_skills_in_action_evidence")
        self.assertEqual(points, 80.0)

    def test_job1_language_group_contains_only_spoken_language_requirement(self):
        job_text = "\n\n".join([
            "We are looking for a talented Software Engineer (Java) to join our growing global team at Sectigo.",
            "The Software Engineer (Java) will contribute to the design, development, and maintenance of Sectigo Certificate Manager.",
            "Build and enhance RESTful APIs and distributed services using Java and Spring technologies.",
            "Strong experience with Java and Spring-based application development.",
            "Strong written and verbal English communication skills.",
            "The targeted compensation package for this role may vary based upon internal equity and years of experience.",
            "Qualifications",
            "Experience:",
            "Ideal Candidate Profiles, Talents, and Desired Qualifications:",
        ])

        def fake_entities(sentence):
            if "Java" in sentence:
                return [{"text": "Java", "type": "language"}]
            if "English" in sentence:
                return [{"text": "English", "type": "language"}]
            return []

        with patch("services.nlp_service._unique_entities", side_effect=fake_entities):
            requirements, excluded = extract_job_criteria(job_text)

        language_groups = [item for item in requirements if item["group_key"] == "language"]
        self.assertEqual(len(language_groups), 1)
        self.assertEqual(
            language_groups[0]["source_statements"],
            ["Strong written and verbal English communication skills."],
        )
        self.assertGreaterEqual(len(excluded), 5)
        requirement_texts = [item["requirement"] for item in requirements]
        self.assertTrue(any("Java" in item for item in requirement_texts))
        self.assertFalse(any(item == "Qualifications" for item in requirement_texts))
        self.assertFalse(any(item == "Experience:" for item in requirement_texts))


if __name__ == "__main__":
    unittest.main()
