import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (
    _assess_evidence,
    _explicit_technical_entity_matches,
)


class JavaTechnicalEntityScoringTests(unittest.TestCase):
    def test_generic_software_word_is_not_treated_as_a_named_technology(self):
        requirement = {
            "requirement": "software-related knowledge",
            "category": "technical skill",
            "criterion_type": "qualification",
            "entities": [{"text": "software", "type": "software"}],
        }
        evidence = {
            "quote": "Master's in Software Development and Business Information Systems",
            "text": "Master's in Software Development and Business Information Systems",
            "evidence_type": "context_window",
        }

        matched, required = _explicit_technical_entity_matches(requirement, evidence)

        self.assertEqual(matched, [])
        self.assertEqual(required, [])

    def test_grouped_id11_troubleshooting_does_not_match_software_degree_text(self):
        requirement = {
            "requirement": "Troubleshooting and incident management",
            "category": "technical skill",
            "criterion_type": "responsibility",
            "group_key": "incident_and_troubleshooting",
            "entities": [
                {"text": "software", "type": "software"},
                {"text": "SCC service desk tool", "type": "tool"},
            ],
        }
        evidence = {
            "label": "Education",
            "quote": "Master's in Software Development and Business Information Systems",
            "text": "Master's in Software Development and Business Information Systems",
            "evidence_type": "context_window",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.90)

        self.assertEqual(status, "Missing")
        self.assertEqual(reason, "retrieved_passage_lacks_sufficient_support")
        self.assertEqual(points, 0.0)

    def test_grouped_id11_still_accepts_contextual_software_debugging(self):
        requirement = {
            "requirement": "Troubleshooting and incident management",
            "category": "technical skill",
            "criterion_type": "responsibility",
            "group_key": "incident_and_troubleshooting",
            "entities": [{"text": "SCC service desk tool", "type": "tool"}],
        }
        evidence = {
            "label": "Project responsibility",
            "quote": "Performed troubleshooting of application bugs during software development.",
            "text": "Performed troubleshooting of application bugs during software development.",
            "evidence_type": "action_statement",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.45)

        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "contextual_transferable_evidence")
        self.assertEqual(points, 50.0)

    def test_java_in_skills_list_remains_partial_without_embedding_cutoff(self):
        requirement = {
            "requirement": "Java development knowledge",
            "category": "technical skill",
            "criterion_type": "qualification",
            "entities": [{"text": "Java", "type": "language"}],
        }
        evidence = {
            "label": "Languages",
            "quote": "Languages: C++, C, Java, JavaScript, TypeScript, Python",
            "text": "Languages: C++, C, Java, JavaScript, TypeScript, Python",
            "evidence_type": "skills_list",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.18)

        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "named_skill_list_without_proficiency_depth")
        self.assertEqual(points, 50.0)

    def test_java_does_not_match_javascript(self):
        requirement = {
            "requirement": "Java development knowledge",
            "category": "technical skill",
            "criterion_type": "qualification",
            "entities": [{"text": "Java", "type": "language"}],
        }
        evidence = {
            "label": "Languages",
            "quote": "Languages: JavaScript, TypeScript",
            "text": "Languages: JavaScript, TypeScript",
            "evidence_type": "skills_list",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.95)

        self.assertEqual(status, "Missing")
        self.assertEqual(reason, "no_exact_named_technical_entity_evidence")
        self.assertEqual(points, 0.0)

    def test_spring_boot_and_springboot_are_equivalent(self):
        requirement = {
            "requirement": "Spring Boot experience",
            "category": "technical skill",
            "criterion_type": "qualification",
            "entities": [{"text": "Spring Boot", "type": "software"}],
        }
        evidence = {
            "label": "Frameworks",
            "quote": "Frameworks and libraries: SpringBoot, React.js, Node.js",
            "text": "Frameworks and libraries: SpringBoot, React.js, Node.js",
            "evidence_type": "skills_list",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.21)

        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "named_skill_list_without_proficiency_depth")
        self.assertEqual(points, 50.0)

    def test_project_action_with_all_named_technologies_can_be_full(self):
        requirement = {
            "requirement": "Develop applications using Java and Spring Boot",
            "category": "technical skill",
            "criterion_type": "responsibility",
            "entities": [
                {"text": "Java", "type": "language"},
                {"text": "Spring Boot", "type": "software"},
            ],
        }
        evidence = {
            "label": "Interview Management project",
            "quote": "Built a web application using Java 21 and Spring Boot with JPA and REST API.",
            "text": "Built a web application using Java 21 and Spring Boot with JPA and REST API.",
            "evidence_type": "action_statement",
        }

        status, reason, points = _assess_evidence(requirement, evidence, 0.28)

        self.assertEqual(status, "Full Match")
        self.assertEqual(reason, "explicit_named_technical_skills_in_action_evidence")
        self.assertEqual(points, 80.0)


if __name__ == "__main__":
    unittest.main()
