import unittest

from services.nlp_service import (
    PRIORITY_WEIGHTS,
    _assess_evidence,
    _calculate_weighted_match_score,
    _evidence_support_level,
    _group_job_requirements,
    _priority,
)


def make_requirement(text, category="job criterion", criterion_type="responsibility"):
    priority, priority_source = _priority(text, criterion_type)
    return {
        "requirement": text,
        "source_text": text,
        "category": category,
        "criterion_type": criterion_type,
        "priority": priority,
        "priority_source": priority_source,
        "entities": [],
    }


class OllamaParityEvidenceTests(unittest.TestCase):
    """Regression coverage for the GLiNER evidence rules aligned with Ollama."""

    def test_it_support_posting_groups_fourteen_statements_into_six_capabilities(self):
        statements = [
            make_requirement(
                "The IT Support Specialist provides technical support and assistance to end-users experiencing IT-related issues, ensuring effective communication and resolution."
            ),
            make_requirement(
                "You will troubleshoot hardware and software problems, manage service requests, and maintain comprehensive documentation to enhance overall IT service efficiency."
            ),
            make_requirement(
                "Act as an initial point of contact for all end-user support requirements;"
            ),
            make_requirement(
                "Act as first-line support for any users experiencing issues with room bookings;"
            ),
            make_requirement("Provide video and audio 1st line support services"),
            make_requirement("Incident logging and tracking;"),
            make_requirement(
                "Attempt to resolve as many incidents as possible at the first point of contact offering advice and guidance;"
            ),
            make_requirement(
                "Provide day-to-day Video Conference session support, including programming call and call monitoring, assist in testing links to authorized external companies, monitor usage and availability of equipment in rooms;"
            ),
            make_requirement(
                "Record all incidents and Video conference requests within the SCC service desk tool;"
            ),
            make_requirement("Monitor mailbox following OLA's and SLA's."),
            make_requirement(
                "Excellent understanding of English – spoken and written;",
                category="language",
                criterion_type="qualification",
            ),
            make_requirement(
                "Troubleshooting experience in a support environment;",
                criterion_type="qualification",
            ),
            make_requirement(
                "Ability to work under pressure;",
                category="soft skill",
                criterion_type="qualification",
            ),
            make_requirement(
                "Basic Audio Visual knowledge preferred.",
                criterion_type="qualification",
            ),
        ]

        grouped = _group_job_requirements(statements)

        self.assertEqual(len(grouped), 6)
        self.assertEqual(
            {item["group_key"] for item in grouped},
            {
                "end_user_support",
                "incident_and_troubleshooting",
                "audiovisual_support",
                "service_level_monitoring",
                "language",
                "working_under_pressure",
            },
        )
        self.assertEqual(sum(len(item["source_statements"]) for item in grouped), 14)
        self.assertTrue(all(item["source_details"] for item in grouped))

        priority_by_group = {item["group_key"]: item["priority"] for item in grouped}
        self.assertEqual(priority_by_group["end_user_support"], "required")
        self.assertEqual(priority_by_group["incident_and_troubleshooting"], "required")
        self.assertEqual(priority_by_group["audiovisual_support"], "required")
        self.assertEqual(priority_by_group["service_level_monitoring"], "unspecified")
        self.assertEqual(priority_by_group["language"], "required")
        self.assertEqual(priority_by_group["working_under_pressure"], "required")

    def test_priority_weights_match_ollama_prototype(self):
        self.assertEqual(PRIORITY_WEIGHTS["required"], 3.0)
        self.assertEqual(PRIORITY_WEIGHTS["preferred"], 1.5)
        self.assertEqual(PRIORITY_WEIGHTS["nice_to_have"], 1.0)
        self.assertEqual(PRIORITY_WEIGHTS["unspecified"], 1.5)

    def test_two_contextual_partials_reproduce_ollama_reference_score(self):
        details = [
            {"score": 50, "priority": "required"},
            {"score": 50, "priority": "required"},
            {"score": 0, "priority": "required"},
            {"score": 0, "priority": "required"},
            {"score": 0, "priority": "required"},
            {"score": 0, "priority": "unspecified"},
        ]
        self.assertEqual(_calculate_weighted_match_score(details), 18.18)

    def test_sales_advisor_is_contextual_for_end_user_communication(self):
        evidence = {
            "label": "Business Sales Advisor — Vodafone",
            "quote": "Business Sales Advisor",
            "text": "Business Sales Advisor",
            "evidence_type": "experience_context",
        }
        requirement = (
            "End-user and first-line support. Act as an initial point of contact "
            "for all end-user support requirements."
        )
        self.assertEqual(_evidence_support_level(requirement, evidence), "contextual")

        status, reason, points = _assess_evidence(
            {"requirement": requirement, "criterion_type": "responsibility"},
            evidence,
            0.20,
        )
        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "contextual_transferable_evidence")
        self.assertEqual(points, 50.0)

    def test_english_resume_is_contextual_written_language_evidence_only(self):
        evidence = {
            "label": "English-language resume excerpt",
            "quote": (
                "Developed a mobile application that helps users monitor and improve "
                "their basic needs through real-time feedback."
            ),
            "text": (
                "Developed a mobile application that helps users monitor and improve "
                "their basic needs through real-time feedback."
            ),
            "evidence_type": "language_context",
            "is_english_writing_sample": True,
        }
        requirement = "Excellent understanding of English spoken and written"
        self.assertEqual(_evidence_support_level(requirement, evidence), "contextual")

        status, reason, points = _assess_evidence(
            {"requirement": requirement, "criterion_type": "qualification"},
            evidence,
            0.10,
        )
        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "contextual_transferable_evidence")
        self.assertEqual(points, 50.0)

    def test_fluent_english_alone_does_not_prove_spoken_and_written_requirement(self):
        evidence = {
            "label": "English",
            "quote": "English — Fluent",
            "text": "English — Fluent",
            "evidence_type": "context",
        }
        requirement = "Excellent understanding of English spoken and written"
        self.assertEqual(_evidence_support_level(requirement, evidence), "direct")

        status, reason, points = _assess_evidence(
            {"requirement": requirement, "criterion_type": "qualification"},
            evidence,
            0.90,
        )
        self.assertEqual(status, "Partial Match")
        self.assertEqual(reason, "explicit_english_level_but_modes_not_both_proven")
        self.assertEqual(points, 60.0)

    def test_software_debugging_is_contextual_not_direct_service_desk_evidence(self):
        evidence = {
            "label": "Junior Java Developer | Responsibility",
            "quote": "Debugged Java application defects and fixed software bugs.",
            "text": "Debugged Java application defects and fixed software bugs.",
            "evidence_type": "action_statement",
        }
        requirement = "Troubleshooting experience in a support environment"
        self.assertEqual(_evidence_support_level(requirement, evidence), "contextual")

        support_role = {
            "label": "IT Support Assistant | Responsibility",
            "quote": "Worked on the IT help desk and resolved user support tickets.",
            "text": "Worked on the IT help desk and resolved user support tickets.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(_evidence_support_level(requirement, support_role), "direct")

    def test_role_title_does_not_prove_working_under_pressure(self):
        evidence = {
            "label": "Software Engineer Intern — Continental Automotive",
            "quote": "Software Engineer Intern July 2023 – September 2023",
            "text": "Software Engineer Intern July 2023 – September 2023",
            "evidence_type": "experience_context",
        }
        self.assertEqual(
            _evidence_support_level("Ability to work under pressure", evidence),
            "none",
        )

    def test_github_tools_do_not_prove_networking(self):
        evidence = {
            "label": "Tools",
            "quote": "Git, GitHub, GitLab",
            "text": "Git, GitHub, GitLab",
            "evidence_type": "skills_list",
        }
        self.assertEqual(
            _evidence_support_level("Basic networking skills", evidence),
            "none",
        )

    def test_unrelated_software_project_does_not_prove_av_support(self):
        evidence = {
            "label": "iNeed | Project",
            "quote": "Developed a mobile app with real-time feedback and mini-games.",
            "text": "Developed a mobile app with real-time feedback and mini-games.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Basic Audio Visual knowledge preferred", evidence),
            "none",
        )


if __name__ == "__main__":
    unittest.main()
