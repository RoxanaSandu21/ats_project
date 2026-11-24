import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.nlp_service import (
    PRIORITY_WEIGHTS,
    _assess_evidence,
    _calculate_weighted_match_score,
    _evidence_support_level,
    _group_job_requirements,
    _priority,
    calculate_hybrid_match_score,
    extract_candidate_info,
    extract_cv_evidence,
    _is_skills_list,
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

        source_by_group = {
            item["group_key"]: item["source_statements"]
            for item in grouped
        }
        self.assertTrue(any(
            "Record all incidents" in statement
            for statement in source_by_group["incident_and_troubleshooting"]
        ))
        self.assertFalse(any(
            "Record all incidents" in statement
            for statement in source_by_group["audiovisual_support"]
        ))

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

    def test_grouped_compatibility_matches_ollama_reference_case(self):
        import torch

        raw_requirements = [
            make_requirement(
                "Act as an initial point of contact for all end-user support requirements;"
            ),
            make_requirement(
                "You will troubleshoot hardware and software problems, manage service requests, and maintain comprehensive documentation."
            ),
            make_requirement("Provide video and audio 1st line support services"),
            make_requirement("Monitor mailbox following OLA's and SLA's."),
            make_requirement(
                "Excellent understanding of English – spoken and written;",
                category="language",
                criterion_type="qualification",
            ),
            make_requirement(
                "Ability to work under pressure;",
                category="soft skill",
                criterion_type="qualification",
            ),
        ]
        grouped = _group_job_requirements(raw_requirements)
        evidence = [
            {
                "text": "Business Sales Advisor February 2025 – September 2025",
                "quote": "Business Sales Advisor February 2025 – September 2025",
                "label": "CV context excerpt",
                "type": "experience_context",
                "evidence_type": "experience_context",
                "entities": [],
            },
            {
                "text": (
                    "Developed a mobile application that helps users monitor and improve "
                    "their basic needs through real-time feedback."
                ),
                "quote": (
                    "Developed a mobile application that helps users monitor and improve "
                    "their basic needs through real-time feedback."
                ),
                "label": "English-language resume excerpt",
                "type": "language_context",
                "evidence_type": "language_context",
                "is_english_writing_sample": True,
                "entities": [],
            },
        ]

        class FixedEmbeddingModel:
            def encode(self, texts, **kwargs):
                return torch.ones((len(texts), 8), dtype=torch.float32)

        with patch(
            "services.nlp_service.extract_job_criteria",
            return_value=(grouped, []),
        ):
            with patch(
                "services.nlp_service.extract_cv_evidence",
                return_value=evidence,
            ):
                with patch(
                    "services.nlp_service.embedding_model",
                    FixedEmbeddingModel(),
                ):
                    result = calculate_hybrid_match_score(
                        "IT Support with English",
                        "A source-grounded English resume with a Business Sales Advisor role.",
                    )

        self.assertEqual(result["requirements_count"], 6)
        self.assertEqual(result["source_statement_count"], 6)
        self.assertEqual(result["partial_matches"], 2)
        self.assertEqual(result["fully_matched"], 0)
        self.assertEqual(result["missing"], 4)
        self.assertEqual(result["evidence_covered_count"], 2)
        self.assertEqual(result["required_count"], 5)
        self.assertEqual(result["required_only_score"], 20.0)
        self.assertEqual(result["required_only_percent"], 25.0)
        self.assertEqual(result["score"], 18.18)
        self.assertEqual(result["ollama_compatible_score"], 18.18)
        self.assertEqual(result["score_scale_max"], 80.0)
        self.assertEqual(result["compatibility_percent"], 22.73)

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

    def test_explicit_spoken_and_written_english_can_be_full_evidence(self):
        evidence = {
            "label": "Languages",
            "quote": "English — Advanced; spoken and written",
            "text": "English — Advanced; spoken and written",
            "evidence_type": "skills_list",
        }
        requirement = "Excellent understanding of English spoken and written"
        self.assertEqual(_evidence_support_level(requirement, evidence), "direct")

        status, reason, points = _assess_evidence(
            {"requirement": requirement, "criterion_type": "qualification"},
            evidence,
            0.85,
        )
        self.assertEqual(status, "Full Match")
        self.assertEqual(reason, "explicit_spoken_and_written_english_evidence")
        self.assertEqual(points, 80.0)

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

    def test_incident_logging_alone_is_partial_for_grouped_troubleshooting(self):
        requirement = (
            "Troubleshooting and incident management. You will troubleshoot hardware "
            "and software problems. Incident logging and tracking."
        )
        evidence = {
            "label": "Support role responsibility",
            "quote": "Recorded all user incidents in the service desk ticketing system.",
            "text": "Recorded all user incidents in the service desk ticketing system.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(_evidence_support_level(requirement, evidence), "contextual")

        direct_evidence = {
            "label": "IT Service Desk | Responsibility",
            "quote": (
                "Worked on the IT service desk, diagnosed hardware and software issues, "
                "and recorded incidents in the support ticketing system."
            ),
            "text": (
                "Worked on the IT service desk, diagnosed hardware and software issues, "
                "and recorded incidents in the support ticketing system."
            ),
            "evidence_type": "action_statement",
        }
        self.assertEqual(_evidence_support_level(requirement, direct_evidence), "direct")

        # A service-desk troubleshooting passage without incident logging must
        # not receive a full match for the combined criterion.
        troubleshooting_only = {
            "label": "Service Desk Analyst | Responsibility",
            "quote": (
                "Service Desk Analyst, SCC. Executed automated system diagnostics "
                "and OS performance troubleshooting using PowerShell scripts."
            ),
            "text": (
                "Service Desk Analyst, SCC. Executed automated system diagnostics "
                "and OS performance troubleshooting using PowerShell scripts."
            ),
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level(requirement, troubleshooting_only),
            "contextual",
        )

    def test_maria_style_service_desk_evidence_is_retained_with_role_context(self):
        source = (
            "Experience\n"
            "Service Desk Analyst, SCC – Iași, Romania June 2026 – Present\n"
            "- Executed automated system diagnostics and OS performance troubleshooting "
            "using PowerShell scripts, minimizing ticket resolution times across enterprise endpoints"
        )
        with patch("services.nlp_service._unique_entities", return_value=[]):
            evidence = extract_cv_evidence(source)

        combined = [
            item for item in evidence
            if "Service Desk Analyst" in item.get("quote", "")
            and "troubleshooting" in item.get("quote", "").casefold()
        ]
        self.assertTrue(
            combined,
            "The service-desk role heading must stay attached to its troubleshooting responsibility.",
        )
        item = combined[0]
        self.assertEqual(item["evidence_type"], "action_statement")
        self.assertEqual(
            _evidence_support_level("Troubleshooting and incident management", item),
            "contextual",
        )
        self.assertEqual(
            _evidence_support_level("End-user and first-line support", item),
            "contextual",
        )

        for requirement in (
            "Troubleshooting and incident management",
            "End-user and first-line support",
        ):
            status, _, points = _assess_evidence(
                {"requirement": requirement, "criterion_type": "responsibility"},
                item,
                0.80,
            )
            self.assertEqual(status, "Partial Match")
            self.assertEqual(points, 50.0)

    def test_video_conference_source_text_cannot_mask_troubleshooting_evidence(self):
        import torch

        raw_requirements = [
            make_requirement(
                "You will troubleshoot hardware and software problems, manage service requests, and maintain comprehensive documentation."
            ),
            make_requirement(
                "Record all incidents and Video conference requests within the SCC service desk tool;"
            ),
            make_requirement("Troubleshooting experience in a support environment;"),
            make_requirement("Provide video and audio 1st line support services"),
        ]
        grouped = _group_job_requirements(raw_requirements)
        evidence = [{
            "text": (
                "Service Desk Analyst, SCC. Executed automated system diagnostics "
                "and OS performance troubleshooting using PowerShell scripts."
            ),
            "quote": (
                "Service Desk Analyst, SCC. Executed automated system diagnostics "
                "and OS performance troubleshooting using PowerShell scripts."
            ),
            "label": "CV context excerpt",
            "type": "action_statement",
            "evidence_type": "action_statement",
            "entities": [],
        }]

        class FixedEmbeddingModel:
            def encode(self, texts, **kwargs):
                return torch.ones((len(texts), 8), dtype=torch.float32)

        with patch(
            "services.nlp_service.extract_job_criteria",
            return_value=(grouped, []),
        ):
            with patch(
                "services.nlp_service.extract_cv_evidence",
                return_value=evidence,
            ):
                with patch(
                    "services.nlp_service.embedding_model",
                    FixedEmbeddingModel(),
                ):
                    result = calculate_hybrid_match_score(
                        "IT Support with English",
                        "Service Desk Analyst with troubleshooting experience.",
                    )

        troubleshooting = next(
            item for item in result["match_results"]
            if item["requirement"] == "Troubleshooting and incident management"
        )
        self.assertEqual(troubleshooting["status"], "Partial Match")
        self.assertEqual(troubleshooting["score"], 50.0)
        self.assertEqual(
            troubleshooting["evidence_strength"],
            "contextual_transferable_evidence",
        )

    def test_english_marker_survives_deduplication_with_context_windows(self):
        writing_sample = (
            "Developed a mobile application that helps users monitor and improve "
            "their basic needs through real-time feedback."
        )
        with patch("services.nlp_service._unique_entities", return_value=[]):
            with patch("services.nlp_service._looks_like_english_resume", return_value=True):
                with patch(
                    "services.nlp_service._find_english_writing_sample",
                    return_value=writing_sample,
                ):
                    evidence = extract_cv_evidence(writing_sample)

        matching = [
            item for item in evidence
            if item.get("text") == writing_sample
        ]
        self.assertEqual(len(matching), 1)
        self.assertTrue(matching[0].get("is_english_writing_sample"))
        self.assertEqual(_evidence_support_level(
            "Excellent understanding of English spoken and written",
            matching[0],
        ), "contextual")

    def test_role_date_context_window_keeps_sales_role_together(self):
        source = (
            "Business Sales Advisor\n"
            "Vodafone\n"
            "February 2025 – September 2025\n"
            "Mathematics and Computer Science Teacher\n"
            "Matematica.md\n"
            "December 2021 – June 2024"
        )
        with patch("services.nlp_service._unique_entities", return_value=[]):
            evidence = extract_cv_evidence(source)

        role_context = [
            item for item in evidence
            if item.get("evidence_type") == "experience_context"
            and "Business Sales Advisor" in item.get("quote", "")
        ]
        self.assertTrue(role_context)
        self.assertTrue(any("February 2025" in item["quote"] for item in role_context))

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

    def test_social_networking_project_does_not_prove_computer_networking(self):
        evidence = {
            "label": "Project responsibility",
            "quote": "Built a social networking application for students.",
            "text": "Built a social networking application for students.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Basic networking skills", evidence),
            "none",
        )

    def test_network_configuration_task_is_direct_evidence(self):
        evidence = {
            "label": "IT operations responsibility",
            "quote": "Configured routers, switches and DNS records for office LAN connectivity.",
            "text": "Configured routers, switches and DNS records for office LAN connectivity.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Basic networking skills", evidence),
            "direct",
        )

    def test_networking_skill_list_is_contextual_not_direct_task_evidence(self):
        evidence = {
            "label": "Technical Skills",
            "quote": "Networking, TCP/IP, DNS, DHCP",
            "text": "Networking, TCP/IP, DNS, DHCP",
            "evidence_type": "skills_list",
        }
        self.assertEqual(
            _evidence_support_level("Basic networking skills", evidence),
            "contextual",
        )

    def test_mailbox_or_sla_mention_without_monitoring_is_not_direct_evidence(self):
        evidence = {
            "label": "Tools and responsibilities",
            "quote": "Shared mailbox, OLA, SLA",
            "text": "Shared mailbox, OLA, SLA",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Mailbox and service-level monitoring", evidence),
            "none",
        )

    def test_explicit_mailbox_and_sla_monitoring_is_direct_evidence(self):
        evidence = {
            "label": "Service Desk responsibility",
            "quote": "Monitored the shared mailbox and tracked SLA performance.",
            "text": "Monitored the shared mailbox and tracked SLA performance.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Mailbox and service-level monitoring", evidence),
            "direct",
        )

    def test_unrelated_management_action_does_not_validate_distant_mailbox_mention(self):
        evidence = {
            "label": "Experience context",
            "quote": (
                "Managed software delivery schedules and coordinated engineering "
                "planning across several teams for multiple quarterly releases. "
                "The role also used a shared mailbox for announcements."
            ),
            "text": (
                "Managed software delivery schedules and coordinated engineering "
                "planning across several teams for multiple quarterly releases. "
                "The role also used a shared mailbox for announcements."
            ),
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Mailbox and service-level monitoring", evidence),
            "none",
        )

    def test_unrelated_management_action_does_not_validate_distant_av_mention(self):
        evidence = {
            "label": "Project experience",
            "quote": (
                "Managed engineering schedules and coordinated several releases "
                "across distributed product teams during the year. "
                "The office had a video conference room."
            ),
            "text": (
                "Managed engineering schedules and coordinated several releases "
                "across distributed product teams during the year. "
                "The office had a video conference room."
            ),
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Video, audio and audiovisual support", evidence),
            "none",
        )

    def test_video_conference_mention_without_av_activity_is_not_direct(self):
        evidence = {
            "label": "Project context",
            "quote": "Attended a video conference in the main conference room.",
            "text": "Attended a video conference in the main conference room.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Video, audio and audiovisual support", evidence),
            "none",
        )

    def test_configuring_and_testing_av_equipment_is_direct_evidence(self):
        evidence = {
            "label": "AV support responsibility",
            "quote": "Set up video-conferencing equipment and tested microphones and projectors.",
            "text": "Set up video-conferencing equipment and tested microphones and projectors.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("Video, audio and audiovisual support", evidence),
            "direct",
        )

    def test_generic_communication_feature_does_not_prove_end_user_support(self):
        evidence = {
            "label": "Software project",
            "quote": "Created a communication feature that allows users to send messages.",
            "text": "Created a communication feature that allows users to send messages.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("End-user and first-line support", evidence),
            "none",
        )

    def test_explicit_communication_with_end_users_is_direct_evidence(self):
        evidence = {
            "label": "Support responsibility",
            "quote": "Communicated with end users and resolved their access requests.",
            "text": "Communicated with end users and resolved their access requests.",
            "evidence_type": "action_statement",
        }
        self.assertEqual(
            _evidence_support_level("End-user and first-line support", evidence),
            "direct",
        )

    def test_phone_extraction_accepts_spaced_romanian_mobile_numbers(self):
        info = extract_candidate_info(
            "RESUME\nMaria Andriescu\n"
            "maria@example.com | +40 735 473 598 | Iași, Romania"
        )
        self.assertEqual(info["full_name"], "Maria Andriescu")
        self.assertEqual(info["email"], "maria@example.com")
        self.assertEqual(info["phone"], "+40735473598")

    def test_phone_extraction_keeps_local_romanian_format(self):
        info = extract_candidate_info("John Doe\n0732 164 275")
        self.assertEqual(info["full_name"], "John Doe")
        self.assertEqual(info["phone"], "0732164275")

    def test_skills_list_detection_recognizes_technologies_heading(self):
        self.assertTrue(_is_skills_list("Technologies"))
        self.assertTrue(_is_skills_list("Core Competencies:"))
        self.assertTrue(_is_skills_list("Technical Skills: Python, Java"))
        self.assertFalse(_is_skills_list("I used various technologies in this project."))

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
