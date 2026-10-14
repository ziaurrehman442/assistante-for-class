import json
import unittest

import requests
from unittest.mock import Mock, patch

from app.services.note_generator import (
    NoteGenerationError,
    chunk_transcript,
    merge_materials,
    _validate_materials,
    check_ollama_available,
    format_dialogue,
    generate_study_materials,
)


def tags_response(*names):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {"models": [{"name": name} for name in names]}
    ).encode("utf-8")
    return response


def available_ollama(*names):
    # new= is used so the decorator injects no argument into the test method.
    return patch(
        "app.services.note_generator.requests.get",
        new=Mock(return_value=tags_response(*(names or ("qwen2.5:3b",)))),
    )


def sample_materials():
    return {
        "detailed_class_notes": {
            "main_topics": ["Topic"],
            "definitions": ["Term: meaning"],
            "teacher_explanations": ["Explanation"],
            "qa": [{"question": "Why?", "answer": "Because."}],
        },
        "easy_notes": ["Remember this point."],
        "revision_quiz": {
            "mcqs": [
                {
                    "question": f"Question {number}?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                }
                for number in range(5)
            ],
            "conceptual_questions": ["Why?", "How?", "What if?"],
        },
    }


class NoteGeneratorTests(unittest.TestCase):
    def test_formats_speaker_dialogue_and_interruptions(self):
        self.assertEqual(
            format_dialogue(
                [
                    {
                        "start": 65.8,
                        "speaker_name": "Teacher",
                        "overlapping_speakers": ["Student 1"],
                        "text": " Welcome back.",
                    }
                ]
            ),
            "[01:05] Teacher (overlapping with Student 1): Welcome back.",
        )

    @available_ollama()
    @patch("app.services.note_generator.requests.post")
    def test_generates_structured_study_materials(self, post):
        post.return_value.json.return_value = {
            "response": json.dumps(sample_materials())
        }

        result = generate_study_materials(
            "Lesson transcript", "http://localhost:11434", "qwen2.5:3b", 300
        )

        self.assertEqual(result, sample_materials())
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/generate")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen2.5:3b")
        prompt = post.call_args.kwargs["json"]["prompt"]
        self.assertIn("code-switch", prompt)
        self.assertIn("Urdu and English", prompt)
        self.assertIn("Do not translate the transcript", prompt)

    @available_ollama()
    @patch("app.services.note_generator.requests.post")
    def test_rejects_incomplete_study_materials(self, post):
        post.return_value.json.return_value = {"response": json.dumps({"easy_notes": []})}

        with self.assertRaises(NoteGenerationError):
            generate_study_materials(
                "Lesson transcript", "http://localhost:11434", "qwen2.5:3b", 300
            )

    @available_ollama()
    @patch("app.services.note_generator.requests.post")
    def test_retries_a_malformed_response_then_succeeds(self, post):
        # A small model following a strict schema is not reliable first time.
        post.return_value.json.side_effect = [
            {"response": json.dumps({"easy_notes": []})},
            {"response": "not json at all"},
            {"response": json.dumps(sample_materials())},
        ]

        materials = generate_study_materials(
            "Lesson transcript", "http://localhost:11434", "qwen2.5:3b", 300
        )

        self.assertEqual(materials["revision_quiz"]["mcqs"][0]["question"], "Question 0?")
        self.assertEqual(post.call_count, 3)

    @available_ollama()
    @patch("app.services.note_generator.requests.post")
    def test_gives_up_after_the_configured_attempts(self, post):
        post.return_value.json.return_value = {"response": json.dumps({"easy_notes": []})}

        with self.assertRaises(NoteGenerationError):
            generate_study_materials(
                "Lesson transcript",
                "http://localhost:11434",
                "qwen2.5:3b",
                300,
                max_attempts=2,
            )

        self.assertEqual(post.call_count, 2)

    @available_ollama()
    @patch("app.services.note_generator.requests.post")
    def test_transport_errors_are_not_retried(self, post):
        post.side_effect = requests.RequestException("connection refused")

        with self.assertRaises(NoteGenerationError):
            generate_study_materials(
                "Lesson transcript",
                "http://localhost:11434",
                "qwen2.5:3b",
                300,
                max_attempts=3,
            )

        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()


class OllamaAvailabilityTests(unittest.TestCase):
    def test_reports_unreachable_ollama_with_a_startup_hint(self):
        with patch(
            "app.services.note_generator.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            with self.assertRaises(NoteGenerationError) as caught:
                check_ollama_available("http://localhost:11434", "qwen2.5:1.5b", 5)

        self.assertIn("not reachable", str(caught.exception))
        self.assertIn("ollama serve", str(caught.exception))

    def test_missing_model_names_the_pull_command(self):
        with patch(
            "app.services.note_generator.requests.get",
            return_value=tags_response("qwen2.5:3b"),
        ):
            with self.assertRaises(NoteGenerationError) as caught:
                check_ollama_available("http://localhost:11434", "qwen2.5:1.5b", 5)

        self.assertEqual(
            str(caught.exception),
            "Ollama model qwen2.5:1.5b is not loaded. "
            "Run 'ollama pull qwen2.5:1.5b' in your terminal.",
        )

    def test_unexpected_tags_payload_is_reported(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"unexpected": true}'
        with patch("app.services.note_generator.requests.get", return_value=response):
            with self.assertRaises(NoteGenerationError):
                check_ollama_available("http://localhost:11434", "qwen2.5:3b", 5)

    def test_installed_model_passes_the_check(self):
        with patch(
            "app.services.note_generator.requests.get",
            return_value=tags_response("qwen2.5:3b", "llama3:8b"),
        ):
            check_ollama_available("http://localhost:11434", "qwen2.5:3b", 5)

    @available_ollama("qwen2.5:3b")
    @patch("app.services.note_generator.requests.post")
    def test_timeout_raises_a_timeout_specific_error(self, post):
        post.side_effect = requests.Timeout("read timed out")

        with self.assertRaises(NoteGenerationError) as caught:
            generate_study_materials("t", "http://localhost:11434", "qwen2.5:3b", 900)

        self.assertIn("900 seconds", str(caught.exception))
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", str(caught.exception))

    @available_ollama("qwen2.5:3b")
    @patch("app.services.note_generator.requests.post")
    def test_model_not_found_during_generation_is_reported(self, post):
        response = requests.Response()
        response.status_code = 404
        response._content = b'{"error":"model \'qwen2.5:3b\' not found"}'
        post.side_effect = requests.HTTPError(response=response)

        with self.assertRaises(NoteGenerationError) as caught:
            generate_study_materials("t", "http://localhost:11434", "qwen2.5:3b", 900)

        self.assertIn("ollama pull qwen2.5:3b", str(caught.exception))

    @available_ollama("qwen2.5:3b")
    @patch("app.services.note_generator.requests.post")
    def test_generation_does_not_reach_ollama_when_the_model_is_absent(self, post):
        with patch(
            "app.services.note_generator.requests.get",
            return_value=tags_response("llama3:8b"),
        ):
            with self.assertRaises(NoteGenerationError):
                generate_study_materials(
                    "t", "http://localhost:11434", "qwen2.5:1.5b", 900
                )

        post.assert_not_called()


class ChunkingTests(unittest.TestCase):
    def test_short_transcript_is_a_single_chunk(self):
        self.assertEqual(chunk_transcript("one line", 6000), ["one line"])

    def test_empty_transcript_yields_no_chunks(self):
        self.assertEqual(chunk_transcript("   ", 6000), [])

    def test_split_preserves_every_word(self):
        lines = [
            f"[00:{i // 60:02d}:{i % 60:02d}] Teacher 1: line {i} " + "word " * 30
            for i in range(60)
        ]
        text = "\n".join(lines)

        chunks = chunk_transcript(text, 6000)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 6000 + 200 for chunk in chunks))
        self.assertEqual(
            " ".join(chunks).split(), text.split(), "no transcript text may be lost"
        )

    def test_split_never_cuts_a_speaker_turn_in_half(self):
        text = "\n".join(f"turn {index}" for index in range(200))

        for chunk in chunk_transcript(text, 200):
            for line in chunk.splitlines():
                self.assertTrue(line.startswith("turn "), line)

    def test_merged_materials_keep_the_client_schema(self):
        chunk_materials = {
            "detailed_class_notes": {
                "main_topics": ["Topic"],
                "definitions": ["Term: meaning"],
                "teacher_explanations": ["Explained"],
                "qa": [{"question": "Q?", "answer": "A."}],
            },
            "easy_notes": ["Recall this"],
            "revision_quiz": {
                "mcqs": [
                    {"question": f"Q{n}?", "options": ["A", "B", "C", "D"], "answer": "A"}
                    for n in range(5)
                ],
                "conceptual_questions": ["a?", "b?", "c?"],
            },
        }

        merged = merge_materials([chunk_materials, chunk_materials], max_mcqs=5)

        self.assertTrue(_validate_materials(merged))
        self.assertEqual(len(merged["revision_quiz"]["mcqs"]), 5)
        self.assertEqual(len(merged["revision_quiz"]["conceptual_questions"]), 3)
        self.assertEqual(len(merged["detailed_class_notes"]["definitions"]), 2)
