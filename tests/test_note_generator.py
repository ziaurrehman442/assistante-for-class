import json
import unittest
from unittest.mock import patch

from app.services.note_generator import (
    NoteGenerationError,
    format_dialogue,
    generate_study_materials,
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

    @patch("app.services.note_generator.requests.post")
    def test_rejects_incomplete_study_materials(self, post):
        post.return_value.json.return_value = {"response": json.dumps({"easy_notes": []})}

        with self.assertRaises(NoteGenerationError):
            generate_study_materials(
                "Lesson transcript", "http://localhost:11434", "qwen2.5:3b", 300
            )


if __name__ == "__main__":
    unittest.main()
