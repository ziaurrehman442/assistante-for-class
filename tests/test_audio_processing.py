import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import LOCAL_ORIGIN_PATTERN
from app.services.audio_processing import process_audio_file
from app.services.note_generator import NoteGenerationError


class TestConfig:
    TESTING = True
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS = 300
    OLLAMA_AVAILABILITY_TIMEOUT_SECONDS = 5
    HUGGINGFACE_TOKEN = "test-token"
    SPEAKER_MATCH_THRESHOLD = 0.78
    ALLOWED_ORIGINS = [LOCAL_ORIGIN_PATTERN]
    SPEAKER_PROFILES_PATH = Path(tempfile.gettempdir()) / "unused-speaker-profiles.json"


class AudioProcessingTests(unittest.TestCase):
    def test_no_speech_returns_notes_instead_of_processing_error(self):
        app = create_app(TestConfig)
        expected_message = "No speech was detected in this recording."

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "silent.wav"
            audio_path.touch()
            with patch("app.services.audio_processing.transcribe_audio", return_value=[]):
                result = process_audio_file(app, audio_path)

        self.assertEqual(result["transcript"], expected_message)
        self.assertEqual(result["dialogue"], [])
        self.assertEqual(result["speakers"], [])
        self.assertEqual(result["detailed_class_notes"]["main_topics"], [expected_message])
        self.assertEqual(
            result["detailed_class_notes"]["teacher_explanations"],
            [expected_message],
        )
        self.assertEqual(result["easy_notes"], [expected_message])
        self.assertEqual(result["revision_quiz"], {
            "mcqs": [],
            "conceptual_questions": [],
        })
        self.assertEqual(result["warnings"], [])

    def test_transcript_is_kept_when_study_material_generation_fails(self):
        """A finished transcript must survive an Ollama failure.

        Transcription costs the recording and CPU minutes, so a dead or missing
        Ollama has to degrade the notes section, not discard the whole job.
        """
        app = create_app(TestConfig)
        dialogue = [
            {
                "speaker_id": "a",
                "speaker_name": "Teacher",
                "overlapping_speakers": [],
                "start": 0.0,
                "end": 1.0,
                "text": "Recursion needs a base case.",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "lesson.wav"
            audio_path.touch()
            with patch(
                "app.services.audio_processing.transcribe_audio",
                return_value=[{"start": 0.0, "end": 1.0, "text": " Recursion"}],
            ), patch(
                "app.services.audio_processing.diarize_and_label",
                return_value=(dialogue, []),
            ), patch(
                "app.services.audio_processing.generate_study_materials",
                side_effect=NoteGenerationError(
                    "Ollama model qwen2.5:1.5b is not loaded. "
                    "Run 'ollama pull qwen2.5:1.5b' in your terminal."
                ),
            ):
                result = process_audio_file(app, audio_path)

        self.assertIn("Recursion needs a base case.", result["transcript"])
        self.assertEqual(result["revision_quiz"]["mcqs"], [])
        self.assertEqual(result["detailed_class_notes"]["definitions"], [])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("ollama pull qwen2.5:1.5b", result["warnings"][0])

    def test_healthy_run_reports_no_warnings(self):
        app = create_app(TestConfig)
        dialogue = [
            {
                "speaker_id": "a",
                "speaker_name": "Teacher",
                "overlapping_speakers": [],
                "start": 0.0,
                "end": 1.0,
                "text": "Fine.",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "lesson.wav"
            audio_path.touch()
            with patch(
                "app.services.audio_processing.transcribe_audio",
                return_value=[{"start": 0.0, "end": 1.0, "text": " Fine"}],
            ), patch(
                "app.services.audio_processing.diarize_and_label",
                return_value=(dialogue, []),
            ), patch(
                "app.services.audio_processing.generate_study_materials",
                return_value={"easy_notes": ["Point"]},
            ):
                result = process_audio_file(app, audio_path)

        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["easy_notes"], ["Point"])


if __name__ == "__main__":
    unittest.main()
