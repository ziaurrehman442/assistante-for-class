import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import LOCAL_ORIGIN_PATTERN
from app.services.audio_processing import process_audio_file


class TestConfig:
    TESTING = True
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS = 300
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


if __name__ == "__main__":
    unittest.main()
