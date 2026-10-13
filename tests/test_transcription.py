import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.transcription import diarize_and_label, transcribe_audio


class Annotation:
    def __init__(self, turns, labels=()):
        self.turns = turns
        self._labels = labels

    def labels(self):
        return list(self._labels)

    def itertracks(self, yield_label=False):
        return iter((turn, None, speaker) for turn, speaker in self.turns)


class TranscriptionTests(unittest.TestCase):
    @patch("app.services.transcription._get_model")
    def test_transcription_uses_auto_language_and_urdu_english_prompt(self, get_model):
        word = SimpleNamespace(start=0.0, end=0.5, word=" Hello")
        get_model.return_value.transcribe.return_value = (
            [SimpleNamespace(words=[word], text=" Hello", start=0.0, end=0.5)],
            None,
        )

        words = transcribe_audio(Path("lesson.wav"))

        self.assertEqual(words, [{"start": 0.0, "end": 0.5, "text": " Hello"}])
        options = get_model.return_value.transcribe.call_args.kwargs
        self.assertIsNone(options["language"])
        self.assertEqual(options["task"], "transcribe")
        self.assertIn("Urdu", options["initial_prompt"])
        self.assertIn("English", options["initial_prompt"])
        self.assertIn("without translating", options["initial_prompt"])

    @patch("app.services.transcription._get_model")
    def test_transcription_enables_vad_and_wide_beam_search(self, get_model):
        get_model.return_value.transcribe.return_value = ([], None)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WHISPER_BEAM_SIZE", None)
            os.environ.pop("WHISPER_MIN_SILENCE_MS", None)
            transcribe_audio(Path("lesson.wav"))

        options = get_model.return_value.transcribe.call_args.kwargs
        self.assertTrue(options["vad_filter"])
        self.assertEqual(options["vad_parameters"]["min_silence_duration_ms"], 500)
        self.assertEqual(options["beam_size"], 5)

    @patch("app.services.transcription._get_model")
    def test_beam_size_and_silence_window_come_from_the_environment(self, get_model):
        get_model.return_value.transcribe.return_value = ([], None)

        with patch.dict(
            os.environ,
            {"WHISPER_BEAM_SIZE": "8", "WHISPER_MIN_SILENCE_MS": "900"},
        ):
            transcribe_audio(Path("lesson.wav"))

        options = get_model.return_value.transcribe.call_args.kwargs
        self.assertEqual(options["beam_size"], 8)
        self.assertEqual(options["vad_parameters"]["min_silence_duration_ms"], 900)

    @patch("app.services.transcription._get_model")
    def test_invalid_accuracy_settings_fall_back_to_the_defaults(self, get_model):
        get_model.return_value.transcribe.return_value = ([], None)

        with patch.dict(
            os.environ,
            {"WHISPER_BEAM_SIZE": "wide", "WHISPER_MIN_SILENCE_MS": "half-a-second"},
        ):
            transcribe_audio(Path("lesson.wav"))

        options = get_model.return_value.transcribe.call_args.kwargs
        self.assertEqual(options["beam_size"], 5)
        self.assertEqual(options["vad_parameters"]["min_silence_duration_ms"], 500)

    @patch(
        "app.services.transcription._assign_speaker_profiles",
        return_value={
            "A": {"id": "teacher-id", "display_name": "Teacher", "is_named": True},
            "B": {"id": "student-id", "display_name": "Student 1", "is_named": True},
        },
    )
    @patch("app.services.transcription._get_diarization_pipeline")
    def test_diarization_aligns_turns_and_marks_interruption(
        self, pipeline, _assign_profiles
    ):
        primary = Annotation(
            [
                (SimpleNamespace(start=0.0, end=0.5), "A"),
                (SimpleNamespace(start=0.5, end=1.0), "B"),
            ]
        )
        overlapping = Annotation(
            [
                (SimpleNamespace(start=0.0, end=0.75), "A"),
                (SimpleNamespace(start=0.5, end=0.8), "B"),
            ],
            labels=["A", "B"],
        )
        pipeline.return_value.return_value = SimpleNamespace(
            exclusive_speaker_diarization=primary,
            speaker_diarization=overlapping,
            speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        with tempfile.TemporaryDirectory() as directory:
            dialogue, speakers = diarize_and_label(
                Path(directory) / "lesson.wav",
                [
                    {"start": 0.1, "end": 0.2, "text": " Hello"},
                    {"start": 0.55, "end": 0.65, "text": " Wait"},
                ],
                token="test-token",
                profiles_path=Path(directory) / "profiles.json",
            )

        self.assertEqual([turn["speaker_name"] for turn in dialogue], ["Teacher", "Student 1"])
        self.assertEqual(dialogue[1]["overlapping_speakers"], ["Teacher"])
        self.assertEqual([speaker["id"] for speaker in speakers], ["teacher-id", "student-id"])


if __name__ == "__main__":
    unittest.main()
