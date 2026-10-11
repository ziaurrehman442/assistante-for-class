import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import LOCAL_ORIGIN_PATTERN
from app.services.speaker_profiles import load_profiles, save_profiles
from app.services.transcription import (
    SpeakerDiarizationError,
    _assign_speaker_profiles,
    diarize_and_label,
)


class TestConfig:
    TESTING = True
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS = 300
    HUGGINGFACE_TOKEN = "test-token"
    SPEAKER_MATCH_THRESHOLD = 0.78
    PROCESS_AUDIO_RATE_LIMIT = 10
    PROCESS_AUDIO_RATE_WINDOW_SECONDS = 60
    ALLOWED_ORIGINS = [LOCAL_ORIGIN_PATTERN, "https://notes.example.com"]


class AudioApiTests(unittest.TestCase):
    def setUp(self):
        self.upload_dir = tempfile.TemporaryDirectory()
        TestConfig.UPLOAD_FOLDER = Path(self.upload_dir.name)
        TestConfig.SPEAKER_PROFILES_PATH = Path(self.upload_dir.name) / "speaker_profiles.json"
        self.client = create_app(TestConfig).test_client()

    def tearDown(self):
        self.upload_dir.cleanup()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"status": "ok", "audio_processing": "async-jobs"},
        )

    def test_cors_allows_configured_origins_only(self):
        local_response = self.client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"}
        )
        configured_response = self.client.get(
            "/api/health", headers={"Origin": "https://notes.example.com"}
        )
        remote_response = self.client.get(
            "/api/health", headers={"Origin": "https://example.com"}
        )
        self.assertEqual(
            local_response.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:5173",
        )
        self.assertEqual(
            configured_response.headers.get("Access-Control-Allow-Origin"),
            "https://notes.example.com",
        )
        self.assertIsNone(remote_response.headers.get("Access-Control-Allow-Origin"))

    def test_process_audio_requires_file(self):
        response = self.client.post("/api/process-audio")
        self.assertEqual(response.status_code, 400)

    def test_process_audio_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/process-audio",
            data={"file": (io.BytesIO(b"not audio"), "lecture.txt")},
        )
        self.assertEqual(response.status_code, 400)

    def test_process_audio_rejects_remote_origins(self):
        response = self.client.post(
            "/api/process-audio",
            headers={"Origin": "https://example.com"},
            data={"file": (io.BytesIO(b"audio bytes"), "lesson.wav")},
        )
        self.assertEqual(response.status_code, 403)

    @patch("app.routes.audio.consume_request", return_value=30)
    def test_process_audio_rate_limit_returns_retry_after(self, _consume_request):
        response = self.client.post(
            "/api/process-audio",
            data={"file": (io.BytesIO(b"audio bytes"), "lesson.wav")},
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "30")

    @patch("app.routes.audio.enqueue_audio_job", return_value="job-123")
    def test_process_audio_queues_upload_and_returns_job_location(self, enqueue):
        response = self.client.post(
            "/api/process-audio",
            data={"file": (io.BytesIO(b"audio bytes"), "lesson.wav")},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"job_id": "job-123", "status": "queued"})
        self.assertEqual(response.headers["Location"], "/api/process-audio/jobs/job-123")
        enqueue.assert_called_once()
        self.assertTrue(enqueue.call_args.args[1].exists())

    @patch(
        "app.routes.audio.get_audio_job",
        return_value={
            "status": "completed",
            "result": {"transcript": "lesson", "easy_notes": []},
            "error": None,
            "status_code": None,
        },
    )
    def test_audio_job_status_returns_completed_result(self, _get_job):
        response = self.client.get("/api/process-audio/jobs/job-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "status": "completed",
                "result": {"transcript": "lesson", "easy_notes": []},
            },
        )

    @patch(
        "app.routes.audio.get_audio_job",
        return_value={
            "status": "processing",
            "result": None,
            "error": None,
            "status_code": None,
        },
    )
    def test_audio_job_status_returns_processing(self, _get_job):
        response = self.client.get("/api/process-audio/jobs/job-123")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"status": "processing"})

    @patch("app.routes.audio.get_audio_job", return_value=None)
    def test_audio_job_status_returns_not_found_for_expired_job(self, _get_job):
        response = self.client.get("/api/process-audio/jobs/missing")
        self.assertEqual(response.status_code, 404)

    @patch("app.routes.audio.generate_study_materials", return_value={
        "detailed_class_notes": {},
        "easy_notes": [],
        "revision_quiz": {},
    })
    def test_renaming_speaker_saves_name_and_regenerates_dialogue(self, generate):
        from app.services.speaker_profiles import save_profiles

        save_profiles(
            TestConfig.SPEAKER_PROFILES_PATH,
            [{"id": "profile-1", "display_name": None, "embedding": [1.0, 0.0]}],
        )
        response = self.client.put(
            "/api/speaker-profiles/profile-1/rename",
            json={
                "display_name": "Teacher",
                "dialogue": [
                    {
                        "speaker_id": "profile-1",
                        "speaker_name": "Speaker 1",
                        "start": 2.0,
                        "end": 4.0,
                        "text": "Welcome to class.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["transcript"],
            "[00:02] Teacher: Welcome to class.",
        )
        self.assertEqual(response.get_json()["dialogue"][0]["speaker_name"], "Teacher")
        generate.assert_called_once()
        saved = load_profiles(TestConfig.SPEAKER_PROFILES_PATH)
        self.assertEqual(saved[0]["display_name"], "Teacher")

    def test_renaming_speaker_rejects_invalid_name(self):
        response = self.client.put(
            "/api/speaker-profiles/profile-1/rename",
            json={"display_name": " "},
        )
        self.assertEqual(response.status_code, 400)

    def test_speaker_voice_profiles_match_and_remember_labels(self):
        path = Path(self.upload_dir.name) / "speaker_profiles.json"
        first = _assign_speaker_profiles(
            ["speaker-a"],
            [[1.0, 0.0]],
            path,
            match_threshold=0.78,
        )
        profile_id = first["speaker-a"]["id"]
        profiles = load_profiles(path)
        profiles[0]["display_name"] = "Teacher"
        save_profiles(path, profiles)

        next_recording = _assign_speaker_profiles(
            ["speaker-x"],
            [[0.99, 0.01]],
            path,
            match_threshold=0.78,
        )
        self.assertEqual(next_recording["speaker-x"]["id"], profile_id)
        self.assertEqual(next_recording["speaker-x"]["display_name"], "Teacher")
        self.assertTrue(next_recording["speaker-x"]["is_named"])

    def test_diarization_requires_hugging_face_token(self):
        with self.assertRaisesRegex(SpeakerDiarizationError, "HF_TOKEN"):
            diarize_and_label(
                Path(self.upload_dir.name) / "not-used.wav",
                [{"start": 0.0, "end": 1.0, "text": " hello"}],
                token=None,
                profiles_path=Path(self.upload_dir.name) / "profiles.json",
            )

    def test_forget_speaker_deletes_saved_voice_profile(self):
        save_profiles(
            TestConfig.SPEAKER_PROFILES_PATH,
            [{"id": "profile-1", "display_name": "Teacher", "embedding": [1.0, 0.0]}],
        )

        response = self.client.delete("/api/speaker-profiles/profile-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "deleted"})
        self.assertEqual(load_profiles(TestConfig.SPEAKER_PROFILES_PATH), [])


if __name__ == "__main__":
    unittest.main()
