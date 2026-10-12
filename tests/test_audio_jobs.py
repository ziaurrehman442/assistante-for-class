import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from unittest.mock import patch

from app.services.audio_jobs import enqueue_audio_job, get_audio_job


class AudioJobTests(unittest.TestCase):
    def test_idempotency_key_reuses_active_job_and_discards_duplicate_upload(self):
        started = Event()

        def process(_app, _audio_path):
            started.set()
            return {"transcript": "complete"}

        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.wav"
            second_path = Path(directory) / "duplicate.wav"
            first_path.write_bytes(b"audio")
            second_path.write_bytes(b"audio")

            with patch("app.services.audio_jobs.process_audio_file", side_effect=process):
                first_id = enqueue_audio_job(object(), first_path, "client:recording")
                second_id = enqueue_audio_job(object(), second_path, "client:recording")

                self.assertEqual(first_id, second_id)
                self.assertFalse(second_path.exists())
                self.assertTrue(started.wait(timeout=2))

                for _ in range(100):
                    job = get_audio_job(first_id)
                    if job["status"] == "completed":
                        break
                    Event().wait(0.01)

                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["result"], {"transcript": "complete"})

    def test_processed_audio_is_deleted_when_retention_is_off(self):
        def process(_app, _audio_path):
            return {"transcript": "complete"}

        app = _StubApp(upload_folder=None, keep=False)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            audio_path.write_bytes(b"audio")
            app.config["UPLOAD_FOLDER"] = directory

            with patch("app.services.audio_jobs.process_audio_file", side_effect=process):
                job_id = enqueue_audio_job(app, audio_path)
                for _ in range(100):
                    if get_audio_job(job_id)["status"] == "completed":
                        break
                    Event().wait(0.01)

            self.assertFalse(audio_path.exists())

    def test_processed_audio_is_retained_when_retention_is_on(self):
        def process(_app, _audio_path):
            return {"transcript": "complete"}

        app = _StubApp(upload_folder=None, keep=True)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            audio_path.write_bytes(b"audio")
            app.config["UPLOAD_FOLDER"] = directory

            with patch("app.services.audio_jobs.process_audio_file", side_effect=process):
                job_id = enqueue_audio_job(app, audio_path)
                for _ in range(100):
                    if get_audio_job(job_id)["status"] == "completed":
                        break
                    Event().wait(0.01)

            self.assertTrue(audio_path.exists())

    def test_expired_uploads_are_swept_on_the_next_upload(self):
        app = _StubApp(upload_folder=None, keep=True)
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old.wav"
            old_path.write_bytes(b"audio")
            fresh_path = Path(directory) / "fresh.wav"
            fresh_path.write_bytes(b"audio")
            stale = datetime.now(timezone.utc) - timedelta(days=8)
            os.utime(old_path, (stale.timestamp(), stale.timestamp()))

            app.config["UPLOAD_FOLDER"] = directory
            app.config["UPLOAD_RETENTION_DAYS"] = 7

            new_path = Path(directory) / "new.wav"
            new_path.write_bytes(b"audio")

            def process(_app, _audio_path):
                return {"transcript": "complete"}

            with patch("app.services.audio_jobs.process_audio_file", side_effect=process):
                job_id = enqueue_audio_job(app, new_path)
                for _ in range(100):
                    if get_audio_job(job_id)["status"] == "completed":
                        break
                    Event().wait(0.01)

            self.assertFalse(old_path.exists())
            self.assertTrue(fresh_path.exists())
            self.assertTrue(new_path.exists())


class _StubApp:
    def __init__(self, upload_folder, keep):
        self.config = {
            "UPLOAD_FOLDER": str(upload_folder) if upload_folder else "",
            "KEEP_UPLOADED_AUDIO": keep,
            "UPLOAD_RETENTION_DAYS": 7,
        }


if __name__ == "__main__":
    unittest.main()
