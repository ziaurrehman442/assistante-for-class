import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
