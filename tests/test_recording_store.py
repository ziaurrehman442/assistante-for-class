import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.recording_store import lookup, prune, remember


class RecordingStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.manifest = self.root / "recordings.json"

    def tearDown(self):
        self.directory.cleanup()

    def _audio(self, name="lecture.m4a"):
        path = self.uploads / name
        path.write_bytes(b"audio")
        return path

    def test_recording_round_trips_to_its_file(self):
        audio = self._audio()
        remember("rec-1", audio, self.manifest)

        self.assertEqual(lookup("rec-1", self.uploads, self.manifest), audio)

    def test_blank_id_is_ignored(self):
        remember("", self._audio(), self.manifest)
        self.assertFalse(self.manifest.exists())

    def test_unknown_recording_is_missing(self):
        self.assertIsNone(lookup("nope", self.uploads, self.manifest))

    def test_deleted_audio_is_reported_missing_and_dropped(self):
        audio = self._audio()
        remember("rec-1", audio, self.manifest)
        audio.unlink()

        self.assertIsNone(lookup("rec-1", self.uploads, self.manifest))
        self.assertEqual(json.loads(self.manifest.read_text()), [])

    def test_re_recording_the_same_id_replaces_the_entry(self):
        first = self._audio("first.m4a")
        second = self._audio("second.m4a")
        remember("rec-1", first, self.manifest)
        remember("rec-1", second, self.manifest)

        entries = json.loads(self.manifest.read_text())
        self.assertEqual(len(entries), 1)
        self.assertEqual(lookup("rec-1", self.uploads, self.manifest), second)

    def test_prune_drops_entries_older_than_the_window(self):
        old = self._audio("old.m4a")
        recent = self._audio("recent.m4a")
        remember("rec-old", old, self.manifest)
        remember("rec-new", recent, self.manifest)

        entries = json.loads(self.manifest.read_text())
        entries[0]["recorded_at"] = (
            datetime.now(timezone.utc) - timedelta(days=9)
        ).isoformat()
        self.manifest.write_text(json.dumps(entries))

        removed = prune(self.uploads, self.manifest, 7)

        self.assertEqual(removed, 1)
        self.assertIsNone(lookup("rec-old", self.uploads, self.manifest))
        self.assertEqual(lookup("rec-new", self.uploads, self.manifest), recent)

    def test_corrupt_manifest_is_treated_as_empty(self):
        self.manifest.write_text("{not json")
        self.assertIsNone(lookup("rec-1", self.uploads, self.manifest))
