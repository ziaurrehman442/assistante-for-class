import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.audio_cleaning import (
    AudioCleaningError,
    build_filter_chain,
    clean_audio,
    detect_clipping,
    ffmpeg_available,
)


class FilterChainTests(unittest.TestCase):
    def test_chain_trims_band_and_normalises(self):
        chain = build_filter_chain(200, 3000, -16.0, -1.5, 11.0)
        self.assertEqual(
            chain,
            "highpass=f=200,lowpass=f=3000,loudnorm=I=-16.0:TP=-1.5:LRA=11.0",
        )

    def test_settings_are_honoured(self):
        chain = build_filter_chain(150, 4000, -19.0, -2.0, 7.0)
        self.assertIn("highpass=f=150", chain)
        self.assertIn("lowpass=f=4000", chain)
        self.assertIn("I=-19.0", chain)


class ClippingTests(unittest.TestCase):
    def test_detects_clipping_in_a_generated_tone(self):
        if not ffmpeg_available():
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loud.wav"
            # A full-scale square-ish tone: write samples that pin to the rails.
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "aevalsrc=0.99*sin(440*2*PI*t):d=1:s=8000",
                    "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", str(path),
                ],
                check=True,
            )
            self.assertIs(detect_clipping(path), True)

    def test_quiet_audio_is_not_reported_as_clipped(self):
        if not ffmpeg_available():
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quiet.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "aevalsrc=0.02*sin(440*2*PI*t):d=1:s=8000",
                    "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", str(path),
                ],
                check=True,
            )
            self.assertIs(detect_clipping(path), False)

    def test_missing_file_is_treated_as_unclipped(self):
        self.assertIs(detect_clipping(Path("/nonexistent/audio.m4a")), False)


class CleanAudioTests(unittest.TestCase):
    def test_cleaning_produces_16k_mono_wav(self):
        if not ffmpeg_available():
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src.m4a"
            destination = Path(directory) / "cleaned.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    str(source),
                ],
                check=True,
            )
            clean_audio(source, destination)
            self.assertTrue(destination.exists())

            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "stream=sample_rate,channels",
                    "-of", "csv=p=0", str(destination),
                ],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), "16000,1")

    def test_failure_is_reported_as_cleaning_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AudioCleaningError):
                clean_audio(
                    Path(directory) / "missing.m4a",
                    Path(directory) / "out.wav",
                )

    def test_empty_output_is_rejected(self):
        if not ffmpeg_available():
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src.m4a"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(source),
                ],
                check=True,
            )
            with patch(
                "app.services.audio_cleaning._run",
                side_effect=lambda *a, **k: None,
            ):
                with self.assertRaises(AudioCleaningError):
                    clean_audio(source, Path(directory) / "out.wav")


class FallbackTests(unittest.TestCase):
    def test_unavailable_ffmpeg_raises_cleaning_error(self):
        with patch("app.services.audio_cleaning.shutil.which", return_value=None):
            self.assertFalse(ffmpeg_available())
            with self.assertRaises(AudioCleaningError):
                clean_audio(Path("in.m4a"), Path("out.wav"))


if __name__ == "__main__":
    unittest.main()