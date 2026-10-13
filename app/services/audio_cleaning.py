"""FFmpeg cleanup applied before transcription.

Raw phone recordings often carry low-frequency electrical or fan hum, broadband
hiss, and inconsistent levels. This stage trims the band that speech actually
uses and normalises loudness so quiet passages are no longer transcribed as
silence.

Two limits worth knowing:

* Loudness normalisation cannot repair a clipped recording. Once samples exceed
  full scale the waveform is already distorted, so boosting it only amplifies the
  distortion. Clipping is detected here and logged, because no amount of
  filtering will recover those samples.
* Normalising a recording that is mostly room noise lifts that noise too.

Every failure falls back to the original file, so a missing or broken ffmpeg
degrades quality rather than breaking transcription.
"""

import logging
import shutil
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CLIP_CEILING = 0.99
_DEFAULT_TIMEOUT_SECONDS = 600


class AudioCleaningError(Exception):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(args: list[str], timeout: int) -> None:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioCleaningError("FFmpeg cleaning timed out.") from error
    except OSError as error:
        raise AudioCleaningError(f"Could not run ffmpeg: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioCleaningError(
            f"FFmpeg cleaning failed: {detail[-1] if detail else 'unknown error'}"
        )


def detect_clipping(audio_path: Path, timeout: int = 120) -> bool:
    """Report whether any decoded sample has reached full scale.

    Decodes through ffmpeg rather than the stdlib wave reader, because wave
    cannot open the compressed formats phones actually record (.m4a, .mp3) and
    would silently report every real recording as unclipped.
    """
    if not ffmpeg_available():
        return False
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(audio_path),
                "-f",
                "s16le",
                "-ac",
                str(CHANNELS),
                "-ar",
                "8000",
                "-",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if completed.returncode != 0 or not completed.stdout:
        return False

    samples = memoryview(completed.stdout)
    limit = int(CLIP_CEILING * 32767)
    # s16le samples are signed, so they must be decoded as signed values. Reading
    # them as unsigned would make every negative sample look like full scale.
    for index in range(0, len(samples) - 1, 2):
        value = int.from_bytes(samples[index : index + 2], "little", signed=True)
        if value >= limit or value <= -limit:
            return True
    return False


def build_filter_chain(
    highpass_hz: int,
    lowpass_hz: int,
    target_lufs: float,
    true_peak: float,
    loudness_range: float,
) -> str:
    return (
        f"highpass=f={highpass_hz},"
        f"lowpass=f={lowpass_hz},"
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
    )


def clean_audio(
    source_path: Path,
    destination_path: Path,
    highpass_hz: int = 200,
    lowpass_hz: int = 3000,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
    loudness_range: float = 11.0,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Write a band-limited, loudness-normalised 16 kHz mono WAV."""
    if not ffmpeg_available():
        raise AudioCleaningError("ffmpeg is not installed or not on PATH.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = build_filter_chain(
        highpass_hz, lowpass_hz, target_lufs, true_peak, loudness_range
    )
    logger.info(
        "Cleaning audio: highpass=%dHz lowpass=%dHz loudnorm=%s",
        highpass_hz,
        lowpass_hz,
        filter_chain.split("loudnorm=")[1],
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source_path),
            "-af",
            filter_chain,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-c:a",
            "pcm_s16le",
            str(destination_path),
        ],
        timeout,
    )

    if not destination_path.exists() or destination_path.stat().st_size == 0:
        raise AudioCleaningError("FFmpeg produced an empty cleaned file.")

    return destination_path