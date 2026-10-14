import os
from pathlib import Path
import re

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
LOCAL_ORIGIN_PATTERN = re.compile(
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
)


def configured_origins():
    origins = os.environ.get("ALLOWED_ORIGINS")
    if origins is None:
        return [LOCAL_ORIGIN_PATTERN]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def origin_is_allowed(origin: str, allowed_origins) -> bool:
    return any(
        allowed.fullmatch(origin) if hasattr(allowed, "fullmatch") else allowed == origin
        for allowed in allowed_origins
    )


class Config:
    SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))
    ALLOWED_ORIGINS = configured_origins()
    UPLOAD_FOLDER = Path(
        os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    )
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
    # Generation on a 4 vCPU box runs at roughly 5 tokens/second, so a full set
    # of study materials takes several minutes. A timeout below the real
    # generation time turns a working Ollama into a failed job, so this is set
    # well above the worst case rather than at the default request timeout.
    OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "900"))
    # Kept short: this check only proves the server is up and the model is
    # installed, so it should fail fast instead of stalling a job.
    OLLAMA_AVAILABILITY_TIMEOUT_SECONDS = int(
        os.environ.get("OLLAMA_AVAILABILITY_TIMEOUT_SECONDS", "5")
    )
    # A client-transcribed lecture can be hours long, so the transcript is split
    # and generated a piece at a time instead of overflowing the context window.
    NOTE_CHUNK_CHARS = int(os.environ.get("NOTE_CHUNK_CHARS", "6000"))
    NOTE_MAX_MCQS = int(os.environ.get("NOTE_MAX_MCQS", "5"))
    MAX_TRANSCRIPT_CHARS = int(
        os.environ.get("MAX_TRANSCRIPT_CHARS", str(2 * 1024 * 1024))
    )
    # Faster-Whisper runs on CPU. "small" mis-transcribes Urdu and
    # English-Urdu code-switching badly, so "medium" is the default; it needs
    # roughly 2.5-3 GB of RAM while a recording is being processed.
    WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
    WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    # Accuracy settings for quiet, distant speech. A wider beam search resolves
    # more soft, ambiguous words at the cost of CPU time, and a short VAD silence
    # window keeps the natural pauses and hesitations of a hesitant speaker
    # instead of treating them as silence to discard.
    WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
    WHISPER_MIN_SILENCE_MS = int(os.environ.get("WHISPER_MIN_SILENCE_MS", "500"))
    # FFmpeg cleanup before transcription: trims the band speech occupies and
    # normalises loudness so quiet passages are not transcribed as silence.
    AUDIO_CLEANING_ENABLED = os.environ.get(
        "AUDIO_CLEANING_ENABLED", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    AUDIO_CLEANING_HIGHPASS_HZ = int(os.environ.get("AUDIO_CLEANING_HIGHPASS_HZ", "200"))
    AUDIO_CLEANING_LOWPASS_HZ = int(os.environ.get("AUDIO_CLEANING_LOWPASS_HZ", "3000"))
    AUDIO_CLEANING_TARGET_LUFS = os.environ.get("AUDIO_CLEANING_TARGET_LUFS", "-16")
    AUDIO_CLEANING_TRUE_PEAK = os.environ.get("AUDIO_CLEANING_TRUE_PEAK", "-1.5")
    AUDIO_CLEANING_LOUDNESS_RANGE = os.environ.get("AUDIO_CLEANING_LOUDNESS_RANGE", "11")
    AUDIO_CLEANING_TIMEOUT_SECONDS = int(
        os.environ.get("AUDIO_CLEANING_TIMEOUT_SECONDS", "600")
    )
    HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
    SPEAKER_PROFILES_PATH = Path(
        os.environ.get(
            "SPEAKER_PROFILES_PATH", str(BASE_DIR / "data" / "speaker_profiles.json")
        )
    )
    SPEAKER_MATCH_THRESHOLD = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.78"))
    PROCESS_AUDIO_RATE_LIMIT = int(os.environ.get("PROCESS_AUDIO_RATE_LIMIT", "10"))
    PROCESS_AUDIO_RATE_WINDOW_SECONDS = int(
        os.environ.get("PROCESS_AUDIO_RATE_WINDOW_SECONDS", "60")
    )
    # Processed audio is normally deleted straight after the job finishes. When
    # retention is enabled the upload is kept so a recording can be transcribed
    # again with a different model, and expired files are swept on the next
    # upload. Recordings are sensitive, so this defaults to off.
    KEEP_UPLOADED_AUDIO = os.environ.get("KEEP_UPLOADED_AUDIO", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    UPLOAD_RETENTION_DAYS = int(os.environ.get("UPLOAD_RETENTION_DAYS", "7"))
    # Optional cleanup of a transcript through a text model, which repairs the
    # mixed-script output that speech-to-text produces for Urdu/English
    # code-switching. It can only reword turns; a failure always falls back to
    # the raw transcript. Off by default because it sends lecture content to a
    # third-party API.
    TRANSCRIPT_REFINER_ENABLED = os.environ.get(
        "TRANSCRIPT_REFINER_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.nexa.website")
    OPENCODE_USER = os.environ.get("OPENCODE_USER")
    OPENCODE_PASSWORD = os.environ.get("OPENCODE_PASSWORD")
    TRANSCRIPT_REFINER_MODEL = os.environ.get(
        "TRANSCRIPT_REFINER_MODEL", "space-bunny-free"
    )
    TRANSCRIPT_REFINER_TIMEOUT_SECONDS = int(
        os.environ.get("TRANSCRIPT_REFINER_TIMEOUT_SECONDS", "120")
    )
