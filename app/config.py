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
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))
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
