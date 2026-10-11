import logging
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.config import origin_is_allowed
from app.services.audio_jobs import enqueue_audio_job, get_audio_job
from app.services.note_generator import (
    NoteGenerationError,
    format_dialogue,
    generate_study_materials,
)
from app.services.rate_limiter import consume_request
from app.services.transcription import (
    SpeakerDiarizationError,
    SpeakerProfileError,
    delete_speaker_profile,
    rename_speaker_profile,
)


audio_bp = Blueprint("audio", __name__)
ALLOWED_EXTENSIONS = {".m4a", ".mp3", ".wav"}
logger = logging.getLogger(__name__)


@audio_bp.get("/api/health")
def health():
    return jsonify(status="ok", audio_processing="async-jobs")


@audio_bp.post("/api/process-audio")
def process_audio():
    origin = request.headers.get("Origin")
    if origin and not origin_is_allowed(origin, current_app.config["ALLOWED_ORIGINS"]):
        return jsonify(error="This request origin is not allowed by the server configuration."), 403

    audio_file = request.files.get("file")
    if audio_file is None:
        return jsonify(error="Provide an audio file in the 'file' form field."), 400
    if not audio_file.filename:
        return jsonify(error="The uploaded file must have a name."), 400

    safe_name = secure_filename(audio_file.filename)
    extension = Path(safe_name).suffix.lower()
    if not safe_name or extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Supported audio formats are .m4a, .mp3, and .wav."), 400

    retry_after = consume_request(
        request.remote_addr or "unknown",
        current_app.config["PROCESS_AUDIO_RATE_LIMIT"],
        current_app.config["PROCESS_AUDIO_RATE_WINDOW_SECONDS"],
    )
    if retry_after is not None:
        response = jsonify(error="Too many audio requests. Please try again later.")
        response.headers["Retry-After"] = str(retry_after)
        return response, 429

    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid.uuid4().hex}{extension}"
    try:
        audio_file.save(saved_path)
        recording_id = request.headers.get("X-Client-Recording-Id", "")
        idempotency_key = (
            f"{request.remote_addr or 'unknown'}:{recording_id}"
            if recording_id and len(recording_id) <= 128
            else None
        )
        job_id = enqueue_audio_job(
            current_app._get_current_object(),
            saved_path,
            client_idempotency_key=idempotency_key,
        )
        logger.info("Queued audio processing job %s", job_id)
        return (
            jsonify(job_id=job_id, status="queued"),
            202,
            {"Location": f"/api/process-audio/jobs/{job_id}"},
        )
    except Exception:
        logger.exception("Could not queue uploaded audio")
        saved_path.unlink(missing_ok=True)
        return jsonify(error="Could not queue audio for processing."), 500


@audio_bp.get("/api/process-audio/jobs/<job_id>")
def audio_job_status(job_id):
    job = get_audio_job(job_id)
    if job is None:
        return jsonify(error="Audio processing job was not found or has expired."), 404
    if job["status"] == "failed":
        return jsonify(status="failed", error=job["error"])
    if job["status"] == "completed":
        return jsonify(status="completed", result=job["result"])
    return jsonify(status=job["status"]), 202


@audio_bp.put("/api/speaker-profiles/<profile_id>/rename")
def rename_speaker(profile_id):
    origin = request.headers.get("Origin")
    if origin and not origin_is_allowed(origin, current_app.config["ALLOWED_ORIGINS"]):
        return jsonify(error="This request origin is not allowed by the server configuration."), 403

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Provide a JSON request body."), 400

    display_name = body.get("display_name")
    dialogue = body.get("dialogue")
    if (
        not isinstance(display_name, str)
        or not display_name.strip()
        or len(display_name.strip()) > 40
    ):
        return jsonify(error="Speaker name must contain 1 to 40 characters."), 400
    if not isinstance(dialogue, list) or not dialogue:
        return jsonify(error="Provide the dialogue to regenerate study materials."), 400
    if not all(
        isinstance(turn, dict)
        and isinstance(turn.get("speaker_id"), str)
        and isinstance(turn.get("speaker_name"), str)
        and isinstance(turn.get("text"), str)
        and isinstance(turn.get("start"), (int, float))
        and isinstance(turn.get("end"), (int, float))
        for turn in dialogue
    ):
        return jsonify(error="The dialogue contains an invalid speaker turn."), 400

    retry_after = consume_request(
        request.remote_addr or "unknown",
        current_app.config["PROCESS_AUDIO_RATE_LIMIT"],
        current_app.config["PROCESS_AUDIO_RATE_WINDOW_SECONDS"],
    )
    if retry_after is not None:
        response = jsonify(error="Too many requests. Please try again later.")
        response.headers["Retry-After"] = str(retry_after)
        return response, 429

    updated_dialogue = [
        {
            **turn,
            "speaker_name": (
                display_name.strip()
                if turn["speaker_id"] == profile_id
                else turn["speaker_name"]
            ),
        }
        for turn in dialogue
    ]
    transcript = format_dialogue(updated_dialogue)
    try:
        materials = generate_study_materials(
            transcript,
            ollama_url=current_app.config["OLLAMA_URL"],
            model=current_app.config["OLLAMA_MODEL"],
            timeout=current_app.config["OLLAMA_TIMEOUT_SECONDS"],
        )
        rename_speaker_profile(
            profile_id,
            display_name.strip(),
            Path(current_app.config["SPEAKER_PROFILES_PATH"]),
        )
        return jsonify(transcript=transcript, dialogue=updated_dialogue, **materials)
    except SpeakerProfileError as error:
        return jsonify(error=str(error)), 404
    except RuntimeError:
        current_app.logger.exception("Could not update the local speaker profile")
        return jsonify(error="Could not save this speaker name locally."), 500
    except NoteGenerationError as error:
        logger.warning("Study material regeneration failed: %s", error)
        return jsonify(error=str(error)), 502


@audio_bp.delete("/api/speaker-profiles/<profile_id>")
def forget_speaker(profile_id):
    origin = request.headers.get("Origin")
    if origin and not origin_is_allowed(origin, current_app.config["ALLOWED_ORIGINS"]):
        return jsonify(error="This request origin is not allowed by the server configuration."), 403

    try:
        if not delete_speaker_profile(
            profile_id,
            Path(current_app.config["SPEAKER_PROFILES_PATH"]),
        ):
            return jsonify(error="Speaker profile was not found."), 404
        return jsonify(status="deleted")
    except RuntimeError:
        current_app.logger.exception("Could not delete local speaker profile")
        return jsonify(error="Could not delete this speaker profile locally."), 500
