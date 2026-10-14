import logging
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.config import origin_is_allowed
from app.services.audio_jobs import enqueue_audio_job, enqueue_notes_job, get_audio_job
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


def _flatten_materials(result):
    """Render the structured study materials as plain text.

    The nested JSON shape stays on /api/process-audio/jobs for the existing
    client; this renders the same data into readable sections for clients that
    want to display a section as one block of text.
    """
    notes = (result or {}).get("detailed_class_notes") or {}
    quiz = (result or {}).get("revision_quiz") or {}

    detailed = []
    for label, key in (
        ("Main topics", "main_topics"),
        ("Definitions", "definitions"),
        ("Teacher explanations", "teacher_explanations"),
    ):
        items = notes.get(key) or []
        if items:
            detailed.append(f"## {label}\n" + "\n".join(f"- {item}" for item in items))
    qa = notes.get("qa") or []
    if qa:
        detailed.append(
            "## Questions and answers\n"
            + "\n".join(f"- Q: {item['question']}\n  A: {item['answer']}" for item in qa)
        )

    mcqs = quiz.get("mcqs") or []
    conceptual = quiz.get("conceptual_questions") or []
    quiz_lines = []
    for number, item in enumerate(mcqs, start=1):
        options = "\n".join(f"   {chr(65 + i)}. {opt}" for i, opt in enumerate(item["options"]))
        quiz_lines.append(f"{number}. {item['question']}\n{options}\n   Answer: {item['answer']}")
    if conceptual:
        quiz_lines.append("## Conceptual questions\n" + "\n".join(f"- {q}" for q in conceptual))

    return (
        "\n\n".join(detailed),
        "\n".join(f"- {item}" for item in ((result or {}).get("easy_notes") or [])),
        "\n\n".join(quiz_lines),
    )


def _queue_upload(request, audio_file, status_label: str, location: str):
    """Validate, persist, and queue an upload. Shared by both upload routes.

    status_label and location differ per route so the original route keeps the
    exact response contract the shipped client already depends on.
    """
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
    safe_name = secure_filename(audio_file.filename)
    extension = Path(safe_name).suffix.lower()
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
        payload = {"job_id": job_id, "status": status_label}
        if status_label == "processing":
            payload["progress"] = 0
        return (
            jsonify(**payload),
            202,
            {"Location": location.format(job_id=job_id)},
        )
    except Exception:
        logger.exception("Could not queue uploaded audio")
        saved_path.unlink(missing_ok=True)
        return jsonify(error="Could not queue audio for processing."), 500


def _upload_request():
    """Shared checks for the two upload routes."""
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
    return audio_file


@audio_bp.post("/api/process-audio")
def process_audio():
    audio_file = _upload_request()
    if not hasattr(audio_file, "filename"):
        return audio_file
    return _queue_upload(request, audio_file, "queued", "/api/process-audio/jobs/{job_id}")


@audio_bp.post("/api/upload-audio")
def upload_audio():
    """Instant, non-blocking upload. Returns immediately with a job to poll."""
    audio_file = _upload_request()
    if not hasattr(audio_file, "filename"):
        return audio_file
    return _queue_upload(request, audio_file, "processing", "/api/job-status/{job_id}")


@audio_bp.get("/api/job-status/<job_id>")
def job_status(job_id):
    """Flat, progress-aware view of a job for simple polling clients."""
    job = get_audio_job(job_id)
    if job is None:
        return jsonify(job_id=job_id, status="failed", progress=0, message="Job not found.", transcript=None, detailed_notes=None, easy_notes=None, quiz=None, error="Audio processing job was not found or has expired."), 404

    result = job["result"] if job["status"] == "completed" else None
    detailed_notes, easy_notes, quiz = _flatten_materials(result)
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "transcript": (result or {}).get("transcript"),
        "detailed_notes": detailed_notes or None,
        "easy_notes": easy_notes or None,
        "quiz": quiz or None,
    }
    if job["status"] == "failed":
        payload["error"] = job["error"]
        return jsonify(**payload), job["status_code"] or 500
    if job["status"] == "completed":
        payload["warnings"] = (result or {}).get("warnings") or []
        return jsonify(**payload), 200
    return jsonify(**payload), 202


@audio_bp.post("/api/generate-notes")
def generate_notes():
    """Accept a transcript the client transcribed, and return notes for it.

    No audio is uploaded and nothing is transcribed here, so this endpoint costs
    only the Ollama generation the client actually asked for.
    """
    origin = request.headers.get("Origin")
    if origin and not origin_is_allowed(origin, current_app.config["ALLOWED_ORIGINS"]):
        return jsonify(error="This request origin is not allowed by the server configuration."), 403

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Provide a JSON request body."), 400

    transcript = body.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        return jsonify(error="Provide a non-empty 'transcript' string."), 400

    limit = current_app.config["MAX_TRANSCRIPT_CHARS"]
    if len(transcript) > limit:
        return jsonify(error=f"The transcript exceeds the {limit} character limit."), 413

    retry_after = consume_request(
        request.remote_addr or "unknown",
        current_app.config["PROCESS_AUDIO_RATE_LIMIT"],
        current_app.config["PROCESS_AUDIO_RATE_WINDOW_SECONDS"],
    )
    if retry_after is not None:
        response = jsonify(error="Too many requests. Please try again later.")
        response.headers["Retry-After"] = str(retry_after)
        return response, 429

    job_id = enqueue_notes_job(current_app._get_current_object(), transcript)
    logger.info("Queued note generation job %s for %d characters", job_id, len(transcript))
    return (
        jsonify(job_id=job_id, status="processing", progress=0),
        202,
        {"Location": f"/api/job-status/{job_id}"},
    )


@audio_bp.get("/api/process-audio/jobs/<job_id>")
def audio_job_status(job_id):
    job = get_audio_job(job_id)
    if job is None:
        return jsonify(error="Audio processing job was not found or has expired."), 404
    # progress and message are additive: the shipped client ignores unknown keys,
    # so this keeps status/result exactly as before while adding a progress bar.
    if job["status"] == "failed":
        return jsonify(
            status="failed",
            error=job["error"],
            progress=job["progress"],
            message=job["message"],
        )
    if job["status"] == "completed":
        return jsonify(
            status="completed",
            result=job["result"],
            progress=100,
            message="Complete",
        )
    return jsonify(
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    ), 202


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
