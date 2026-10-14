import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from app.services.audio_processing import AudioProcessingError, process_audio_file


logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-processing")
_jobs = {}
_client_jobs = {}
_jobs_lock = Lock()
_JOB_TTL = timedelta(hours=2)


def _now():
    return datetime.now(timezone.utc)


def _prune_expired_jobs(now):
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if job["status"] != "processing" and now - job["updated_at"] > _JOB_TTL
    ]
    for job_id in expired:
        del _jobs[job_id]
    expired_client_ids = [
        client_id
        for client_id, (job_id, updated_at) in _client_jobs.items()
        if job_id not in _jobs or now - updated_at > _JOB_TTL
    ]
    for client_id in expired_client_ids:
        del _client_jobs[client_id]


def _keep_uploaded_audio(app) -> bool:
    config = getattr(app, "config", None)
    if not isinstance(config, dict):
        return False
    return bool(config.get("KEEP_UPLOADED_AUDIO"))


def _prune_expired_uploads(app, now):
    """Delete retained uploads once they are older than the retention window."""
    retention_days = app.config.get("UPLOAD_RETENTION_DAYS")
    if not isinstance(retention_days, int) or retention_days <= 0:
        return
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    if not upload_folder.is_dir():
        return
    cutoff = now - timedelta(days=retention_days)
    for path in upload_folder.iterdir():
        try:
            if not path.is_file():
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove expired upload %s", path, exc_info=True)


def _run_job(app, job_id: str, audio_path: Path):
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["updated_at"] = _now()

    def report(progress: int, message: str) -> None:
        update_job_progress(job_id, progress, message)

    try:
        result = process_audio_file(app, audio_path, progress=report)
        with _jobs_lock:
            _jobs[job_id].update(
                status="completed",
                result=result,
                progress=100,
                message="Complete",
                updated_at=_now(),
            )
    except AudioProcessingError as error:
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error=str(error),
                status_code=error.status_code,
                message=str(error),
                updated_at=_now(),
            )
    except Exception:
        logger.exception("Unexpected audio processing job failure")
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error="An unexpected audio processing error occurred.",
                status_code=500,
                message="An unexpected audio processing error occurred.",
                updated_at=_now(),
            )
    finally:
        if not _keep_uploaded_audio(app):
            audio_path.unlink(missing_ok=True)


def update_job_progress(job_id: str, progress: int, message: str) -> None:
    """Record how far a running job has got, for the client's progress bar.

    A finished job keeps its final state: a late callback from a stage that was
    still unwinding must not turn a completed job back into a partial one.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] in ("completed", "failed"):
            return
        job["progress"] = max(0, min(100, int(progress)))
        job["message"] = message
        job["updated_at"] = _now()


def _run_notes_job(app, job_id: str, transcript: str):
    """Generate notes from a transcript the client already transcribed.

    Shares the job store with audio jobs so one status endpoint covers both, and
    has no audio_path cleanup because there is no upload to remove.
    """
    from app.services.note_generator import NoteGenerationError, generate_notes_for_transcript

    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["message"] = "Generating notes..."
        _jobs[job_id]["updated_at"] = _now()

    def report(progress: int, message: str) -> None:
        update_job_progress(job_id, progress, message)

    try:
        with app.app_context():
            materials = generate_notes_for_transcript(
                transcript,
                ollama_url=app.config["OLLAMA_URL"],
                model=app.config["OLLAMA_MODEL"],
                timeout=app.config["OLLAMA_TIMEOUT_SECONDS"],
                chunk_chars=app.config["NOTE_CHUNK_CHARS"],
                max_mcqs=app.config["NOTE_MAX_MCQS"],
                progress=report,
            )
        with _jobs_lock:
            _jobs[job_id].update(
                status="completed",
                result={"transcript": transcript, **materials},
                progress=100,
                message="Complete",
                updated_at=_now(),
            )
    except NoteGenerationError as error:
        logger.warning("Note generation job failed: %s", error)
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error=str(error),
                status_code=502,
                message=str(error),
                updated_at=_now(),
            )
    except Exception:
        logger.exception("Unexpected note generation job failure")
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error="An unexpected error occurred while generating notes.",
                status_code=500,
                message="An unexpected error occurred while generating notes.",
                updated_at=_now(),
            )


def enqueue_notes_job(app, transcript: str) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _prune_expired_jobs(_now())
        _jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "status_code": None,
            "progress": 0,
            "message": "Queued",
            "updated_at": _now(),
        }
    _executor.submit(_run_notes_job, app, job_id, transcript)
    return job_id
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] in ("completed", "failed"):
            return
        job["progress"] = max(0, min(100, int(progress)))
        job["message"] = message
        job["updated_at"] = _now()


def enqueue_audio_job(
    app,
    audio_path: Path,
    client_idempotency_key: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex
    now = _now()
    with _jobs_lock:
        _prune_expired_jobs(now)
        if _keep_uploaded_audio(app):
            _prune_expired_uploads(app, now)
        if client_idempotency_key:
            existing = _client_jobs.get(client_idempotency_key)
            if existing:
                existing_job_id, _ = existing
                existing_job = _jobs.get(existing_job_id)
                if existing_job and existing_job["status"] != "failed":
                    audio_path.unlink(missing_ok=True)
                    _client_jobs[client_idempotency_key] = (existing_job_id, now)
                    return existing_job_id

        _jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "status_code": None,
            "progress": 0,
            "message": "Queued",
            "updated_at": now,
        }
        if client_idempotency_key:
            _client_jobs[client_idempotency_key] = (job_id, now)

    try:
        _executor.submit(_run_job, app, job_id, audio_path)
    except Exception:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        raise
    return job_id


def get_audio_job(job_id: str):
    with _jobs_lock:
        _prune_expired_jobs(_now())
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
            "status_code": job["status_code"],
            "progress": job["progress"],
            "message": job["message"],
        }
