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


def _run_job(app, job_id: str, audio_path: Path):
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["updated_at"] = _now()

    try:
        result = process_audio_file(app, audio_path)
        with _jobs_lock:
            _jobs[job_id].update(
                status="completed",
                result=result,
                updated_at=_now(),
            )
    except AudioProcessingError as error:
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error=str(error),
                status_code=error.status_code,
                updated_at=_now(),
            )
    except Exception:
        logger.exception("Unexpected audio processing job failure")
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error="An unexpected audio processing error occurred.",
                status_code=500,
                updated_at=_now(),
            )
    finally:
        audio_path.unlink(missing_ok=True)


def enqueue_audio_job(
    app,
    audio_path: Path,
    client_idempotency_key: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex
    now = _now()
    with _jobs_lock:
        _prune_expired_jobs(now)
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
        }
