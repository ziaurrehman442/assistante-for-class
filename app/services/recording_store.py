"""Mapping from a client's recording id to the audio file kept on disk.

Job state lives in memory and expires after two hours, but retained audio lives
in the upload folder for the retention window. Without a durable mapping there is
no way to answer "reprocess that old recording": the job id and the stored
filename are generated independently, so neither one can be derived from the
other.

The manifest is a small JSON file written atomically, mirroring the speaker
profile store.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock


_manifest_lock = Lock()


def _read(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic so a crash mid-write cannot leave a truncated manifest that would
    # orphan every retained recording.
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(entries, stream, indent=2)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _now() -> datetime:
    return datetime.now(timezone.utc)


def remember(recording_id: str, audio_path: Path, path: Path) -> None:
    if not recording_id:
        return

    def store(entries):
        entries[:] = [
            entry for entry in entries if entry.get("recording_id") != recording_id
        ]
        entries.append(
            {
                "recording_id": recording_id,
                "filename": audio_path.name,
                "recorded_at": _now().isoformat(),
            }
        )
        return None, True

    with _manifest_lock:
        entries = _read(path)
        store(entries)
        _write(path, entries)


def lookup(recording_id: str, upload_dir: Path, manifest_path: Path) -> Path | None:
    """Return the retained file for a recording, or None if it is not kept.

    Entries are dropped once the file is gone, so a deleted or swept upload
    reports as missing instead of failing later on a missing path.
    """
    with _manifest_lock:
        entries = _read(manifest_path)
        entry = next(
            (item for item in entries if item.get("recording_id") == recording_id),
            None,
        )
        if entry is None:
            return None

        candidate = upload_dir / str(entry.get("filename") or "")
        if candidate.is_file():
            return candidate

        entries[:] = [
            item for item in entries if item.get("recording_id") != recording_id
        ]
        _write(manifest_path, entries)
        return None


def prune(upload_dir: Path, manifest_path: Path, retention_days: int) -> int:
    """Drop manifest entries whose audio has been swept or deleted."""
    if retention_days <= 0:
        return 0
    cutoff = _now() - timedelta(days=retention_days)
    with _manifest_lock:
        entries = _read(manifest_path)
        kept = []
        removed = 0
        for entry in entries:
            audio = upload_dir / str(entry.get("filename") or "")
            if not audio.is_file():
                removed += 1
                continue
            recorded_at = str(entry.get("recorded_at") or "")
            try:
                stamp = datetime.fromisoformat(recorded_at)
            except ValueError:
                removed += 1
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            if stamp >= cutoff:
                kept.append(entry)
            else:
                removed += 1
        if removed:
            _write(manifest_path, kept)
        return removed