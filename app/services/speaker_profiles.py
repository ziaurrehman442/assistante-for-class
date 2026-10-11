import json
import os
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar


_profile_lock = Lock()
Result = TypeVar("Result")


def _read_profiles(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not read the local speaker profile store.") from error
    if not isinstance(profiles, list) or not all(
        isinstance(profile, dict)
        and isinstance(profile.get("id"), str)
        and isinstance(profile.get("embedding"), list)
        and all(isinstance(value, (int, float)) for value in profile["embedding"])
        and (
            profile.get("display_name") is None
            or isinstance(profile.get("display_name"), str)
        )
        for profile in profiles
    ):
        raise RuntimeError("The local speaker profile store is invalid.")
    return profiles


def _write_profiles(path: Path, profiles: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(profiles, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as error:
        raise RuntimeError("Could not save local speaker profiles.") from error


def load_profiles(path: Path) -> list[dict]:
    with _profile_lock:
        return _read_profiles(path)


def save_profiles(path: Path, profiles: list[dict]) -> None:
    with _profile_lock:
        _write_profiles(path, profiles)


def update_profiles(
    path: Path,
    update: Callable[[list[dict]], tuple[Result, bool]],
) -> Result:
    with _profile_lock:
        profiles = _read_profiles(path)
        result, changed = update(profiles)
        if changed:
            _write_profiles(path, profiles)
        return result


def delete_profile(profile_id: str, path: Path) -> bool:
    def remove(profiles):
        remaining = [profile for profile in profiles if profile["id"] != profile_id]
        if len(remaining) == len(profiles):
            return False, False
        profiles[:] = remaining
        return True, True

    return update_profiles(path, remove)
