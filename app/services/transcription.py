from pathlib import Path
from threading import Lock


class TranscriptionError(Exception):
    pass


class SpeakerDiarizationError(Exception):
    pass


class SpeakerProfileError(Exception):
    pass


_model = None
_model_lock = Lock()
_diarization_pipeline = None
_diarization_lock = Lock()
MULTILINGUAL_TRANSCRIPTION_PROMPT = (
    "This recording may switch between Urdu and English, including within a "
    "sentence. Transcribe the words as spoken without translating them. Write "
    "Urdu words in Urdu script and English words in Latin script. Preserve "
    "English technical terms and names."
)


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from faster_whisper import WhisperModel

                    _model = WhisperModel("small", device="cpu", compute_type="int8")
                except Exception as error:
                    raise TranscriptionError(
                        "Could not initialize Faster-Whisper. Check the installation "
                        "and that the small model can be downloaded."
                    ) from error
    return _model


def transcribe_audio(audio_path: Path) -> list[dict[str, float | str]]:
    try:
        segments, _ = _get_model().transcribe(
            str(audio_path),
            language=None,
            task="transcribe",
            initial_prompt=MULTILINGUAL_TRANSCRIPTION_PROMPT,
            word_timestamps=True,
            vad_filter=True,
        )
        words = []
        for segment in segments:
            if segment.words:
                words.extend(
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "text": word.word,
                    }
                    for word in segment.words
                    if word.word.strip()
                )
            elif segment.text.strip():
                words.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text,
                    }
                )
        return words
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError(
            "Could not transcribe this audio file. Check that it is a valid recording."
        ) from error


def _get_diarization_pipeline(token: str | None):
    global _diarization_pipeline
    if not token:
        raise SpeakerDiarizationError(
            "Speaker diarization needs a Hugging Face token. Set HF_TOKEN and "
            "accept the pyannote/speaker-diarization-community-1 model conditions."
        )

    if _diarization_pipeline is None:
        with _diarization_lock:
            if _diarization_pipeline is None:
                try:
                    from pyannote.audio import Pipeline
                    import torch

                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=token,
                    )
                    if pipeline is None:
                        raise RuntimeError("The diarization model could not be loaded.")
                    pipeline.to(torch.device("cpu"))
                    _diarization_pipeline = pipeline
                except Exception as error:
                    raise SpeakerDiarizationError(
                        "Could not initialize local speaker diarization. Verify "
                        "HF_TOKEN, accept the model conditions, and install pyannote.audio."
                    ) from error
    return _diarization_pipeline


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _normalize_embedding(values) -> list[float]:
    import numpy as np

    embedding = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm == 0 or not np.isfinite(embedding).all():
        raise SpeakerDiarizationError("Diarization produced an invalid speaker profile.")
    return (embedding / norm).astype(float).tolist()


def _assign_speaker_profiles(
    labels: list[str],
    embeddings,
    profiles_path: Path,
    match_threshold: float,
) -> dict[str, dict[str, str | bool]]:
    import uuid

    from app.services.speaker_profiles import update_profiles

    vectors = [_normalize_embedding(embeddings[index]) for index in range(len(labels))]

    def assign(profiles):
        matched_in_recording = set()
        speakers = {}
        unnamed_number = 0

        for label, vector in zip(labels, vectors):
            candidates = [
                (profile, _cosine_similarity(vector, profile["embedding"]))
                for profile in profiles
                if profile["id"] not in matched_in_recording
            ]
            best = max(candidates, key=lambda candidate: candidate[1], default=None)

            if best is not None and best[1] >= match_threshold:
                profile = best[0]
                profile["embedding"] = _normalize_embedding(
                    [old + new for old, new in zip(profile["embedding"], vector)]
                )
            else:
                profile = {
                    "id": uuid.uuid4().hex,
                    "display_name": None,
                    "embedding": vector,
                }
                profiles.append(profile)

            matched_in_recording.add(profile["id"])
            if profile["display_name"] is None:
                unnamed_number += 1
            display_name = profile["display_name"] or f"Speaker {unnamed_number}"
            speakers[label] = {
                "id": profile["id"],
                "display_name": display_name,
                "is_named": profile["display_name"] is not None,
            }

        return speakers, True

    return update_profiles(profiles_path, assign)


def diarize_and_label(
    audio_path: Path,
    words: list[dict[str, float | str]],
    token: str | None,
    profiles_path: Path,
    match_threshold: float = 0.78,
) -> tuple[
    list[dict[str, float | str | list[str]]],
    list[dict[str, str | bool]],
]:
    if not words:
        raise SpeakerDiarizationError("No spoken words were found to diarize.")

    try:
        pipeline = _get_diarization_pipeline(token)
        output = pipeline(str(audio_path))
        diarization = output.exclusive_speaker_diarization
        labels = list(output.speaker_diarization.labels())
        embeddings = output.speaker_embeddings
        if embeddings is None or len(embeddings) != len(labels):
            raise SpeakerDiarizationError(
                "The diarization model did not return speaker voice profiles."
            )

        profiles_by_label = _assign_speaker_profiles(
            labels, embeddings, profiles_path, match_threshold
        )
        turns = [
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        all_turns = [
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in output.speaker_diarization.itertracks(
                yield_label=True
            )
        ]

        dialogue: list[dict[str, float | str | list[str]]] = []
        current = None
        for word in words:
            start = float(word["start"])
            end = float(word["end"])
            midpoint = (start + end) / 2
            active = [
                (turn_start, turn_end, speaker)
                for turn_start, turn_end, speaker in turns
                if turn_start <= midpoint <= turn_end
            ]
            if not active:
                overlaps = [
                    (
                        max(0.0, min(end, turn_end) - max(start, turn_start)),
                        speaker,
                    )
                    for turn_start, turn_end, speaker in turns
                ]
                overlap, speaker_label = max(overlaps, default=(0.0, None))
                if overlap <= 0:
                    continue
            else:
                speaker_label = max(
                    active,
                    key=lambda turn: min(end, turn[1]) - max(start, turn[0]),
                )[2]

            speaker = profiles_by_label[speaker_label]
            overlap_names = sorted(
                {
                    str(profiles_by_label[other_label]["display_name"])
                    for turn_start, turn_end, other_label in all_turns
                    if other_label != speaker_label
                    and turn_start <= midpoint <= turn_end
                }
            )
            text = str(word["text"])
            if (
                current is None
                or current["speaker_id"] != speaker["id"]
                or start - current["end"] > 1.2
                or current["overlapping_speakers"] != overlap_names
            ):
                current = {
                    "speaker_id": speaker["id"],
                    "speaker_name": speaker["display_name"],
                    "overlapping_speakers": overlap_names,
                    "start": start,
                    "end": end,
                    "text": text.strip(),
                }
                dialogue.append(current)
            else:
                current["text"] = f'{current["text"]}{text}'
                current["end"] = end

        if not dialogue:
            raise SpeakerDiarizationError(
                "Could not align transcribed words with detected speakers."
            )

        speakers = []
        seen_ids = set()
        for label in labels:
            speaker = profiles_by_label[label]
            if speaker["id"] in seen_ids:
                continue
            seen_ids.add(speaker["id"])
            speakers.append(dict(speaker))
        return dialogue, speakers
    except SpeakerDiarizationError:
        raise
    except Exception as error:
        raise SpeakerDiarizationError(
            "Could not separate speakers in this audio. Check that the recording "
            "is clear and the pyannote model is available."
        ) from error


def rename_speaker_profile(
    profile_id: str,
    display_name: str,
    profiles_path: Path,
) -> None:
    from app.services.speaker_profiles import update_profiles

    def rename(profiles):
        profile = next((item for item in profiles if item["id"] == profile_id), None)
        if profile is None:
            raise SpeakerProfileError("Speaker profile was not found.")
        profile["display_name"] = display_name
        return None, True

    update_profiles(profiles_path, rename)


def delete_speaker_profile(profile_id: str, profiles_path: Path) -> bool:
    from app.services.speaker_profiles import delete_profile

    return delete_profile(profile_id, profiles_path)
