import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path

from app.services.note_generator import format_dialogue, generate_study_materials
from app.services.transcription import (
    SpeakerDiarizationError,
    TranscriptionError,
    diarize_and_label,
    transcribe_audio,
)


logger = logging.getLogger(__name__)


def _refine_transcript_if_enabled(app, transcript: str) -> str:
    """Clean up the transcript wording, falling back to the original on any error.

    Refinement is a best-effort improvement: a network failure, a model error or
    a reply that changed the number of turns must never fail an otherwise good
    recording, so every failure is logged and the original transcript is used.
    """
    if not app.config.get("TRANSCRIPT_REFINER_ENABLED"):
        return transcript

    from app.services.transcript_refiner import TranscriptRefinementError, refine_transcript

    try:
        refined = refine_transcript(
            transcript,
            base_url=app.config["OPENCODE_BASE_URL"],
            user=app.config.get("OPENCODE_USER"),
            password=app.config.get("OPENCODE_PASSWORD"),
            model=app.config["TRANSCRIPT_REFINER_MODEL"],
            timeout=app.config["TRANSCRIPT_REFINER_TIMEOUT_SECONDS"],
        )
    except TranscriptRefinementError as error:
        logger.warning("Transcript refinement skipped: %s", error)
        return transcript

    logger.info("Refined transcript from %d to %d characters", len(transcript), len(refined))
    return refined


class AudioProcessingError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _empty_materials(message: str, include_in_topics: bool = False) -> dict[str, object]:
    """Placeholder study materials with the same keys the mobile tabs render."""
    return {
        "detailed_class_notes": {
            "main_topics": [message] if include_in_topics else [],
            "definitions": [],
            "teacher_explanations": [message],
            "qa": [],
        },
        "easy_notes": [message],
        "revision_quiz": {
            "mcqs": [],
            "conceptual_questions": [],
        },
    }


def _generate_materials_or_degrade(app, transcript: str):
    """Generate study materials, but never lose a finished transcript to them.

    Transcription is the expensive, irreplaceable part: it needs the recording,
    which is deleted once the job ends, and CPU minutes that cannot be recovered.
    If Ollama is unreachable, slow, or missing the model, returning an empty
    notes section plus a warning keeps that transcript in the user's hands
    instead of failing the whole job.
    """
    from app.services.note_generator import NoteGenerationError

    try:
        materials = generate_study_materials(
            transcript,
            ollama_url=app.config["OLLAMA_URL"],
            model=app.config["OLLAMA_MODEL"],
            timeout=app.config["OLLAMA_TIMEOUT_SECONDS"],
            availability_timeout=app.config["OLLAMA_AVAILABILITY_TIMEOUT_SECONDS"],
        )
    except NoteGenerationError as error:
        logger.warning("Study material generation degraded: %s", error)
        return _empty_materials("Study materials could not be generated."), [str(error)]

    return materials, []


def _cleaned_audio_for(app, audio_path: Path):
    """Clean the upload before transcription, falling back to the original.

    Returns (audio_path_to_use, cleanup) where cleanup removes any temporary
    file when the context exits.
    """
    if not app.config.get("AUDIO_CLEANING_ENABLED"):
        return audio_path, nullcontext()

    from app.services.audio_cleaning import (
        AudioCleaningError,
        clean_audio,
        detect_clipping,
    )

    if detect_clipping(audio_path):
        # Loudness normalisation cannot undo clipping; loudening a clipped
        # signal only makes the distortion louder.
        logger.warning(
            "Recording reaches full scale, so it is clipped. Loudness "
            "normalisation cannot repair those samples; re-record with the "
            "microphone further from any loud source."
        )

    cleaned_path = audio_path.with_name(f"{audio_path.stem}.cleaned.wav")
    try:
        clean_audio(
            audio_path,
            cleaned_path,
            highpass_hz=app.config["AUDIO_CLEANING_HIGHPASS_HZ"],
            lowpass_hz=app.config["AUDIO_CLEANING_LOWPASS_HZ"],
            target_lufs=app.config["AUDIO_CLEANING_TARGET_LUFS"],
            true_peak=app.config["AUDIO_CLEANING_TRUE_PEAK"],
            loudness_range=app.config["AUDIO_CLEANING_LOUDNESS_RANGE"],
            timeout=app.config["AUDIO_CLEANING_TIMEOUT_SECONDS"],
        )
    except AudioCleaningError as error:
        logger.warning("Audio cleaning skipped, using the original: %s", error)
        cleaned_path.unlink(missing_ok=True)
        return audio_path, nullcontext()

    logger.info("Using cleaned audio %s", cleaned_path.name)
    return cleaned_path, _remove_when_done(cleaned_path)


@contextmanager
def _remove_when_done(path: Path):
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def process_audio_file(app, audio_path: Path) -> dict:
    with app.app_context():
        try:
            logger.info("Processing uploaded audio file")
            cleaned_path, cleanup = _cleaned_audio_for(app, audio_path)
            with cleanup as audio_for_inference:
                words = transcribe_audio(audio_for_inference)
                if not words:
                    message = "No speech was detected in this recording."
                    logger.info(message)
                    return {
                        "transcript": message,
                        "dialogue": [],
                        "speakers": [],
                        **_empty_materials(message, include_in_topics=True),
                        "warnings": [],
                    }

                dialogue, speakers = diarize_and_label(
                    audio_for_inference,
                    words,
                    token=app.config["HUGGINGFACE_TOKEN"],
                    profiles_path=Path(app.config["SPEAKER_PROFILES_PATH"]),
                    match_threshold=app.config["SPEAKER_MATCH_THRESHOLD"],
                )
                transcript = format_dialogue(dialogue)
                transcript = _refine_transcript_if_enabled(app, transcript)
                materials, warnings = _generate_materials_or_degrade(app, transcript)
                return {
                    "transcript": transcript,
                    "dialogue": dialogue,
                    "speakers": speakers,
                    **materials,
                    # An empty list on a healthy run, so the response shape never
                    # changes. The mobile client ignores it, but a client that
                    # wants to surface the degraded case can check it.
                    "warnings": warnings,
                }
        except TranscriptionError as error:
            logger.warning("Audio transcription failed: %s", error)
            raise AudioProcessingError(str(error), 422) from error
        except SpeakerDiarizationError as error:
            logger.warning("Speaker diarization failed: %s", error)
            raise AudioProcessingError(str(error), 503) from error
        except Exception as error:
            from app.services.note_generator import NoteGenerationError

            if isinstance(error, NoteGenerationError):
                logger.warning("Study material generation failed: %s", error)
                raise AudioProcessingError(str(error), 502) from error
            raise
