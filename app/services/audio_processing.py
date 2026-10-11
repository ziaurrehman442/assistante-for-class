import logging
from pathlib import Path

from app.services.note_generator import format_dialogue, generate_study_materials
from app.services.transcription import (
    SpeakerDiarizationError,
    TranscriptionError,
    diarize_and_label,
    transcribe_audio,
)


logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def process_audio_file(app, audio_path: Path) -> dict:
    with app.app_context():
        try:
            logger.info("Processing uploaded audio file")
            words = transcribe_audio(audio_path)
            if not words:
                message = "No speech was detected in this recording."
                logger.info(message)
                return {
                    "transcript": message,
                    "dialogue": [],
                    "speakers": [],
                    "detailed_class_notes": {
                        "main_topics": [message],
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

            dialogue, speakers = diarize_and_label(
                audio_path,
                words,
                token=app.config["HUGGINGFACE_TOKEN"],
                profiles_path=Path(app.config["SPEAKER_PROFILES_PATH"]),
                match_threshold=app.config["SPEAKER_MATCH_THRESHOLD"],
            )
            transcript = format_dialogue(dialogue)
            materials = generate_study_materials(
                transcript,
                ollama_url=app.config["OLLAMA_URL"],
                model=app.config["OLLAMA_MODEL"],
                timeout=app.config["OLLAMA_TIMEOUT_SECONDS"],
            )
            return {
                "transcript": transcript,
                "dialogue": dialogue,
                "speakers": speakers,
                **materials,
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
