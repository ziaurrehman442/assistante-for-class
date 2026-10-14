import json
import logging

import requests


logger = logging.getLogger(__name__)


class NoteGenerationError(Exception):
    pass


def model_missing_message(model: str) -> str:
    """The actionable message shown when Ollama does not have the model."""
    return (
        f"Ollama model {model} is not loaded. "
        f"Run 'ollama pull {model}' in your terminal."
    )


def _split_model_name(value: str) -> tuple[str, str]:
    base, _, tag = value.strip().partition(":")
    return base, tag or "latest"


def _installed_model_names(ollama_url: str, timeout: int) -> list[str]:
    try:
        response = requests.get(
            f"{ollama_url.rstrip('/')}/api/tags",
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise NoteGenerationError(
            f"Ollama is not reachable at {ollama_url}. Start it with 'ollama serve' "
            "and check that OLLAMA_URL points at the right host and port."
        ) from error

    if response.status_code != 200:
        raise NoteGenerationError(
            f"Ollama answered /api/tags with HTTP {response.status_code}, so its "
            "state could not be checked."
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise NoteGenerationError(
            "Ollama returned an unreadable response from /api/tags."
        ) from error

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise NoteGenerationError(
            "Ollama returned an unexpected response from /api/tags."
        )
    return [
        entry["name"]
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]


def check_ollama_available(ollama_url: str, model: str, timeout: int) -> None:
    """Fail fast, with an actionable message, before a long generation starts.

    Ollama stays up and answers happily while a model it does not have is
    requested, and the failure only surfaces as a bare 404 several minutes into
    a generation. Checking first turns that into an immediate, specific error.
    """
    if not any(
        _split_model_name(name) == _split_model_name(model)
        for name in _installed_model_names(ollama_url, timeout)
    ):
        raise NoteGenerationError(model_missing_message(model))


def _ollama_error_detail(response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:300]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"][:300]
    return ""


def format_dialogue(dialogue: list[dict]) -> str:
    lines = []
    for turn in dialogue:
        minutes, seconds = divmod(int(float(turn["start"])), 60)
        overlap = turn.get("overlapping_speakers")
        overlap_note = (
            f" (overlapping with {', '.join(overlap)})"
            if isinstance(overlap, list) and overlap
            else ""
        )
        lines.append(
            f"[{minutes:02d}:{seconds:02d}] {turn['speaker_name']}{overlap_note}: "
            f"{turn['text'].strip()}"
        )
    return "\n".join(lines)


def chunk_transcript(transcript: str, chunk_chars: int, min_chars: int = 200) -> list[str]:
    """Split a transcript into chunks that each fit the model's context.

    A long lecture does not fit in one prompt. Ollama truncates whatever exceeds
    the context window, so the tail of a 3-hour lecture would be silently ignored
    and the model would still be asked to describe the whole thing.

    Splits on line boundaries, because each line here is one speaker turn and
    cutting mid-turn produces a fragment that reads as broken speech.
    """
    text = transcript.strip()
    if not text:
        return []

    lines = text.splitlines() or [text]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1
        if current and current_length + line_length > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += line_length

    if current:
        tail = "\n".join(current)
        # Never emit a stub that would waste a generation call.
        if chunks and len(tail) < min_chars:
            chunks[-1] = f"{chunks[-1]}\n{tail}"
        else:
            chunks.append(tail)

    return chunks


def merge_materials(per_chunk: list[dict[str, object]], max_mcqs: int) -> dict[str, object]:
    """Combine per-chunk materials into one set, keeping the schema intact.

    The mobile client renders a fixed set of keys, so the merged result has to
    match STUDY_MATERIAL_SCHEMA exactly or the client rejects it.
    """
    topics: list[str] = []
    definitions: list[str] = []
    explanations: list[str] = []
    qa: list[dict[str, str]] = []
    easy_notes: list[str] = []
    mcqs: list[dict] = []
    conceptual: list[str] = []

    for index, materials in enumerate(per_chunk, start=1):
        notes = materials.get("detailed_class_notes") or {}
        quiz = materials.get("revision_quiz") or {}

        label = "" if len(per_chunk) == 1 else f"Part {index}: "
        topics.extend(f"{label}{item}" for item in notes.get("main_topics") or [])
        definitions.extend(f"{label}{item}" for item in notes.get("definitions") or [])
        explanations.extend(
            f"{label}{item}" for item in notes.get("teacher_explanations") or []
        )
        qa.extend(notes.get("qa") or [])
        easy_notes.extend(f"{label}{item}" for item in materials.get("easy_notes") or [])

        # Taking five MCQs from every chunk would produce fifty for one lecture,
        # so the quiz is capped and drawn from the earliest chunks.
        for item in quiz.get("mcqs") or []:
            if len(mcqs) >= max_mcqs:
                break
            mcqs.append(item)
        if len(conceptual) < 3:
            conceptual.extend((quiz.get("conceptual_questions") or [])[: 3 - len(conceptual)])

    return {
        "detailed_class_notes": {
            "main_topics": topics,
            "definitions": definitions,
            "teacher_explanations": explanations,
            "qa": qa,
        },
        "easy_notes": easy_notes,
        "revision_quiz": {"mcqs": mcqs, "conceptual_questions": conceptual},
    }


def _is_string_list(value, expected_length=None):
    return (
        isinstance(value, list)
        and (expected_length is None or len(value) == expected_length)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _validate_materials(materials):
    required_sections = {"detailed_class_notes", "easy_notes", "revision_quiz"}
    if set(materials) != required_sections:
        return False

    notes = materials["detailed_class_notes"]
    quiz = materials["revision_quiz"]
    if not isinstance(notes, dict) or set(notes) != {
        "main_topics",
        "definitions",
        "teacher_explanations",
        "qa",
    }:
        return False
    if not all(
        _is_string_list(notes[key])
        for key in ("main_topics", "definitions", "teacher_explanations")
    ):
        return False
    if not isinstance(notes["qa"], list) or not all(
        isinstance(item, dict)
        and set(item) == {"question", "answer"}
        and all(isinstance(item[key], str) and item[key].strip() for key in item)
        for item in notes["qa"]
    ):
        return False
    if not _is_string_list(materials["easy_notes"]):
        return False
    if not isinstance(quiz, dict) or set(quiz) != {
        "mcqs",
        "conceptual_questions",
    }:
        return False
    if not _is_string_list(quiz["conceptual_questions"], expected_length=3):
        return False

    mcqs = quiz["mcqs"]
    return isinstance(mcqs, list) and len(mcqs) == 5 and all(
        isinstance(item, dict)
        and set(item) == {"question", "options", "answer"}
        and isinstance(item["question"], str)
        and item["question"].strip()
        and _is_string_list(item["options"], expected_length=4)
        and isinstance(item["answer"], str)
        and item["answer"] in item["options"]
        for item in mcqs
    )


STUDY_MATERIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "detailed_class_notes": {
            "type": "object",
            "properties": {
                "main_topics": {"type": "array", "items": {"type": "string"}},
                "definitions": {"type": "array", "items": {"type": "string"}},
                "teacher_explanations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "qa": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                },
            },
            "required": [
                "main_topics",
                "definitions",
                "teacher_explanations",
                "qa",
            ],
        },
        "easy_notes": {"type": "array", "items": {"type": "string"}},
        "revision_quiz": {
            "type": "object",
            "properties": {
                "mcqs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "options", "answer"],
                    },
                    "minItems": 5,
                    "maxItems": 5,
                },
                "conceptual_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["mcqs", "conceptual_questions"],
        },
    },
    "required": ["detailed_class_notes", "easy_notes", "revision_quiz"],
}


def generate_study_materials(
    transcript: str,
    ollama_url: str,
    model: str,
    timeout: int,
    max_attempts: int = 3,
    availability_timeout: int = 5,
    num_ctx: int = 8192,
) -> dict[str, object]:
    check_ollama_available(ollama_url, model, availability_timeout)

    prompt = (
        "Turn the following class transcript into useful study materials. "
        "The class may code-switch between spoken Urdu and English within a "
        "sentence. Understand both languages; do not mistake Urdu words for "
        "noise. Do not translate the transcript. Write study materials in clear "
        "English, retaining important Urdu terms in Urdu script with a short "
        "English meaning when helpful. Keep all explanations faithful to the "
        "transcript and do not invent facts. "
        # The response keys below are fixed by STUDY_MATERIAL_SCHEMA and by the
        # mobile client's tabs, so depth is requested through the wording of each
        # section rather than by adding new fields.
        "Write detailed class notes that go well beyond a summary: list the main "
        "topics in order; give every technical term, definition, model, formula, "
        "rule, and named theorem its own entry in definitions, written precisely "
        "enough to be revised from on its own; expand teacher explanations into "
        "step-by-step reasoning, the sequence of steps or workflow as it was "
        "presented, the principles and rules behind it, worked examples, and the "
        "key takeaways a student must remember; and cover the teacher's questions "
        "and answers explicitly in question-and-answer pairs. "
        "Write easy notes as a concise bullet checklist a student can read in "
        "five minutes before an exam, covering the core ideas as short bullets "
        "and closing with a plain-language glossary of the key terms. "
        "Write a revision quiz with exactly five multiple-choice questions, each "
        "with exactly four options and exactly three short conceptual questions. "
        "Make every question rigorous and exam-style rather than trivia. The "
        "answer field of each multiple-choice question is the answer key entry: "
        "it must be copied verbatim from that question's own four options and must "
        "be the single correct choice. "
        "Treat speaker names as labels, not identities inferred from the audio. "
        "Use an explicitly assigned Teacher label when identifying instructor "
        "explanations, questions, and answers. Do not invent or reassign roles. "
        "Return only data matching the requested JSON schema.\n\n"
        f"CLASS DIALOGUE:\n{transcript}"
    )

    last_error: NoteGenerationError | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": STUDY_MATERIAL_SCHEMA,
                    # num_ctx defaults to 4096, which silently truncates any
                    # real lecture. Sizing it to hold a chunk plus the prompt and
                    # the reply is what stops generation running past the client
                    # timeout and wasting CPU on a truncated lecture.
                    "options": {"num_ctx": num_ctx},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise NoteGenerationError("Ollama returned an invalid response.")
            materials = result.get("response")
            if not isinstance(materials, str) or not materials.strip():
                raise NoteGenerationError("Ollama returned an empty response.")
            parsed = json.loads(materials)
            if not isinstance(parsed, dict) or not _validate_materials(parsed):
                # A small model following a strict schema is not reliable on the
                # first try, so a retry is far cheaper than failing the whole
                # recording after transcription has already succeeded.
                raise NoteGenerationError(
                    "Ollama returned study materials that do not match the "
                    "required structure."
                )
            return parsed
        except json.JSONDecodeError as error:
            last_error = NoteGenerationError(
                "Ollama returned a response that was not valid JSON."
            )
            logger.warning(
                "Study material attempt %d/%d returned invalid JSON: %s",
                attempt,
                max_attempts,
                error,
            )
        except NoteGenerationError as error:
            last_error = error
            logger.warning(
                "Study material attempt %d/%d failed: %s",
                attempt,
                max_attempts,
                error,
            )
        except requests.Timeout as error:
            # The most common failure on a small CPU server: generation is slow
            # but healthy, and only the configured timeout decides it failed.
            raise NoteGenerationError(
                f"Ollama was still generating after {timeout} seconds. Raise "
                "OLLAMA_TIMEOUT_SECONDS, or use a smaller model, so the "
                "recording is not discarded after transcription has already "
                "succeeded."
            ) from error
        except requests.HTTPError as error:
            response = error.response
            status = response.status_code if response is not None else "unknown"
            detail = _ollama_error_detail(response) if response is not None else ""
            # Ollama reports an unknown model as 404, and as 400 on some builds.
            if status in (400, 404) and "not found" in detail.lower():
                raise NoteGenerationError(model_missing_message(model)) from error
            raise NoteGenerationError(
                f"Ollama returned HTTP {status} while generating study materials"
                + (f": {detail}" if detail else ".")
            ) from error
        except requests.RequestException as error:
            raise NoteGenerationError(
                f"Ollama became unreachable while generating study materials "
                f"({type(error).__name__}). Confirm 'ollama serve' is still "
                "running and the server has not run out of memory."
            ) from error
        except (ValueError, TypeError) as error:
            # Transport-level problems will not improve by asking again.
            raise NoteGenerationError(
                "Could not generate study materials. Confirm Ollama is running "
                "locally and that the configured model is available."
            ) from error

    raise last_error or NoteGenerationError("Could not generate study materials.")


def generate_notes_for_transcript(
    transcript: str,
    ollama_url: str,
    model: str,
    timeout: int,
    chunk_chars: int = 6000,
    max_mcqs: int = 5,
    progress=None,
) -> dict[str, object]:
    """Generate study materials for a whole lecture, one chunk at a time.

    Used by the text-only endpoint, where the client sends a transcript that can
    be far longer than one model context. Each chunk is generated and validated
    independently so one bad chunk cannot discard the others.
    """
    chunks = chunk_transcript(transcript, chunk_chars)
    if not chunks:
        raise NoteGenerationError("The transcript was empty.")

    check_ollama_available(ollama_url, model, 5)

    logger.info("Generating notes for %d transcript chunk(s)", len(chunks))
    per_chunk: list[dict[str, object]] = []
    failures: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if progress is not None:
            percent = int((index - 1) / len(chunks) * 100)
            progress(percent, f"Generating notes for part {index} of {len(chunks)}...")
        try:
            per_chunk.append(
                generate_study_materials(
                    chunk,
                    ollama_url=ollama_url,
                    model=model,
                    timeout=timeout,
                    num_ctx=max(8192, chunk_chars),
                )
            )
        except NoteGenerationError as error:
            # Keep going: a chunk that times out should cost that chunk's notes,
            # not the entire lecture's.
            logger.warning("Chunk %d/%d failed: %s", index, len(chunks), error)
            failures.append(f"Part {index}: {error}")

    if not per_chunk:
        raise NoteGenerationError(
            "Could not generate study materials for any part of this lecture. "
            + (failures[0] if failures else "")
        )

    if progress is not None:
        progress(95, "Merging the generated notes...")

    merged = merge_materials(per_chunk, max_mcqs)
    if failures:
        merged["warnings"] = failures
    return merged
