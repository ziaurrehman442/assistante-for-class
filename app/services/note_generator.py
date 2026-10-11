import json

import requests


class NoteGenerationError(Exception):
    pass


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
) -> dict[str, object]:
    prompt = (
        "Turn the following class transcript into useful study materials. "
        "The class may code-switch between spoken Urdu and English within a "
        "sentence. Understand both languages; do not mistake Urdu words for "
        "noise. Do not translate the transcript. Write study materials in clear "
        "English, retaining important Urdu terms in Urdu script with a short "
        "English meaning when helpful. Keep all explanations faithful to the "
        "transcript and do not invent facts. "
        "Write detailed class notes with main topics, definitions, teacher "
        "explanations, and question-and-answer pairs; easy notes as concise "
        "bullets suitable for a five-minute revision; and a revision quiz with "
        "exactly five multiple-choice questions (four options and the correct "
        "answer for each) plus exactly three short conceptual questions. "
        "Treat speaker names as labels, not identities inferred from the audio. "
        "Use an explicitly assigned Teacher label when identifying instructor "
        "explanations, questions, and answers. Do not invent or reassign roles. "
        "Return only data matching the requested JSON schema.\n\n"
        f"CLASS DIALOGUE:\n{transcript}"
    )

    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": STUDY_MATERIAL_SCHEMA,
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
            raise NoteGenerationError(
                "Ollama returned study materials that do not match the required structure."
            )
        return parsed
    except NoteGenerationError:
        raise
    except (requests.RequestException, ValueError, TypeError) as error:
        raise NoteGenerationError(
            "Could not generate study materials. Confirm Ollama is running locally "
            "and that the configured model is available."
        ) from error
