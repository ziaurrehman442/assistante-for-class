"""Optional transcript cleanup through a text model.

Speech-to-text output for Urdu/English code-switching is often littered with
mixed script: Urdu words leaking into Latin characters, English words dropped
or mangled, and stray punctuation. A text model repairs that far better than
the transcriber can.

This step only rewrites wording. It must never change the structure of a
recording: speaker labels, timestamps and turn boundaries are parsed back out of
the refined text, so anything the model does to those lines would corrupt the
dialogue. Every failure path falls back to the original transcript, because a
refiner that silently invents content is worse than no refiner at all.
"""

import logging
import re

import requests


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://opencode.nexa.website"
DEFAULT_MODEL = "space-bunny-free"
_REFINE_TIMEOUT_SECONDS = 120

_TURN_PATTERN = re.compile(r"^\[(\d{2}:\d{2})\]\s*(.*?):\s*(.*)$")

_PROMPT = (
    "You are correcting the output of an Urdu/English speech-to-text system. "
    "Fix only the writing, never the content.\n"
    "Rules:\n"
    "1. Write Urdu words in Urdu script and English words in Latin script.\n"
    "2. Fix words that were written in the wrong script.\n"
    "3. Remove stutters, repeated filler and obvious speech-recognition noise.\n"
    "4. Do NOT translate, summarise, explain or add anything that was not said.\n"
    "5. Do NOT merge, split, reorder or drop any line.\n"
    "6. Keep every timestamp and speaker label exactly as given.\n"
    "7. Output only the corrected transcript, with no preamble or commentary.\n\n"
    "Each line is formatted '[mm:ss] Speaker: text'. Keep that exact shape.\n\n"
    "Transcript:\n"
)


class TranscriptRefinementError(Exception):
    pass


def split_turns(transcript: str) -> list[tuple[str, str, str]]:
    """Split a formatted transcript into (timestamp, speaker, text) turns."""
    turns = []
    for line in transcript.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _TURN_PATTERN.match(stripped)
        if match:
            turns.append((match.group(1), match.group(2), match.group(3)))
        else:
            turns.append(("", "", stripped))
    return turns


def join_turns(turns: list[tuple[str, str, str]]) -> str:
    lines = []
    for timestamp, speaker, text in turns:
        if timestamp:
            lines.append(f"[{timestamp}] {speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def build_refinement_request(transcript: str) -> str:
    return _PROMPT + transcript


def parse_refinement_response(
    response: str,
    original_turns: list[tuple[str, str, str]],
) -> str:
    """Re-apply the original timestamps and speakers to the refined wording.

    The model is asked to echo the timestamps and labels, but it does not always
    do so reliably. Matching by position and restoring the originals means a
    malformed or reordered reply cannot corrupt speaker attribution.
    """
    refined_lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not refined_lines:
        raise TranscriptRefinementError("The refinement response was empty.")

    parsed = []
    for line in refined_lines:
        match = _TURN_PATTERN.match(line)
        parsed.append(match.group(3).strip() if match else line)

    if len(parsed) != len(original_turns):
        # A different number of lines means the model split, merged or dropped
        # turns. Keeping the original text is the safe outcome.
        raise TranscriptRefinementError(
            f"Refinement changed the number of turns ({len(original_turns)} "
            f"to {len(parsed)})."
        )

    rebuilt = []
    for (timestamp, speaker, _original_text), new_text in zip(original_turns, parsed):
        if not new_text:
            raise TranscriptRefinementError("Refinement produced an empty turn.")
        rebuilt.append((timestamp, speaker, new_text))
    return join_turns(rebuilt)


def refine_transcript(
    transcript: str,
    base_url: str = DEFAULT_BASE_URL,
    user: str | None = None,
    password: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = _REFINE_TIMEOUT_SECONDS,
) -> str:
    """Return a cleaned transcript, or raise TranscriptRefinementError."""
    if not transcript.strip():
        return transcript
    if not user or not password:
        raise TranscriptRefinementError(
            "Transcript refinement needs OPENCODE_USER and OPENCODE_PASSWORD."
        )

    turns = split_turns(transcript)
    if not turns:
        return transcript

    session_url = f"{base_url.rstrip('/')}/session"
    message_url = None
    try:
        session_response = requests.post(
            session_url,
            json={"title": "transcript-refinement"},
            auth=(user, password),
            timeout=30,
        )
        session_response.raise_for_status()
        session_id = session_response.json().get("id")
        if not session_id:
            raise TranscriptRefinementError("Could not open a refinement session.")
        message_url = f"{session_url}/{session_id}/message"

        message_response = requests.post(
            message_url,
            json={
                "model": {"providerID": "opencode", "modelID": model},
                "parts": [
                    {
                        "type": "text",
                        "text": build_refinement_request(transcript),
                    }
                ],
            },
            auth=(user, password),
            timeout=timeout,
        )
        message_response.raise_for_status()
        payload = message_response.json()

        error = (payload.get("info") or {}).get("error")
        if error:
            raise TranscriptRefinementError(
                f"The refinement model returned an error: {error.get('data', {}).get('message', error)}"
            )

        refined_text = "\n".join(
            part.get("text", "")
            for part in payload.get("parts", [])
            if part.get("type") == "text"
        )
        if not refined_text.strip():
            raise TranscriptRefinementError("The refinement model returned no text.")

        return parse_refinement_response(refined_text, turns)
    except requests.RequestException as error:
        raise TranscriptRefinementError(
            f"Could not reach the refinement service: {error}"
        ) from error
    finally:
        if message_url:
            try:
                requests.delete(
                    message_url.rsplit("/message", 1)[0],
                    auth=(user, password),
                    timeout=15,
                )
            except requests.RequestException:
                logger.debug("Could not close the refinement session", exc_info=True)