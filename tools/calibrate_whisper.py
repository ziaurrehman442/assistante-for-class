"""Calibration harness for the lecture transcription pipeline.

Sweeps faster-whisper parameters over a labelled recording and scores each run
against a reference transcript with Word Error Rate, so a configuration is
chosen from measurements instead of intuition.

Two things this tool deliberately does NOT do:

* It never writes a transcript into storage. A score is not proof of
  correctness; see "Reference quality" below.
* It does not auto-promote a winning configuration to production. A sweep over
  one recording describes that recording. Promoting it globally changes every
  other lecture on the strength of one file.

Usage:
    python tools/calibrate_whisper.py \\
        --audio path/to/lecture.m4a \\
        --reference data/reference_transcript.txt \\
        --model medium --out calibration-report.json

The reference is read from a file so the benchmark stays reviewable and
versionable rather than buried in code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


CLEAN_FILTER = "highpass=f=200,lowpass=f=3000,loudnorm=I=-16:TP=-1.5:LRA=11"

BASE_PROMPT = (
    "This is a computer science lecture on Theory of Automata, Formal Languages, "
    "Computability, Decidability, Complexity, Alphabet, String, Word, "
    "Concatenation, and Recursive Definition spoken in Urdu and English."
)


@dataclass(frozen=True)
class WhisperSettings:
    """One point in the search space."""

    model: str
    beam_size: int
    temperature: float
    language: str | None
    min_silence_duration_ms: int
    initial_prompt: str

    def label(self) -> str:
        return (
            f"{self.model}/beam{self.beam_size}/temp{self.temperature}"
            f"/{self.language or 'auto'}/silence{self.min_silence_duration_ms}"
        )


# Arabic, Urdu and extended-Arabic blocks.
ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def script_of(text: str) -> str:
    """Classify the dominant script of a string."""
    counts = {"latin": 0, "arabic": 0}
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if any(low <= code <= high for low, high in ARABIC_RANGES):
            counts["arabic"] += 1
        elif char in "LATIN" or unicodedata.name(char, "").startswith("LATIN"):
            counts["latin"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "unknown"


def normalise(text: str) -> list[str]:
    """Lowercase and drop punctuation so scoring is not punc/whitespace noise."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", text)
    return [token for token in text.split() if token]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over token lists, computed row-wise."""
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def clean_audio(source: Path, destination: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af", CLEAN_FILTER,
            "-ar", "16000", "-ac", "1", str(destination),
        ],
        check=True,
        capture_output=True,
    )
    return destination


def transcribe(model, audio: Path, settings: WhisperSettings) -> str:
    segments, _ = model.transcribe(
        str(audio),
        language=settings.language,
        condition_on_previous_text=False,
        initial_prompt=settings.initial_prompt,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=settings.min_silence_duration_ms),
        beam_size=settings.beam_size,
        temperature=settings.temperature,
    )
    return " ".join(segment.text for segment in segments).strip()


def build_grid(args) -> list[WhisperSettings]:
    grid = []
    for beam, temperature, silence in itertools.product(
        args.beam_sizes, args.temperatures, args.silence_ms
    ):
        grid.append(
            WhisperSettings(
                model=args.model,
                beam_size=beam,
                temperature=temperature,
                language=args.language,
                min_silence_duration_ms=silence,
                initial_prompt=args.prompt or BASE_PROMPT,
            )
        )
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[5])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--silence-ms", type=int, nargs="+", default=[800])
    parser.add_argument("--language", default="ur")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reference_text = args.reference.read_text(encoding="utf-8").strip()
    reference_tokens = normalise(reference_text)
    if not reference_tokens:
        print("Reference file contains no comparable tokens.", flush=True)
        return 2

    reference_script = script_of(reference_text)
    print(f"reference: {len(reference_tokens)} tokens, script={reference_script}")
    if reference_script != "unknown":
        print(
            "NOTE: the reference is in a different script from most Whisper "
            "outputs for this language. Raw WER is dominated by script "
            "differences, not by recognition errors. Score with a transliterated "
            "reference before trusting these numbers.",
            flush=True,
        )

    from faster_whisper import WhisperModel

    workdir = Path(tempfile.mkdtemp(prefix="calibrate-"))
    cleaned = clean_audio(args.audio, workdir / "cleaned.wav")
    if args.max_seconds:
        trimmed = workdir / "trimmed.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(cleaned), "-t", str(args.max_seconds), str(trimmed)],
            check=True,
            capture_output=True,
        )
        cleaned = trimmed

    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4)
    results = []
    for settings in build_grid(args):
        started = time.time()
        hypothesis = transcribe(model, cleaned, settings)
        tokens = normalise(hypothesis)
        wer = word_error_rate(reference_tokens, tokens)
        row = {
            "settings": asdict(settings),
            "label": settings.label(),
            "wer": round(wer, 4),
            "hypothesis_tokens": len(tokens),
            "hypothesis_script": script_of(hypothesis),
            "elapsed_seconds": round(time.time() - started, 1),
            "transcript": hypothesis,
        }
        results.append(row)
        print(
            f"WER {wer:7.2%}  script={row['hypothesis_script']:6} "
            f"{row['elapsed_seconds']:6.0f}s  {settings.label()}",
            flush=True,
        )

    results.sort(key=lambda row: row["wer"])
    best = results[0]
    print(f"\nbest: WER {best['wer']:.2%}  {best['label']}")
    print(
        "This is a measurement for this recording only. Promote it manually "
        "after checking it against other lectures."
    )

    if args.out:
        args.out.write_text(
            json.dumps({"reference_script": reference_script, "results": results}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
