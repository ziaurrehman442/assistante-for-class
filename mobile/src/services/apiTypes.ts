import { SpeakerTurn, StudyMaterials } from "../types";

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "string")
  );
}

export function isStudyMaterials(value: unknown): value is StudyMaterials {
  if (!isObject(value)) {
    return false;
  }
  const notes = value.detailed_class_notes;
  const quiz = value.revision_quiz;
  if (!isObject(notes) || !isObject(quiz)) {
    return false;
  }
  const validQa =
    Array.isArray(notes.qa) &&
    notes.qa.every(
      (item) =>
        isObject(item) &&
        typeof item.question === "string" &&
        typeof item.answer === "string",
    );
  const validMcqs =
    Array.isArray(quiz.mcqs) &&
    quiz.mcqs.every(
      (item) =>
        isObject(item) &&
        typeof item.question === "string" &&
        isStringArray(item.options) &&
        typeof item.answer === "string",
    );
  return (
    isStringArray(notes.main_topics) &&
    isStringArray(notes.definitions) &&
    isStringArray(notes.teacher_explanations) &&
    validQa &&
    isStringArray(value.easy_notes) &&
    validMcqs &&
    isStringArray(quiz.conceptual_questions)
  );
}

export function isSpeakerTurn(value: unknown): value is SpeakerTurn {
  return (
    isObject(value) &&
    typeof value.speaker_id === "string" &&
    typeof value.speaker_name === "string" &&
    typeof value.text === "string" &&
    typeof value.start === "number" &&
    typeof value.end === "number" &&
    (value.overlapping_speakers === undefined ||
      isStringArray(value.overlapping_speakers))
  );
}

export function isRenamedSpeakerResponse(
  value: unknown,
): value is Record<string, unknown> &
  StudyMaterials & { transcript: string; dialogue: SpeakerTurn[] } {
  return (
    isObject(value) &&
    typeof value.transcript === "string" &&
    Array.isArray(value.dialogue) &&
    value.dialogue.every(isSpeakerTurn) &&
    isStudyMaterials(value)
  );
}
