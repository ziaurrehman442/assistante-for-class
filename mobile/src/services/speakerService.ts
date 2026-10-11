import { SPEAKER_PROFILES_ENDPOINT } from "../config";
import { isObject, isRenamedSpeakerResponse } from "./apiTypes";
import { SpeakerTurn, StudyMaterials } from "../types";

export type RenamedSpeakerResult = {
  transcript: string;
  dialogue: SpeakerTurn[];
  materials: StudyMaterials;
};

function getErrorMessage(payload: unknown, status: number): string {
  if (
    isObject(payload) &&
    typeof payload.error === "string"
  ) {
    return payload.error;
  }
  return `The server returned HTTP ${status}.`;
}

export async function renameSpeakerAndRegenerate(
  speakerId: string,
  displayName: string,
  dialogue: SpeakerTurn[],
): Promise<RenamedSpeakerResult> {
  const response = await fetch(
    `${SPEAKER_PROFILES_ENDPOINT}/${encodeURIComponent(speakerId)}/rename`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName, dialogue }),
    },
  );
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The server returned invalid JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, response.status));
  }
  if (!isRenamedSpeakerResponse(payload)) {
    throw new Error("The server returned invalid regenerated study materials.");
  }

  return {
    transcript: payload.transcript,
    dialogue: payload.dialogue,
    materials: payload,
  };
}

export async function forgetSpeakerProfile(speakerId: string): Promise<void> {
  const response = await fetch(
    `${SPEAKER_PROFILES_ENDPOINT}/${encodeURIComponent(speakerId)}`,
    { method: "DELETE" },
  );
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The server returned invalid JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, response.status));
  }
}
