import {
  AUDIO_PROCESS_ENDPOINT,
  AUDIO_PROCESS_JOBS_ENDPOINT,
} from "../config";
import NetInfo from "@react-native-community/netinfo";
import { File, UploadType } from "expo-file-system";
import { loadRecordings, saveRecordings } from "./recordingStore";
import { ClassRecording, RecordingSpeaker, SpeakerTurn, StudyMaterials } from "../types";
import { isObject, isSpeakerTurn, isStudyMaterials } from "./apiTypes";

let activeSync: Promise<void> | undefined;
const JOB_POLL_INTERVAL_MS = 3000;
const JOB_POLL_TIMEOUT_MS = 60 * 60 * 1000;
const JOB_REQUEST_TIMEOUT_MS = 15_000;
// How long to wait for the network before giving the job up entirely. Without
// this a device that reconnects after a long outage would abandon a job that the
// server is still working on.
const JOB_OFFLINE_GRACE_MS = 10 * 60 * 1000;

class AudioJobUnavailableError extends Error {}

const NO_SPEECH_MESSAGE = "No speech was detected in this recording.";

function isNoSpeechMessage(message: string): boolean {
  return /no speech was detected/i.test(message);
}

function createNoSpeechResult(): {
  transcript: string;
  dialogue: SpeakerTurn[];
  speakers: RecordingSpeaker[];
} & StudyMaterials {
  return {
    transcript: NO_SPEECH_MESSAGE,
    dialogue: [],
    speakers: [],
    detailed_class_notes: {
      main_topics: [NO_SPEECH_MESSAGE],
      definitions: [],
      teacher_explanations: [NO_SPEECH_MESSAGE],
      qa: [],
    },
    easy_notes: [NO_SPEECH_MESSAGE],
    revision_quiz: { mcqs: [], conceptual_questions: [] },
  };
}

function isRecordingSpeaker(value: unknown): value is RecordingSpeaker {
  return (
    isObject(value) &&
    typeof value.id === "string" &&
    typeof value.display_name === "string" &&
    typeof value.is_named === "boolean"
  );
}

function isSpeakerTurnArray(value: unknown): value is SpeakerTurn[] {
  return Array.isArray(value) && value.every(isSpeakerTurn);
}

function isRecordingSpeakerArray(value: unknown): value is RecordingSpeaker[] {
  return Array.isArray(value) && value.every(isRecordingSpeaker);
}

function isProcessAudioResponse(
  value: unknown,
): value is { transcript: string; dialogue: SpeakerTurn[]; speakers: RecordingSpeaker[] } & StudyMaterials {
  return (
    isObject(value) &&
    typeof value.transcript === "string" &&
    isSpeakerTurnArray(value.dialogue) &&
    isRecordingSpeakerArray(value.speakers) &&
    isStudyMaterials(value)
  );
}

function normalizeProcessAudioResponse(
  value: unknown,
): ({ transcript: string; dialogue: SpeakerTurn[]; speakers: RecordingSpeaker[] } & StudyMaterials) | null {
  if (isProcessAudioResponse(value)) {
    return value;
  }
  if (!isObject(value)) {
    return null;
  }
  const transcript = value.transcript;
  const dialogue = value.dialogue;
  const speakers = value.speakers;
  if (typeof transcript !== "string" || !isStudyMaterials(value)) {
    return null;
  }
  let normalizedDialogue: SpeakerTurn[] = [];
  if (dialogue !== undefined) {
    if (!isSpeakerTurnArray(dialogue)) {
      return null;
    }
    normalizedDialogue = dialogue;
  }
  let normalizedSpeakers: RecordingSpeaker[] = [];
  if (speakers !== undefined) {
    if (!isRecordingSpeakerArray(speakers)) {
      return null;
    }
    normalizedSpeakers = speakers;
  }
  return {
    transcript,
    dialogue: normalizedDialogue,
    speakers: normalizedSpeakers,
    detailed_class_notes: value.detailed_class_notes,
    easy_notes: value.easy_notes,
    revision_quiz: value.revision_quiz,
  };
}

function getErrorMessage(payload: unknown, status: number): string {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "error" in payload &&
    typeof payload.error === "string"
  ) {
    return payload.error;
  }
  return `The server returned HTTP ${status}.`;
}

function waitForNextPoll(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL_MS));
}

/** Progress is advisory: an older server simply omits it. */
function readProgress(payload: unknown): number | undefined {
  if (
    isObject(payload) &&
    typeof payload.progress === "number" &&
    Number.isFinite(payload.progress)
  ) {
    return Math.max(0, Math.min(100, Math.round(payload.progress)));
  }
  return undefined;
}

function readMessage(payload: unknown): string | undefined {
  if (isObject(payload) && typeof payload.message === "string" && payload.message.trim()) {
    return payload.message.trim();
  }
  return undefined;
}

/**
 * Connectivity is tracked by one long-lived subscription rather than a fetch
 * per poll: NetInfo does not expose current state synchronously, and polling a
 * 3-second loop through fetch() would add real overhead.
 */
let networkAvailable = true;
let networkWatcherStarted = false;

function ensureNetworkWatcher(): void {
  if (networkWatcherStarted) {
    return;
  }
  networkWatcherStarted = true;
  NetInfo.addEventListener((state) => {
    networkAvailable = state.isConnected !== false && state.isInternetReachable !== false;
  });
}

/** Resolves immediately when online, otherwise on the next reconnect. */
function waitForConnection(): Promise<void> {
  ensureNetworkWatcher();
  if (networkAvailable) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      if (state.isConnected === false || state.isInternetReachable === false) {
        // Still down. Keep this single listener until the device is actually
        // back, rather than churning a subscription per poll.
        return;
      }
      unsubscribe();
      resolve();
    });
  });
}

/** True for the transport failures that a dropped connection produces. */
function isTransientNetworkError(error: unknown): boolean {
  return (
    error instanceof TypeError ||
    (error instanceof Error && /network|fetch|abort|timeout/i.test(error.message))
  );
}

async function getAudioJobResult(
  jobId: string,
  recordings: ClassRecording[],
  recordingId: string,
): Promise<unknown> {
  const deadline = Date.now() + JOB_POLL_TIMEOUT_MS;
  let offlineSince: number | undefined;
  while (Date.now() < deadline) {
    // A dropped connection pauses the loop instead of throwing. The job keeps
    // running on the server and the same job id is reused when the device is
    // back, so no work is lost and the UI keeps its last known progress.
    await waitForConnection();
    if (Date.now() >= deadline) {
      break;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), JOB_REQUEST_TIMEOUT_MS);
    let response: Response;
    try {
      response = await fetch(
        `${AUDIO_PROCESS_JOBS_ENDPOINT}/${encodeURIComponent(jobId)}`,
        { signal: controller.signal },
      );
      offlineSince = undefined;
    } catch (error) {
      if (!isTransientNetworkError(error) || Date.now() >= deadline) {
        throw error;
      }
      offlineSince ??= Date.now();
      const latestRecordings = await loadRecordings();
      await updateRecording(latestRecordings, recordingId, {
        status: "SYNCING",
        processingJobId: jobId,
        syncMessage: "Waiting for a connection…",
      });
      if (Date.now() - offlineSince > JOB_OFFLINE_GRACE_MS) {
        throw new Error(
          "The connection was lost for too long. The job is saved and will resume when sync runs again.",
        );
      }
      await waitForConnection();
      continue;
    } finally {
      clearTimeout(timeout);
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`The job status endpoint returned invalid JSON (HTTP ${response.status}).`);
    }

    if (response.status !== 200 && response.status !== 202) {
      if (response.status === 404) {
        throw new AudioJobUnavailableError(getErrorMessage(payload, response.status));
      }
      throw new Error(getErrorMessage(payload, response.status));
    }
    if (!isObject(payload) || typeof payload.status !== "string") {
      throw new Error("The job status endpoint returned an invalid response.");
    }
    if (payload.status === "completed") {
      return payload.result;
    }
    if (payload.status === "failed") {
      if (typeof payload.error === "string" && isNoSpeechMessage(payload.error)) {
        return createNoSpeechResult();
      }
      throw new AudioJobUnavailableError(
        typeof payload.error === "string"
          ? payload.error
          : "Audio processing failed.",
      );
    }
    if (payload.status !== "queued" && payload.status !== "processing") {
      throw new Error(`Unknown audio job status: ${payload.status}`);
    }

    const latestRecordings = await loadRecordings();
    // The job id is re-saved on every poll, so an app kill, a route change or a
    // dropped connection resumes from this exact job rather than re-uploading.
    // Progress rides along so the list can show a live percentage.
    await updateRecording(latestRecordings, recordingId, {
      status: "SYNCING",
      processingJobId: jobId,
      syncProgress: readProgress(payload),
      syncMessage: readMessage(payload),
    });
    await waitForNextPoll();
  }
  throw new Error(
    "Audio processing is taking longer than expected. The job is saved and will resume when sync runs again.",
  );
}

async function updateRecording(
  recordings: ClassRecording[],
  id: string,
  update: Partial<ClassRecording>,
): Promise<ClassRecording[]> {
  const updated = recordings.map((recording) =>
    recording.id === id ? { ...recording, ...update } : recording,
  );
  await saveRecordings(updated);
  return updated;
}

async function syncQueue(): Promise<void> {
  let recordings = await loadRecordings();
  for (const recording of recordings) {
    if (
      recording.status !== "PENDING_SYNC" &&
      !(recording.status === "SYNCING" && recording.processingJobId)
    ) {
      continue;
    }
    let processingJobId = recording.processingJobId;

    if (recording.status !== "SYNCING") {
      recordings = await updateRecording(recordings, recording.id, {
        status: "SYNCING",
        lastError: undefined,
      });
    }

    try {
      let latestRecordings = recordings;
      let payload: unknown;
      if (!processingJobId) {
        const response = await new File(recording.uri).upload(AUDIO_PROCESS_ENDPOINT, {
          httpMethod: "POST",
          uploadType: UploadType.MULTIPART,
          fieldName: "file",
          mimeType: "audio/mp4",
          // The attempt counter keeps the server's two-hour idempotency cache
          // from handing back the previous job when a recording is deliberately
          // re-processed after a bad transcript.
          headers: {
            "X-Client-Recording-Id": `${recording.id}:${recording.attempt ?? 1}`,
          },
        });
        let queued: unknown;
        try {
          queued = JSON.parse(response.body);
        } catch {
          throw new Error(`The upload endpoint returned invalid JSON (HTTP ${response.status}).`);
        }
        if (response.status === 200 && normalizeProcessAudioResponse(queued)) {
          payload = normalizeProcessAudioResponse(queued);
        } else if (
          response.status === 200 &&
          isObject(queued) &&
          queued.status === "completed" &&
          normalizeProcessAudioResponse(queued.result)
        ) {
          payload = normalizeProcessAudioResponse(queued.result);
        } else if (
          response.status === 422 &&
          isObject(queued) &&
          typeof queued.error === "string" &&
          isNoSpeechMessage(queued.error)
        ) {
          payload = createNoSpeechResult();
        } else if (
          response.status === 200 &&
          isObject(queued) &&
          typeof queued.status === "string" &&
          (queued.status === "queued" || queued.status === "processing") &&
          typeof queued.job_id === "string"
        ) {
          processingJobId = queued.job_id;
          latestRecordings = await updateRecording(recordings, recording.id, {
            status: "SYNCING",
            processingJobId,
          });
          payload = await getAudioJobResult(
            processingJobId,
            latestRecordings,
            recording.id,
          );
        } else if (
          response.status !== 202 ||
          !isObject(queued) ||
          typeof queued.job_id !== "string"
        ) {
          const serverMessage = getErrorMessage(queued, response.status);
          throw new Error(
            response.status >= 200 && response.status < 300
              ? "The server returned HTTP 200 but the response did not contain a complete transcript or audio job ID. Restart the updated backend and retry."
              : serverMessage,
          );
        } else {
          processingJobId = queued.job_id;
          latestRecordings = await updateRecording(recordings, recording.id, {
            status: "SYNCING",
            processingJobId,
          });
          payload = await getAudioJobResult(
            processingJobId,
            latestRecordings,
            recording.id,
          );
        }
      } else {
        payload = await getAudioJobResult(
          processingJobId,
          latestRecordings,
          recording.id,
        );
      }

      const result = normalizeProcessAudioResponse(payload);
      if (!result) {
        throw new Error("The server response did not contain the expected study materials.");
      }

      recordings = await updateRecording(latestRecordings, recording.id, {
        status: "SYNCED",
        processingJobId: undefined,
        syncProgress: undefined,
        syncMessage: undefined,
        lastError: undefined,
        transcript: result.transcript,
        dialogue: result.dialogue,
        speakers: result.speakers,
        materials: result,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not sync this recording.";
      const latestRecordings = await loadRecordings();
      recordings = await updateRecording(latestRecordings, recording.id, {
        status: "PENDING_SYNC",
        processingJobId:
          error instanceof AudioJobUnavailableError ? undefined : processingJobId,
        lastError: message,
      });
      console.warn("Audio sync failed:", message);
      break;
    }
  }
}

export function syncPendingRecordings(): Promise<void> {
  if (!activeSync) {
    activeSync = syncQueue().finally(() => {
      activeSync = undefined;
    });
  }
  return activeSync;
}
