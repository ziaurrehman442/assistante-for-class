export const LOCAL_SERVER_URL = (
  process.env.EXPO_PUBLIC_API_URL ?? "http://192.168.100.15:5000"
).replace(/\/+$/, "");

export const AUDIO_PROCESS_ENDPOINT = `${LOCAL_SERVER_URL}/api/process-audio`;

export const AUDIO_PROCESS_JOBS_ENDPOINT = `${AUDIO_PROCESS_ENDPOINT}/jobs`;

export const SPEAKER_PROFILES_ENDPOINT = `${LOCAL_SERVER_URL}/api/speaker-profiles`;

// Text-only path: the client already has the transcript, so only the notes are
// regenerated. This is what "reprocess" does after a prompt or model change,
// and it never re-uploads audio.
export const GENERATE_NOTES_ENDPOINT = `${LOCAL_SERVER_URL}/api/generate-notes`;
export const JOB_STATUS_ENDPOINT = `${LOCAL_SERVER_URL}/api/job-status`;
