export const LOCAL_SERVER_URL = (
  process.env.EXPO_PUBLIC_API_URL ?? "http://192.168.100.15:5000"
).replace(/\/+$/, "");

export const AUDIO_PROCESS_ENDPOINT = `${LOCAL_SERVER_URL}/api/process-audio`;

export const AUDIO_PROCESS_JOBS_ENDPOINT = `${AUDIO_PROCESS_ENDPOINT}/jobs`;

export const SPEAKER_PROFILES_ENDPOINT = `${LOCAL_SERVER_URL}/api/speaker-profiles`;
