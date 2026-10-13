# Local Class Notes API

A local Flask backend that transcribes class recordings with Faster-Whisper and
generates notes and revision questions with Ollama.

## Project layout

```text
app/
  routes/       HTTP endpoints and request validation
  services/     Transcription and study-material generation
  config.py     Environment-based runtime configuration
run.py          Local development entry point
mobile/         Expo React Native offline-first recorder app
tests/          API tests using Python's standard unittest framework
uploads/        Temporary audio uploads (removed after each request)
```

## Requirements

- Python 3.10+
- FFmpeg available on `PATH` for audio formats supported by Faster-Whisper.
- A Hugging Face account and access token for local speaker diarization.
- Ollama running locally, with the `qwen2.5:3b` model available:

  ```powershell
  ollama pull qwen2.5:3b
  ```

Accept the user conditions for
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
while signed in to Hugging Face, create a read token, and place it in a local
`.env` file (never commit this file):

```powershell
Copy-Item .env.example .env
# Edit .env and replace the HF_TOKEN value with your Hugging Face read token.
```

Install dependencies and start the server:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

The first transcription request downloads the Faster-Whisper `small` model and
the Pyannote diarization model. Both run locally on CPU; the Hugging Face token
is used to access/download the gated Pyannote model and is not sent with audio.
Transcription automatically detects the language and is prompted to preserve
English-Urdu code-switching, using Urdu script for Urdu words and Latin script
for English words. Generated notes are in clear English while retaining
important Urdu terms with short meanings when useful.
Uploads are cleaned with FFmpeg before transcription: the band speech occupies
is trimmed, loudness is normalised to EBU R128, and the result is resampled to
16 kHz mono. Configure this with the `AUDIO_CLEANING_*` values, including
`AUDIO_CLEANING_ENABLED=false` to transcribe the upload untouched.
Transcription accuracy for quiet or distant speech is tuned with
`WHISPER_BEAM_SIZE` (higher resolves more soft words, more CPU time) and
`WHISPER_MIN_SILENCE_MS` (lower keeps a hesitant speaker's short pauses).
By default the API listens on port 5000 on all network interfaces so a phone
on the same trusted LAN can reach it.
Set `SERVER_HOST=127.0.0.1` to restrict it to this computer, or configure
`SERVER_HOST` and `SERVER_PORT` for a different local setup. Configure local
Ollama with `OLLAMA_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS`.
`UPLOAD_FOLDER` can override the default project-local `uploads` directory.

## Endpoints

- `GET /api/health` returns `{"status":"ok","audio_processing":"async-jobs"}`.
- `POST /api/process-audio` accepts multipart form data with an audio file in
  the `file` field. Accepted extensions: `.m4a`, `.mp3`, and `.wav`. The maximum
  request size is 100 MB. It quickly returns `202` with a `job_id` and a
  `Location` header instead of keeping the upload request open during inference.
- `GET /api/process-audio/jobs/<job_id>` returns `202` while processing and
  `200` with the completed result. Failed jobs return a JSON `status: "failed"`
  and an error message. The mobile app polls this endpoint and stores the job
  ID locally, so reconnects or app restarts resume polling rather than uploading
  the same recording again.

On job completion, the job status endpoint returns a speaker-labeled `transcript`,
timestamped `dialogue` turns, and a `speakers` list, as well as
`detailed_class_notes` (topics, definitions, teacher explanations, and Q&A),
`easy_notes`, and `revision_quiz` (five MCQs with answers and three conceptual
questions). When transcription finds no speech, the job still completes
successfully and displays “No speech was detected in this recording” in the
transcript and notes; no quiz questions are generated for silent audio. Uploads
are deleted after processing, including when processing fails. Job state and results are held in memory for up to two hours; restarting
the backend clears that state, and the mobile app will re-upload if its saved
job is no longer found.

Pyannote initially labels distinct voices as `Speaker 1`, `Speaker 2`, etc. In
the app's Transcript tab, rename them (for example `Teacher` or `Student 1`),
then press **Save & regenerate**.
The backend regenerates the transcript-based study materials using the updated
labels and saves a local voice embedding so it can recognize that voice in
future recordings. Voice profiles are stored locally in
`data/speaker_profiles.json` and are ignored by Git. Use **Forget voice
profile** in the app to remove an identity profile; previously saved transcripts
are not changed. Get consent before recording or saving voice identities.
Speaker identity is an audio-based estimate, not guaranteed recognition. Review
the transcript and speaker assignments, especially when several people speak
over one another. The transcript marks detected overlapping voices.

CORS is enabled for HTTP and HTTPS origins on `localhost` and `127.0.0.1` with
any port. The audio endpoint also rejects requests carrying a non-local
`Origin` header and limits each client to 10 audio requests per 60 seconds
(configurable with `PROCESS_AUDIO_RATE_LIMIT` and
`PROCESS_AUDIO_RATE_WINDOW_SECONDS`); credentials are not enabled. The in-memory
rate limiter applies to this server process only. This server is intended for
local development and should not be exposed directly to untrusted networks.

## Mobile app (Expo React Native)

The Expo SDK 57 app in `mobile/` records `.m4a` audio to device document storage
and keeps its queue and study results on-device. It can record and play audio
offline. Pending audio is submitted to the API when a network connection is
available; a reachable server is required to create notes and quiz results.
Create or select a subject folder before recording. Each class is saved under
that subject with its recording date, transcript, notes, and quiz, and existing
recordings without a subject are moved into a local `General` folder on upgrade.
Speaker voice profiles and saved names are shared across subjects on this
backend: label a speaker once in the Transcript tab, and later recordings can
reuse that voice identity and display name in the dialogue.

1. Start the backend and Ollama using the setup above. For an Android emulator,
   uses `http://10.0.2.2:5000` for the Android emulator. For a physical phone,
   `mobile/src/config.ts` must use the computer's LAN address. It is currently
   set to `http://192.168.100.15:5000`; update it if the computer's LAN address
   changes. Keep both devices on the same trusted Wi-Fi and
   allow port 5000 through the computer's firewall for that private network.
   For iOS Simulator, use `http://localhost:5000`. Expo's `--tunnel` option
   tunnels the JavaScript bundle only; the phone still needs LAN access to the
   backend URL.
2. Install Node.js 20 LTS and the current Expo Go app (SDK 57 compatible), then
   install and start the app:

   ```powershell
   cd mobile
   npm install
   npx expo start --lan -c
   ```

3. Scan the QR code with Expo Go on a physical device or open the Android/iOS
   emulator from the Expo CLI. Microphone access is requested on the first
   recording. The recorder saves audio in device storage before attempting any
   network sync.

HTTP is enabled for local testing in the Expo native app configuration. The
backend binds to all interfaces by default to support a phone on the LAN; use
only on a trusted private network and prefer `SERVER_HOST=127.0.0.1` when phone
access is not needed.

## Tests

```powershell
python -m unittest discover -s tests
```
