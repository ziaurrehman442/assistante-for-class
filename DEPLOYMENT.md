# Deploying to a Contabo Ubuntu server

This guide deploys the API behind Nginx with HTTPS. Keep Gunicorn bound to
loopback; do not expose port 5000 to the Internet. The configured backend
domain is `classproject.nexa.website`; adjust `/opt/class-notes` if your
installation uses a different path.

## Before exposing the API

The current API does not authenticate users. CORS is not authentication.
Do not expose it publicly to arbitrary clients: uploads consume CPU and memory,
and speaker-profile endpoints can modify local profiles. For a private test,
restrict Nginx/firewall access to your own IP or use a VPN. Add proper API
authentication and authorization before operating a public service.

## Environment settings

Create `/etc/class-notes/class-notes.env`, owned by root and readable only by
root and the service group. Do not commit production secrets:

```dotenv
SERVER_HOST=127.0.0.1
SERVER_PORT=5000
ALLOWED_ORIGINS=https://classproject.nexa.website

OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT_SECONDS=300
HF_TOKEN=replace-with-your-Hugging-Face-read-token

UPLOAD_FOLDER=/var/lib/class-notes/uploads
SPEAKER_PROFILES_PATH=/var/lib/class-notes/data/speaker_profiles.json
PROCESS_AUDIO_RATE_LIMIT=10
PROCESS_AUDIO_RATE_WINDOW_SECONDS=60
```

`ALLOWED_ORIGINS` is a comma-separated list of exact browser origins including
scheme and port when nonstandard, for example
`https://classproject.nexa.website,https://admin.classproject.nexa.website`.
Do not include paths or
trailing slashes. If unset, only localhost/127.0.0.1 development origins are
accepted. Native React Native clients commonly omit the `Origin` header; CORS
does not provide authentication for them.

Set the mobile API URL at Expo bundle time. For local testing keep the existing
LAN URL; for production use the HTTPS domain:

```powershell
$env:EXPO_PUBLIC_API_URL = "https://classproject.nexa.website"
npx expo start --lan
```

For a production app build, set `EXPO_PUBLIC_API_URL` in the build environment
before running the Expo/EAS build. Do not use `localhost` or the server's port
5000 in a client build.

## Ubuntu install and Python environment

Install Python, FFmpeg, Nginx, and the Ollama runtime/model using their official
Ubuntu installation instructions. Place the project at `/opt/class-notes`, then
create the service account and writable state directories:

```bash
sudo useradd --system --home-dir /var/lib/class-notes --create-home \
  --shell /usr/sbin/nologin class-notes
sudo install -d -o class-notes -g class-notes /opt/class-notes
sudo install -d -o class-notes -g class-notes \
  /var/lib/class-notes/uploads /var/lib/class-notes/data \
  /var/cache/class-notes/huggingface
```

Copy the source into `/opt/class-notes` and make it readable by `class-notes`.
Create the environment file and restrict its permissions:

```bash
sudo install -d -m 0750 -o root -g class-notes /etc/class-notes
sudo install -m 0640 -o root -g class-notes /dev/null \
  /etc/class-notes/class-notes.env
sudoedit /etc/class-notes/class-notes.env
```

Create the virtual environment and install the Ubuntu production dependencies:

```bash
cd /opt/class-notes
sudo -u class-notes python3 -m venv .venv
sudo -u class-notes .venv/bin/pip install --upgrade pip
sudo -u class-notes .venv/bin/pip install -r deploy/requirements-production.txt
```

The first audio job downloads local Faster-Whisper and Pyannote models. Ensure
the service can reach Hugging Face for initial downloads and that `HF_TOKEN`
has access to the accepted Pyannote model. Ollama must be running on the same
server and have `qwen2.5:3b` pulled.

## Gunicorn systemd service

Install the provided unit and start it:

```bash
sudo install -m 0644 deploy/systemd/class-notes.service \
  /etc/systemd/system/class-notes.service
sudo systemctl daemon-reload
sudo systemctl enable --now class-notes
sudo systemctl status class-notes
sudo journalctl -u class-notes -f
```

The service deliberately uses **one Gunicorn worker**. Audio jobs and completed
results are held in process memory; multiple workers would not share job state,
and a client could poll a different worker and receive 404. The worker pool
serializes heavy audio processing. A service restart clears active job state,
so use a persistent job queue/database before scaling workers or claiming
durable production processing.

## Nginx and HTTPS (recommended)

Install Nginx and Certbot:

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

Point the domain's DNS A/AAAA record to the server, then install the Nginx
configuration:

```bash
sudo install -m 0644 deploy/nginx/class-notes.conf \
  /etc/nginx/sites-available/class-notes
sudo ln -s /etc/nginx/sites-available/class-notes \
  /etc/nginx/sites-enabled/class-notes
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d classproject.nexa.website
```

Certbot obtains and installs the TLS certificate and configures HTTPS. Verify
renewal is enabled with `sudo certbot renew --dry-run`.

Allow only SSH and web traffic through UFW. Do **not** open port 5000 publicly:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

Port 5000 remains reachable only over loopback (`127.0.0.1`) for Nginx-to-
Gunicorn proxying.

## Direct port 5000 (not recommended)

If HTTPS/reverse proxy is not yet set up, a direct public `http://` API sends
audio and results without transport encryption and exposes the unauthenticated
development API. Prefer a VPN or restrict port 5000 to a known client IP. Do
not use an unrestricted `ufw allow 5000/tcp` rule for sensitive recordings.

After deployment, check `https://classproject.nexa.website/api/health` and confirm
`audio_processing` is `async-jobs`. Then set the mobile build's
`EXPO_PUBLIC_API_URL` to `https://classproject.nexa.website`.
