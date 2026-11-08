# StayStation

Raspberry Pi operant-conditioning system that trains a cat to stay. A camera watches
for the cat, a YOLO model running in a Docker container detects it, and a conditioning
model decides when to sound the buzzer and dispense a treat via the motor.

The system has two processes:

| Process | What it is | Starts on boot? |
| --- | --- | --- |
| **Inference server** | YOLO model behind FastAPI in the `staystation-yolo` Docker container, listening on `localhost:8080` | **Yes**: the container runs with `--restart unless-stopped`, so Docker brings it back up after a reboot (as long as the Docker service is enabled, which is the default) |
| **StayStation app** | `staystation/main.py`: camera, conditioning loop, buzzer, motor | **No**: you start it by hand, or set up the optional systemd service below |

The app refuses to start unless the inference server is answering `GET /health`.

---

## Installing on a Raspberry Pi

Tested target: Raspberry Pi 4 (64-bit Raspberry Pi OS), Python 3.11+.

### 1. Install Docker (one time)

`bin/setup.sh` does not install Docker for you.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER     # run docker without sudo
sudo systemctl enable docker      # start Docker on boot (normally already on)
```

Log out and back in (or reboot) so the `docker` group membership takes effect.

### 2. Clone the repo

```bash
cd ~
git clone https://github.com/lpekelis/staystation.git
cd staystation
```

### 3. Run setup

```bash
bin/setup.sh
```

In production mode (the default), this:

1. Installs Python 3.11+ and Poetry if they are missing.
2. Installs apt packages `swig`, `build-essential`, `python3-opencv`, and `python3-picamera2`.
3. Creates `.venv` with `--system-site-packages` so the apt-installed `picamera2` and `cv2` are visible.
4. Runs `poetry install --only main,tests`.
5. Builds the `staystation-inference` Docker image, using the model named in `.env` (`YOLO_MODEL_NAME`).
6. Starts the `staystation-yolo` container on port 8080 and waits for it to report healthy.

The first Docker build takes a while on a Pi.

Flags:

- `--dirty`: reuse the existing `.venv` instead of deleting and recreating it (faster re-runs).
- `--dev`: development setup with dev dependencies and pre-commit hooks. This **skips Docker**; use it on a laptop, not on the Pi.

---

## Updating

```bash
cd ~/staystation
git pull
bin/setup.sh --dirty
```

This reinstalls Python dependencies, rebuilds the Docker image (cached layers make
it fast unless `docker/` or `.env` changed), and restarts the inference container.
If you hit dependency problems, drop `--dirty` to rebuild the venv from scratch.

If you set up the systemd service below, restart it afterwards:

```bash
sudo systemctl restart staystation
```

---

## Running

### Inference server (background)

The container keeps running in the background and restarts on boot by itself.
You don't normally need to start it, but these commands manage it:

```bash
docker ps --filter name=staystation-yolo   # is it running?
curl http://localhost:8080/health          # should print {"status":"ok"}
docker logs -f staystation-yolo            # tail logs
docker restart staystation-yolo            # restart
docker stop staystation-yolo               # stop (stays stopped across reboots)
docker start staystation-yolo              # start again
```

### StayStation app

```bash
cd ~/staystation
source .venv/bin/activate
python -m staystation.main     # or just: staystation
```

Stop it with `Ctrl+C`; the camera, motor, and GPIO are cleaned up on exit.
Training data is appended to `data/dataset.csv`.

Common options (see `python -m staystation.main --help` for all of them):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model {1,2,3,4}` | `1` | Which conditioning model to use |
| `--camera {picamera2,usb}` | `picamera2` | Camera backend |
| `--camera-device` | `auto` | USB device index or path (e.g. `/dev/video2`) |
| `--confidence` | `0.5` | Detection confidence threshold |
| `--sigma` / `--gamma` / `--horizon` / `--p-explore` | | Shared conditioning-model parameters (`--horizon` defaults to 10, or 5 for model 4) |
| `--k` / `--p-promote` / `--p-demote` | | Model 3 level-change parameters |
| `--cat-id` | new id | Model 4: which cat's saved params to resume (see below) |
| `--lookback` / `--half-life` | `600` / `300` | Model 4: steps of history per fit, and age half-life for weighting them |
| `--threshold` | `0.66` | Model 4: treat when P(cat on mat in 5 s \| no treat) is below this |
| `--explore-interval` | `180` | Model 4: ~1 exploration treat every N steps (seconds) |
| `--max-lure-treats` | `5` | Model 4: max treats released while the cat is off the mat |
| `--save-interval` | `10` | Save the dataset every N steps |
| `--viz` | off | Show the live camera feed with detections (needs a display; press `q` to quit) |
| `--debug` | off | Print per-frame timing |

Example:

```bash
python -m staystation.main --model 3 --camera usb --debug
```

#### Model 4 (bandit) and cat ids

Model 4 refits a small logistic regression every second on the last 10 minutes
of data (newer observations weigh more). It predicts whether the cat will be on
the mat one second from now from: on/off mat, time on the mat, time away, and
whether a treat was just dispensed. Rolling that forward 5 seconds gives
P(cat on mat in 5 s). It treats when that probability (without a treat) is
below `--threshold`, plus an occasional exploration treat (`--explore-interval`).
While the cat is away, at most `--max-lure-treats` treats are released.
Each fit starts from the previous one, so what it has learned carries forward.
It starts out assuming the cat is not conditioned at all, so early on it treats
often. As the cat learns to stay, the model waits longer before treating.

Fitted parameters are saved under `data/cats/<cat-id>/`:

- `params.json` — latest fitted coefficients (rewritten after every fit)
- `diagnostics.csv` — one row per decision: coefficients, state, P(treat/no treat),
  current treat wait, lure count, and the decision

```bash
# First run: a new id like cat-20261108-153012-a1b2 is generated and printed
python -m staystation.main --model 4 --camera usb
# Later runs: resume that cat's learned params
python -m staystation.main --model 4 --camera usb --cat-id cat-20261108-153012-a1b2
```

If you pass a `--cat-id` that has no saved params yet (e.g. `--cat-id mochi`),
it starts from the defaults and saves under that name.

### Optional: start the app on boot with systemd

The app does not start on boot by default. To make it start on boot, create
`/etc/systemd/system/staystation.service` (replace `pi` with your username if it's different):

```ini
[Unit]
Description=StayStation conditioning app
After=docker.service network-online.target
Requires=docker.service

[Service]
User=pi
WorkingDirectory=/home/pi/staystation
ExecStart=/home/pi/staystation/.venv/bin/python -m staystation.main --model 1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`Restart=on-failure` matters here: right after boot the inference container may
not be ready yet, the app's health check fails, and systemd retries it 10 seconds later.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now staystation   # start now and on every boot
journalctl -u staystation -f              # follow logs
sudo systemctl stop staystation           # stop
sudo systemctl disable staystation        # stop starting on boot
```

Don't use `--viz` in the service, because it has no display.

---

## Development (laptop)

```bash
bin/setup.sh --dev
poetry run pytest
```

Dev mode doesn't run the inference server. To run the app end-to-end locally,
build and run the container from `docker/Dockerfile` yourself. Note that the base
image is `arm64`.
