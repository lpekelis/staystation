#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────────
DEV_MODE=false
for arg in "$@"; do
  case "$arg" in
    --dev) DEV_MODE=true ;;
    *) echo "Unknown argument: $arg"; echo "Usage: $0 [--dev]"; exit 1 ;;
  esac
done

if $DEV_MODE; then
  echo "Running in dev mode (no Docker, dev dependencies included)"
else
  echo "Running in production mode (Docker required, no dev dependencies)"
fi

# ── Detect platform ────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *) echo "Unsupported OS: $OS" && exit 1 ;;
esac

# ── Ensure Python 3.11+ ────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "Python 3 not found. Installing..."
  if [ "$PLATFORM" = "macos" ]; then
    if ! command -v brew &>/dev/null; then
      echo "Homebrew is required to install Python on macOS."
      echo "Install it from https://brew.sh and re-run this script."
      exit 1
    fi
    brew install python@3.11
  else
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv
  fi
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "$PYTHON_VERSION" -lt 11 ]; then
  echo "Python 3.11+ is required (found 3.${PYTHON_VERSION})."
  if [ "$PLATFORM" = "linux" ]; then
    echo "Run: sudo apt-get install python3.11"
  else
    echo "Run: brew install python@3.11"
  fi
  exit 1
fi

# ── Ensure Poetry ──────────────────────────────────────────────────────────────
if ! command -v poetry &>/dev/null; then
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v poetry &>/dev/null; then
  echo "Poetry not found. Attempting to install..."

  # Preferred: pipx already present
  if command -v pipx &>/dev/null; then
    echo "Installing Poetry via pipx..."
    pipx install poetry
    pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"

  # Fallback: install pipx first (apt on Linux, pip on macOS), then Poetry
  else
    echo "Installing pipx, then Poetry..."
    if [ "$PLATFORM" = "linux" ]; then
      sudo apt-get update -qq
      sudo apt-get install -y pipx
    else
      pip3 install --user pipx
    fi
    python3 -m pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"
    pipx install poetry
  fi

  if ! command -v poetry &>/dev/null; then
    echo ""
    echo "Poetry was installed but is not on PATH."
    echo "Add the following to your shell profile and re-run:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
  fi
fi

echo "Using Poetry: $(poetry --version)"

# ── Install Linux build dependencies ──────────────────────────────────────────
# lgpio requires swig and build tools to compile its C extension
# system-site-packages allows the venv to see picamera2 installed via apt
if [ "$PLATFORM" = "linux" ]; then
  sudo apt-get update -qq
  sudo apt-get install -y swig build-essential
  poetry config virtualenvs.options.system-site-packages true
fi

# ── Install project dependencies ───────────────────────────────────────────────
cd "$REPO_ROOT"
if $DEV_MODE; then
  poetry install --with dev,tests
else
  poetry install --only main,tests
fi

# ── Dev-only: install pre-commit hooks ────────────────────────────────────────
if $DEV_MODE; then
  poetry run pre-commit install
fi

# ── Production-only: build and start Docker inference server ──────────────────
if ! $DEV_MODE; then
  IMAGE_NAME="staystation-inference"
  CONTAINER_NAME="staystation-yolo"

  if ! command -v docker &>/dev/null; then
    echo "Docker not found — install Docker and re-run to enable YOLO detection."
    exit 1
  fi

  echo "Building inference server image (this may take a while on first run)..."
  docker build -t "$IMAGE_NAME" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"

  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

  echo "Starting inference server..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 8080:8080 \
    "$IMAGE_NAME"

  echo "Waiting for inference server to be ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
      echo "Inference server ready!"
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "Inference server did not become ready in time — check: docker logs $CONTAINER_NAME"
      exit 1
    fi
    sleep 2
  done
fi

echo ""
echo "Setup complete. Activate the virtualenv with:"
echo "  source .venv/bin/activate"
echo "Or prefix commands with: poetry run"
