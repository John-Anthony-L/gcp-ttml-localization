#!/usr/bin/env bash

# Minimal setup: enable required Google Cloud APIs for this project.
# Enables:
#   - Vertex AI API (aiplatform.googleapis.com)
#   - Video Intelligence API (videointelligence.googleapis.com)
#
# Usage:
#   1) Put PROJECT_ID in a .env file (PROJECT_ID=your-project) and run:
#        bash setup.sh
#   2) Or override via flag:
#        bash setup.sh -p <PROJECT_ID>
#
# Requirements:
#   - gcloud CLI installed and authenticated (gcloud auth login)

PROJECT_ID=""

usage() {
  cat <<EOF
Usage: $0 [-p <PROJECT_ID>]

This script enables the following services on the given project:
  - aiplatform.googleapis.com (Vertex AI)
  - videointelligence.googleapis.com (Video Intelligence)
EOF
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not found in PATH." >&2
    echo "Aborting setup without error code. Please install '$1' and re-run." >&2
    exit 0
  fi
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo ""
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# If not provided via flag, try to load from .env
if [[ -z "$PROJECT_ID" ]] && [[ -f .env ]]; then
  echo "Loading PROJECT_ID from .env"
  set -o allexport
  # shellcheck disable=SC1091
  source .env || true
  set +o allexport
  PROJECT_ID="${PROJECT_ID:-}"
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: PROJECT_ID not set. Provide with -p or set PROJECT_ID in .env" >&2
  usage
  echo "Aborting setup without error code. Set PROJECT_ID and re-run." >&2
  exit 0
fi

need_cmd gcloud

echo "Setting gcloud project to $PROJECT_ID"
gcloud config set project "$PROJECT_ID" --quiet

echo "Enabling required APIs on project: $PROJECT_ID"
gcloud services enable \
  aiplatform.googleapis.com \
  videointelligence.googleapis.com \
  translate.googleapis.com \
  --project="$PROJECT_ID" --quiet

# Create Python virtual environment and install dependencies
PY_CMD=$(find_python)
if [[ -z "$PY_CMD" ]]; then
  echo "Python 3 not found. Please install Python 3.x and re-run." >&2
  echo "Skipping venv creation and dependency installation."
  echo "Done."
  exit 0
fi

if [[ -d .venv ]]; then
  echo "Virtual environment .venv already exists. Reusing it."
else
  echo "Creating Python virtual environment at .venv"
  "$PY_CMD" -m venv .venv || {
    echo "Failed to create virtual environment. Skipping installs." >&2
    echo "Done."
    exit 0
  }
fi

if [[ -x .venv/bin/python ]]; then
  echo "Upgrading pip, setuptools, wheel..."
  .venv/bin/python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
  if [[ -f requirements.txt ]]; then
    echo "Installing dependencies from requirements.txt..."
    .venv/bin/pip install -r requirements.txt || true
  else
    echo "requirements.txt not found. Skipping dependency installation."
  fi
  echo "\nSetup complete. To activate the virtual environment, run:"
  echo "  source .venv/bin/activate"
else
  echo "Virtual environment not usable (.venv/bin/python missing)."
fi

echo "Done."
