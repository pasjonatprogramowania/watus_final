#!/usr/bin/env bash
set -euo pipefail

# Ścieżki
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_VENV="$ROOT/.api_venv"
WATUS_ROOT="$ROOT/watus_project"
WATUS_VENV="$WATUS_ROOT/.watus_venv"

ensure_venv() {
  local venv="$1"
  if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
  fi
}

term() {
  local title="$1"
  local cmd="$2"
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$cmd; exec bash"
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -T "$title" -e bash -lc "$cmd; exec bash"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -e bash -lc "$cmd; exec bash"
  else
    echo "Brak terminala GUI (gnome-terminal/xterm)."
    exit 1
  fi
}

# Tworzenie venv (jeśli brak)
ensure_venv "$API_VENV"
ensure_venv "$WATUS_VENV"

# Komendy
API_CMD="cd \"$ROOT\" && source \"$API_VENV/bin/activate\" && uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload"
REPORTER_CMD="cd \"$WATUS_ROOT\" && source \"$WATUS_VENV/bin/activate\" && python reporter.py"
CAMERA_CMD="cd \"$WATUS_ROOT\" && source \"$WATUS_VENV/bin/activate\" && python camera_runner.py --jsonl ./camera.jsonl --device 0"
WATUS_CMD="cd \"$WATUS_ROOT\" && source \"$WATUS_VENV/bin/activate\" && python watus.py"

# Start w osobnych oknach
term "API (uvicorn)" "$API_CMD"
term "Reporter" "$REPORTER_CMD"
term "Camera Runner" "$CAMERA_CMD"
term "Watus" "$WATUS_CMD"