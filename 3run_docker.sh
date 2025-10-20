#!/usr/bin/env bash
set -euo pipefail

# Przeznaczone dla WSL. Uruchamia każdy proces w osobnym oknie:
# - preferuje gnome-terminal (WSLg lub X-serwer + DISPLAY),
# - w przeciwnym razie używa nowych okien Windows (cmd.exe -> wsl.exe),
# - ostatnia deska ratunku: tmux, jeśli brak okien GUI i brak cmd.exe.

if ! grep -qi "microsoft" /proc/sys/kernel/osrelease; then
  echo "Ten skrypt jest przeznaczony dla WSL." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_VENV="$ROOT/.api_venv"
WATUS_ROOT="$ROOT/watus_project"
WATUS_VENV="$WATUS_ROOT/.watus_venv"

ensure_venv() {
  local venv="$1"
  [[ -x "$venv/bin/python" ]] || python3 -m venv "$venv"
}

ensure_display_if_possible() {
  # Jeśli nie ma DISPLAY, spróbuj ustawić (przydatne z X-serwerem typu VcXsrv)
  if [[ -z "${DISPLAY:-}" ]]; then
    if [[ -f /etc/resolv.conf ]]; then
      local host_ip
      host_ip="$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)"
      export DISPLAY="${host_ip}:0"
    else
      export DISPLAY=":0"
    fi
  fi
}

can_use_gnome_terminal() {
  # gnome-terminal działa gdy:
  #  - jest zainstalowany
  #  - i mamy WSLg (WAYLAND_DISPLAY) lub ustawione DISPLAY do X-serwera
  command -v gnome-terminal >/dev/null 2>&1 || return 1
  if [[ -n "${WAYLAND_DISPLAY:-}" || -n "${DISPLAY:-}" ]]; then
    return 0
  fi
  return 1
}

term() {
  local title="$1"
  local cmd="$2"

  # 1) Spróbuj WSLg / X-serwer + gnome-terminal
  ensure_display_if_possible
  if can_use_gnome_terminal; then
    nohup gnome-terminal --title="$title" -- bash -lc "$cmd; exec bash" >/dev/null 2>&1 &
    return
  fi

  # 2) Fallback: nowe okno Windows (CMD) z WSL
  if command -v cmd.exe >/dev/null 2>&1; then
    local distro="${WSL_DISTRO_NAME:-}"
    if [[ -n "$distro" ]]; then
      cmd.exe /C start "$title" wsl.exe -d "$distro" bash -lc "$cmd; exec bash"
    else
      cmd.exe /C start "$title" wsl.exe bash -lc "$cmd; exec bash"
    fi
    return
  fi

  # 3) Ostateczny fallback: tmux
  if command -v tmux >/dev/null 2>&1; then
    local session="watus"
    if ! tmux has-session -t "$session" 2>/dev/null; then
      tmux new-session -d -s "$session" -n "$title" "bash -lc '$cmd; exec bash'"
    else
      tmux new-window -t "$session" -n "$title" "bash -lc '$cmd; exec bash'"
    fi
    echo "Uruchomiono w tmux (sesja: $session). Aby dołączyć: tmux attach -t $session"
    return
  fi

  echo "Brak GUI oraz brak cmd.exe/tmux. Zainstaluj X-serwer (np. VcXsrv) i ustaw DISPLAY lub użyj Windows Terminal/CMD." >&2
  exit 1
}

# venv-y
ensure_venv "$API_VENV"
ensure_venv "$WATUS_VENV"

# Polecenia do uruchomienia
API_CMD="cd '$ROOT' && . '$API_VENV/bin/activate' && uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload"
REPORTER_CMD="cd '$WATUS_ROOT' && . '$WATUS_VENV/bin/activate' && python reporter.py"
CAMERA_CMD="cd '$WATUS_ROOT' && . '$WATUS_VENV/bin/activate' && python camera_runner.py --jsonl ./camera.jsonl --device 0"
WATUS_CMD="cd '$WATUS_ROOT' && . '$WATUS_VENV/bin/activate' && python watus.py"

echo "Uruchamianie procesów w osobnych oknach..."
term "API-Uvicorn" "$API_CMD"
sleep 0.5
term "Reporter" "$REPORTER_CMD"
sleep 0.5
term "Camera-Runner" "$CAMERA_CMD"
sleep 0.5
term "Watus" "$WATUS_CMD"
echo "Gotowe."