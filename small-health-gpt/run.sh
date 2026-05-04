set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"

cleanup() {
  if [ -n "${API_PID:-}" ]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$VENV_DIR/bin/python" -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --app-dir "$ROOT_DIR" &
API_PID=$!

echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both servers."

"$VENV_DIR/bin/python" -m http.server 5173 --directory "$ROOT_DIR/frontend"
