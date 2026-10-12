#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"
port=8765
page=auto
open_browser=1
while (($#)); do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { printf '%s\n' '--port needs a value.' >&2; exit 2; }
      port="$2"; shift 2 ;;
    --page)
      [[ $# -ge 2 ]] || { printf '%s\n' '--page needs auto or demo.' >&2; exit 2; }
      page="$2"; shift 2 ;;
    --no-browser) open_browser=0; shift ;;
    --offline) shift ;; # Accepted for compatibility; replay is always offline.
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1 && 10#$port <= 65535)) || {
  printf '%s\n' 'Port must be between 1 and 65535.' >&2; exit 2;
}
[[ "$page" == auto || "$page" == demo ]] || {
  printf '%s\n' 'Page must be auto or demo.' >&2; exit 2;
}

probe="import importlib.util,pathlib,sys; missing=[n for n in ('uvicorn','fastapi','pydantic') if importlib.util.find_spec(n) is None]; s=importlib.util.find_spec('polymer_agent_demo'); raise SystemExit(0 if sys.version_info>=(3,11) and not missing and s and s.origin and pathlib.Path(s.origin).resolve()==pathlib.Path(sys.argv[1]).resolve()/'__init__.py' else 1)"
python_exe=''
for candidate in "${POLYMER_AGENT_PYTHON:-}" "$repo_root/.venv/bin/python" "$repo_root/.venv-platform/bin/python" python3 python; do
  [[ -n "$candidate" ]] || continue
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$probe" "$repo_root" >/dev/null 2>&1; then
    python_exe="$candidate"; break
  fi
done
if [[ -z "$python_exe" ]]; then
  printf '%s\n' \
    'Replay needs Python 3.11+, FastAPI, uvicorn and Pydantic installed for this checkout.' \
    'From this directory run:' \
    '  python3 -m venv .venv' \
    '  .venv/bin/python -m pip install -e .' \
    'Then run: bash run_demo.sh' \
    'For offline installation, add --no-index --find-links /path/to/wheels.' >&2
  exit 1
fi
export POLYMER_AGENT_APP_MODE=demo POLYMER_AGENT_DEMO_MODE=replay PYTHONUTF8=1
"$python_exe" -m polymer_agent_demo.scripts.doctor --stage demo

if ((open_browser)); then
  (
    for ((attempt=0; attempt<60; attempt++)); do
      if "$python_exe" -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1],timeout=1).close()' "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
        if command -v xdg-open >/dev/null 2>&1; then
          xdg-open "http://127.0.0.1:$port/$page" >/dev/null 2>&1 || true
        elif command -v open >/dev/null 2>&1; then
          open "http://127.0.0.1:$port/$page" >/dev/null 2>&1 || true
        fi
        break
      fi
      sleep 0.5
    done
  ) &
fi
printf 'Offline Polymer-Agent replay: http://127.0.0.1:%s/%s\n' "$port" "$page"
exec "$python_exe" -m uvicorn polymer_agent_demo.app:app --host 127.0.0.1 --port "$port"
