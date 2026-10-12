#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/site-pipeline-adapter" >&2
  exit 2
fi
adapter="$1"
[[ "$adapter" = /* && -x "$adapter" ]] || { echo "site adapter must be an absolute executable" >&2; exit 2; }

prefix="${HOME}/.local"
mkdir -p "${prefix}/lib/polymer-agent" "${prefix}/bin"
install -m 0755 "$(dirname "$0")/remote_runner.py" "${prefix}/lib/polymer-agent/remote_runner.py"
cat > "${prefix}/bin/polymer-agent-remote" <<EOF
#!/usr/bin/env bash
export POLYMER_AGENT_SITE_PIPELINE='$adapter'
exec python3 '${prefix}/lib/polymer-agent/remote_runner.py' "\$@"
EOF
chmod 0755 "${prefix}/bin/polymer-agent-remote"
echo "Installed ${prefix}/bin/polymer-agent-remote"
echo "Ensure ${prefix}/bin is on PATH, then run: polymer-agent-remote self-test --json"
