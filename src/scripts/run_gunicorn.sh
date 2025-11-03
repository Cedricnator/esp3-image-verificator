#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_MODULE="${APP_MODULE:-src.main:app}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-1}"

EXTRA_ARGS=()
if [[ "${RELOAD:-0}" != "0" ]]; then
	EXTRA_ARGS+=(--reload)
fi

cd "${REPO_ROOT}"

CMD=(
	gunicorn "${APP_MODULE}"
	--bind "${HOST}:${PORT}"
	--workers "${WORKERS}"
)

if (( ${#EXTRA_ARGS[@]} )); then
	CMD+=("${EXTRA_ARGS[@]}")
fi

if (( $# )); then
	CMD+=("$@")
fi

exec "${CMD[@]}"
