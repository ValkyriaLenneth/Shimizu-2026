#!/usr/bin/env bash
set -euo pipefail

pid="$1"
shift

while kill -0 "$pid" 2>/dev/null; do
  sleep 60
done

exec "$@"
