#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a public address, behind an authenticating gateway.
#
#   scripts/serve-public.sh              # start (or reuse) the container, then the gateway on :8080
#   MODE=hybrid scripts/serve-public.sh  # same, hybrid checkpoint
#   GW_PORT=9000 scripts/serve-public.sh
#
# The container's API port is published on 127.0.0.1 only, so vLLM never faces the
# network; gateway.py does, and every request to it needs a bearer key from the
# dashboard. Every variable scripts/serve-legacy.sh understands (MODE, CTX, MTP, SEQS, ...)
# is passed straight through.
#
# Ctrl-C stops the gateway, not the container: scripts/serve-legacy.sh runs it detached with
# --restart unless-stopped, and a ~10-minute weight load is not something to throw away
# on a terminal hangup. Stop it explicitly with `docker rm -f $NAME`.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${NAME:-qwen38-flash}"
PORT="${PORT:-18300}"             # vLLM, loopback only
GW_PORT="${GW_PORT:-8080}"        # gateway, public
GW_HOST="${GW_HOST:-0.0.0.0}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # first boot loads ~76 GiB of weights

if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo ">> reusing the server already answering on 127.0.0.1:${PORT}"
  published=$(docker inspect -f \
    '{{range .NetworkSettings.Ports}}{{range .}}{{.HostIP}}:{{.HostPort}} {{end}}{{end}}' \
    "$NAME" 2>/dev/null || true)
  case " $published" in
    *" 0.0.0.0:"*|*" ::"*)
      echo "!! note: $NAME publishes its API on all interfaces, so the unauthenticated"
      echo "   vLLM port is reachable directly and the gateway is not the only way in."
      echo "   Restart it through this script (docker rm -f $NAME) to pin it to loopback." ;;
  esac
else
  echo ">> starting $NAME on 127.0.0.1:${PORT}"
  BIND_ADDR=127.0.0.1 PORT="$PORT" NAME="$NAME" scripts/serve-legacy.sh
  printf '>> waiting for the model to load'
  deadline=$((SECONDS + READY_TIMEOUT))
  until curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    # A container that died (bad flags, OOM) would otherwise be waited on for half an hour.
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
      echo; echo "!! $NAME exited before becoming ready. Last lines:"
      docker logs --tail 40 "$NAME" 2>&1 || true
      exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo; echo "!! not ready after ${READY_TIMEOUT}s -- docker logs -f $NAME"; exit 1
    fi
    printf '.'; sleep 2
  done
  echo; echo ">> up"
fi

# uv resolves gateway.py's inline dependency block (aiohttp); plain python3 works if
# aiohttp is already installed system-wide.
if command -v uv >/dev/null 2>&1; then
  exec uv run scripts/gateway.py --upstream "http://127.0.0.1:${PORT}" --host "$GW_HOST" --port "$GW_PORT"
else
  exec python3 scripts/gateway.py --upstream "http://127.0.0.1:${PORT}" --host "$GW_HOST" --port "$GW_PORT"
fi
