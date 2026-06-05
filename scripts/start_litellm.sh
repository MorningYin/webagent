#!/usr/bin/env bash
# Start the LiteLLM proxy (local AI gateway) on :4000. Idempotent-ish: kills any prior proxy first.
# Secrets come from env (set here, NOT in the committed config). Logs -> /root/autodl-tmp/litellm.log
set -u

LITELLM=/root/autodl-tmp/litellm-venv/bin/litellm
CONFIG=/root/webgym/scripts/litellm_config.yaml
PORT=4000

# Secrets (gateway bases + API keys + master key) live in .env (gitignored).
# Copy .env.example -> .env and fill real values.
ENV_FILE="${ENV_FILE:-/root/webgym/.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Copy .env.example to .env and fill in keys." >&2
  exit 1
fi
set -a; . "$ENV_FILE"; set +a
# don't fetch the remote model-cost map on startup (SSL-times-out here); use local backup
export LITELLM_LOCAL_MODEL_COST_MAP="True"

# the proxy itself must reach the gateways DIRECTLY (do NOT route via the sealed browser proxy)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# kill any prior litellm cleanly (avoid pkill self-match / multi-worker restart races)
/root/miniconda3/bin/python - <<'PYKILL'
import os, signal, time
me = {os.getpid(), os.getppid()}
def sweep(sig):
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in me: continue
        try: cl = open(f'/proc/{d}/cmdline','rb').read().decode('utf8','ignore')
        except: continue
        if 'litellm' in cl and 'start_litellm' not in cl:
            try: os.kill(int(d), sig)
            except: pass
sweep(signal.SIGTERM); time.sleep(2); sweep(signal.SIGKILL)
PYKILL
sleep 1

# single async worker: policy calls are I/O-bound (wait on upstream), one uvicorn worker
# handles 50 concurrent fine, and avoids the multi-process restart race that killed it.
nohup "$LITELLM" --config "$CONFIG" --port "$PORT" \
  > /root/autodl-tmp/litellm.log 2>&1 &
echo "litellm proxy pid $! on :${PORT}  (log: /root/autodl-tmp/litellm.log)"
