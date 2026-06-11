#!/bin/bash
# tools/start_space.sh — Docker Space entrypoint: bring up llama-server
# and ComfyUI, wait until both answer, then run the Gradio app.
#
# The app tolerates either backend being down (mock LLM / curated cast),
# so health waits are best-effort with generous timeouts: first boot reads
# ~16GB of weights from image layers, which can take a few minutes.
set -uo pipefail

echo "[space] starting llama-server…"
nohup bash /app/vn/tools/space_llm_start.sh > /tmp/llama.log 2>&1 &

echo "[space] starting ComfyUI…"
cd /app/comfy
nohup python3 main.py --listen 127.0.0.1 --port 8188 \
    > /tmp/comfy.log 2>&1 &
cd /app/vn

wait_http () { # url, name, tries (2s apart)
    local url="$1" name="$2" tries="$3"
    for _ in $(seq 1 "$tries"); do
        if curl -sf -o /dev/null "$url"; then
            echo "[space] $name is up"
            return 0
        fi
        sleep 2
    done
    echo "[space] WARNING: $name not healthy yet — app starts anyway"
    return 1
}

wait_http http://127.0.0.1:8188/system_stats "ComfyUI" 60       # ~2 min
wait_http http://127.0.0.1:8110/health      "llama-server" 240 # ~8 min

echo "[space] starting Gradio app"
exec python3 app.py
