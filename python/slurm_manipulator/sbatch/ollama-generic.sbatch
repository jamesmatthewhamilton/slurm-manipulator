#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --gres=${gres}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${log_dir}/${job_name}-%j.log
${extra_sbatch_directives}

set -euo pipefail

# Environment
OLLAMA="${ollama_bin}"
PORT=$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
export PORT
export MODEL="${model}"

# Ollama-specific environment
export OLLAMA_MODELS="${ollama_models_dir}"
export OLLAMA_HOST=0.0.0.0:${PORT}
export OLLAMA_CONTEXT_LENGTH=${num_ctx}
export OLLAMA_LOAD_TIMEOUT=${load_timeout}

ENDPOINT_FILE="${endpoint_file}"
SERVER_PID=
cleanup() {
    rm -f "$ENDPOINT_FILE" "${ENDPOINT_FILE}.tmp"
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
rm -f "$ENDPOINT_FILE"

# Print connection info
echo "========================================"
echo "STARTING OLLAMA"
echo "Hostname: $(hostname -f)"
echo "Port: ${PORT}"
echo "API URL: http://$(hostname -f):${PORT}"
echo "Model: ${MODEL}"

echo "========================================"

# Start Ollama and preload model
"${OLLAMA}" serve &
SERVER_PID=$!
READY=false
for attempt in {1..60}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Ollama exited before the server was ready." >&2
        exit 1
    fi
    if VERSION=$(curl --fail --silent --show-error --connect-timeout 1 --max-time 1 \
        "http://127.0.0.1:${PORT}/api/version"); then
        READY=true
        break
    fi
    sleep 1
done
if [[ "$READY" != true ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Ollama did not become ready." >&2
    exit 1
fi
echo "Ollama version: ${VERSION}"
curl --fail --silent --show-error --max-time ${preload_timeout} "http://127.0.0.1:${PORT}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":-1,\"options\":{\"num_ctx\":${OLLAMA_CONTEXT_LENGTH},\"num_thread\":${SLURM_CPUS_PER_TASK}}}" >/dev/null || exit 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Ollama exited while loading ${MODEL}." >&2
    exit 1
fi
printf 'http://%s:%s\n' "$(hostname -f)" "$PORT" > "${ENDPOINT_FILE}.tmp"
mv "${ENDPOINT_FILE}.tmp" "$ENDPOINT_FILE"
echo "Model ${MODEL} loaded and ready."

# Keep server running until job ends
wait "$SERVER_PID"
