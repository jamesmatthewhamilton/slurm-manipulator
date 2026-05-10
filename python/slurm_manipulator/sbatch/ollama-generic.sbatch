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

# Optional bootstrap (auto-install Ollama) — wired up in a future PR.
# if ! command -v "${ollama_bin}" >/dev/null 2>&1; then
#     echo "ERROR: ollama not found at ${ollama_bin}" >&2
#     exit 1
# fi

OLLAMA=${ollama_bin}
export PORT=${port}
export MODEL="${model}"
export OLLAMA_MODELS=${ollama_models_dir}
export OLLAMA_HOST=0.0.0.0:$PORT
# Keep loaded models pinned in VRAM for the entire job. Default is 5min, after
# which Ollama unloads the model and the next request triggers a slow reload
# from the networked filesystem (can take 30+ min for 100B+ models).
export OLLAMA_KEEP_ALIVE=-1

echo "========================================"
echo "OLLAMA SERVER READY"
echo "Hostname: $(hostname -f)"
echo "Port:     $PORT"
echo "Model:    $MODEL"
echo "JobID:    $SLURM_JOB_ID"
echo "========================================"

"$OLLAMA" serve &
sleep 10
# Preload the model into VRAM. For large models (e.g. gpt-oss:120b at ~70GB)
# this can take several minutes; the client polls for the endpoint file below
# and won't unblock until preload completes.
"$OLLAMA" run "$MODEL" "" 2>/dev/null
echo "Model $MODEL loaded and ready."

# Rendezvous file: the slurm_manipulator client reads this to discover host:port.
# Written AFTER preload so the REPL doesn't appear until the model is actually
# loaded — otherwise the user's first prompt triggers a slow on-demand load.
# $SLURM_JOB_ID is expanded by Bash at runtime; safe_substitute leaves it alone.
echo "http://$(hostname -f):$PORT" > ${endpoint_file}

wait
