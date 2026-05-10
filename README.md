# slurm-manipulator

Primitive tools for Slurm job manipulation over SSH. **Stdlib only.**

This package is intentionally small and unopinionated. It does:

- Submit / find / poll / cancel Slurm jobs (via ssh + squeue/sbatch/scancel/scontrol)
- Render parameterized sbatch templates (`string.Template.safe_substitute`)
- Read a remote rendezvous file
- Parse the standard outputs (squeue, sbatch, scontrol, endpoint URLs)

It does **not**:

- Open SSH tunnels
- Probe HTTP endpoints
- Parse cluster/session YAML config
- Manage long-running session lifecycles

Those concerns belong in the *consumer* package (e.g. `llm_connections`, which
depends on this one for its Slurm bits and adds the higher-level orchestration).

## Layout

```
slurm-manipulator/
├── python/
│   └── slurm_manipulator/
│       ├── __init__.py
│       ├── slurm.py              # find/submit/poll/cancel/query_gpu_info, parsers
│       ├── ssh_cmd.py            # _ssh_cmd, _scp_cmd (sshpass-aware shell builders)
│       ├── sbatch_render.py      # render_sbatch + endpoint_file_for_client
│       ├── sbatch/
│       │   └── ollama-generic.sbatch    # bundled parameterized template
│       ├── scripts/
│       │   ├── slurm-kill-all.sh        # bash <user@host> — scancel all your jobs
│       │   └── slurm-gpu-status.sh      # bash <user@host> — sinfo + your queue
│       └── tests/
└── pyproject.toml
```

## Usage

```python
from slurm_manipulator import (
    submit_sbatch, find_existing_job, poll_until_running,
    cancel_job, query_gpu_info, read_endpoint_file, render_sbatch,
)

ssh_target = "alice@login.example-hpc.edu"

# Existing job?
existing = find_existing_job(ssh_target, "alice", "my-job-name")
if existing:
    state, job_id = existing
else:
    rendered = render_sbatch(template_path, cluster_cfg, "my-session", session_cfg)
    job_id = submit_sbatch(ssh_target, rendered, "my-job.sbatch", "~/log")
    poll_until_running(ssh_target, job_id)

# Read the rendezvous file the sbatch wrote
host, port = read_endpoint_file(ssh_target, f"~/endpoint-{job_id}.txt")

# ... do something with host:port (e.g. tunnel to it from a higher layer) ...

cancel_job(ssh_target, job_id)
```

## Install

```
pip install -e .
```

Or just put `python/` on your `PYTHONPATH`.

## Run tests

```
PYTHONPATH=python pytest python/slurm_manipulator/tests/
```

## The rendezvous-file contract

The sbatch script's job is to write `http://<hostname>:<port>` (or any
service-discoverable string) to a file path that the client can construct.
Convention: pattern with `${job_id}` placeholder, e.g.
`~/endpoint-${job_id}.txt`. The sbatch substitutes `$SLURM_JOB_ID` at
runtime; the client substitutes the parsed job id with
`endpoint_file_for_client(pattern, job_id)`.

Any service that obeys this contract works — Ollama, vLLM, llama.cpp,
your own Flask server, etc. The bundled `ollama-generic.sbatch` is one
example.
