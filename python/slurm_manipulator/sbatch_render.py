"""Render an sbatch template by substituting ${var} placeholders.

Uses string.Template.safe_substitute so any unresolved ${var} (e.g., bash's
$SLURM_JOB_ID) survives untouched into the rendered file.

The rendezvous-file pattern lives in cluster.endpoint.file_pattern (e.g.
"~/ollama-endpoint-${job_id}.txt"). Before substitution, we replace
${job_id} with $SLURM_JOB_ID so Bash resolves it at runtime on the cluster.
"""

import os
import string

# Directives that, if set in sbatch_params, get rendered as their own
# #SBATCH lines via the ${extra_sbatch_directives} hook in the template.
_OPTIONAL_DIRECTIVES = ("account", "qos", "constraint", "nodelist", "exclude")


def _build_extra_directives(sbatch_params: dict) -> str:
    """Return a newline-joined block of optional #SBATCH lines."""
    lines = []
    for key in _OPTIONAL_DIRECTIVES:
        val = sbatch_params.get(key)
        if val:
            lines.append(f"#SBATCH --{key}={val}")
    return "\n".join(lines)


def _endpoint_file_for_sbatch(file_pattern: str) -> str:
    """Convert config-time ${job_id} into runtime $SLURM_JOB_ID for Bash."""
    return file_pattern.replace("${job_id}", "$SLURM_JOB_ID")


def render_sbatch(template_path: str, cluster_cfg: dict,
                  session_name: str, session_cfg: dict) -> str:
    """Render an sbatch template to a string.

    Args:
        template_path: path to a sbatch template with ${var} placeholders
        cluster_cfg: cluster config dict (ssh, paths, endpoint, ...)
        session_name: name of the session (used as ${job_name} default)
        session_cfg: session config dict (model, sbatch_params, ...)

    Returns:
        Rendered sbatch script as a string. Bash variables ($SLURM_JOB_ID,
        $PORT, $MODEL, etc.) are preserved for runtime expansion.
    """
    template_path = os.path.expanduser(template_path)
    with open(template_path, "r") as f:
        raw = f.read()

    # Strip CRLF if the file was edited on Windows; the cluster will get LF.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    paths = cluster_cfg.get("paths") or {}
    endpoint = cluster_cfg.get("endpoint") or {}
    sbatch_params = dict(session_cfg.get("sbatch_params") or {})

    sub = {
        "job_name": sbatch_params.get("job_name") or session_name,
        "model": session_cfg.get("model", ""),
        "ollama_bin": paths.get("ollama_bin", "ollama"),
        "ollama_models_dir": paths.get("ollama_models_dir", ""),
        "log_dir": paths.get("log_dir", "."),
        "endpoint_file": _endpoint_file_for_sbatch(endpoint.get("file_pattern", "")),
        "extra_sbatch_directives": _build_extra_directives(sbatch_params),
    }
    # Pass through anything in sbatch_params (partition, gres, cpus, mem, time,
    # port, ...). Optional directive keys are also fine to leave in — they're
    # consumed by extra_sbatch_directives but harmless if referenced.
    for k, v in sbatch_params.items():
        sub.setdefault(k, str(v))

    return string.Template(raw).safe_substitute(sub)


def endpoint_file_for_client(file_pattern: str, job_id: str) -> str:
    """Resolve the rendezvous file path for a known job_id (client-side)."""
    return file_pattern.replace("${job_id}", str(job_id))
