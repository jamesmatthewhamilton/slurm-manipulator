"""slurm_manipulator — primitive tools for Slurm job manipulation over SSH.

This package is intentionally small and stdlib-only. It does NOT open SSH
tunnels, probe HTTP endpoints, parse cluster/session YAML, or manage
high-level lifecycles. Those concerns belong in the consumer (e.g.
llm_connections.SlurmSession).

What's here:
    submit_sbatch, find_existing_job, poll_until_running, cancel_job,
    query_gpu_info, read_endpoint_file
    parse_squeue_output, parse_sbatch_job_id, parse_endpoint_url,
    parse_scontrol_gpu_info, parse_job_name_from_sbatch
    render_sbatch, endpoint_file_for_client
    _ssh_cmd, _scp_cmd  (in .ssh_cmd)

Bundled assets:
    sbatch/ollama-generic.sbatch        — parameterized template
    scripts/slurm-kill-all.sh           — bash <user@host>
    scripts/slurm-gpu-status.sh         — bash <user@host>
"""

from .sbatch_render import endpoint_file_for_client, render_sbatch
from .slurm import (
    cancel_job,
    find_existing_job,
    parse_endpoint_url,
    parse_job_name_from_sbatch,
    parse_sbatch_job_id,
    parse_scontrol_gpu_info,
    parse_squeue_output,
    poll_until_running,
    query_gpu_info,
    read_endpoint_file,
    submit_sbatch,
)

__all__ = [
    # lifecycle
    "submit_sbatch",
    "find_existing_job",
    "poll_until_running",
    "cancel_job",
    "query_gpu_info",
    "read_endpoint_file",
    # parsers
    "parse_squeue_output",
    "parse_sbatch_job_id",
    "parse_endpoint_url",
    "parse_scontrol_gpu_info",
    "parse_job_name_from_sbatch",
    # rendering
    "render_sbatch",
    "endpoint_file_for_client",
]
