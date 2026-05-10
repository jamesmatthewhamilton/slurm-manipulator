"""Slurm job lifecycle helpers: find/submit, poll, read endpoint, query GPU info.

All side-effecting functions take an ssh_target string and ssh_password so
they can be wrapped by _ssh_cmd from .ssh_cmd. Pure-string parsers are split
out (parse_*) to keep them unit-testable without subprocess.
"""

import logging
import re
import subprocess
import time
import urllib.parse

from .ssh_cmd import _ssh_cmd

logger = logging.getLogger(__name__)


# ── Pure-string parsers ──────────────────────────────────────────────────

def parse_job_name_from_sbatch(sbatch_text: str) -> str | None:
    """Extract '#SBATCH --job-name=<name>' from a rendered sbatch script."""
    for line in sbatch_text.splitlines():
        m = re.match(r"#SBATCH\s+--job-name=(\S+)", line)
        if m:
            return m.group(1)
    return None


def parse_squeue_output(stdout: str, target_job_name: str) -> tuple[str, str] | None:
    """Find a job in squeue output (format: 'STATE NAME JOBID' per line).

    Returns (state, job_id) or None.
    """
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[1] == target_job_name:
            return parts[0], parts[2]
    return None


def parse_sbatch_job_id(stdout: str) -> str | None:
    """Extract the numeric job id from sbatch's stdout."""
    for word in stdout.split():
        if word.isdigit():
            return word
    return None


def parse_endpoint_url(stdout: str) -> tuple[str, int] | None:
    """Parse 'http://host:port' from the rendezvous file's contents."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            parsed = urllib.parse.urlparse(line)
            if parsed.hostname and parsed.port:
                return parsed.hostname, parsed.port
    return None


def parse_scontrol_gpu_info(stdout: str) -> str:
    """Build a one-line summary from `scontrol show job <id> -o`."""
    parts = []
    m = re.search(r"Partition=(\S+)", stdout)
    if m:
        parts.append(f"partition: {m.group(1)}")
    m = re.search(r"Gres=(\S+)", stdout)
    if m:
        parts.append(f"gres: {m.group(1)}")
    m = re.search(r"NodeList=(\S+)", stdout)
    if m:
        parts.append(f"node: {m.group(1)}")
    return ", ".join(parts)


# ── Side-effecting helpers (subprocess + ssh) ────────────────────────────

def find_existing_job(ssh_target: str, ssh_user: str, job_name: str,
                      ssh_password: str = "") -> tuple[str, str] | None:
    """squeue the cluster looking for a job with `job_name`.

    Returns (state, job_id) or None.
    """
    try:
        result = subprocess.run(
            _ssh_cmd(
                [ssh_target, f"squeue -u {ssh_user} -h -o '%T %j %i'"],
                ssh_password,
            ),
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return parse_squeue_output(result.stdout, job_name)


def submit_sbatch(ssh_target: str, sbatch_text: str, remote_basename: str,
                  log_dir: str, ssh_password: str = "") -> str:
    """Send rendered sbatch to remote, fix CRLF, sbatch it, return job_id."""
    # Ensure log_dir exists.
    try:
        subprocess.run(
            _ssh_cmd([ssh_target, f"mkdir -p {log_dir}"], ssh_password),
            capture_output=True, timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConnectionError(
            f"SSH connection to {ssh_target} timed out. Are you on the VPN?"
        ) from exc

    # SCP via stdin: write the rendered text to a file on the remote.
    remote_path = f"~/{remote_basename}"
    proc = subprocess.run(
        _ssh_cmd([ssh_target, f"cat > {remote_path}"], ssh_password),
        input=sbatch_text, text=True,
        capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to write sbatch to {ssh_target}:{remote_path}: "
            f"{proc.stderr.strip()}"
        )

    # Belt-and-braces CRLF strip on the remote.
    subprocess.run(
        _ssh_cmd([ssh_target, f"sed -i 's/\\r$//' {remote_path}"], ssh_password),
        capture_output=True, timeout=15,
    )

    submit = subprocess.run(
        _ssh_cmd([ssh_target, f"sbatch {remote_path}"], ssh_password),
        capture_output=True, text=True, timeout=30,
    )
    if submit.returncode != 0:
        raise RuntimeError(f"sbatch submission failed:\n{submit.stderr.strip()}")

    job_id = parse_sbatch_job_id(submit.stdout)
    if not job_id:
        raise RuntimeError(
            f"Could not parse job id from sbatch stdout:\n{submit.stdout.strip()}"
        )
    logger.info(f"  Submitted Slurm job {job_id}")
    return job_id


def poll_until_running(ssh_target: str, job_id: str,
                       ssh_password: str = "",
                       poll_interval: int = 10) -> None:
    """Poll squeue until job_id is RUNNING. Raise on terminal states."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        try:
            sq = subprocess.run(
                _ssh_cmd(
                    [ssh_target, f"squeue -j {job_id} -h -o '%T'"],
                    ssh_password,
                ),
                capture_output=True, text=True, timeout=15,
            )
            state = sq.stdout.strip()
            if state == "RUNNING":
                logger.info(f"  Job {job_id} is RUNNING ({elapsed:.0f}s elapsed)")
                return
            if state == "" or any(s in state for s in ("FAILED", "CANCELLED",
                                                       "TIMEOUT", "BOOT_FAIL",
                                                       "NODE_FAIL")):
                raise RuntimeError(
                    f"Slurm job {job_id} ended with state: "
                    f"{state or 'COMPLETED/UNKNOWN'}"
                )
            logger.info(
                f"  Waiting for Slurm job to start... "
                f"({elapsed:.0f}s elapsed, state: {state})"
            )
        except subprocess.TimeoutExpired:
            pass
        time.sleep(poll_interval)


def read_endpoint_file(ssh_target: str, endpoint_path: str,
                       ssh_password: str = "",
                       total_timeout: int = 60,
                       poll_interval: int = 3) -> tuple[str, int]:
    """Cat the rendezvous file until it contains 'http://host:port'.

    Returns (remote_host, remote_port). Raises ConnectionError on timeout.
    """
    logger.info(f"  Waiting for endpoint file: {endpoint_path}")
    start = time.time()
    while time.time() - start < total_timeout:
        try:
            result = subprocess.run(
                _ssh_cmd(
                    [ssh_target, f"cat {endpoint_path} 2>/dev/null"],
                    ssh_password,
                ),
                capture_output=True, text=True, timeout=15,
            )
            parsed = parse_endpoint_url(result.stdout)
            if parsed:
                logger.info(f"  Found endpoint: http://{parsed[0]}:{parsed[1]}")
                return parsed
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(poll_interval)

    raise ConnectionError(
        f"Endpoint file not found after {total_timeout}s: "
        f"{ssh_target}:{endpoint_path}"
    )


def query_gpu_info(ssh_target: str, job_id: str,
                   ssh_password: str = "") -> str:
    """Best-effort: ask scontrol for partition/gres/nodelist."""
    try:
        result = subprocess.run(
            _ssh_cmd(
                [ssh_target, f"scontrol show job {job_id} -o"],
                ssh_password,
            ),
            capture_output=True, text=True, timeout=15,
        )
        return parse_scontrol_gpu_info(result.stdout)
    except Exception:
        return ""


def cancel_job(ssh_target: str, job_id: str, ssh_password: str = "") -> None:
    """scancel a job. Best-effort; logs failures."""
    try:
        result = subprocess.run(
            _ssh_cmd([ssh_target, f"scancel {job_id}"], ssh_password),
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            logger.info(f"  Cancelled Slurm job {job_id}")
        else:
            logger.warning(f"  scancel {job_id} failed: {result.stderr.strip()}")
    except Exception as exc:
        logger.warning(f"  Could not cancel job {job_id}: {exc}")
