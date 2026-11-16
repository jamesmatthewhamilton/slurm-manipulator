"""Slurm job lifecycle helpers: find/submit, poll, read endpoint, query GPU info.

All side-effecting functions take an ssh_target string and ssh_password so
they can be wrapped by _ssh_cmd from .ssh_cmd. Pure-string parsers are split
out (parse_*) to keep them unit-testable without subprocess.
"""

import logging
import ipaddress
import re
import shlex
import subprocess
import time
import urllib.parse

from .ssh_cmd import _ssh_cmd, _run_ssh

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
    jobs = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[1] == target_job_name and parts[0] in {"RUNNING", "CONFIGURING", "PENDING"}:
            jobs.append((parts[0], parts[2]))
    priority = {"RUNNING": 0, "CONFIGURING": 1, "PENDING": 2}
    return min(jobs, key=lambda item: priority[item[0]]) if jobs else None


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
    result = _run_ssh(
        [ssh_target, f"squeue -u {shlex.quote(ssh_user)} -h -o '%T %j %i'"],
        ssh_password, retry_nonzero=True, operation="Slurm job discovery",
    )
    if result.returncode != 0:
        raise ConnectionError(
            f"Cannot read Slurm job state (exit status {result.returncode}; "
            f"stderr: {result.stderr.strip() or '(empty)'}); no replacement job was submitted"
        )
    return parse_squeue_output(result.stdout, job_name)


def submit_sbatch(ssh_target: str, sbatch_text: str, remote_basename: str,
                  log_dir: str, ssh_password: str = "") -> str:
    """Send rendered sbatch to remote, fix CRLF, sbatch it, return job_id."""
    # Preparation is idempotent, so a transport failure can safely be retried.
    _run_ssh(
        [ssh_target, f"mkdir -p {_remote_path(log_dir)}"], ssh_password,
        operation="remote log directory creation",
    ).check_returncode()

    # Write the entire script again if an upload is interrupted.
    remote_path = _remote_path(f"~/{remote_basename}")
    proc = _run_ssh(
        [ssh_target, f"cat > {remote_path}"], ssh_password,
        input_text=sbatch_text, timeout=30, operation="batch script upload",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to write sbatch to {ssh_target}:{remote_path}: "
            f"{proc.stderr.strip()}"
        )

    # Belt-and-braces CRLF strip on the remote.
    _run_ssh(
        [ssh_target, f"sed -i 's/\\r$//' {remote_path}"], ssh_password,
        operation="batch script line-ending cleanup",
    ).check_returncode()

    # Never retry submission: the job may exist even when its reply is lost.
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


TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
    "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED",
}


def _state(ssh_target, job_id, ssh_password, accounting=False):
    command = (f"sacct -j {shlex.quote(job_id)} -X -n -o State%40" if accounting else
               f"squeue -j {shlex.quote(job_id)} -h -o '%T'")
    operation = "Slurm accounting state read" if accounting else "Slurm queue state read"
    result = _run_ssh([ssh_target, command], ssh_password, retry_nonzero=True,
                      operation=operation)
    result.check_returncode()
    state = result.stdout.strip()
    return state.split()[0].rstrip("+") if state else ""


def raise_if_job_ended(ssh_target, job_id, ssh_password=""):
    state = _state(ssh_target, job_id, ssh_password)
    if not state:
        state = _state(ssh_target, job_id, ssh_password, accounting=True)
    if state in TERMINAL_STATES:
        raise RuntimeError(f"Slurm job {job_id} ended before Ollama was ready (state: {state}).")


def poll_until_running(ssh_target, job_id, ssh_password="", poll_interval=10, total_timeout=3600):
    start = time.time()
    empty = 0
    while time.time() - start < total_timeout:
        state = _state(ssh_target, job_id, ssh_password)
        if state == "RUNNING":
            return
        empty = empty + 1 if not state else 0
        if empty >= 3:
            state = _state(ssh_target, job_id, ssh_password, accounting=True)
        if state in TERMINAL_STATES:
            raise RuntimeError(f"Slurm job {job_id} ended with state: {state}")
        logger.info("Waiting for Slurm job %s (%s)", job_id, state or "temporarily absent")
        time.sleep(poll_interval)
    raise TimeoutError(f"Slurm job {job_id} did not start within {total_timeout}s")


def read_endpoint_file(ssh_target: str, endpoint_path: str,
                       ssh_password: str = "",
                       total_timeout: int = 1080,
                       poll_interval: int = 3, *, job_id=None,
                       include_gateway=False):
    """Cat the rendezvous file until it contains 'http://host:port'.

    Returns (remote_host, remote_port). Raises ConnectionError on timeout.
    """
    logger.info(f"  Waiting for endpoint file: {endpoint_path}")
    start = time.time()
    last_state_error = None
    while time.time() - start < total_timeout:
        try:
            result = subprocess.run(
                _ssh_cmd(
                    [ssh_target, 'printf "SSH_CONNECTION=%s\\n" "$SSH_CONNECTION"; '
                     f"cat {_remote_path(endpoint_path)} 2>/dev/null"],
                    ssh_password,
                ),
                capture_output=True, text=True, timeout=15,
            )
            parsed = parse_endpoint_url(result.stdout) if result.returncode == 0 else None
            if parsed:
                logger.info(f"  Found endpoint: http://{parsed[0]}:{parsed[1]}")
                gateway = None
                for line in result.stdout.splitlines():
                    if line.startswith("SSH_CONNECTION="):
                        parts = line.partition("=")[2].split()
                        if len(parts) == 4:
                            gateway = str(ipaddress.ip_address(parts[2]))
                return (*parsed, gateway) if include_gateway else parsed
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if job_id:
            try:
                raise_if_job_ended(ssh_target, job_id, ssh_password)
            except ConnectionError as exc:
                # An unavailable scheduler does not mean this known job ended.
                # Keep checking its endpoint within the existing readiness budget.
                last_state_error = exc
                logger.info(
                    "  Slurm state unavailable for job %s; continuing endpoint wait: %s",
                    job_id, exc,
                )
        time.sleep(poll_interval)

    state_detail = f"; last Slurm state read error: {last_state_error}" if last_state_error else ""
    raise ConnectionError(
        f"Endpoint file not found after {total_timeout}s: "
        f"{ssh_target}:{endpoint_path}{state_detail}"
    ) from last_state_error


def _remote_path(path):
    # Keep remote tilde expansion, but quote every user-controlled path segment.
    return "~/" + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)


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
