"""Bounded SSH commands for scheduler reads and idempotent operations."""
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def _ssh_options(multiplex: bool = True) -> list:
    options = ["-o", "ConnectTimeout=10"]
    if not multiplex:
        return options + ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
    if platform.system() != "Windows":
        directory = Path(tempfile.gettempdir()) / f"llm-connections-ssh-{os.getuid()}"
        # macOS has a 104-byte Unix socket limit; OpenSSH also appends a
        # temporary suffix when creating the 40-character %C socket name.
        if len(os.fsencode(directory)) + 58 >= 104:
            directory = Path('/tmp') / f"llm-connections-ssh-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        if (directory.is_symlink() or directory.stat().st_uid != os.getuid()
                or directory.stat().st_mode & 0o077):
            raise ConnectionError("SSH control directory must be private and owned by the current user")
        options += ["-o", "ControlMaster=auto", "-o", "ControlPersist=600",
                    "-o", f"ControlPath={directory}/%C"]
    return options


def _ssh_cmd(extra_args: list, ssh_password: str = "", *, multiplex: bool = True) -> list:
    """Build SSH command, using sshpass if password is provided."""
    if ssh_password:
        if not shutil.which("sshpass"):
            raise ConnectionError(
                "Password auth requires 'sshpass' but it's not installed.\n"
                "Install it with:\n"
                "  brew install sshpass    # macOS (via homebrew)\n"
                "  apt install sshpass     # Ubuntu/Debian\n"
                "Or set up SSH keys instead."
            )
        return ["sshpass", "-p", ssh_password, "ssh"] + _ssh_options(multiplex) + extra_args
    return ["ssh"] + _ssh_options(multiplex) + extra_args


def _scp_cmd(extra_args: list, ssh_password: str = "") -> list:
    """Build SCP command, using sshpass if password is provided."""
    if ssh_password:
        return ["sshpass", "-p", ssh_password, "scp"] + _ssh_options() + extra_args
    return ["scp"] + _ssh_options() + extra_args


def _run_ssh(extra_args: list, ssh_password: str = "", timeout: int = 15,
             *, retry_nonzero: bool = False, input_text: str | None = None):
    """Retry transport failures for reads and idempotent remote commands.

    Scheduler reads may opt into retrying remote nonzero exits in the same budget.
    Idempotent uploads can provide input_text, replayed unchanged on each attempt.
    Job submission deliberately does not use this helper: a lost sbatch reply
    does not establish that submission failed.
    """
    for attempt in range(3):
        try:
            result = subprocess.run(_ssh_cmd(extra_args, ssh_password),
                                    capture_output=True, text=True, timeout=timeout,
                                    input=input_text)
            if result.returncode != 255 and (
                    not retry_nonzero or result.returncode == 0 or attempt == 2):
                return result
            last_failure = f"exit status {result.returncode}; stderr: {result.stderr.strip() or '(empty)'}"
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            last_failure = f"timed out after {timeout}s; stderr: {stderr.strip() or '(empty)'}"
        if attempt < 2:
            logger.info(f"  SSH command unavailable ({last_failure}); retrying remote command...")
            time.sleep(2 ** attempt)
    raise ConnectionError(f"SSH command failed after three attempts; remote state is unknown ({last_failure})")
