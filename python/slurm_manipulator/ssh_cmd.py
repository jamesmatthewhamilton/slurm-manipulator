"""SSH command-line builders. Stdlib only.

This module exists so the rest of the package can issue `ssh` and `scp`
commands without re-implementing the sshpass branch each time. It does NOT
open tunnels, probe HTTP endpoints, or maintain connection state — those
concerns live in `llm_connections.ssh` (the consumer of this package).
"""

import shutil


def _ssh_cmd(extra_args: list, ssh_password: str = "") -> list:
    """Build an ssh command list, optionally wrapping with sshpass."""
    if ssh_password:
        if not shutil.which("sshpass"):
            raise ConnectionError(
                "Password auth requires 'sshpass' but it's not installed.\n"
                "Install it with:\n"
                "  brew install sshpass     # macOS\n"
                "  apt install sshpass      # Debian/Ubuntu\n"
                "Or set up SSH keys (ssh-keygen + ssh-copy-id) instead."
            )
        return ["sshpass", "-p", ssh_password, "ssh"] + extra_args
    return ["ssh"] + extra_args


def _scp_cmd(extra_args: list, ssh_password: str = "") -> list:
    """Build an scp command list, optionally wrapping with sshpass."""
    if ssh_password:
        return ["sshpass", "-p", ssh_password, "scp"] + extra_args
    return ["scp"] + extra_args
