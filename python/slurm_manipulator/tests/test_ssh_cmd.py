"""Tests for the ssh/scp command builders."""

import logging
import subprocess

import pytest

from slurm_manipulator import ssh_cmd
from slurm_manipulator.ssh_cmd import _run_ssh, _scp_cmd, _ssh_cmd


def test_ssh_cmd_no_password():
    cmd = _ssh_cmd(["alice@host", "echo ok"])
    assert cmd[0] == "ssh"
    assert "ConnectTimeout=10" in cmd
    assert cmd[-2:] == ["alice@host", "echo ok"]


def test_ssh_cmd_with_password_uses_sshpass(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    cmd = _ssh_cmd(["alice@host"], "secret")
    assert cmd[:3] == ["sshpass", "-p", "secret"]
    assert cmd[3] == "ssh"


def test_ssh_cmd_with_password_missing_sshpass(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(ConnectionError, match="sshpass"):
        _ssh_cmd(["alice@host"], "secret")


def test_scp_cmd_no_password():
    cmd = _scp_cmd(["a", "b"])
    assert cmd[0] == "scp"
    assert "ConnectTimeout=10" in cmd
    assert cmd[-2:] == ["a", "b"]


def test_scp_cmd_with_password(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    cmd = _scp_cmd(["a", "b"], "secret")
    assert cmd[:3] == ["sshpass", "-p", "secret"]
    assert cmd[3] == "scp"


@pytest.mark.parametrize("operation", [None, "batch script upload"])
@pytest.mark.parametrize("failure", ["timeout", "disconnect"])
def test_retry_diagnostics_identify_operation_without_command_secrets(
        monkeypatch, caplog, operation, failure):
    calls = []
    delays = []
    password = "private-password"
    remote_command = "cat > private-remote-path"
    script = "private-uploaded-script"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"],
                                            output="private-partial-output",
                                            stderr=b"connection stalled")
        return subprocess.CompletedProcess(command, 255, "private-partial-output",
                                           "connection stalled")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    monkeypatch.setattr(ssh_cmd.subprocess, "run", run)
    monkeypatch.setattr(ssh_cmd.time, "sleep", delays.append)
    with caplog.at_level(logging.INFO, logger=ssh_cmd.__name__):
        with pytest.raises(ConnectionError) as caught:
            _run_ssh(["private-target", remote_command], password,
                     input_text=script, operation=operation)

    assert len(calls) == 3
    assert delays == [1, 2]
    assert all(kwargs["timeout"] == 15 and kwargs["input"] == script
               for _, kwargs in calls)
    assert len(caplog.records) == 2
    messages = [record.getMessage() for record in caplog.records] + [str(caught.value)]
    assert all("connection stalled" in message for message in messages)
    if operation:
        assert all(operation in message for message in messages)
    else:
        assert str(caught.value).startswith("SSH command failed after three attempts;")
    for secret in (password, remote_command, "private-target", script, "private-partial-output"):
        assert all(secret not in message for message in messages)
