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
    assert "ControlPath=none" not in calls[0][0]
    assert all("ControlMaster=no" in command and "ControlPath=none" in command
               for command, _ in calls[1:])
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


@pytest.mark.parametrize("outcomes, shared", [
    (["timeout", 0], [True, False]),
    ([255, 0], [True, False]),
    (["timeout", 255, 0], [True, False, False]),
    ([1, 255, 0], [True, True, False]),
    ([124, 1, 0], [True, True, True]),
    ([255, 1, 0], [True, False, False]),
])
def test_transport_failure_bypasses_sharing_within_scheduler_retry_budget(
        monkeypatch, outcomes, shared):
    commands = []
    delays = []
    options = ssh_cmd._ssh_options
    # Keep the first command's normal shape without touching a real socket directory.
    monkeypatch.setattr(ssh_cmd, "_ssh_options", lambda multiplex=True:
                        ["-o", "ControlMaster=auto"] if multiplex else options(False))

    def run(command, **kwargs):
        commands.append(command)
        outcome = outcomes[len(commands) - 1]
        assert command[-2:] == ["target", "squeue -u user"]
        assert kwargs == dict(capture_output=True, text=True, timeout=15, input=None)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, outcome, "RUNNING model 7", "")

    monkeypatch.setattr(ssh_cmd.subprocess, "run", run)
    monkeypatch.setattr(ssh_cmd.time, "sleep", delays.append)
    result = _run_ssh(["target", "squeue -u user"], retry_nonzero=True)
    assert result.returncode == 0
    assert result.stdout == "RUNNING model 7"
    assert len(commands) == len(outcomes) <= 3
    assert delays == [1, 2][:len(commands) - 1]
    for command, multiplex in zip(commands, shared):
        assert ("ControlMaster=auto" in command) is multiplex
        assert ("ControlMaster=no" in command) is not multiplex
        assert ("ControlPath=none" in command) is not multiplex
        assert "-O" not in command  # Never stop or mutate the shared master.


@pytest.mark.parametrize("status", [1, 124])
def test_ordinary_remote_failure_does_not_retry_or_change_transport(monkeypatch, status):
    calls = []
    monkeypatch.setattr(ssh_cmd, "_ssh_options", lambda multiplex=True: [])
    result = subprocess.CompletedProcess([], status, "", "command rejected")
    monkeypatch.setattr(ssh_cmd.subprocess, "run", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or result)
    monkeypatch.setattr(ssh_cmd.time, "sleep", lambda delay: pytest.fail("unexpected retry"))
    assert _run_ssh(["target", "idempotent-command"]) is result
    assert len(calls) == 1


def test_fresh_transport_replays_identical_authenticated_upload(monkeypatch):
    commands = []
    uploads = []
    options = ssh_cmd._ssh_options
    monkeypatch.setattr(ssh_cmd, "_ssh_options", lambda multiplex=True:
                        ["-o", "ControlMaster=auto"] if multiplex else options(False))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    monkeypatch.setattr(ssh_cmd.time, "sleep", lambda delay: None)
    script = "#!/bin/sh\nprintf '%s\\n' 'unaltered script'\n"

    def run(command, **kwargs):
        commands.append(command)
        uploads.append(kwargs["input"])
        assert command[:4] == ["sshpass", "-p", "password", "ssh"]
        assert command[-3:] == ["-T", "user@alias", "cat > script"]
        assert kwargs["timeout"] == 30
        if len(commands) == 1:
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ssh_cmd.subprocess, "run", run)
    _run_ssh(["-T", "user@alias", "cat > script"], "password", timeout=30,
             input_text=script, operation="batch script upload")
    assert len(commands) == 2
    assert "ControlMaster=auto" in commands[0]
    assert "ControlMaster=no" in commands[1] and "ControlPath=none" in commands[1]
    assert uploads == [script, script]
