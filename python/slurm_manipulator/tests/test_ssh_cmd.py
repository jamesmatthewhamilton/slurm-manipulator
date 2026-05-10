"""Tests for the ssh/scp command builders."""

import pytest

from slurm_manipulator.ssh_cmd import _scp_cmd, _ssh_cmd


def test_ssh_cmd_no_password():
    cmd = _ssh_cmd(["alice@host", "echo ok"])
    assert cmd == ["ssh", "alice@host", "echo ok"]


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
    assert cmd == ["scp", "a", "b"]


def test_scp_cmd_with_password(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    cmd = _scp_cmd(["a", "b"], "secret")
    assert cmd[:3] == ["sshpass", "-p", "secret"]
    assert cmd[3] == "scp"
