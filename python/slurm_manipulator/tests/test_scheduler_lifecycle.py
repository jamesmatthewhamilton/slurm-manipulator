"""Recovery must never turn an unknown remote state into a duplicate job."""
import subprocess
from unittest.mock import Mock, patch

import pytest

from slurm_manipulator import slurm


def test_running_job_preferred_over_queued_or_finishing_jobs():
    assert slurm.parse_squeue_output(
        "PENDING model 1\nCONFIGURING model 2\nCOMPLETING model 3\nRUNNING model 4", "model"
    ) == ("RUNNING", "4")


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED by 123", "OUT_OF_MEMORY", "TIMEOUT"])
def test_terminal_state_in_accounting_aborts_preload(state):
    with patch.object(slurm, "_run_ssh", side_effect=[Mock(stdout=""), Mock(stdout=state)]):
        with pytest.raises(RuntimeError, match="ended before Ollama was ready"):
            slurm.raise_if_job_ended("target", "7")


def test_absent_queue_and_accounting_records_do_not_confirm_job_end():
    with patch.object(slurm, "_run_ssh", return_value=Mock(stdout="")) as run:
        slurm.raise_if_job_ended("target", "7")
    assert run.call_count == 2


def test_endpoint_wait_covers_long_preload_and_preserves_gateway():
    clock = [0]
    with patch.object(slurm.time, "time", side_effect=lambda: clock[0]), \
         patch.object(slurm.time, "sleep", side_effect=lambda _: clock.__setitem__(0, 950)), \
         patch.object(slurm, "raise_if_job_ended") as check, \
         patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm.subprocess, "run", side_effect=[
             Mock(returncode=0, stdout=""),
             Mock(returncode=0, stdout="SSH_CONNECTION=192.0.2.1 50000 2001:db8::7 22\nhttp://node:11434\n"),
         ]):
        assert slurm.read_endpoint_file("target", "~/endpoint-7", job_id="7", include_gateway=True) == (
            "node", 11434, "2001:db8::7")
    check.assert_called_once_with("target", "7", "")


@pytest.mark.parametrize("failure", ["copy", "submission", "disconnect"])
def test_failed_copy_or_ambiguous_submission_is_not_retried(failure):
    ok = Mock(returncode=0)
    responses = [ok, Mock(returncode=1, stderr="copy failed")] if failure == "copy" else [
        ok, ok, ok, subprocess.TimeoutExpired("ssh sbatch", 30) if failure == "submission"
        else Mock(returncode=255, stderr="reply lost")]
    with patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm.subprocess, "run", side_effect=responses) as run:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            slurm.submit_sbatch("target", "script", "model.sbatch", "~/log")
    submissions = [call for call in run.call_args_list if call.args[0][-1].startswith("sbatch ")]
    assert len(submissions) == (0 if failure == "copy" else 1)


@pytest.mark.parametrize("stage", ["mkdir -p", "cat >", "sed -i"])
@pytest.mark.parametrize("failure", ["disconnect", "timeout"])
def test_submission_preparation_recovers_transport_failure(stage, failure):
    attempts = []
    uploaded = []

    def run(command, **kwargs):
        remote = command[-1]
        attempts.append(remote)
        if remote.startswith("cat >"):
            uploaded.append(kwargs.get("input"))
        if remote.startswith(stage) and attempts.count(remote) == 1:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            result = subprocess.CompletedProcess(command, 255, "", "connection lost")
            if kwargs.get("check"):
                result.check_returncode()
            return result
        stdout = "Submitted batch job 42\n" if remote.startswith("sbatch ") else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    script = "#!/bin/bash\n#SBATCH --job-name=model\necho ready\n"
    with patch("slurm_manipulator.ssh_cmd._ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch("slurm_manipulator.ssh_cmd.time.sleep"), \
         patch.object(slurm.subprocess, "run", side_effect=run):
        assert slurm.submit_sbatch("target", script, "model.sbatch", "~/log") == "42"
    assert sum(command.startswith(stage) for command in attempts) == 2
    assert sum(command.startswith("sbatch ") for command in attempts) == 1
    assert uploaded and all(text == script for text in uploaded)


@pytest.mark.parametrize("stage", ["mkdir -p", "cat >", "sed -i"])
def test_preparation_exhaustion_never_submits(stage):
    attempts = []

    def run(command, **kwargs):
        attempts.append(command[-1])
        status = 255 if command[-1].startswith(stage) else 0
        result = subprocess.CompletedProcess(command, status, "", "connection lost")
        if kwargs.get("check"):
            result.check_returncode()
        return result

    with patch("slurm_manipulator.ssh_cmd._ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch("slurm_manipulator.ssh_cmd.time.sleep"), \
         patch.object(slurm.subprocess, "run", side_effect=run):
        with pytest.raises(ConnectionError, match="three attempts"):
            slurm.submit_sbatch("target", "script", "model.sbatch", "~/log")
    assert sum(command.startswith(stage) for command in attempts) == 3
    assert not any(command.startswith("sbatch ") for command in attempts)


@pytest.mark.parametrize("status", [1, 255])
def test_failed_endpoint_read_does_not_accept_partial_output(status):
    with patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm, "raise_if_job_ended") as check, \
         patch.object(slurm.time, "sleep"), \
         patch.object(slurm.subprocess, "run", side_effect=[
             subprocess.CompletedProcess([], status, "http://partial:11434\n", "read failed"),
             subprocess.CompletedProcess([], 0, "http://ready:11435\n", ""),
         ]) as run:
        assert slurm.read_endpoint_file("target", "~/endpoint-7", job_id="7") == ("ready", 11435)
    assert run.call_count == 2
    check.assert_called_once_with("target", "7", "")
