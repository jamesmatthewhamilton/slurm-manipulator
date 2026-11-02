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
             Mock(stdout=""),
             Mock(stdout="SSH_CONNECTION=192.0.2.1 50000 2001:db8::7 22\nhttp://node:11434\n"),
         ]):
        assert slurm.read_endpoint_file("target", "~/endpoint-7", job_id="7", include_gateway=True) == (
            "node", 11434, "2001:db8::7")
    check.assert_called_once_with("target", "7", "")


@pytest.mark.parametrize("failure", ["copy", "submission"])
def test_failed_copy_or_ambiguous_submission_is_not_retried(failure):
    ok = Mock(returncode=0)
    responses = [ok, Mock(returncode=1, stderr="copy failed")] if failure == "copy" else [
        ok, ok, ok, subprocess.TimeoutExpired("ssh sbatch", 30)]
    with patch.object(slurm, "_ssh_cmd", side_effect=lambda args, pwd: args), \
         patch.object(slurm.subprocess, "run", side_effect=responses) as run:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            slurm.submit_sbatch("target", "script", "model.sbatch", "~/log")
    submissions = [call for call in run.call_args_list if call.args[0][-1].startswith("sbatch ")]
    assert len(submissions) == (0 if failure == "copy" else 1)
