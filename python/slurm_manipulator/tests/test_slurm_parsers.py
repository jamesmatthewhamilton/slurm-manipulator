"""Tests for the pure-string parsers in slurm.py."""

from slurm_manipulator.slurm import (
    parse_endpoint_url,
    parse_job_name_from_sbatch,
    parse_sbatch_job_id,
    parse_scontrol_gpu_info,
    parse_squeue_output,
)


def test_parse_job_name_from_sbatch_typical():
    text = "#!/bin/bash\n#SBATCH --job-name=ollama-70b\n#SBATCH --time=08:00:00\n"
    assert parse_job_name_from_sbatch(text) == "ollama-70b"


def test_parse_job_name_missing():
    assert parse_job_name_from_sbatch("#!/bin/bash\necho hi\n") is None


def test_parse_squeue_match():
    out = "RUNNING ollama-70b 12345\nPENDING someone-elses 99999\n"
    assert parse_squeue_output(out, "ollama-70b") == ("RUNNING", "12345")


def test_parse_squeue_no_match():
    assert parse_squeue_output("RUNNING other 1\n", "ollama-70b") is None


def test_parse_squeue_empty():
    assert parse_squeue_output("", "x") is None


def test_parse_sbatch_job_id():
    assert parse_sbatch_job_id("Submitted batch job 8675309\n") == "8675309"


def test_parse_sbatch_job_id_missing():
    assert parse_sbatch_job_id("submission failed\n") is None


def test_parse_endpoint_url_typical():
    assert parse_endpoint_url("http://atl1-1-01-009-9-0.pace.gatech.edu:11070\n") == (
        "atl1-1-01-009-9-0.pace.gatech.edu", 11070,
    )


def test_parse_endpoint_url_https():
    assert parse_endpoint_url("https://node:443\n") == ("node", 443)


def test_parse_endpoint_url_garbage():
    assert parse_endpoint_url("not a url\n") is None


def test_parse_endpoint_url_no_port():
    assert parse_endpoint_url("http://node\n") is None


def test_parse_scontrol_gpu_info_full():
    out = "JobId=1 Partition=gpu-h200 ... Gres=gpu:1 ... NodeList=atl-001 ..."
    info = parse_scontrol_gpu_info(out)
    assert "partition: gpu-h200" in info
    assert "gres: gpu:1" in info
    assert "node: atl-001" in info


def test_parse_scontrol_gpu_info_partial():
    info = parse_scontrol_gpu_info("Partition=gpu Gres=gpu:1")
    assert info == "partition: gpu, gres: gpu:1"


def test_parse_scontrol_gpu_info_empty():
    assert parse_scontrol_gpu_info("") == ""
