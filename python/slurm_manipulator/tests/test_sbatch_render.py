"""Tests for sbatch template rendering."""

import textwrap

import pytest

from slurm_manipulator.sbatch_render import (
    _build_extra_directives,
    _endpoint_file_for_sbatch,
    endpoint_file_for_client,
    render_sbatch,
)


CLUSTER_CFG = {
    "ssh": {"user": "alice", "host": "login.acme.example"},
    "paths": {
        "log_dir": "~/log",
        "ollama_bin": "/opt/ollama",
        "ollama_models_dir": "/scratch/models",
    },
    "endpoint": {"file_pattern": "~/ollama-endpoint-${job_id}.txt"},
}

SESSION_CFG = {
    "cluster": "acme",
    "model": "llama3.1:70b",
    "sbatch": "ignored-by-renderer",
    "sbatch_params": {
        "partition": "gpu-h200",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "64G",
        "time": "08:00:00",
        "port": 11070,
    },
}


def _write_template(tmp_path, body):
    p = tmp_path / "ollama.sbatch"
    p.write_text(body)
    return str(p)


def test_basic_substitution(tmp_path):
    template = textwrap.dedent("""\
        #SBATCH --job-name=${job_name}
        #SBATCH --partition=${partition}
        #SBATCH --gres=${gres}
        ${extra_sbatch_directives}
        OLLAMA=${ollama_bin}
        echo "rendezvous" > ${endpoint_file}
        $OLLAMA run "${model}"
        echo "$SLURM_JOB_ID"
    """)
    path = _write_template(tmp_path, template)
    out = render_sbatch(path, CLUSTER_CFG, "acme-llama-70b", SESSION_CFG)

    assert "#SBATCH --job-name=acme-llama-70b" in out
    assert "#SBATCH --partition=gpu-h200" in out
    assert "OLLAMA=/opt/ollama" in out
    assert 'run "llama3.1:70b"' in out
    # ${job_id} -> $SLURM_JOB_ID, not the actual id (we don't know it yet).
    assert "~/ollama-endpoint-$SLURM_JOB_ID.txt" in out
    # $SLURM_JOB_ID survives — bash will resolve at runtime.
    assert "echo \"$SLURM_JOB_ID\"" in out


def test_optional_directives(tmp_path):
    template = "${extra_sbatch_directives}\n"
    path = _write_template(tmp_path, template)
    cfg = dict(SESSION_CFG, sbatch_params=dict(SESSION_CFG["sbatch_params"],
                                               account="my-acct", qos="embers"))
    out = render_sbatch(path, CLUSTER_CFG, "x", cfg)
    assert "#SBATCH --account=my-acct" in out
    assert "#SBATCH --qos=embers" in out


def test_optional_directives_omitted_when_empty(tmp_path):
    template = "before\n${extra_sbatch_directives}\nafter\n"
    path = _write_template(tmp_path, template)
    out = render_sbatch(path, CLUSTER_CFG, "x", SESSION_CFG)
    assert "#SBATCH --account" not in out
    assert "before\n\nafter" in out  # empty block leaves a blank line


def test_crlf_input_normalized(tmp_path):
    p = tmp_path / "x.sbatch"
    p.write_bytes(b"line1\r\n${model}\r\nline3\r\n")
    out = render_sbatch(str(p), CLUSTER_CFG, "x", SESSION_CFG)
    assert "\r" not in out
    assert "\nllama3.1:70b\n" in out


def test_unknown_variable_left_intact(tmp_path):
    template = "echo ${not_in_dict}\n"
    path = _write_template(tmp_path, template)
    out = render_sbatch(path, CLUSTER_CFG, "x", SESSION_CFG)
    # safe_substitute leaves unknown ${var} alone.
    assert "${not_in_dict}" in out


def test_endpoint_file_for_sbatch():
    assert _endpoint_file_for_sbatch("~/x-${job_id}.txt") == "~/x-$SLURM_JOB_ID.txt"


def test_endpoint_file_for_client():
    assert endpoint_file_for_client("~/x-${job_id}.txt", "12345") == "~/x-12345.txt"


def test_build_extra_directives_sorted():
    out = _build_extra_directives({"qos": "embers", "account": "my-acct"})
    # Order follows _OPTIONAL_DIRECTIVES tuple, not dict order.
    assert out.startswith("#SBATCH --account=my-acct")
    assert "#SBATCH --qos=embers" in out
