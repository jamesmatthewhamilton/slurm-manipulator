import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import unittest
import slurm_manipulator
from slurm_manipulator import render_sbatch


class SlurmStartupTests(unittest.TestCase):
    def test_owned_server_preload_and_endpoint_lifecycle(self):
        script_path = Path(slurm_manipulator.__file__).parent / 'sbatch/ollama-generic.sbatch'
        for mode in ('success', 'server_fail', 'preload_fail'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                server = directory / 'ollama'
                server.write_text('''#!/usr/bin/env python3
import os, pathlib, sys, time
root = pathlib.Path(os.environ['PACE_TEST_DIR'])
(root / 'server.pid').write_text(str(os.getpid()))
if os.environ['PACE_TEST_MODE'] == 'server_fail':
    sys.exit(17)
while True:
    time.sleep(1)
''')
                curl = directory / 'curl'
                curl.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ['PACE_TEST_DIR'])
assert not (root / 'endpoint').exists(), 'endpoint published before preload'
if any(arg.endswith('/api/version') for arg in sys.argv):
    for _ in range(200):
        if (root / 'server.pid').exists():
            break
        time.sleep(0.01)
    time.sleep(0.05)
    print('{"version":"test"}')
else:
    (root / 'preload.json').write_text(json.dumps({
        'load_timeout': os.environ['OLLAMA_LOAD_TIMEOUT'],
        'port': os.environ['PORT'],
        'payload': json.loads(sys.argv[sys.argv.index('-d') + 1]),
    }))
    if os.environ['PACE_TEST_MODE'] == 'preload_fail':
        sys.exit(22)
    while not (root / 'release-preload').exists():
        time.sleep(0.01)
''')
                server.chmod(0o700)
                curl.chmod(0o700)
                script = render_sbatch(str(script_path), {"paths": {"log_dir": "~/log", "ollama_bin": "/opt/ollama", "ollama_models_dir": "/models"}, "endpoint": {"file_pattern": "~/ollama-endpoint-${job_id}.txt"}}, "demo", {"model": "gpt-oss:120b", "sbatch_params": {"cpus": 8, "num_ctx": 65536, "partition": "gpu", "gres": "gpu:1", "mem": "96G", "time": "02:00:00"}})
                script = re.sub(r'^OLLAMA=.*$', lambda _: f'OLLAMA={shlex.quote(str(server))}',
                                script, flags=re.MULTILINE)
                script = re.sub(r'^ENDPOINT_FILE=.*$',
                                lambda _: f'ENDPOINT_FILE={shlex.quote(str(directory / "endpoint"))}',
                                script, flags=re.MULTILINE)
                local_script = directory / 'startup.sh'
                local_script.write_text(script)
                env = {**os.environ, 'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                       'PACE_TEST_DIR': str(directory), 'PACE_TEST_MODE': mode,
                       'SLURM_JOB_ID': '7', 'SLURM_CPUS_PER_TASK': '8'}
                process = subprocess.Popen(['bash', str(local_script)], env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    if mode == 'success':
                        self._wait_for_file(directory / 'preload.json', process)
                        self.assertFalse((directory / 'endpoint').exists())
                        preload = json.loads((directory / 'preload.json').read_text())
                        self.assertEqual('10m', preload['load_timeout'])
                        self.assertEqual('', preload['payload']['prompt'])
                        self.assertEqual(65536, preload['payload']['options']['num_ctx'])
                        (directory / 'release-preload').touch()
                        self._wait_for_file(directory / 'endpoint', process)
                        endpoint = (directory / 'endpoint').read_text().strip()
                        self.assertTrue(endpoint.endswith(':' + preload['port']))
                        self.assertIsNone(process.poll())
                        process.terminate()
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertNotEqual(0, process.returncode)  # terminated or startup failed
                    self.assertFalse((directory / 'endpoint').exists(), (stdout, stderr))
                    if mode == 'server_fail':
                        self.assertFalse((directory / 'preload.json').exists())
                    server_pid = int((directory / 'server.pid').read_text())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(server_pid, 0)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.communicate(timeout=5)

    def _wait_for_file(self, path, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                return
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail(f'Startup exited before {path.name}: {stdout}\n{stderr}')
            time.sleep(0.01)
        self.fail(f'Startup did not create {path.name}')
