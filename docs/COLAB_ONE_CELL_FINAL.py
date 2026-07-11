# v3 SLA — paste into one Colab cell (A100, InferenceLab_v3.zip in /content)

import os, shutil, zipfile, json, subprocess, time, sys, urllib.request
from pathlib import Path

os.chdir('/content')
assert os.path.exists('InferenceLab_v3.zip'), 'Upload InferenceLab_v3.zip first'

shutil.rmtree('/content/inference_lab', ignore_errors=True)
with zipfile.ZipFile('InferenceLab_v3.zip') as z:
    z.extractall('/content/inference_lab')
ROOT = '/content/inference_lab'
os.chdir(ROOT)

os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
!pip install -q -r requirements.txt
!pip uninstall -y vllm transformers tokenizers 2>/dev/null
!pip install -q "vllm==0.8.5" "transformers==4.51.3" "tokenizers<0.22"

env = os.environ.copy()
env.update({
    'PYTHONPATH': ROOT,
    'VLLM_MODEL': 'zephyr-7b',
    'SLA_P95_BUDGET_SEC': '3.0',
    'SLA_MAX_QUEUE': '32',
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
})

check = subprocess.run(
    [sys.executable, '-c',
     'from vllm import AsyncLLMEngine; import google.protobuf; '
     'print("OK protobuf", google.protobuf.__version__)'],
    capture_output=True, text=True, env=env, cwd=ROOT,
)
print(check.stdout.strip() or check.stderr.strip())
assert check.returncode == 0, 'vllm import failed'

logf = open('/content/server_sla.log', 'w')
srv = subprocess.Popen(
    ['uvicorn', 'server.sla_server:app', '--host', '127.0.0.1', '--port', '8000'],
    env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT,
)
t0 = time.time()
while time.time() - t0 < 1800:
    if srv.poll() is not None:
        print(''.join(open('/content/server_sla.log').readlines()[-50:]))
        raise RuntimeError('Server died during load')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as r:
            if json.loads(r.read()).get('status') == 'ok':
                print('Server ready')
                break
    except Exception:
        pass
    if int(time.time() - t0) % 30 == 0 and time.time() - t0 > 0:
        print(f'  loading... {int(time.time() - t0)}s')
    time.sleep(3)
else:
    raise RuntimeError('Server load timed out')

import httpx
smoke = httpx.post('http://127.0.0.1:8000/generate',
    json={'prompt': 'Say OK.', 'max_new_tokens': 8}, timeout=120.0)
assert smoke.status_code == 200, f'Smoke failed: {smoke.status_code}'

!cd /content/inference_lab && PYTHONPATH=/content/inference_lab \
  VLLM_MODEL=zephyr-7b SLA_P95_BUDGET_SEC=3.0 SLA_MAX_QUEUE=32 \
  python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 \
  --strategy sla --runs 2 --monitor-gpu --capstone

srv.terminate()
try:
    srv.wait(timeout=15)
except subprocess.TimeoutExpired:
    srv.kill()
logf.close()

for f in sorted(Path('results').glob('load_sla_*.json')):
    print(f.name)
