# Colab v3 — SLA benchmark

**Runtime:** A100 GPU · **Notebook:** `v3/colab_run_v3.ipynb` · **Zip:** `v3/InferenceLab_v3.zip`

Rebuild zip locally: `bash scripts/build_colab_zips.sh`

## Run order

| Cell | Action |
|------|--------|
| 1 | GPU check |
| 2 | Upload `InferenceLab_v3.zip` |
| 3 | Unzip → `/content/inference_lab` |
| 4 | Install vLLM 0.8.5 + transformers 4.51.3 |
| 5 | Helpers |
| 6 | SLA benchmark (4 loads, 2 runs) |
| 7–9 | Charts + download zip |

## Env (set in notebook)

- `VLLM_MODEL=zephyr-7b`
- `SLA_P95_BUDGET_SEC=3.0`

## Success criteria

JSON in `results/load_sla_*.json` must have:

- `"sla_policy": "reject_e2e_v2"`
- `failed_requests_total` > 0 on medium/heavy/long_context (503 shedding)

Copy JSON + `gpu_traces/` into `v3/results/`, then run `python scripts/generate_tier_charts.py` locally.

## Troubleshooting

- **Upload fails:** `os.chdir('/content')` then re-run upload cell.
- **Server not ready:** wait for `/health` → `status=ok` (Zephyr load ~5–10 min). Check `!tail -60 /content/server_sla.log`.
- **Wrong paths:** use `/content/inference_lab`, not `/content/scripts` or Gemini-invented folders.
