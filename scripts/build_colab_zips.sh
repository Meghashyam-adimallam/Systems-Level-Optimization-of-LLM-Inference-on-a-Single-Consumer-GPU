#!/usr/bin/env bash
# Rebuild Colab upload zips from current repo (excludes results, tiers, private files).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pack() {
  local name="$1"
  local out="v${name}/InferenceLab_v${name}.zip"
  rm -f "$out"
  zip -r "$out" \
    server benchmark scripts \
    sitecustomize.py \
    requirements.txt requirements-vllm.txt \
    docs/project_flow.html docs/tail_latency_analysis.md docs/v3_sla_findings.md docs/COLAB_SLA_RERUN.md \
    -x "*__pycache__*" "*.pyc" "*DS_Store*"
  echo "Wrote $out"
}

pack 2
pack 3
echo "Done. Upload InferenceLab_v2.zip / InferenceLab_v3.zip with matching notebooks."
