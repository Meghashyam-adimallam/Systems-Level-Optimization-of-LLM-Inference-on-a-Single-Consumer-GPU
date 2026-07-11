#!/usr/bin/env bash
# Fresh git history — run from Terminal.app (NOT Cursor Agent).
#
#   bash scripts/fresh_git_start.sh
#
# Then commit + push yourself (see end of script output).
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Removing old git history..."
rm -rf .git

git init
git branch -M main

git config user.name "Achuth Reddy"
git config user.email "bangaruachuth@gmail.com"

git add -A

echo ""
echo "=== Staged. Now run (Terminal.app only): ==="
echo ""
cat <<'EOF'
git commit -m "Systems-Level Optimization of LLM Inference on a Single Consumer GPU"

git log -1 --format=full
# Must show ONLY your name/email. NO Co-authored-by: Cursor.

git remote add origin git@github.com:Meghashyam-adimallam/Systems-Level-Optimization-of-LLM-Inference-on-a-Single-Consumer-GPU.git
# If remote exists: git remote set-url origin git@github.com:Meghashyam-adimallam/Systems-Level-Optimization-of-LLM-Inference-on-a-Single-Consumer-GPU.git

git push -u origin main --force
EOF
echo ""
echo "If cursoragent STILL shows: GitHub → Settings → Collaborators → Remove cursoragent"
echo "Or delete repo on GitHub, create empty repo same name, push again."
