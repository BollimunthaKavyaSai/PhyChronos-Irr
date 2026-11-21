#!/usr/bin/env bash
# Sequential queue for memory-heavy jobs (8 GB RAM machine). Logs in results/queue_*.log
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=../.venv/Scripts/python
$PY e11_external.py            > ../results/queue_e11.log 2>&1
$PY e5_generalization.py       > ../results/queue_e5.log 2>&1
$PY e2_foundation.py tirex     > ../results/queue_tirex.log 2>&1
echo QUEUE_DONE
