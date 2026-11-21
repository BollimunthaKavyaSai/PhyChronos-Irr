#!/usr/bin/env bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=../.venv/Scripts/python
$PY e8b_forecaster_mpc.py          > ../results/e8/e8b_log.txt 2>&1
$PY e3c_dose_response.py           > ../results/queue2_e3c.log 2>&1
$PY e8b_forecaster_mpc.py --noise  > ../results/queue2_e8b_noise.log 2>&1
$PY e6b_fault_aug.py --no-physics  > ../results/queue2_e6b_nophys.log 2>&1
$PY e6b_fault_aug.py               > ../results/queue2_e6b_fa.log 2>&1
echo QUEUE3_DONE
