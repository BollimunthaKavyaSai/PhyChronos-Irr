#!/usr/bin/env bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=../.venv/Scripts/python
while ps -ef | grep -q "[e]3c_dose_response"; do sleep 20; done
for k in 1.5 3; do $PY e8b_forecaster_mpc.py --plants A --ctrls Chronos-BPT,RACER-Irr --kappa $k --tag _k$k > ../results/e8/e8b_k$k.log 2>&1; done
for k in -1.5 -3; do $PY e8b_forecaster_mpc.py --plants A --ctrls Physics --kappa $k --tag _k$k > ../results/e8/e8b_k$k.log 2>&1; done
$PY e8b_forecaster_mpc.py --noise --plants A --ctrls Hysteresis,Physics,Chronos-2,Chronos-BPT,RACER-Irr > ../results/queue2_e8b_noise.log 2>&1
$PY e6b_fault_aug.py --no-physics  > ../results/queue2_e6b_nophys.log 2>&1
echo QUEUE7_DONE
