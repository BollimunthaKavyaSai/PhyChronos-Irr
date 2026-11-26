#!/bin/bash
# E13: BPT with the five-TSFM ensemble backbone in closed loop (main plant and independently calibrated plant)
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
L=../results/e8
( $PY e8b_forecaster_mpc.py --plants A --ctrls Ensemble-BPT --tag _ensbpt > $L/ensbpt_k0.log 2>&1
  $PY e8b_forecaster_mpc.py --plants A --ctrls Ensemble-BPT --kappa 3 --tag _ensbpt_k3 > $L/ensbpt_k3.log 2>&1
  echo LANE_A_DONE >> $L/queue9.status ) &
( $PY e8b_forecaster_mpc.py --plant-loso --plants A --ctrls Ensemble-BPT --tag _plantloso_ensbpt > $L/ploso_ensbpt_k0.log 2>&1
  $PY e8b_forecaster_mpc.py --plant-loso --plants A --ctrls Ensemble-BPT --kappa 3 --tag _plantloso_ensbpt_k3 > $L/ploso_ensbpt_k3.log 2>&1
  echo LANE_B_DONE >> $L/queue9.status ) &
wait
echo QUEUE9_DONE >> $L/queue9.status
