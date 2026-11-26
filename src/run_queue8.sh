#!/usr/bin/env bash
# Reviewer-driven experiments: lane A (BPT physics sensitivity + independent-plant sweep), lanes B/C (RACER-Irr ablations)
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=../.venv/Scripts/python
laneA() {
  for p in "alpha*0.5" "alpha*2" "c*0.7" "c*1.4" "d*0.5" "d*2" "r=0.2" "r=0.9" "beta=0.2" "beta=0.9"; do
    t=$(echo "$p" | tr '*=' 'xe')
    $PY e8b_forecaster_mpc.py --plants A --ctrls Chronos-BPT --perturb "$p" --tag "_sens_$t" > ../results/e8/sens_$t.log 2>&1
  done
  for k in 1.5 3; do $PY e8b_forecaster_mpc.py --plant-loso --plants A --ctrls Chronos-BPT --kappa $k --tag _plantloso_k$k > ../results/e8/ploso_bpt_k$k.log 2>&1; done
  $PY e8b_forecaster_mpc.py --plant-loso --plants A --ctrls RACER-Irr --kappa 3 --tag _plantloso_racer_k3 > ../results/e8/ploso_racer_k3.log 2>&1
  for k in -3 -6; do $PY e8b_forecaster_mpc.py --plant-loso --plants A --ctrls Physics --kappa $k --tag _plantloso_phys_k$k > ../results/e8/ploso_phys_k$k.log 2>&1; done
  echo LANE_A_DONE
}
laneB() { for v in full nocf nomono nofa; do $PY e12_racer.py $v > ../results/e12_abl_$v.log 2>&1; done; echo LANE_B_DONE; }
laneC() { for v in noband nocaus nosat; do $PY e12_racer.py $v > ../results/e12_abl_$v.log 2>&1; done; echo LANE_C_DONE; }
laneA & laneB & laneC & wait
echo QUEUE8_DONE
