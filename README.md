# PhyChronos-Irr: implementation, data and results

This folder contains the code, datasets and outputs for the paper *Physics-Guided Counterfactual Forecasting with Time-Series Foundation Models for Water-Efficient Greenhouse Irrigation* (AgriEngineering submission).

## Folder layout

| Path | Contents |
|---|---|
| `src/` | All implementation code (one script per experiment; see the table below) |
| `data/raw/primary_c837v6p8ph/extracted/` | Main dataset: Arnesano greenhouse, 2025, Mendeley Data https://doi.org/10.17632/c837v6p8ph.2 |
| `data/raw/secondary/cjb4vy4mzj/` | External-farm dataset, Mendeley Data https://data.mendeley.com/datasets/cjb4vy4mzj |
| `data/processed/` | Corrected hourly data produced by `build_dataset.py` (`zone_1..5.csv`) and `fetch_weather.py` (`era5_hourly_extra.csv`) |
| `results/` | All experiment outputs (CSV files and logs); `results/e2/preds/` holds every model's forecasts |
| `paper/figures/` | All figures used in the manuscript |
| `external/Time-Series-Library/` | Third-party library used for the deep baselines (github.com/thuml/Time-Series-Library) |
| `requirements.txt` | Python packages and the exact versions used |

## Setup

Use Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

The foundation models are downloaded automatically from Hugging Face on first use:
- `amazon/chronos-2`
- `google/timesfm-2.5-200m-pytorch`
- `Salesforce/moirai-2.0-R-small`
- `NX-AI/TiRex`

After the first download, set `HF_HUB_OFFLINE=1` to run offline. Run every script from inside `src/`. The scripts find all paths relative to this folder.

## Scripts

| Step / experiment | Script(s) | Main outputs |
|---|---|---|
| Configuration (paths, agronomic thresholds, seed) | `config.py` | – |
| C1: corrected dataset | `build_dataset.py`, `fetch_weather.py` | `data/processed/` |
| E1: protocol audit / EDML reproduction | `e1_reproduce.py [--season]` | `results/e1/` |
| E2: forecasting benchmark | `e2_panel.py` (protocol), `e2_baselines.py`, `e2_foundation.py <chronos2\|chronos2cov\|timesfm\|moirai2\|tirex>`, `e2_deep.py <models>`, `e2_eval.py` | `results/e2/` |
| C2/E3: dual-store physics model | `physics.py`, `e3_physics.py` | `results/e3/params_*.csv` |
| C2: PhyChronos-Irr hybrid (+ E9 ablations) | `e3_hybrid.py [--no-physics] [--no-fault] [--single]` | `results/e2/preds/` |
| E3b/E3c: action sensitivity, dose-response | `e3b_action_sensitivity.py`, `e3c_dose_response.py` | `results/e3/` |
| E4: identifiability | `e4_identifiability.py` | `results/e3/e4_*.csv` |
| E5: unseen sector, post-season | `e5_generalization.py` | `results/e5/` |
| E6: sensor faults, fault augmentation | `e6_faults.py`, `e6b_fault_aug.py [--no-physics]` | `results/e6/` |
| E7: conformal calibration | `e7_conformal.py <models>` | `results/e7/` |
| E8: closed-loop control (simulators, reference controllers) | `e8_control.py --validate`, `e8_refs.py` | `results/e8/` |
| C3/C4: same planner with different forecasters, BPT, risk sweep, noise | `e8b_forecaster_mpc.py [--ctrls ...] [--plants A] [--kappa k] [--noise] [--tag t]`, `e8b_summary.py`, `e8_pareto.py` | `results/e8/` |
| E8: independently calibrated plant (simulator sensitivity check) | `e8b_forecaster_mpc.py --plant-loso --plants A --ctrls Fixed-interval,Hysteresis,EDML,Physics,Chronos-2,Chronos-BPT,RACER-Irr --tag _plantloso` | `results/e8/e8b_results_plantloso*.csv` (Tables 18 and 20); add `--kappa k` for the risk sweep |
| E8: sensitivity of BPT to physics-model errors | `e8b_forecaster_mpc.py --plants A --ctrls Chronos-BPT --perturb alpha*0.5 --tag _sens_alphax0.5` (also `alpha*2`, `c*0.7`, `c*1.4`, `d*0.5`, `d*2`, `r=0.2`, `r=0.9`, `beta=0.2`, `beta=0.9`) | `results/e8/e8b_results_sens_*.csv` (Table 21) |
| E13: Ensemble-BPT (BPT on the five-TSFM ensemble) | `e13_ensemble.py` (benchmark and bootstrap), `run_queue9.sh` (closed loop, `--ctrls Ensemble-BPT`) | `results/e2/e13_ensemble_stats.csv`, `results/e2/preds/TSFM-Ensemble.csv.gz`, `results/e8/e8b_results_*ensbpt*.csv` |
| E10: data efficiency | `e10_data_efficiency.py` | `results/e10_data_efficiency.csv` |
| E11: external farm | `e11_external.py` | `results/e11/` |
| E12: RACER-Irr (C4) and component ablations | `e12_racer.py [full\|nocf\|nomono\|noband\|nocaus\|nosat\|nofa]`, then `e2_eval.py` | `results/e12/racer_dose.csv`, `results/e2/e12_ablation_metrics.csv`, `results/e2/preds/` |
| Statistics (block bootstrap) | `stats_bootstrap.py` | `results/stats_*.csv` |
| Figures | `figures.py`, `fig_architecture.py`, `fig_clear.py`, `fig_review.py` | `paper/figures/` |
| Job queues used on the 8 GB machine | `run_queue*.sh` | logs in `results/` |

## Where the proposed methods are implemented
- **Behaviour-Policy Transport (BPT):** in `src/e8b_forecaster_mpc.py`, function `behaviour_policy` and the controller `Chronos-BPT` in `lower_bounds`.
- **RACER-Irr:** in `src/e12_racer.py`, functions `cf_set`, `racer_loss` and `train_racer`.
- **PhyChronos-Irr fusion head:** in `src/e3_hybrid.py`, class `Head`.
- **Dual-store soil-water model:** in `src/physics.py`, class `Bucket`.

The random seed is 42 throughout.

## Suggested order to reproduce everything
1. `build_dataset.py`, then `fetch_weather.py`.
2. `e1_reproduce.py` and `e1_reproduce.py --season`.
3. `e2_baselines.py`, `e2_foundation.py` (one call per model), `e2_deep.py`, then `e3_physics.py` and `e3_hybrid.py`.
4. `e3b`, `e3c`, `e4`, `e5`, `e6`, `e6b`, `e7`, `e10`, `e11` and `e12`, then `e2_eval.py` and `stats_bootstrap.py`.
5. `e8_control.py --validate`, `e8_refs.py`, `e8b_forecaster_mpc.py` (the runs described in the paper), `e8b_summary.py` and `e8_pareto.py`.
6. `figures.py` and `fig_architecture.py`.

Datasets: CC BY 4.0, by their original authors (see the Data Availability Statement in the paper).
