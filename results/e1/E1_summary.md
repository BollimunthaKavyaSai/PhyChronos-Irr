# E1: Reproduction and protocol correction (2026-11-17)

Code: `src/build_dataset.py`, `src/e1_reproduce.py`. Raw numbers: `e1_results_{full,season}.csv`.
Target: mean soil moisture over the next 24 h (as in Adamo et al. 2026). Model: XGBoost with a monotone constraint on planned water, using the competitor's 15 features.

## 1. Reproduction succeeds (published data, their 3-day stratified group CV, full period)
| Sector | Paper R² | Ours R² | Paper MAE | Ours MAE |
|---|---|---|---|---|
| 1 | 0.89 | 0.97 | 1.71 | 1.10 |
| 2 | 0.99 | 0.97 | 0.60 | 0.74 |
| 3 | 0.65 | 0.72 | 3.69 | 4.56 |
| 4 | 0.92 | 0.92 | 2.07 | 1.73 |
| 5 | 0.98 | 0.99 | 2.08 | 1.85 |

## 2. The headline accuracy depends on the protocol (mean over sectors 1, 2, 4, 5)
| Data | Protocol | Period | Persistence MAE | XGBoost MAE | Skill vs persistence |
|---|---|---|---|---|---|
| published | 3-day group CV (shuffled) | full | 3.12 | **1.35** (R² 0.96) | +57 % |
| published | rolling-origin + 24 h embargo | full | 2.13 | 9.34 (R² −0.02) | **−286 %** |
| rebuilt | rolling-origin + 24 h embargo | full | 2.76 | 6.87 | −181 % |
| rebuilt | 3-day group CV | in-season | 4.90 | 2.03 | +58 % |
| rebuilt | rolling-origin + 24 h embargo | **in-season (main protocol)** | 4.63 | 2.60 (delta target) | **+41 %** |

## 3. Findings to report in the paper
1. **Shuffled block CV overstates accuracy.** Its R² is inflated because the variance between seasons (in-season vs post-season levels) makes R² easy to achieve. Within the season, the same CV gives R² of 0.45–0.65.
2. **Tree models fail when the regime shifts.** Irrigation stops in July (sectors 1, 2) or September (4, 5). Chronological test windows then fall in a post-season drying regime that the training data never contains, and XGBoost does 3–60× worse than persistence. This is a natural distribution-shift case, used later for E5/E7.
3. **Main protocol:** active irrigation season (first event to last event + 2 days), rolling-origin with 4 folds over the last 40 % of the season and a 24 h embargo. Under this protocol XGBoost still beats persistence by 34–41 %.
4. **R² is unreliable within the season** (it goes negative while MAE skill is +41 %), because short test windows have low variance. Report MAE/RMSE and skill as primary metrics; NSE/KGE only as supplementary.
5. **Planned water adds no information to the learned model.** Removing `water_to_24h` slightly improves MAE (2.54 vs 2.60), consistent with audit finding A8. A purely data-driven water→moisture "lever", which the competitor's MILP relies on, is weakly identified. This justifies a physics prior for the decision layer (C2/C4).
6. EC/pH help a little (removing them: 2.86 vs 2.61), but they come from a co-located probe and fail together with the moisture sensor (relevant for E6).

## 4. Fixes adopted in the rebuilt dataset
Rain ÷ 6; true-time targets with ≥ 80 % coverage; 3 % deadband gaps filled up to 6 h and flagged as censored; flatlines ≥ 24 h masked (readings are integer-quantised, so shorter constant runs are normal); sector 3 masked after probe death (2025-06-25); EC/pH error codes removed.
