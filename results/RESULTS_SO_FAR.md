# Final results summary (2026-11-17 evening)

**Paper:** *Accurate Is Not Enough: Counterfactual, Physics-Consistent Foundation-Model Forecasting for Risk-Aware Greenhouse Irrigation*

**Manuscript:** `paper/manuscript.md` · **Figures:** `paper/figures/`

## Proposed methods

| Method | Type | Idea |
|---|---|---|
| **BPT: Behaviour-Policy Transport** (new) | Training-free | Reads the foundation-model forecast as the outcome under the farm's logged policy, then transports it to any candidate dose with the physics model: q(u) = q_FM − p(π_b) + p(u) |
| **RACER-Irr** (new) | Learning algorithm | Reliability-aware counterfactual event-response learning: paired factual and counterfactual doses, plus losses for monotonicity, causality, a physical response band and the saturation range, plus fault-augmented reliability learning |
| PhyChronos-Irr / -FA | Hybrid forecaster | Chronos-2 + dual-store physics + fault features, action-effect fusion head |
| Differentiable dual-store soil-water model | Physics | FAO-56 style; α, c and κ identifiable |
| ACI conformal + chance-constrained planner | Uncertainty and decision | P(θ ≥ MADP) ≥ 0.9, least water |

## Headline numbers

1. **Audit:** the competitor's R² of 0.99 becomes 286 % worse than persistence under leakage-free evaluation.
2. **Forecasting (1,711 origins, 20 models):**
   - Chronos-2 3.040 ≈ TiRex 3.041 ≈ Chronos-2+cov ≈ PhyChronos-Irr 3.09.
   - All foundation models beat the 7 trained deep models plus XGBoost (p < 0.001).
   - RACER-Irr 3.12 (+0.08 vs Chronos-2).
3. **Policy confounding (E3c):** Chronos-2 predicts a response that grows with dose in only 7–62 % of cases, and in sector 2 more water predicts *drier* soil. RACER-Irr is monotone in 100 % of cases (0–24 % without its losses).
4. **Closed loop on the physics plant, 4 sectors, vs the farm rule:**

   | Controller | Water | Stress hours |
   |---|---|---|
   | EDML (competitor) | −82 % | +193 % |
   | Chronos-2 planner | −95 % | +188 % |
   | **BPT (κ = +3)** | **−26 %** | **−95 %** |
   | BPT (κ = +1.5) | −29 % | −87 % |
   | RACER-Irr (κ = +3) | −27 % | −85 % |
   | Physics (κ = −6) | −27.5 % | −80 % |
   | Physics (κ = −4.5) | −21 % | −95 % |

   BPT lies on or below the physics frontier: at equal water it has 37 % fewer stress hours, and at equal stress it uses 6 % less water.
5. **Noise:** BPT −27 % water / −63 % stress; RACER-Irr −34 % / −56 %; the ranking is preserved.
6. **Faults:**
   - Fault augmentation gives robustness comparable to Chronos-2, and a significant gain under stuck sensors.
   - Coverage under stuck sensors: 0.28 vs 0.12 for Chronos-2.
   - The physics branch does not add drift robustness (reported honestly).
7. **Calibration:** ACI gives 0.80–0.81 coverage for all 7 probabilistic models.
8. **Generalisation:**
   - Unseen sector: PhyChronos 3.66 vs Chronos-2 3.55.
   - Post-season: XGBoost collapses (5–16× worse than persistence), while the hybrid is the best learned model.
   - External farm: Chronos-2+cov skill 51 %.
9. **Simulator audit:** the data-driven simulator reverses the sign of the water effect, so it is invalid for testing interventions.

## Honest limitations

- One greenhouse and one season.
- Simulation-based control, on a plant that shares the physics model family.
- BPT assumes the behaviour policy is known.
- In sector 5 there is no observed response to water, but the physics model predicts one.
- The relative probe scale has no volumetric calibration.
- References marked † still need checking.
