# UFC Predictor — Post-Implementation Improvement Plan

**Context:** After implementing the initial plan, the tuned RandomForest winner model achieved 73.49% accuracy (up from 73.25%), and the tuned method model held at 74.91%. These are modest improvements in accuracy. The next phase should focus on calibration, probability quality, stacking ensembles, and betting performance.

---

## 0) Objectives & Success Criteria

### Goals

* Improve **probability calibration** to reduce Brier score and Expected Calibration Error (ECE).
* Increase **log loss improvement** over baseline by ≥ 5%.
* Achieve **CLV win rate > 55%** and positive backtested ROI with realistic constraints.
* Enhance betting performance metrics (ROI, Sharpe ratio, drawdown control).

### Guardrails

* Maintain strict temporal validation.
* Ensure reproducibility and document randomness sources.

---

## 1) Evaluation & Diagnostics

1. **Run calibration diagnostics** (Brier, ECE, reliability curves) for both models.
2. **Feature impact analysis** using SHAP or permutation importance to identify top-performing features.
3. **Market baseline comparison** — Compare predictions to vig-free implied probabilities from closing odds.
4. **Walk-forward evaluation** — Verify all results using temporal folds.

---

## 2) Modeling Enhancements

### 2.1 Stacking Ensemble

* Base models: tuned RandomForest, XGBoost, LightGBM.
* Generate out-of-fold predictions in a temporal scheme.
* Meta-model: LogisticRegression for winner prediction, calibrated per segment.

### 2.2 Method-of-Victory Improvement

* Implement competing risks survival modeling for KO/Sub vs Decision.
* Compare method model performance in log loss and class-wise calibration.

### 2.3 Calibration Refinement

* Apply isotonic regression per segment (division, gender, 3R/5R).
* Save and version calibrators.

---

## 3) Feature Refinement

1. **Prune non-contributing features** identified in feature impact analysis.
2. Test new contextual features: fight-week weigh-in result flags, historical round-by-round strike differential trends.
3. Add opponent-quality adjusted stats if not already in place.

---

## 4) Betting Strategy Improvements

1. **Pessimistic Kelly staking** using lower-bound probability from bootstrap/meta-model variance.
2. Enforce **exposure caps** by uncertainty and fight importance.
3. **CLV tracking** — Log closing odds for every bet to measure value capture.
4. Implement **portfolio optimization** for each card with correlation-aware constraints.

---

## 5) Backtesting & Simulation

1. Backtest updated models with walk-forward folds.
2. Apply realistic constraints (odds slippage, limits, minimum bet sizes).
3. Track equity curve, drawdown, ROI, and CLV.
4. Compare results with and without stacking.

---

## 6) Monitoring

* Create dashboards for:

  * Calibration drift.
  * CLV trend.
  * ROI and equity curve.
  * Feature importance over time.

---

## 7) Timeline

**Week 1–2:** Calibration diagnostics, feature impact analysis, market comparison.
**Week 2–3:** Implement stacking ensemble and calibrators.
**Week 3–4:** Enhance method-of-victory modeling.
**Week 4–5:** Betting strategy refinements and backtesting.
**Week 5:** Dashboard integration, documentation, CI automation.

---

## 8) Acceptance Criteria

* Winner model log loss improved ≥ 5% vs current tuned RF.
* ECE < 0.03 overall; per-division ECE < 0.05.
* Positive walk-forward ROI, CLV win rate > 55%.
* Reproducible results with artifact versioning.

---

**End of Plan**
