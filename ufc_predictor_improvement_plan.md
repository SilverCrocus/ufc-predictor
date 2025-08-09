# UFC Predictor — Improvement Plan

**Audience:** Another LLM/engineer to implement.

**Objective:** Improve predictive performance, probability calibration, and real-money betting performance (ROI, CLV) using trustworthy evaluation (walk‑forward), richer features, better modeling (stacking + survival/competing risks), and risk-aware staking. Maintain production quality (tests, monitoring, reproducibility).

---

## 0) Outcomes & Success Criteria

### Primary KPIs (must report per release)

- **Calibration**: Brier Score ↓ 5–10%, ECE < 0.03, reliability curves by division.
- **Market Baseline**: Beat closing line **>55%** of recommended bets (CLV win rate). Average CLV ≥ +1.5% vs close.
- **Backtested ROI**: Positive ROI in walk‑forward simulation with slippage; Max drawdown ≤ 25% of bankroll under quarter-Kelly.
- **Predictive**: Log loss ↓ vs current baseline; AUC ↑; overall accuracy secondary.

### Guardrails

- No data leakage. All model selection and calibration use temporal OOF (out‑of‑fold) predictions.
- Reproducible seeds, deterministic runs where applicable; document non-deterministic parts (e.g., LightGBM multi-threading).

---

## 1) Project Structure Additions

Create new modules and folders to keep work isolated.

```
ufc-predictor/
├── configs/
│   ├── backtest.yaml                 # parameters for simulation
│   └── features.yaml                 # global feature flags/window sizes
├── src/ufc_predictor/
│   ├── evaluation/
│   │   ├── temporal_split.py         # walk-forward split utilities
│   │   ├── metrics.py                # brier, ece, clv, calibration curves
│   │   └── calibration.py            # isotonic/platt by segment
│   ├── features/
│   │   ├── context_features.py       # venue altitude, apex cage, short notice, tz delta
│   │   ├── matchup_features.py       # stance matrix, reach/height splines, weight-class change
│   │   ├── quality_adjustment.py     # opponent-strength normalized per-round stats
│   │   └── rolling_profiles.py       # decay-weighted rolling features
│   ├── models/
│   │   ├── stacking.py               # OOF stacking meta-learner
│   │   ├── bradley_terry.py          # BT/elo-logistic head-to-head layer
│   │   ├── survival_moi.py           # method-of-victory via competing risks
│   │   └── registry.py               # save/load with versioning
│   ├── betting/
│   │   ├── odds_utils.py             # vig removal, implied prob conversions
│   │   ├── staking.py                # pessimistic kelly, caps, portfolio optimizer
│   │   ├── correlation.py            # leg correlation estimation
│   │   └── simulator.py              # walk-forward bankroll sim, slippage, limits
│   ├── monitoring/
│   │   ├── dashboards.py             # static plots, json summaries
│   │   └── drift.py                  # calibration drift, ROI drift
│   └── utils/
│       └── time_index.py             # event-date indexing helpers
└── notebooks/
    ├── 01_walkforward_eval.ipynb
    ├── 02_calibration_plots.ipynb
    └── 03_betting_simulation.ipynb
```

---

## 2) Data Contracts & Inputs

### Required columns in the modeling dataset

- **Identifiers**: `event_id`, `fight_id`, `date` (YYYY-MM-DD), `division`, `rounds` (3 or 5)
- **Fighters**: `fighter_a`, `fighter_b`, `winner` (A/B), `method` (KO/TKO/DEC/SUB), `finish_round`, `finish_time_sec`
- **Odds**: `open_odds_a`, `open_odds_b`, `close_odds_a`, `close_odds_b`, `book` (TAB/FightOdds), timestamps
- **Features (existing)**: The 70+ differential features already in your pipeline
- **New features (see §3)**: context/matchup/quality/rolling fields

### Data quality rules (failing rows should be dropped with reason logged)

- Missing `date`, `fighter_a`, `fighter_b`, or `winner` → drop.
- `open_odds_*` or `close_odds_*` not both present → mark `odds_available=False` for CLV calc, keep for modeling.
- Inconsistent \(odds for A × odds for B\) within 0.01 of fair two-way market after vig removal → log.

---

## 3) Feature Engineering (New)

Implement these as pure functions operating on a pandas DataFrame. Feature flags in `configs/features.yaml` toggle each block.

### 3.1 Contextual Features

- `is_apex_small_cage`: bool (Apex events)
- `is_altitude_high`: bool (venue elevation ≥ 1500m; allow 1000m/1500m threshold config)
- `short_notice_flag`: bool (fighter accepted bout ≤ 14 days; if unknown, impute False)
- `time_zone_delta_hours`: absolute hours between fighter’s home (last camp loc if available) and venue
- `is_5_round_main`: bool; `is_title_fight`: bool

### 3.2 Matchup Interaction Features

- `stance_combo`: categorical {O-S, O-O, S-S, Switch-\*}; one-hot
- `reach_diff_spline`: piecewise linear bins: [≤3cm], (3–7], (7–12], (12+]
- `height_diff_bins`: similar piecewise bins
- `weight_class_change`: {down, up, same} based on last fight’s division
- `age_curve_feature`: spline on age with division-specific knots

### 3.3 Opponent Quality Normalization

- Build `opponent_strength` via Bradley–Terry/Elo up to previous fight date.
- Normalize per-round stats: e.g., `adj_strikes_lpm = raw_strikes_lpm * f(opponent_strength)`.
- Provide both raw and adjusted variants; meta-model will weight.

### 3.4 Rolling & Decay-Weighted Profiles

- For each fighter stat, compute decay-weighted means over last `N={3,5}` fights with factor `DECAY=0.90–0.95`.
- Include late-round performance deltas (e.g., R3 strike differential minus R1).

**Acceptance Criteria:** All new features are deterministic, unit-tested, and documented. Feature generation runs under 60s for full dataset on 4 cores.

---

## 4) Evaluation: Temporal Walk‑Forward

### 4.1 Split Strategy

- Sort by `date`. Build folds yearly or by `K` events:
  - Train: start→T, Validate: T→T+Δ (development tuning), Test: T+Δ→T+2Δ (report) in a rolling window.
- Keep **all rematches** within the same fold.
- Stratify by `division` and `rounds` if fold sizes allow.

### 4.2 Metrics to Report (per fold and aggregated)

- **Classification**: Log loss, Brier, AUC, Accuracy.
- **Calibration**: ECE (10-bin), reliability curves, per-division ECE.
- **Market**: CLV win rate, mean CLV (%), hit rate of recommended bets.
- **Betting** (from simulator): ROI, Sharpe ≈ mean/Std, Max Drawdown, turnover.

**Acceptance Criteria:** A single command runs the full walk‑forward evaluation and writes JSON + PNG outputs to `artifacts/eval/YYYYMMDD/`.

---

## 5) Modeling Improvements

### 5.1 Head-to-Head Layer (Bradley–Terry / Elo-Logistic)

- Train a BT model on historical outcomes up to each fold’s train end.
- Output feature: `bt_prob_a_beats_b` used as an input to main classifiers.

### 5.2 Stacking Ensemble (OOF)

- Base learners: RF, XGBoost, LightGBM (existing).
- Generate **OOF predictions** for each base learner in walk-forward scheme.
- Meta-learner: `LogisticRegression(C=1.0)` on OOF probs → final `win_prob_a`.
- Save class-wise predicted probabilities and uncertainty (std across learners + bootstrap if available).

### 5.3 Method-of-Victory via Competing Risks (Survival)

- Fit hazard models for KO/Sub vs Decision (non-finish) using discrete-time survival (per round) or CoxPH with cause-specific hazards.
- Derive `P(KO)`, `P(Sub)`, `P(Dec)` s.t. they sum to 1; ensure monotonicity over time.

### 5.4 Probability Calibration

- Apply **isotonic calibration** trained on temporal OOF predictions.
- Calibrate **per segment**: male/female × division groups or 3R/5R.
- Store calibrators with version tags `calib_{segment}_{date}.joblib`.

**Acceptance Criteria:**

- Stacked model improves log loss ≥ 2% vs best single base on average across folds.
- Calibrated ECE < 0.03 overall, < 0.05 for each major division bucket.

---

## 6) Betting Layer Upgrades

### 6.1 Odds Processing

- Implement vig removal to derive **fair implied probabilities** from two-way markets.
- Support both open and close; if multiple books, store per-book and compute a consensus.

### 6.2 Pessimistic Kelly Staking

- From bootstrap/meta-ensemble uncertainty, compute a lower-bound `p_low` (e.g., 20th percentile) and use it in Kelly.
- Cap per-bet exposure by config (default ≤ 5% single, ≤ 2% multi), and by uncertainty (higher var → smaller cap).

### 6.3 Parlay Correlation

- Estimate correlation between legs using shared-feature cosine similarity + historical co‑movement of model residuals.
- If correlation > threshold, either reduce combined EV by factor `f(rho)` or **disallow** the parlay.

### 6.4 Portfolio Allocation

- Build a per-card optimizer: select bets that maximize expected log-growth subject to caps and correlation constraints.
- Export a card-level ticket plan with stakes and confidence bands.

**Acceptance Criteria:** Simulator shows improved risk-adjusted returns vs baseline (higher Sharpe or same Sharpe with lower drawdown) over the last 3 years.

---

## 7) Backtesting Simulator (Walk‑Forward)

### 7.1 Mechanics

- For each test fold date range:
  - Freeze model and calibrator from training window.
  - Pull **available odds at recommendation time** (use open by default; optionally mid‑line).
  - Apply staking policy to produce bets.
  - Apply **slippage model**: when evaluating, replace stake odds with min(close odds, recommended odds + slip), configurable.
  - Apply bookmaker **limits** and min bet sizes (configurable).

### 7.2 Outputs

- Per-card and cumulative equity curve.
- ROI, turnover, CLV distribution, drawdown chart.
- JSON summary for dashboards.

**Acceptance Criteria:** `python -m ufc_predictor.betting.simulator --config configs/backtest.yaml` produces artifacts and summary JSON; unit tests cover slippage and Kelly edge cases.

---

## 8) Monitoring & Reporting

- Generate static dashboards (matplotlib) for:
  - Calibration reliability plots overall and by division/rounds.
  - Equity curve, drawdown, distribution of EV vs realized.
  - CLV trend over time.
- Write `artifacts/monitoring/summary_{YYYYMMDD}.json` with KPIs for CI to pick up.

---

## 9) Testing Strategy

### Unit Tests (add to `tests/unit/`)

- `test_temporal_split.py`: no leakage; rematches stay within fold; dates monotonic.
- `test_calibration.py`: ECE decreases after calibration on synthetic set.
- `test_odds_utils.py`: vig removal correctness.
- `test_staking.py`: Kelly with p\_low, caps, and zero/negative edges.
- `test_features_context.py`: deterministic feature outputs given inputs.

### Integration Tests (add to `tests/integration/`)

- End‑to‑end walk-forward run on a small, fixed sample dataset (5 events) → snapshot JSON.
- Simulator with known bets → expected bankroll path.

**Coverage Target:** ≥ 80% for new modules.

---

## 10) Configs (initial suggestions)

``

```yaml
context:
  apex_small_cage: true
  altitude_threshold_m: 1500
  short_notice_days: 14
  time_zone_delta: true
matchup:
  stance_combo: true
  reach_spline_bins: [3, 7, 12]
  height_spline_bins: [3, 7, 12]
  weight_class_change: true
  age_curve_by_division: true
quality_adjustment:
  enabled: true
  method: bradley_terry
rolling:
  windows: [3, 5]
  decay: 0.93
```

``

```yaml
folding:
  mode: rolling
  train_years: 5
  dev_years: 0
  test_years: 1
odds:
  use: open
  slippage_bp: 25   # basis points
  consensus: false
staking:
  kelly_fraction: 0.25
  p_lower_quantile: 0.20
  max_single_pct: 0.05
  max_multi_pct: 0.02
  allow_parlays: false
limits:
  max_stake_per_bet: 500
  min_stake: 5
```

---

## 11) Function Signatures (Guidance)

```python
# evaluation/temporal_split.py
def make_rolling_folds(df, date_col: str, years_train: int, years_test: int, group_cols=("division", "rounds")) -> list:
    """Return list of (train_idx, test_idx) respecting chronology and group balance."""

# evaluation/metrics.py
def brier_score(y_true, p_true): ...
def expected_calibration_error(y_true, p_true, n_bins=10): ...

def clv_stats(reco_odds, close_odds, stakes): ...  # return clv_win_rate, avg_clv

# evaluation/calibration.py
def fit_isotonic_by_segment(oof_df, segment_cols=("division","rounds")) -> Dict[str, Any]: ...

def apply_calibration(prob_df, calibrators, segment_cols) -> np.ndarray: ...

# models/bradley_terry.py
def fit_bt(train_df) -> Any: ...

def predict_bt_prob(bt_model, pairs_df) -> np.ndarray: ...

# models/stacking.py
def fit_stack(base_models: Dict[str, Any], X_train, y_train, folds) -> Tuple[Any, Dict[str, Any]]: ...  # returns meta, fitted_bases

def predict_stack(meta, bases, X): ...

# models/survival_moi.py
def fit_competing_risks(train_df) -> Any: ...

def predict_method_probs(model, X_pairs) -> np.ndarray: ...  # shape [n, 3]

# betting/odds_utils.py
def remove_vig_two_way(odds_a: float, odds_b: float) -> Tuple[float,float]: ...  # fair probs

# betting/staking.py
def kelly_pessimistic(p_mean, p_low, odds, kelly_fraction=0.25, cap=0.05) -> float: ...

# betting/simulator.py
def run_backtest(df, config) -> Dict[str, Any]: ...
```

---

## 12) CLI Additions

```
# Full walk-forward eval + backtest
python scripts/main.py --mode eval --config configs/backtest.yaml

# Train & persist stack + calibrators for latest window
python scripts/main.py --mode train_stack --features configs/features.yaml

# Generate dashboards from latest artifacts
python scripts/main.py --mode report --artifacts artifacts/eval/2025XXXX/
```

---

## 13) Rollout Plan

**Phase 1 (Week 1–2):** Temporal CV + metrics + calibration (OOF). Baseline artifacts and dashboards.

**Phase 2 (Week 2–3):** New features (context, matchup, rolling, quality). Re‑eval.

**Phase 3 (Week 3–4):** Stacking meta‑learner + BT layer. Re‑eval.

**Phase 4 (Week 4–5):** Betting upgrades (vig removal, pessimistic Kelly, portfolio). Backtesting with slippage/limits.

**Phase 5 (Week 5):** Documentation, test hardening, CI job, champion/challenger shadow run.

---

## 14) Deliverables Checklist

-

---

## 15) Documentation Notes to Include in Code

- Every module: short docstring, input/output schema, deterministic notes.
- `README` section: how to reproduce the latest evaluation; link to artifacts.
- Version tags: data hash, feature flags, model versions, calibrator versions.

---

## 16) Acceptance Criteria Recap

- ECE < 0.03 overall; CLV win rate > 55%; positive walk-forward ROI with drawdown ≤ 25%; reproducible artifacts and comprehensive tests.

---

**End of Plan**

