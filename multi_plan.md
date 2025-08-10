# UFC Predictor — Conditional Multibets Plan (When Singles Don’t Qualify)

**Purpose:** On cards where strict single‑bet filters (calibrated EV 5–15% + market gap) yield 0–1 opportunities, selectively construct low‑correlation 2‑leg parlays from small‑edge legs to deploy capital without compromising risk discipline.

---

## 0) Outcomes & Guardrails

**Primary KPIs**

* **Parlay CLV ≥ 55%** over a rolling 100‑parlay window.
* **Parlay ROI > 0** in walk‑forward backtests with slippage and limits.
* **Total strategy drawdown ≤ baseline** (no greater max drawdown versus “singles‑only” policy).

**Guardrails**

* Do **not** weaken single‑bet rules; parlays are **conditional** (only when <2 singles qualify).
* Only **2‑leg** parlays initially; expand to 3 legs **after** 6 months of positive CLV.
* No legs from the **same fight** or with estimated correlation **ρ > 0.20**.

---

## 1) Decision Logic (High Level)

1. Run standard single‑bet pipeline (calibrated EV 5–15% + ≥5% market gap + confirmation checks).
2. If `qualified_singles < 2`, enable **Parlay Builder**.
3. Collect **eligible legs** (small‑edge pool) using relaxed thresholds (see §2).
4. Enumerate 2‑leg pairs, estimate correlation (see §3), compute combined EV (see §4).
5. Keep candidates that pass **combined EV**, **correlation**, and **market confirmation**.
6. Size stakes with **pessimistic Kelly** and strict caps (see §5) and apply portfolio limits (see §6).

---

## 2) Leg Eligibility (Relaxed for Parlay Pool)

Apply **after calibration** and on **vig‑free** market probabilities.

* **Calibrated EV per leg:** `EV_leg ≥ 0.02` (2%).
* **Market gap:** `p_model − p_market ≥ 0.03` (3%)
* **Confidence:** Bootstrap lower‑bound `p_low ≥ 0.48` if favorite, `p_low ≥ 0.28` if underdog (tunable by division).
* **Line movement:** No adverse move > 3% against model in last N hours (configurable, default N=24h).
* **Exclusions:** No props that are inherently correlated across legs (e.g., same fighter ML + over/under where style links outcomes).

> Note: These thresholds are **only** for the parlay leg pool; single‑bet rules remain stricter.

---

## 3) Correlation Estimation (ρ)

Estimate ρ between legs A and B using a blended score:

1. **Feature Overlap Cosine (0–1):** cosine similarity of model feature vectors used for the two matchups; map to \[0, 0.5].
2. **Residual Co‑Movement (−1…1):** Pearson correlation of **OOF residuals** (predicted − market fair prob) across historical bouts of similar type/division; map to \[−0.5, 0.5].
3. **Heuristic Penalties:** +0.1 if same event weight class; +0.1 if both legs rely heavily on the same modeled dimension (e.g., grappling ELO).

```
rho = clamp( 0.6 * cos_mapped + 0.4 * resid_mapped + penalties, 0.0, 1.0 )
```

**Rule:** If `ρ > 0.20` → **reject** pair. If `0.10 < ρ ≤ 0.20`, apply a probability penalty (see §4).

---

## 4) Combined Probability & EV

For legs with calibrated probabilities `p1, p2` and decimal odds `o1, o2`:

1. **Base parlay prob:** `p_base = p1 * p2`.
2. **Correlation penalty:**

   * If `ρ ≤ 0.10`: `p_adj = p_base`.
   * If `0.10 < ρ ≤ 0.20`: `p_adj = p_base * (1 − α * (ρ − 0.10))`, with `α = 1.5`.
3. **Pessimistic bound:** compute `p1_low, p2_low` from bootstrap (e.g., 20th percentile) → `p_low_adj` via same penalty.
4. **Parlay odds:** `o_parlay = o1 * o2`.
5. **Combined EV:** `EV_parlay = p_adj * o_parlay − 1`.

**Acceptance Rule:** `EV_parlay ≥ 0.10` (10%) **and** `p_low_adj * o_parlay − 1 ≥ 0.02` (safety margin).

---

## 5) Staking for Parlays (Capital at Risk)

* **Pessimistic Kelly:**

  ```python
  k = max(0.0, (p_low_adj * o_parlay - (1 - p_low_adj)) / o_parlay)
  stake = bankroll * 0.25 * k   # quarter Kelly
  ```
* **Hard caps:** `min(stake, 0.5% * bankroll)` initially; can raise to `1.0%` after 100 parlays with CLV ≥ 55%.
* **Floor:** do not place if stake < min bet size or < 0.1% bankroll.

---

## 6) Portfolio & Exposure Limits (Per Card)

* Max **2 parlays per card**, total parlay exposure ≤ **1.5% bankroll** (or ≤ exposure used for one standard single bet in a high‑signal card).
* Never exceed total event exposure cap (singles + parlays) of **5% bankroll**.
* No more than **one leg per weight class** unless correlation proof shows `ρ ≤ 0.05`.

---

## 7) Backtesting Protocol (Must Pass Before Live)

1. **Data:** ≥ 3 years historical cards; freeze model per walk‑forward fold.
2. **Friction:** include slippage (25–50 bp), availability filters, limits, void/push rules.
3. **Comparators:**

   * Baseline A: Singles‑only strategy.
   * Baseline B: No‑bet on low‑signal cards.
   * Strategy C: Conditional parlays as defined here.
4. **Metrics:** Overall ROI, Sharpe (mean/Std of weekly returns), Max DD, Parlay CLV, ROI/CLV by EV bucket and by ρ bucket.
5. **Sign‑off:** Use bootstrap on fold returns to estimate confidence intervals; require Strategy C ≥ Baseline B on ROI with no worse drawdown.

---

## 8) Monitoring After Launch

* **Weekly:** Parlay hit rate, EV distribution, CLV, average ρ, equity curve.
* **Monthly:** Compare C vs A and B; disable parlays if 30‑day CLV < 50% or drawdown > 10% bankroll.
* **Drift checks:** If ECE worsens, tighten eligibility (raise EV\_leg threshold or lower caps).

---

## 9) Implementation Sketch (Modules & Signatures)

```
src/ufc_predictor/betting/
├── parlay_builder.py
├── correlation.py
└── selection.py
```

**`selection.py`**

```python
def eligible_parlay_legs(df, cfg) -> pd.DataFrame:
    """Return candidate legs meeting relaxed EV, gap, confidence, and line-move rules."""
```

**`correlation.py`**

```python
def leg_feature_vector(leg) -> np.ndarray: ...

def cosine_overlap(v1, v2) -> float: ...  # 0..1

def residual_comovement(leg1, leg2, hist_df) -> float: ...  # -1..1 (OOF residuals)

def estimate_rho(leg1, leg2, hist_df) -> float: ...  # blend & clamp to 0..1
```

**`parlay_builder.py`**

```python
def enumerate_pairs(legs_df) -> Iterable[Tuple[Leg, Leg]]: ...

def parlay_metrics(leg1, leg2, rho, cfg) -> Dict[str, float]:
    """Return p_adj, p_low_adj, o_parlay, EV_parlay, passes flags."""


def select_parlays(legs_df, cfg, max_per_card=2) -> List[Parlay]:
    """Filter by rho, EV thresholds, and portfolio caps; return ranked parlays."""
```

---

## 10) CLI & Config

**Config (`configs/parlay.yaml`)**

```yaml
eligibility:
  ev_leg_min: 0.02
  market_gap_min: 0.03
  p_low_min_fav: 0.48
  p_low_min_dog: 0.28
  adverse_move_max: 0.03
correlation:
  rho_reject: 0.20
  rho_penalty_start: 0.10
  alpha: 1.5
parlay:
  min_ev_combined: 0.10
  min_ev_combined_low: 0.02
staking:
  kelly_fraction: 0.25
  max_pct_bankroll: 0.005
portfolio:
  max_parlays_per_card: 2
  max_parlay_exposure_pct: 0.015
  max_event_exposure_pct: 0.05
```

**CLI**

```
python scripts/main.py --mode parlay --config configs/parlay.yaml --card "UFC XYZ"
```

---

## 11) Timeline

* **Week 1:** Implement modules, add tests, wire into pipeline; dry‑run on last 12 months.
* **Week 2:** Full 3‑year walk‑forward backtest vs baselines; tune thresholds.
* **Week 3:** Add dashboards, ship challenger mode; live shadow for two cards.
* **Week 4:** Evaluate CLV/ROI/drawdown; decide on full enablement.

---

## 12) Acceptance Criteria

* Conditional parlay strategy beats “no‑bet” baseline on low‑signal cards without increasing max drawdown; parlay CLV ≥ 55% over 100 parlays; configuration is versioned and reproducible.

---

**End of Conditional Multibets Plan**
