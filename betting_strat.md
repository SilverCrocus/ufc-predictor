# UFC Predictor — Revised Betting Strategy Plan (Including Multi-bets)

**Context:** Backtest results show:

* Win rate: 20% overall (-89.8% ROI)
* Calibration Error (ECE): 0.380 (Target: <0.03)
* CLV: 50% (Target: >55%)
* EV bucket analysis:

  * 5–10% EV bets: 50% win rate, +43% ROI ✅
  * 10–15% EV bets: 0% win rate, -100% ROI ❌
  * 25%+ EV bets: 0% win rate, -100% ROI ❌

**Goal:** Eliminate overconfident, poorly calibrated high-EV bets; focus on EV ranges that historically perform; integrate calibration, market confirmation, disciplined staking, and structured multi-bet logic.

---

## 0) Objectives & Success Criteria

**Primary Goals:**

* Reduce ECE from 0.380 → <0.05.
* Achieve CLV win rate > 55%.
* Maintain positive ROI in backtest with realistic market constraints.
* Eliminate large drawdowns from high-EV false positives.
* Integrate multi-bet strategy to increase portfolio efficiency without increasing risk disproportionately.

**Guardrails:**

* All EV calculations must use *calibrated* probabilities.
* No bets placed without market confirmation.
* Multi-bets only allowed when legs are uncorrelated and pass individual filters.

---

## 1) Probability Calibration Before EV

1. Use **temporal out-of-fold predictions** to fit per-segment calibrators:

   * Segment by gender × weight class × fight length (3R/5R).
2. Apply isotonic regression or Platt scaling for each segment.
3. Store and version calibrators.
4. Apply calibrated probabilities before calculating EV:

   ```python
   ev = (calibrated_p * odds) - 1
   ```

**Acceptance Criteria:** ECE < 0.05 for all major segments.

---

## 2) EV Bucket Filtering

* **Primary range:** Auto-bet if EV is between **5% and 15%** after calibration.
* **High EV (>15%) bets**: require two confirmations:

  1. CLV projection > 55%.
  2. Market odds stable or moving in model’s direction.
* **Low EV (<5%) bets**: skip unless included in a low-correlation multi-bet.

---

## 3) Market Confirmation Filter

1. Convert odds to vig-free market probability.
2. Require min probability gap (default: 5%) between model and market.
3. Reject bets with >5% adverse market movement pre-fight.
4. Weight CLV history in ranking bets.

---

## 4) Staking Policy

1. **Base stake:** quarter-Kelly using pessimistic probability (`p_low` from 20th percentile bootstrap):

   ```python
   stake = bankroll * 0.25 * max(0, (p_low * odds - (1 - p_low)) / odds)
   ```
2. Apply caps:

   * Max 2% bankroll for single bets in 5–10% EV range.
   * Max 1% bankroll for 10–15% EV bets.
   * Max 0.5% bankroll for >15% EV bets (even if confirmed).
3. For parlays (multibets): **use only under strict conditions** (see §5a) and size at **half** the single-bet stake for the same EV.

---

## 5) Backtesting Framework Updates

### 5a) Multibets / Parlays Strategy (New)

**When to consider parlays**

* Only if **each leg** individually meets the single-bet rules (calibrated EV ≥ 5%, market confirmation) and **legs are weakly correlated**.
* Limit to **2 legs** initially (max **3** after proving edge over 6+ months). Avoid method-of-victory props in the same fight as a moneyline leg.
* Do **not** include two legs that depend on the same fighter outcome (e.g., Fighter A ML + Over 1.5 if their styles are strongly linked).

**Correlation control**

* Compute leg correlation ρ via:

  1. **Feature-overlap cosine** between legs’ feature vectors (high overlap ⇒ higher ρ), and
  2. **Historical residual correlation** of model errors for similar markets.
* If estimated ρ > 0.2, **disallow** the parlay. If 0.1 < ρ ≤ 0.2, apply a **penalty** to combined probability: `p_adj = p1*p2*(1 - α*ρ)` with α ∈ \[1.0, 1.5]. Default α=1.2.

**Combined EV & staking**

* Compute vig-free **fair probs** per leg, then combined parlay probability with correlation adjustment.
* Require **combined EV ≥ 10%** (post-calibration, post-correlation).
* Size stake with **pessimistic Kelly** using `p_low` for each leg and propagate to combined `p_low_combined`. Cap at **0.5–1.0%** of bankroll.

**Operational limits**

* **One parlay max per card** until live CLV > 55% on parlays over a 100-parlay sample.
* Respect book rules (voids, push rules) in simulation.

**Implementation sketch**

```python
# betting/correlation.py

def leg_feature_vector(bet) -> np.ndarray: ...

def feature_cosine_similarity(v1, v2) -> float: ...

def historical_residual_corr(bet1, bet2, history_df) -> float: ...

def estimate_rho(bet1, bet2, history_df) -> float:
    s = feature_cosine_similarity(leg_feature_vector(bet1), leg_feature_vector(bet2))
    r = historical_residual_corr(bet1, bet2, history_df)
    return 0.5*s + 0.5*r

# betting/odds_utils.py

def parlay_prob_with_corr(p1, p2, rho, alpha=1.2):
    base = p1 * p2
    penalty = (1 - alpha * max(0.0, rho))
    return max(0.0, min(1.0, base * penalty))
```

**Monitoring**

* Track parlay-specific metrics: hit rate, ROI, CLV, average ρ, and **incremental** ROI over equivalent singles.
* Auto-disable parlays if 30-day parlay CLV < 50% or Max Drawdown > 10% of bankroll.

---

### Requirements:

1. Test new filters and staking on at least 3+ years of historical fights.
2. Record performance by EV bucket, CLV bucket, and fight segment.
3. Metrics to monitor:

   * ROI & Sharpe ratio per bucket.
   * CLV win rate overall and per bucket.
   * Max drawdown.
   * **Parlay metrics** from §5a.
4. Compare new strategy vs old EV-max strategy and vs **no-parlay** baseline.

**Acceptance Criteria:**

* New strategy shows reduced drawdown and higher risk-adjusted return.
* High-EV bucket drawdowns reduced by ≥ 50%.
* Parlays (if enabled) show **positive CLV** and do not increase total drawdown beyond baseline.

---

1. Evaluate singles and multi-bets separately and combined.
2. Record ROI, Sharpe ratio, CLV win rate, drawdown for each type.
3. Compare new strategy (with multi-bets) vs old EV-max singles-only.

**Acceptance Criteria:**

* Positive ROI with lower drawdown than baseline.
* Multi-bets add incremental ROI without significantly increasing variance.

---

## 6) Monitoring & Deployment

1. Dashboards:

   * ROI by EV bucket (singles & multi-bets).
   * CLV trend.
   * ECE trend.
2. Shadow test multi-bet strategy alongside singles for 3 months before live deployment.

---

## 7) Timeline

* **Week 1:** Calibration integration.
* **Week 2:** EV bucket filtering + market confirmation.
* **Week 3:** Singles staking policy.
* **Week 4:** Multi-bet logic + staking.
* **Week 5–6:** Historical backtest and comparison.
* **Week 7:** Deploy in shadow mode.

---

## 8) Acceptance Criteria Recap

* ECE < 0.05.
* CLV win rate > 55%.
* Positive ROI in singles and combined portfolio.
* Multi-bets show positive contribution to portfolio without >20% increase in drawdown.

---

**End of Revised Betting Strategy Plan (Including Multi-bets)**
