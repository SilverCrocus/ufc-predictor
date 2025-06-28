# 🎲 Multi-Bet Expected Value Analysis Guide

## Overview

The UFC Predictor now includes **sophisticated multi-bet (parlay) analysis** that identifies profitable combinations of single bets with proper Expected Value calculations.

## Key Features

### ✅ **Mathematical Accuracy**
- Proper Expected Value: `EV = (Combined_Probability × Combined_Odds) - 1`
- Kelly Criterion sizing for optimal stake calculation
- Correlation penalties for same-event fights

### ✅ **Risk Management** 
- Conservative stake limits (2% max for multi-bets vs 5% for singles)
- Risk level assessment (LOW/MEDIUM/HIGH/VERY HIGH)
- Maximum 4-leg limit to prevent excessive risk

### ✅ **Smart Analysis**
- Only analyzes combinations from profitable single bets
- Filters by minimum EV threshold (15% default)
- Accounts for fight correlations on same card

## How It Works

### Step 1: Find Profitable Singles
```python
# System automatically identifies profitable single bets
# Example: Joshua Van (9.8% EV), Beneil Dariush (21.8% EV)
```

### Step 2: Generate Combinations
```python
# Creates all possible 2-4 leg combinations
# Example: Van + Dariush = 2-leg multi-bet
```

### Step 3: Calculate Multi-Bet EV
```python
Combined_Probability = 0.61 × 0.58 = 35.4%
Combined_Odds = 1.80 × 2.10 = 3.78
Raw_EV = (0.354 × 3.78) - 1 = 33.7%
Adjusted_EV = 33.7% × (1 - 8% correlation penalty) = 31.0%
```

### Step 4: Risk Assessment & Sizing
```python
Risk Level: MEDIUM RISK (2-leg, 35.4% win probability)
Optimal Stake: $20 (2% of $1000 bankroll)
Expected Profit: $6.21
```

## Usage in Notebook

The multi-bet analysis is automatically integrated into the stealth profitability analyzer:

```python
# Run analysis with multi-bet detection
profitability_results = run_stealth_profitability_analysis(
    card_results=card_results, 
    bankroll=1000
)

# Results include both singles and multi-bets
single_bets = profitability_results['profitable_bets']
multi_bets = profitability_results['multi_bet_opportunities']
```

## Example Output

```
🎯 MULTI-BET ANALYSIS
==============================
📊 Analyzing combinations from 8 profitable singles
🎲 Max legs: 4, Min EV threshold: 15%

💎 FOUND 5 PROFITABLE MULTI-BETS
--------------------------------------------------
1. 2-LEG MULTI (MEDIUM RISK)
   Fighters: Joshua Van + Beneil Dariush
   Combined Odds: 3.78
   Win Probability: 35.4%
   Expected Value: 31.0%
   💰 Optimal Stake: $20.00
   💵 Expected Profit: $6.21
```

## Strategic Considerations

### ✅ **When Multi-Bets Are Good**
- Multiple high-EV single opportunities available
- Seeking higher returns with acceptable risk
- Capital efficiency (less total stake required)

### ⚠️ **When to Avoid Multi-Bets**
- Limited profitable singles available
- Risk tolerance is low
- Prefer steady, diversified returns

### 🎯 **Optimal Strategy**
- Use 2-3 leg multi-bets for balance of risk/reward
- Avoid 4+ leg bets unless EV is exceptional
- Consider both single and multi-bet portfolios

## Risk Levels Explained

| Risk Level | Criteria | Strategy |
|------------|----------|----------|
| **LOW RISK** | 2 legs, >40% win prob | Safe multi-bet option |
| **MEDIUM RISK** | 2-3 legs, 20-40% win prob | Balanced risk/reward |
| **HIGH RISK** | 3-4 legs, 10-20% win prob | High EV, high volatility |
| **VERY HIGH RISK** | 4+ legs, <10% win prob | Lottery ticket plays |

## Configuration Options

```python
multi_opportunities = analyzer.analyze_multi_bet_opportunities(
    profitable_bets,
    max_legs=4,           # Maximum legs per multi-bet
    min_multi_ev=0.15     # Minimum 15% EV threshold
)
```

## Mathematical Foundation

The system uses **proper Expected Value calculations** that account for:

1. **Compound Probability**: All legs must win
2. **Multiplicative Odds**: Payouts multiply across legs  
3. **Correlation Effects**: Same-event fight dependencies
4. **Kelly Sizing**: Optimal stake calculation for multi-bets

This ensures mathematically sound analysis that maximizes long-term profitability while managing risk appropriately.

---

**🎯 The multi-bet system transforms your UFC predictions into a comprehensive betting strategy that identifies the highest-value opportunities across both single and combination bets!** 