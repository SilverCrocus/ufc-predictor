# UFC Betting Results Update Guide

## After UFC Fight Night: Whittaker vs de Ridder (July 27, 2025)

### Method 1: Individual Bet Updates (Recommended)

Use the bet tracking system to update each bet individually:

```python
from src.bet_tracking import BetTracker

# Initialize tracker
tracker = BetTracker()

# Example: Update a winning bet
tracker.update_fight_result(
    bet_id="BET_20250727_014013_82247693",  # Marcus McGhee bet
    actual_result="WIN",  # or "LOSS"
    profit_loss=6.71,    # Amount won (bet_size * (odds - 1)) or -bet_size for loss
    fight_date="2025-07-27",
    method_actual="Decision"  # Decision/KO/TKO/Submission
)

# Example: Update a losing bet  
tracker.update_fight_result(
    bet_id="BET_20250727_014013_c261e3d6",  # Reinier de Ridder bet
    actual_result="LOSS",
    profit_loss=-0.86,   # Negative bet size for losses
    fight_date="2025-07-27", 
    method_actual="Decision"
)
```

### Method 2: Bulk Event Update

Update all bets for the entire event at once:

```python
from src.bet_tracking import BetTracker

tracker = BetTracker()

# Define results for all fighters you bet on
results = {
    "Marcus McGhee": {"result": "WIN", "method": "Decision", "date": "2025-07-27"},
    "Reinier de Ridder": {"result": "LOSS", "method": "Decision", "date": "2025-07-27"},
    "Asu Almabayev": {"result": "WIN", "method": "Submission", "date": "2025-07-27"},
    "Nikita Krylov": {"result": "WIN", "method": "KO/TKO", "date": "2025-07-27"},
    # Add more fighters as needed
}

# Update all bets for the event
updated_count = tracker.update_event_results("UFC Fight Night: Whittaker vs de Ridder", results)
print(f"Updated {updated_count} bets")
```

### Your Exact Bet IDs to Update

**Single Bets:**
- Marcus McGhee: `BET_20250727_014013_82247693`
- Reinier de Ridder: `BET_20250727_014013_c261e3d6` 
- Asu Almabayev: `BET_20250727_014013_6d059470`
- Nikita Krylov: `BET_20250727_014013_974380a4`
- Marcus McGhee (2nd bet): `BET_20250727_023549_d5397a38`
- Reinier de Ridder (2nd bet): `BET_20250727_023549_0cf9772a`
- Marc-Andre Barriault: `BET_20250727_023549_3aa2d7ee`
- Jose Ochoa: `BET_20250727_023549_40c40bad`
- Bogdan Guskov: `BET_20250727_023549_482ce22b`

**Parlay Bets (update if ALL fighters win):**
- 3-leg parlay: `BET_20250727_014013_dfa2712f`
- 2-leg parlay: `BET_20250727_014013_1f15789d`
- (Plus 8 more parlay combinations)

### Profit/Loss Calculation

**For Winning Bets:**
```
Profit = bet_size × (decimal_odds - 1)
Example: $2.14 bet at 4.0 odds = $2.14 × (4.0 - 1) = $6.42 profit
```

**For Losing Bets:**
```
Loss = -bet_size
Example: $2.14 bet loses = -$2.14 loss
```

**For Parlays:**
- Only wins if ALL fighters in the parlay win
- If any fighter loses, entire parlay loses

### After Updating Results

Generate your performance report:

```python
# View your updated performance
tracker.generate_performance_report()

# Export to Excel for detailed analysis
tracker.export_to_excel("ufc_results_july27.xlsx")
```

### Quick Commands

**View pending bets:**
```python
tracker = BetTracker()
pending = tracker.get_pending_bets()
print(f"Pending bets: {len(pending)}")
```

**View bets for this event:**
```python
event_bets = tracker.get_bets_by_event("UFC Fight Night: Whittaker vs de Ridder")
print(f"Event bets: {len(event_bets)}")
```