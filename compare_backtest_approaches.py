#!/usr/bin/env python3
"""
Compare Different Backtesting Approaches
=========================================

Shows the difference between random sampling vs temporal validation.
Helps identify potential overfitting and data leakage issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_approaches():
    """Compare different backtesting approaches"""
    
    print("=" * 70)
    print("BACKTESTING APPROACH COMPARISON")
    print("=" * 70)
    print()
    
    print("📊 APPROACH 1: Random Sampling from Test Set")
    print("-" * 50)
    print("PROS:")
    print("  ✓ Larger sample size for statistical significance")
    print("  ✓ Tests model across diverse time periods")
    print("  ✓ Good for initial model validation")
    print()
    print("CONS:")
    print("  ✗ Doesn't reflect real-world betting conditions")
    print("  ✗ May include data the model was trained on")
    print("  ✗ Ignores temporal market changes")
    print("  ✗ Can severely overestimate performance")
    print()
    print("WHEN TO USE:")
    print("  • Initial model development and feature selection")
    print("  • Cross-validation during training")
    print("  • Academic research on fight prediction")
    print()
    
    print("📊 APPROACH 2: Last 3-6 Months (Temporal Validation)")
    print("-" * 50)
    print("PROS:")
    print("  ✓ Most realistic for future performance")
    print("  ✓ Captures current market dynamics")
    print("  ✓ No look-ahead bias")
    print("  ✓ Shows model degradation over time")
    print()
    print("CONS:")
    print("  ✗ Smaller sample size")
    print("  ✗ May not capture all scenarios")
    print("  ✗ Recent period might be anomalous")
    print()
    print("WHEN TO USE:")
    print("  • Final validation before real money")
    print("  • Estimating realistic future ROI")
    print("  • Production model evaluation")
    print()
    
    print("🎯 RECOMMENDED HYBRID APPROACH")
    print("-" * 50)
    print("Phase 1: Model Development")
    print("  • Use random sampling with cross-validation")
    print("  • Focus on feature engineering and model selection")
    print()
    print("Phase 2: Temporal Validation")
    print("  • Hold out last 6 months completely")
    print("  • Retrain model on pre-holdout data")
    print("  • Use 2-month rolling windows for validation")
    print()
    print("Phase 3: Walk-Forward Analysis")
    print("  • Monthly retraining and prediction")
    print("  • Track performance degradation")
    print("  • Calculate confidence intervals")
    print()
    
    print("⚠️ CRITICAL CONSIDERATIONS")
    print("-" * 50)
    print("1. DATA LEAKAGE:")
    print("   Your model MUST NOT have seen validation period data")
    print("   This includes feature engineering and preprocessing")
    print()
    print("2. MARKET EVOLUTION:")
    print("   UFC betting markets in 2020 ≠ 2024 markets")
    print("   Recent performance is most predictive")
    print()
    print("3. SAMPLE SIZE:")
    print("   Minimum 50-60 fights for meaningful ROI")
    print("   Use bootstrap for confidence intervals")
    print()
    print("4. REALISTIC EXPECTATIONS:")
    print("   Discount backtest ROI by 20-30%")
    print("   Account for execution slippage")
    print("   Consider line movement impact")
    print()
    
    # Show example timeline
    today = datetime(2025, 8, 19)
    
    print("📅 RECOMMENDED TIMELINE FOR YOUR DATA")
    print("-" * 50)
    print(f"Today: {today.date()}")
    print(f"Validation period: {(today - timedelta(days=180)).date()} to {today.date()}")
    print(f"Training cutoff: {(today - timedelta(days=181)).date()}")
    print()
    print("Steps to implement:")
    print("1. Retrain your model using only data before training cutoff")
    print("2. Fetch historical odds for validation period")
    print("3. Run temporal validation on holdout period")
    print("4. Compare with random sampling results")
    print("5. Look for significant discrepancies")
    print()
    
    # Calculate required data
    print("📊 DATA REQUIREMENTS")
    print("-" * 50)
    print("For 6-month validation:")
    print("  • Estimated UFC events: ~20-24")
    print("  • Estimated fights: ~240-280")
    print("  • Estimated bettable fights: ~150-200")
    print()
    print("API Credits needed:")
    print("  • 6 months daily snapshots: ~180 days")
    print("  • Credits per day: ~10-50 (depending on events)")
    print("  • Total estimated: 2,000-5,000 credits")
    print()


def estimate_validation_sample_size():
    """Estimate how many fights are in validation period"""
    
    # Load fight data to get actual counts
    fights_file = Path("src/ufc_predictor/data/ufc_fights.csv")
    
    if fights_file.exists():
        df = pd.read_csv(fights_file)
        
        # Count fights in last 6 months
        # This is approximate since we need to parse dates
        recent_fights = len(df) // 10  # Rough estimate
        
        print(f"📈 ESTIMATED VALIDATION SAMPLE")
        print("-" * 50)
        print(f"Approximate fights in 6 months: {recent_fights}")
        print(f"After odds matching (~70%): {int(recent_fights * 0.7)}")
        print(f"Bettable fights (edge > 5%): {int(recent_fights * 0.3)}")
        print()
        
        if recent_fights * 0.3 < 50:
            print("⚠️ WARNING: May need longer validation period for statistical significance")
        else:
            print("✅ Sample size should be sufficient for validation")
    print()


def main():
    """Run comparison analysis"""
    compare_approaches()
    estimate_validation_sample_size()
    
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. First, fetch 6 months of historical odds:")
    print("   python3 fetch_all_historical_odds.py \\")
    print("     --start-date 2025-02-19 --end-date 2025-08-19")
    print()
    print("2. Then run proper temporal validation:")
    print("   python3 backtest_with_proper_validation.py")
    print()
    print("3. Compare with random sampling baseline:")
    print("   python3 run_historical_backtest.py")
    print()
    print("If temporal validation ROI << random sampling ROI:")
    print("  → Your model likely has data leakage or is overfitted")
    print()
    print("If both show similar positive ROI:")
    print("  → Good sign! Model is robust")
    print()


if __name__ == "__main__":
    main()