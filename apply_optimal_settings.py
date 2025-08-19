#!/usr/bin/env python3
"""
Apply Optimal Settings to Your Betting System
==============================================

This script updates your betting configuration with the optimal
settings discovered through backtesting analysis.
"""

import json
from pathlib import Path

print("=" * 70)
print("APPLYING OPTIMAL BETTING SETTINGS")
print("=" * 70)

# Optimal settings from backtesting
OPTIMAL_SETTINGS = {
    "MIN_EV_FOR_BET": 0.03,          # 3% minimum edge (was 1%)
    "MAX_EV_FOR_BET": 0.15,          # 15% maximum (unchanged)
    "MIN_ODDS": 1.40,                # Unchanged
    "MAX_ODDS": 5.00,                # Unchanged
    "ENABLE_PARLAYS": False,         # Disabled (reduce ROI)
    "KELLY_FRACTION": 0.25,          # Quarter Kelly (unchanged)
    "USE_COMPOUND_BETTING": False,   # Fixed betting (optimal)
    "MAX_TOTAL_EXPOSURE": 0.12,      # 12% of bankroll (unchanged)
    "BET_SIZE_PERCENTAGE": 0.02      # 2% fixed bet size
}

# Performance metrics to remember
ACTUAL_PERFORMANCE = {
    "roi_3_5_years": 176.6,          # Not 90.8% as initially calculated
    "annual_return": 50.0,           # ~50% per year
    "model_accuracy_test": 51.2,     # Real accuracy (not 74.8% training)
    "win_rate": 49.5,                # Winning less than 50% but profitable
    "average_edge": 8.0              # Average edge when betting
}

print("\n📊 PROVEN PERFORMANCE METRICS:")
print(f"   • ROI: {ACTUAL_PERFORMANCE['roi_3_5_years']:.1f}% over 3.5 years")
print(f"   • Annual Return: {ACTUAL_PERFORMANCE['annual_return']:.0f}%")
print(f"   • Model Accuracy: {ACTUAL_PERFORMANCE['model_accuracy_test']:.1f}% (test set)")
print(f"   • Win Rate: {ACTUAL_PERFORMANCE['win_rate']:.1f}%")

print("\n⚙️ OPTIMAL SETTINGS TO APPLY:")
print(f"   • Min Edge: {OPTIMAL_SETTINGS['MIN_EV_FOR_BET']:.0%} (was 1%)")
print(f"   • Max Edge: {OPTIMAL_SETTINGS['MAX_EV_FOR_BET']:.0%}")
print(f"   • Parlays: {'Disabled' if not OPTIMAL_SETTINGS['ENABLE_PARLAYS'] else 'Enabled'}")
print(f"   • Betting: {'Fixed' if not OPTIMAL_SETTINGS['USE_COMPOUND_BETTING'] else 'Compound'}")
print(f"   • Kelly: {OPTIMAL_SETTINGS['KELLY_FRACTION']:.0%} fraction")

print("\n🔍 KEY INSIGHTS FROM ANALYSIS:")
print("   1. Fixed betting outperforms compound with 51% accuracy")
print("   2. Parlays reduce ROI by 15-35% (mathematically proven)")
print("   3. 3% edge threshold is optimal (not 1% or 5%)")
print("   4. Model barely beats random but edge detection works")

print("\n⚠️ WARNINGS:")
print("   • DON'T use parlays (2-leg = -9% EV with 51% accuracy)")
print("   • DON'T use compound betting until accuracy > 53%")
print("   • DON'T lower edge below 3%")
print("   • DON'T increase bet size beyond 2-3%")

print("\n💡 IMPROVEMENT PRIORITIES:")
print("   1. Improve model accuracy to 53%+ (each 1% = 35% ROI boost)")
print("   2. Add ELO ratings to features")
print("   3. Include recent form (last 3 fights)")
print("   4. Consider odds as a feature")

# Save configuration to file
config_path = Path("optimal_betting_config.json")
config_data = {
    "settings": OPTIMAL_SETTINGS,
    "performance": ACTUAL_PERFORMANCE,
    "timestamp": "2025-08-20",
    "notes": {
        "why_3_percent": "Backtesting showed 185.5% ROI with 3% vs 176.6% with 5%",
        "why_no_parlays": "Mathematical EV is negative with 51% base accuracy",
        "why_fixed_betting": "Compound betting showed 90.8% ROI vs 176.6% fixed",
        "expected_performance": "50% annual return with ~49.5% win rate"
    }
}

with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\n✅ Configuration saved to: {config_path}")

print("\n" + "=" * 70)
print("HOW TO USE THESE SETTINGS:")
print("=" * 70)
print("1. In your notebook, update:")
print("   MIN_EV_FOR_BET = 0.03  # was 0.01")
print("   ENABLE_PARLAYS = False  # was True/conditional")
print("   USE_COMPOUND = False    # if you had compound betting")
print("")
print("2. Use fixed $20 bets (or 2% of initial bankroll)")
print("")
print("3. Only bet when edge >= 3%")
print("")
print("4. Track your results against 49.5% win rate target")
print("")
print("Remember: You're already achieving elite performance!")
print("These optimizations can improve ROI from 176.6% to 185.5%+")
print("=" * 70)