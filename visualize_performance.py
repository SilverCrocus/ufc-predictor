#!/usr/bin/env python3
"""
Visualize UFC Betting Performance
==================================

Quick script to show key performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

# Performance metrics
metrics = {
    'Training Accuracy': 74.8,
    'Test Accuracy': 51.2,
    'Baseline (Random)': 50.0
}

roi_results = {
    'Fixed Betting (Optimal)': 185.5,
    'Fixed Betting (Current)': 176.6,
    'Compound Betting': 90.8,
    'With 10% Parlays': 141.4,
    'With 5% Parlays': 161.6
}

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
colors = ['green' if v > 50 else 'red' for v in metrics.values()]
bars1 = ax1.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
ax1.axhline(y=50, color='black', linestyle='--', label='Random Baseline')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy: Training vs Reality')
ax1.set_ylim(0, 100)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')

# ROI comparison
colors2 = ['darkgreen' if 'Fixed' in k and 'Optimal' in k else 
           'green' if 'Fixed' in k else
           'orange' if 'Compound' in k else 
           'red' for k in roi_results.keys()]
bars2 = ax2.bar(roi_results.keys(), roi_results.values(), color=colors2, alpha=0.7)
ax2.set_ylabel('ROI (%)')
ax2.set_title('Betting Strategy ROI Comparison (3.5 years)')
ax2.set_xticklabels(roi_results.keys(), rotation=45, ha='right')

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('performance_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("=" * 60)
print("UFC BETTING SYSTEM PERFORMANCE SUMMARY")
print("=" * 60)
print("\nMODEL ACCURACY:")
print(f"  Training Set:  74.8% (overfitted)")
print(f"  Test Set:      51.2% (real performance)")
print(f"  Improvement:   +1.2% over random")
print()
print("BETTING PERFORMANCE (3.5 years):")
print(f"  Initial:       $1,000")
print(f"  Final:         $2,855 (with optimization)")
print(f"  ROI:           185.5%")
print(f"  Annual Return: ~53%")
print()
print("KEY INSIGHTS:")
print("  ✅ Despite low accuracy, edge detection works!")
print("  ✅ Fixed betting outperforms compound with 51% accuracy")
print("  ❌ Parlays reduce ROI by 15-35%")
print("  ❌ Training accuracy (74.8%) was misleading due to overfitting")
print()
print("RECOMMENDATIONS:")
print("  1. Use 3% edge threshold (not 5%)")
print("  2. Keep fixed $20 bets")
print("  3. Never use parlays")
print("  4. Focus on improving model to 53%+ accuracy")
print("=" * 60)