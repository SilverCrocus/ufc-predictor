#!/bin/bash

# 🥊 Quick UFC Profitability Analysis Scripts

echo "🥊 UFC PROFITABILITY ANALYSIS - QUICK LAUNCHER"
echo "=============================================="
echo "1. 🚀 Full analysis with live odds (slow but accurate)"
echo "2. ⚡ Quick analysis with cached odds (fast)"
echo "3. 💰 Custom bankroll analysis"
echo "4. 📝 Interactive prediction entry"
echo "5. 🧪 Test with sample predictions"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo "🚀 Running full analysis with live TAB Australia odds..."
        python3 run_profitability_analysis.py --sample
        ;;
    2)
        echo "⚡ Running quick analysis with cached odds..."
        python3 run_profitability_analysis.py --sample --no-live-odds
        ;;
    3)
        read -p "💰 Enter your bankroll in AUD: $" bankroll
        echo "Running analysis with bankroll: \$${bankroll}..."
        python3 run_profitability_analysis.py --sample --bankroll $bankroll
        ;;
    4)
        echo "📝 Interactive prediction entry mode..."
        python3 run_profitability_analysis.py
        ;;
    5)
        echo "🧪 Testing with sample predictions (no live odds)..."
        python3 run_profitability_analysis.py --sample --no-live-odds
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-5."
        exit 1
        ;;
esac 