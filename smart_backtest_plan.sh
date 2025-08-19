#!/bin/bash

echo "========================================================================="
echo "SMART BACKTESTING PLAN FOR YOUR TEMPORAL MODEL"
echo "========================================================================="
echo ""
echo "Your test set: Dec 2021 - Aug 2025 (4,336 unseen fights)"
echo ""
echo "RECOMMENDED APPROACH:"
echo "---------------------"
echo ""
echo "STEP 1: Start with recent 6 months (most relevant)"
echo "This uses ~2,000 API credits and gives you the most relevant data"
echo ""

cat << 'EOF'
python3 fetch_all_historical_odds.py \
  --start-date 2025-02-01 \
  --end-date 2025-08-09
EOF

echo ""
echo "STEP 2: Run backtest on this period"
echo ""

cat << 'EOF'
python3 backtest_temporal_model.py
EOF

echo ""
echo "STEP 3: If results are promising, fetch 2024 data"
echo "This adds another year of validation"
echo ""

cat << 'EOF'
python3 fetch_all_historical_odds.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
EOF

echo ""
echo "STEP 4: If you need maximum validation, fetch 2023"
echo ""

cat << 'EOF'
python3 fetch_all_historical_odds.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31
EOF

echo ""
echo "========================================================================="
echo "WHY THIS APPROACH?"
echo "========================================================================="
echo ""
echo "1. RECENT DATA FIRST: Most relevant to current betting markets"
echo "2. PRESERVE CREDITS: Start small, expand if needed"
echo "3. ITERATIVE VALIDATION: See results before committing more credits"
echo "4. MARKET EVOLUTION: 2025 odds are most predictive of future"
echo ""
echo "========================================================================="
echo "EXPECTED RESULTS"
echo "========================================================================="
echo ""
echo "With 6 months of data (~500 fights):"
echo "  • Statistical significance: ✓ (sufficient for confidence intervals)"
echo "  • Market relevance: ✓✓✓ (most recent = most relevant)"
echo "  • API cost: ~2,000 credits (10% of your budget)"
echo ""
echo "If you see >5% ROI with 95% confidence → Worth expanding to more data"
echo "If you see <5% ROI or high variance → Focus on model improvement first"
echo ""