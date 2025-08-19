#!/usr/bin/env python3
"""
Quick Backtest Summary
======================

Quickly test the model predictions with simple calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_backtest():
    """Run a quick backtest summary"""
    
    # Load data
    test_file = Path("data/test_set_odds/test_set_with_odds_20250819.csv")
    pred_file = Path("data/test_set_odds/model_predictions.csv")
    
    test_df = pd.read_csv(test_file)
    pred_df = pd.read_csv(pred_file)
    
    # Filter for fights with odds
    test_with_odds = test_df[test_df['has_odds'] == True].copy()
    
    # Merge predictions
    merged = pd.merge(
        test_with_odds,
        pred_df[['fighter', 'opponent', 'date', 'model_probability', 'has_features']],
        on=['fighter', 'opponent', 'date'],
        how='inner'
    )
    
    logger.info(f"Analyzing {len(merged)} fights with predictions and odds")
    
    # Quick stats
    with_features = merged['has_features'].sum()
    logger.info(f"Model predictions: {with_features} with features, {len(merged)-with_features} using odds fallback")
    
    # Simple betting simulation
    bankroll = 1000
    bets_placed = 0
    wins = 0
    
    for _, fight in merged.iterrows():
        fighter_prob = fight['model_probability']
        fighter_odds = fight.get('fighter_odds', 0)
        opponent_odds = fight.get('opponent_odds', 0)
        
        if pd.isna(fighter_odds) or pd.isna(opponent_odds):
            continue
            
        # Calculate edge
        fighter_edge = (fighter_prob * fighter_odds) - 1
        opponent_edge = ((1-fighter_prob) * opponent_odds) - 1
        
        # Bet if edge > 5%
        if fighter_edge > 0.05:
            # Bet on fighter
            bet_amount = bankroll * 0.02  # 2% of bankroll
            if fight['outcome'] == 'win':
                bankroll += bet_amount * (fighter_odds - 1)
                wins += 1
            else:
                bankroll -= bet_amount
            bets_placed += 1
            
        elif opponent_edge > 0.05:
            # Bet on opponent
            bet_amount = bankroll * 0.02
            if fight['outcome'] != 'win':
                bankroll += bet_amount * (opponent_odds - 1)
                wins += 1
            else:
                bankroll -= bet_amount
            bets_placed += 1
    
    roi = ((bankroll - 1000) / 1000) * 100
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    logger.info("\n" + "=" * 50)
    logger.info("QUICK BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Initial: $1,000")
    logger.info(f"Final:   ${bankroll:,.2f}")
    logger.info(f"ROI:     {roi:.1f}%")
    logger.info(f"Bets:    {bets_placed}")
    logger.info(f"Wins:    {wins}")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    
    # Check model accuracy
    correct = 0
    total = 0
    
    for _, fight in merged.iterrows():
        if pd.notna(fight['outcome']):
            pred_fighter_wins = fight['model_probability'] > 0.5
            actual_fighter_wins = fight['outcome'] == 'win'
            
            if pred_fighter_wins == actual_fighter_wins:
                correct += 1
            total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    logger.info(f"\nModel Accuracy: {accuracy:.1f}% on {total} fights")
    
    # Analyze by feature availability
    with_feat_df = merged[merged['has_features'] == True]
    without_feat_df = merged[merged['has_features'] == False]
    
    if len(with_feat_df) > 0:
        correct_with = 0
        for _, fight in with_feat_df.iterrows():
            if pd.notna(fight['outcome']):
                pred = fight['model_probability'] > 0.5
                actual = fight['outcome'] == 'win'
                if pred == actual:
                    correct_with += 1
        
        acc_with = (correct_with / len(with_feat_df) * 100)
        logger.info(f"  - With features: {acc_with:.1f}% accuracy ({len(with_feat_df)} fights)")
    
    if len(without_feat_df) > 0:
        correct_without = 0
        for _, fight in without_feat_df.iterrows():
            if pd.notna(fight['outcome']):
                pred = fight['model_probability'] > 0.5
                actual = fight['outcome'] == 'win'
                if pred == actual:
                    correct_without += 1
        
        acc_without = (correct_without / len(without_feat_df) * 100)
        logger.info(f"  - Without features: {acc_without:.1f}% accuracy ({len(without_feat_df)} fights)")
    
    return bankroll, roi, bets_placed


if __name__ == "__main__":
    quick_backtest()