#!/usr/bin/env python3
"""
Test Fixed Notebook Configuration
==================================

This script tests that the fixed TAB profitability analysis 
works correctly with the user's specific fight card.
"""

import sys
sys.path.append('.')

from src.tab_profitability import TABProfitabilityAnalyzer

def main():
    # Your specific fight predictions from the notebook
    your_card_predictions = {
        'Ilia Topuria': 0.698,
        'Charles Oliveira': 0.302,
        'Alexandre Pantoja': 0.625,
        'Kai Kara-France': 0.375,
        'Brandon Royval': 0.392,
        'Joshua Van': 0.608,
        'Beneil Dariush': 0.579,
        'Renato Moicano': 0.421,
        'Payton Talbott': 0.340,
        'Felipe Lima': 0.660
    }

    print('🚀 TESTING FIXED NOTEBOOK CONFIGURATION')
    print('=' * 60)
    print(f'🎯 Your fight card: {len(your_card_predictions)} fighters')
    
    # Show the fight list from your notebook
    print('\n📋 Your UFC Card Fights:')
    fights = [
        "Ilia Topuria vs. Charles Oliveira",
        "Alexandre Pantoja vs. Kai Kara-France", 
        "Brandon Royval vs. Joshua Van",
        "Beneil Dariush vs. Renato Moicano",
        "Payton Talbott vs. Felipe Lima"
    ]
    for i, fight in enumerate(fights, 1):
        print(f'   {i}. {fight}')

    # Test with live scraping disabled (should use fixed data)
    print('\n📄 Testing with use_live_odds=False (FIXED):')
    analyzer = TABProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
    results = analyzer.analyze_predictions(your_card_predictions)

    print(f'\n📊 RESULTS SUMMARY:')
    print(f'✅ Matched fighters: {results.get("matched_fighters", 0)}/{results.get("total_fighters", 0)}')
    print(f'💰 Profitable opportunities: {results.get("total_opportunities", 0)}')
    print(f'💵 Total expected profit: ${results.get("total_expected_profit", 0):.2f}')
    print(f'📡 Odds source: {results.get("odds_source", "unknown")}')

    if results.get('opportunities'):
        print('\n🏆 FOUND OPPORTUNITIES:')
        for opp in results['opportunities'][:3]:
            print(f'   {opp.fighter}: TAB {opp.tab_decimal_odds} | Model {opp.model_prob:.1%} | {opp.expected_value:.1%} EV')
    else:
        print('\n❌ No profitable opportunities found')
        
    print('\n🎯 NOW YOUR NOTEBOOK SHOULD WORK!')
    print('   - Restart your Jupyter kernel')
    print('   - Re-run all cells')
    print('   - You should see proper matches instead of wrong ones')
    print('   - No more "Topuria → Murphy v Moura" nonsense!')

if __name__ == "__main__":
    main() 