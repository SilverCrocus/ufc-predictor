"""
Test UFC 317 Profitability Analysis

This script demonstrates the profitability analysis using real odds data
from the UFC 317 card shown in the screenshot.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.profitability import ProfitabilityOptimizer, BettingOdds

def test_ufc317_profitability_simple():
    """
    Simple profitability analysis with UFC 317 card data from fightodds.io screenshot
    """
    print("🎯 UFC 317 PROFITABILITY ANALYSIS (SIMPLIFIED)")
    print("Data from fightodds.io screenshot")
    print("=" * 60)
    
    # Initialize profitability optimizer
    optimizer = ProfitabilityOptimizer(bankroll=5000.0, max_kelly_fraction=0.05)
    
    # UFC 317 fights with odds from screenshot
    ufc317_analysis = [
        {
            'fighter_a': 'Charles Oliveira',
            'fighter_b': 'Ilia Topuria', 
            'odds_a': +400,
            'odds_b': -500,
            'model_prob_a': 0.35,  # Example: model gives Oliveira 35% chance
            'model_prob_b': 0.65   # Model gives Topuria 65% chance
        },
        {
            'fighter_a': 'Alexandre Pantoja',
            'fighter_b': 'Kai Kara-France',
            'odds_a': -245,
            'odds_b': +210,
            'model_prob_a': 0.72,  # Model gives Pantoja 72% chance
            'model_prob_b': 0.28
        },
        {
            'fighter_a': 'Brandon Royval',
            'fighter_b': 'Josh Van',
            'odds_a': +107,
            'odds_b': -125,
            'model_prob_a': 0.48,  # Close fight
            'model_prob_b': 0.52
        },
        {
            'fighter_a': 'Beneil Dariush',
            'fighter_b': 'Renato Moicano',
            'odds_a': +130,
            'odds_b': -155,
            'model_prob_a': 0.45,
            'model_prob_b': 0.55
        },
        {
            'fighter_a': 'Niko Price',
            'fighter_b': 'Jacobe Smith',
            'odds_a': +1200,  # Huge underdog
            'odds_b': -2500,  # Massive favorite
            'model_prob_a': 0.15,  # Model gives Price 15% chance
            'model_prob_b': 0.85
        }
    ]
    
    profitable_opportunities = []
    
    print(f"💰 ANALYZING {len(ufc317_analysis)} KEY FIGHTS:")
    print("=" * 60)
    
    for i, fight in enumerate(ufc317_analysis, 1):
        print(f"\n{i}. {fight['fighter_a']} vs {fight['fighter_b']}")
        print(f"   Market odds: {fight['fighter_a']} ({fight['odds_a']:+d}), {fight['fighter_b']} ({fight['odds_b']:+d})")
        
        # Calculate market probabilities
        market_prob_a = optimizer.american_odds_to_probability(fight['odds_a'])
        market_prob_b = optimizer.american_odds_to_probability(fight['odds_b'])
        
        print(f"   Market probabilities: {fight['fighter_a']} ({market_prob_a*100:.1f}%), {fight['fighter_b']} ({market_prob_b*100:.1f}%)")
        print(f"   Model probabilities:  {fight['fighter_a']} ({fight['model_prob_a']*100:.1f}%), {fight['fighter_b']} ({fight['model_prob_b']*100:.1f}%)")
        
        # Check expected value for each fighter
        ev_a = optimizer.calculate_expected_value(fight['model_prob_a'], fight['odds_a'])
        ev_b = optimizer.calculate_expected_value(fight['model_prob_b'], fight['odds_b'])
        
        print(f"   Expected Values: {fight['fighter_a']} ({ev_a*100:+.2f}%), {fight['fighter_b']} ({ev_b*100:+.2f}%)")
        
        # Check for profitable opportunities
        if ev_a > 0:
            kelly_a = optimizer.calculate_kelly_fraction(fight['model_prob_a'], fight['odds_a'])
            bet_amount_a = kelly_a * optimizer.bankroll
            profitable_opportunities.append({
                'fight': f"{fight['fighter_a']} vs {fight['fighter_b']}",
                'fighter': fight['fighter_a'],
                'odds': fight['odds_a'],
                'expected_value': ev_a,
                'recommended_bet': bet_amount_a,
                'kelly_fraction': kelly_a,
                'edge': fight['model_prob_a'] - market_prob_a
            })
            print(f"   🎯 OPPORTUNITY: Bet ${bet_amount_a:.2f} on {fight['fighter_a']} (EV: {ev_a*100:+.2f}%)")
        
        if ev_b > 0:
            kelly_b = optimizer.calculate_kelly_fraction(fight['model_prob_b'], fight['odds_b'])
            bet_amount_b = kelly_b * optimizer.bankroll
            profitable_opportunities.append({
                'fight': f"{fight['fighter_a']} vs {fight['fighter_b']}",
                'fighter': fight['fighter_b'],
                'odds': fight['odds_b'],
                'expected_value': ev_b,
                'recommended_bet': bet_amount_b,
                'kelly_fraction': kelly_b,
                'edge': fight['model_prob_b'] - market_prob_b
            })
            print(f"   🎯 OPPORTUNITY: Bet ${bet_amount_b:.2f} on {fight['fighter_b']} (EV: {ev_b*100:+.2f}%)")
        
        if ev_a <= 0 and ev_b <= 0:
            print(f"   ❌ No profitable opportunities")
    
    # Summary
    print(f"\n📊 PROFITABILITY SUMMARY:")
    print("=" * 40)
    
    if profitable_opportunities:
        print(f"💎 Found {len(profitable_opportunities)} profitable opportunities!")
        
        # Sort by expected value
        profitable_opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        print(f"\n🏆 TOP OPPORTUNITIES:")
        total_recommended_bets = 0
        expected_profit = 0
        
        for i, opp in enumerate(profitable_opportunities[:5], 1):  # Top 5
            print(f"{i}. {opp['fighter']} in {opp['fight']}")
            print(f"   EV: {opp['expected_value']*100:+.2f}%, Bet: ${opp['recommended_bet']:.2f}, Edge: {opp['edge']*100:+.1f}%")
            
            total_recommended_bets += opp['recommended_bet']
            expected_profit += opp['recommended_bet'] * opp['expected_value']
        
        print(f"\n💰 BETTING SUMMARY:")
        print(f"   Total recommended bets: ${total_recommended_bets:.2f}")
        print(f"   Bankroll exposure: {(total_recommended_bets/optimizer.bankroll)*100:.1f}%")
        print(f"   Expected profit: ${expected_profit:.2f}")
        print(f"   Expected ROI: {(expected_profit/total_recommended_bets)*100:.1f}%")
        
    else:
        print("❌ No profitable opportunities found on this card.")
        print("The market odds appear well-calibrated to model predictions.")
    
    return profitable_opportunities

if __name__ == "__main__":
    print("🥊 UFC 317 Profitability Analysis using fightodds.io data")
    print("Note: This uses example model probabilities for demonstration")
    print()
    
    opportunities = test_ufc317_profitability_simple()
    
    print(f"\n💡 KEY INSIGHTS:")
    print("• This analysis shows how to identify profitable betting opportunities")
    print("• Expected Value (EV) > 0 indicates a profitable bet")
    print("• Kelly Criterion determines optimal bet sizing")
    print("• Higher model edge = higher confidence in the opportunity")
    print()
    print("🎯 To use with real data:")
    print("1. Train your prediction models on historical fight data")
    print("2. Scrape current odds from fightodds.io")
    print("3. Compare model probabilities vs market probabilities") 
    print("4. Bet only when Expected Value is positive!") 