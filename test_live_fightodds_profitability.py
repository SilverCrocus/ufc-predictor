"""
Live FightOdds.io Profitability Analysis

This script combines the Selenium scraper with the profitable predictor system
to analyze real-time betting opportunities from fightodds.io
"""

import json
from typing import List, Dict
from webscraper.fightodds_selenium_scraper import FightOddsSeleniumScraper, FightOddsData
from src.profitable_predictor import ProfitableUFCPredictor, create_profitable_predictor_from_latest
from src.profitability import BettingOdds as ProfitabilityBettingOdds, BettingOpportunity

def convert_fightodds_to_profitability_format(fight_odds: FightOddsData) -> Dict:
    """Convert FightOddsData to the format expected by the profitability system"""
    
    # Find the best odds for each fighter across all sportsbooks
    best_fighter_a_odds = None
    best_fighter_b_odds = None
    
    if fight_odds.fighter_a_odds:
        # For favorites (negative odds), we want the least negative (closest to 0)
        # For underdogs (positive odds), we want the highest positive
        fighter_a_american_odds = [odds.american_odds for odds in fight_odds.fighter_a_odds]
        if any(odds < 0 for odds in fighter_a_american_odds):
            # Fighter A is favored - get least negative odds
            best_odds = max([odds for odds in fighter_a_american_odds if odds < 0])
        else:
            # Fighter A is underdog - get highest positive odds
            best_odds = max(fighter_a_american_odds)
        
        best_fighter_a_odds = next(odds for odds in fight_odds.fighter_a_odds if odds.american_odds == best_odds)
    
    if fight_odds.fighter_b_odds:
        fighter_b_american_odds = [odds.american_odds for odds in fight_odds.fighter_b_odds]
        if any(odds < 0 for odds in fighter_b_american_odds):
            # Fighter B is favored - get least negative odds
            best_odds = max([odds for odds in fighter_b_american_odds if odds < 0])
        else:
            # Fighter B is underdog - get highest positive odds
            best_odds = max(fighter_b_american_odds)
        
        best_fighter_b_odds = next(odds for odds in fight_odds.fighter_b_odds if odds.american_odds == best_odds)
    
    return {
        'fight_id': f"{fight_odds.fighter_a}_vs_{fight_odds.fighter_b}".replace(' ', '_'),
        'event_name': fight_odds.event_name,
        'fighter_a': fight_odds.fighter_a,
        'fighter_b': fight_odds.fighter_b,
        'fighter_a_odds': best_fighter_a_odds.american_odds if best_fighter_a_odds else None,
        'fighter_b_odds': best_fighter_b_odds.american_odds if best_fighter_b_odds else None,
        'best_fighter_a_sportsbook': best_fighter_a_odds.sportsbook if best_fighter_a_odds else None,
        'best_fighter_b_sportsbook': best_fighter_b_odds.sportsbook if best_fighter_b_odds else None
    }

def main():
    """Main analysis function"""
    print("🎯 LIVE FIGHTODDS.IO PROFITABILITY ANALYSIS")
    print("=" * 55)
    
    # Step 1: Scrape live odds
    print("📡 STEP 1: SCRAPING LIVE ODDS")
    print("-" * 30)
    
    scraper = FightOddsSeleniumScraper(headless=True)  # Headless for efficiency
    live_odds = scraper.scrape_fightodds()
    
    print(f"✅ Successfully scraped {len(live_odds)} fights")
    
    # Step 2: Initialize the profitable predictor
    print("\n🤖 STEP 2: INITIALIZING PREDICTOR")
    print("-" * 30)
    
    try:
        predictor = create_profitable_predictor_from_latest()
        print("✅ Profitable predictor system initialized")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        print("💡 Using sample model probabilities for demonstration")
        predictor = None
    
    # Step 3: Analyze each fight for profitability
    print("\n💰 STEP 3: PROFITABILITY ANALYSIS")
    print("-" * 30)
    
    total_opportunities = 0
    total_expected_profit = 0
    profitable_fights = []
    
    for i, fight in enumerate(live_odds, 1):
        print(f"\n🥊 Fight {i}: {fight.fighter_a} vs {fight.fighter_b}")
        
        # Convert to profitability format
        fight_data = convert_fightodds_to_profitability_format(fight)
        
        print(f"   📊 Best odds: {fight.fighter_a} ({fight_data['fighter_a_odds'] if fight_data['fighter_a_odds'] else 'N/A'}) "
              f"vs {fight.fighter_b} ({fight_data['fighter_b_odds'] if fight_data['fighter_b_odds'] else 'N/A'})")
        
        if predictor:
            try:
                # Use real model predictions
                fighter_a_odds = fight_data['fighter_a_odds'] if fight_data['fighter_a_odds'] else 0
                fighter_b_odds = fight_data['fighter_b_odds'] if fight_data['fighter_b_odds'] else 0
                
                prediction_result = predictor.predict_with_profitability(
                    fight_data['fighter_a'], 
                    fight_data['fighter_b'], 
                    fighter_a_odds, 
                    fighter_b_odds
                )
                
                # Convert to our format
                opportunities = []
                if 'betting_opportunities' in prediction_result and prediction_result['betting_opportunities']:
                    for opp in prediction_result['betting_opportunities']:
                        # Extract numeric values from formatted strings
                        ev = float(opp['expected_value'].replace('%', '')) / 100
                        bet_amount = float(opp['recommended_bet'].replace('$', ''))
                        
                        # Create a simple opportunity object for our display
                        class SimpleOpportunity:
                            def __init__(self, fighter_name, expected_value, recommended_bet, expected_profit):
                                self.fighter_name = fighter_name
                                self.expected_value = expected_value
                                self.recommended_bet = recommended_bet
                                self.expected_profit = expected_profit
                        
                        opportunities.append(SimpleOpportunity(
                            fighter_name=opp['fighter'],
                            expected_value=ev,
                            recommended_bet=bet_amount,
                            expected_profit=bet_amount * ev
                        ))
                        
            except Exception as e:
                print(f"   ❌ Error with real predictor: {e}")
                import traceback
                traceback.print_exc()
                opportunities = []
        else:
            # Use sample predictions for demonstration
            opportunities = analyze_with_sample_predictions(fight_data)
        
        if opportunities:
            fight_profit = sum(opp.expected_profit for opp in opportunities)
            total_expected_profit += fight_profit
            total_opportunities += len(opportunities)
            profitable_fights.append((fight, opportunities))
            
            print(f"   💰 Found {len(opportunities)} profitable opportunities!")
            for opp in opportunities:
                print(f"      🎯 {opp.fighter_name}: {opp.expected_value:.1%} EV, ${opp.recommended_bet:.2f} bet, ${opp.expected_profit:.2f} profit")
        else:
            print(f"   📈 No profitable opportunities found")
    
    # Step 4: Summary
    print(f"\n📊 FINAL SUMMARY")
    print("=" * 30)
    print(f"✅ Total fights analyzed: {len(live_odds)}")
    print(f"💰 Profitable fights found: {len(profitable_fights)}")
    print(f"🎯 Total opportunities: {total_opportunities}")
    print(f"💵 Total expected profit: ${total_expected_profit:.2f}")
    
    if profitable_fights:
        print(f"\n🏆 TOP OPPORTUNITIES:")
        
        # Sort by expected profit
        all_opportunities = []
        for fight, opportunities in profitable_fights:
            for opp in opportunities:
                all_opportunities.append((fight, opp))
        
        all_opportunities.sort(key=lambda x: x[1].expected_profit, reverse=True)
        
        for i, (fight, opp) in enumerate(all_opportunities[:5], 1):
            print(f"   {i}. {opp.fighter_name} in {fight.fighter_a} vs {fight.fighter_b}")
            print(f"      💰 ${opp.recommended_bet:.2f} bet → ${opp.expected_profit:.2f} profit ({opp.expected_value:.1%} EV)")
    
    # Save results
    results = {
        'total_fights': len(live_odds),
        'profitable_fights': len(profitable_fights),
        'total_opportunities': total_opportunities,
        'total_expected_profit': total_expected_profit,
        'opportunities': []
    }
    
    for fight, opportunities in profitable_fights:
        for opp in opportunities:
            results['opportunities'].append({
                'fighter_a': fight.fighter_a,
                'fighter_b': fight.fighter_b,
                'fighter_name': opp.fighter_name,
                'expected_value': opp.expected_value,
                'recommended_bet': opp.recommended_bet,
                'expected_profit': opp.expected_profit
            })
    
    with open('live_profitability_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: live_profitability_analysis.json")

def analyze_with_sample_predictions(fight_data: Dict) -> List:
    """Analyze profitability using sample model predictions"""
    from src.profitability import ProfitabilityOptimizer
    
    # Sample model probabilities (you would replace these with real model predictions)
    sample_predictions = {
        'Charles_Oliveira': 0.65,  # Model thinks Oliveira has 65% chance despite being underdog
        'Ilia_Topuria': 0.35,
        'Alexandre_Pantoja': 0.75,
        'Brandon_Royval': 0.25,
        'Beneil_Dariush': 0.60,  # Model thinks Dariush has good value as slight underdog
        'Josh_Van': 0.40,
        'Niko_Price': 0.15,  # Extreme long shot
        'Alvin_Hines': 0.85,
    }
    
    fighter_a_clean = fight_data['fighter_a'].replace(' ', '_')
    fighter_b_clean = fight_data['fighter_b'].replace(' ', '_')
    
    model_prob_a = sample_predictions.get(fighter_a_clean, 0.5)
    model_prob_b = sample_predictions.get(fighter_b_clean, 1 - model_prob_a)
    
    optimizer = ProfitabilityOptimizer(bankroll=1000)  # $1000 bankroll
    
    opportunities = []
    
    # Simple opportunity class for our analysis
    class SimpleOpportunity:
        def __init__(self, fighter_name, expected_value, recommended_bet, expected_profit):
            self.fighter_name = fighter_name
            self.expected_value = expected_value
            self.recommended_bet = recommended_bet
            self.expected_profit = expected_profit
    
    # Check Fighter A
    if fight_data['fighter_a_odds']:
        ev_a = optimizer.calculate_expected_value(model_prob_a, fight_data['fighter_a_odds'])
        if ev_a > 0:
            kelly_a = optimizer.calculate_kelly_fraction(model_prob_a, fight_data['fighter_a_odds'])
            bet_amount = kelly_a * optimizer.bankroll
            opportunities.append(SimpleOpportunity(
                fighter_name=fight_data['fighter_a'],
                expected_value=ev_a,
                recommended_bet=bet_amount,
                expected_profit=bet_amount * ev_a
            ))
    
    # Check Fighter B
    if fight_data['fighter_b_odds']:
        ev_b = optimizer.calculate_expected_value(model_prob_b, fight_data['fighter_b_odds'])
        if ev_b > 0:
            kelly_b = optimizer.calculate_kelly_fraction(model_prob_b, fight_data['fighter_b_odds'])
            bet_amount = kelly_b * optimizer.bankroll
            opportunities.append(SimpleOpportunity(
                fighter_name=fight_data['fighter_b'],
                expected_value=ev_b,
                recommended_bet=bet_amount,
                expected_profit=bet_amount * ev_b
            ))
    
    return opportunities

if __name__ == "__main__":
    main() 