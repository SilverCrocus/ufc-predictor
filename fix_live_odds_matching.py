#!/usr/bin/env python3
"""
Fix Live Odds Matching - Smart TAB Profitability
================================================

This fixes the real problems with TAB profitability analysis:
1. Focus live scraping on user's specific fight card  
2. Fix broken fighter matching logic
3. Handle missing H2H data intelligently

This keeps live odds (which you want!) but makes them work correctly.
"""

import sys
sys.path.append('.')

from src.tab_profitability import TABProfitabilityAnalyzer
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import json

class SmartTABAnalyzer(TABProfitabilityAnalyzer):
    """Enhanced TAB analyzer that focuses on specific fight cards"""
    
    def __init__(self, bankroll: float = 1000, target_fights: List[str] = None):
        super().__init__(bankroll=bankroll, use_live_odds=True)
        self.target_fights = target_fights or []
        self.target_fighters = self._extract_target_fighters()
        
    def _extract_target_fighters(self) -> List[str]:
        """Extract individual fighter names from fight card"""
        fighters = []
        for fight in self.target_fights:
            if " vs. " in fight:
                f1, f2 = fight.split(" vs. ")
                fighters.extend([f1.strip(), f2.strip()])
            elif " vs " in fight:
                f1, f2 = fight.split(" vs ")
                fighters.extend([f1.strip(), f2.strip()])
        return fighters
    
    def enhanced_fighter_matching(self, target_name: str, scraped_odds: Dict[str, float]) -> Tuple[str, float, float]:
        """Enhanced matching that focuses on target fighters and handles missing data"""
        
        print(f"🎯 Looking for: {target_name}")
        
        # Direct H2H match (best case)
        for market_name, odds in scraped_odds.items():
            if market_name.startswith("H2H "):
                h2h_part = market_name.replace("H2H ", "").strip()
                parts = h2h_part.split()
                
                if len(parts) >= 2:
                    # TAB format: "H2H LASTNAME Firstname"  
                    tab_first = parts[1].lower()
                    tab_last = parts[0].lower()
                    full_name = f"{tab_first} {tab_last}".title()
                    
                    # Check similarity
                    similarity = SequenceMatcher(None, target_name.lower(), full_name.lower()).ratio()
                    if similarity >= 0.8:
                        print(f"   ✅ Direct H2H match: {market_name} → {full_name} (sim: {similarity:.2f}, odds: {odds})")
                        return market_name, odds, similarity
        
        # Fight market match (calculate opponent)
        target_parts = target_name.lower().split()
        target_last = target_parts[-1] if target_parts else ""
        
        for market_name, odds in scraped_odds.items():
            if " v " in market_name:
                fighters = market_name.split(" v ")
                if len(fighters) == 2:
                    left_fighter = fighters[0].strip()
                    right_fighter = fighters[1].strip()
                    
                    # Check if target matches left fighter
                    left_sim = SequenceMatcher(None, target_name.lower(), left_fighter.lower()).ratio()
                    if left_sim >= 0.7:
                        print(f"   ✅ Fight market (LEFT): {market_name} → {left_fighter} (sim: {left_sim:.2f}, odds: {odds})")
                        return market_name, odds, left_sim
                    
                    # Check if target matches right fighter (calculate implied odds)
                    right_sim = SequenceMatcher(None, target_name.lower(), right_fighter.lower()).ratio()
                    if right_sim >= 0.7:
                        # Calculate implied odds for right fighter
                        if odds > 1.0:
                            left_prob = 1.0 / odds
                            right_prob = 1.0 - left_prob
                            right_odds = round(1.0 / right_prob, 2) if right_prob > 0 else 99.0
                            print(f"   ✅ Fight market (RIGHT): {market_name} → {right_fighter} (sim: {right_sim:.2f}, calculated odds: {right_odds})")
                            return f"{market_name} (RIGHT)", right_odds, right_sim
        
        print(f"   ❌ No match found for {target_name}")
        return "", 0.0, 0.0
    
    def smart_analyze_predictions(self, model_predictions: Dict[str, float]) -> Dict:
        """Smart analysis that focuses on your specific fighters"""
        
        print("🎯 SMART TAB PROFITABILITY ANALYSIS")
        print("=" * 50)
        print(f"📋 Target fights: {len(self.target_fights)}")
        for i, fight in enumerate(self.target_fights, 1):
            print(f"   {i}. {fight}")
        
        print(f"🎯 Target fighters: {len(self.target_fighters)}")
        for fighter in self.target_fighters:
            print(f"   • {fighter}")
        
        # Get live odds
        print("\n🔄 Scraping LIVE TAB odds...")
        raw_odds = self.scrape_live_tab_odds()
        
        if not raw_odds:
            return {
                'error': 'No live odds available',
                'total_opportunities': 0,
                'opportunities': []
            }
        
        print(f"📊 Scraped {len(raw_odds)} live odds entries")
        
        # Enhanced matching for your specific fighters only
        matched_odds = {}
        
        print("\n🎯 ENHANCED MATCHING FOR YOUR FIGHTERS:")
        print("-" * 40)
        
        for fighter, model_prob in model_predictions.items():
            if fighter in self.target_fighters:
                market_name, odds, similarity = self.enhanced_fighter_matching(fighter, raw_odds)
                if odds > 0:
                    matched_odds[fighter] = odds
            else:
                print(f"⏭️  Skipping {fighter} (not in target card)")
        
        print(f"\n📊 Matched {len(matched_odds)}/{len([f for f in model_predictions.keys() if f in self.target_fighters])} target fighters")
        
        # Continue with normal profitability analysis
        opportunities = []
        total_expected_profit = 0
        
        for fighter, model_prob in model_predictions.items():
            if fighter in matched_odds:
                decimal_odds = matched_odds[fighter]
                american_odds = self.decimal_to_american_odds(decimal_odds)
                
                ev = self.optimizer.calculate_expected_value(model_prob, american_odds)
                
                if ev > 0:
                    kelly = self.optimizer.calculate_kelly_fraction(model_prob, american_odds)
                    bet_amount = kelly * self.bankroll
                    expected_profit = bet_amount * ev
                    market_prob = self.optimizer.american_odds_to_probability(american_odds)
                    
                    from src.tab_profitability import TABOpportunity
                    opportunity = TABOpportunity(
                        fighter=fighter,
                        opponent=self._find_opponent(fighter, model_predictions),
                        event="UFC Live",
                        tab_decimal_odds=decimal_odds,
                        american_odds=american_odds,
                        model_prob=model_prob,
                        market_prob=market_prob,
                        expected_value=ev,
                        recommended_bet=bet_amount,
                        expected_profit=expected_profit
                    )
                    
                    opportunities.append(opportunity)
                    total_expected_profit += expected_profit
                    
                    print(f"💰 {fighter}: {ev:.1%} EV, ${bet_amount:.2f} bet, ${expected_profit:.2f} profit")
        
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
        
        return {
            'total_opportunities': len(opportunities),
            'total_expected_profit': total_expected_profit,
            'bankroll': self.bankroll,
            'opportunities': opportunities,
            'odds_source': 'smart_live',
            'matched_fighters': len(matched_odds),
            'total_fighters': len(model_predictions),
            'target_fighters': len(self.target_fighters)
        }

def test_smart_analyzer():
    """Test the smart analyzer with user's fight card"""
    
    # User's fight card
    target_fights = [
        "Ilia Topuria vs. Charles Oliveira",
        "Alexandre Pantoja vs. Kai Kara-France",
        "Brandon Royval vs. Joshua Van",
        "Beneil Dariush vs. Renato Moicano",
        "Payton Talbott vs. Felipe Lima"
    ]
    
    # User's predictions
    model_predictions = {
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
    
    print("🚀 TESTING SMART TAB ANALYZER")
    print("=" * 50)
    print("💡 This keeps LIVE odds but fixes the matching!")
    print()
    
    # Create smart analyzer
    analyzer = SmartTABAnalyzer(bankroll=1000, target_fights=target_fights)
    
    # Run smart analysis
    results = analyzer.smart_analyze_predictions(model_predictions)
    
    print(f"\n📊 SMART ANALYSIS RESULTS:")
    print(f"✅ Target fighters: {results.get('target_fighters', 0)}")
    print(f"✅ Matched fighters: {results.get('matched_fighters', 0)}")
    print(f"💰 Opportunities: {results.get('total_opportunities', 0)}")
    print(f"💵 Expected profit: ${results.get('total_expected_profit', 0):.2f}")
    print(f"📡 Odds source: {results.get('odds_source', 'unknown')}")
    
    if results.get('opportunities'):
        print("\n🏆 PROFITABLE OPPORTUNITIES:")
        for opp in results['opportunities']:
            print(f"   {opp.fighter}: {opp.expected_value:.1%} EV, ${opp.expected_profit:.2f} profit")
    
    print("\n🎯 SOLUTION SUMMARY:")
    print("✅ Keeps LIVE odds scraping (what you want!)")
    print("✅ Focuses on YOUR fight card only")  
    print("✅ Fixed matching logic")
    print("✅ Handles missing H2H data")
    print("\n💡 This gives you real-time odds for actual betting!")

if __name__ == "__main__":
    test_smart_analyzer() 