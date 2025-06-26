#!/usr/bin/env python3
"""
Stealth TAB Profitability Analyzer
================================

Uses the stealth search scraper to get real TAB odds and perform 
profitability analysis with actual live data.
"""

import sys
import os
sys.path.append('.')

from webscraper.stealth_search_scraper import StealthTABScraper
from typing import Dict, List, Tuple, Optional
import json
from difflib import SequenceMatcher

class StealthTABProfitabilityAnalyzer:
    """Enhanced profitability analyzer using stealth scraping for live odds"""
    
    def __init__(self, bankroll: float = 1000):
        self.bankroll = bankroll
        self.stealth_scraper = StealthTABScraper(headless=True)
        
    def get_live_tab_odds(self, fight_list: List[str]) -> Dict[str, Dict[str, float]]:
        """Get live TAB odds using stealth scraping"""
        
        print("🕵️  GETTING LIVE TAB ODDS WITH STEALTH SCRAPING")
        print("=" * 60)
        
        try:
            # Use stealth scraper to get live odds
            live_odds = self.stealth_scraper.scrape_multiple_fights_stealth(fight_list)
            
            print(f"\n📊 Live Odds Summary:")
            for fight, odds_data in live_odds.items():
                print(f"🥊 {fight}: {len(odds_data)} markets found")
                
            return live_odds
            
        except Exception as e:
            print(f"❌ Failed to get live odds: {e}")
            return {}
    
    def analyze_profitability_live(self, my_predictions: Dict[str, float], 
                                  fight_card: List[str]) -> Dict:
        """Analyze profitability using live TAB odds - FIXED VERSION"""
        
        print("💰 LIVE TAB PROFITABILITY ANALYSIS")
        print("=" * 50)
        
        # Get live odds
        live_odds_data = self.get_live_tab_odds(fight_card)
        
        if not live_odds_data:
            print("❌ No live odds available")
            return {}
        
        # Clean and filter odds to H2H only
        clean_odds_data = self.filter_h2h_odds_only(live_odds_data)
        
        print(f"\n📊 H2H Odds Summary:")
        for fight, h2h_odds in clean_odds_data.items():
            print(f"🥊 {fight}: {len(h2h_odds)} H2H markets")
            for market, odds in h2h_odds.items():
                print(f"   {market}: {odds}")
        
        # Analyze each fight for profitability
        profitable_bets = []
        total_expected_profit = 0.0
        processed_fighters = set()  # Prevent duplicates
        
        for fight_name, h2h_odds in clean_odds_data.items():
            print(f"\n🥊 ANALYZING: {fight_name}")
            print("-" * 40)
            
            # Extract fighter names from fight string
            if " vs. " in fight_name:
                fighter1, fighter2 = fight_name.split(" vs. ")
            elif " vs " in fight_name:
                fighter1, fighter2 = fight_name.split(" vs ")
            else:
                continue
                
            fighter1 = fighter1.strip()
            fighter2 = fighter2.strip()
            
            # Process each H2H market for this fight
            for market_name, tab_odds in h2h_odds.items():
                
                # Skip unrealistic odds
                if tab_odds < 1.01 or tab_odds > 50.0:
                    continue
                
                # Find exact fighter match for this market
                matched_fighter = self.match_market_to_fighter(market_name, my_predictions, fighter1, fighter2)
                
                if matched_fighter and matched_fighter not in processed_fighters:
                    pred_fighter, my_prob, similarity = matched_fighter
                    
                    # Calculate expected value
                    expected_value = self.calculate_expected_value(my_prob, tab_odds)
                    
                    print(f"   🎯 {pred_fighter} ({similarity:.2f} match)")
                    print(f"      My prediction: {my_prob:.1%}")
                    print(f"      TAB H2H odds: {tab_odds}")
                    print(f"      Expected value: {expected_value:.3f}")
                    
                    if expected_value > 0.05:  # 5% edge threshold
                        bet_size = self.calculate_kelly_bet_size(my_prob, tab_odds)
                        expected_profit = bet_size * expected_value
                        
                        profitable_bets.append({
                            'fighter': pred_fighter,
                            'fight': fight_name,
                            'market': market_name,
                            'my_probability': my_prob,
                            'tab_odds': tab_odds,
                            'expected_value': expected_value,
                            'bet_size': bet_size,
                            'expected_profit': expected_profit,
                            'match_quality': similarity
                        })
                        
                        total_expected_profit += expected_profit
                        processed_fighters.add(pred_fighter)  # Mark as processed
                        
                        print(f"      💰 PROFITABLE! Bet ${bet_size:.2f} for ${expected_profit:.2f} expected profit")
                    else:
                        print(f"      ❌ Not profitable (EV: {expected_value:.3f})")
        
        # Summary
        print(f"\n📊 PROFITABILITY SUMMARY")
        print("=" * 30)
        print(f"💰 Total profitable bets: {len(profitable_bets)}")
        print(f"💰 Total expected profit: ${total_expected_profit:.2f}")
        
        if profitable_bets:
            print(f"\n🎯 RECOMMENDED BETS:")
            for i, bet in enumerate(profitable_bets, 1):
                opponent = self.get_opponent_name(bet['fight'], bet['fighter'])
                print(f"{i}. {bet['fighter']} @ {bet['tab_odds']} (vs {opponent}) - Bet ${bet['bet_size']:.2f} (EV: +${bet['expected_profit']:.2f})")
        
        return {
            'profitable_bets': profitable_bets,
            'total_expected_profit': total_expected_profit,
            'live_odds_data': clean_odds_data,
            'analysis_summary': {
                'total_bets_found': len(profitable_bets),
                'bankroll': self.bankroll,
                'total_expected_return': total_expected_profit
            }
        }
    
    def calculate_expected_value(self, my_prob: float, odds: float) -> float:
        """Calculate expected value of a bet"""
        implied_prob = 1.0 / odds
        edge = my_prob - implied_prob
        return edge
    
    def calculate_kelly_bet_size(self, my_prob: float, odds: float, 
                               max_bet_pct: float = 0.05) -> float:
        """Calculate Kelly Criterion bet size with conservative cap"""
        
        # Kelly formula: (bp - q) / b
        # where b = odds - 1, p = my probability, q = 1 - p
        b = odds - 1
        p = my_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at max_bet_pct of bankroll for safety
        kelly_fraction = min(kelly_fraction, max_bet_pct)
        kelly_fraction = max(kelly_fraction, 0)  # Never negative
        
        return self.bankroll * kelly_fraction
    
    def filter_h2h_odds_only(self, live_odds_data: Dict) -> Dict:
        """Filter to only H2H (Head to Head) odds, removing other bet types"""
        
        clean_data = {}
        
        for fight_name, all_markets in live_odds_data.items():
            h2h_markets = {}
            
            for market_name, odds_value in all_markets.items():
                # Only keep H2H markets
                if self.is_h2h_market(market_name):
                    h2h_markets[market_name] = odds_value
            
            if h2h_markets:
                clean_data[fight_name] = h2h_markets
        
        return clean_data
    
    def is_h2h_market(self, market_name: str) -> bool:
        """Check if a market is a Head to Head (winner) market"""
        
        market_lower = market_name.lower()
        
        # H2H indicators
        h2h_indicators = [
            'h2h',
            'head to head',
            'winner',
            'to win'
        ]
        
        # Non-H2H indicators to exclude
        exclude_indicators = [
            'method',
            'round',
            'total rounds',
            'ko/tko',
            'submission',
            'decision',
            'distance',
            'time',
            'combined',
            'exact',
            'over',
            'under'
        ]
        
        # Must have H2H indicator
        has_h2h = any(indicator in market_lower for indicator in h2h_indicators)
        
        # Must not have exclude indicators
        has_exclude = any(indicator in market_lower for indicator in exclude_indicators)
        
        return has_h2h and not has_exclude
    
    def match_market_to_fighter(self, market_name: str, my_predictions: Dict[str, float], 
                               fighter1: str, fighter2: str) -> Optional[Tuple[str, float, float]]:
        """Match a market to exactly one fighter from predictions"""
        
        best_match = None
        best_similarity = 0.0
        
        print(f"      🔍 Matching market '{market_name}' to predictions...")
        
        for pred_fighter, my_prob in my_predictions.items():
            
            # Multiple matching strategies
            similarities = []
            
            # Strategy 1: Direct name match in market (e.g., "H2H TOPURIA Ilia" -> "Ilia Topuria")
            if self.name_in_market(pred_fighter, market_name):
                similarities.append(0.95)  # High confidence for name match
                print(f"         📍 Name match: {pred_fighter} found in {market_name}")
                
                # CRITICAL FIX: Only match if this fighter should be in this market
                # Check if market is for this specific fighter
                if self.market_is_for_fighter(market_name, pred_fighter):
                    similarities.append(0.98)  # Even higher confidence
                    print(f"         ✅ Market confirmed for {pred_fighter}")
                else:
                    # This market is for the opponent, not this fighter
                    print(f"         ❌ Market is for opponent, not {pred_fighter}")
                    continue
            
            # Strategy 2: Check if prediction fighter matches fight fighters AND market
            fighter1_match = self.names_similar(pred_fighter, fighter1)
            fighter2_match = self.names_similar(pred_fighter, fighter2)
            
            if fighter1_match > 0.8 and self.market_is_for_fighter(market_name, fighter1):
                similarities.append(fighter1_match)
                print(f"         📍 Fighter1 match: {pred_fighter} ≈ {fighter1} ({fighter1_match:.2f}) AND market matches")
            
            if fighter2_match > 0.8 and self.market_is_for_fighter(market_name, fighter2):
                similarities.append(fighter2_match)  
                print(f"         📍 Fighter2 match: {pred_fighter} ≈ {fighter2} ({fighter2_match:.2f}) AND market matches")
            
            if similarities:
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity and max_similarity > 0.7:  # Lowered threshold
                    best_similarity = max_similarity
                    best_match = (pred_fighter, my_prob, best_similarity)
                    print(f"         ✅ Best match so far: {pred_fighter} ({max_similarity:.2f})")
        
        if best_match:
            pred_fighter, my_prob, similarity = best_match
            print(f"      🎯 Final match: {pred_fighter} ({similarity:.2f})")
        else:
            print(f"      ❌ No match found for {market_name}")
        
        return best_match
    
    def market_is_for_fighter(self, market_name: str, fighter_name: str) -> bool:
        """Check if a market is specifically for a given fighter"""
        
        market_lower = market_name.lower()
        fighter_parts = fighter_name.lower().split()
        
        # For H2H markets like "H2H TOPURIA Ilia", check if fighter parts are in market
        for part in fighter_parts:
            if len(part) >= 3 and part in market_lower:
                return True
        
        # Also check for last name specifically (most common in markets)
        if fighter_parts:
            last_name = fighter_parts[-1]
            if len(last_name) >= 4 and last_name in market_lower:
                return True
        
        return False
    
    def name_in_market(self, fighter_name: str, market_name: str) -> bool:
        """Check if fighter name appears in market name"""
        
        fighter_parts = fighter_name.split()
        market_lower = market_name.lower()
        
        # Check if all parts of fighter name are in market
        for part in fighter_parts:
            if len(part) >= 3 and part.lower() in market_lower:
                return True
        
        return False
    
    def names_similar(self, name1: str, name2: str) -> float:
        """Calculate similarity between two fighter names"""
        
        from difflib import SequenceMatcher
        
        # Direct comparison
        direct_sim = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        
        # Compare parts (first/last name flexibility)
        parts1 = name1.lower().split()
        parts2 = name2.lower().split()
        
        part_similarities = []
        
        for p1 in parts1:
            for p2 in parts2:
                if len(p1) >= 3 and len(p2) >= 3:
                    part_sim = SequenceMatcher(None, p1, p2).ratio()
                    part_similarities.append(part_sim)
        
        max_part_sim = max(part_similarities) if part_similarities else 0
        
        return max(direct_sim, max_part_sim)
    
    def partial_name_match(self, fighter_name: str, market_name: str) -> bool:
        """Check for partial name matches (last name, etc.)"""
        
        fighter_parts = fighter_name.split()
        market_lower = market_name.lower()
        
        # Check last name (usually most distinctive)
        if fighter_parts:
            last_name = fighter_parts[-1].lower()
            if len(last_name) >= 4 and last_name in market_lower:
                return True
        
        # Check first name
        if len(fighter_parts) >= 2:
            first_name = fighter_parts[0].lower()
            if len(first_name) >= 4 and first_name in market_lower:
                return True
        
        return False
    
    def get_opponent_name(self, fight_name: str, fighter_name: str) -> str:
        """Get the opponent's name from a fight string"""
        
        if " vs. " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs. ")
        elif " vs " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs ")
        else:
            return "opponent"
        
        fighter1 = fighter1.strip()
        fighter2 = fighter2.strip()
        
        # Return the fighter that isn't the input fighter
        if fighter_name.lower() in fighter1.lower():
            return fighter2
        elif fighter_name.lower() in fighter2.lower():
            return fighter1
        else:
            return "opponent"
    
    def close(self):
        """Clean up resources"""
        if self.stealth_scraper:
            self.stealth_scraper.close()

def test_stealth_profitability():
    """Test stealth profitability analysis with sample data"""
    
    # Sample fight card
    fight_card = [
        "Ilia Topuria vs. Charles Oliveira",
        "Alexandre Pantoja vs. Kai Kara-France"
    ]
    
    # Sample predictions
    my_predictions = {
        "Ilia Topuria": 0.75,      # I think Topuria has 75% chance
        "Charles Oliveira": 0.25,   # Oliveira 25% chance
        "Alexandre Pantoja": 0.60,  # Pantoja 60% chance  
        "Kai Kara-France": 0.40     # Kara-France 40% chance
    }
    
    print("🕵️  TESTING STEALTH TAB PROFITABILITY")
    print("=" * 50)
    
    analyzer = StealthTABProfitabilityAnalyzer(bankroll=1000)
    
    try:
        results = analyzer.analyze_profitability_live(my_predictions, fight_card)
        
        # Save results
        with open('stealth_profitability_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to stealth_profitability_results.json")
        
        print(f"\n🎯 STEALTH PROFITABILITY SUMMARY:")
        print("✅ Uses real live TAB odds")
        print("✅ Bypasses bot detection") 
        print("✅ Accurate fighter matching")
        print("✅ Kelly Criterion bet sizing")
        print("✅ Conservative profit estimates")
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    test_stealth_profitability() 