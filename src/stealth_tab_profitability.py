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
from itertools import combinations

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
        """Analyze profitability using live TAB odds - CLEAN VERSION"""
        
        print("💰 LIVE TAB PROFITABILITY ANALYSIS")
        print("=" * 50)
        
        # Get live odds
        live_odds_data = self.get_live_tab_odds(fight_card)
        
        if not live_odds_data:
            print("❌ No live odds available")
            return {}
        
        # Clean and filter odds to H2H only
        clean_odds_data = self.filter_h2h_odds_only(live_odds_data)
        
        print(f"\n🎯 Found TAB odds for {len(clean_odds_data)} fights")
        
        # Analyze each fight for profitability
        profitable_bets = []
        total_expected_profit = 0.0
        processed_fighters = set()  # Prevent duplicates
        
        for fight_name, h2h_odds in clean_odds_data.items():
            
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
                matched_fighter = self.match_market_to_fighter_clean(market_name, my_predictions, fighter1, fighter2)
                
                if matched_fighter and matched_fighter not in processed_fighters:
                    pred_fighter, my_prob, similarity = matched_fighter
                    
                    # Calculate expected value
                    expected_value = self.calculate_expected_value(my_prob, tab_odds)
                    
                    if expected_value > 0.05:  # 5% EV threshold
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
        
        # Clean summary
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 30)
        print(f"💰 Profitable opportunities: {len(profitable_bets)}")
        print(f"💵 Total expected profit: ${total_expected_profit:.2f}")
        
        if profitable_bets:
            print(f"\n🎯 RECOMMENDED BETS:")
            print("-" * 40)
            for i, bet in enumerate(profitable_bets, 1):
                opponent = self.get_opponent_name(bet['fight'], bet['fighter'])
                print(f"{i}. {bet['fighter']} @ {bet['tab_odds']} (vs {opponent})")
                print(f"   💰 Bet ${bet['bet_size']:.2f} → Expected profit: ${bet['expected_profit']:.2f}")
                print(f"   📈 Expected Value: {bet['expected_value']:.1%}")
                print()
        
        # Analyze multi-bet opportunities if we have enough profitable singles
        multi_bet_opportunities = []
        if len(profitable_bets) >= 2:
            multi_bet_opportunities = self.analyze_multi_bet_opportunities(profitable_bets)
        
        return {
            'profitable_bets': profitable_bets,
            'multi_bet_opportunities': multi_bet_opportunities,
            'total_expected_profit': total_expected_profit,
            'live_odds_data': clean_odds_data,
            'analysis_summary': {
                'total_single_bets': len(profitable_bets),
                'total_multi_bets': len(multi_bet_opportunities),
                'bankroll': self.bankroll,
                'total_expected_return': total_expected_profit
            }
        }
    
    def calculate_expected_value(self, my_prob: float, odds: float) -> float:
        """
        Calculate proper expected value of a bet using decimal odds.
        
        Formula: EV = (My_Probability × Decimal_Odds) - 1
        
        Args:
            my_prob: Model's probability of the outcome (0.0 to 1.0)
            odds: Decimal odds from TAB (e.g., 1.67, 2.50)
            
        Returns:
            Expected value as decimal (e.g., 0.15 = 15% expected return)
        """
        # Proper expected value formula
        ev = (my_prob * odds) - 1
        return ev
    
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
    
    def match_market_to_fighter_clean(self, market_name: str, my_predictions: Dict[str, float], 
                                     fighter1: str, fighter2: str) -> Optional[Tuple[str, float, float]]:
        """Clean version - match market to fighter without verbose logging"""
        
        best_match = None
        best_similarity = 0.0
        
        for pred_fighter, my_prob in my_predictions.items():
            
            # Multiple matching strategies
            similarities = []
            
            # Strategy 1: Direct name match in market (e.g., "H2H TOPURIA Ilia" -> "Ilia Topuria")
            if self.name_in_market(pred_fighter, market_name):
                # CRITICAL FIX: Only match if this fighter should be in this market
                if self.market_is_for_fighter(market_name, pred_fighter):
                    similarities.append(0.98)  # High confidence
                else:
                    continue  # This market is for the opponent
            
            # Strategy 2: Check if prediction fighter matches fight fighters AND market
            fighter1_match = self.names_similar(pred_fighter, fighter1)
            fighter2_match = self.names_similar(pred_fighter, fighter2)
            
            if fighter1_match > 0.8 and self.market_is_for_fighter(market_name, fighter1):
                similarities.append(fighter1_match)
            
            if fighter2_match > 0.8 and self.market_is_for_fighter(market_name, fighter2):
                similarities.append(fighter2_match)
            
            if similarities:
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity and max_similarity > 0.7:
                    best_similarity = max_similarity
                    best_match = (pred_fighter, my_prob, best_similarity)
        
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
        """Extract opponent name from fight string"""
        
        # Handle different fight name formats
        if " vs. " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs. ")
        elif " vs " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs ")
        else:
            return "Unknown"
            
        fighter1 = fighter1.strip()
        fighter2 = fighter2.strip()
        
        # Return the opponent (the one that's not the given fighter)
        if self.names_similar(fighter_name, fighter1) > 0.8:
            return fighter2
        elif self.names_similar(fighter_name, fighter2) > 0.8:
            return fighter1
        else:
            # If no strong match, return the first one that's different
            return fighter2 if fighter_name != fighter1 else fighter1
    
    def extract_fighters_from_fight_name(self, fight_name: str) -> Tuple[str, str]:
        """
        Extract fighter names from fight string.
        
        Args:
            fight_name: Fight string like "Fighter A vs. Fighter B"
            
        Returns:
            Tuple of (fighter1, fighter2)
        """
        # Handle different fight name formats
        if " vs. " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs. ", 1)
        elif " vs " in fight_name:
            fighter1, fighter2 = fight_name.split(" vs ", 1)
        else:
            # Fallback - assume it's a single fighter name or unknown format
            return fight_name.strip(), "Unknown"
            
        return fighter1.strip(), fighter2.strip()
    
    def close(self):
        """Clean up resources"""
        if self.stealth_scraper:
            self.stealth_scraper.close()
    
    def analyze_multi_bet_opportunities(self, profitable_bets: List[Dict], 
                                       max_legs: int = 4, 
                                       min_multi_ev: float = 0.15) -> List[Dict]:
        """
        Analyze multi-bet (parlay) opportunities from profitable single bets.
        
        Args:
            profitable_bets: List of profitable single bet opportunities
            max_legs: Maximum number of legs in a multi-bet (risk management)
            min_multi_ev: Minimum Expected Value threshold for multi-bets
            
        Returns:
            List of profitable multi-bet combinations sorted by EV
        """
        if len(profitable_bets) < 2:
            return []
        
        multi_bet_opportunities = []
        
        print(f"\n🎯 MULTI-BET ANALYSIS")
        print("=" * 30)
        print(f"📊 Analyzing combinations from {len(profitable_bets)} profitable singles")
        print(f"🎲 Max legs: {max_legs}, Min EV threshold: {min_multi_ev:.0%}")
        print()
        
        # Generate all possible combinations (2 to max_legs)
        for num_legs in range(2, min(max_legs + 1, len(profitable_bets) + 1)):
            
            for combo in combinations(profitable_bets, num_legs):
                # Calculate combined probability and odds
                combined_prob = 1.0
                combined_odds = 1.0
                total_stake = 0
                fighters = []
                
                for bet in combo:
                    combined_prob *= bet['my_probability']
                    combined_odds *= bet['tab_odds']
                    total_stake += bet['bet_size']  # For comparison
                    fighters.append(bet['fighter'])
                
                # Calculate multi-bet Expected Value
                multi_ev = (combined_prob * combined_odds) - 1
                
                # Apply correlation penalty (fights on same card may be correlated)
                correlation_penalty = self.calculate_correlation_penalty(combo)
                adjusted_ev = multi_ev * (1 - correlation_penalty)
                
                if adjusted_ev > min_multi_ev:
                    # Calculate optimal stake using modified Kelly
                    optimal_stake = self.calculate_multi_bet_kelly(combined_prob, combined_odds)
                    expected_profit = optimal_stake * adjusted_ev
                    
                    multi_bet_opportunities.append({
                        'legs': num_legs,
                        'fighters': fighters,
                        'fights': [bet['fight'] for bet in combo],
                        'combined_probability': combined_prob,
                        'combined_odds': combined_odds,
                        'raw_ev': multi_ev,
                        'adjusted_ev': adjusted_ev,
                        'correlation_penalty': correlation_penalty,
                        'optimal_stake': optimal_stake,
                        'expected_profit': expected_profit,
                        'risk_level': self.assess_multi_bet_risk(num_legs, combined_prob),
                        'individual_bets': combo
                    })
        
        # Sort by adjusted Expected Value
        multi_bet_opportunities.sort(key=lambda x: x['adjusted_ev'], reverse=True)
        
        # Display balanced opportunities across risk categories
        if multi_bet_opportunities:
            print(f"💎 FOUND {len(multi_bet_opportunities)} PROFITABLE MULTI-BETS")
            print("-" * 50)
            
            # Categorize by risk level
            risk_categories = {
                'LOW RISK': [],
                'MEDIUM RISK': [],
                'HIGH RISK': [],
                'VERY HIGH RISK': []
            }
            
            for multi in multi_bet_opportunities:
                risk_level = multi['risk_level']
                risk_categories[risk_level].append(multi)
            
            # Sort each category separately by EV to ensure best examples from each risk level
            for risk_level in risk_categories:
                risk_categories[risk_level].sort(key=lambda x: x['adjusted_ev'], reverse=True)
            
            # Display strategy: Show at least 1 from each category, up to 2 per category
            display_count = 0
            max_total = 8  # Increased to ensure all categories can be shown
            
            # First pass: Show at least 1 from each category that has bets
            categories_with_bets = []
            for risk_level in ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK', 'VERY HIGH RISK']:
                if risk_categories[risk_level]:
                    categories_with_bets.append(risk_level)
            
            for risk_level in categories_with_bets:
                category_bets = risk_categories[risk_level]
                if category_bets and display_count < max_total:
                    
                    print(f"\n🎯 {risk_level} MULTI-BETS:")
                    print("." * 30)
                    
                    # Determine how many to show from this category
                    remaining_slots = max_total - display_count
                    remaining_categories = len([cat for cat in categories_with_bets[categories_with_bets.index(risk_level):] if risk_categories[cat]])
                    
                    # Ensure at least 1 slot for remaining categories, but allow up to 2 per category
                    slots_for_this_category = min(2, max(1, remaining_slots - remaining_categories + 1))
                    
                    shown_in_category = 0
                    for multi in category_bets[:slots_for_this_category]:
                        if display_count >= max_total:
                            break
                            
                        display_count += 1
                        shown_in_category += 1
                        
                        print(f"{display_count}. {multi['legs']}-LEG MULTI")
                        print(f"   Fighters: {' + '.join(multi['fighters'])}")
                        print(f"   Combined Odds: {multi['combined_odds']:.2f}")
                        print(f"   Win Probability: {multi['combined_probability']:.1%}")
                        print(f"   Expected Value: {multi['adjusted_ev']:.1%}")
                        print(f"   💰 Stake: ${multi['optimal_stake']:.2f} → Profit: ${multi['expected_profit']:.2f}")
                        print()
                    
                    if len(category_bets) > shown_in_category:
                        remaining = len(category_bets) - shown_in_category
                        print(f"   ... and {remaining} more {risk_level.lower()} options")
                        print()
            
            # Summary recommendation
            print("💡 MULTI-BET STRATEGY GUIDE:")
            print("=" * 35)
            
            low_count = len(risk_categories['LOW RISK'])
            medium_count = len(risk_categories['MEDIUM RISK']) 
            high_count = len(risk_categories['HIGH RISK'])
            very_high_count = len(risk_categories['VERY HIGH RISK'])
            
            if low_count > 0:
                print("✅ CONSERVATIVE: Start with LOW RISK 2-leg combinations")
            if medium_count > 0:
                print("⚖️  BALANCED: MEDIUM RISK offers good risk/reward balance")
            if high_count > 0:
                print("⚡ AGGRESSIVE: HIGH RISK for experienced bettors only")
            if very_high_count > 0:
                print("🎰 SPECULATIVE: VERY HIGH RISK = lottery ticket plays")
            
            print(f"\n📊 Distribution: {low_count} Low | {medium_count} Medium | {high_count} High | {very_high_count} Very High")
            print("\n🧮 WIN PROBABILITY CALCULATION:")
            print("   Win % = Fighter1_Prob × Fighter2_Prob × ... × FighterN_Prob")
            print("   (All legs must win for multi-bet to pay out)")
        
        return multi_bet_opportunities
    
    def calculate_correlation_penalty(self, combo: List[Dict]) -> float:
        """
        Calculate correlation penalty for multi-bet legs.
        Fights on the same card may have correlated outcomes.
        """
        # Check if fights are on the same card
        same_card_count = 0
        fight_names = [bet['fight'] for bet in combo]
        
        # Simple heuristic: if fights share common elements, they might be correlated
        for i, fight1 in enumerate(fight_names):
            for fight2 in fight_names[i+1:]:
                # Very basic correlation detection
                if self.fights_potentially_correlated(fight1, fight2):
                    same_card_count += 1
        
        # Apply penalty based on correlation
        base_penalty = 0.05  # 5% base penalty for any multi-bet
        correlation_penalty = same_card_count * 0.03  # 3% per correlated pair
        
        return min(base_penalty + correlation_penalty, 0.25)  # Max 25% penalty
    
    def fights_potentially_correlated(self, fight1: str, fight2: str) -> bool:
        """Simple heuristic to detect potentially correlated fights."""
        # This is a placeholder - in practice, you'd want more sophisticated correlation detection
        # For now, assume fights on the same card (same event) might be correlated
        return True  # Conservative approach - assume some correlation exists
    
    def calculate_multi_bet_kelly(self, combined_prob: float, combined_odds: float, 
                                 max_stake_pct: float = 0.02) -> float:
        """
        Calculate Kelly stake for multi-bet with conservative limits.
        Multi-bets are riskier, so use smaller maximum stakes.
        """
        # Kelly formula for multi-bet
        kelly_fraction = (combined_prob * combined_odds - 1) / (combined_odds - 1)
        
        # Conservative caps for multi-bets (riskier than singles)
        kelly_fraction = min(max(kelly_fraction, 0), max_stake_pct)
        
        return self.bankroll * kelly_fraction
    
    def assess_multi_bet_risk(self, num_legs: int, combined_prob: float) -> str:
        """Assess risk level of multi-bet with more appropriate thresholds."""
        if num_legs <= 2 and combined_prob > 0.35:
            return "LOW RISK"
        elif num_legs <= 2 and combined_prob > 0.25:
            return "MEDIUM RISK"
        elif num_legs == 3 and combined_prob > 0.15:
            return "MEDIUM RISK"
        elif combined_prob > 0.08:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"
    
    def process_cached_odds(self, my_predictions: Dict[str, float], 
                          live_odds_data: Dict) -> Dict:
        """
        Process cached odds data without scraping (for split cell workflow).
        
        Args:
            my_predictions: Dictionary of fighter -> probability
            live_odds_data: Previously scraped odds data
            
        Returns:
            Analysis results including multi-bets
        """
        print("💰 PROCESSING CACHED ODDS DATA")
        print("=" * 35)
        
        if not live_odds_data:
            print("❌ No odds data provided")
            return {'error': 'No odds data'}
        
        # Filter to H2H odds only
        clean_odds_data = self.filter_h2h_odds_only(live_odds_data)
        
        profitable_bets = []
        total_expected_profit = 0
        processed_fighters = set()
        
        print(f"📊 Analyzing {len(clean_odds_data)} fights with H2H odds")
        print("-" * 35)
        
        for fight_name, markets in clean_odds_data.items():
            
            fighter1, fighter2 = self.extract_fighters_from_fight_name(fight_name)
            
            for market_name, tab_odds in markets.items():
                
                matched_fighter = self.match_market_to_fighter_clean(
                    market_name, my_predictions, fighter1, fighter2
                )
                
                if matched_fighter and matched_fighter not in processed_fighters:
                    pred_fighter, my_prob, similarity = matched_fighter
                    
                    # Calculate expected value
                    expected_value = self.calculate_expected_value(my_prob, tab_odds)
                    
                    if expected_value > 0.05:  # 5% EV threshold
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
                        processed_fighters.add(pred_fighter)
        
        # Display single bet results
        print(f"\n📊 SINGLE BET ANALYSIS RESULTS")
        print("=" * 35)
        print(f"💰 Profitable opportunities: {len(profitable_bets)}")
        print(f"💵 Total expected profit: ${total_expected_profit:.2f}")
        
        if profitable_bets:
            print(f"\n🎯 RECOMMENDED SINGLE BETS:")
            print("-" * 30)
            for i, bet in enumerate(profitable_bets, 1):
                opponent = self.get_opponent_name(bet['fight'], bet['fighter'])
                print(f"{i}. {bet['fighter']} @ {bet['tab_odds']} (vs {opponent})")
                print(f"   💰 Bet ${bet['bet_size']:.2f} → Expected profit: ${bet['expected_profit']:.2f}")
                print(f"   📈 Expected Value: {bet['expected_value']:.1%}")
                print()
        
        # Analyze multi-bet opportunities
        multi_bet_opportunities = []
        if len(profitable_bets) >= 2:
            multi_bet_opportunities = self.analyze_multi_bet_opportunities(profitable_bets)
        
        return {
            'profitable_bets': profitable_bets,
            'multi_bet_opportunities': multi_bet_opportunities,
            'total_expected_profit': total_expected_profit,
            'live_odds_data': clean_odds_data,
            'analysis_summary': {
                'total_single_bets': len(profitable_bets),
                'total_multi_bets': len(multi_bet_opportunities),
                'bankroll': self.bankroll,
                'total_expected_return': total_expected_profit
            }
        }

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