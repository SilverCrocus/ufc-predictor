"""
TAB Australia Profitability Module

Import this into your UFC predictions notebook to analyze profitability
based on actual LIVE TAB Australia odds scraped in real-time.

Usage:
    from src.tab_profitability import TABProfitabilityAnalyzer
    
    analyzer = TABProfitabilityAnalyzer(bankroll=1000)
    results = analyzer.analyze_predictions(model_predictions)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from src.profitability import ProfitabilityOptimizer

# Import the TAB Australia scraper
sys.path.append(str(Path(__file__).parent.parent))
from webscraper.tab_australia_scraper import TABAustraliaUFCScraper

@dataclass
class TABOpportunity:
    """TAB Australia betting opportunity"""
    fighter: str
    opponent: str
    event: str
    tab_decimal_odds: float
    american_odds: int
    model_prob: float
    market_prob: float
    expected_value: float
    recommended_bet: float
    expected_profit: float

class TABProfitabilityAnalyzer:
    """Analyze profitability for TAB Australia UFC betting with LIVE odds scraping"""
    
    def __init__(self, bankroll: float = 1000, use_live_odds: bool = True):
        self.bankroll = bankroll
        self.optimizer = ProfitabilityOptimizer(bankroll=bankroll)
        self.use_live_odds = use_live_odds
        self.current_tab_odds = {}
        self.scraper = None
        
        # Initialize scraper if using live odds
        if self.use_live_odds:
            self.scraper = TABAustraliaUFCScraper(headless=True)
            print("🔄 Initializing TAB Australia live odds scraper...")
        else:
            print("⚠️  Live scraping disabled - no odds available")
    
    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings (0.0 to 1.0)"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def find_best_match(self, target_name: str, scraped_names: List[str], threshold: float = 0.6) -> Tuple[str, float]:
        """Find the best matching scraped name for a target fighter name"""
        best_match = ""
        best_score = 0.0
        
        # Split target name into parts
        target_parts = target_name.lower().split()
        target_first = target_parts[0] if target_parts else ""
        target_last = target_parts[-1] if len(target_parts) > 1 else target_parts[0] if target_parts else ""
        
        for scraped_name in scraped_names:
            # Skip obvious non-fighter entries
            if any(skip in scraped_name.lower() for skip in ['market', 'vs', 'ufc', 'round', 'method', 'total', 'over', 'under']):
                continue
            
            # Try different variations of the scraped name
            variations = [
                scraped_name,
                scraped_name.replace("H2H ", ""),
                scraped_name.replace("v ", "vs "),
            ]
            
            # Extract names from "Fighter v Fighter" format
            if " v " in scraped_name:
                parts = scraped_name.split(" v ")
                if len(parts) == 2:
                    variations.extend([parts[0].strip(), parts[1].strip()])
            
            # Extract from "H2H LASTNAME Firstname" format
            if "H2H " in scraped_name:
                h2h_part = scraped_name.replace("H2H ", "").strip()
                h2h_parts = h2h_part.split()
                if len(h2h_parts) >= 2:
                    # TAB format: "H2H LASTNAME Firstname"
                    tab_last = h2h_parts[0].lower()
                    tab_first = h2h_parts[1].lower()
                    variations.extend([
                        f"{tab_first} {tab_last}",
                        f"{tab_first.title()} {tab_last.title()}",
                        tab_last,
                        tab_first
                    ])
            
            for variation in variations:
                variation = variation.strip()
                if not variation:
                    continue
                
                # Multiple scoring approaches
                scores = []
                
                # 1. Direct similarity with full name
                scores.append(self.similarity(target_name, variation))
                
                # 2. Last name matching (most important for fighters)
                if target_last:
                    variation_parts = variation.lower().split()
                    for part in variation_parts:
                        if len(part) > 2:  # Avoid short words
                            scores.append(self.similarity(target_last, part) * 1.2)  # Boost last name matches
                
                # 3. First name matching
                if target_first:
                    variation_parts = variation.lower().split()
                    for part in variation_parts:
                        if len(part) > 2:
                            scores.append(self.similarity(target_first, part))
                
                # 4. Contains matching (for partial names)
                if target_last in variation.lower() or variation.lower() in target_last:
                    scores.append(0.9)
                if target_first in variation.lower() or variation.lower() in target_first:
                    scores.append(0.8)
                
                # 5. Reverse name order matching (First Last vs Last First)
                if len(target_parts) >= 2:
                    reversed_target = f"{target_last} {target_first}"
                    scores.append(self.similarity(reversed_target, variation))
                
                # Take the best score for this variation
                if scores:
                    max_score = max(scores)
                    if max_score > best_score:
                        best_score = max_score
                        best_match = scraped_name
        
        return (best_match, best_score) if best_score >= threshold else ("", 0.0)
    
    def scrape_live_tab_odds(self) -> Dict[str, float]:
        """Scrape live odds from TAB Australia"""
        if not self.use_live_odds or not self.scraper:
            print("⚠️  Live scraping disabled")
            return {}
        
        try:
            print("🇦🇺 Scraping live TAB Australia odds...")
            
            # Get all current UFC fights from TAB
            tab_fights = self.scraper.scrape_all_ufc_odds()
            
            # Extract raw odds data
            raw_odds = {}
            for fight in tab_fights:
                if fight.fighter_a and fight.fighter_a_decimal_odds:
                    raw_odds[fight.fighter_a] = fight.fighter_a_decimal_odds
                if fight.fighter_b and fight.fighter_b_decimal_odds:
                    raw_odds[fight.fighter_b] = fight.fighter_b_decimal_odds
            
            print(f"📊 Scraped {len(raw_odds)} raw odds entries")
            
            # Save raw data for debugging
            with open('raw_tab_odds.json', 'w') as f:
                json.dump(raw_odds, f, indent=2)
            print("💾 Raw odds saved to raw_tab_odds.json")
            
            return raw_odds
            
        except Exception as e:
            print(f"❌ Error scraping live odds: {e}")
            print("⚠️  No odds available - analysis will show no opportunities")
            return {}
    
    def match_fighters_to_odds(self, model_predictions: Dict[str, float], raw_odds: Dict[str, float]) -> Dict[str, float]:
        """Match prediction fighter names to scraped odds using fuzzy matching with smart fight pairing"""
        
        matched_odds = {}
        scraped_names = list(raw_odds.keys())
        used_markets = set()  # Track which markets we've already used
        
        print("🔍 MATCHING FIGHTERS TO ODDS:")
        print("-" * 40)
        
        # First pass: match fighters to unique markets
        for fighter_name in model_predictions.keys():
            best_match, score = self.find_best_match(fighter_name, scraped_names)
            
            if best_match and score > 0.6:
                # For fight markets like "Oliveira v Topuria", we need to assign different odds to each fighter
                if best_match not in used_markets:
                    # First fighter gets the odds as-is
                    matched_odds[fighter_name] = raw_odds[best_match]
                    used_markets.add(best_match)
                    print(f"✅ {fighter_name} → {best_match} ({score:.2f} match, odds: {raw_odds[best_match]})")
                else:
                    # Second fighter in same fight - calculate implied opponent odds
                    original_odds = raw_odds[best_match]
                    # If original odds are for the other fighter, calculate the opponent odds
                    # For decimal odds: opponent_odds = 1 / (1 - 1/original_odds)
                    if original_odds > 1:
                        opponent_prob = 1 - (1 / original_odds)
                        opponent_odds = 1 / opponent_prob if opponent_prob > 0 else 1.01
                        matched_odds[fighter_name] = round(opponent_odds, 2)
                        print(f"✅ {fighter_name} → {best_match} ({score:.2f} match, calculated opponent odds: {opponent_odds:.2f})")
                    else:
                        print(f"❌ {fighter_name} → Invalid odds calculation for {best_match}")
            else:
                print(f"❌ {fighter_name} → No match found (best: {score:.2f})")
        
        print(f"\n📊 Successfully matched {len(matched_odds)}/{len(model_predictions)} fighters")
        
        # If we have missing fighters, run debug mode
        if len(matched_odds) < len(model_predictions):
            self.debug_missing_fighters(model_predictions, raw_odds)
        
        return matched_odds
    
    def decimal_to_american_odds(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def analyze_predictions(self, model_predictions: Dict[str, float]) -> Dict:
        """
        Analyze profitability for all fights with available TAB odds
        
        Args:
            model_predictions: Dict mapping fighter names to win probabilities
        
        Returns:
            Dict containing analysis results
        """
        
        # Get fresh odds from TAB Australia
        raw_odds = self.scrape_live_tab_odds()
        
        if not raw_odds:
            print("❌ No live odds available - cannot perform analysis")
            return {
                'total_opportunities': 0,
                'total_expected_profit': 0,
                'bankroll': self.bankroll,
                'opportunities': [],
                'odds_source': 'none',
                'error': 'No live odds available'
            }
        
        # Match fighter names to odds
        matched_odds = self.match_fighters_to_odds(model_predictions, raw_odds)
        
        opportunities = []
        total_expected_profit = 0
        
        print("\n🇦🇺 TAB AUSTRALIA PROFITABILITY ANALYSIS")
        print("=" * 50)
        print("📡 Using LIVE TAB Australia odds")
        print("-" * 50)
        
        for fighter, model_prob in model_predictions.items():
            if fighter in matched_odds:
                decimal_odds = matched_odds[fighter]
                american_odds = self.decimal_to_american_odds(decimal_odds)
                
                # Calculate profitability
                ev = self.optimizer.calculate_expected_value(model_prob, american_odds)
                
                if ev > 0:
                    kelly = self.optimizer.calculate_kelly_fraction(model_prob, american_odds)
                    bet_amount = kelly * self.bankroll
                    expected_profit = bet_amount * ev
                    market_prob = self.optimizer.american_odds_to_probability(american_odds)
                    
                    opportunity = TABOpportunity(
                        fighter=fighter,
                        opponent=self._find_opponent(fighter, model_predictions),
                        event="UFC 317",
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
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
        
        results = {
            'total_opportunities': len(opportunities),
            'total_expected_profit': total_expected_profit,
            'bankroll': self.bankroll,
            'opportunities': opportunities,
            'odds_source': 'live',
            'matched_fighters': len(matched_odds),
            'total_fighters': len(model_predictions)
        }
        
        # Don't auto-print summary - let notebook handle display
        return results
    
    def _find_opponent(self, fighter: str, model_predictions: Dict[str, float]) -> str:
        """Find the opponent for a given fighter"""
        
        # Known fight matchups for UFC 317
        matchups = {
            'Charles Oliveira': 'Ilia Topuria',
            'Ilia Topuria': 'Charles Oliveira',
            'Alexandre Pantoja': 'Kai Kara-France',
            'Kai Kara-France': 'Alexandre Pantoja',
            'Brandon Royval': 'Joshua Van',
            'Joshua Van': 'Brandon Royval',
            'Beneil Dariush': 'Renato Moicano',
            'Renato Moicano': 'Beneil Dariush',
            'Payton Talbott': 'Felipe Lima',
            'Felipe Lima': 'Payton Talbott',
        }
        
        return matchups.get(fighter, "Unknown")
    
    def _print_summary(self, results: Dict):
        """Print formatted summary of results"""
        
        print(f"\n📊 SUMMARY")
        print("-" * 30)
        print(f"💰 Profitable opportunities: {results['total_opportunities']}")
        print(f"💵 Total expected profit: ${results['total_expected_profit']:.2f} AUD")
        print(f"💳 Bankroll: ${results['bankroll']:.2f} AUD")
        print(f"📡 Odds source: {results['odds_source'].upper()}")
        print(f"🎯 Matched fighters: {results.get('matched_fighters', 0)}/{results.get('total_fighters', 0)}")
        
        if results['opportunities']:
            print(f"\n🏆 TOP OPPORTUNITIES:")
            for i, opp in enumerate(results['opportunities'][:3], 1):
                print(f"   {i}. {opp.fighter} vs {opp.opponent}")
                print(f"      💰 TAB: {opp.tab_decimal_odds} ({opp.american_odds:+d})")
                print(f"      🎯 Bet ${opp.recommended_bet:.2f} → ${opp.expected_profit:.2f} profit ({opp.expected_value:.1%} EV)")
        else:
            if results.get('matched_fighters', 0) == 0:
                print("\n⚠️  No fighters could be matched to TAB odds")
                print("   Check raw_tab_odds.json to see what was scraped")
            else:
                print("\n📊 No profitable opportunities found with current odds")
    
    def get_betting_instructions(self, results: Dict) -> List[str]:
        """Get step-by-step betting instructions"""
        
        if not results['opportunities']:
            if results.get('error'):
                return [f"❌ {results['error']}"]
            else:
                return ["❌ No profitable opportunities found"]
        
        instructions = [
            "1. 🔍 Log into TAB Australia",
            "2. 🥊 Navigate to UFC section", 
            "3. 💰 Place the following bets:",
        ]
        
        for opp in results['opportunities']:
            instructions.append(
                f"   • {opp.fighter}: ${opp.recommended_bet:.2f} at {opp.tab_decimal_odds} odds"
            )
        
        instructions.extend([
            f"4. 💵 Total stake: ${sum(opp.recommended_bet for opp in results['opportunities']):.2f}",
            f"5. 🎯 Expected profit: ${results['total_expected_profit']:.2f}",
            "6. 📊 Track results to validate model performance",
            "",
            f"⚠️  IMPORTANT: Verify odds haven't changed since analysis (live odds used)"
        ])
        
        return instructions

    def debug_missing_fighters(self, model_predictions: Dict[str, float], raw_odds: Dict[str, float]):
        """Debug why certain fighters aren't matching"""
        
        scraped_names = list(raw_odds.keys())
        
        print("\n🔍 DEBUGGING MISSING FIGHTERS:")
        print("-" * 50)
        
        for fighter_name in model_predictions.keys():
            best_match, score = self.find_best_match(fighter_name, scraped_names, threshold=0.4)  # Lower threshold for debugging
            
            if score < 0.6:
                print(f"\n❌ {fighter_name} (best match: {score:.2f})")
                print(f"   Best candidate: '{best_match}'")
                
                # Show potential matches
                fighter_parts = fighter_name.lower().split()
                print(f"   Looking for: {fighter_parts}")
                
                potential_matches = []
                for scraped_name in scraped_names:
                    temp_score = 0
                    for part in fighter_parts:
                        if part in scraped_name.lower():
                            temp_score += 0.5
                    if temp_score > 0:
                        potential_matches.append((scraped_name, temp_score))
                
                potential_matches.sort(key=lambda x: x[1], reverse=True)
                if potential_matches:
                    print("   Potential candidates:")
                    for name, score in potential_matches[:3]:
                        print(f"     - '{name}' (contains score: {score})")
                else:
                    print("   No potential candidates found")

def analyze_notebook_predictions(model_predictions: Dict[str, float], bankroll: float = 1000, use_live_odds: bool = True) -> Dict:
    """
    Convenient function for notebook use with LIVE TAB Australia odds
    
    Usage in notebook:
        from src.tab_profitability import analyze_notebook_predictions
        
        predictions = {
            'Ilia Topuria': 0.6953,
            'Charles Oliveira': 0.3047,
            # ... other predictions
        }
        
        # Use live odds (recommended)
        results = analyze_notebook_predictions(predictions, bankroll=1000, use_live_odds=True)
        
        # Or disable live scraping (for testing)
        results = analyze_notebook_predictions(predictions, bankroll=1000, use_live_odds=False)
    """
    
    analyzer = TABProfitabilityAnalyzer(bankroll=bankroll, use_live_odds=use_live_odds)
    return analyzer.analyze_predictions(model_predictions) 