"""
TAB Australia Profitability Module

Import this into your UFC predictions notebook to analyze profitability
based on actual LIVE TAB Australia odds scraped in real-time.

Usage:
    from ufc_predictor.betting.tab_profitability import TABProfitabilityAnalyzer
    
    analyzer = TABProfitabilityAnalyzer(bankroll=1000)
    results = analyzer.analyze_predictions(model_predictions)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from ufc_predictor.betting.profitability import ProfitabilityOptimizer

# Import the TAB Australia scraper
sys.path.append(str(Path(__file__).parent.parent))
from ufc_predictor.scrapers.tab_australia_scraper import TABAustraliaUFCScraper

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
        """Match prediction fighter names to scraped odds using proper TAB data structure"""
        
        matched_odds = {}
        
        print("🔍 MATCHING FIGHTERS TO ODDS:")
        print("-" * 40)
        
        # Parse TAB odds data structure
        # TAB has multiple formats:
        # 1. "Oliveira v Topuria": 4.25 (fight market - left fighter's odds)
        # 2. "H2H LASTNAME Firstname": 4.25 (head-to-head market - specific fighter's odds)
        # 3. "Markets": X.XX (non-fighter data - skip)
        
        # Create mapping of fight markets to individual fighters
        fight_markets = {}  # fight_key -> full fight data
        h2h_markets = {}    # cleaned_name -> odds
        fight_pairs = {}    # track fighter pairs for missing H2H data
        
        print("📊 Parsing TAB odds structure...")
        
        for market_name, odds in raw_odds.items():
            # Skip non-fighter entries
            if market_name.lower() in ['markets', 'market']:
                print(f"⏭️  Skipping: {market_name}")
                continue
            
            if " v " in market_name:
                # Fight market format: "Oliveira v Topuria"
                fighters = market_name.split(" v ")
                if len(fighters) == 2:
                    fighter_a = fighters[0].strip()
                    fighter_b = fighters[1].strip()
                    
                    # Store this as a fight market (left fighter gets these odds)
                    fight_key = f"{fighter_a} vs {fighter_b}"
                    fight_markets[fight_key] = {
                        'fighter_a': fighter_a,
                        'fighter_b': fighter_b,
                        'fighter_a_odds': odds,
                        'market_name': market_name
                    }
                    
                    # Track the pair for later H2H matching
                    fight_pairs[fighter_a.lower()] = fighter_b
                    fight_pairs[fighter_b.lower()] = fighter_a
                    
                    print(f"🥊 Fight market: {market_name} → {fighter_a} gets odds {odds}")
                    
            elif market_name.startswith("H2H "):
                # Head-to-head format: "H2H LASTNAME Firstname"
                h2h_part = market_name.replace("H2H ", "").strip()
                
                # Parse "LASTNAME Firstname" format
                parts = h2h_part.split()
                if len(parts) >= 2:
                    # TAB format: "H2H LASTNAME Firstname"
                    last_name = parts[0].title()  # OLIVEIRA -> Oliveira
                    first_name = parts[1].title()  # charles -> Charles
                    
                    # Create possible name variations
                    full_name = f"{first_name} {last_name}"
                    h2h_markets[full_name] = odds
                    h2h_markets[last_name] = odds  # Also by last name
                    
                    print(f"👤 H2H market: {market_name} → {full_name} gets odds {odds}")
        
        print(f"\n📈 Parsed {len(fight_markets)} fight markets and {len(h2h_markets)} H2H entries")
        
        # CRITICAL FIX: Create complete fight data by calculating missing odds
        print("\n⚡ CALCULATING MISSING FIGHTER ODDS:")
        print("-" * 40)
        
        complete_fight_data = {}
        for fight_key, fight_data in fight_markets.items():
            enhanced_data = fight_data.copy()
            
            # Try to find right fighter's odds in H2H markets first
            right_fighter = fight_data['fighter_b']
            right_fighter_odds = None
            
            # Look for H2H odds for right fighter
            for h2h_name, h2h_odds in h2h_markets.items():
                if (self.similarity(right_fighter, h2h_name) > 0.8 or 
                    any(part.lower() in right_fighter.lower() for part in h2h_name.split())):
                    right_fighter_odds = h2h_odds
                    print(f"🔗 Found H2H for {right_fighter}: {h2h_odds}")
                    break
            
            if right_fighter_odds:
                enhanced_data['fighter_b_odds'] = right_fighter_odds
            else:
                # CALCULATE implied odds for right fighter
                # In a two-outcome fight: P(A) + P(B) = 1
                # If left fighter odds = X, then P(A) = 1/X
                # P(B) = 1 - P(A) = 1 - 1/X
                # Right fighter odds = 1/P(B) = 1/(1 - 1/X)
                left_odds = fight_data['fighter_a_odds']
                if left_odds > 1.0:
                    left_prob = 1.0 / left_odds
                    right_prob = 1.0 - left_prob
                    if right_prob > 0:
                        implied_right_odds = 1.0 / right_prob
                        enhanced_data['fighter_b_odds'] = round(implied_right_odds, 2)
                        print(f"🧮 Calculated {right_fighter} odds: {implied_right_odds:.2f} (from {fight_data['fighter_a']} odds {left_odds})")
                    else:
                        print(f"❌ Cannot calculate odds for {right_fighter} - invalid probability")
                else:
                    print(f"❌ Invalid left fighter odds for {fight_data['fighter_a']}: {left_odds}")
            
            complete_fight_data[fight_key] = enhanced_data
        
        # Now match our model predictions to available odds
        print("\n🎯 MATCHING MODEL PREDICTIONS:")
        print("-" * 40)
        
        for fighter_name in model_predictions.keys():
            matched = False
            
            # Strategy 1: Direct match in H2H markets (most reliable)
            best_h2h_match = None
            best_h2h_score = 0
            
            for h2h_name, odds in h2h_markets.items():
                # Try exact similarity
                similarity = self.similarity(fighter_name, h2h_name)
                
                # Try individual name parts
                fighter_parts = fighter_name.lower().split()
                h2h_parts = h2h_name.lower().split()
                
                part_matches = 0
                for f_part in fighter_parts:
                    for h_part in h2h_parts:
                        if len(f_part) > 2 and len(h_part) > 2:  # Avoid short words
                            if f_part in h_part or h_part in f_part or self.similarity(f_part, h_part) > 0.8:
                                part_matches += 1
                
                # Boost score if we have good part matches
                if part_matches > 0:
                    similarity = max(similarity, part_matches / max(len(fighter_parts), len(h2h_parts)))
                
                if similarity > best_h2h_score:
                    best_h2h_score = similarity
                    best_h2h_match = (h2h_name, odds, similarity)
            
            if best_h2h_match and best_h2h_score > 0.7:
                h2h_name, odds, similarity = best_h2h_match
                matched_odds[fighter_name] = odds
                matched = True
                print(f"✅ {fighter_name} → H2H {h2h_name} (similarity: {similarity:.2f}, odds: {odds})")
            
            if matched:
                continue
            
            # Strategy 2: Match to complete fight data
            best_fight_match = None
            best_fight_score = 0
            
            for fight_key, fight_data in complete_fight_data.items():
                # Check match with left fighter
                left_sim = self.similarity(fighter_name, fight_data['fighter_a'])
                # Also try last name matching
                fighter_last = fighter_name.split()[-1] if fighter_name.split() else fighter_name
                left_last = fight_data['fighter_a'].split()[-1] if fight_data['fighter_a'].split() else fight_data['fighter_a']
                left_sim = max(left_sim, self.similarity(fighter_last, left_last))
                
                # Check match with right fighter  
                right_sim = self.similarity(fighter_name, fight_data['fighter_b'])
                right_last = fight_data['fighter_b'].split()[-1] if fight_data['fighter_b'].split() else fight_data['fighter_b']
                right_sim = max(right_sim, self.similarity(fighter_last, right_last))
                
                if left_sim > 0.6 and left_sim > best_fight_score:
                    best_fight_score = left_sim
                    best_fight_match = (fight_data, 'left', left_sim)
                elif right_sim > 0.6 and right_sim > best_fight_score:
                    best_fight_score = right_sim  
                    best_fight_match = (fight_data, 'right', right_sim)
            
            if best_fight_match:
                fight_data, position, similarity = best_fight_match
                
                if position == 'left':
                    matched_odds[fighter_name] = fight_data['fighter_a_odds']
                    print(f"✅ {fighter_name} → {fight_data['market_name']} (LEFT, sim: {similarity:.2f}, odds: {fight_data['fighter_a_odds']})")
                else:
                    if 'fighter_b_odds' in fight_data:
                        matched_odds[fighter_name] = fight_data['fighter_b_odds']
                        odds_type = "calculated" if fight_data['fighter_b_odds'] != fight_data.get('original_b_odds') else "actual"
                        print(f"✅ {fighter_name} → {fight_data['market_name']} (RIGHT, sim: {similarity:.2f}, odds: {fight_data['fighter_b_odds']} {odds_type})")
                    else:
                        print(f"❌ {fighter_name} → {fight_data['market_name']} (RIGHT but no odds available)")
                
                matched = True
            
            if not matched:
                print(f"❌ {fighter_name} → No suitable match found")
        
        print(f"\n📊 Successfully matched {len(matched_odds)}/{len(model_predictions)} fighters")
        
        # Show final summary
        if matched_odds:
            print(f"\n✅ FINAL MATCHED ODDS:")
            for fighter, odds in matched_odds.items():
                print(f"   • {fighter}: {odds}")
        
        # If we still have missing fighters, run debug mode
        if len(matched_odds) < len(model_predictions):
            unmatched = [f for f in model_predictions.keys() if f not in matched_odds]
            print(f"\n❌ UNMATCHED FIGHTERS ({len(unmatched)}):")
            for fighter in unmatched:
                print(f"   • {fighter}")
        
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
        
        # Get odds - either from live scraping or fixed data file
        if self.use_live_odds:
            raw_odds = self.scrape_live_tab_odds()
            odds_source = 'live'
        else:
            # Use fixed data file when live scraping is disabled
            try:
                with open('raw_tab_odds.json', 'r') as f:
                    raw_odds = json.load(f)
                print(f"📄 Loaded {len(raw_odds)} odds from raw_tab_odds.json")
                odds_source = 'fixed_data'
            except FileNotFoundError:
                print("❌ raw_tab_odds.json not found - cannot perform analysis")
                return {
                    'total_opportunities': 0,
                    'total_expected_profit': 0,
                    'bankroll': self.bankroll,
                    'opportunities': [],
                    'odds_source': 'none',
                    'error': 'raw_tab_odds.json file not found'
                }
            except Exception as e:
                print(f"❌ Error loading raw_tab_odds.json: {e}")
                return {
                    'total_opportunities': 0,
                    'total_expected_profit': 0,
                    'bankroll': self.bankroll,
                    'opportunities': [],
                    'odds_source': 'none',
                    'error': f'Error loading odds data: {e}'
                }
        
        if not raw_odds:
            print("❌ No odds available - cannot perform analysis")
            return {
                'total_opportunities': 0,
                'total_expected_profit': 0,
                'bankroll': self.bankroll,
                'opportunities': [],
                'odds_source': 'none',
                'error': 'No odds data available'
            }
        
        # Match fighter names to odds
        matched_odds = self.match_fighters_to_odds(model_predictions, raw_odds)
        
        opportunities = []
        total_expected_profit = 0
        
        print("\n🇦🇺 TAB AUSTRALIA PROFITABILITY ANALYSIS")
        print("=" * 50)
        
        if odds_source == 'live':
            print("📡 Using LIVE TAB Australia odds")
        else:
            print("📄 Using FIXED TAB Australia odds (raw_tab_odds.json)")
        
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
            'odds_source': odds_source,
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
        """Debug missing fighter matches by showing available TAB odds"""
        
        missing_fighters = []
        for fighter in model_predictions.keys():
            # Find which fighters didn't get matched (not in our final matched_odds)
            # This is called from match_fighters_to_odds so we need to check differently
            missing_fighters.append(fighter)
        
        print("\n🔍 DEBUG: MISSING FIGHTER ANALYSIS")
        print("=" * 50)
        
        print(f"📋 Model Predictions ({len(model_predictions)} fighters):")
        for fighter, prob in model_predictions.items():
            print(f"   • {fighter}: {prob:.1%}")
        
        print(f"\n📊 Available TAB Odds ({len(raw_odds)} markets):")
        
        # Group by type for better visualization
        fight_markets = []
        h2h_markets = []
        other_markets = []
        
        for market_name, odds in raw_odds.items():
            if market_name.lower() in ['markets', 'market']:
                other_markets.append(f"   • {market_name}: {odds} (SKIPPED - not fighter data)")
            elif " v " in market_name:
                fight_markets.append(f"   • {market_name}: {odds} (FIGHT - left fighter odds)")
            elif market_name.startswith("H2H "):
                h2h_name = market_name.replace("H2H ", "")
                parts = h2h_name.split()
                if len(parts) >= 2:
                    cleaned = f"{parts[1].title()} {parts[0].title()}"
                    h2h_markets.append(f"   • {market_name}: {odds} → {cleaned}")
                else:
                    h2h_markets.append(f"   • {market_name}: {odds} (MALFORMED)")
            else:
                other_markets.append(f"   • {market_name}: {odds} (UNKNOWN FORMAT)")
        
        if fight_markets:
            print(f"\n🥊 Fight Markets ({len(fight_markets)}):")
            for market in fight_markets:
                print(market)
        
        if h2h_markets:
            print(f"\n👤 Head-to-Head Markets ({len(h2h_markets)}):")
            for market in h2h_markets:
                print(market)
        
        if other_markets:
            print(f"\n❓ Other Markets ({len(other_markets)}):")
            for market in other_markets:
                print(market)
        
        print("\n💡 MATCHING SUGGESTIONS:")
        print("-" * 30)
        
        for fighter in model_predictions.keys():
            print(f"\n🎯 For '{fighter}':")
            
            # Show top similarity matches
            matches = []
            for market_name in raw_odds.keys():
                if market_name.lower() in ['markets', 'market']:
                    continue
                
                # For H2H markets, clean the name first
                target_name = market_name
                if market_name.startswith("H2H "):
                    h2h_part = market_name.replace("H2H ", "")
                    parts = h2h_part.split()
                    if len(parts) >= 2:
                        target_name = f"{parts[1].title()} {parts[0].title()}"
                
                similarity = self.similarity(fighter, target_name)
                if similarity > 0.3:  # Show any reasonable match
                    matches.append((market_name, target_name, similarity, raw_odds[market_name]))
            
            # Sort by similarity
            matches.sort(key=lambda x: x[2], reverse=True)
            
            if matches:
                print("   Top matches:")
                for market_name, target_name, sim, odds in matches[:5]:
                    match_type = "H2H" if market_name.startswith("H2H") else "FIGHT" if " v " in market_name else "OTHER"
                    print(f"      {sim:.2f}: {market_name} → {target_name} (odds: {odds}) [{match_type}]")
                else:
                    print("      No reasonable matches found")
        
        print(f"\n💾 Full raw odds saved to 'raw_tab_odds.json' for manual inspection")

def analyze_notebook_predictions(model_predictions: Dict[str, float], bankroll: float = 1000, use_live_odds: bool = True) -> Dict:
    """
    Convenient function for notebook use with LIVE TAB Australia odds
    
    Usage in notebook:
        from ufc_predictor.betting.tab_profitability import analyze_notebook_predictions
        
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