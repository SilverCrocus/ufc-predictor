"""
UFC Betting Odds Scraper

This module extends the existing webscraper functionality to collect betting odds
for UFC fights, enabling automated profitability analysis.

Note: This is a basic implementation. For production use, consider:
- Using official odds APIs (The Odds API, etc.)
- Implementing rate limiting and error handling
- Adding multiple sportsbook support
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import re
from dataclasses import dataclass

from config import HEADERS  # Use existing headers from config


@dataclass
class FightOdds:
    """Container for fight odds data."""
    event_name: str
    fighter_a: str
    fighter_b: str
    fighter_a_odds: Optional[float]
    fighter_b_odds: Optional[float]
    sportsbook: str
    weight_class: Optional[str]
    fight_type: str  # "Main Event", "Co-Main", "Prelim", etc.
    scraped_at: datetime


class UFCOddsScraper:
    """
    Scraper for UFC betting odds from various sources.
    
    This builds on the existing scraping infrastructure to add odds collection
    for profitability analysis.
    """
    
    def __init__(self):
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def clean_fighter_name(self, name: str) -> str:
        """Clean and normalize fighter names for matching with database."""
        # Remove common prefixes/suffixes
        name = re.sub(r'\s+\([^)]*\)', '', name)  # Remove parenthetical info
        name = re.sub(r'\s+vs\.?\s+.*', '', name)  # Remove "vs Fighter" portions
        name = name.strip()
        
        # Common name normalizations
        name_replacements = {
            'Jon "Bones" Jones': 'Jon Jones',
            'Daniel "DC" Cormier': 'Daniel Cormier',
            'José Aldo': 'Jose Aldo',
            'Conor McGregor': 'Conor McGregor',
        }
        
        return name_replacements.get(name, name)
    
    def parse_american_odds(self, odds_text: str) -> Optional[float]:
        """Parse American odds from text (e.g., '-150', '+200')."""
        if not odds_text:
            return None
            
        # Clean the text
        odds_text = odds_text.strip().replace(',', '')
        
        # Extract numeric part
        match = re.search(r'([+-]?\d+)', odds_text)
        if match:
            return float(match.group(1))
        
        return None
    
    def scrape_draftkings_odds(self, event_url: str = None) -> List[FightOdds]:
        """
        Scrape odds from DraftKings (example implementation).
        
        Note: This is a simplified example. Real implementation would need:
        - Proper URL handling
        - Dynamic content loading (Selenium)
        - Rate limiting
        - Error handling
        """
        odds_list = []
        
        # This is a placeholder - in reality you'd scrape actual sportsbook sites
        # or use APIs like The Odds API
        
        # Example mock data for demonstration
        mock_odds = [
            {
                'event': 'UFC 309',
                'fighter_a': 'Jon Jones',
                'fighter_b': 'Stipe Miocic',
                'odds_a': -300,
                'odds_b': +250,
                'weight_class': 'Heavyweight',
                'fight_type': 'Main Event'
            },
            {
                'event': 'UFC 309',
                'fighter_a': 'Charles Oliveira',
                'fighter_b': 'Michael Chandler',
                'odds_a': -150,
                'odds_b': +130,
                'weight_class': 'Lightweight',
                'fight_type': 'Co-Main Event'
            }
        ]
        
        for fight_data in mock_odds:
            odds_list.append(FightOdds(
                event_name=fight_data['event'],
                fighter_a=self.clean_fighter_name(fight_data['fighter_a']),
                fighter_b=self.clean_fighter_name(fight_data['fighter_b']),
                fighter_a_odds=fight_data['odds_a'],
                fighter_b_odds=fight_data['odds_b'],
                sportsbook='DraftKings',
                weight_class=fight_data.get('weight_class'),
                fight_type=fight_data.get('fight_type', 'Regular'),
                scraped_at=datetime.now()
            ))
        
        return odds_list
    
    def scrape_odds_api(self, api_key: str, sport: str = 'mma_mixed_martial_arts') -> List[FightOdds]:
        """
        Scrape odds using The Odds API (recommended approach).
        
        Args:
            api_key: Your Odds API key
            sport: Sport key for MMA
            
        Returns:
            List of FightOdds objects
        """
        odds_list = []
        
        try:
            # The Odds API endpoint
            url = 'https://api.the-odds-api.com/v4/sports/{}/odds/'.format(sport)
            params = {
                'api_key': api_key,
                'regions': 'us',
                'markets': 'h2h',  # Head-to-head (moneyline)
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for game in data:
                if 'UFC' in game.get('sport_title', ''):
                    home_team = game.get('home_team', '')
                    away_team = game.get('away_team', '')
                    
                    # Extract odds from bookmakers
                    for bookmaker in game.get('bookmakers', []):
                        sportsbook = bookmaker.get('title', 'Unknown')
                        
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'h2h':
                                outcomes = market.get('outcomes', [])
                                
                                if len(outcomes) >= 2:
                                    fighter_a_odds = None
                                    fighter_b_odds = None
                                    
                                    for outcome in outcomes:
                                        if outcome.get('name') == home_team:
                                            fighter_a_odds = outcome.get('price')
                                        elif outcome.get('name') == away_team:
                                            fighter_b_odds = outcome.get('price')
                                    
                                    if fighter_a_odds and fighter_b_odds:
                                        odds_list.append(FightOdds(
                                            event_name=game.get('sport_title', 'UFC Event'),
                                            fighter_a=self.clean_fighter_name(home_team),
                                            fighter_b=self.clean_fighter_name(away_team),
                                            fighter_a_odds=fighter_a_odds,
                                            fighter_b_odds=fighter_b_odds,
                                            sportsbook=sportsbook,
                                            weight_class=None,
                                            fight_type='Regular',
                                            scraped_at=datetime.now()
                                        ))
            
        except Exception as e:
            print(f"Error scraping odds from API: {e}")
        
        return odds_list
    
    def scrape_multiple_sportsbooks(self, api_key: str = None) -> List[FightOdds]:
        """
        Scrape odds from multiple sources and return the best available.
        
        Args:
            api_key: Optional API key for odds services
            
        Returns:
            Combined list of odds from all sources
        """
        all_odds = []
        
        print("🎯 Scraping UFC betting odds...")
        
        # Method 1: Use Odds API if available
        if api_key:
            print("  📡 Fetching from Odds API...")
            try:
                api_odds = self.scrape_odds_api(api_key)
                all_odds.extend(api_odds)
                print(f"     Found {len(api_odds)} fights from API")
            except Exception as e:
                print(f"     API error: {e}")
        
        # Method 2: Scrape DraftKings (mock for demo)
        print("  🎰 Fetching from DraftKings...")
        try:
            dk_odds = self.scrape_draftkings_odds()
            all_odds.extend(dk_odds)
            print(f"     Found {len(dk_odds)} fights from DraftKings")
        except Exception as e:
            print(f"     DraftKings error: {e}")
        
        # Remove duplicates and find best odds
        all_odds = self.consolidate_odds(all_odds)
        
        print(f"✅ Total unique fights found: {len(all_odds)}")
        return all_odds
    
    def consolidate_odds(self, odds_list: List[FightOdds]) -> List[FightOdds]:
        """
        Consolidate odds from multiple sportsbooks, keeping the best odds for each fight.
        
        Args:
            odds_list: List of FightOdds from various sources
            
        Returns:
            Consolidated list with best odds for each unique fight
        """
        fight_odds_map = {}
        
        for odds in odds_list:
            fight_key = f"{odds.fighter_a}_vs_{odds.fighter_b}"
            
            if fight_key not in fight_odds_map:
                fight_odds_map[fight_key] = odds
            else:
                # Keep the odds that provide better value
                existing = fight_odds_map[fight_key]
                
                # Simple heuristic: keep odds with higher absolute values (better for bettors)
                existing_avg = (abs(existing.fighter_a_odds or 0) + abs(existing.fighter_b_odds or 0)) / 2
                new_avg = (abs(odds.fighter_a_odds or 0) + abs(odds.fighter_b_odds or 0)) / 2
                
                if new_avg > existing_avg:
                    fight_odds_map[fight_key] = odds
        
        return list(fight_odds_map.values())
    
    def save_odds_to_file(self, odds_list: List[FightOdds], filename: str = None):
        """Save scraped odds to JSON file for later use."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"../data/ufc_odds_{timestamp}.json"
        
        odds_data = []
        for odds in odds_list:
            odds_data.append({
                'event_name': odds.event_name,
                'fighter_a': odds.fighter_a,
                'fighter_b': odds.fighter_b,
                'fighter_a_odds': odds.fighter_a_odds,
                'fighter_b_odds': odds.fighter_b_odds,
                'sportsbook': odds.sportsbook,
                'weight_class': odds.weight_class,
                'fight_type': odds.fight_type,
                'scraped_at': odds.scraped_at.isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(odds_data, f, indent=2)
        
        print(f"💾 Odds saved to: {filename}")
        return filename
    
    def load_odds_from_file(self, filename: str) -> List[FightOdds]:
        """Load previously scraped odds from JSON file."""
        odds_list = []
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for item in data:
                odds_list.append(FightOdds(
                    event_name=item['event_name'],
                    fighter_a=item['fighter_a'],
                    fighter_b=item['fighter_b'],
                    fighter_a_odds=item['fighter_a_odds'],
                    fighter_b_odds=item['fighter_b_odds'],
                    sportsbook=item['sportsbook'],
                    weight_class=item.get('weight_class'),
                    fight_type=item.get('fight_type', 'Regular'),
                    scraped_at=datetime.fromisoformat(item['scraped_at'])
                ))
                
        except Exception as e:
            print(f"Error loading odds from file: {e}")
        
        return odds_list


def get_upcoming_ufc_odds(api_key: str = None) -> List[FightOdds]:
    """
    Convenience function to get upcoming UFC odds.
    
    Args:
        api_key: Optional Odds API key for real data
        
    Returns:
        List of upcoming fight odds
    """
    scraper = UFCOddsScraper()
    return scraper.scrape_multiple_sportsbooks(api_key)


def demo_odds_scraping():
    """Demonstrate the odds scraping functionality."""
    print("🎯 UFC Odds Scraping Demo")
    print("=" * 40)
    
    # Create scraper
    scraper = UFCOddsScraper()
    
    # Get odds (using mock data for demo)
    odds_list = scraper.scrape_multiple_sportsbooks()
    
    if odds_list:
        print(f"\n📊 SCRAPED ODDS SUMMARY:")
        print(f"Total fights found: {len(odds_list)}")
        
        for i, odds in enumerate(odds_list, 1):
            print(f"\nFight {i}: {odds.fighter_a} vs {odds.fighter_b}")
            print(f"  Odds: {odds.fighter_a} ({odds.fighter_a_odds}), {odds.fighter_b} ({odds.fighter_b_odds})")
            print(f"  Sportsbook: {odds.sportsbook}")
            print(f"  Event: {odds.event_name}")
        
        # Save to file
        filename = scraper.save_odds_to_file(odds_list)
        
        # Test loading from file
        loaded_odds = scraper.load_odds_from_file(filename)
        print(f"\n✅ Successfully saved and loaded {len(loaded_odds)} odds")
        
    else:
        print("❌ No odds found")
    
    return odds_list


if __name__ == "__main__":
    demo_odds_scraping() 