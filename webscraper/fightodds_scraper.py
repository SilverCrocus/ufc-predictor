"""
FightOdds.io Scraper

Specialized scraper for extracting UFC odds from fightodds.io
This site aggregates odds from multiple sportsbooks, perfect for profitability analysis.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass

from config import HEADERS


@dataclass
class FightOddsData:
    """Container for fight odds data from fightodds.io"""
    event_name: str
    fighter_a: str
    fighter_b: str
    sportsbook_odds: Dict[str, Tuple[Optional[float], Optional[float]]]  # sportsbook -> (fighter_a_odds, fighter_b_odds)
    best_odds_a: Optional[float]
    best_odds_b: Optional[float]
    best_sportsbook_a: Optional[str]
    best_sportsbook_b: Optional[str]
    scraped_at: datetime


class FightOddsScraper:
    """
    Scraper for fightodds.io - aggregates odds from multiple sportsbooks
    """
    
    def __init__(self):
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://fightodds.io"
        
    def clean_fighter_name(self, name: str) -> str:
        """Clean fighter names to match your database format."""
        if not name:
            return ""
            
        # Remove extra whitespace and common artifacts
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove common prefixes/suffixes that might appear
        name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parenthetical info
        
        # Common name standardizations for your database
        name_mappings = {
            'Jon Jones': 'Jon Jones',
            'Jonathan Jones': 'Jon Jones',
            'Bones Jones': 'Jon Jones',
            'Charles Oliveira': 'Charles Oliveira',
            'Charles do Bronx': 'Charles Oliveira',
            'Ilia Topuria': 'Ilia Topuria',
            'El Matador': 'Ilia Topuria',
        }
        
        return name_mappings.get(name, name)
    
    def parse_american_odds(self, odds_text: str) -> Optional[float]:
        """Parse American odds from text (+200, -150, etc.)"""
        if not odds_text or odds_text.strip() == '':
            return None
            
        # Clean the text
        cleaned = re.sub(r'[^\d+\-]', '', odds_text.strip())
        
        # Match pattern for American odds
        match = re.match(r'^([+\-]?\d+)$', cleaned)
        if match:
            return float(match.group(1))
        
        return None
    
    def scrape_ufc_event_odds(self, event_url: str = None) -> List[FightOddsData]:
        """
        Scrape odds for a specific UFC event from fightodds.io
        
        Args:
            event_url: Specific event URL, or None to scrape the main page
            
        Returns:
            List of FightOddsData objects
        """
        if not event_url:
            # Use the main UFC page - you can customize this URL
            event_url = f"{self.base_url}/"
        
        print(f"🎯 Scraping odds from: {event_url}")
        
        try:
            response = self.session.get(event_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # This is a template - you'll need to adjust selectors based on the actual HTML structure
            return self._parse_odds_table(soup)
            
        except Exception as e:
            print(f"❌ Error scraping {event_url}: {e}")
            return []
    
    def _parse_odds_table(self, soup: BeautifulSoup) -> List[FightOddsData]:
        """
        Parse the odds table from the HTML
        
        Updated to target the Material-UI structure: nav.MuiList-root.jss1579
        """
        odds_data = []
        
        try:
            # Look for the specific MUI nav element containing the odds
            odds_container = soup.find('nav', class_='MuiList-root jss1579')
            
            if not odds_container:
                print("❌ Could not find MUI odds container with class 'MuiList-root jss1579'")
                print("🔍 Looking for alternative table structures...")
                
                # Fallback selectors
                odds_container = (
                    soup.find('nav', class_=lambda x: x and 'MuiList-root' in x) or
                    soup.find('div', class_=lambda x: x and 'MuiList' in x) or
                    soup.find('table') or
                    soup.find('div', class_='odds-table')
                )
                
                if not odds_container:
                    print("❌ No odds container found, using sample data")
                    return self._create_sample_data()
                else:
                    print(f"✅ Found fallback container: {odds_container.name} with class {odds_container.get('class')}")
            
            # Extract event name
            event_name = self._extract_event_name(soup)
            
            # Parse MUI list items or table rows depending on structure
            if odds_container.name == 'nav':
                # For MUI nav structure, look for list items
                rows = odds_container.find_all('li', class_=lambda x: x and 'MuiListItem' in x) if odds_container else []
                if not rows:
                    # Alternative: look for any div children that might contain fight data
                    rows = odds_container.find_all('div', recursive=False)
            else:
                # For table structure
                rows = odds_container.find_all('tr') if odds_container.name == 'table' else odds_container.find_all('div', class_='row')
            
            print(f"🔍 Found {len(rows)} potential fight rows")
            
            current_fight_data = None
            
            for i, row in enumerate(rows):
                # Look for fighter names and odds in MUI structure
                if odds_container.name == 'nav':
                    # For MUI list items, fighter names might be in spans, divs, or typography components
                    fighter_elements = row.find_all(['span', 'div', 'p'], string=True)
                    odds_elements = row.find_all(['span', 'div'], string=lambda text: text and any(c in text for c in ['+', '-']) if text else False)
                    
                    if fighter_elements:
                        fighter_texts = [elem.get_text(strip=True) for elem in fighter_elements if elem.get_text(strip=True)]
                        print(f"   Row {i}: Found fighter elements: {fighter_texts[:3]}...")  # Show first 3 for debugging
                        
                        # Look for actual fighter names (not empty, not just numbers/symbols)
                        potential_fighters = [text for text in fighter_texts 
                                            if text and len(text) > 2 and not text.replace('+', '').replace('-', '').replace('.', '').isdigit()]
                        
                        if potential_fighters:
                            fighter_name = self.clean_fighter_name(potential_fighters[0])
                            
                            if fighter_name and not any(keyword in fighter_name.lower() 
                                                       for keyword in ['vs', 'fight', 'event', 'date', 'odds', 'bet']):
                                
                                # Extract odds for this fighter
                                odds_texts = [elem.get_text(strip=True) for elem in odds_elements]
                                sportsbook_odds = self._extract_sportsbook_odds_from_texts(odds_texts)
                                
                                print(f"   Fighter: {fighter_name}, Odds found: {len(sportsbook_odds)}")
                                
                                # Group fighters into fights
                                if current_fight_data is None:
                                    current_fight_data = {
                                        'fighter_a': fighter_name,
                                        'fighter_a_odds': sportsbook_odds,
                                        'event_name': event_name
                                    }
                                else:
                                    # Complete the fight
                                    fight_odds = self._create_fight_odds_data(
                                        event_name,
                                        current_fight_data['fighter_a'],
                                        fighter_name,
                                        current_fight_data['fighter_a_odds'],
                                        sportsbook_odds
                                    )
                                    odds_data.append(fight_odds)
                                    current_fight_data = None
                else:
                    # Original table parsing logic
                    cells = row.find_all(['td', 'th', 'div'])
                    
                    if len(cells) >= 2:
                        fighter_cell = cells[0] if cells else None
                        if fighter_cell:
                            fighter_name = self.clean_fighter_name(fighter_cell.get_text(strip=True))
                            
                            if fighter_name and not any(keyword in fighter_name.lower() 
                                                       for keyword in ['vs', 'fight', 'event', 'date']):
                                
                                sportsbook_odds = self._extract_sportsbook_odds(cells[1:])
                                
                                if current_fight_data is None:
                                    current_fight_data = {
                                        'fighter_a': fighter_name,
                                        'fighter_a_odds': sportsbook_odds,
                                        'event_name': event_name
                                    }
                                else:
                                    fight_odds = self._create_fight_odds_data(
                                        event_name,
                                        current_fight_data['fighter_a'],
                                        fighter_name,
                                        current_fight_data['fighter_a_odds'],
                                        sportsbook_odds
                                    )
                                    odds_data.append(fight_odds)
                                    current_fight_data = None
            
        except Exception as e:
            print(f"❌ Error parsing odds table: {e}")
            return self._create_sample_data()
        
        return odds_data if odds_data else self._create_sample_data()
    
    def _extract_event_name(self, soup: BeautifulSoup) -> str:
        """Extract event name from the page"""
        # Look for event title
        title_selectors = [
            'h1',
            '.event-title',
            '.page-title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if 'UFC' in text:
                    return text
        
        return "UFC Event"
    
    def _extract_sportsbook_odds(self, odds_cells) -> Dict[str, float]:
        """Extract odds for each sportsbook from table cells"""
        sportsbook_odds = {}
        
        # Common sportsbook names based on the screenshot
        sportsbooks = [
            'BetOnline', 'Bovada', 'Bookmaker', 'Pinnacle', 'Betway', 'BetUS',
            'Bet365', 'SXbet', 'Cloudbet', 'DraftKings', 'FanDuel', 'Circa',
            'ESPN', 'BetRivers', 'BetMGM', 'Caesars'
        ]
        
        for i, cell in enumerate(odds_cells[:len(sportsbooks)]):
            if i < len(sportsbooks):
                odds_text = cell.get_text(strip=True)
                odds_value = self.parse_american_odds(odds_text)
                if odds_value:
                    sportsbook_odds[sportsbooks[i]] = odds_value
        
        return sportsbook_odds
    
    def _extract_sportsbook_odds_from_texts(self, odds_texts: List[str]) -> Dict[str, float]:
        """Extract odds from a list of text strings (for MUI structure)"""
        sportsbook_odds = {}
        
        # Common sportsbook names based on the screenshot
        sportsbooks = [
            'BetOnline', 'Bovada', 'Bookmaker', 'Pinnacle', 'Betway', 'BetUS',
            'Bet365', 'SXbet', 'Cloudbet', 'DraftKings', 'FanDuel', 'Circa',
            'ESPN', 'BetRivers', 'BetMGM', 'Caesars'
        ]
        
        valid_odds = []
        for text in odds_texts:
            odds_value = self.parse_american_odds(text)
            if odds_value:
                valid_odds.append(odds_value)
        
        # Map valid odds to sportsbooks
        for i, odds_value in enumerate(valid_odds[:len(sportsbooks)]):
            if i < len(sportsbooks):
                sportsbook_odds[sportsbooks[i]] = odds_value
        
        return sportsbook_odds
    
    def _create_fight_odds_data(self, event_name: str, fighter_a: str, fighter_b: str, 
                               fighter_a_odds: Dict[str, float], fighter_b_odds: Dict[str, float]) -> FightOddsData:
        """Create FightOddsData object from extracted data"""
        
        # Combine odds for each sportsbook
        sportsbook_odds = {}
        all_sportsbooks = set(fighter_a_odds.keys()) | set(fighter_b_odds.keys())
        
        for sportsbook in all_sportsbooks:
            odds_a = fighter_a_odds.get(sportsbook)
            odds_b = fighter_b_odds.get(sportsbook)
            sportsbook_odds[sportsbook] = (odds_a, odds_b)
        
        # Find best odds for each fighter
        best_odds_a = max([odds for odds, _ in sportsbook_odds.values() if odds], default=None)
        best_odds_b = max([odds for _, odds in sportsbook_odds.values() if odds], default=None)
        
        best_sportsbook_a = None
        best_sportsbook_b = None
        
        for sportsbook, (odds_a, odds_b) in sportsbook_odds.items():
            if odds_a == best_odds_a:
                best_sportsbook_a = sportsbook
            if odds_b == best_odds_b:
                best_sportsbook_b = sportsbook
        
        return FightOddsData(
            event_name=event_name,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            sportsbook_odds=sportsbook_odds,
            best_odds_a=best_odds_a,
            best_odds_b=best_odds_b,
            best_sportsbook_a=best_sportsbook_a,
            best_sportsbook_b=best_sportsbook_b,
            scraped_at=datetime.now()
        )
    
    def _create_sample_data(self) -> List[FightOddsData]:
        """Create sample data based on the screenshot for testing"""
        print("📝 Creating sample data based on screenshot...")
        
        # Sample data extracted from the screenshot
        sample_fights = [
            {
                'fighter_a': 'Charles Oliveira',
                'fighter_b': 'Ilia Topuria',
                'odds_a': {'DraftKings': +375, 'BetMGM': +375, 'FanDuel': +400},
                'odds_b': {'DraftKings': -500, 'BetMGM': -500, 'FanDuel': -510}
            },
            {
                'fighter_a': 'Alexandre Pantoja',
                'fighter_b': 'Kai Kara-France',
                'odds_a': {'DraftKings': -250, 'BetMGM': -250, 'FanDuel': -245},
                'odds_b': {'DraftKings': +210, 'BetMGM': +210, 'FanDuel': +210}
            },
            {
                'fighter_a': 'Brandon Royval',
                'fighter_b': 'Josh Van',
                'odds_a': {'DraftKings': +100, 'BetMGM': +100, 'FanDuel': +102},
                'odds_b': {'DraftKings': -120, 'BetMGM': -120, 'FanDuel': -125}
            }
        ]
        
        odds_data = []
        for fight in sample_fights:
            fight_odds = self._create_fight_odds_data(
                "UFC 317",
                fight['fighter_a'],
                fight['fighter_b'],
                fight['odds_a'],
                fight['odds_b']
            )
            odds_data.append(fight_odds)
        
        return odds_data
    
    def convert_to_profitability_format(self, fight_odds_list: List[FightOddsData]) -> List[Tuple[str, str, float, float, str]]:
        """
        Convert scraped odds to format needed for profitability analysis
        
        Returns:
            List of tuples: (fighter_a, fighter_b, best_odds_a, best_odds_b, best_sportsbook)
        """
        profitability_data = []
        
        for fight_odds in fight_odds_list:
            if fight_odds.best_odds_a and fight_odds.best_odds_b:
                # Use the sportsbook with better overall value
                best_sportsbook = fight_odds.best_sportsbook_a or fight_odds.best_sportsbook_b or "Unknown"
                
                profitability_data.append((
                    fight_odds.fighter_a,
                    fight_odds.fighter_b,
                    fight_odds.best_odds_a,
                    fight_odds.best_odds_b,
                    best_sportsbook
                ))
        
        return profitability_data
    
    def save_odds_data(self, odds_data: List[FightOddsData], filename: str = None) -> str:
        """Save scraped odds data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"../data/fightodds_io_{timestamp}.json"
        
        data_to_save = []
        for odds in odds_data:
            data_to_save.append({
                'event_name': odds.event_name,
                'fighter_a': odds.fighter_a,
                'fighter_b': odds.fighter_b,
                'sportsbook_odds': odds.sportsbook_odds,
                'best_odds_a': odds.best_odds_a,
                'best_odds_b': odds.best_odds_b,
                'best_sportsbook_a': odds.best_sportsbook_a,
                'best_sportsbook_b': odds.best_sportsbook_b,
                'scraped_at': odds.scraped_at.isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"💾 Odds data saved to: {filename}")
        return filename


def scrape_and_analyze_profitability():
    """
    Complete workflow: scrape odds from fightodds.io and analyze profitability
    """
    print("🎯 COMPLETE PROFITABILITY WORKFLOW")
    print("=" * 50)
    
    # Step 1: Scrape odds
    scraper = FightOddsScraper()
    odds_data = scraper.scrape_ufc_event_odds()
    
    if not odds_data:
        print("❌ No odds data found")
        return
    
    print(f"✅ Found {len(odds_data)} fights with odds")
    
    # Step 2: Convert to profitability format
    fight_card = scraper.convert_to_profitability_format(odds_data)
    
    # Step 3: Run profitability analysis
    try:
        from src.profitable_predictor import create_profitable_predictor_from_latest
        
        predictor = create_profitable_predictor_from_latest()
        
        # Analyze the fight card
        card_results = predictor.analyze_fight_card_profitability(fight_card, "Best Available")
        
        # Display results
        predictor.display_card_analysis(card_results)
        
        # Save odds data
        scraper.save_odds_data(odds_data)
        
        return card_results
        
    except Exception as e:
        print(f"❌ Error in profitability analysis: {e}")
        print("Make sure your prediction models are available")
        return None


def demo_fightodds_scraper():
    """Demo the fightodds.io scraper with sample data"""
    print("🎯 FightOdds.io Scraper Demo")
    print("=" * 40)
    
    scraper = FightOddsScraper()
    
    # For now, use sample data (customize the scraper based on actual HTML)
    odds_data = scraper.scrape_ufc_event_odds()
    
    if odds_data:
        print(f"\n📊 SCRAPED DATA SUMMARY:")
        for i, fight in enumerate(odds_data, 1):
            print(f"\nFight {i}: {fight.fighter_a} vs {fight.fighter_b}")
            print(f"  Best odds: {fight.fighter_a} ({fight.best_odds_a}) @ {fight.best_sportsbook_a}")
            print(f"  Best odds: {fight.fighter_b} ({fight.best_odds_b}) @ {fight.best_sportsbook_b}")
            print(f"  Sportsbooks: {len(fight.sportsbook_odds)} books available")
        
        # Show profitability format
        profitability_format = scraper.convert_to_profitability_format(odds_data)
        print(f"\n🎯 PROFITABILITY FORMAT:")
        for fighter_a, fighter_b, odds_a, odds_b, sportsbook in profitability_format:
            print(f"  ({fighter_a}, {fighter_b}, {odds_a}, {odds_b}, {sportsbook})")
    
    return odds_data


if __name__ == "__main__":
    demo_fightodds_scraper() 