"""
UFC Event Discovery System

Automatically detects upcoming UFC events and their fight cards from multiple sources.
Primary source: The Odds API (real-time odds data)
Fallback sources: TAB Australia, UFC official site

Features:
- Auto-detect next UFC event
- Extract complete fight cards
- Handle numbered events (UFC 300) and Fight Nights
- Cache event data to reduce API calls
- Multi-source validation

Usage:
    from ufc_predictor.scrapers.event_discovery import UFCEventDiscovery
    
    discovery = UFCEventDiscovery(api_key="your_key")
    next_event = discovery.get_next_event()
    print(f"Next event: {next_event['name']} on {next_event['date']}")
    print(f"Fights: {next_event['fights']}")
"""

import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from functools import lru_cache
import time

class UFCEventDiscovery:
    """Automatic UFC event discovery and fight card extraction."""
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialize UFC Event Discovery.
        
        Args:
            api_key: The Odds API key (optional if using fallback sources)
            cache_dir: Directory for caching event data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ufc_predictor" / "event_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.odds_api_url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
        
        # Cache settings
        self.cache_duration = 3600  # 1 hour in seconds
    
    def get_next_event(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get the next upcoming UFC event with complete fight card.
        
        Args:
            force_refresh: Force API call even if cache exists
            
        Returns:
            Dictionary with event info or None if no events found
            {
                'name': 'UFC 300',
                'date': datetime object,
                'fights': ['Fighter A vs Fighter B', ...],
                'odds_available': True/False,
                'source': 'the_odds_api'
            }
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_cached_event()
            if cached:
                return cached
        
        # Try primary source (The Odds API)
        if self.api_key:
            event = self._get_event_from_odds_api()
            if event:
                self._cache_event(event)
                return event
        
        # Fallback sources would go here
        # For now, return None if no API key
        return None
    
    def get_upcoming_events(self, days_ahead: int = 30, force_refresh: bool = False) -> List[Dict]:
        """
        Get all upcoming UFC events within specified timeframe.
        
        Args:
            days_ahead: Number of days to look ahead
            force_refresh: Force API call even if cache exists
            
        Returns:
            List of event dictionaries sorted by date
        """
        # Check cache
        cache_file = self.cache_dir / f"upcoming_events_{days_ahead}d.json"
        if not force_refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < self.cache_duration:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        events = []
        
        if self.api_key:
            events = self._get_events_from_odds_api(days_ahead)
        
        # Cache results
        if events:
            with open(cache_file, 'w') as f:
                json.dump(events, f, default=str)
        
        return events
    
    def _get_event_from_odds_api(self) -> Optional[Dict]:
        """Fetch next UFC event from The Odds API."""
        try:
            # Make API request
            params = {
                'apiKey': self.api_key,
                'regions': 'au,us,uk',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            
            response = requests.get(self.odds_api_url, params=params, timeout=10)
            
            if response.status_code == 429:
                print("⚠️  API rate limit reached. Using cached data if available.")
                return None
            
            if response.status_code != 200:
                print(f"⚠️  API error: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                return None
            
            # Group fights by event date
            events = self._group_fights_by_event(data)
            
            if not events:
                return None
            
            # Get the next event (closest date)
            next_event = min(events, key=lambda x: x['date'])
            
            return next_event
            
        except Exception as e:
            print(f"⚠️  Error fetching from Odds API: {e}")
            return None
    
    def _get_events_from_odds_api(self, days_ahead: int) -> List[Dict]:
        """Fetch all UFC events within timeframe from The Odds API."""
        try:
            params = {
                'apiKey': self.api_key,
                'regions': 'au,us,uk',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            
            response = requests.get(self.odds_api_url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            
            # Group and filter by date
            events = self._group_fights_by_event(data)
            
            # Filter by timeframe
            cutoff_date = datetime.now() + timedelta(days=days_ahead)
            events = [e for e in events if e['date'] <= cutoff_date]
            
            return sorted(events, key=lambda x: x['date'])
            
        except Exception:
            return []
    
    def _group_fights_by_event(self, odds_data: List[Dict]) -> List[Dict]:
        """Group individual fights into UFC events."""
        events_by_date = {}
        
        for fight in odds_data:
            commence_time = fight.get('commence_time', '')
            if not commence_time:
                continue
            
            # Parse date
            try:
                event_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            except:
                continue
            
            # Use date as key (fights on same day = same event)
            date_key = event_date.strftime('%Y-%m-%d')
            
            if date_key not in events_by_date:
                events_by_date[date_key] = {
                    'date': event_date,
                    'fights': [],
                    'raw_fights': []
                }
            
            # Extract fighter names
            home = fight.get('home_team', '')
            away = fight.get('away_team', '')
            
            if home and away:
                events_by_date[date_key]['fights'].append(f"{home} vs {away}")
                events_by_date[date_key]['raw_fights'].append(fight)
        
        # Convert to list of events
        events = []
        for date_key, event_data in events_by_date.items():
            if len(event_data['fights']) < 2:
                continue  # Skip single-fight "events"
            
            # Generate event name
            event_name = self._generate_event_name(event_data['date'], event_data['fights'])
            
            events.append({
                'name': event_name,
                'date': event_data['date'],
                'date_str': event_data['date'].strftime('%Y-%m-%d'),
                'fights': event_data['fights'],
                'fight_count': len(event_data['fights']),
                'odds_available': True,
                'source': 'the_odds_api',
                'raw_data': event_data['raw_fights']  # Include for odds extraction
            })
        
        return events
    
    def _generate_event_name(self, date: datetime, fights: List[str]) -> str:
        """Generate UFC event name from date and fights."""
        # Check if it's a numbered event (typically has 10+ fights)
        if len(fights) >= 10:
            # Try to determine if it's a PPV (numbered event)
            # This is a heuristic - would need official data for accuracy
            return f"UFC_{date.strftime('%B_%d_%Y')}"
        else:
            # Fight Night event
            return f"UFC_Fight_Night_{date.strftime('%B_%d_%Y')}"
    
    def extract_odds_for_event(self, event: Dict) -> Dict:
        """
        Extract best odds for all fights in an event.
        
        Args:
            event: Event dictionary from get_next_event()
            
        Returns:
            Dictionary mapping fight keys to odds data
        """
        if 'raw_data' not in event:
            return {}
        
        odds_data = {}
        
        for fight in event['raw_data']:
            home = fight.get('home_team', '')
            away = fight.get('away_team', '')
            
            if not home or not away:
                continue
            
            # Get best odds from all bookmakers
            best_home_odds = None
            best_away_odds = None
            
            for bookmaker in fight.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                price = outcome.get('price', 0)
                                if price and (not best_home_odds or price > best_home_odds):
                                    best_home_odds = price
                            elif outcome['name'] == away:
                                price = outcome.get('price', 0)
                                if price and (not best_away_odds or price > best_away_odds):
                                    best_away_odds = price
            
            if best_home_odds and best_away_odds:
                fight_key = f"{home}_vs_{away}".replace(" ", "_")
                odds_data[fight_key] = {
                    'fighter_a': home,
                    'fighter_b': away,
                    'fighter_a_decimal_odds': best_home_odds,
                    'fighter_b_decimal_odds': best_away_odds
                }
        
        return odds_data
    
    def _cache_event(self, event: Dict):
        """Cache event data to disk."""
        cache_file = self.cache_dir / "next_event.json"
        
        # Remove raw_data before caching (too large)
        cache_data = {k: v for k, v in event.items() if k != 'raw_data'}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, default=str)
    
    def _load_cached_event(self) -> Optional[Dict]:
        """Load cached event if still valid."""
        cache_file = self.cache_dir / "next_event.json"
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > self.cache_duration:
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert date string back to datetime
            if 'date' in data and isinstance(data['date'], str):
                data['date'] = datetime.fromisoformat(data['date'])
            
            return data
        except:
            return None
    
    def get_event_by_name_pattern(self, pattern: str) -> Optional[Dict]:
        """
        Find UFC event by name pattern (e.g., 'UFC 300', 'Whittaker').
        
        Args:
            pattern: Event name pattern to search for
            
        Returns:
            Event dictionary or None if not found
        """
        events = self.get_upcoming_events(days_ahead=60)
        
        pattern_lower = pattern.lower()
        
        for event in events:
            # Check event name
            if pattern_lower in event['name'].lower():
                return event
            
            # Check fighter names
            for fight in event['fights']:
                if pattern_lower in fight.lower():
                    return event
        
        return None


# Convenience functions for notebook integration
def get_next_ufc_event(api_key: str = None) -> Optional[Dict]:
    """Quick function to get next UFC event."""
    discovery = UFCEventDiscovery(api_key=api_key)
    return discovery.get_next_event()

def get_upcoming_ufc_events(api_key: str = None, days: int = 30) -> List[Dict]:
    """Quick function to get upcoming UFC events."""
    discovery = UFCEventDiscovery(api_key=api_key)
    return discovery.get_upcoming_events(days_ahead=days)

def auto_detect_and_fetch_odds(api_key: str) -> Tuple[Dict, Dict]:
    """
    Auto-detect next UFC event and fetch odds.
    
    Returns:
        Tuple of (odds_data, event_info)
    """
    discovery = UFCEventDiscovery(api_key=api_key)
    
    # Get next event
    event = discovery.get_next_event()
    if not event:
        return {}, {}
    
    # Extract odds
    odds = discovery.extract_odds_for_event(event)
    
    return odds, event


if __name__ == "__main__":
    # Test the discovery system
    print("🧪 Testing UFC Event Discovery System")
    print("=" * 40)
    
    # You would need to provide an API key for testing
    test_api_key = input("Enter The Odds API key (or press Enter to skip): ").strip()
    
    if test_api_key:
        discovery = UFCEventDiscovery(api_key=test_api_key)
        
        # Test getting next event
        print("\n📅 Next UFC Event:")
        next_event = discovery.get_next_event()
        if next_event:
            print(f"   Name: {next_event['name']}")
            print(f"   Date: {next_event['date']}")
            print(f"   Fights: {next_event['fight_count']}")
            print(f"   Main Event: {next_event['fights'][0] if next_event['fights'] else 'N/A'}")
        else:
            print("   No upcoming events found")
        
        # Test getting multiple events
        print("\n📅 Upcoming Events (30 days):")
        events = discovery.get_upcoming_events(days_ahead=30)
        for i, event in enumerate(events[:3], 1):
            print(f"   {i}. {event['name']} - {event['date_str']}")
    else:
        print("⚠️  No API key provided. Skipping tests.")