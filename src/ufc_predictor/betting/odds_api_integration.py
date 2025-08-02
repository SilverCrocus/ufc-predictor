"""
The Odds API Integration for UFC Predictor
==========================================

Clean integration with The Odds API for UFC betting odds.
No fallbacks - shows clear errors when API fails.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

class OddsAPIError(Exception):
    """Custom exception for Odds API errors"""
    pass

class UFCOddsAPIClient:
    """Client for fetching UFC odds from The Odds API"""
    
    def __init__(self, api_key: str):
        """
        Initialize the Odds API client
        
        Args:
            api_key: Your The Odds API key (required)
        """
        if not api_key:
            raise OddsAPIError("API key is required. Get your free key at: https://the-odds-api.com/")
        
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "mma_mixed_martial_arts"
        
    def get_ufc_odds(self, region: str = "au") -> List[Dict]:
        """
        Fetch current UFC odds from The Odds API
        
        Args:
            region: Betting region ('au' for Australia, 'us' for US, etc.)
            
        Returns:
            List of fight data with odds
            
        Raises:
            OddsAPIError: If API call fails
        """
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': region,
            'markets': 'h2h',  # Head-to-head (moneyline)
            'oddsFormat': 'decimal'
        }
        
        print(f"🔄 Fetching UFC odds from The Odds API...")
        print(f"   Region: {region}")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Check for API errors
            if response.status_code == 401:
                raise OddsAPIError("Invalid API key. Check your key at: https://the-odds-api.com/account/")
            elif response.status_code == 422:
                raise OddsAPIError("Invalid parameters. Check region and sport settings.")
            elif response.status_code == 429:
                raise OddsAPIError("Rate limit exceeded. You've used all your API requests.")
            elif response.status_code != 200:
                raise OddsAPIError(f"API error {response.status_code}: {response.text}")
            
            # Parse response
            data = response.json()
            
            # Check API usage
            remaining = response.headers.get('x-requests-remaining', 'Unknown')
            used = response.headers.get('x-requests-used', 'Unknown')
            print(f"📊 API Usage - Remaining: {remaining}, Used: {used}")
            
            if not data:
                raise OddsAPIError("No UFC events found. Check if there are upcoming UFC fights.")
            
            print(f"✅ Found {len(data)} UFC events")
            return data
            
        except requests.RequestException as e:
            raise OddsAPIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise OddsAPIError(f"Invalid JSON response: {str(e)}")
    
    def format_odds_for_analysis(self, api_data: List[Dict], target_fights: List[str] = None) -> Dict[str, Dict]:
        """
        Format The Odds API data for the UFC predictor analysis
        
        Args:
            api_data: Raw data from The Odds API
            target_fights: Optional list of specific fights to extract
            
        Returns:
            Dictionary formatted for profitability analysis
        """
        formatted_odds = {}
        
        for event in api_data:
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            
            if not home_team or not away_team:
                continue
            
            # Create fight key
            fight_key = f"{home_team} vs {away_team}"
            
            # Filter for target fights if specified
            if target_fights:
                fight_found = False
                for target in target_fights:
                    if self._fights_match(fight_key, target):
                        fight_found = True
                        break
                if not fight_found:
                    continue
            
            # Extract best odds from all bookmakers
            best_home_odds = None
            best_away_odds = None
            bookmaker_info = []
            
            for bookmaker in event.get('bookmakers', []):
                bm_name = bookmaker.get('title', 'Unknown')
                
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            name = outcome.get('name', '')
                            price = outcome.get('price', 0)
                            
                            if name == home_team:
                                if best_home_odds is None or price > best_home_odds:
                                    best_home_odds = price
                            elif name == away_team:
                                if best_away_odds is None or price > best_away_odds:
                                    best_away_odds = price
                        
                        bookmaker_info.append(bm_name)
            
            if best_home_odds and best_away_odds:
                formatted_odds[fight_key] = {
                    'fighter_a': home_team,
                    'fighter_b': away_team,
                    'fighter_a_decimal_odds': best_home_odds,
                    'fighter_b_decimal_odds': best_away_odds,
                    'bookmakers': list(set(bookmaker_info)),
                    'commence_time': event.get('commence_time', ''),
                    'api_timestamp': datetime.now().isoformat()
                }
                
                print(f"📋 {fight_key}")
                print(f"   {home_team}: {best_home_odds}")
                print(f"   {away_team}: {best_away_odds}")
                print(f"   Bookmakers: {', '.join(set(bookmaker_info))}")
        
        return formatted_odds
    
    def _fights_match(self, api_fight: str, target_fight: str, threshold: float = 0.8) -> bool:
        """
        Check if an API fight matches a target fight using fuzzy matching
        
        Args:
            api_fight: Fight string from API
            target_fight: Target fight string to match
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if fights match above threshold
        """
        from difflib import SequenceMatcher
        
        # Normalize fight strings
        api_normalized = api_fight.lower().replace(' vs ', ' vs. ').strip()
        target_normalized = target_fight.lower().replace(' vs ', ' vs. ').strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, api_normalized, target_normalized).ratio()
        
        return similarity >= threshold

def get_ufc_304_odds(api_key: str) -> Dict[str, Dict]:
    """
    Get odds specifically for UFC 304 fights
    
    Args:
        api_key: Your The Odds API key
        
    Returns:
        Formatted odds data for UFC 304
        
    Raises:
        OddsAPIError: If API call fails
    """
    # UFC 304 Fight Card
    ufc_304_fights = [
        "Amir Albazi vs Tatsuro Taira",
        "Mateusz Rebecki vs Chris Duncan", 
        "Elves Brener vs Esteban Ribovics",
        "Karol Rosa vs Nora Cornolle",
        "Neil Magny vs Elizeu Zaleski dos Santos",
        "Danny Silva vs Kevin Vallejos"
    ]
    
    print("🏆 UFC 304 - August 3rd Odds Fetching")
    print("=" * 45)
    
    # Initialize client
    client = UFCOddsAPIClient(api_key)
    
    # Fetch all UFC odds
    api_data = client.get_ufc_odds(region="au")  # Australian odds
    
    # Format for UFC 304 specifically
    ufc_304_odds = client.format_odds_for_analysis(api_data, ufc_304_fights)
    
    if not ufc_304_odds:
        raise OddsAPIError(
            "No UFC 304 odds found. Possible reasons:\n"
            "1. UFC 304 fights not available yet\n"
            "2. Event has already occurred\n"
            "3. Fighter name mismatches\n"
            "4. Bookmakers not offering odds yet"
        )
    
    print(f"\n✅ Successfully retrieved odds for {len(ufc_304_odds)} UFC 304 fights")
    return ufc_304_odds

def convert_to_profitability_format(odds_data: Dict[str, Dict]) -> Dict[str, float]:
    """
    Convert Odds API format to the format expected by profitability analysis
    
    Args:
        odds_data: Formatted odds data from get_ufc_304_odds()
        
    Returns:
        Dictionary mapping fighter names to decimal odds
    """
    fighter_odds = {}
    
    for fight_key, fight_data in odds_data.items():
        fighter_a = fight_data['fighter_a']
        fighter_b = fight_data['fighter_b']
        
        fighter_odds[fighter_a] = fight_data['fighter_a_decimal_odds']
        fighter_odds[fighter_b] = fight_data['fighter_b_decimal_odds']
    
    return fighter_odds

if __name__ == "__main__":
    # Test the integration
    test_api_key = "your_api_key_here"
    
    try:
        odds = get_ufc_304_odds(test_api_key)
        print(f"\n🎯 Test successful! Found odds for {len(odds)} fights")
        
        # Show profitability format conversion
        fighter_odds = convert_to_profitability_format(odds)
        print(f"\n📊 Profitability format: {len(fighter_odds)} fighters")
        
    except OddsAPIError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")