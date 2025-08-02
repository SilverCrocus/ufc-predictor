"""
TAB Australia Scraper Integration for Enhanced Odds Service

Provides seamless integration between the enhanced odds service and the existing
TAB Australia scraper, with async support and error handling.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import sys

# Add the webscraper directory to the path to import TAB scraper
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "webscraper"))

try:
    from tab_australia_scraper import TABAustraliaUFCScraper, TABFightOdds
except ImportError:
    # Fallback for when scraper isn't available
    TABAustraliaUFCScraper = None
    TABFightOdds = None

logger = logging.getLogger(__name__)


class TABOddsIntegration:
    """
    Integration wrapper for TAB Australia scraper
    
    Provides async interface and error handling for the TAB scraper
    to work seamlessly with the enhanced odds service.
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize TAB integration
        
        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self.scraper = None
        self.is_available = TABAustraliaUFCScraper is not None
        self.last_scrape_time = None
        self.scrape_cache = {}
        self.cache_duration_minutes = 15  # Cache results for 15 minutes
        
        if not self.is_available:
            logger.warning("TAB Australia scraper not available")
        else:
            logger.info("TAB integration initialized")
    
    async def fetch_event_odds(self, event_name: str, 
                             target_fights: Optional[List[str]] = None) -> Optional[List[TABFightOdds]]:
        """
        Fetch odds for a specific event from TAB Australia
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            
        Returns:
            List of TABFightOdds objects or None if failed
        """
        if not self.is_available:
            logger.warning("TAB scraper not available")
            return None
        
        # Check cache first
        cache_key = f"{event_name}_{hash(str(target_fights))}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached TAB data for {event_name}")
            return self.scrape_cache[cache_key]['data']
        
        try:
            logger.info(f"Scraping TAB Australia for {event_name}")
            
            # Run scraper in executor to avoid blocking
            loop = asyncio.get_event_loop()
            odds_data = await loop.run_in_executor(
                None, self._run_scraper_sync, target_fights
            )
            
            if odds_data:
                # Filter for target fights if specified
                if target_fights:
                    odds_data = self._filter_for_target_fights(odds_data, target_fights)
                
                # Cache the results
                self.scrape_cache[cache_key] = {
                    'data': odds_data,
                    'timestamp': datetime.now(),
                    'event_name': event_name
                }
                
                self.last_scrape_time = datetime.now()
                logger.info(f"Successfully scraped {len(odds_data)} fights from TAB")
                
                return odds_data
            else:
                logger.warning("No odds data returned from TAB scraper")
                return None
                
        except Exception as e:
            logger.error(f"TAB scraping failed for {event_name}: {str(e)}")
            return None
    
    def _run_scraper_sync(self, target_fights: Optional[List[str]] = None) -> Optional[List[TABFightOdds]]:
        """
        Run the TAB scraper synchronously
        
        Args:
            target_fights: Optional list of specific fights
            
        Returns:
            List of TABFightOdds objects or None if failed
        """
        try:
            scraper = TABAustraliaUFCScraper(headless=self.headless)
            odds_data = scraper.scrape_all_ufc_odds()
            
            # Clean up the scraper
            if hasattr(scraper, 'driver') and scraper.driver:
                scraper.driver.quit()
            
            return odds_data if odds_data else None
            
        except Exception as e:
            logger.error(f"Sync TAB scraper failed: {str(e)}")
            return None
    
    def _filter_for_target_fights(self, odds_data: List[TABFightOdds], 
                                target_fights: List[str]) -> List[TABFightOdds]:
        """
        Filter odds data for specific target fights
        
        Args:
            odds_data: List of all TABFightOdds
            target_fights: List of target fight strings
            
        Returns:
            Filtered list of TABFightOdds
        """
        filtered_odds = []
        
        for fight_odds in odds_data:
            fight_string = f"{fight_odds.fighter_a} vs {fight_odds.fighter_b}"
            
            # Check if this fight matches any target fight
            for target in target_fights:
                if self._fights_match(fight_string, target):
                    filtered_odds.append(fight_odds)
                    break
        
        return filtered_odds
    
    def _fights_match(self, tab_fight: str, target_fight: str, threshold: float = 0.8) -> bool:
        """
        Check if a TAB fight matches a target fight using fuzzy matching
        
        Args:
            tab_fight: Fight string from TAB
            target_fight: Target fight string
            threshold: Similarity threshold
            
        Returns:
            True if fights match above threshold
        """
        from difflib import SequenceMatcher
        
        # Normalize fight strings
        tab_normalized = tab_fight.lower().replace(' vs ', ' vs. ').strip()
        target_normalized = target_fight.lower().replace(' vs ', ' vs. ').strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, tab_normalized, target_normalized).ratio()
        
        return similarity >= threshold
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.scrape_cache:
            return False
        
        cache_entry = self.scrape_cache[cache_key]
        cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds() / 60
        
        return cache_age < self.cache_duration_minutes
    
    def clear_cache(self):
        """Clear the scraping cache"""
        self.scrape_cache.clear()
        logger.info("TAB scraping cache cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get cache status and statistics
        
        Returns:
            Dict with cache information
        """
        cache_entries = []
        for key, entry in self.scrape_cache.items():
            age_minutes = (datetime.now() - entry['timestamp']).total_seconds() / 60
            cache_entries.append({
                'event_name': entry['event_name'],
                'fights_count': len(entry['data']),
                'age_minutes': age_minutes,
                'valid': age_minutes < self.cache_duration_minutes
            })
        
        return {
            'total_entries': len(self.scrape_cache),
            'valid_entries': len([e for e in cache_entries if e['valid']]),
            'cache_duration_minutes': self.cache_duration_minutes,
            'last_scrape_time': self.last_scrape_time.isoformat() if self.last_scrape_time else None,
            'entries': cache_entries
        }
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """
        Test TAB Australia website connectivity
        
        Returns:
            Dict with connectivity test results
        """
        if not self.is_available:
            return {
                'status': 'unavailable',
                'error': 'TAB scraper module not available'
            }
        
        try:
            logger.info("Testing TAB Australia connectivity")
            
            # Run a minimal scraper test
            loop = asyncio.get_event_loop()
            test_result = await loop.run_in_executor(
                None, self._test_connectivity_sync
            )
            
            return test_result
            
        except Exception as e:
            logger.error(f"TAB connectivity test failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_connectivity_sync(self) -> Dict[str, Any]:
        """
        Synchronous connectivity test
        
        Returns:
            Dict with test results
        """
        try:
            scraper = TABAustraliaUFCScraper(headless=True)
            
            # Just test driver setup
            if scraper.setup_driver():
                # Quick page load test
                scraper.driver.get("https://www.tab.com.au/sports/betting/UFC")
                page_title = scraper.driver.title
                
                # Clean up
                scraper.driver.quit()
                
                return {
                    'status': 'success',
                    'page_title': page_title,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Could not initialize WebDriver'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_sample_data(self) -> List[Dict[str, Any]]:
        """
        Get sample TAB odds data for testing/fallback
        
        Returns:
            List of sample fight odds in dictionary format
        """
        sample_data = [
            {
                'event_name': 'UFC 307',
                'fighter_a': 'Alex Pereira',
                'fighter_b': 'Khalil Rountree Jr',
                'fighter_a_decimal_odds': 1.33,
                'fighter_b_decimal_odds': 3.25,
                'fighter_a_american_odds': -303,
                'fighter_b_american_odds': 225,
                'fight_time': '2024-10-05 22:00:00',
                'is_featured_fight': True
            },
            {
                'event_name': 'UFC 307',
                'fighter_a': 'Raquel Pennington',
                'fighter_b': 'Julianna Pena',
                'fighter_a_decimal_odds': 2.10,
                'fighter_b_decimal_odds': 1.75,
                'fighter_a_american_odds': 110,
                'fighter_b_american_odds': -133,
                'fight_time': '2024-10-05 22:00:00',
                'is_featured_fight': True
            }
        ]
        
        return sample_data