#!/usr/bin/env python3
"""
Unified Odds Scraper - Consolidated scraping functionality
Combines all odds scraping into a single, robust module.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path


@dataclass
class FightOdds:
    """Represents odds for a single fight."""
    fighter1: str
    fighter2: str
    fighter1_odds: float
    fighter2_odds: float
    source: str
    timestamp: str
    event: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class UnifiedOddsScraper:
    """
    Unified odds scraper supporting multiple sources.
    Includes caching, fallback mechanisms, and async support.
    """
    
    def __init__(self, cache_dir: str = "cache/odds"):
        """Initialize scraper with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure sources
        self.sources = {
            'tab': self._scrape_tab_australia,
            'fightodds': self._scrape_fightodds,
            'cached': self._load_cached_odds
        }
        
        # Stealth browser setup
        self.browser_options = None
        self.driver = None
        
    def get_odds(self, 
                event: Optional[str] = None,
                source: str = 'tab',
                use_cache: bool = True,
                cache_hours: int = 24) -> List[FightOdds]:
        """
        Get odds from specified source.
        
        Args:
            event: Event name (for filtering)
            source: Source to scrape ('tab', 'fightodds', 'cached')
            use_cache: Whether to check cache first
            cache_hours: Hours before cache expires
            
        Returns:
            List of FightOdds objects
        """
        # Check cache first if enabled
        if use_cache:
            cached = self._check_cache(event, source, cache_hours)
            if cached:
                return cached
        
        # Scrape fresh odds
        if source in self.sources:
            odds = self.sources[source](event)
            
            # Cache results
            if odds and source != 'cached':
                self._save_to_cache(odds, event, source)
            
            return odds
        else:
            raise ValueError(f"Unknown source: {source}")
    
    async def get_odds_async(self,
                           events: List[str],
                           source: str = 'tab') -> Dict[str, List[FightOdds]]:
        """
        Get odds for multiple events asynchronously.
        
        Args:
            events: List of event names
            source: Source to scrape
            
        Returns:
            Dictionary mapping event to odds
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for event in events:
                tasks.append(self._fetch_odds_async(session, event, source))
            
            results = await asyncio.gather(*tasks)
            
        return dict(zip(events, results))
    
    def _setup_stealth_browser(self):
        """Setup Selenium browser with stealth options."""
        if self.browser_options is None:
            options = Options()
            
            # Stealth settings
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            # Performance settings
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--headless')
            
            self.browser_options = options
    
    def _scrape_tab_australia(self, event: Optional[str] = None) -> List[FightOdds]:
        """Scrape TAB Australia odds."""
        try:
            self._setup_stealth_browser()
            
            if self.driver is None:
                self.driver = webdriver.Chrome(options=self.browser_options)
            
            # Navigate to UFC betting page
            url = "https://www.tab.com.au/sports/betting/Mixed%20Martial%20Arts/competitions/UFC"
            self.driver.get(url)
            
            # Wait for odds to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "match-content")))
            
            # Parse odds
            odds_list = []
            matches = self.driver.find_elements(By.CLASS_NAME, "match-content")
            
            for match in matches:
                try:
                    fighters = match.find_elements(By.CLASS_NAME, "participant-name")
                    odds = match.find_elements(By.CLASS_NAME, "odds-button")
                    
                    if len(fighters) >= 2 and len(odds) >= 2:
                        fight_odds = FightOdds(
                            fighter1=fighters[0].text,
                            fighter2=fighters[1].text,
                            fighter1_odds=float(odds[0].text),
                            fighter2_odds=float(odds[1].text),
                            source='tab',
                            timestamp=datetime.now().isoformat(),
                            event=event
                        )
                        odds_list.append(fight_odds)
                
                except Exception as e:
                    continue
            
            return odds_list
            
        except Exception as e:
            print(f"Error scraping TAB: {e}")
            return []
        
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def _scrape_fightodds(self, event: Optional[str] = None) -> List[FightOdds]:
        """Scrape FightOdds.io (simplified)."""
        try:
            url = "https://fightodds.io/events"
            
            # Use requests for simpler sites
            import requests
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse odds (simplified - would need real selectors)
            odds_list = []
            
            # This is a placeholder - actual implementation would parse the HTML
            # For now, return empty list
            
            return odds_list
            
        except Exception as e:
            print(f"Error scraping FightOdds: {e}")
            return []
    
    def _load_cached_odds(self, event: Optional[str] = None) -> List[FightOdds]:
        """Load odds from cache files."""
        odds_list = []
        
        # Find cache files
        cache_files = self.cache_dir.glob("*.json")
        
        for cache_file in cache_files:
            if event and event.lower() not in cache_file.name.lower():
                continue
            
            with open(cache_file) as f:
                data = json.load(f)
                
                for fight in data.get('fights', []):
                    odds_list.append(FightOdds(**fight))
        
        return odds_list
    
    def _check_cache(self, event: Optional[str], 
                    source: str, hours: int) -> Optional[List[FightOdds]]:
        """Check if valid cache exists."""
        cache_file = self.cache_dir / f"{source}_{event}_{datetime.now().date()}.json"
        
        if cache_file.exists():
            # Check age
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            
            if age_hours < hours:
                with open(cache_file) as f:
                    data = json.load(f)
                    
                return [FightOdds(**fight) for fight in data['fights']]
        
        return None
    
    def _save_to_cache(self, odds: List[FightOdds], 
                      event: Optional[str], source: str):
        """Save odds to cache."""
        cache_file = self.cache_dir / f"{source}_{event}_{datetime.now().date()}.json"
        
        data = {
            'source': source,
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'fights': [odds.to_dict() for odds in odds]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _fetch_odds_async(self, session: aiohttp.ClientSession,
                               event: str, source: str) -> List[FightOdds]:
        """Async helper for fetching odds."""
        # Simplified async implementation
        # In practice, would make actual async HTTP requests
        
        # For now, use sync method
        return self.get_odds(event, source)
    
    def export_to_csv(self, odds: List[FightOdds], filepath: str):
        """Export odds to CSV file."""
        df = pd.DataFrame([odd.to_dict() for odd in odds])
        df.to_csv(filepath, index=False)
        print(f"Exported {len(odds)} odds to {filepath}")
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None