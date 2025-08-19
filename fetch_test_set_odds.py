#!/usr/bin/env python3
"""
Fetch Historical Odds for Test Set Only
========================================

Smart fetcher that only gets odds for fights in your test set.
Saves massive API credits by not fetching unnecessary data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import pickle
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.ufc_predictor.betting.historical_odds_fetcher import (
    HistoricalOddsAPIClient,
    HistoricalOddsRecord
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSetOddsFetcher:
    """Fetches historical odds ONLY for test set fights"""
    
    def __init__(self, api_key: str = None):
        """Initialize the test set odds fetcher"""
        self.project_root = Path(__file__).parent
        self.parent_root = self.project_root.parent
        
        # Set up directories
        self.cache_dir = self.project_root / "data" / "test_set_odds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get API key
        if api_key is None:
            api_key = os.getenv('UFC_ODDS_API_KEY')
            if not api_key:
                # Try .env file
                env_file = self.project_root / '.env'
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            if 'UFC_ODDS_API_KEY' in line:
                                api_key = line.split('=')[1].strip().strip('"').strip("'")
                                break
        
        if not api_key:
            raise ValueError("API key required. Set UFC_ODDS_API_KEY environment variable")
        
        self.api_client = HistoricalOddsAPIClient(api_key, cache_dir=self.cache_dir)
        
        # Cache management
        self.cache_file = self.cache_dir / "test_set_odds_cache.json"
        self.progress_file = self.cache_dir / "fetch_progress.json"
        self.cached_odds = self.load_cache()
        self.progress = self.load_progress()
    
    def load_cache(self) -> Dict:
        """Load cached odds data"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        """Save cached odds data"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cached_odds, f, indent=2, default=str)
    
    def load_progress(self) -> Dict:
        """Load fetch progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'fetched_fights': [], 'last_index': 0}
    
    def save_progress(self):
        """Save fetch progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def identify_test_set_fights(self) -> pd.DataFrame:
        """Identify fights in the test set based on temporal split"""
        
        logger.info("Identifying test set fights...")
        
        # Find fights data
        fights_paths = [
            self.project_root / "src" / "ufc_predictor" / "data" / "ufc_fights.csv",
            self.parent_root / "src" / "ufc_predictor" / "data" / "ufc_fights.csv",
            self.project_root / "data" / "ufc_fights.csv",
            self.parent_root / "data" / "ufc_fights.csv"
        ]
        
        fights_path = None
        for path in fights_paths:
            if path.exists():
                fights_path = path
                break
        
        if not fights_path:
            raise FileNotFoundError("Could not find ufc_fights.csv")
        
        # Load fights
        fights_df = pd.read_csv(fights_path)
        logger.info(f"Loaded {len(fights_df)} total fights")
        
        # Parse dates
        import re
        dates = []
        for idx, row in fights_df.iterrows():
            date = None
            
            # Try Date column first
            if 'Date' in fights_df.columns and pd.notna(row.get('Date')):
                try:
                    date = pd.to_datetime(row['Date'])
                except:
                    pass
            
            # Try extracting from Event column
            if date is None and 'Event' in fights_df.columns:
                event = row.get('Event', '')
                match = re.search(r'([A-Z][a-z]+)\. (\d{1,2}), (\d{4})', event)
                if match:
                    month_str, day, year = match.groups()
                    months = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    month = months.get(month_str[:3], 1)
                    try:
                        date = datetime(int(year), month, int(day))
                    except:
                        pass
            
            dates.append(date)
        
        fights_df['parsed_date'] = dates
        
        # Remove invalid dates and sort
        valid_dates = fights_df['parsed_date'].notna()
        fights_with_dates = fights_df[valid_dates].sort_values('parsed_date')
        
        # Get test set (last 20%)
        split_idx = int(len(fights_with_dates) * 0.8)
        test_fights = fights_with_dates.iloc[split_idx:].copy()
        
        # Add unique fight identifier
        test_fights['fight_id'] = test_fights.apply(
            lambda x: f"{x['Fighter']}_{x['Opponent']}_{x['parsed_date'].date()}", 
            axis=1
        )
        
        logger.info(f"Identified {len(test_fights)} test set fights")
        logger.info(f"Date range: {test_fights['parsed_date'].min().date()} to {test_fights['parsed_date'].max().date()}")
        
        return test_fights
    
    def create_fight_key(self, fighter: str, opponent: str, date: datetime) -> str:
        """Create a unique key for a fight"""
        return f"{fighter}_{opponent}_{date.date()}"
    
    def fetch_odds_for_fight(self, fight: pd.Series) -> Dict:
        """Fetch odds for a specific fight"""
        
        fighter = fight['Fighter']
        opponent = fight['Opponent']
        fight_date = fight['parsed_date']
        fight_id = fight['fight_id']
        
        # Check cache first
        if fight_id in self.cached_odds:
            return self.cached_odds[fight_id]
        
        # Fetch odds for the fight date
        try:
            # Get odds for the day of the fight
            odds_data = self.api_client.get_historical_odds(fight_date)
            
            # Look for this specific fight in the odds
            fight_odds = None
            for event in odds_data.get('data', []):
                # Check if this event matches our fight
                home = event.get('home_team', '').lower()
                away = event.get('away_team', '').lower()
                
                fighter_lower = fighter.lower()
                opponent_lower = opponent.lower()
                
                # Check both orderings
                if ((fighter_lower in home and opponent_lower in away) or
                    (opponent_lower in home and fighter_lower in away) or
                    (home in fighter_lower and away in opponent_lower) or
                    (away in fighter_lower and home in opponent_lower)):
                    
                    # Found the fight!
                    fight_odds = {
                        'fight_id': fight_id,
                        'fighter': fighter,
                        'opponent': opponent,
                        'date': fight_date.isoformat(),
                        'odds_data': event,
                        'fetched_at': datetime.now().isoformat()
                    }
                    break
            
            # Cache the result (even if None to avoid re-fetching)
            self.cached_odds[fight_id] = fight_odds
            
            return fight_odds
            
        except Exception as e:
            logger.warning(f"Failed to fetch odds for {fighter} vs {opponent}: {e}")
            # Cache None to avoid re-fetching
            self.cached_odds[fight_id] = None
            return None
    
    def fetch_all_test_set_odds(self, resume: bool = True, 
                               batch_size: int = 50,
                               delay: float = 1.0):
        """
        Fetch odds for all test set fights
        
        Args:
            resume: Whether to resume from last checkpoint
            batch_size: Number of fights to fetch before saving
            delay: Delay between API calls in seconds
        """
        
        # Get test set fights
        test_fights = self.identify_test_set_fights()
        
        # Resume from checkpoint if requested
        start_idx = 0
        if resume and self.progress['last_index'] > 0:
            start_idx = self.progress['last_index']
            logger.info(f"Resuming from fight {start_idx}")
        
        # Group fights by date to optimize API calls
        fights_by_date = test_fights.groupby(test_fights['parsed_date'].dt.date)
        
        logger.info(f"\n{'='*60}")
        logger.info("FETCHING ODDS FOR TEST SET FIGHTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total unique dates: {len(fights_by_date)}")
        logger.info(f"Total fights: {len(test_fights)}")
        logger.info(f"Estimated API calls: {len(fights_by_date)}")
        logger.info(f"{'='*60}\n")
        
        fetched_count = 0
        matched_count = 0
        
        # Process each date
        with tqdm(total=len(fights_by_date), desc="Processing dates") as pbar:
            for date, date_fights in fights_by_date:
                # Skip if already processed
                all_cached = all(
                    fight['fight_id'] in self.cached_odds 
                    for _, fight in date_fights.iterrows()
                )
                
                if all_cached:
                    pbar.update(1)
                    continue
                
                # Fetch odds for this date once
                try:
                    date_obj = pd.to_datetime(date)
                    odds_data = self.api_client.get_historical_odds(date_obj)
                    
                    # Match each fight on this date
                    for _, fight in date_fights.iterrows():
                        fight_id = fight['fight_id']
                        
                        if fight_id in self.cached_odds:
                            continue
                        
                        # Look for this fight in the odds data
                        matched = False
                        for event in odds_data.get('data', []):
                            if self.fights_match(fight, event):
                                # Found match!
                                self.cached_odds[fight_id] = {
                                    'fight_id': fight_id,
                                    'fighter': fight['Fighter'],
                                    'opponent': fight['Opponent'],
                                    'date': date_obj.isoformat(),
                                    'odds_data': event,
                                    'fetched_at': datetime.now().isoformat()
                                }
                                matched = True
                                matched_count += 1
                                break
                        
                        if not matched:
                            # Cache as not found
                            self.cached_odds[fight_id] = None
                        
                        fetched_count += 1
                    
                    # Save progress periodically
                    if fetched_count % batch_size == 0:
                        self.save_cache()
                        self.progress['last_index'] = fetched_count
                        self.save_progress()
                        logger.info(f"Progress saved: {fetched_count} fights processed, {matched_count} matched")
                    
                    # Rate limiting
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Failed to process date {date}: {e}")
                
                pbar.update(1)
        
        # Final save
        self.save_cache()
        self.save_progress()
        
        # Generate summary
        logger.info(f"\n{'='*60}")
        logger.info("FETCH COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total fights processed: {fetched_count}")
        logger.info(f"Fights with odds found: {matched_count}")
        logger.info(f"Match rate: {matched_count/fetched_count*100:.1f}%")
        logger.info(f"API credits used: ~{len(fights_by_date)}")
        
        # Save consolidated dataset
        self.create_consolidated_dataset(test_fights)
    
    def fights_match(self, fight: pd.Series, event: Dict) -> bool:
        """Check if a fight matches an event from the API"""
        
        fighter = fight['Fighter'].lower()
        opponent = fight['Opponent'].lower()
        
        home = event.get('home_team', '').lower()
        away = event.get('away_team', '').lower()
        
        # Fuzzy matching with various strategies
        strategies = [
            # Exact match
            (fighter == home and opponent == away),
            (opponent == home and fighter == away),
            
            # Contains match
            (fighter in home and opponent in away),
            (opponent in home and fighter in away),
            (home in fighter and away in opponent),
            (away in fighter and home in opponent),
            
            # Last name match (common in UFC)
            (fighter.split()[-1] in home and opponent.split()[-1] in away),
            (opponent.split()[-1] in home and fighter.split()[-1] in away),
        ]
        
        return any(strategies)
    
    def create_consolidated_dataset(self, test_fights: pd.DataFrame):
        """Create a consolidated dataset with fights and odds"""
        
        logger.info("\nCreating consolidated dataset...")
        
        # Build consolidated data
        consolidated = []
        
        for _, fight in test_fights.iterrows():
            fight_id = fight['fight_id']
            odds_info = self.cached_odds.get(fight_id)
            
            record = {
                'fighter': fight['Fighter'],
                'opponent': fight['Opponent'],
                'outcome': fight.get('Outcome', ''),
                'date': fight['parsed_date'],
                'event': fight.get('Event', ''),
                'has_odds': odds_info is not None and odds_info != {}
            }
            
            if odds_info and odds_info != {}:
                odds_data = odds_info.get('odds_data', {})
                
                # Extract best odds from bookmakers
                best_home_odds = None
                best_away_odds = None
                
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                name = outcome.get('name', '')
                                price = outcome.get('price', 0)
                                
                                if name == odds_data.get('home_team'):
                                    if best_home_odds is None or price > best_home_odds:
                                        best_home_odds = price
                                elif name == odds_data.get('away_team'):
                                    if best_away_odds is None or price > best_away_odds:
                                        best_away_odds = price
                
                # Determine which odds correspond to which fighter
                if self.fighter_is_home(fight['Fighter'], odds_data):
                    record['fighter_odds'] = best_home_odds
                    record['opponent_odds'] = best_away_odds
                else:
                    record['fighter_odds'] = best_away_odds
                    record['opponent_odds'] = best_home_odds
            
            consolidated.append(record)
        
        # Create DataFrame
        consolidated_df = pd.DataFrame(consolidated)
        
        # Save to CSV
        output_file = self.cache_dir / f"test_set_with_odds_{datetime.now().strftime('%Y%m%d')}.csv"
        consolidated_df.to_csv(output_file, index=False)
        
        logger.info(f"Consolidated dataset saved to: {output_file}")
        logger.info(f"Total fights: {len(consolidated_df)}")
        logger.info(f"Fights with odds: {consolidated_df['has_odds'].sum()}")
        logger.info(f"Coverage: {consolidated_df['has_odds'].mean()*100:.1f}%")
        
        return consolidated_df
    
    def fighter_is_home(self, fighter: str, odds_data: Dict) -> bool:
        """Determine if fighter is the home team in odds data"""
        home = odds_data.get('home_team', '').lower()
        fighter_lower = fighter.lower()
        
        return (fighter_lower in home or 
                home in fighter_lower or 
                fighter_lower.split()[-1] in home)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch odds for test set fights only')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls')
    
    args = parser.parse_args()
    
    try:
        fetcher = TestSetOddsFetcher()
        fetcher.fetch_all_test_set_odds(resume=args.resume, delay=args.delay)
        
        logger.info("\n✅ Success! Your test set odds are ready for backtesting")
        logger.info("\nNext step: Run backtesting with:")
        logger.info("python3 backtest_temporal_model.py")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())