"""
Historical Odds Fetcher for UFC Predictor
=========================================

Fetches historical UFC betting odds from The Odds API for backtesting.
Covers all available data from June 2020 to present.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/#historical
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HistoricalOddsRecord:
    """Data class for storing historical odds records"""
    event_id: str
    event_name: str
    commence_time: str
    fighter_a: str
    fighter_b: str
    fighter_a_odds: float
    fighter_b_odds: float
    bookmaker: str
    market: str
    timestamp: str
    snapshot_time: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return asdict(self)

class HistoricalOddsAPIClient:
    """Client for fetching historical UFC odds from The Odds API"""
    
    # Historical data availability
    MIN_HISTORICAL_DATE = datetime(2020, 6, 6)  # API historical data starts
    
    def __init__(self, api_key: str, cache_dir: Path = None):
        """
        Initialize the Historical Odds API client
        
        Args:
            api_key: Your The Odds API key (paid tier required)
            cache_dir: Directory to cache historical data
        """
        if not api_key:
            raise ValueError("API key is required for historical data")
        
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "mma_mixed_martial_arts"
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "historical_odds"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track API usage
        self.credits_used = 0
        self.credits_remaining = None
        
        logger.info(f"Historical Odds Client initialized")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def get_historical_events(self, date: datetime) -> List[Dict]:
        """
        Get list of historical UFC events for a specific date
        
        Args:
            date: Date to fetch events for
            
        Returns:
            List of event dictionaries
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/events"
        
        params = {
            'apiKey': self.api_key,
            'date': date.isoformat() + 'Z'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self._update_api_usage(response)
            
            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text}")
                return []
            
            events = response.json().get('data', [])
            logger.info(f"Found {len(events)} events for {date.date()}")
            
            # Return all MMA events (we'll filter for UFC when processing odds)
            return events
            
        except Exception as e:
            logger.error(f"Error fetching events for {date}: {e}")
            return []
    
    def get_historical_odds(self, date: datetime, event_id: str = None, 
                           markets: List[str] = ['h2h'], regions: List[str] = ['au']) -> Dict:
        """
        Fetch historical odds for a specific date/event
        
        Args:
            date: Date to fetch odds for
            event_id: Optional specific event ID
            markets: List of markets to fetch (h2h, totals, etc)
            regions: List of regions (au, us, uk, etc)
            
        Returns:
            Dictionary with odds data
        """
        if event_id:
            url = f"{self.base_url}/historical/sports/{self.sport}/events/{event_id}/odds"
        else:
            url = f"{self.base_url}/historical/sports/{self.sport}/odds"
        
        all_odds = []
        
        for region in regions:
            for market in markets:
                params = {
                    'apiKey': self.api_key,
                    'date': date.isoformat() + 'Z',
                    'regions': region,
                    'markets': market,
                    'oddsFormat': 'decimal'
                }
                
                try:
                    response = requests.get(url, params=params, timeout=30)
                    self._update_api_usage(response)
                    
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch {market} odds for {region}: {response.status_code}")
                        continue
                    
                    data = response.json()
                    
                    # Extract odds data
                    if 'data' in data:
                        odds_data = data['data']
                        timestamp = data.get('timestamp', date.isoformat())
                        
                        for event in odds_data:
                            # Create a copy to avoid modifying the original
                            event_copy = dict(event)
                            event_copy['snapshot_timestamp'] = timestamp
                            event_copy['region'] = region
                            event_copy['market_type'] = market
                            all_odds.append(event_copy)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error fetching {market} odds for {region}: {e}")
                    continue
        
        return {'data': all_odds, 'date': date.isoformat()}
    
    def fetch_date_range(self, start_date: datetime, end_date: datetime, 
                        interval_hours: int = 24) -> List[HistoricalOddsRecord]:
        """
        Fetch historical odds for a date range
        
        Args:
            start_date: Start date for fetching
            end_date: End date for fetching
            interval_hours: Hours between snapshots (24 = daily)
            
        Returns:
            List of HistoricalOddsRecord objects
        """
        records = []
        current_date = start_date
        
        # Calculate total days
        total_days = (end_date - start_date).days
        
        with tqdm(total=total_days, desc="Fetching historical odds") as pbar:
            while current_date <= end_date:
                # Check cache first
                cache_file = self._get_cache_path(current_date)
                
                if cache_file.exists():
                    logger.info(f"Loading cached data for {current_date.date()}")
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        records.extend(self._parse_odds_data(cached_data))
                else:
                    # Fetch from API
                    logger.info(f"Fetching odds for {current_date.date()}")
                    
                    # First get events for this date
                    events = self.get_historical_events(current_date)
                    
                    if events:
                        # Fetch odds for each event
                        for event in events:
                            event_id = event.get('id')
                            odds_data = self.get_historical_odds(current_date, event_id)
                            
                            if odds_data['data']:
                                # Parse and store records
                                event_records = self._parse_odds_data(odds_data)
                                records.extend(event_records)
                                
                                # Cache the data
                                self._cache_data(current_date, odds_data)
                    
                    # Rate limiting
                    time.sleep(1)
                
                # Move to next date
                current_date += timedelta(hours=interval_hours)
                pbar.update(interval_hours / 24)
                
                # Check API limits
                if self.credits_remaining and self.credits_remaining < 100:
                    logger.warning(f"Low API credits remaining: {self.credits_remaining}")
                    break
        
        logger.info(f"Fetched {len(records)} total odds records")
        return records
    
    def fetch_all_historical_ufc(self, start_date: datetime = None, 
                                 end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch ALL available historical UFC odds data
        
        Args:
            start_date: Start date (defaults to June 2020)
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with all historical odds
        """
        if start_date is None:
            start_date = self.MIN_HISTORICAL_DATE
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Fetching ALL UFC historical odds from {start_date.date()} to {end_date.date()}")
        logger.info(f"This may take a while and use significant API credits...")
        
        # Fetch in monthly chunks to manage memory
        all_records = []
        current_start = start_date
        
        while current_start < end_date:
            # Process one month at a time
            current_end = min(current_start + timedelta(days=30), end_date)
            
            logger.info(f"Processing {current_start.date()} to {current_end.date()}")
            
            # Fetch this chunk
            chunk_records = self.fetch_date_range(current_start, current_end)
            all_records.extend(chunk_records)
            
            # Save intermediate results
            if len(all_records) > 0:
                self._save_intermediate_results(all_records)
            
            # Move to next month
            current_start = current_end + timedelta(days=1)
            
            # Check if we should stop
            if self.credits_remaining and self.credits_remaining < 50:
                logger.warning("Stopping due to low API credits")
                break
        
        # Convert to DataFrame
        if all_records:
            df = pd.DataFrame([r.to_dict() for r in all_records])
            
            # Save final results
            output_file = self.cache_dir / f"ufc_historical_odds_{start_date.date()}_{end_date.date()}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} records to {output_file}")
            
            return df
        else:
            logger.warning("No historical odds records found")
            return pd.DataFrame()
    
    def _parse_odds_data(self, odds_data: Dict) -> List[HistoricalOddsRecord]:
        """Parse raw odds data into HistoricalOddsRecord objects"""
        records = []
        
        for event in odds_data.get('data', []):
            event_id = event.get('id', '')
            event_title = event.get('title', '')
            commence_time = event.get('commence_time', '')
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            snapshot_time = event.get('snapshot_timestamp', '')
            
            # Parse bookmaker odds
            for bookmaker in event.get('bookmakers', []):
                bm_name = bookmaker.get('title', '')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')
                    
                    # Extract h2h odds
                    if market_key == 'h2h':
                        home_odds = None
                        away_odds = None
                        
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home_team:
                                home_odds = outcome['price']
                            elif outcome['name'] == away_team:
                                away_odds = outcome['price']
                        
                        if home_odds and away_odds:
                            record = HistoricalOddsRecord(
                                event_id=event_id,
                                event_name=event_title,
                                commence_time=commence_time,
                                fighter_a=home_team,
                                fighter_b=away_team,
                                fighter_a_odds=home_odds,
                                fighter_b_odds=away_odds,
                                bookmaker=bm_name,
                                market=market_key,
                                timestamp=datetime.now().isoformat(),
                                snapshot_time=snapshot_time
                            )
                            records.append(record)
        
        return records
    
    def _update_api_usage(self, response: requests.Response):
        """Update API usage tracking from response headers"""
        self.credits_remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        if self.credits_remaining:
            self.credits_remaining = int(self.credits_remaining)
            logger.debug(f"API Credits - Remaining: {self.credits_remaining}, Used: {used}")
    
    def _get_cache_path(self, date: datetime) -> Path:
        """Get cache file path for a specific date"""
        year_month = date.strftime('%Y-%m')
        cache_subdir = self.cache_dir / year_month
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"odds_{date.strftime('%Y-%m-%d')}.json"
    
    def _cache_data(self, date: datetime, data: Dict):
        """Cache odds data to disk"""
        cache_file = self._get_cache_path(date)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Cached data to {cache_file}")
    
    def _save_intermediate_results(self, records: List[HistoricalOddsRecord]):
        """Save intermediate results during long fetching operations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = self.cache_dir / f"intermediate_{timestamp}.json"
        
        data = [r.to_dict() for r in records]
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved intermediate results ({len(records)} records) to {temp_file}")


class HistoricalOddsProcessor:
    """Process and analyze historical odds data"""
    
    def __init__(self, odds_df: pd.DataFrame):
        """
        Initialize processor with historical odds DataFrame
        
        Args:
            odds_df: DataFrame with historical odds records
        """
        self.odds_df = odds_df
        self._prepare_dataframe()
    
    def _prepare_dataframe(self):
        """Prepare DataFrame for analysis"""
        # Convert timestamp columns to datetime
        if 'commence_time' in self.odds_df.columns:
            self.odds_df['commence_time'] = pd.to_datetime(self.odds_df['commence_time'])
        
        if 'snapshot_time' in self.odds_df.columns:
            self.odds_df['snapshot_time'] = pd.to_datetime(self.odds_df['snapshot_time'])
        
        # Calculate implied probabilities
        self.odds_df['fighter_a_implied_prob'] = 1 / self.odds_df['fighter_a_odds']
        self.odds_df['fighter_b_implied_prob'] = 1 / self.odds_df['fighter_b_odds']
        
        # Calculate bookmaker margin
        self.odds_df['bookmaker_margin'] = (
            self.odds_df['fighter_a_implied_prob'] + 
            self.odds_df['fighter_b_implied_prob'] - 1
        )
    
    def get_fight_odds(self, fighter_a: str, fighter_b: str, 
                      event_date: datetime = None) -> pd.DataFrame:
        """
        Get historical odds for a specific fight
        
        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name
            event_date: Optional event date to filter
            
        Returns:
            DataFrame with fight odds history
        """
        # Filter for the specific fight (check both orderings)
        fight_mask = (
            ((self.odds_df['fighter_a'] == fighter_a) & 
             (self.odds_df['fighter_b'] == fighter_b)) |
            ((self.odds_df['fighter_a'] == fighter_b) & 
             (self.odds_df['fighter_b'] == fighter_a))
        )
        
        fight_odds = self.odds_df[fight_mask].copy()
        
        # Filter by date if provided
        if event_date:
            date_mask = fight_odds['commence_time'].dt.date == event_date.date()
            fight_odds = fight_odds[date_mask]
        
        return fight_odds.sort_values('snapshot_time')
    
    def get_best_odds(self, as_of_date: datetime = None) -> pd.DataFrame:
        """
        Get best available odds for each fight
        
        Args:
            as_of_date: Get best odds as of this date
            
        Returns:
            DataFrame with best odds per fight
        """
        if as_of_date:
            df = self.odds_df[self.odds_df['snapshot_time'] <= as_of_date].copy()
        else:
            df = self.odds_df.copy()
        
        # Group by fight and get best odds
        best_odds = df.groupby(['event_name', 'fighter_a', 'fighter_b']).agg({
            'fighter_a_odds': 'max',
            'fighter_b_odds': 'max',
            'commence_time': 'first',
            'bookmaker': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        return best_odds
    
    def calculate_odds_movement(self) -> pd.DataFrame:
        """
        Calculate odds movement over time for each fight
        
        Returns:
            DataFrame with odds movement statistics
        """
        movements = []
        
        # Group by unique fights
        for (event, f_a, f_b), group in self.odds_df.groupby(['event_name', 'fighter_a', 'fighter_b']):
            if len(group) < 2:
                continue
            
            sorted_group = group.sort_values('snapshot_time')
            
            # Calculate movement
            opening_odds_a = sorted_group.iloc[0]['fighter_a_odds']
            closing_odds_a = sorted_group.iloc[-1]['fighter_a_odds']
            opening_odds_b = sorted_group.iloc[0]['fighter_b_odds']
            closing_odds_b = sorted_group.iloc[-1]['fighter_b_odds']
            
            movement = {
                'event': event,
                'fighter_a': f_a,
                'fighter_b': f_b,
                'fighter_a_opening': opening_odds_a,
                'fighter_a_closing': closing_odds_a,
                'fighter_a_movement': closing_odds_a - opening_odds_a,
                'fighter_a_movement_pct': (closing_odds_a - opening_odds_a) / opening_odds_a * 100,
                'fighter_b_opening': opening_odds_b,
                'fighter_b_closing': closing_odds_b,
                'fighter_b_movement': closing_odds_b - opening_odds_b,
                'fighter_b_movement_pct': (closing_odds_b - opening_odds_b) / opening_odds_b * 100,
                'snapshots': len(group)
            }
            movements.append(movement)
        
        return pd.DataFrame(movements)


def main():
    """Example usage of the historical odds fetcher"""
    
    # Get API key from environment
    api_key = os.getenv('UFC_ODDS_API_KEY')
    if not api_key:
        logger.error("Please set UFC_ODDS_API_KEY environment variable")
        return
    
    # Initialize client
    client = HistoricalOddsAPIClient(api_key)
    
    # Example 1: Fetch last 3 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logger.info(f"Fetching 3 months of historical odds...")
    records = client.fetch_date_range(start_date, end_date)
    
    if records:
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in records])
        
        # Process the data
        processor = HistoricalOddsProcessor(df)
        
        # Get best odds
        best_odds = processor.get_best_odds()
        logger.info(f"Found best odds for {len(best_odds)} fights")
        
        # Calculate odds movements
        movements = processor.calculate_odds_movement()
        logger.info(f"Calculated odds movements for {len(movements)} fights")
        
        # Save results
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "historical_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_odds.to_csv(output_dir / "best_historical_odds.csv", index=False)
        movements.to_csv(output_dir / "odds_movements.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    # Example 2: Fetch ALL historical data (use with caution - uses many credits)
    # Uncomment to fetch all data since June 2020
    # all_data = client.fetch_all_historical_ufc()
    # logger.info(f"Fetched {len(all_data)} total historical records")


if __name__ == "__main__":
    main()