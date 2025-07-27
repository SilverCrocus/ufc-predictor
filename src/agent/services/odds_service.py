"""
UFC Odds Service

Professional odds fetching and management service extracted from notebook workflow.
Provides live odds integration with structured storage and analysis capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OddsRecord:
    """Structured odds record for CSV storage"""
    
    def __init__(self, timestamp: str, event_name: str, fight_key: str, 
                 fighter: str, opponent: str, decimal_odds: float, position: str):
        self.timestamp = timestamp
        self.event_name = event_name
        self.fight_key = fight_key
        self.fighter = fighter
        self.opponent = opponent
        self.decimal_odds = decimal_odds
        self.american_odds = self._decimal_to_american(decimal_odds)
        self.implied_probability = 1 / decimal_odds
        self.position = position
        self.bookmakers = []
    
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American format"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage"""
        return {
            'timestamp': self.timestamp,
            'event_name': self.event_name,
            'fight_key': self.fight_key,
            'fighter': self.fighter,
            'opponent': self.opponent,
            'decimal_odds': self.decimal_odds,
            'american_odds': self.american_odds,
            'implied_probability': self.implied_probability,
            'bookmakers': ', '.join(self.bookmakers),
            'position': self.position
        }


class OddsResult:
    """Result of odds fetching operation"""
    
    def __init__(self, event_name: str):
        self.event_name = event_name
        self.status = 'pending'
        self.odds_data: Dict[str, Dict] = {}
        self.csv_path: Optional[str] = None
        self.total_fights = 0
        self.fetch_timestamp: Optional[str] = None
        self.error: Optional[str] = None
    
    def mark_success(self, odds_data: Dict, csv_path: str):
        """Mark operation as successful"""
        self.status = 'success'
        self.odds_data = odds_data
        self.csv_path = csv_path
        self.total_fights = len(odds_data)
        self.fetch_timestamp = datetime.now().isoformat()
    
    def mark_failure(self, error: str):
        """Mark operation as failed"""
        self.status = 'failed'
        self.error = error


class UFCOddsService:
    """
    Professional UFC odds service with CSV storage and analysis capabilities
    
    Extracted from notebook Cell 2 for production use in automated agent.
    """
    
    def __init__(self, odds_client, storage_base_path: str = 'odds'):
        """
        Initialize odds service
        
        Args:
            odds_client: UFCOddsAPIClient instance
            storage_base_path: Base directory for odds storage
        """
        self.odds_client = odds_client
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(exist_ok=True)
    
    def fetch_and_store_odds(self, event_name: str, 
                           target_fights: Optional[List[str]] = None) -> OddsResult:
        """
        Fetch live odds and store in organized CSV format
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights to fetch
            
        Returns:
            OddsResult: Complete result of the odds fetching operation
        """
        logger.info(f"Fetching odds for event: {event_name}")
        
        result = OddsResult(event_name)
        timestamp = datetime.now()
        
        try:
            # Fetch odds from The Odds API
            logger.info("Fetching odds from The Odds API")
            api_data = self.odds_client.get_ufc_odds(region="au")
            
            # Format for target fights
            formatted_odds = self.odds_client.format_odds_for_analysis(
                api_data, target_fights
            )
            
            if not formatted_odds:
                error_msg = (
                    f"No odds found for target fights. "
                    f"Available events: {len(api_data)}. "
                    f"Target fights: {target_fights}"
                )
                logger.warning(error_msg)
                result.mark_failure(error_msg)
                return result
            
            logger.info(f"Retrieved odds for {len(formatted_odds)} fights")
            
            # Store to CSV
            csv_path = self._store_odds_to_csv(
                formatted_odds, event_name, timestamp
            )
            
            # Mark success
            result.mark_success(formatted_odds, str(csv_path))
            
            logger.info(f"Odds fetching completed successfully: {csv_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"Odds fetching failed: {str(e)}"
            logger.error(error_msg)
            result.mark_failure(error_msg)
            return result
    
    def _store_odds_to_csv(self, odds_data: Dict[str, Dict], 
                          event_name: str, timestamp: datetime) -> Path:
        """
        Store odds data to CSV with organized folder structure
        
        Args:
            odds_data: Formatted odds data
            event_name: Name of the event
            timestamp: Timestamp of the fetch
            
        Returns:
            Path: Path to the created CSV file
        """
        # Create event-specific folder
        event_folder = self.storage_base_path / event_name
        event_folder.mkdir(exist_ok=True)
        
        # Generate CSV filename with timestamp
        csv_filename = f"odds_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = event_folder / csv_filename
        
        # Prepare odds records
        odds_records = []
        timestamp_str = timestamp.isoformat()
        
        for fight_key, fight_data in odds_data.items():
            # Create record for fighter A
            record_a = OddsRecord(
                timestamp=timestamp_str,
                event_name=event_name,
                fight_key=fight_key,
                fighter=fight_data['fighter_a'],
                opponent=fight_data['fighter_b'],
                decimal_odds=fight_data['fighter_a_decimal_odds'],
                position='fighter_a'
            )
            record_a.bookmakers = fight_data.get('bookmakers', ['Unknown'])
            odds_records.append(record_a.to_dict())
            
            # Create record for fighter B
            record_b = OddsRecord(
                timestamp=timestamp_str,
                event_name=event_name,
                fight_key=fight_key,
                fighter=fight_data['fighter_b'],
                opponent=fight_data['fighter_a'],
                decimal_odds=fight_data['fighter_b_decimal_odds'],
                position='fighter_b'
            )
            record_b.bookmakers = fight_data.get('bookmakers', ['Unknown'])
            odds_records.append(record_b.to_dict())
        
        # Save to CSV
        df_odds = pd.DataFrame(odds_records)
        df_odds.to_csv(csv_path, index=False)
        
        logger.info(f"Odds saved to CSV: {csv_path}")
        
        return csv_path
    
    def get_odds_summary(self, odds_data: Dict[str, Dict]) -> str:
        """
        Generate formatted odds summary for display
        
        Args:
            odds_data: Formatted odds data
            
        Returns:
            str: Formatted summary text
        """
        output = [
            f"📋 ODDS SUMMARY",
            f"=" * 30
        ]
        
        for fight_key, fight_data in odds_data.items():
            fighter_a = fight_data['fighter_a']
            fighter_b = fight_data['fighter_b']
            odds_a = fight_data['fighter_a_decimal_odds']
            odds_b = fight_data['fighter_b_decimal_odds']
            
            # Identify favorite
            if odds_a < odds_b:
                favorite = f"⭐ {fighter_a}"
                underdog = fighter_b
                fav_odds, dog_odds = odds_a, odds_b
            else:
                favorite = f"⭐ {fighter_b}"
                underdog = fighter_a
                fav_odds, dog_odds = odds_b, odds_a
            
            output.append(f"\n🥊 {fight_key}")
            output.append(f"   {favorite} ({fav_odds:.2f}) vs {underdog} ({dog_odds:.2f})")
            output.append(f"   Market edge: {abs(odds_a - odds_b):.2f}")
        
        return "\n".join(output)
    
    def check_api_usage(self) -> Dict[str, Any]:
        """
        Check current API usage status
        
        Returns:
            Dict: API usage information
        """
        try:
            # Make a quick API call to check usage
            response = self.odds_client.get_ufc_odds(region="au")
            
            return {
                'status': 'active',
                'available_events': len(response) if response else 0,
                'connection': True
            }
        except Exception as e:
            logger.warning(f"API usage check failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'connection': False
            }
    
    def get_latest_odds_for_event(self, event_name: str) -> Optional[Tuple[Dict, str]]:
        """
        Get the latest stored odds for an event
        
        Args:
            event_name: Name of the event
            
        Returns:
            Tuple of (odds_data, csv_path) or None if not found
        """
        event_folder = self.storage_base_path / event_name
        
        if not event_folder.exists():
            logger.warning(f"No odds folder found for event: {event_name}")
            return None
        
        # Find latest CSV file
        csv_files = list(event_folder.glob("odds_*.csv"))
        if not csv_files:
            logger.warning(f"No odds files found for event: {event_name}")
            return None
        
        latest_csv = sorted(csv_files, key=lambda x: x.name)[-1]
        
        try:
            # Load and reconstruct odds data
            df = pd.read_csv(latest_csv)
            odds_data = self._reconstruct_odds_data(df)
            
            logger.info(f"Loaded latest odds from: {latest_csv}")
            return odds_data, str(latest_csv)
            
        except Exception as e:
            logger.error(f"Error loading odds from {latest_csv}: {str(e)}")
            return None
    
    def _reconstruct_odds_data(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Reconstruct odds data from CSV DataFrame
        
        Args:
            df: DataFrame loaded from CSV
            
        Returns:
            Dict: Reconstructed odds data in original format
        """
        odds_data = {}
        
        # Group by fight_key to reconstruct fight data
        for fight_key, group in df.groupby('fight_key'):
            fighter_a_row = group[group['position'] == 'fighter_a'].iloc[0]
            fighter_b_row = group[group['position'] == 'fighter_b'].iloc[0]
            
            odds_data[fight_key] = {
                'fighter_a': fighter_a_row['fighter'],
                'fighter_b': fighter_b_row['fighter'],
                'fighter_a_decimal_odds': fighter_a_row['decimal_odds'],
                'fighter_b_decimal_odds': fighter_b_row['decimal_odds'],
                'bookmakers': fighter_a_row['bookmakers'].split(', ') if pd.notna(fighter_a_row['bookmakers']) else []
            }
        
        return odds_data
    
    def monitor_odds_changes(self, event_name: str, 
                           threshold_change: float = 0.1) -> Optional[Dict[str, List]]:
        """
        Monitor odds changes for an event
        
        Args:
            event_name: Name of the event
            threshold_change: Minimum change to report (decimal odds)
            
        Returns:
            Dict with 'significant_changes' and 'new_opportunities' or None
        """
        event_folder = self.storage_base_path / event_name
        
        if not event_folder.exists():
            return None
        
        # Get all CSV files for the event
        csv_files = sorted(list(event_folder.glob("odds_*.csv")), key=lambda x: x.name)
        
        if len(csv_files) < 2:
            logger.info(f"Not enough historical data to monitor changes for {event_name}")
            return None
        
        try:
            # Load latest two files
            df_current = pd.read_csv(csv_files[-1])
            df_previous = pd.read_csv(csv_files[-2])
            
            significant_changes = []
            new_opportunities = []
            
            # Compare odds for each fighter
            for _, current_row in df_current.iterrows():
                fighter = current_row['fighter']
                fight_key = current_row['fight_key']
                current_odds = current_row['decimal_odds']
                
                # Find corresponding previous row
                previous_match = df_previous[
                    (df_previous['fighter'] == fighter) & 
                    (df_previous['fight_key'] == fight_key)
                ]
                
                if not previous_match.empty:
                    previous_odds = previous_match.iloc[0]['decimal_odds']
                    change = abs(current_odds - previous_odds)
                    
                    if change >= threshold_change:
                        change_data = {
                            'fighter': fighter,
                            'fight_key': fight_key,
                            'previous_odds': previous_odds,
                            'current_odds': current_odds,
                            'change': change,
                            'change_percent': (change / previous_odds) * 100
                        }
                        significant_changes.append(change_data)
                        
                        # Check if this creates a new opportunity
                        if current_odds > previous_odds:  # Odds got worse for fighter
                            new_opportunities.append(change_data)
            
            return {
                'significant_changes': significant_changes,
                'new_opportunities': new_opportunities
            }
            
        except Exception as e:
            logger.error(f"Error monitoring odds changes: {str(e)}")
            return None