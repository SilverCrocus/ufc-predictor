"""
Context features for UFC fight predictions.
Includes venue, altitude, short notice, and timezone features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VenueInfo:
    """Container for venue information."""
    name: str
    city: str
    country: str
    altitude_m: float
    timezone: str
    is_apex: bool
    capacity: Optional[int] = None


class ContextFeatureGenerator:
    """
    Generates contextual features for UFC fights.
    """
    
    # Known UFC venues with altitude and characteristics
    VENUE_DATABASE = {
        'UFC APEX': VenueInfo('UFC APEX', 'Las Vegas', 'USA', 610, 'America/Los_Angeles', True, 1000),
        'T-Mobile Arena': VenueInfo('T-Mobile Arena', 'Las Vegas', 'USA', 610, 'America/Los_Angeles', False, 20000),
        'Ball Arena': VenueInfo('Ball Arena', 'Denver', 'USA', 1609, 'America/Denver', False, 19520),
        'Vivint Arena': VenueInfo('Vivint Arena', 'Salt Lake City', 'USA', 1288, 'America/Denver', False, 18306),
        'Arena Ciudad de México': VenueInfo('Arena Ciudad de México', 'Mexico City', 'Mexico', 2240, 'America/Mexico_City', False, 22300),
        'Madison Square Garden': VenueInfo('Madison Square Garden', 'New York', 'USA', 10, 'America/New_York', False, 20789),
        'Barclays Center': VenueInfo('Barclays Center', 'Brooklyn', 'USA', 10, 'America/New_York', False, 19000),
        'American Airlines Center': VenueInfo('American Airlines Center', 'Dallas', 'USA', 131, 'America/Chicago', False, 20000),
        'United Center': VenueInfo('United Center', 'Chicago', 'USA', 181, 'America/Chicago', False, 23500),
        'Honda Center': VenueInfo('Honda Center', 'Anaheim', 'USA', 48, 'America/Los_Angeles', False, 18900),
        'Etihad Arena': VenueInfo('Etihad Arena', 'Abu Dhabi', 'UAE', 5, 'Asia/Dubai', False, 18000),
        'O2 Arena': VenueInfo('O2 Arena', 'London', 'UK', 7, 'Europe/London', False, 20000),
        'RAC Arena': VenueInfo('RAC Arena', 'Perth', 'Australia', 20, 'Australia/Perth', False, 15000),
        'Qudos Bank Arena': VenueInfo('Qudos Bank Arena', 'Sydney', 'Australia', 19, 'Australia/Sydney', False, 21000),
        'Singapore Indoor Stadium': VenueInfo('Singapore Indoor Stadium', 'Singapore', 'Singapore', 15, 'Asia/Singapore', False, 12000),
        'Jeunesse Arena': VenueInfo('Jeunesse Arena', 'Rio de Janeiro', 'Brazil', 5, 'America/Sao_Paulo', False, 15000),
        'Ibirapuera Gymnasium': VenueInfo('Ibirapuera Gymnasium', 'São Paulo', 'Brazil', 760, 'America/Sao_Paulo', False, 10000),
        'Rogers Arena': VenueInfo('Rogers Arena', 'Vancouver', 'Canada', 2, 'America/Vancouver', False, 19700),
        'Scotiabank Arena': VenueInfo('Scotiabank Arena', 'Toronto', 'Canada', 76, 'America/Toronto', False, 19800),
    }
    
    # Fighter home locations/training camps (sample - would be expanded)
    FIGHTER_LOCATIONS = {
        'American Top Team': {'city': 'Coconut Creek', 'country': 'USA', 'timezone': 'America/New_York'},
        'Jackson Wink MMA': {'city': 'Albuquerque', 'country': 'USA', 'timezone': 'America/Denver'},
        'Team Alpha Male': {'city': 'Sacramento', 'country': 'USA', 'timezone': 'America/Los_Angeles'},
        'City Kickboxing': {'city': 'Auckland', 'country': 'New Zealand', 'timezone': 'Pacific/Auckland'},
        'Tiger Muay Thai': {'city': 'Phuket', 'country': 'Thailand', 'timezone': 'Asia/Bangkok'},
        'Chute Boxe': {'city': 'Curitiba', 'country': 'Brazil', 'timezone': 'America/Sao_Paulo'},
        'Kings MMA': {'city': 'Huntington Beach', 'country': 'USA', 'timezone': 'America/Los_Angeles'},
    }
    
    def __init__(
        self,
        altitude_threshold_m: float = 1500,
        short_notice_days: int = 14,
        enable_all: bool = True
    ):
        """
        Initialize context feature generator.
        
        Args:
            altitude_threshold_m: Threshold for high altitude venues (meters)
            short_notice_days: Days threshold for short notice fights
            enable_all: Whether to generate all features
        """
        self.altitude_threshold = altitude_threshold_m
        self.short_notice_days = short_notice_days
        self.enable_all = enable_all
    
    def generate_features(
        self,
        df: pd.DataFrame,
        venue_col: str = 'venue',
        date_col: str = 'date',
        announcement_col: Optional[str] = None,
        fighter_camp_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate all context features.
        
        Args:
            df: DataFrame with fight data
            venue_col: Column containing venue names
            date_col: Column containing fight dates
            announcement_col: Optional column with fight announcement dates
            fighter_camp_col: Optional column with fighter camp/team info
            
        Returns:
            DataFrame with new context features added
        """
        df = df.copy()
        
        # Venue-based features
        df = self._add_venue_features(df, venue_col)
        
        # Short notice features
        if announcement_col and announcement_col in df.columns:
            df = self._add_short_notice_features(df, date_col, announcement_col)
        else:
            df['is_short_notice'] = False
            df['notice_days'] = np.nan
        
        # Timezone features
        if fighter_camp_col and fighter_camp_col in df.columns:
            df = self._add_timezone_features(df, venue_col, fighter_camp_col)
        else:
            df['time_zone_delta_hours'] = 0
        
        # Event type features
        df = self._add_event_type_features(df)
        
        return df
    
    def _add_venue_features(self, df: pd.DataFrame, venue_col: str) -> pd.DataFrame:
        """Add venue-related features."""
        # Initialize features
        df['is_apex_small_cage'] = False
        df['is_altitude_high'] = False
        df['venue_altitude_m'] = 0
        df['venue_capacity'] = np.nan
        
        for venue_name, venue_info in self.VENUE_DATABASE.items():
            venue_mask = df[venue_col].str.contains(venue_name, case=False, na=False)
            
            if venue_mask.any():
                df.loc[venue_mask, 'is_apex_small_cage'] = venue_info.is_apex
                df.loc[venue_mask, 'is_altitude_high'] = venue_info.altitude_m >= self.altitude_threshold
                df.loc[venue_mask, 'venue_altitude_m'] = venue_info.altitude_m
                if venue_info.capacity:
                    df.loc[venue_mask, 'venue_capacity'] = venue_info.capacity
        
        # Log unknown venues
        known_venues = df['venue_altitude_m'] > 0
        unknown_venues = ~known_venues & df[venue_col].notna()
        if unknown_venues.any():
            unique_unknown = df.loc[unknown_venues, venue_col].unique()
            logger.debug(f"Unknown venues encountered: {unique_unknown[:5]}")
        
        return df
    
    def _add_short_notice_features(
        self,
        df: pd.DataFrame,
        date_col: str,
        announcement_col: str
    ) -> pd.DataFrame:
        """Add short notice fight features."""
        # Ensure datetime format
        df[date_col] = pd.to_datetime(df[date_col])
        df[announcement_col] = pd.to_datetime(df[announcement_col])
        
        # Calculate notice period
        df['notice_days'] = (df[date_col] - df[announcement_col]).dt.days
        
        # Short notice flag
        df['is_short_notice'] = df['notice_days'] <= self.short_notice_days
        
        # Additional short notice categories
        df['notice_category'] = pd.cut(
            df['notice_days'],
            bins=[-np.inf, 7, 14, 30, 60, np.inf],
            labels=['very_short', 'short', 'normal', 'standard', 'long']
        )
        
        # Handle missing values
        df['is_short_notice'] = df['is_short_notice'].fillna(False)
        
        return df
    
    def _add_timezone_features(
        self,
        df: pd.DataFrame,
        venue_col: str,
        fighter_camp_col: str
    ) -> pd.DataFrame:
        """Add timezone-related features."""
        # This is simplified - in production would use actual timezone calculations
        df['time_zone_delta_hours'] = 0
        
        # Map venues to timezones
        venue_timezones = {}
        for venue_name, venue_info in self.VENUE_DATABASE.items():
            venue_timezones[venue_name] = venue_info.timezone
        
        # Calculate timezone differences (simplified)
        # In production, would use pytz or similar for accurate calculations
        timezone_offsets = {
            'America/Los_Angeles': -8,
            'America/Denver': -7,
            'America/Chicago': -6,
            'America/New_York': -5,
            'America/Sao_Paulo': -3,
            'Europe/London': 0,
            'Asia/Dubai': 4,
            'Asia/Bangkok': 7,
            'Asia/Singapore': 8,
            'Australia/Perth': 8,
            'Australia/Sydney': 10,
            'Pacific/Auckland': 12,
        }
        
        # Add timezone delta calculation here if camp data available
        # This is a placeholder for the actual implementation
        
        return df
    
    def _add_event_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event type features."""
        # Title fight indicator
        if 'is_title_fight' not in df.columns:
            # Simple heuristic - would be better with actual data
            df['is_title_fight'] = df.get('rounds', 3) == 5
        
        # Main event indicator
        if 'is_main_event' not in df.columns:
            df['is_main_event'] = df.get('fight_number', 0) == df.groupby('event_id')['fight_number'].transform('max')
        
        # Five round non-title fight
        df['is_5_round_main'] = (df.get('rounds', 3) == 5) & (~df.get('is_title_fight', False))
        
        # Fight Night vs PPV (simplified heuristic)
        if 'event_name' in df.columns:
            df['is_ppv'] = df['event_name'].str.contains(r'UFC \d{3}', case=False, na=False)
            df['is_fight_night'] = df['event_name'].str.contains('Fight Night', case=False, na=False)
        else:
            df['is_ppv'] = False
            df['is_fight_night'] = False
        
        return df
    
    def create_differential_context_features(
        self,
        df: pd.DataFrame,
        fighter_a_cols: List[str],
        fighter_b_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create differential features between fighters for context.
        
        Args:
            df: DataFrame with fight data
            fighter_a_cols: Columns for fighter A
            fighter_b_cols: Columns for fighter B
            
        Returns:
            DataFrame with differential context features
        """
        # This would create differential features based on fighter-specific context
        # For example, if one fighter is local and the other is traveling
        
        # Placeholder for differential calculations
        df['home_advantage'] = 0  # Would calculate based on fighter location vs venue
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created by this generator."""
        return [
            'is_apex_small_cage',
            'is_altitude_high',
            'venue_altitude_m',
            'venue_capacity',
            'is_short_notice',
            'notice_days',
            'notice_category',
            'time_zone_delta_hours',
            'is_title_fight',
            'is_main_event',
            'is_5_round_main',
            'is_ppv',
            'is_fight_night',
            'home_advantage'
        ]
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate generated features.
        
        Returns:
            Dictionary with validation results
        """
        feature_names = self.get_feature_names()
        existing_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        validation_results = {
            'n_features_expected': len(feature_names),
            'n_features_found': len(existing_features),
            'missing_features': missing_features,
            'feature_completeness': len(existing_features) / len(feature_names),
        }
        
        # Check for NaN values
        nan_counts = {}
        for feature in existing_features:
            nan_count = df[feature].isna().sum()
            if nan_count > 0:
                nan_counts[feature] = nan_count
        
        validation_results['features_with_nan'] = nan_counts
        
        # Check value distributions
        value_distributions = {}
        for feature in existing_features:
            if df[feature].dtype in ['bool', 'object', 'category']:
                value_distributions[feature] = df[feature].value_counts().to_dict()
        
        validation_results['value_distributions'] = value_distributions
        
        return validation_results