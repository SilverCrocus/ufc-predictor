"""
Rolling and decay-weighted fighter profile features.
Captures recent form and performance trends with temporal weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RollingProfileGenerator:
    """
    Generates rolling and decay-weighted features for fighter profiles.
    Emphasizes recent performance while maintaining historical context.
    """
    
    # Statistics to track in rolling windows
    ROLLING_STATS = [
        'strikes_landed_per_min',
        'strikes_absorbed_per_min',
        'takedowns_per_15min',
        'takedown_accuracy',
        'takedown_defense',
        'significant_strikes_landed',
        'significant_strikes_accuracy',
        'significant_strikes_defense',
        'submission_attempts_per_15min',
        'control_time_pct',
        'knockdowns_landed',
        'reversals'
    ]
    
    # Round-by-round statistics for momentum analysis
    ROUND_STATS = [
        'r1_strikes_landed',
        'r2_strikes_landed',
        'r3_strikes_landed',
        'r1_strikes_absorbed',
        'r2_strikes_absorbed',
        'r3_strikes_absorbed'
    ]
    
    def __init__(
        self,
        windows: List[int] = None,
        decay_factor: float = 0.93,
        min_fights_for_rolling: int = 2
    ):
        """
        Initialize rolling profile generator.
        
        Args:
            windows: List of window sizes (number of fights)
            decay_factor: Exponential decay factor for weighting
            min_fights_for_rolling: Minimum fights needed for rolling stats
        """
        self.windows = windows or [3, 5]
        self.decay_factor = decay_factor
        self.min_fights_for_rolling = min_fights_for_rolling
    
    def generate_features(
        self,
        fighter_df: pd.DataFrame,
        fight_history_df: pd.DataFrame,
        fighter_col: str = 'fighter',
        date_col: str = 'date',
        target_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate rolling profile features for fighters.
        
        Args:
            fighter_df: DataFrame with current fighter data
            fight_history_df: DataFrame with historical fight data
            fighter_col: Column name for fighter
            date_col: Column name for fight date
            target_date: Date to calculate features for (None = latest)
            
        Returns:
            DataFrame with rolling profile features
        """
        df = fighter_df.copy()
        
        # Generate rolling features for each window
        for window in self.windows:
            df = self._add_rolling_features(
                df, fight_history_df, window, fighter_col, date_col, target_date
            )
        
        # Add decay-weighted features
        df = self._add_decay_weighted_features(
            df, fight_history_df, fighter_col, date_col, target_date
        )
        
        # Add momentum features
        df = self._add_momentum_features(
            df, fight_history_df, fighter_col, date_col
        )
        
        # Add activity features
        df = self._add_activity_features(
            df, fight_history_df, fighter_col, date_col, target_date
        )
        
        # Add late-round performance features
        df = self._add_late_round_features(
            df, fight_history_df, fighter_col
        )
        
        return df
    
    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        window: int,
        fighter_col: str,
        date_col: str,
        target_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Add rolling window features."""
        for stat in self.ROLLING_STATS:
            if stat not in history_df.columns:
                continue
            
            # Calculate rolling mean
            rolling_col = f'{fighter_col}_{stat}_roll{window}_mean'
            df[rolling_col] = df[fighter_col].apply(
                lambda x: self._calculate_rolling_stat(
                    x, history_df, stat, window, 'mean', date_col, target_date
                )
            )
            
            # Calculate rolling std (volatility)
            rolling_std_col = f'{fighter_col}_{stat}_roll{window}_std'
            df[rolling_std_col] = df[fighter_col].apply(
                lambda x: self._calculate_rolling_stat(
                    x, history_df, stat, window, 'std', date_col, target_date
                )
            )
            
            # Calculate trend (linear regression slope)
            trend_col = f'{fighter_col}_{stat}_roll{window}_trend'
            df[trend_col] = df[fighter_col].apply(
                lambda x: self._calculate_trend(
                    x, history_df, stat, window, date_col, target_date
                )
            )
        
        return df
    
    def _calculate_rolling_stat(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        stat: str,
        window: int,
        agg_func: str,
        date_col: str,
        target_date: Optional[datetime]
    ) -> float:
        """Calculate rolling statistic for a fighter."""
        # Get fighter's fight history
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].copy()
        
        if len(fighter_fights) < self.min_fights_for_rolling:
            return np.nan
        
        # Sort by date and take last N fights before target date
        fighter_fights = fighter_fights.sort_values(date_col)
        
        if target_date:
            fighter_fights = fighter_fights[fighter_fights[date_col] < target_date]
        
        recent_fights = fighter_fights.tail(window)
        
        # Get statistic values
        stat_values = []
        for _, fight in recent_fights.iterrows():
            if fight['fighter_a'] == fighter:
                stat_values.append(fight.get(f'fighter_a_{stat}', np.nan))
            else:
                stat_values.append(fight.get(f'fighter_b_{stat}', np.nan))
        
        # Remove NaN values
        stat_values = [v for v in stat_values if not pd.isna(v)]
        
        if not stat_values:
            return np.nan
        
        # Apply aggregation function
        if agg_func == 'mean':
            return np.mean(stat_values)
        elif agg_func == 'std':
            return np.std(stat_values) if len(stat_values) > 1 else 0
        elif agg_func == 'max':
            return np.max(stat_values)
        elif agg_func == 'min':
            return np.min(stat_values)
        else:
            return np.nan
    
    def _calculate_trend(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        stat: str,
        window: int,
        date_col: str,
        target_date: Optional[datetime]
    ) -> float:
        """Calculate trend (slope) of statistic over window."""
        # Get fighter's fight history
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].copy()
        
        if len(fighter_fights) < self.min_fights_for_rolling:
            return 0.0
        
        # Sort by date and take last N fights
        fighter_fights = fighter_fights.sort_values(date_col)
        
        if target_date:
            fighter_fights = fighter_fights[fighter_fights[date_col] < target_date]
        
        recent_fights = fighter_fights.tail(window)
        
        # Get statistic values
        stat_values = []
        for _, fight in recent_fights.iterrows():
            if fight['fighter_a'] == fighter:
                stat_values.append(fight.get(f'fighter_a_{stat}', np.nan))
            else:
                stat_values.append(fight.get(f'fighter_b_{stat}', np.nan))
        
        # Remove NaN values
        stat_values = [v for v in stat_values if not pd.isna(v)]
        
        if len(stat_values) < 2:
            return 0.0
        
        # Calculate linear trend
        x = np.arange(len(stat_values))
        try:
            slope, _ = np.polyfit(x, stat_values, 1)
            return slope
        except:
            return 0.0
    
    def _add_decay_weighted_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        fighter_col: str,
        date_col: str,
        target_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Add exponentially decay-weighted features."""
        for stat in self.ROLLING_STATS:
            if stat not in history_df.columns:
                continue
            
            decay_col = f'{fighter_col}_{stat}_decay'
            df[decay_col] = df[fighter_col].apply(
                lambda x: self._calculate_decay_weighted_stat(
                    x, history_df, stat, date_col, target_date
                )
            )
        
        return df
    
    def _calculate_decay_weighted_stat(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        stat: str,
        date_col: str,
        target_date: Optional[datetime]
    ) -> float:
        """Calculate decay-weighted average of statistic."""
        # Get fighter's fight history
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].copy()
        
        if len(fighter_fights) == 0:
            return np.nan
        
        # Sort by date
        fighter_fights = fighter_fights.sort_values(date_col)
        
        if target_date:
            fighter_fights = fighter_fights[fighter_fights[date_col] < target_date]
        
        # Calculate weights based on recency
        weights = []
        values = []
        
        for i, (_, fight) in enumerate(fighter_fights.iterrows()):
            # Weight decays exponentially with each fight
            weight = self.decay_factor ** (len(fighter_fights) - i - 1)
            weights.append(weight)
            
            # Get statistic value
            if fight['fighter_a'] == fighter:
                value = fight.get(f'fighter_a_{stat}', np.nan)
            else:
                value = fight.get(f'fighter_b_{stat}', np.nan)
            
            if not pd.isna(value):
                values.append(value)
            else:
                weights[-1] = 0  # Zero weight for missing values
        
        # Calculate weighted average
        if sum(weights) > 0 and len(values) > 0:
            weighted_avg = np.average(values, weights=[w for w in weights if w > 0])
            return weighted_avg
        else:
            return np.nan
    
    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        fighter_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """Add momentum-based features (win streaks, performance changes)."""
        # Win/loss streak
        df[f'{fighter_col}_win_streak'] = df[fighter_col].apply(
            lambda x: self._calculate_streak(x, history_df, 'win')
        )
        
        df[f'{fighter_col}_loss_streak'] = df[fighter_col].apply(
            lambda x: self._calculate_streak(x, history_df, 'loss')
        )
        
        # Finish streak
        df[f'{fighter_col}_finish_streak'] = df[fighter_col].apply(
            lambda x: self._calculate_finish_streak(x, history_df)
        )
        
        # Recent form (last 3 fights win rate)
        df[f'{fighter_col}_recent_form'] = df[fighter_col].apply(
            lambda x: self._calculate_recent_form(x, history_df, 3)
        )
        
        # Performance volatility
        df[f'{fighter_col}_performance_volatility'] = df[fighter_col].apply(
            lambda x: self._calculate_performance_volatility(x, history_df)
        )
        
        return df
    
    def _calculate_streak(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        streak_type: str
    ) -> int:
        """Calculate current win/loss streak."""
        # Get fighter's fight history
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].sort_values('date', ascending=False)
        
        streak = 0
        for _, fight in fighter_fights.iterrows():
            fighter_won = (
                (fight['fighter_a'] == fighter and fight['winner'] == 1) or
                (fight['fighter_b'] == fighter and fight['winner'] == 0)
            )
            
            if streak_type == 'win' and fighter_won:
                streak += 1
            elif streak_type == 'loss' and not fighter_won:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_finish_streak(self, fighter: str, history_df: pd.DataFrame) -> int:
        """Calculate consecutive finishes (KO/TKO/SUB)."""
        # Placeholder - would calculate from fight history
        return 0
    
    def _calculate_recent_form(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        n_fights: int
    ) -> float:
        """Calculate win rate in last N fights."""
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].sort_values('date', ascending=False).head(n_fights)
        
        if len(fighter_fights) == 0:
            return 0.5
        
        wins = 0
        for _, fight in fighter_fights.iterrows():
            if ((fight['fighter_a'] == fighter and fight['winner'] == 1) or
                (fight['fighter_b'] == fighter and fight['winner'] == 0)):
                wins += 1
        
        return wins / len(fighter_fights)
    
    def _calculate_performance_volatility(
        self,
        fighter: str,
        history_df: pd.DataFrame
    ) -> float:
        """Calculate volatility in fighter's recent performances."""
        # Placeholder - would calculate variance in key metrics
        return 0.5
    
    def _add_activity_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        fighter_col: str,
        date_col: str,
        target_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Add activity-related features (layoffs, fight frequency)."""
        target_date = target_date or datetime.now()
        
        # Days since last fight
        df[f'{fighter_col}_days_since_last_fight'] = df[fighter_col].apply(
            lambda x: self._calculate_layoff(x, history_df, date_col, target_date)
        )
        
        # Fight frequency (fights per year in last 2 years)
        df[f'{fighter_col}_fight_frequency'] = df[fighter_col].apply(
            lambda x: self._calculate_fight_frequency(x, history_df, date_col, target_date)
        )
        
        # Ring rust indicator (>365 days layoff)
        df[f'{fighter_col}_ring_rust'] = (
            df[f'{fighter_col}_days_since_last_fight'] > 365
        ).astype(int)
        
        return df
    
    def _calculate_layoff(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        date_col: str,
        target_date: datetime
    ) -> int:
        """Calculate days since last fight."""
        fighter_fights = history_df[
            (history_df['fighter_a'] == fighter) | 
            (history_df['fighter_b'] == fighter)
        ].sort_values(date_col, ascending=False)
        
        if len(fighter_fights) == 0:
            return 999  # No previous fights
        
        last_fight_date = pd.to_datetime(fighter_fights.iloc[0][date_col])
        return (target_date - last_fight_date).days
    
    def _calculate_fight_frequency(
        self,
        fighter: str,
        history_df: pd.DataFrame,
        date_col: str,
        target_date: datetime
    ) -> float:
        """Calculate fights per year in recent period."""
        two_years_ago = target_date - timedelta(days=730)
        
        fighter_fights = history_df[
            ((history_df['fighter_a'] == fighter) | 
             (history_df['fighter_b'] == fighter)) &
            (pd.to_datetime(history_df[date_col]) > two_years_ago) &
            (pd.to_datetime(history_df[date_col]) < target_date)
        ]
        
        return len(fighter_fights) / 2  # Fights per year
    
    def _add_late_round_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        fighter_col: str
    ) -> pd.DataFrame:
        """Add late-round performance delta features."""
        # Calculate difference between early and late round performance
        df[f'{fighter_col}_late_round_improvement'] = df[fighter_col].apply(
            lambda x: self._calculate_late_round_delta(x, history_df)
        )
        
        # Championship rounds performance (R4-R5)
        df[f'{fighter_col}_championship_rounds_exp'] = df[fighter_col].apply(
            lambda x: self._calculate_championship_experience(x, history_df)
        )
        
        return df
    
    def _calculate_late_round_delta(
        self,
        fighter: str,
        history_df: pd.DataFrame
    ) -> float:
        """Calculate performance difference between R1 and R3."""
        # Placeholder - would calculate from round-by-round stats
        return 0.0
    
    def _calculate_championship_experience(
        self,
        fighter: str,
        history_df: pd.DataFrame
    ) -> int:
        """Count number of championship round experiences."""
        # Placeholder - would count 5-round fights
        return 0
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created by this generator."""
        features = []
        
        # Rolling features for each window and stat
        for window in self.windows:
            for stat in self.ROLLING_STATS:
                features.extend([
                    f'fighter_{stat}_roll{window}_mean',
                    f'fighter_{stat}_roll{window}_std',
                    f'fighter_{stat}_roll{window}_trend'
                ])
        
        # Decay-weighted features
        for stat in self.ROLLING_STATS:
            features.append(f'fighter_{stat}_decay')
        
        # Momentum features
        features.extend([
            'fighter_win_streak',
            'fighter_loss_streak',
            'fighter_finish_streak',
            'fighter_recent_form',
            'fighter_performance_volatility'
        ])
        
        # Activity features
        features.extend([
            'fighter_days_since_last_fight',
            'fighter_fight_frequency',
            'fighter_ring_rust'
        ])
        
        # Late round features
        features.extend([
            'fighter_late_round_improvement',
            'fighter_championship_rounds_exp'
        ])
        
        return features