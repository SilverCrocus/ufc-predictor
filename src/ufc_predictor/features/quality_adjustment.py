"""
Opponent quality adjustment features for UFC predictions.
Normalizes fighter statistics based on opponent strength.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ..models.bradley_terry import BradleyTerryModel, EloRatingSystem
import logging

logger = logging.getLogger(__name__)


class QualityAdjustmentFeatureGenerator:
    """
    Adjusts fighter statistics based on opponent quality.
    Uses Bradley-Terry or Elo ratings to normalize performance metrics.
    """
    
    # Statistics to adjust for opponent quality
    ADJUSTABLE_STATS = [
        'strikes_landed_per_min',
        'strikes_absorbed_per_min',
        'takedowns_per_15min',
        'takedown_defense',
        'submission_attempts_per_15min',
        'significant_strikes_accuracy',
        'significant_strikes_defense'
    ]
    
    def __init__(
        self,
        rating_system: str = 'bradley_terry',
        adjustment_method: str = 'multiplicative',
        min_fights_for_adjustment: int = 3
    ):
        """
        Initialize quality adjustment generator.
        
        Args:
            rating_system: 'bradley_terry' or 'elo'
            adjustment_method: 'multiplicative' or 'additive'
            min_fights_for_adjustment: Minimum fights needed for adjustment
        """
        self.rating_system = rating_system
        self.adjustment_method = adjustment_method
        self.min_fights_for_adjustment = min_fights_for_adjustment
        self.ratings_model = None
        self.opponent_strengths: Dict[str, float] = {}
    
    def fit_ratings(self, historical_fights: pd.DataFrame):
        """
        Fit rating model on historical fight data.
        
        Args:
            historical_fights: DataFrame with fight history
        """
        if self.rating_system == 'bradley_terry':
            self.ratings_model = BradleyTerryModel(regularization=0.01)
            self.ratings_model.fit(historical_fights)
            self.opponent_strengths = self.ratings_model.ratings
        elif self.rating_system == 'elo':
            self.ratings_model = EloRatingSystem()
            # Process fights chronologically for Elo
            historical_fights = historical_fights.sort_values('date')
            for _, fight in historical_fights.iterrows():
                self.ratings_model.update_ratings(
                    fight['fighter_a'],
                    fight['fighter_b'],
                    int(fight['winner']),
                    fight.get('date')
                )
            self.opponent_strengths = self.ratings_model.ratings
        else:
            raise ValueError(f"Unknown rating system: {self.rating_system}")
        
        logger.info(f"Fitted {self.rating_system} ratings for {len(self.opponent_strengths)} fighters")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        fighter_stats_df: pd.DataFrame,
        fighter_col: str = 'fighter',
        stats_to_adjust: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate quality-adjusted features.
        
        Args:
            df: DataFrame with fight data
            fighter_stats_df: DataFrame with fighter statistics
            fighter_col: Column name for fighter
            stats_to_adjust: List of statistics to adjust (None = use defaults)
            
        Returns:
            DataFrame with quality-adjusted features
        """
        if self.ratings_model is None:
            raise ValueError("Must fit ratings model first")
        
        df = df.copy()
        stats_to_adjust = stats_to_adjust or self.ADJUSTABLE_STATS
        
        # Calculate average opponent strength for each fighter
        df = self._add_opponent_strength_features(df, fighter_col)
        
        # Adjust each statistic
        for stat in stats_to_adjust:
            if stat in fighter_stats_df.columns:
                df = self._adjust_statistic(df, fighter_stats_df, stat, fighter_col)
        
        # Add rating-based features
        df = self._add_rating_features(df, fighter_col)
        
        return df
    
    def _add_opponent_strength_features(
        self,
        df: pd.DataFrame,
        fighter_col: str
    ) -> pd.DataFrame:
        """Add features related to opponent strength."""
        # Average opponent rating
        df[f'{fighter_col}_avg_opponent_rating'] = df[fighter_col].map(
            self._get_average_opponent_rating
        )
        
        # Strength of schedule (variance in opponent ratings)
        df[f'{fighter_col}_opponent_rating_variance'] = df[fighter_col].map(
            self._get_opponent_rating_variance
        )
        
        # Quality of wins/losses
        df[f'{fighter_col}_quality_of_wins'] = df[fighter_col].map(
            self._get_quality_of_wins
        )
        
        df[f'{fighter_col}_quality_of_losses'] = df[fighter_col].map(
            self._get_quality_of_losses
        )
        
        return df
    
    def _adjust_statistic(
        self,
        df: pd.DataFrame,
        fighter_stats_df: pd.DataFrame,
        stat: str,
        fighter_col: str
    ) -> pd.DataFrame:
        """
        Adjust a single statistic for opponent quality.
        
        The adjustment formula:
        - Multiplicative: adjusted = raw * (1 + opponent_strength_factor)
        - Additive: adjusted = raw + opponent_strength_factor
        """
        raw_stat_col = f'{fighter_col}_{stat}'
        adjusted_stat_col = f'{fighter_col}_{stat}_adj'
        
        if raw_stat_col not in fighter_stats_df.columns:
            logger.debug(f"Statistic {raw_stat_col} not found, skipping")
            return df
        
        # Get raw statistics
        df = df.merge(
            fighter_stats_df[[fighter_col, raw_stat_col]],
            on=fighter_col,
            how='left',
            suffixes=('', '_raw')
        )
        
        # Calculate adjustment factor based on opponent strength
        avg_opponent_rating = df[f'{fighter_col}_avg_opponent_rating'].fillna(0)
        
        # Normalize opponent rating to adjustment factor
        # Stronger opponents -> stats worth more
        # Weaker opponents -> stats worth less
        adjustment_factor = self._calculate_adjustment_factor(avg_opponent_rating)
        
        # Apply adjustment
        if self.adjustment_method == 'multiplicative':
            df[adjusted_stat_col] = df[raw_stat_col] * (1 + adjustment_factor)
        else:  # additive
            df[adjusted_stat_col] = df[raw_stat_col] + adjustment_factor
        
        # Keep both raw and adjusted versions
        df[f'{fighter_col}_{stat}_adjustment'] = adjustment_factor
        
        return df
    
    def _calculate_adjustment_factor(self, opponent_ratings: pd.Series) -> pd.Series:
        """
        Calculate adjustment factor from opponent ratings.
        
        Returns values typically in range [-0.2, 0.2] for multiplicative adjustment.
        """
        # Normalize ratings (assume mean=0, std=1 for fitted ratings)
        if len(opponent_ratings) == 0:
            return pd.Series([0])
        
        # Scale to reasonable adjustment range
        # Facing top opponents -> positive adjustment (up to 20%)
        # Facing weak opponents -> negative adjustment (down to -20%)
        adjustment = opponent_ratings / 5  # Scale factor
        adjustment = adjustment.clip(-0.2, 0.2)  # Limit adjustment range
        
        return adjustment
    
    def _get_average_opponent_rating(self, fighter: str) -> float:
        """Get average rating of fighter's past opponents."""
        # This would be calculated from historical fight data
        # For now, return a placeholder
        return self.opponent_strengths.get(fighter, 0.0)
    
    def _get_opponent_rating_variance(self, fighter: str) -> float:
        """Get variance in ratings of fighter's past opponents."""
        # Placeholder - would calculate from historical data
        return 0.5
    
    def _get_quality_of_wins(self, fighter: str) -> float:
        """Get average rating of opponents fighter has beaten."""
        # Placeholder - would calculate from historical data
        return 0.0
    
    def _get_quality_of_losses(self, fighter: str) -> float:
        """Get average rating of opponents fighter has lost to."""
        # Placeholder - would calculate from historical data
        return 0.0
    
    def _add_rating_features(self, df: pd.DataFrame, fighter_col: str) -> pd.DataFrame:
        """Add features based on fighter ratings."""
        # Direct fighter rating
        df[f'{fighter_col}_rating'] = df[fighter_col].map(
            lambda x: self.opponent_strengths.get(x, 0.0)
        )
        
        # Rating percentile
        all_ratings = list(self.opponent_strengths.values())
        df[f'{fighter_col}_rating_percentile'] = df[f'{fighter_col}_rating'].apply(
            lambda x: self._get_percentile(x, all_ratings)
        )
        
        # Rating tier (categorical)
        df[f'{fighter_col}_rating_tier'] = pd.cut(
            df[f'{fighter_col}_rating_percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['bottom', 'low', 'mid', 'high', 'elite']
        )
        
        return df
    
    def _get_percentile(self, value: float, values: List[float]) -> float:
        """Get percentile rank of value in list."""
        if not values:
            return 0.5
        return sum(v <= value for v in values) / len(values)
    
    def create_differential_quality_features(
        self,
        df: pd.DataFrame,
        fighter_a_col: str = 'fighter_a',
        fighter_b_col: str = 'fighter_b'
    ) -> pd.DataFrame:
        """
        Create differential features between two fighters' quality metrics.
        
        Args:
            df: DataFrame with fight data
            fighter_a_col: Column for fighter A
            fighter_b_col: Column for fighter B
            
        Returns:
            DataFrame with differential quality features
        """
        # Rating difference
        df['rating_diff'] = (
            df[f'{fighter_a_col}_rating'] - df[f'{fighter_b_col}_rating']
        )
        
        # Average opponent strength difference
        df['avg_opponent_diff'] = (
            df[f'{fighter_a_col}_avg_opponent_rating'] - 
            df[f'{fighter_b_col}_avg_opponent_rating']
        )
        
        # Quality of wins difference
        df['quality_wins_diff'] = (
            df[f'{fighter_a_col}_quality_of_wins'] - 
            df[f'{fighter_b_col}_quality_of_wins']
        )
        
        # Tier mismatch indicator
        df['tier_mismatch'] = (
            df[f'{fighter_a_col}_rating_tier'] != df[f'{fighter_b_col}_rating_tier']
        ).astype(int)
        
        # Elite vs non-elite
        df['elite_vs_non_elite'] = (
            (df[f'{fighter_a_col}_rating_tier'] == 'elite') & 
            (df[f'{fighter_b_col}_rating_tier'] != 'elite')
        ).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created by this generator."""
        base_features = []
        
        # Per-fighter features
        for prefix in ['fighter_a', 'fighter_b']:
            base_features.extend([
                f'{prefix}_rating',
                f'{prefix}_rating_percentile',
                f'{prefix}_rating_tier',
                f'{prefix}_avg_opponent_rating',
                f'{prefix}_opponent_rating_variance',
                f'{prefix}_quality_of_wins',
                f'{prefix}_quality_of_losses'
            ])
            
            # Adjusted statistics
            for stat in self.ADJUSTABLE_STATS:
                base_features.extend([
                    f'{prefix}_{stat}_adj',
                    f'{prefix}_{stat}_adjustment'
                ])
        
        # Differential features
        base_features.extend([
            'rating_diff',
            'avg_opponent_diff',
            'quality_wins_diff',
            'tier_mismatch',
            'elite_vs_non_elite'
        ])
        
        return base_features
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate quality-adjusted features."""
        validation_results = {
            'n_fighters_with_ratings': len(self.opponent_strengths),
            'rating_range': (
                min(self.opponent_strengths.values()),
                max(self.opponent_strengths.values())
            ) if self.opponent_strengths else (None, None),
            'features_created': len([c for c in df.columns if '_adj' in c or 'rating' in c]),
            'missing_ratings': df[[c for c in df.columns if 'rating' in c]].isnull().sum().to_dict()
        }
        
        return validation_results