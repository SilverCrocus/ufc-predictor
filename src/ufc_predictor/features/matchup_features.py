"""
Matchup interaction features for UFC fight predictions.
Includes stance combinations, reach/height splines, and weight class changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MatchupFeatureGenerator:
    """
    Generates matchup-specific interaction features for UFC fights.
    """
    
    # Stance interaction matrix
    STANCE_ADVANTAGES = {
        ('Orthodox', 'Orthodox'): 0.0,
        ('Orthodox', 'Southpaw'): 0.1,  # Slight advantage to orthodox (more common)
        ('Orthodox', 'Switch'): -0.05,   # Slight disadvantage vs switch
        ('Southpaw', 'Orthodox'): -0.1,
        ('Southpaw', 'Southpaw'): 0.0,
        ('Southpaw', 'Switch'): -0.05,
        ('Switch', 'Orthodox'): 0.05,
        ('Switch', 'Southpaw'): 0.05,
        ('Switch', 'Switch'): 0.0,
    }
    
    # Weight class ordering for detecting changes
    WEIGHT_CLASSES = [
        'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
        'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight',
        'Heavyweight'
    ]
    
    WOMEN_WEIGHT_CLASSES = [
        "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
        "Women's Featherweight"
    ]
    
    def __init__(
        self,
        reach_bins: List[float] = None,
        height_bins: List[float] = None,
        age_knots: Dict[str, List[float]] = None
    ):
        """
        Initialize matchup feature generator.
        
        Args:
            reach_bins: Bin boundaries for reach differences (cm)
            height_bins: Bin boundaries for height differences (cm)
            age_knots: Division-specific knots for age splines
        """
        self.reach_bins = reach_bins or [3, 7, 12]
        self.height_bins = height_bins or [3, 7, 12]
        self.age_knots = age_knots or {
            'default': [23, 27, 32, 36],
            'heavyweight': [25, 30, 35, 38],
            'flyweight': [22, 26, 30, 34]
        }
    
    def generate_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str = 'fighter_a_',
        fighter_b_prefix: str = 'fighter_b_'
    ) -> pd.DataFrame:
        """
        Generate all matchup interaction features.
        
        Args:
            df: DataFrame with fighter data
            fighter_a_prefix: Prefix for fighter A columns
            fighter_b_prefix: Prefix for fighter B columns
            
        Returns:
            DataFrame with new matchup features added
        """
        df = df.copy()
        
        # Stance combination features
        df = self._add_stance_features(df, fighter_a_prefix, fighter_b_prefix)
        
        # Reach difference features
        df = self._add_reach_features(df, fighter_a_prefix, fighter_b_prefix)
        
        # Height difference features
        df = self._add_height_features(df, fighter_a_prefix, fighter_b_prefix)
        
        # Weight class change features
        df = self._add_weight_class_features(df, fighter_a_prefix, fighter_b_prefix)
        
        # Age curve features
        df = self._add_age_features(df, fighter_a_prefix, fighter_b_prefix)
        
        # Physical advantage composite scores
        df = self._add_composite_physical_features(df)
        
        return df
    
    def _add_stance_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str,
        fighter_b_prefix: str
    ) -> pd.DataFrame:
        """Add stance combination features."""
        stance_a_col = f'{fighter_a_prefix}stance'
        stance_b_col = f'{fighter_b_prefix}stance'
        
        if stance_a_col not in df.columns or stance_b_col not in df.columns:
            logger.warning("Stance columns not found, skipping stance features")
            return df
        
        # Create stance combination string
        df['stance_combo'] = df[stance_a_col].astype(str) + '_vs_' + df[stance_b_col].astype(str)
        
        # One-hot encode stance combinations
        stance_dummies = pd.get_dummies(df['stance_combo'], prefix='stance')
        df = pd.concat([df, stance_dummies], axis=1)
        
        # Add stance advantage score
        df['stance_advantage'] = df.apply(
            lambda row: self._calculate_stance_advantage(
                row[stance_a_col], row[stance_b_col]
            ), axis=1
        )
        
        # Specific stance matchup flags
        df['is_orthodox_vs_southpaw'] = (
            (df[stance_a_col] == 'Orthodox') & (df[stance_b_col] == 'Southpaw')
        ).astype(int)
        
        df['is_southpaw_vs_orthodox'] = (
            (df[stance_a_col] == 'Southpaw') & (df[stance_b_col] == 'Orthodox')
        ).astype(int)
        
        df['has_switch_fighter'] = (
            (df[stance_a_col] == 'Switch') | (df[stance_b_col] == 'Switch')
        ).astype(int)
        
        df['same_stance'] = (df[stance_a_col] == df[stance_b_col]).astype(int)
        
        return df
    
    def _calculate_stance_advantage(self, stance_a: str, stance_b: str) -> float:
        """Calculate stance advantage score."""
        key = (stance_a, stance_b)
        return self.STANCE_ADVANTAGES.get(key, 0.0)
    
    def _add_reach_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str,
        fighter_b_prefix: str
    ) -> pd.DataFrame:
        """Add reach-based features with splines."""
        reach_a_col = f'{fighter_a_prefix}reach'
        reach_b_col = f'{fighter_b_prefix}reach'
        
        if reach_a_col not in df.columns or reach_b_col not in df.columns:
            logger.warning("Reach columns not found, skipping reach features")
            return df
        
        # Calculate reach difference (in cm)
        df['reach_diff_cm'] = df[reach_a_col] - df[reach_b_col]
        
        # Create piecewise linear spline features
        df['reach_diff_spline'] = pd.cut(
            df['reach_diff_cm'].abs(),
            bins=[-np.inf] + self.reach_bins + [np.inf],
            labels=[f'reach_bin_{i}' for i in range(len(self.reach_bins) + 1)]
        )
        
        # One-hot encode reach bins
        reach_dummies = pd.get_dummies(df['reach_diff_spline'], prefix='reach_spline')
        df = pd.concat([df, reach_dummies], axis=1)
        
        # Additional reach features
        df['reach_advantage'] = df['reach_diff_cm'].apply(lambda x: 1 if x > 0 else 0)
        df['significant_reach_advantage'] = (df['reach_diff_cm'].abs() > 7).astype(int)
        df['extreme_reach_advantage'] = (df['reach_diff_cm'].abs() > 12).astype(int)
        
        # Reach ratio
        df['reach_ratio'] = df[reach_a_col] / df[reach_b_col].replace(0, 1)
        
        return df
    
    def _add_height_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str,
        fighter_b_prefix: str
    ) -> pd.DataFrame:
        """Add height-based features with splines."""
        height_a_col = f'{fighter_a_prefix}height'
        height_b_col = f'{fighter_b_prefix}height'
        
        if height_a_col not in df.columns or height_b_col not in df.columns:
            logger.warning("Height columns not found, skipping height features")
            return df
        
        # Calculate height difference (in cm)
        df['height_diff_cm'] = df[height_a_col] - df[height_b_col]
        
        # Create piecewise linear spline features
        df['height_diff_bins'] = pd.cut(
            df['height_diff_cm'].abs(),
            bins=[-np.inf] + self.height_bins + [np.inf],
            labels=[f'height_bin_{i}' for i in range(len(self.height_bins) + 1)]
        )
        
        # One-hot encode height bins
        height_dummies = pd.get_dummies(df['height_diff_bins'], prefix='height_spline')
        df = pd.concat([df, height_dummies], axis=1)
        
        # Additional height features
        df['height_advantage'] = df['height_diff_cm'].apply(lambda x: 1 if x > 0 else 0)
        df['significant_height_advantage'] = (df['height_diff_cm'].abs() > 7).astype(int)
        
        # Height ratio
        df['height_ratio'] = df[height_a_col] / df[height_b_col].replace(0, 1)
        
        return df
    
    def _add_weight_class_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str,
        fighter_b_prefix: str
    ) -> pd.DataFrame:
        """Add weight class change features."""
        # This requires historical fight data to determine if fighters changed weight classes
        # For now, we'll add placeholder columns
        
        df['fighter_a_weight_class_change'] = 'same'  # Would be: 'up', 'down', or 'same'
        df['fighter_b_weight_class_change'] = 'same'
        
        # One-hot encode weight class changes
        for fighter in ['fighter_a', 'fighter_b']:
            col = f'{fighter}_weight_class_change'
            dummies = pd.get_dummies(df[col], prefix=f'{fighter}_weight_change')
            df = pd.concat([df, dummies], axis=1)
        
        # Both fighters changing weight
        df['both_weight_class_change'] = (
            (df['fighter_a_weight_class_change'] != 'same') & 
            (df['fighter_b_weight_class_change'] != 'same')
        ).astype(int)
        
        return df
    
    def _add_age_features(
        self,
        df: pd.DataFrame,
        fighter_a_prefix: str,
        fighter_b_prefix: str
    ) -> pd.DataFrame:
        """Add age-based features with division-specific curves."""
        age_a_col = f'{fighter_a_prefix}age'
        age_b_col = f'{fighter_b_prefix}age'
        
        if age_a_col not in df.columns or age_b_col not in df.columns:
            logger.warning("Age columns not found, skipping age features")
            return df
        
        # Age difference
        df['age_diff'] = df[age_a_col] - df[age_b_col]
        
        # Age curve features (simplified - would use actual splines in production)
        df['fighter_a_age_prime'] = df[age_a_col].apply(self._age_prime_score)
        df['fighter_b_age_prime'] = df[age_b_col].apply(self._age_prime_score)
        df['age_prime_diff'] = df['fighter_a_age_prime'] - df['fighter_b_age_prime']
        
        # Veteran vs prospect
        df['veteran_vs_prospect'] = (
            (df[age_a_col] > 35) & (df[age_b_col] < 28)
        ).astype(int)
        
        df['prospect_vs_veteran'] = (
            (df[age_a_col] < 28) & (df[age_b_col] > 35)
        ).astype(int)
        
        # Both in prime
        df['both_in_prime'] = (
            (df[age_a_col].between(26, 32)) & (df[age_b_col].between(26, 32))
        ).astype(int)
        
        return df
    
    def _age_prime_score(self, age: float, division: str = 'default') -> float:
        """
        Calculate age prime score based on division-specific curves.
        Peak performance typically 27-32, varies by division.
        """
        if pd.isna(age):
            return 0.5
        
        # Simplified scoring - in production would use proper splines
        if division == 'heavyweight':
            # Heavyweights peak later
            if 28 <= age <= 35:
                return 1.0
            elif 25 <= age < 28 or 35 < age <= 38:
                return 0.8
            elif 22 <= age < 25 or 38 < age <= 40:
                return 0.6
            else:
                return 0.4
        else:
            # Standard divisions
            if 26 <= age <= 32:
                return 1.0
            elif 23 <= age < 26 or 32 < age <= 35:
                return 0.8
            elif 20 <= age < 23 or 35 < age <= 38:
                return 0.6
            else:
                return 0.4
    
    def _add_composite_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite physical advantage scores."""
        # Physical advantage composite
        physical_features = []
        
        if 'reach_diff_cm' in df.columns:
            physical_features.append('reach_diff_cm')
        if 'height_diff_cm' in df.columns:
            physical_features.append('height_diff_cm')
        
        if physical_features:
            # Normalize and combine
            scaler = StandardScaler()
            normalized = scaler.fit_transform(df[physical_features].fillna(0))
            df['physical_advantage_score'] = normalized.mean(axis=1)
            
            # Categories
            df['physical_advantage_category'] = pd.cut(
                df['physical_advantage_score'],
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=['disadvantage', 'neutral', 'advantage']
            )
        
        return df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df: DataFrame with features
            feature_pairs: List of (feature1, feature2) tuples to interact
            
        Returns:
            DataFrame with interaction features added
        """
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Difference interaction
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
                
                # Ratio interaction (with protection against division by zero)
                df[f'{feat1}_over_{feat2}'] = df[feat1] / df[feat2].replace(0, 1)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created by this generator."""
        base_features = [
            'stance_combo', 'stance_advantage',
            'is_orthodox_vs_southpaw', 'is_southpaw_vs_orthodox',
            'has_switch_fighter', 'same_stance',
            'reach_diff_cm', 'reach_diff_spline',
            'reach_advantage', 'significant_reach_advantage', 
            'extreme_reach_advantage', 'reach_ratio',
            'height_diff_cm', 'height_diff_bins',
            'height_advantage', 'significant_height_advantage', 'height_ratio',
            'fighter_a_weight_class_change', 'fighter_b_weight_class_change',
            'both_weight_class_change',
            'age_diff', 'fighter_a_age_prime', 'fighter_b_age_prime',
            'age_prime_diff', 'veteran_vs_prospect', 'prospect_vs_veteran',
            'both_in_prime', 'physical_advantage_score', 'physical_advantage_category'
        ]
        
        # Add dynamic features (one-hot encoded)
        # These would be generated based on actual data
        
        return base_features
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""
        validation_results = {
            'total_features': len(df.columns),
            'matchup_features': len([c for c in df.columns if any(
                keyword in c for keyword in ['stance', 'reach', 'height', 'age', 'weight_class']
            )]),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_types': df.dtypes.value_counts().to_dict()
        }
        
        return validation_results