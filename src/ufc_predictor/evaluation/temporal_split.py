"""
Temporal walk-forward cross-validation for UFC fight predictions.
Ensures no data leakage and proper time-series evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TemporalFold:
    """Container for temporal fold information."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metadata: Dict[str, Any]


class TemporalWalkForwardSplitter:
    """
    Implements temporal walk-forward cross-validation for UFC fight data.
    Prevents data leakage and handles rematches appropriately.
    """
    
    def __init__(
        self,
        date_col: str = 'date',
        train_years: int = 5,
        test_months: int = 3,
        gap_days: int = 14,
        stratify_cols: Optional[List[str]] = None,
        handle_rematches: bool = True
    ):
        """
        Initialize temporal splitter.
        
        Args:
            date_col: Column name containing fight dates
            train_years: Number of years for training window
            test_months: Number of months for test window
            gap_days: Gap between train and test to prevent leakage
            stratify_cols: Columns to stratify on (e.g., 'division', 'rounds')
            handle_rematches: Whether to keep rematches in same fold
        """
        self.date_col = date_col
        self.train_years = train_years
        self.test_months = test_months
        self.gap_days = gap_days
        self.stratify_cols = stratify_cols or []
        self.handle_rematches = handle_rematches
        
    def make_rolling_folds(
        self, 
        df: pd.DataFrame,
        min_train_samples: int = 100,
        min_test_samples: int = 20
    ) -> List[TemporalFold]:
        """
        Create rolling temporal folds for walk-forward validation.
        
        Args:
            df: DataFrame with fight data
            min_train_samples: Minimum samples required for training fold
            min_test_samples: Minimum samples required for test fold
            
        Returns:
            List of TemporalFold objects
        """
        # Ensure date column is datetime
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        
        # Get date range
        min_date = df[self.date_col].min()
        max_date = df[self.date_col].max()
        
        # Calculate initial training end date
        initial_train_end = min_date + pd.DateOffset(years=self.train_years)
        
        if initial_train_end >= max_date:
            raise ValueError(f"Not enough data for {self.train_years} years of training")
        
        folds = []
        fold_id = 0
        
        # Create rolling windows
        current_train_end = initial_train_end
        
        while current_train_end < max_date:
            # Define test period
            test_start = current_train_end + pd.Timedelta(days=self.gap_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            # Define training period
            train_start = min_date
            train_end = current_train_end
            
            # Get indices
            train_mask = (df[self.date_col] >= train_start) & (df[self.date_col] <= train_end)
            test_mask = (df[self.date_col] >= test_start) & (df[self.date_col] <= test_end)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            # Check minimum sample requirements
            if len(train_indices) < min_train_samples:
                logger.warning(f"Fold {fold_id}: Insufficient training samples ({len(train_indices)})")
                current_train_end = test_end
                continue
                
            if len(test_indices) < min_test_samples:
                logger.warning(f"Fold {fold_id}: Insufficient test samples ({len(test_indices)})")
                break
            
            # Handle rematches if requested
            if self.handle_rematches:
                train_indices, test_indices = self._handle_rematches(df, train_indices, test_indices)
            
            # Calculate fold metadata
            metadata = self._calculate_fold_metadata(df, train_indices, test_indices)
            
            # Create fold
            fold = TemporalFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metadata=metadata
            )
            
            folds.append(fold)
            fold_id += 1
            
            # Move to next window
            current_train_end = test_end
            
        logger.info(f"Created {len(folds)} temporal folds")
        return folds
    
    def _handle_rematches(
        self, 
        df: pd.DataFrame, 
        train_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust fold indices to handle rematches appropriately.
        Ensures that same fighter pairs don't appear in both train and test.
        """
        # Get fighter pairs in test set
        test_df = df.iloc[test_indices]
        test_pairs = set()
        
        for _, row in test_df.iterrows():
            # Add both orderings to handle symmetric matchups
            if 'fighter_a' in row and 'fighter_b' in row:
                test_pairs.add((row['fighter_a'], row['fighter_b']))
                test_pairs.add((row['fighter_b'], row['fighter_a']))
            elif 'fighter1' in row and 'fighter2' in row:
                test_pairs.add((row['fighter1'], row['fighter2']))
                test_pairs.add((row['fighter2'], row['fighter1']))
        
        # Filter training indices to exclude rematches
        clean_train_indices = []
        train_df = df.iloc[train_indices]
        
        for idx, row in train_df.iterrows():
            if 'fighter_a' in row and 'fighter_b' in row:
                pair = (row['fighter_a'], row['fighter_b'])
            elif 'fighter1' in row and 'fighter2' in row:
                pair = (row['fighter1'], row['fighter2'])
            else:
                clean_train_indices.append(idx)
                continue
                
            if pair not in test_pairs:
                clean_train_indices.append(idx)
            else:
                logger.debug(f"Removing rematch from training: {pair}")
        
        return np.array(clean_train_indices), test_indices
    
    def _calculate_fold_metadata(
        self, 
        df: pd.DataFrame, 
        train_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate metadata for a fold."""
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
        
        metadata = {
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_events': train_df['event_id'].nunique() if 'event_id' in train_df else None,
            'test_events': test_df['event_id'].nunique() if 'event_id' in test_df else None,
        }
        
        # Add stratification statistics if columns exist
        for col in self.stratify_cols:
            if col in df.columns:
                metadata[f'train_{col}_dist'] = train_df[col].value_counts().to_dict()
                metadata[f'test_{col}_dist'] = test_df[col].value_counts().to_dict()
        
        return metadata
    
    def make_expanding_folds(
        self, 
        df: pd.DataFrame,
        test_months: int = 3,
        step_months: int = 1
    ) -> List[TemporalFold]:
        """
        Create expanding window folds where training data grows over time.
        Better for capturing long-term fighter evolution.
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        
        min_date = df[self.date_col].min()
        max_date = df[self.date_col].max()
        
        # Start with initial training period
        initial_train_end = min_date + pd.DateOffset(years=self.train_years)
        
        folds = []
        fold_id = 0
        current_test_start = initial_train_end + pd.Timedelta(days=self.gap_days)
        
        while current_test_start < max_date:
            test_end = current_test_start + pd.DateOffset(months=test_months)
            train_end = current_test_start - pd.Timedelta(days=self.gap_days)
            
            # Expanding window: always start from beginning
            train_mask = (df[self.date_col] >= min_date) & (df[self.date_col] <= train_end)
            test_mask = (df[self.date_col] >= current_test_start) & (df[self.date_col] <= test_end)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                if self.handle_rematches:
                    train_indices, test_indices = self._handle_rematches(df, train_indices, test_indices)
                
                metadata = self._calculate_fold_metadata(df, train_indices, test_indices)
                
                fold = TemporalFold(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=min_date,
                    train_end=train_end,
                    test_start=current_test_start,
                    test_end=test_end,
                    metadata=metadata
                )
                
                folds.append(fold)
                fold_id += 1
            
            current_test_start += pd.DateOffset(months=step_months)
        
        logger.info(f"Created {len(folds)} expanding window folds")
        return folds
    
    def validate_temporal_integrity(self, folds: List[TemporalFold]) -> bool:
        """
        Validate that folds maintain temporal integrity (no future leakage).
        """
        for fold in folds:
            if fold.train_end >= fold.test_start:
                logger.error(f"Fold {fold.fold_id}: Training end ({fold.train_end}) overlaps with test start ({fold.test_start})")
                return False
            
            gap = (fold.test_start - fold.train_end).days
            if gap < self.gap_days:
                logger.warning(f"Fold {fold.fold_id}: Gap ({gap} days) less than required ({self.gap_days} days)")
        
        return True
    
    def get_fold_summary(self, folds: List[TemporalFold]) -> pd.DataFrame:
        """
        Generate summary DataFrame of all folds.
        """
        summaries = []
        
        for fold in folds:
            summary = {
                'fold_id': fold.fold_id,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_size': fold.metadata['train_size'],
                'test_size': fold.metadata['test_size'],
                'train_events': fold.metadata.get('train_events', 0),
                'test_events': fold.metadata.get('test_events', 0),
                'gap_days': (fold.test_start - fold.train_end).days
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def make_event_based_folds(
    df: pd.DataFrame,
    event_col: str = 'event_id',
    date_col: str = 'date',
    events_per_fold: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create folds based on UFC events rather than time periods.
    Ensures all fights from an event stay together.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get unique events sorted by date
    event_dates = df.groupby(event_col)[date_col].min().sort_values()
    events = event_dates.index.tolist()
    
    folds = []
    
    for i in range(events_per_fold, len(events), events_per_fold):
        train_events = events[:i]
        test_events = events[i:min(i + events_per_fold, len(events))]
        
        train_mask = df[event_col].isin(train_events)
        test_mask = df[event_col].isin(test_events)
        
        train_indices = df[train_mask].index.values
        test_indices = df[test_mask].index.values
        
        if len(train_indices) > 0 and len(test_indices) > 0:
            folds.append((train_indices, test_indices))
    
    logger.info(f"Created {len(folds)} event-based folds")
    return folds