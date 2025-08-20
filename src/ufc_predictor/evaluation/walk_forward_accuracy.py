#!/usr/bin/env python3
"""
Walk-Forward Accuracy Validation
=================================
Tests model accuracy using walk-forward validation on fight outcomes.
NO BETTING ODDS INVOLVED - purely predictive accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardAccuracyValidator:
    """
    Validates model accuracy using walk-forward approach.
    This is ONLY for testing prediction accuracy - no odds involved.
    """
    
    def __init__(self,
                 data_path: str = 'model/ufc_fight_dataset_with_diffs.csv',
                 feature_config_path: str = 'model/optimized/feature_selector_latest.json',
                 train_cutoff: str = '2021-10-01',
                 retrain_months: int = 6):
        """
        Initialize accuracy validator
        
        Args:
            data_path: Path to fight dataset
            feature_config_path: Path to feature configuration  
            train_cutoff: Date to split train/test (80/20 temporal split)
            retrain_months: How often to retrain in walk-forward
        """
        self.data_path = Path(data_path)
        self.feature_config_path = Path(feature_config_path)
        self.train_cutoff = pd.to_datetime(train_cutoff)
        self.retrain_months = retrain_months
        
        self.data = None
        self.features = None
        
    def load_data(self) -> pd.DataFrame:
        """Load fight outcome data (no odds needed)"""
        logger.info(f"Loading fight data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Parse dates
        df['date'] = pd.to_datetime(df['Event'].str.extract(r'(\w+\.?\s+\d{1,2},\s+\d{4})')[0], errors='coerce')
        
        # Handle missing dates
        mask = df['date'].isna()
        if mask.any():
            years = df.loc[mask, 'Event'].str.extract(r'(\d{4})')[0]
            df.loc[mask, 'date'] = pd.to_datetime(years + '-01-01', errors='coerce')
        
        df = df.dropna(subset=['date'])
        df['target'] = (df['Outcome'] == 'win').astype(int)
        df = df.sort_values('date').reset_index(drop=True)
        
        self.data = df
        logger.info(f"Loaded {len(df)} fights from {df['date'].min().date()} to {df['date'].max().date()}")
        return df
    
    def load_features(self) -> List[str]:
        """Load feature configuration"""
        if not self.feature_config_path.exists():
            logger.warning("Feature config not found, using default features")
            return self._get_default_features()
        
        with open(self.feature_config_path, 'r') as f:
            config = json.load(f)
        
        self.features = config['selected_features'][:32]
        logger.info(f"Loaded {len(self.features)} features")
        return self.features
    
    def _get_default_features(self) -> List[str]:
        """Default feature set"""
        return [
            'wins_diff', 'slpm_diff', 'td_def_diff', 'age_diff', 'sapm_diff',
            'losses_diff', 'str_acc_diff', 'str_def_diff', 'red_Wins', 'td_avg_diff',
            'blue_Wins', 'red_SLpM', 'blue_SLpM', 'red_TD Avg.', 'blue_TD Acc.',
            'red_TD Acc.', 'red_SApM', 'td_acc_diff', 'red_TD Def.', 'blue_TD Def.',
            'red_Reach (in)', 'blue_SApM', 'blue_TD Avg.', 'blue_Reach (in)',
            'blue_Age', 'red_Age', 'blue_Str. Acc.', 'reach_(in)_diff',
            'red_Losses', 'red_Str. Acc.', 'blue_Losses', 'sub_avg_diff'
        ]
    
    def run_validation(self) -> Dict:
        """
        Run walk-forward validation on test set.
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            self.load_data()
        if self.features is None:
            self.load_features()
        
        # Split data: 80% train (before cutoff), 20% test (after cutoff)
        train_data = self.data[self.data['date'] < self.train_cutoff]
        test_data = self.data[self.data['date'] >= self.train_cutoff]
        
        logger.info("\n" + "="*70)
        logger.info("WALK-FORWARD ACCURACY VALIDATION")
        logger.info("="*70)
        logger.info(f"Training Set: {len(train_data)} fights before {self.train_cutoff.date()}")
        logger.info(f"Test Set: {len(test_data)} fights after {self.train_cutoff.date()}")
        logger.info(f"Split Ratio: {len(train_data)/(len(train_data)+len(test_data))*100:.1f}% / {len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")
        logger.info("-"*70)
        
        # Get available features
        available_features = [f for f in self.features if f in self.data.columns]
        
        # Initial training
        X_train = train_data[available_features]
        y_train = train_data['target']
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Walk forward through test set
        results = []
        current_model = model
        months_since_retrain = 0
        
        test_start = test_data['date'].min()
        test_end = test_data['date'].max()
        current_date = test_start
        
        fold = 1
        all_predictions = []
        all_actuals = []
        
        while current_date < test_end:
            # Get next period's data
            period_end = current_date + pd.DateOffset(months=3)
            period_mask = (test_data['date'] >= current_date) & (test_data['date'] < period_end)
            period_data = test_data[period_mask]
            
            if len(period_data) == 0:
                current_date = period_end
                continue
            
            # Retrain if needed
            if months_since_retrain >= self.retrain_months:
                # Retrain on all data up to current date
                retrain_data = self.data[self.data['date'] < current_date]
                if len(retrain_data) > len(train_data):
                    logger.info(f"\n📊 Fold {fold}: Retraining with {len(retrain_data)} samples")
                    X_retrain = retrain_data[available_features]
                    y_retrain = retrain_data['target']
                    current_model.fit(X_retrain, y_retrain)
                    months_since_retrain = 0
            
            # Test on period
            X_test = period_data[available_features]
            y_test = period_data['target']
            
            y_pred = current_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            
            logger.info(f"   Period: {current_date.date()} to {period_end.date()} | Fights: {len(period_data)} | Accuracy: {accuracy:.3f}")
            
            results.append({
                'fold': fold,
                'start_date': current_date,
                'end_date': period_end,
                'n_fights': len(period_data),
                'accuracy': accuracy
            })
            
            # Move forward
            current_date = period_end
            months_since_retrain += 3
            fold += 1
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        
        results_df = pd.DataFrame(results)
        avg_accuracy = results_df['accuracy'].mean()
        
        logger.info(f"\n📊 Model Performance (No Odds):")
        logger.info(f"   Overall Test Accuracy: {overall_accuracy:.3f}")
        logger.info(f"   Average Fold Accuracy: {avg_accuracy:.3f}")
        logger.info(f"   Best Fold: {results_df['accuracy'].max():.3f}")
        logger.info(f"   Worst Fold: {results_df['accuracy'].min():.3f}")
        
        # Temporal stability
        early = results_df.iloc[:len(results_df)//2]['accuracy'].mean()
        late = results_df.iloc[len(results_df)//2:]['accuracy'].mean()
        
        logger.info(f"\n⏱️ Temporal Stability:")
        logger.info(f"   Early Period: {early:.3f}")
        logger.info(f"   Late Period: {late:.3f}")
        logger.info(f"   Degradation: {early - late:+.3f}")
        
        logger.info("\n" + "="*70)
        
        return {
            'overall_accuracy': overall_accuracy,
            'average_fold_accuracy': avg_accuracy,
            'results_df': results_df,
            'n_test_fights': len(test_data),
            'n_train_fights': len(train_data)
        }


def main():
    """Run walk-forward accuracy validation"""
    validator = WalkForwardAccuracyValidator()
    results = validator.run_validation()
    
    # Save results
    results_df = results['results_df']
    output_path = Path('model/walk_forward_accuracy_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n📁 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()