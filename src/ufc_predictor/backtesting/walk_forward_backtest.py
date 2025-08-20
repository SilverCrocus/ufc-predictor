#!/usr/bin/env python3
"""
Walk-Forward Backtesting System
Provides realistic backtesting by retraining models periodically, just like production usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import json
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestFold:
    """Results for a single walk-forward fold"""
    fold_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    train_accuracy: float
    test_accuracy: float
    predictions: pd.DataFrame
    betting_roi: float = 0.0
    n_bets: int = 0


class WalkForwardBacktester:
    """
    Implements walk-forward backtesting for UFC predictions.
    This provides more realistic accuracy estimates by retraining models periodically.
    """
    
    def __init__(self, 
                 data_path: str = 'model/ufc_fight_dataset_with_diffs.csv',
                 feature_config_path: str = 'model/optimized/feature_selector_latest.json',
                 initial_train_years: int = 3,
                 test_window_months: int = 3,
                 retrain_months: int = 6):
        """
        Initialize walk-forward backtester.
        
        Args:
            data_path: Path to fight dataset
            feature_config_path: Path to feature configuration
            initial_train_years: Years of initial training data
            test_window_months: Size of each test window in months
            retrain_months: How often to retrain the model
        """
        self.data_path = Path(data_path)
        self.feature_config_path = Path(feature_config_path)
        self.initial_train_years = initial_train_years
        self.test_window_months = test_window_months
        self.retrain_months = retrain_months
        
        self.data = None
        self.features = None
        self.results = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the fight dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Parse dates from Event column
        df['date'] = pd.to_datetime(df['Event'].str.extract(r'(\w+\.?\s+\d{1,2},\s+\d{4})')[0], errors='coerce')
        
        # Fallback to year extraction if full date parsing fails
        mask = df['date'].isna()
        if mask.any():
            years = df.loc[mask, 'Event'].str.extract(r'(\d{4})')[0]
            df.loc[mask, 'date'] = pd.to_datetime(years + '-01-01', errors='coerce')
        
        # Remove rows without dates
        df = df.dropna(subset=['date'])
        
        # Convert outcome to binary
        df['target'] = (df['Outcome'] == 'win').astype(int)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} fights from {df['date'].min().date()} to {df['date'].max().date()}")
        
        self.data = df
        return df
    
    def load_features(self) -> List[str]:
        """Load feature configuration"""
        if not self.feature_config_path.exists():
            # Fallback to default features if config doesn't exist
            logger.warning("Feature config not found, using default features")
            return self._get_default_features()
        
        with open(self.feature_config_path, 'r') as f:
            config = json.load(f)
        
        self.features = config['selected_features'][:32]  # Use top 32 features
        logger.info(f"Loaded {len(self.features)} features")
        return self.features
    
    def _get_default_features(self) -> List[str]:
        """Get default feature set if config is missing"""
        return [
            'wins_diff', 'slpm_diff', 'td_def_diff', 'age_diff', 'sapm_diff',
            'losses_diff', 'str_acc_diff', 'str_def_diff', 'red_Wins', 'td_avg_diff',
            'blue_Wins', 'red_SLpM', 'blue_SLpM', 'red_TD Avg.', 'blue_TD Acc.',
            'red_TD Acc.', 'red_SApM', 'td_acc_diff', 'red_TD Def.', 'blue_TD Def.',
            'red_Reach (in)', 'blue_SApM', 'blue_TD Avg.', 'blue_Reach (in)',
            'blue_Age', 'red_Age', 'blue_Str. Acc.', 'reach_(in)_diff',
            'red_Losses', 'red_Str. Acc.', 'blue_Losses', 'sub_avg_diff'
        ]
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train a Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def calculate_betting_roi(self, predictions_df: pd.DataFrame) -> Tuple[float, int]:
        """
        Calculate ROI from predictions assuming simple betting strategy.
        
        Args:
            predictions_df: DataFrame with columns ['actual', 'predicted', 'confidence']
            
        Returns:
            (roi, n_bets) tuple
        """
        if len(predictions_df) == 0:
            return 0.0, 0
        
        # Simple strategy: Bet when confidence > 60%
        betting_threshold = 0.60
        bets = predictions_df[predictions_df['confidence'] > betting_threshold].copy()
        
        if len(bets) == 0:
            return 0.0, 0
        
        # Assume flat betting with 2:1 odds (typical UFC odds)
        stake_per_bet = 1.0
        odds = 2.0
        
        # Calculate returns
        bets['return'] = bets.apply(
            lambda row: odds if row['actual'] == row['predicted'] else 0,
            axis=1
        )
        
        total_stake = len(bets) * stake_per_bet
        total_return = bets['return'].sum()
        roi = ((total_return - total_stake) / total_stake) * 100
        
        return roi, len(bets)
    
    def run_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run walk-forward backtest on historical data.
        
        Args:
            start_date: Start date for backtesting (format: 'YYYY-MM-DD')
            end_date: End date for backtesting (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with backtest results
        """
        # Load data and features
        if self.data is None:
            self.load_data()
        if self.features is None:
            self.load_features()
        
        # Filter available features
        available_features = [f for f in self.features if f in self.data.columns]
        if len(available_features) < len(self.features):
            logger.warning(f"Only {len(available_features)}/{len(self.features)} features available")
        
        # Apply date filters if provided
        df = self.data.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Initialize walk-forward
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        initial_train_end = min_date + pd.DateOffset(years=self.initial_train_years)
        
        logger.info("\n" + "="*70)
        logger.info("WALK-FORWARD BACKTESTING")
        logger.info("="*70)
        logger.info(f"Date Range: {min_date.date()} to {max_date.date()}")
        logger.info(f"Initial Training: {self.initial_train_years} years")
        logger.info(f"Test Window: {self.test_window_months} months")
        logger.info(f"Retrain Every: {self.retrain_months} months")
        logger.info("-"*70)
        
        # Walk forward through time
        results = []
        fold_num = 1
        current_train_start = min_date
        current_train_end = initial_train_end
        months_since_retrain = 0
        current_model = None
        
        while current_train_end < max_date - pd.DateOffset(months=self.test_window_months):
            test_start = current_train_end
            test_end = test_start + pd.DateOffset(months=self.test_window_months)
            
            # Get train and test data
            train_mask = (df['date'] >= current_train_start) & (df['date'] < current_train_end)
            test_mask = (df['date'] >= test_start) & (df['date'] < test_end)
            
            train_data = df[train_mask]
            test_data = df[test_mask]
            
            if len(train_data) < 100 or len(test_data) < 10:
                current_train_end = test_end
                continue
            
            # Retrain if needed
            if current_model is None or months_since_retrain >= self.retrain_months:
                logger.info(f"\n📊 Fold {fold_num}: Retraining model")
                logger.info(f"   Training: {current_train_start.date()} to {current_train_end.date()} ({len(train_data)} samples)")
                
                X_train = train_data[available_features]
                y_train = train_data['target']
                
                current_model = self.train_model(X_train, y_train)
                train_accuracy = accuracy_score(y_train, current_model.predict(X_train))
                months_since_retrain = 0
            else:
                # Use existing model
                X_train = train_data[available_features]
                y_train = train_data['target']
                train_accuracy = accuracy_score(y_train, current_model.predict(X_train))
            
            # Test the model
            X_test = test_data[available_features]
            y_test = test_data['target']
            
            y_pred = current_model.predict(X_test)
            y_prob = current_model.predict_proba(X_test)[:, 1]
            
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Create predictions DataFrame for ROI calculation
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'confidence': y_prob
            })
            
            roi, n_bets = self.calculate_betting_roi(predictions_df)
            
            logger.info(f"   Testing:  {test_start.date()} to {test_end.date()} ({len(test_data)} samples)")
            logger.info(f"   Train Acc: {train_accuracy:.3f} | Test Acc: {test_accuracy:.3f} | Gap: {train_accuracy - test_accuracy:+.3f}")
            logger.info(f"   Betting: {n_bets} bets | ROI: {roi:+.1f}%")
            
            # Store results
            fold = BacktestFold(
                fold_num=fold_num,
                train_start=current_train_start,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                predictions=predictions_df,
                betting_roi=roi,
                n_bets=n_bets
            )
            results.append(fold)
            
            # Move forward
            current_train_end = test_end
            months_since_retrain += self.test_window_months
            fold_num += 1
            
            # Limit folds for demonstration
            if fold_num > 20:
                logger.info("\n... (limiting to 20 folds)")
                break
        
        self.results = results
        return self.summarize_results()
    
    def summarize_results(self) -> pd.DataFrame:
        """Summarize backtest results"""
        if not self.results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        summary_data = []
        for fold in self.results:
            summary_data.append({
                'fold': fold.fold_num,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_samples': fold.train_samples,
                'test_samples': fold.test_samples,
                'train_accuracy': fold.train_accuracy,
                'test_accuracy': fold.test_accuracy,
                'overfitting_gap': fold.train_accuracy - fold.test_accuracy,
                'betting_roi': fold.betting_roi,
                'n_bets': fold.n_bets
            })
        
        results_df = pd.DataFrame(summary_data)
        
        # Calculate overall statistics
        logger.info("\n" + "="*70)
        logger.info("WALK-FORWARD BACKTEST SUMMARY")
        logger.info("="*70)
        
        avg_train_acc = results_df['train_accuracy'].mean()
        avg_test_acc = results_df['test_accuracy'].mean()
        avg_gap = results_df['overfitting_gap'].mean()
        
        logger.info(f"\n📊 Model Performance:")
        logger.info(f"   Average Train Accuracy: {avg_train_acc:.3f}")
        logger.info(f"   Average Test Accuracy:  {avg_test_acc:.3f}")
        logger.info(f"   Average Overfitting Gap: {avg_gap:+.3f}")
        
        # Temporal stability
        early_acc = results_df.iloc[:len(results_df)//2]['test_accuracy'].mean()
        late_acc = results_df.iloc[len(results_df)//2:]['test_accuracy'].mean()
        
        logger.info(f"\n⏱️ Temporal Stability:")
        logger.info(f"   Early Period: {early_acc:.3f}")
        logger.info(f"   Late Period:  {late_acc:.3f}")
        logger.info(f"   Degradation:  {early_acc - late_acc:+.3f}")
        
        # Betting performance
        total_bets = results_df['n_bets'].sum()
        if total_bets > 0:
            weighted_roi = (results_df['betting_roi'] * results_df['n_bets']).sum() / total_bets
            
            logger.info(f"\n💰 Betting Performance:")
            logger.info(f"   Total Bets: {total_bets}")
            logger.info(f"   Average ROI: {weighted_roi:+.1f}%")
            logger.info(f"   Best Fold ROI: {results_df['betting_roi'].max():+.1f}%")
            logger.info(f"   Worst Fold ROI: {results_df['betting_roi'].min():+.1f}%")
            
            # Bankroll simulation
            logger.info(f"\n💵 Bankroll Simulation:")
            initial_bankroll = 1000
            
            # Calculate total profit/loss
            # Assuming $20 flat bet per bet (2% of initial bankroll)
            bet_size = 20  
            
            # Calculate actual returns based on ROI
            total_profit = (weighted_roi / 100) * bet_size * total_bets
            final_bankroll = initial_bankroll + total_profit
            
            # Calculate time period
            date_range = results_df['test_end'].max() - results_df['test_start'].min()
            months = date_range.days / 30.44  # Average days per month
            
            logger.info(f"   Starting Bankroll: ${initial_bankroll:,.0f}")
            logger.info(f"   Bet Size: ${bet_size} per bet")
            logger.info(f"   Total Bets: {total_bets} over {months:.1f} months")
            logger.info(f"   Total Profit: ${total_profit:+,.0f}")
            logger.info(f"   Final Bankroll: ${final_bankroll:,.0f}")
            logger.info(f"   Total Return: {((final_bankroll - initial_bankroll) / initial_bankroll * 100):+.1f}%")
            
            if months > 0:
                monthly_profit = total_profit / months
                logger.info(f"   Monthly Profit: ${monthly_profit:+,.0f}")
                logger.info(f"   Monthly ROI: {(monthly_profit / initial_bankroll * 100):+.1f}%")
            
            # Performance message
            if final_bankroll > initial_bankroll * 1.5:
                logger.info(f"\n   🎯 EXCELLENT: Your bankroll would grow {(final_bankroll/initial_bankroll):.1f}x!")
            elif final_bankroll > initial_bankroll * 1.2:
                logger.info(f"\n   ✅ GOOD: Solid profitable strategy with {((final_bankroll - initial_bankroll) / initial_bankroll * 100):.0f}% growth")
            elif final_bankroll > initial_bankroll:
                logger.info(f"\n   📈 POSITIVE: Modest gains of {((final_bankroll - initial_bankroll) / initial_bankroll * 100):.0f}%")
            else:
                logger.info(f"\n   ⚠️ LOSS: Strategy would lose ${initial_bankroll - final_bankroll:.0f}")
        
        logger.info("\n" + "="*70)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        """
        Save backtest results to CSV file.
        
        Args:
            results_df: DataFrame with backtest results
        """
        if not results_df.empty:
            output_path = Path('model/walk_forward_backtest_results.csv')
            results_df.to_csv(output_path, index=False)
            logger.info(f"\n📁 Results saved to: {output_path}")


def main():
    """Run walk-forward backtest demonstration"""
    backtester = WalkForwardBacktester(
        initial_train_years=3,
        test_window_months=3,
        retrain_months=6
    )
    
    # Run backtest only on period with real historical odds
    # We have historical odds from Oct 2021 to Aug 2025
    results = backtester.run_backtest(
        start_date='2021-10-01',
        end_date='2025-08-31'
    )
    
    # Save results
    backtester.save_results(results)
    
    return results


if __name__ == "__main__":
    main()