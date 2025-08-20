#!/usr/bin/env python3
"""
Walk-Forward ROI Backtesting with Real Historical Odds
=======================================================
Uses actual historical betting odds to calculate realistic ROI.
Works only on the test set period where we have real odds data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardROIBacktester:
    """
    Backtests ROI using real historical odds and walk-forward approach.
    This uses ACTUAL betting odds from your scraped data.
    """
    
    def __init__(self,
                 fight_data_path: str = 'model/ufc_fight_dataset_with_diffs.csv',
                 odds_data_path: str = 'data/test_set_odds/test_set_with_odds_20250819.csv',
                 feature_config_path: str = 'model/optimized/feature_selector_latest.json',
                 train_cutoff: str = '2021-10-01',
                 retrain_months: int = 6,
                 bankroll: float = 1000.0,
                 bet_size: float = 20.0):
        """
        Initialize ROI backtester with real odds
        
        Args:
            fight_data_path: Path to fight dataset with features
            odds_data_path: Path to historical odds data
            feature_config_path: Path to feature configuration
            train_cutoff: Date to split train/test
            retrain_months: How often to retrain model
            bankroll: Starting bankroll
            bet_size: Fixed bet size
        """
        self.fight_data_path = Path(fight_data_path)
        self.odds_data_path = Path(odds_data_path)
        self.feature_config_path = Path(feature_config_path)
        self.train_cutoff = pd.to_datetime(train_cutoff)
        self.retrain_months = retrain_months
        self.initial_bankroll = bankroll
        self.bet_size = bet_size
        
        self.fight_data = None
        self.odds_data = None
        self.features = None
        
    def load_fight_data(self) -> pd.DataFrame:
        """Load fight outcome and feature data"""
        logger.info(f"Loading fight data from {self.fight_data_path}")
        df = pd.read_csv(self.fight_data_path)
        
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
        
        # Normalize names for matching
        df['fighter_normalized'] = df['Fighter'].str.lower().str.strip()
        df['opponent_normalized'] = df['Opponent'].str.lower().str.strip()
        
        self.fight_data = df
        logger.info(f"Loaded {len(df)} fights")
        return df
    
    def load_odds_data(self) -> pd.DataFrame:
        """Load historical betting odds"""
        logger.info(f"Loading odds data from {self.odds_data_path}")
        odds_df = pd.read_csv(self.odds_data_path)
        
        # Parse dates
        odds_df['date'] = pd.to_datetime(odds_df['date'])
        
        # Filter to rows with odds
        odds_df = odds_df[odds_df['has_odds'] == True].copy()
        
        # Normalize names for matching
        odds_df['fighter_normalized'] = odds_df['fighter'].str.lower().str.strip()
        odds_df['opponent_normalized'] = odds_df['opponent'].str.lower().str.strip()
        
        self.odds_data = odds_df
        logger.info(f"Loaded {len(odds_df)} fights with odds from {odds_df['date'].min().date()} to {odds_df['date'].max().date()}")
        return odds_df
    
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
    
    def match_fight_with_odds(self, fighter: str, opponent: str, date: pd.Timestamp) -> Optional[Dict]:
        """
        Match a fight prediction with historical odds
        
        Returns:
            Dict with odds info or None if no match
        """
        # Try within 7 days window
        date_range = (self.odds_data['date'] >= date - pd.Timedelta(days=7)) & \
                    (self.odds_data['date'] <= date + pd.Timedelta(days=7))
        date_odds = self.odds_data[date_range]
        
        if len(date_odds) == 0:
            return None
        
        # Normalize names - more aggressive normalization
        fighter_norm = fighter.lower().strip().replace(' ', '')
        opponent_norm = opponent.lower().strip().replace(' ', '')
        
        # Try to find matching fight
        for _, row in date_odds.iterrows():
            row_fighter = row['fighter_normalized'].replace(' ', '')
            row_opponent = row['opponent_normalized'].replace(' ', '')
            
            # Check if last names match at minimum
            fighter_last = fighter_norm.split()[-1] if ' ' in fighter else fighter_norm
            opponent_last = opponent_norm.split()[-1] if ' ' in opponent else opponent_norm
            
            # More lenient matching
            if (fighter_last in row_fighter or row_fighter in fighter_norm) and \
               (opponent_last in row_opponent or row_opponent in opponent_norm):
                return {
                    'fighter_odds': row['fighter_odds'],
                    'opponent_odds': row['opponent_odds'],
                    'matched': True
                }
            # Check reverse
            elif (opponent_last in row_fighter or row_fighter in opponent_norm) and \
                 (fighter_last in row_opponent or row_opponent in fighter_norm):
                return {
                    'fighter_odds': row['opponent_odds'],
                    'opponent_odds': row['fighter_odds'],
                    'matched': True
                }
        
        return None
    
    def calculate_bet_return(self, predicted: int, actual: int, odds: float, bet_amount: float) -> float:
        """
        Calculate return from a bet
        
        Args:
            predicted: 1 if predicted winner, 0 if predicted loser
            actual: 1 if actual winner, 0 if actual loser
            odds: Decimal odds for the predicted fighter
            bet_amount: Amount bet
            
        Returns:
            Profit/loss from bet
        """
        if predicted != 1:  # Only bet on predicted winners
            return 0
        
        if predicted == actual:  # Won the bet
            return (odds - 1) * bet_amount
        else:  # Lost the bet
            return -bet_amount
    
    def run_backtest(self) -> Dict:
        """
        Run walk-forward ROI backtest with real odds
        
        Returns:
            Dictionary with backtest results
        """
        # Load all data
        if self.fight_data is None:
            self.load_fight_data()
        if self.odds_data is None:
            self.load_odds_data()
        if self.features is None:
            self.load_features()
        
        # Split data
        train_data = self.fight_data[self.fight_data['date'] < self.train_cutoff]
        test_data = self.fight_data[self.fight_data['date'] >= self.train_cutoff]
        
        logger.info("\n" + "="*70)
        logger.info("WALK-FORWARD ROI BACKTEST WITH REAL ODDS")
        logger.info("="*70)
        logger.info(f"Training Set: {len(train_data)} fights before {self.train_cutoff.date()}")
        logger.info(f"Test Set: {len(test_data)} fights after {self.train_cutoff.date()}")
        logger.info(f"Historical Odds Available: {len(self.odds_data)} fights")
        logger.info(f"Starting Bankroll: ${self.initial_bankroll:,.0f}")
        logger.info(f"Bet Size: ${self.bet_size}")
        logger.info("-"*70)
        
        # Get available features
        available_features = [f for f in self.features if f in self.fight_data.columns]
        
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
        bankroll = self.initial_bankroll
        results = []
        current_model = model
        months_since_retrain = 0
        
        test_start = test_data['date'].min()
        test_end = test_data['date'].max()
        current_date = test_start
        
        fold = 1
        total_bets = 0
        total_matched = 0
        total_wins = 0
        
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
                retrain_data = self.fight_data[self.fight_data['date'] < current_date]
                if len(retrain_data) > len(train_data):
                    logger.info(f"\n📊 Fold {fold}: Retraining with {len(retrain_data)} samples")
                    X_retrain = retrain_data[available_features]
                    y_retrain = retrain_data['target']
                    current_model.fit(X_retrain, y_retrain)
                    months_since_retrain = 0
            
            # Test on period with real odds
            period_bets = 0
            period_matched = 0
            period_profit = 0
            period_wins = 0
            
            for _, fight in period_data.iterrows():
                # Get prediction
                X_fight = fight[available_features].values.reshape(1, -1)
                y_pred = current_model.predict(X_fight)[0]
                y_prob = current_model.predict_proba(X_fight)[0, 1]
                y_actual = fight['target']
                
                # Match with odds
                odds_info = self.match_fight_with_odds(
                    fight.get('Fighter', ''),
                    fight.get('Opponent', ''),
                    fight['date']
                )
                
                if odds_info and y_prob > 0.55:  # Only bet if confident
                    period_matched += 1
                    total_matched += 1
                    
                    # Calculate bet return
                    fighter_odds = odds_info['fighter_odds']
                    profit = self.calculate_bet_return(y_pred, y_actual, fighter_odds, self.bet_size)
                    
                    if profit > 0:
                        period_wins += 1
                        total_wins += 1
                    
                    period_profit += profit
                    bankroll += profit
                    period_bets += 1
                    total_bets += 1
            
            # Log period results
            logger.info(f"   Period: {current_date.date()} to {period_end.date()}")
            logger.info(f"   Fights: {len(period_data)} | Matched with odds: {period_matched}")
            
            if period_matched > 0:
                period_roi = (period_profit / (period_matched * self.bet_size)) * 100
                win_rate = (period_wins / period_matched) * 100 if period_matched > 0 else 0
                
                logger.info(f"   Bets: {period_bets} | Wins: {period_wins} ({win_rate:.1f}%)")
                logger.info(f"   Profit: ${period_profit:+.0f} | ROI: {period_roi:+.1f}%")
                logger.info(f"   Bankroll: ${bankroll:,.0f}")
            
            results.append({
                'fold': fold,
                'start_date': current_date,
                'end_date': period_end,
                'n_fights': len(period_data),
                'n_matched': period_matched,
                'n_bets': period_bets,
                'n_wins': period_wins,
                'profit': period_profit,
                'bankroll': bankroll
            })
            
            # Move forward
            current_date = period_end
            months_since_retrain += 3
            fold += 1
        
        # Calculate overall metrics
        total_profit = bankroll - self.initial_bankroll
        total_roi = (total_profit / (total_bets * self.bet_size)) * 100 if total_bets > 0 else 0
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0
        
        logger.info("\n" + "="*70)
        logger.info("ROI BACKTEST SUMMARY (WITH REAL ODDS)")
        logger.info("="*70)
        
        logger.info(f"\n💰 Overall Performance:")
        logger.info(f"   Total Bets Placed: {total_bets}")
        logger.info(f"   Fights Matched with Odds: {total_matched}/{len(test_data)} ({total_matched/len(test_data)*100:.1f}%)")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Total ROI: {total_roi:+.1f}%")
        
        logger.info(f"\n💵 Bankroll Growth:")
        logger.info(f"   Starting: ${self.initial_bankroll:,.0f}")
        logger.info(f"   Final: ${bankroll:,.0f}")
        logger.info(f"   Total Profit: ${total_profit:+,.0f}")
        logger.info(f"   Growth: {(bankroll/self.initial_bankroll - 1)*100:+.1f}%")
        
        # Performance rating
        if bankroll > self.initial_bankroll * 1.5:
            logger.info(f"\n🎯 EXCELLENT: Your bankroll grew {bankroll/self.initial_bankroll:.1f}x with real odds!")
        elif bankroll > self.initial_bankroll * 1.2:
            logger.info(f"\n✅ GOOD: Solid {(bankroll/self.initial_bankroll - 1)*100:.0f}% growth with real odds")
        elif bankroll > self.initial_bankroll:
            logger.info(f"\n📈 POSITIVE: Modest {(bankroll/self.initial_bankroll - 1)*100:.0f}% gains")
        else:
            logger.info(f"\n⚠️ LOSS: Strategy lost ${self.initial_bankroll - bankroll:.0f}")
        
        logger.info("\n" + "="*70)
        
        return {
            'total_bets': total_bets,
            'total_matched': total_matched,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'results_df': pd.DataFrame(results)
        }


def main():
    """Run walk-forward ROI backtest with real odds"""
    backtester = WalkForwardROIBacktester()
    results = backtester.run_backtest()
    
    # Save results
    results_df = results['results_df']
    output_path = Path('model/walk_forward_roi_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n📁 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()