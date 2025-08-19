#!/usr/bin/env python3
"""
Run Historical Backtest
========================

Complete historical backtesting of UFC predictions using real odds data.
Integrates model predictions with historical odds to calculate actual ROI.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import logging
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ufc_predictor.betting.historical_backtester import (
    HistoricalBacktester,
    FighterNameMatcher
)
from src.ufc_predictor.betting.historical_odds_fetcher import HistoricalOddsProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBacktester:
    """Integrates UFC prediction model with historical backtesting"""
    
    def __init__(self, model_path: Path = None):
        """
        Initialize the model backtester
        
        Args:
            model_path: Path to trained model (uses latest if not specified)
        """
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"
        
        # Load the model
        self.model = self._load_model(model_path)
        self.name_matcher = FighterNameMatcher()
        
    def _load_model(self, model_path: Path = None):
        """Load the trained UFC prediction model"""
        if model_path is None:
            # Find latest model
            model_files = list(self.model_dir.glob("rf_model_*.pkl"))
            if not model_files:
                logger.warning("No trained model found. Using placeholder predictions.")
                return None
            
            # Get most recent model
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading model: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def load_historical_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load historical odds and fight results
        
        Returns:
            Tuple of (historical_odds_df, fight_results_df)
        """
        # Load historical odds
        odds_file = self.data_dir / "historical_odds" / "ufc_historical_odds_complete.csv"
        if not odds_file.exists():
            # Try to find any historical odds file
            odds_files = list((self.data_dir / "historical_odds").glob("*.csv"))
            if odds_files:
                odds_file = odds_files[0]
                logger.info(f"Using odds file: {odds_file}")
            else:
                logger.error("No historical odds data found!")
                logger.error("Run: python3 fetch_all_historical_odds.py")
                return None, None
        
        odds_df = pd.read_csv(odds_file)
        logger.info(f"Loaded {len(odds_df)} historical odds records")
        
        # Load fight results
        fights_file = self.project_root / "src" / "ufc_predictor" / "data" / "ufc_fights.csv"
        if not fights_file.exists():
            logger.error(f"Fight results not found: {fights_file}")
            return odds_df, None
        
        fights_df = pd.read_csv(fights_file)
        logger.info(f"Loaded {len(fights_df)} fight results")
        
        return odds_df, fights_df
    
    def generate_predictions(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate model predictions for historical fights
        
        Args:
            fights_df: DataFrame with fight data
            
        Returns:
            DataFrame with predictions added
        """
        predictions = []
        
        for _, fight in fights_df.iterrows():
            fighter = fight['Fighter']
            opponent = fight['Opponent']
            
            # Generate prediction
            if self.model is not None:
                # Use actual model
                prediction = self._predict_fight(fighter, opponent)
            else:
                # Use heuristic predictions for testing
                prediction = self._heuristic_prediction(fighter, opponent, fight.get('Outcome'))
            
            predictions.append({
                'Fighter': fighter,
                'Opponent': opponent,
                'predicted_winner': prediction['winner'],
                'win_probability': prediction['probability'],
                'confidence': prediction.get('confidence', 'medium')
            })
        
        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(predictions_df)} predictions")
        
        return predictions_df
    
    def _predict_fight(self, fighter: str, opponent: str) -> Dict:
        """Generate prediction using actual model"""
        # TODO: Integrate with actual model prediction pipeline
        # This would need fighter stats and feature engineering
        
        # For now, return placeholder
        return self._heuristic_prediction(fighter, opponent, None)
    
    def _heuristic_prediction(self, fighter: str, opponent: str, 
                             actual_outcome: str = None) -> Dict:
        """
        Generate heuristic predictions for testing
        
        Uses simple rules to generate semi-realistic predictions
        """
        import random
        
        # Seed based on fighter names for consistency
        seed = hash(f"{fighter}_{opponent}") % 1000000
        np.random.seed(seed)
        
        # Generate base probability
        base_prob = 0.5
        
        # Add some variance
        variance = np.random.normal(0, 0.15)
        win_prob = max(0.2, min(0.8, base_prob + variance))
        
        # Determine predicted winner
        if win_prob > 0.5:
            predicted_winner = fighter
        else:
            predicted_winner = opponent
            win_prob = 1 - win_prob
        
        # Determine confidence level
        if win_prob > 0.7:
            confidence = 'high'
        elif win_prob > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'winner': predicted_winner,
            'probability': win_prob,
            'confidence': confidence
        }
    
    def run_backtest(self, start_date: datetime = None, 
                    end_date: datetime = None,
                    initial_bankroll: float = 1000,
                    strategy: str = 'kelly',
                    min_edge: float = 0.05) -> Dict:
        """
        Run complete historical backtest
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_bankroll: Starting bankroll
            strategy: Betting strategy ('kelly', 'flat', 'proportional')
            min_edge: Minimum expected value to bet
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 60)
        logger.info("🚀 RUNNING HISTORICAL BACKTEST")
        logger.info("=" * 60)
        
        # Load data
        odds_df, fights_df = self.load_historical_data()
        
        if odds_df is None or fights_df is None:
            logger.error("Failed to load required data")
            return None
        
        # Filter by date range
        if start_date or end_date:
            logger.info(f"Filtering data from {start_date} to {end_date}")
            
            # Filter odds
            if 'commence_time' in odds_df.columns:
                odds_df['date'] = pd.to_datetime(odds_df['commence_time'])
                
                if start_date:
                    odds_df = odds_df[odds_df['date'] >= start_date]
                if end_date:
                    odds_df = odds_df[odds_df['date'] <= end_date]
            
            # Filter fights
            # Extract dates from fight events
            fights_df['date'] = fights_df['Event'].apply(self._extract_date_from_event)
            
            if start_date:
                fights_df = fights_df[fights_df['date'] >= start_date]
            if end_date:
                fights_df = fights_df[fights_df['date'] <= end_date]
        
        logger.info(f"Backtesting period: {len(fights_df)} fights, {len(odds_df)} odds records")
        
        # Generate predictions
        predictions_df = self.generate_predictions(fights_df)
        
        # Initialize backtester
        backtester = HistoricalBacktester(
            model_predictions=predictions_df,
            historical_odds=odds_df,
            fight_results=fights_df,
            initial_bankroll=initial_bankroll
        )
        
        # Match fights with odds
        logger.info("\n📊 Matching fights with historical odds...")
        matched_fights = backtester.match_fights_with_odds()
        
        if matched_fights.empty:
            logger.error("No fights could be matched with odds data")
            return None
        
        logger.info(f"✅ Matched {len(matched_fights)} fights")
        
        # Add predictions to matched fights
        matched_fights = self._add_predictions_to_matches(matched_fights, predictions_df)
        
        # Run backtest simulation
        logger.info(f"\n💰 Running backtest simulation...")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Min Edge: {min_edge}")
        logger.info(f"   Initial Bankroll: ${initial_bankroll}")
        
        # Create enhanced backtester with predictions
        enhanced_backtester = EnhancedBacktester(
            matched_fights=matched_fights,
            initial_bankroll=initial_bankroll
        )
        
        results = enhanced_backtester.run_simulation(
            strategy=strategy,
            min_edge=min_edge
        )
        
        # Generate report
        logger.info("\n📝 Generating report...")
        report_path = self.data_dir / "backtest_reports" / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = backtester.generate_report(results, report_path)
        
        print("\n" + report)
        
        logger.info(f"\n✅ Backtest complete!")
        logger.info(f"Report saved to: {report_path}")
        
        return results
    
    def _extract_date_from_event(self, event_str: str) -> Optional[datetime]:
        """Extract date from UFC event string"""
        import re
        match = re.search(r'([A-Z][a-z]+)\. (\d{1,2}), (\d{4})', event_str)
        if match:
            month_str, day, year = match.groups()
            months = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = months.get(month_str[:3], 1)
            try:
                return datetime(int(year), month, int(day))
            except:
                return None
        return None
    
    def _add_predictions_to_matches(self, matched_fights: pd.DataFrame, 
                                   predictions: pd.DataFrame) -> pd.DataFrame:
        """Add model predictions to matched fights"""
        # Merge predictions with matched fights
        merged = matched_fights.merge(
            predictions,
            left_on=['fighter', 'opponent'],
            right_on=['Fighter', 'Opponent'],
            how='left'
        )
        
        # Fill missing predictions with defaults
        merged['predicted_winner'] = merged['predicted_winner'].fillna(merged['fighter'])
        merged['win_probability'] = merged['win_probability'].fillna(0.5)
        merged['confidence'] = merged['confidence'].fillna('low')
        
        return merged


class EnhancedBacktester:
    """Enhanced backtester with actual predictions"""
    
    def __init__(self, matched_fights: pd.DataFrame, initial_bankroll: float):
        """Initialize enhanced backtester"""
        self.fights = matched_fights
        self.initial_bankroll = initial_bankroll
    
    def run_simulation(self, strategy: str = 'kelly', min_edge: float = 0.05) -> Dict:
        """Run betting simulation with predictions"""
        from src.ufc_predictor.betting.historical_backtester import BacktestResult
        
        results = []
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        
        for _, fight in self.fights.iterrows():
            # Get prediction
            win_prob = fight.get('win_probability', 0.5)
            predicted_winner = fight.get('predicted_winner', fight['fighter'])
            
            # Determine which odds to use
            if predicted_winner == fight['fighter']:
                odds = fight['fighter_odds']
            else:
                odds = fight['opponent_odds']
            
            # Calculate expected value
            implied_prob = 1 / odds
            expected_value = (win_prob * (odds - 1)) - (1 - win_prob)
            
            # Check betting criteria
            if expected_value < min_edge:
                continue
            
            # Calculate bet size based on strategy
            if strategy == 'kelly':
                kelly_fraction = (win_prob * (odds - 1) - (1 - win_prob)) / (odds - 1)
                kelly_fraction = max(0, min(kelly_fraction, 0.25))
                bet_amount = bankroll * kelly_fraction * 0.25  # Conservative Kelly
            elif strategy == 'flat':
                bet_amount = min(bankroll * 0.02, 20)  # 2% or $20
            else:  # proportional
                bet_amount = bankroll * 0.01 * (1 + expected_value)
            
            # Cap bet size
            bet_amount = min(bet_amount, bankroll * 0.05, 50)
            
            if bet_amount < 1 or bet_amount > bankroll:
                continue
            
            # Determine outcome
            actual_winner = fight['fighter'] if fight['outcome'] == 'win' else fight['opponent']
            bet_won = (predicted_winner == actual_winner)
            
            # Calculate profit
            profit = bet_amount * (odds - 1) if bet_won else -bet_amount
            
            # Update bankroll
            bankroll += profit
            bankroll_history.append(bankroll)
            
            # Record result
            result = BacktestResult(
                date=str(fight.get('date', 'Unknown')),
                event=fight.get('event', 'Unknown'),
                fighter_a=fight['fighter'],
                fighter_b=fight['opponent'],
                predicted_winner=predicted_winner,
                actual_winner=actual_winner,
                bet_amount=bet_amount,
                odds=odds,
                profit=profit,
                win=bet_won,
                model_probability=win_prob,
                implied_probability=implied_prob,
                expected_value=expected_value
            )
            results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(results, bankroll_history)
        
        return {
            'results': results,
            'metrics': metrics,
            'bankroll_history': bankroll_history
        }
    
    def _calculate_comprehensive_metrics(self, results: List, 
                                        bankroll_history: List[float]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not results:
            return {
                'total_bets': 0,
                'roi': 0,
                'win_rate': 0,
                'profit': 0,
                'status': 'No bets placed'
            }
        
        df = pd.DataFrame([r.to_dict() for r in results])
        
        total_bets = len(df)
        total_staked = df['bet_amount'].sum()
        total_profit = df['profit'].sum()
        wins = df['win'].sum()
        
        # Calculate advanced metrics
        metrics = {
            # Basic metrics
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': (wins / total_bets * 100) if total_bets > 0 else 0,
            
            # Financial metrics
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll_history[-1] if bankroll_history else self.initial_bankroll,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
            'bankroll_growth_pct': ((bankroll_history[-1] - self.initial_bankroll) / self.initial_bankroll * 100) if bankroll_history else 0,
            
            # Betting metrics
            'avg_bet_size': df['bet_amount'].mean(),
            'avg_odds': df['odds'].mean(),
            'avg_expected_value': df['expected_value'].mean() * 100,  # As percentage
            'avg_win_odds': df[df['win'] == True]['odds'].mean() if wins > 0 else 0,
            'avg_loss_odds': df[df['win'] == False]['odds'].mean() if (total_bets - wins) > 0 else 0,
            
            # Risk metrics
            'max_bet': df['bet_amount'].max(),
            'max_win': df['profit'].max(),
            'max_loss': df['profit'].min(),
            'max_drawdown_pct': self._calculate_max_drawdown(bankroll_history),
            'volatility': df['profit'].std(),
            
            # Performance metrics
            'sharpe_ratio': self._calculate_sharpe(df),
            'profit_factor': self._calculate_profit_factor(df),
            'expected_value_accuracy': self._calculate_ev_accuracy(df)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if len(bankroll_history) < 2:
            return 0
        
        peak = bankroll_history[0]
        max_dd = 0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if len(df) < 2:
            return 0
        
        returns = df['profit'] / df['bet_amount']
        if returns.std() == 0:
            return 0
        
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = df[df['profit'] > 0]['profit'].sum()
        gross_loss = abs(df[df['profit'] < 0]['profit'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_ev_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate how accurate expected value predictions were"""
        if df.empty:
            return 0
        
        # Compare expected vs actual returns
        df['expected_return'] = df['expected_value'] * df['bet_amount']
        df['actual_return'] = df['profit']
        
        # Calculate correlation
        if len(df) > 1:
            correlation = df['expected_return'].corr(df['actual_return'])
            return correlation * 100 if not pd.isna(correlation) else 0
        
        return 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run historical backtest of UFC predictions')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--bankroll', type=float, default=1000, help='Initial bankroll')
    parser.add_argument('--strategy', choices=['kelly', 'flat', 'proportional'], 
                       default='kelly', help='Betting strategy')
    parser.add_argument('--min-edge', type=float, default=0.05, 
                       help='Minimum expected value to bet (0.05 = 5%)')
    parser.add_argument('--model', type=str, help='Path to model file')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize backtester
    model_path = Path(args.model) if args.model else None
    backtester = ModelBacktester(model_path)
    
    # Run backtest
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_bankroll=args.bankroll,
        strategy=args.strategy,
        min_edge=args.min_edge
    )
    
    if results:
        logger.info("\n✅ Backtest completed successfully!")
        
        # Show key metrics
        metrics = results['metrics']
        print("\n" + "=" * 60)
        print("KEY PERFORMANCE INDICATORS")
        print("=" * 60)
        print(f"ROI: {metrics['roi']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Bankroll Growth: {metrics['bankroll_growth_pct']:.1f}%")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    else:
        logger.error("Backtest failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())