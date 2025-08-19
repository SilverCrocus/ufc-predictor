#!/usr/bin/env python3
"""
Backtest with Real Model Predictions
=====================================

Uses your actual model predictions to calculate realistic ROI.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BettingStrategy:
    """Betting strategy parameters"""
    name: str
    max_bet_percentage: float
    min_edge: float
    kelly_fraction: float
    temperature_scaling: float = 1.0
    enable_parlays: bool = False
    max_exposure: float = 0.10


# Your actual betting strategies from notebooks
STRATEGIES = {
    'original': BettingStrategy(
        name='Original Strategy',
        max_bet_percentage=0.05,      # 5% max from your notebooks
        min_edge=0.05,                # 5% min edge
        kelly_fraction=0.25,          # Quarter Kelly
        temperature_scaling=1.0,      # No calibration
        enable_parlays=True,          # Parlays enabled
        max_exposure=0.20             # 20% total exposure
    ),
    
    'v2_conservative': BettingStrategy(
        name='V2 Conservative',
        max_bet_percentage=0.02,      # 2% max (from your V2)
        min_edge=0.03,                # 3% min edge (from your V2)
        kelly_fraction=0.25,          # Quarter Kelly
        temperature_scaling=1.1,      # Conservative calibration
        enable_parlays=False,         # Disabled in V2
        max_exposure=0.10             # 10% total exposure
    ),
    
    'calibrated': BettingStrategy(
        name='Calibrated Model',
        max_bet_percentage=0.03,      # 3% max
        min_edge=0.05,                # 5% min edge
        kelly_fraction=0.20,          # More conservative Kelly
        temperature_scaling=1.15,     # Temperature scaling
        enable_parlays=True,          # Allow parlays
        max_exposure=0.15             # 15% exposure
    )
}


class RealBacktester:
    """Backtest using actual model predictions"""
    
    def __init__(self, strategy: BettingStrategy):
        self.strategy = strategy
        self.initial_bankroll = 1000
        self.bankroll = self.initial_bankroll
        self.bets = []
        self.results = []
        
    def apply_temperature_scaling(self, prob: float) -> float:
        """Apply temperature scaling for calibration"""
        if self.strategy.temperature_scaling == 1.0:
            return prob
        
        # Temperature scaling formula
        logit = np.log(prob / (1 - prob))
        scaled_logit = logit / self.strategy.temperature_scaling
        return 1 / (1 + np.exp(-scaled_logit))
    
    def calculate_kelly_bet(self, prob: float, odds: float) -> float:
        """Calculate bet size using Kelly criterion"""
        # Apply temperature scaling
        calibrated_prob = self.apply_temperature_scaling(prob)
        
        # Calculate edge
        edge = (calibrated_prob * odds) - 1
        
        # Skip if below minimum edge
        if edge < self.strategy.min_edge:
            return 0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win prob, q = lose prob, b = decimal odds - 1
        q = 1 - calibrated_prob
        b = odds - 1
        kelly_fraction = (calibrated_prob * b - q) / b
        
        # Apply conservative fraction
        bet_fraction = kelly_fraction * self.strategy.kelly_fraction
        
        # Cap at max bet percentage
        bet_fraction = min(bet_fraction, self.strategy.max_bet_percentage)
        
        # Ensure positive bet
        return max(0, bet_fraction)
    
    def process_fight(self, fight: pd.Series, prediction: pd.Series) -> Dict:
        """Process a single fight"""
        
        # Get model prediction
        fighter_prob = prediction['model_probability']
        
        # Get odds
        fighter_odds = fight.get('fighter_odds', 0)
        opponent_odds = fight.get('opponent_odds', 0)
        
        # Skip if no valid odds
        if pd.isna(fighter_odds) or pd.isna(opponent_odds) or fighter_odds <= 1 or opponent_odds <= 1:
            return None
        
        # Determine which fighter to bet on
        opponent_prob = 1 - fighter_prob
        
        # Calculate Kelly bets for both sides
        fighter_bet_fraction = self.calculate_kelly_bet(fighter_prob, fighter_odds)
        opponent_bet_fraction = self.calculate_kelly_bet(opponent_prob, opponent_odds)
        
        # Choose the better bet
        if fighter_bet_fraction > opponent_bet_fraction and fighter_bet_fraction > 0:
            bet_on = 'fighter'
            bet_fraction = fighter_bet_fraction
            bet_odds = fighter_odds
            bet_prob = fighter_prob
        elif opponent_bet_fraction > 0:
            bet_on = 'opponent'
            bet_fraction = opponent_bet_fraction
            bet_odds = opponent_odds
            bet_prob = opponent_prob
        else:
            return None  # No bet
        
        # Check bankroll and exposure limits
        current_exposure = sum(b['amount'] for b in self.bets if b.get('status') != 'settled')
        max_exposure_amount = self.bankroll * self.strategy.max_exposure
        
        if current_exposure >= max_exposure_amount:
            return None  # Exposure limit reached
        
        # Calculate bet amount
        bet_amount = min(
            self.bankroll * bet_fraction,
            max_exposure_amount - current_exposure
        )
        
        if bet_amount < 1:  # Minimum bet size
            return None
        
        # Create bet
        bet = {
            'date': fight['date'],
            'event': fight.get('event', ''),
            'fighter': fight['fighter'],
            'opponent': fight['opponent'],
            'bet_on': bet_on,
            'amount': bet_amount,
            'odds': bet_odds,
            'probability': bet_prob,
            'edge': (bet_prob * bet_odds) - 1,
            'outcome': fight.get('outcome'),
            'status': 'pending'
        }
        
        return bet
    
    def settle_bet(self, bet: Dict) -> float:
        """Settle a bet and return profit/loss"""
        
        # Determine if bet won
        if bet['bet_on'] == 'fighter':
            won = bet['outcome'] == 'win'
        else:
            won = bet['outcome'] != 'win'
        
        # Calculate profit/loss
        if won:
            profit = bet['amount'] * (bet['odds'] - 1)
        else:
            profit = -bet['amount']
        
        bet['profit'] = profit
        bet['won'] = won
        bet['status'] = 'settled'
        
        return profit
    
    def run_backtest(self, test_df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict:
        """Run the backtest"""
        
        logger.info(f"Starting backtest with {self.strategy.name}")
        logger.info(f"Initial bankroll: ${self.initial_bankroll:,.2f}")
        
        # Merge predictions with test data
        merged = pd.merge(
            test_df,
            predictions_df[['fighter', 'opponent', 'date', 'model_probability']],
            on=['fighter', 'opponent', 'date'],
            how='inner'
        )
        
        logger.info(f"Processing {len(merged)} fights with predictions")
        
        # Process fights chronologically
        merged = merged.sort_values('date')
        
        daily_results = {}
        
        for idx, fight in merged.iterrows():
            # Find matching prediction
            pred_mask = (
                (predictions_df['fighter'] == fight['fighter']) & 
                (predictions_df['opponent'] == fight['opponent']) &
                (predictions_df['date'] == fight['date'])
            )
            
            if not pred_mask.any():
                continue
            
            prediction = predictions_df[pred_mask].iloc[0]
            
            # Process fight
            bet = self.process_fight(fight, prediction)
            
            if bet:
                # Place bet
                self.bets.append(bet)
                
                # Settle immediately (we know the outcome)
                profit = self.settle_bet(bet)
                self.bankroll += profit
                
                # Track daily results
                date = fight['date']
                if date not in daily_results:
                    daily_results[date] = {
                        'bets': 0,
                        'won': 0,
                        'profit': 0,
                        'bankroll': self.bankroll
                    }
                
                daily_results[date]['bets'] += 1
                daily_results[date]['won'] += 1 if bet['won'] else 0
                daily_results[date]['profit'] += profit
                daily_results[date]['bankroll'] = self.bankroll
                
                # Log significant bets
                if abs(profit) > 50 or self.bankroll < 100:
                    result = "WON" if bet['won'] else "LOST"
                    logger.info(f"  {bet['date']}: {result} ${abs(profit):.2f} on {bet['fighter'] if bet['bet_on'] == 'fighter' else bet['opponent']} @ {bet['odds']:.2f}")
                    logger.info(f"    Bankroll: ${self.bankroll:.2f}")
        
        # Calculate final stats
        total_bets = len(self.bets)
        winning_bets = sum(1 for b in self.bets if b['won'])
        total_profit = self.bankroll - self.initial_bankroll
        roi = (total_profit / self.initial_bankroll) * 100
        
        # Win rate
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # Average bet size
        avg_bet = np.mean([b['amount'] for b in self.bets]) if self.bets else 0
        
        # Average edge
        avg_edge = np.mean([b['edge'] for b in self.bets]) if self.bets else 0
        
        # Max drawdown
        running_bankroll = self.initial_bankroll
        peak = self.initial_bankroll
        max_drawdown = 0
        
        for date in sorted(daily_results.keys()):
            running_bankroll = daily_results[date]['bankroll']
            peak = max(peak, running_bankroll)
            drawdown = (peak - running_bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'strategy': self.strategy.name,
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': win_rate,
            'avg_bet_size': avg_bet,
            'avg_edge': avg_edge,
            'max_drawdown': max_drawdown,
            'daily_results': daily_results
        }
        
        return results


def main():
    """Run backtests with all strategies"""
    
    logger.info("=" * 70)
    logger.info("BACKTESTING WITH REAL MODEL PREDICTIONS")
    logger.info("=" * 70)
    
    # Load test set with odds
    test_file = Path("data/test_set_odds/test_set_with_odds_20250819.csv")
    test_df = pd.read_csv(test_file)
    test_df = test_df[test_df['has_odds'] == True]
    
    # Load model predictions
    pred_file = Path("data/test_set_odds/model_predictions.csv")
    if not pred_file.exists():
        logger.error(f"Predictions not found: {pred_file}")
        logger.info("Run: python3 generate_model_predictions.py first")
        return
    
    predictions_df = pd.read_csv(pred_file)
    
    logger.info(f"Loaded {len(predictions_df)} predictions")
    logger.info(f"Testing on {len(test_df)} fights with odds")
    
    # Analyze prediction quality
    with_features = predictions_df['has_features'].sum()
    logger.info(f"\nModel predictions: {with_features} with features, {len(predictions_df)-with_features} using odds fallback")
    
    # Run backtests for each strategy
    all_results = {}
    
    for strategy_name, strategy in STRATEGIES.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing: {strategy.name}")
        logger.info(f"{'=' * 50}")
        
        backtester = RealBacktester(strategy)
        results = backtester.run_backtest(test_df, predictions_df)
        all_results[strategy_name] = results
        
        # Print summary
        logger.info(f"\n📊 RESULTS - {strategy.name}")
        logger.info("-" * 40)
        logger.info(f"Initial: ${results['initial_bankroll']:,.2f}")
        logger.info(f"Final:   ${results['final_bankroll']:,.2f}")
        logger.info(f"Profit:  ${results['total_profit']:,.2f}")
        logger.info(f"ROI:     {results['roi']:.1f}%")
        logger.info(f"Bets:    {results['total_bets']} ({results['winning_bets']} won)")
        logger.info(f"Win Rate: {results['win_rate']:.1%}")
        logger.info(f"Avg Bet: ${results['avg_bet_size']:.2f}")
        logger.info(f"Avg Edge: {results['avg_edge']:.1%}")
        logger.info(f"Max DD:  {results['max_drawdown']:.1%}")
    
    # Compare strategies
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 70)
    
    comparison_data = []
    for name, results in all_results.items():
        comparison_data.append({
            'Strategy': results['strategy'],
            'ROI': f"{results['roi']:.1f}%",
            'Final': f"${results['final_bankroll']:,.0f}",
            'Bets': results['total_bets'],
            'Win%': f"{results['win_rate']:.1%}",
            'MaxDD': f"{results['max_drawdown']:.1%}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Best strategy
    best_strategy = max(all_results.items(), key=lambda x: x[1]['roi'])
    logger.info(f"\n🏆 Best Strategy: {best_strategy[1]['strategy']} with {best_strategy[1]['roi']:.1f}% ROI")
    
    # Save detailed results
    output_file = Path("data/test_set_odds/backtest_results_real_model.csv")
    
    # Create detailed results DataFrame
    detailed_results = []
    for name, results in all_results.items():
        for bet in results.get('bets', [])[:100]:  # Save first 100 bets per strategy
            detailed_results.append({
                'strategy': name,
                'date': bet['date'],
                'fighter': bet['fighter'],
                'opponent': bet['opponent'],
                'bet_on': bet['bet_on'],
                'amount': bet['amount'],
                'odds': bet['odds'],
                'edge': bet['edge'],
                'won': bet['won'],
                'profit': bet['profit']
            })
    
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved detailed results to {output_file}")
    
    # Check for concerning patterns
    for name, results in all_results.items():
        if results['roi'] > 1000:
            logger.warning(f"\n⚠️ WARNING: {results['strategy']} shows unrealistic ROI of {results['roi']:.1f}%")
            logger.warning("This might indicate:")
            logger.warning("  - Look-ahead bias in features")
            logger.warning("  - Data leakage")
            logger.warning("  - Overfitting on test set")
            logger.warning("Consider reviewing the model training pipeline")


if __name__ == "__main__":
    main()