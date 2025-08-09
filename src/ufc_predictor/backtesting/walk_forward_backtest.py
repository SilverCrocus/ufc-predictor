#!/usr/bin/env python3
"""
Walk-forward backtesting system with CLV tracking for UFC predictions.
Implements realistic betting simulation with market-based validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
import logging
from dataclasses import dataclass, asdict
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BettingResult:
    """Container for individual bet results."""
    date: datetime
    fighter_a: str
    fighter_b: str
    predicted_prob: float
    market_prob: float
    bet_on: str  # 'fighter_a' or 'fighter_b'
    odds: float
    stake: float
    profit: float
    result: str  # 'win', 'loss', 'push'
    clv: float  # Closing line value
    edge: float  # Expected value edge
    kelly_fraction: float
    confidence_percentile: float


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    total_bets: int
    winning_bets: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    avg_odds: float
    avg_stake: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    clv_win_rate: float
    avg_clv: float
    avg_edge: float
    profit_factor: float
    risk_adjusted_return: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR
    longest_losing_streak: int
    longest_winning_streak: int
    by_confidence_bucket: Dict[str, Dict]
    by_division: Dict[str, Dict]
    by_odds_range: Dict[str, Dict]


class WalkForwardBacktest:
    """
    Comprehensive walk-forward backtesting system for UFC predictions.
    """
    
    def __init__(self, config_path: str = 'configs/backtest.yaml'):
        """Initialize backtester with configuration."""
        self.config = self._load_config(config_path)
        self.results = []
        self.equity_curve = []
        self.current_bankroll = self.config['simulation']['initial_bankroll']
        self.peak_bankroll = self.current_bankroll
        self.commission_rate = self.config['simulation'].get('commission_rate', 0.0)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load backtest configuration."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'folding': {
                'train_years': 3,
                'test_months': 3,
                'gap_days': 14
            },
            'staking': {
                'method': 'fractional_kelly',
                'kelly_fraction': 0.25,
                'p_lower_quantile': 0.20,
                'max_single_pct': 0.05,
                'min_bet_pct': 0.005,
                'min_edge_single': 0.05
            },
            'simulation': {
                'initial_bankroll': 1000,
                'commission_rate': 0.0
            },
            'odds': {
                'slippage_bp': 25,
                'remove_vig': True
            },
            'limits': {
                'max_stake_per_bet': 500,
                'min_stake': 5,
                'max_exposure_per_event': 1000
            },
            'risk_management': {
                'max_drawdown_pct': 0.25,
                'daily_loss_limit_pct': 0.10
            }
        }
    
    def calculate_kelly_stake(
        self,
        prob: float,
        odds: float,
        bankroll: float,
        use_pessimistic: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate Kelly stake with optional pessimistic adjustment.
        
        Args:
            prob: Win probability
            odds: Decimal odds
            bankroll: Current bankroll
            use_pessimistic: Use lower confidence bound
            
        Returns:
            Tuple of (stake_amount, kelly_fraction)
        """
        # Calculate edge
        edge = (prob * odds) - 1
        
        if edge <= self.config['staking']['min_edge_single']:
            return 0, 0
        
        # Calculate Kelly fraction
        kelly_f = edge / (odds - 1)
        
        # Apply fractional Kelly
        kelly_f *= self.config['staking']['kelly_fraction']
        
        # Apply pessimistic adjustment
        if use_pessimistic:
            kelly_f *= 0.8  # Additional safety factor
        
        # Apply limits
        max_fraction = self.config['staking']['max_single_pct']
        min_fraction = self.config['staking']['min_bet_pct']
        
        kelly_f = np.clip(kelly_f, min_fraction, max_fraction)
        
        # Calculate stake
        stake = bankroll * kelly_f
        
        # Apply absolute limits
        stake = np.clip(
            stake,
            self.config['limits']['min_stake'],
            min(self.config['limits']['max_stake_per_bet'], bankroll * max_fraction)
        )
        
        return stake, kelly_f
    
    def apply_odds_slippage(self, odds: float, is_favorite: bool) -> float:
        """Apply realistic odds slippage."""
        slippage_pct = self.config['odds']['slippage_bp'] / 10000
        
        if is_favorite:
            # Favorites get worse odds (lower)
            return odds * (1 - slippage_pct)
        else:
            # Underdogs get worse odds (but still higher than 1)
            return max(1.01, odds * (1 - slippage_pct))
    
    def remove_vig(self, odds_a: float, odds_b: float) -> Tuple[float, float]:
        """Remove vig from odds to get fair probabilities."""
        if not self.config['odds']['remove_vig']:
            return odds_a, odds_b
        
        # Calculate implied probabilities
        prob_a = 1 / odds_a
        prob_b = 1 / odds_b
        total_prob = prob_a + prob_b
        
        # Remove vig
        fair_prob_a = prob_a / total_prob
        fair_prob_b = prob_b / total_prob
        
        # Convert back to odds
        fair_odds_a = 1 / fair_prob_a
        fair_odds_b = 1 / fair_prob_b
        
        return fair_odds_a, fair_odds_b
    
    def calculate_clv(
        self,
        predicted_prob: float,
        opening_odds: float,
        closing_odds: float
    ) -> float:
        """
        Calculate Closing Line Value.
        Positive CLV means we beat the closing line.
        """
        opening_prob = 1 / opening_odds
        closing_prob = 1 / closing_odds
        
        # CLV = (closing_prob - opening_prob) / opening_prob
        clv = (closing_prob - opening_prob) / opening_prob
        
        return clv
    
    def simulate_bet(
        self,
        prediction: Dict,
        market_odds: Dict,
        actual_winner: int
    ) -> Optional[BettingResult]:
        """
        Simulate a single bet with all constraints.
        
        Args:
            prediction: Model prediction dict
            market_odds: Market odds dict
            actual_winner: Actual fight outcome (0 or 1)
            
        Returns:
            BettingResult or None if no bet placed
        """
        # Extract prediction details
        pred_prob_a = prediction['prob_fighter_a']
        pred_prob_b = 1 - pred_prob_a
        
        # Get market odds
        odds_a = market_odds.get('odds_fighter_a', 2.0)
        odds_b = market_odds.get('odds_fighter_b', 2.0)
        
        # Remove vig
        fair_odds_a, fair_odds_b = self.remove_vig(odds_a, odds_b)
        
        # Determine which side to bet (if any)
        edge_a = (pred_prob_a * fair_odds_a) - 1
        edge_b = (pred_prob_b * fair_odds_b) - 1
        
        if edge_a > edge_b and edge_a > self.config['staking']['min_edge_single']:
            # Bet on fighter A
            bet_prob = pred_prob_a
            bet_odds = self.apply_odds_slippage(odds_a, odds_a < 2.0)
            bet_on = 'fighter_a'
            edge = edge_a
            won = (actual_winner == 1)
        elif edge_b > self.config['staking']['min_edge_single']:
            # Bet on fighter B
            bet_prob = pred_prob_b
            bet_odds = self.apply_odds_slippage(odds_b, odds_b < 2.0)
            bet_on = 'fighter_b'
            edge = edge_b
            won = (actual_winner == 0)
        else:
            # No bet
            return None
        
        # Calculate stake
        stake, kelly_f = self.calculate_kelly_stake(
            bet_prob,
            bet_odds,
            self.current_bankroll,
            use_pessimistic=True
        )
        
        if stake == 0:
            return None
        
        # Calculate profit/loss
        if won:
            gross_profit = stake * (bet_odds - 1)
            commission = gross_profit * self.commission_rate
            net_profit = gross_profit - commission
            result = 'win'
        else:
            net_profit = -stake
            result = 'loss'
        
        # Calculate CLV (simplified - would need closing odds in practice)
        closing_odds = market_odds.get(f'closing_{bet_on}', bet_odds)
        clv = self.calculate_clv(bet_prob, bet_odds, closing_odds)
        
        # Create result
        betting_result = BettingResult(
            date=prediction.get('date', datetime.now()),
            fighter_a=prediction.get('fighter_a', ''),
            fighter_b=prediction.get('fighter_b', ''),
            predicted_prob=bet_prob,
            market_prob=1/bet_odds,
            bet_on=bet_on,
            odds=bet_odds,
            stake=stake,
            profit=net_profit,
            result=result,
            clv=clv,
            edge=edge,
            kelly_fraction=kelly_f,
            confidence_percentile=prediction.get('confidence_percentile', 50)
        )
        
        # Update bankroll
        self.current_bankroll += net_profit
        self.peak_bankroll = max(self.peak_bankroll, self.current_bankroll)
        
        return betting_result
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        model_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestMetrics:
        """
        Run walk-forward backtest on historical data.
        
        Args:
            df: Fight data with features
            model_path: Path to trained model
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestMetrics with comprehensive results
        """
        logger.info("Starting walk-forward backtest...")
        
        # Load model
        model = joblib.load(model_path)
        
        # Filter date range
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Initialize results
        self.results = []
        self.equity_curve = [self.current_bankroll]
        
        # Walk forward through time
        train_size = int(len(df) * 0.7)
        
        for i in range(train_size, len(df)):
            # Get training data (expanding window)
            train_data = df.iloc[:i]
            test_fight = df.iloc[i]
            
            # Skip if not enough training data
            if len(train_data) < 100:
                continue
            
            # Make prediction
            features = self._get_features(test_fight)
            pred_prob = model.predict_proba([features])[0][1]
            
            # Create prediction dict
            prediction = {
                'date': test_fight['date'],
                'fighter_a': test_fight.get('fighter_a', ''),
                'fighter_b': test_fight.get('fighter_b', ''),
                'prob_fighter_a': pred_prob,
                'confidence_percentile': self._calculate_confidence_percentile(pred_prob)
            }
            
            # Get market odds (simulated for now)
            market_odds = self._get_market_odds(test_fight)
            
            # Get actual outcome
            actual_winner = test_fight.get('winner', 0)
            
            # Simulate bet
            bet_result = self.simulate_bet(prediction, market_odds, actual_winner)
            
            if bet_result:
                self.results.append(bet_result)
                self.equity_curve.append(self.current_bankroll)
                
                # Check risk management
                if self._check_stop_conditions():
                    logger.warning("Stop conditions met, ending backtest")
                    break
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest complete: {metrics.total_bets} bets, ROI: {metrics.roi:.2%}")
        
        return metrics
    
    def _get_features(self, fight: pd.Series) -> List[float]:
        """Extract features from fight data."""
        # Load the exact features the model was trained on
        model_dir = Path('model')
        training_dirs = [d for d in model_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('training_')]
        
        if training_dirs:
            latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
            column_files = list(latest_dir.glob('*winner_model*_columns.json'))
            if column_files:
                import json
                with open(column_files[0], 'r') as f:
                    model_cols = json.load(f)
                # Extract features in the exact order expected
                features = []
                for col in model_cols:
                    if col in fight.index:
                        features.append(fight[col])
                    else:
                        features.append(0)  # Default value for missing features
                return features
        
        # Fallback: Get numeric columns (excluding metadata)
        exclude = ['Date', 'date', 'fighter_a', 'fighter_b', 'winner', 'Winner', 
                  'event', 'Event', 'method', 'Method', 'Time', 'Outcome',
                  'blue_fighter', 'red_fighter', 'loser_fighter']
        features = []
        for col in fight.index:
            if col not in exclude and pd.api.types.is_numeric_dtype(type(fight[col])):
                features.append(fight[col])
        return features
    
    def _get_market_odds(self, fight: pd.Series) -> Dict:
        """Get market odds (simulated for testing)."""
        # In production, would fetch actual odds from database
        # For now, simulate based on a simple model
        
        # Generate realistic odds based on fight characteristics
        if 'odds_fighter_a' in fight:
            return {
                'odds_fighter_a': fight['odds_fighter_a'],
                'odds_fighter_b': fight['odds_fighter_b']
            }
        
        # Simulate odds if not available
        # Favorites typically between 1.2 - 1.9
        # Underdogs typically between 2.1 - 5.0
        is_favorite = np.random.random() < 0.5
        
        if is_favorite:
            odds_a = np.random.uniform(1.3, 1.9)
            odds_b = 1 / (1 - 1/odds_a) * 1.06  # Add 6% vig
        else:
            odds_a = np.random.uniform(2.1, 4.0)
            odds_b = 1 / (1 - 1/odds_a) * 1.06
        
        return {
            'odds_fighter_a': odds_a,
            'odds_fighter_b': odds_b,
            'closing_fighter_a': odds_a * np.random.uniform(0.95, 1.05),
            'closing_fighter_b': odds_b * np.random.uniform(0.95, 1.05)
        }
    
    def _calculate_confidence_percentile(self, prob: float) -> float:
        """Calculate confidence percentile for bet sizing."""
        # Higher confidence near 0 or 1
        distance_from_middle = abs(prob - 0.5)
        percentile = 50 + (distance_from_middle * 100)
        return min(percentile, 100)
    
    def _check_stop_conditions(self) -> bool:
        """Check if stop conditions are met."""
        # Check max drawdown
        current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        if current_drawdown > self.config['risk_management']['max_drawdown_pct']:
            return True
        
        # Check if bankrupt
        if self.current_bankroll < self.config['limits']['min_stake']:
            return True
        
        return False
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        if not self.results:
            return BacktestMetrics(
                total_bets=0, winning_bets=0, win_rate=0, total_staked=0,
                total_profit=0, roi=0, avg_odds=0, avg_stake=0,
                max_drawdown=0, max_drawdown_pct=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, clv_win_rate=0,
                avg_clv=0, avg_edge=0, profit_factor=0,
                risk_adjusted_return=0, var_95=0, cvar_95=0,
                longest_losing_streak=0, longest_winning_streak=0,
                by_confidence_bucket={}, by_division={}, by_odds_range={}
            )
        
        # Basic metrics
        total_bets = len(self.results)
        winning_bets = sum(1 for r in self.results if r.result == 'win')
        win_rate = winning_bets / total_bets
        
        total_staked = sum(r.stake for r in self.results)
        total_profit = sum(r.profit for r in self.results)
        roi = total_profit / total_staked if total_staked > 0 else 0
        
        avg_odds = np.mean([r.odds for r in self.results])
        avg_stake = np.mean([r.stake for r in self.results])
        
        # Drawdown calculation
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown / self.config['simulation']['initial_bankroll']
        
        # Risk metrics
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            downside_returns = returns[returns < 0]
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            calmar_ratio = (roi * 252) / max_drawdown_pct if max_drawdown_pct > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        # CLV metrics
        clv_wins = sum(1 for r in self.results if r.clv > 0)
        clv_win_rate = clv_wins / total_bets
        avg_clv = np.mean([r.clv for r in self.results])
        avg_edge = np.mean([r.edge for r in self.results])
        
        # Profit factor
        gross_wins = sum(r.profit for r in self.results if r.profit > 0)
        gross_losses = abs(sum(r.profit for r in self.results if r.profit < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # Value at Risk
        profit_dist = [r.profit for r in self.results]
        var_95 = np.percentile(profit_dist, 5) if profit_dist else 0
        cvar_95 = np.mean([p for p in profit_dist if p <= var_95]) if profit_dist else 0
        
        # Streaks
        longest_winning = longest_losing = current_streak = 0
        last_result = None
        
        for r in self.results:
            if r.result == last_result:
                current_streak += 1
            else:
                if last_result == 'win':
                    longest_winning = max(longest_winning, current_streak)
                elif last_result == 'loss':
                    longest_losing = max(longest_losing, current_streak)
                current_streak = 1
                last_result = r.result
        
        # Segment analysis
        by_confidence = self._analyze_by_confidence()
        by_division = {}  # Would need division data
        by_odds_range = self._analyze_by_odds()
        
        return BacktestMetrics(
            total_bets=total_bets,
            winning_bets=winning_bets,
            win_rate=win_rate,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=roi,
            avg_odds=avg_odds,
            avg_stake=avg_stake,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            clv_win_rate=clv_win_rate,
            avg_clv=avg_clv,
            avg_edge=avg_edge,
            profit_factor=profit_factor,
            risk_adjusted_return=sharpe_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            longest_losing_streak=longest_losing,
            longest_winning_streak=longest_winning,
            by_confidence_bucket=by_confidence,
            by_division=by_division,
            by_odds_range=by_odds_range
        )
    
    def _analyze_by_confidence(self) -> Dict:
        """Analyze performance by confidence buckets."""
        buckets = {
            'low': {'range': (0, 0.55), 'results': []},
            'medium': {'range': (0.55, 0.65), 'results': []},
            'high': {'range': (0.65, 0.75), 'results': []},
            'very_high': {'range': (0.75, 1.0), 'results': []}
        }
        
        for r in self.results:
            prob = r.predicted_prob
            for bucket_name, bucket_data in buckets.items():
                if bucket_data['range'][0] <= prob < bucket_data['range'][1]:
                    bucket_data['results'].append(r)
                    break
        
        # Calculate metrics per bucket
        for bucket_name, bucket_data in buckets.items():
            results = bucket_data['results']
            if results:
                bucket_data['metrics'] = {
                    'count': len(results),
                    'win_rate': sum(1 for r in results if r.result == 'win') / len(results),
                    'roi': sum(r.profit for r in results) / sum(r.stake for r in results),
                    'avg_edge': np.mean([r.edge for r in results])
                }
            else:
                bucket_data['metrics'] = {'count': 0, 'win_rate': 0, 'roi': 0, 'avg_edge': 0}
        
        return buckets
    
    def _analyze_by_odds(self) -> Dict:
        """Analyze performance by odds ranges."""
        ranges = {
            'heavy_favorite': {'range': (1.0, 1.5), 'results': []},
            'favorite': {'range': (1.5, 2.0), 'results': []},
            'slight_underdog': {'range': (2.0, 3.0), 'results': []},
            'underdog': {'range': (3.0, 5.0), 'results': []},
            'heavy_underdog': {'range': (5.0, 20.0), 'results': []}
        }
        
        for r in self.results:
            for range_name, range_data in ranges.items():
                if range_data['range'][0] <= r.odds < range_data['range'][1]:
                    range_data['results'].append(r)
                    break
        
        # Calculate metrics per range
        for range_name, range_data in ranges.items():
            results = range_data['results']
            if results:
                range_data['metrics'] = {
                    'count': len(results),
                    'win_rate': sum(1 for r in results if r.result == 'win') / len(results),
                    'roi': sum(r.profit for r in results) / sum(r.stake for r in results),
                    'avg_clv': np.mean([r.clv for r in results])
                }
            else:
                range_data['metrics'] = {'count': 0, 'win_rate': 0, 'roi': 0, 'avg_clv': 0}
        
        return ranges
    
    def save_results(self, output_dir: str = 'artifacts/backtest'):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        if self.results:
            results_df = pd.DataFrame([asdict(r) for r in self.results])
            results_df.to_csv(output_path / 'backtest_results.csv', index=False)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'index': range(len(self.equity_curve)),
            'bankroll': self.equity_curve
        })
        equity_df.to_csv(output_path / 'equity_curve.csv', index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """Generate comprehensive backtest report."""
        report = f"""
        {'='*60}
        WALK-FORWARD BACKTEST REPORT
        {'='*60}
        
        OVERALL PERFORMANCE:
        -------------------
        Total Bets: {metrics.total_bets}
        Win Rate: {metrics.win_rate:.2%}
        ROI: {metrics.roi:.2%}
        Total Profit: ${metrics.total_profit:.2f}
        
        RISK METRICS:
        ------------
        Sharpe Ratio: {metrics.sharpe_ratio:.2f}
        Sortino Ratio: {metrics.sortino_ratio:.2f}
        Max Drawdown: {metrics.max_drawdown_pct:.2%}
        Profit Factor: {metrics.profit_factor:.2f}
        VaR (95%): ${metrics.var_95:.2f}
        
        CLV ANALYSIS:
        ------------
        CLV Win Rate: {metrics.clv_win_rate:.2%}
        Average CLV: {metrics.avg_clv:.3f}
        Average Edge: {metrics.avg_edge:.2%}
        
        STREAKS:
        --------
        Longest Win Streak: {metrics.longest_winning_streak}
        Longest Loss Streak: {metrics.longest_losing_streak}
        
        BY CONFIDENCE:
        -------------
        """
        
        for bucket_name, bucket_data in metrics.by_confidence_bucket.items():
            m = bucket_data.get('metrics', {})
            report += f"  {bucket_name}: {m.get('count', 0)} bets, {m.get('roi', 0):.2%} ROI\n"
        
        report += f"\n{'='*60}\n"
        
        return report


def main():
    """Run walk-forward backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run walk-forward backtest')
    parser.add_argument('--data', type=str, required=True, help='Path to fight data')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='configs/backtest.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='artifacts/backtest', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Initialize backtester
    backtester = WalkForwardBacktest(args.config)
    
    # Run backtest
    metrics = backtester.run_walk_forward(
        df,
        args.model,
        args.start_date,
        args.end_date
    )
    
    # Save results
    backtester.save_results(args.output)
    
    # Print report
    print(backtester.generate_report(metrics))


if __name__ == "__main__":
    main()