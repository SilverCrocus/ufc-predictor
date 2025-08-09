"""
Walk-forward backtesting simulator for UFC betting strategies.
Simulates realistic betting with slippage, limits, and portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import yaml
from ..evaluation.temporal_split import TemporalWalkForwardSplitter
from ..evaluation.metrics import UFCMetricsCalculator, calculate_portfolio_metrics
from .staking import KellyStaking, PortfolioOptimizer
from .odds_utils import VigCalculator, calculate_expected_value
from .correlation import ParlayCorrelationEstimator

logger = logging.getLogger(__name__)


@dataclass
class BetResult:
    """Container for individual bet result."""
    date: datetime
    event: str
    fighter: str
    bet_odds: float
    close_odds: float
    stake: float
    probability: float
    outcome: bool
    profit: float
    running_bankroll: float
    bet_type: str = 'single'
    confidence: Optional[float] = None


@dataclass
class BacktestResult:
    """Container for complete backtest results."""
    bet_history: List[BetResult]
    performance_metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    fold_results: List[Dict[str, Any]]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WalkForwardSimulator:
    """
    Walk-forward backtesting simulator for UFC betting.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize simulator with configuration.
        
        Args:
            config_path: Path to backtest configuration YAML
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.temporal_splitter = TemporalWalkForwardSplitter(
            train_years=self.config['folding']['train_years'],
            test_months=self.config['folding']['test_years'] * 12,
            gap_days=self.config['folding']['gap_days']
        )
        
        self.kelly_staking = KellyStaking(
            kelly_fraction=self.config['staking']['kelly_fraction'],
            max_bet_pct=self.config['staking']['max_single_pct'],
            min_bet_pct=self.config['staking'].get('min_bet_pct', 0.005),
            use_pessimistic=True,
            confidence_level=self.config['staking']['p_lower_quantile']
        )
        
        self.portfolio_optimizer = PortfolioOptimizer(
            max_bets=self.config['portfolio'].get('max_bets_per_card', 5),
            max_exposure_pct=self.config['staking']['max_single_pct'] * 
                           self.config['portfolio'].get('max_bets_per_card', 5)
        )
        
        self.parlay_estimator = ParlayCorrelationEstimator() if self.config['staking']['allow_parlays'] else None
        
        self.metrics_calculator = UFCMetricsCalculator()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided."""
        return {
            'folding': {
                'train_years': 5,
                'test_years': 1,
                'gap_days': 14
            },
            'odds': {
                'use': 'open',
                'slippage_bp': 25,
                'remove_vig': True
            },
            'staking': {
                'kelly_fraction': 0.25,
                'p_lower_quantile': 0.20,
                'max_single_pct': 0.05,
                'min_bet_pct': 0.005,
                'min_edge_single': 0.05,
                'allow_parlays': False
            },
            'portfolio': {
                'enabled': True,
                'max_bets_per_card': 5
            },
            'limits': {
                'max_stake_per_bet': 500,
                'min_stake': 5
            },
            'simulation': {
                'initial_bankroll': 1000,
                'commission_rate': 0.0,
                'random_seed': 42
            },
            'risk_management': {
                'max_drawdown_pct': 0.25,
                'daily_loss_limit_pct': 0.10
            }
        }
    
    def run_backtest(
        self,
        fights_df: pd.DataFrame,
        model_pipeline,
        feature_pipeline=None
    ) -> BacktestResult:
        """
        Run complete walk-forward backtest.
        
        Args:
            fights_df: DataFrame with fight data and odds
            model_pipeline: Model training/prediction pipeline
            feature_pipeline: Optional feature engineering pipeline
            
        Returns:
            BacktestResult with complete analysis
        """
        # Set random seed
        np.random.seed(self.config['simulation']['random_seed'])
        
        # Create temporal folds
        folds = self.temporal_splitter.make_rolling_folds(fights_df)
        
        # Initialize tracking
        bet_history = []
        fold_results = []
        current_bankroll = self.config['simulation']['initial_bankroll']
        peak_bankroll = current_bankroll
        
        logger.info(f"Starting backtest with {len(folds)} folds")
        
        for fold_idx, fold in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            # Split data
            train_data = fights_df.iloc[fold.train_indices]
            test_data = fights_df.iloc[fold.test_indices]
            
            # Apply feature engineering if provided
            if feature_pipeline:
                train_data = feature_pipeline.transform(train_data)
                test_data = feature_pipeline.transform(test_data)
            
            # Train model on training data
            model = model_pipeline.fit(train_data)
            
            # Generate predictions for test period
            predictions = model.predict_proba(test_data)
            
            # Apply calibration if configured
            if hasattr(model_pipeline, 'calibrator'):
                predictions = model_pipeline.calibrator.apply_calibration(predictions)
            
            # Simulate betting on test period
            fold_bets, fold_metrics = self._simulate_fold_betting(
                test_data,
                predictions,
                current_bankroll,
                fold.test_start,
                fold.test_end
            )
            
            # Update bankroll
            fold_profit = sum(bet.profit for bet in fold_bets)
            current_bankroll += fold_profit
            peak_bankroll = max(peak_bankroll, current_bankroll)
            
            # Check risk limits
            drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
            if drawdown > self.config['risk_management']['max_drawdown_pct']:
                logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
                break
            
            # Store results
            bet_history.extend(fold_bets)
            fold_results.append({
                'fold_id': fold_idx,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'n_bets': len(fold_bets),
                'profit': fold_profit,
                'roi': fold_profit / sum(bet.stake for bet in fold_bets) if fold_bets else 0,
                'metrics': fold_metrics
            })
        
        # Calculate overall performance metrics
        performance_metrics = self._calculate_performance_metrics(
            bet_history,
            self.config['simulation']['initial_bankroll']
        )
        
        # Create equity curve
        equity_curve = self._create_equity_curve(
            bet_history,
            self.config['simulation']['initial_bankroll']
        )
        
        return BacktestResult(
            bet_history=bet_history,
            performance_metrics=performance_metrics,
            equity_curve=equity_curve,
            fold_results=fold_results,
            config=self.config,
            metadata={
                'n_folds': len(folds),
                'total_fights': len(fights_df),
                'date_range': (fights_df['date'].min(), fights_df['date'].max())
            }
        )
    
    def _simulate_fold_betting(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        current_bankroll: float,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[List[BetResult], Dict[str, float]]:
        """
        Simulate betting for a single fold.
        
        Returns:
            Tuple of (bet_results, fold_metrics)
        """
        bet_results = []
        running_bankroll = current_bankroll
        
        # Group by event/date
        test_data['prediction'] = predictions[:, 1] if predictions.ndim > 1 else predictions
        
        for event_date in test_data['date'].unique():
            event_fights = test_data[test_data['date'] == event_date].copy()
            
            # Check daily loss limit
            daily_loss = current_bankroll - running_bankroll
            if daily_loss > current_bankroll * self.config['risk_management']['daily_loss_limit_pct']:
                logger.info(f"Daily loss limit reached for {event_date}")
                continue
            
            # Get betting opportunities
            opportunities = self._identify_opportunities(event_fights, running_bankroll)
            
            if len(opportunities) == 0:
                continue
            
            # Portfolio optimization if enabled
            if self.config['portfolio']['enabled'] and len(opportunities) > 1:
                portfolio = self.portfolio_optimizer.optimize_portfolio(
                    opportunities,
                    running_bankroll
                )
                selected_bets = portfolio['bets']
            else:
                # Simple selection
                selected_bets = opportunities.nlargest(
                    self.config['portfolio'].get('max_bets_per_card', 5),
                    'edge'
                ).to_dict('records')
            
            # Place bets with slippage
            for bet in selected_bets:
                # Apply slippage
                bet_odds = bet['odds']
                slippage = self.config['odds']['slippage_bp'] / 10000
                actual_odds = bet_odds * (1 - slippage)
                
                # Apply limits
                stake = min(bet['stake'], self.config['limits']['max_stake_per_bet'])
                stake = max(stake, self.config['limits']['min_stake'])
                
                # Determine outcome
                actual_winner = event_fights[
                    event_fights['fighter_a'] == bet['fighter']
                ]['winner'].values[0] if bet['fighter'] in event_fights['fighter_a'].values else 0
                
                outcome = actual_winner == 1 if bet['fighter'] in event_fights['fighter_a'].values else actual_winner == 0
                
                # Calculate profit
                if outcome:
                    profit = stake * (actual_odds - 1) * (1 - self.config['simulation']['commission_rate'])
                else:
                    profit = -stake
                
                running_bankroll += profit
                
                # Store result
                bet_result = BetResult(
                    date=event_date,
                    event=event_fights.iloc[0].get('event', 'Unknown'),
                    fighter=bet['fighter'],
                    bet_odds=bet_odds,
                    close_odds=bet.get('close_odds', bet_odds),
                    stake=stake,
                    probability=bet['probability'],
                    outcome=outcome,
                    profit=profit,
                    running_bankroll=running_bankroll,
                    confidence=bet.get('confidence')
                )
                
                bet_results.append(bet_result)
        
        # Calculate fold metrics
        if bet_results:
            fold_metrics = {
                'n_bets': len(bet_results),
                'win_rate': sum(1 for b in bet_results if b.outcome) / len(bet_results),
                'total_staked': sum(b.stake for b in bet_results),
                'total_profit': sum(b.profit for b in bet_results),
                'roi': sum(b.profit for b in bet_results) / sum(b.stake for b in bet_results),
                'avg_odds': np.mean([b.bet_odds for b in bet_results]),
                'clv_rate': self._calculate_clv_rate(bet_results)
            }
        else:
            fold_metrics = {'n_bets': 0, 'win_rate': 0, 'roi': 0}
        
        return bet_results, fold_metrics
    
    def _identify_opportunities(
        self,
        event_fights: pd.DataFrame,
        bankroll: float
    ) -> pd.DataFrame:
        """
        Identify betting opportunities from predictions.
        """
        opportunities = []
        
        for _, fight in event_fights.iterrows():
            # Get prediction and odds
            prob = fight['prediction']
            odds = fight.get('odds_a', 2.0)  # Default odds if missing
            
            # Remove vig if configured
            if self.config['odds']['remove_vig']:
                vig_calc = VigCalculator()
                fair_prob_a, _ = vig_calc.remove_vig_two_way(
                    odds,
                    fight.get('odds_b', 2.0),
                    method='balanced'
                )
                # Adjust odds based on fair probability
                fair_odds = 1 / fair_prob_a if fair_prob_a > 0 else odds
            else:
                fair_odds = odds
            
            # Calculate edge
            edge = (prob * fair_odds) - 1
            
            # Check minimum edge
            if edge < self.config['staking']['min_edge_single']:
                continue
            
            # Calculate stake
            stake_rec = self.kelly_staking.calculate_kelly_stake(
                prob=prob,
                odds=fair_odds,
                bankroll=bankroll
            )
            
            if stake_rec.bet_amount > 0:
                opportunities.append({
                    'fighter': fight['fighter_a'],
                    'probability': prob,
                    'odds': fair_odds,
                    'edge': edge,
                    'stake': stake_rec.bet_amount,
                    'confidence': stake_rec.confidence_level
                })
        
        return pd.DataFrame(opportunities)
    
    def _calculate_clv_rate(self, bet_results: List[BetResult]) -> float:
        """Calculate closing line value win rate."""
        clv_wins = 0
        for bet in bet_results:
            if bet.close_odds > bet.bet_odds:
                clv_wins += 1
        
        return clv_wins / len(bet_results) if bet_results else 0
    
    def _calculate_performance_metrics(
        self,
        bet_history: List[BetResult],
        initial_bankroll: float
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not bet_history:
            return {'n_bets': 0, 'roi': 0}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'date': b.date,
                'stake': b.stake,
                'return': b.profit + b.stake,
                'profit': b.profit,
                'odds': b.bet_odds,
                'outcome': b.outcome
            }
            for b in bet_history
        ])
        
        # Use portfolio metrics calculator
        portfolio_metrics = calculate_portfolio_metrics(df, initial_bankroll)
        
        # Add additional metrics
        portfolio_metrics.update({
            'n_bets': len(bet_history),
            'win_rate': sum(1 for b in bet_history if b.outcome) / len(bet_history),
            'avg_stake': np.mean([b.stake for b in bet_history]),
            'avg_odds': np.mean([b.bet_odds for b in bet_history]),
            'clv_rate': self._calculate_clv_rate(bet_history),
            'final_bankroll': bet_history[-1].running_bankroll if bet_history else initial_bankroll
        })
        
        return portfolio_metrics
    
    def _create_equity_curve(
        self,
        bet_history: List[BetResult],
        initial_bankroll: float
    ) -> pd.DataFrame:
        """Create equity curve DataFrame."""
        if not bet_history:
            return pd.DataFrame()
        
        equity_data = []
        
        for bet in bet_history:
            equity_data.append({
                'date': bet.date,
                'bankroll': bet.running_bankroll,
                'drawdown': (bet.running_bankroll - initial_bankroll) / initial_bankroll,
                'bet_count': len(equity_data) + 1
            })
        
        equity_df = pd.DataFrame(equity_data)
        
        # Calculate running max and drawdown
        equity_df['peak'] = equity_df['bankroll'].expanding().max()
        equity_df['drawdown_pct'] = (equity_df['bankroll'] - equity_df['peak']) / equity_df['peak']
        
        return equity_df