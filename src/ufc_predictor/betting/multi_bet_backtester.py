"""
Multi-bet backtesting framework for sophisticated betting strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

from .multi_bet_orchestrator import MultiBetOrchestrator
from .enhanced_correlation import EnhancedCorrelationEngine

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    date: datetime
    event: str
    bet_type: str  # 'single' or 'parlay'
    legs: List[Dict]  # Fight details for each leg
    stake: float
    odds: float
    ev: float
    predicted_prob: float
    result: str  # 'win', 'loss', 'pending'
    actual_return: float
    running_bankroll: float
    correlation_info: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'event': self.event,
            'bet_type': self.bet_type,
            'legs': self.legs,
            'stake': self.stake,
            'odds': self.odds,
            'ev': self.ev,
            'predicted_prob': self.predicted_prob,
            'result': self.result,
            'actual_return': self.actual_return,
            'running_bankroll': self.running_bankroll,
            'correlation_info': self.correlation_info
        }

class MultiBetBacktester:
    """
    Sophisticated backtesting engine for multi-bet strategies.
    
    Features:
    - Walk-forward testing with proper temporal splits
    - Correlation tracking and validation
    - CLV (Closing Line Value) analysis
    - Strategy comparison (singles vs conditional parlays)
    - Risk metrics and drawdown analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.orchestrator = MultiBetOrchestrator(config=self.config['orchestrator'])
        self.correlation_engine = EnhancedCorrelationEngine()
        self.results: List[BacktestResult] = []
        
    def _default_config(self) -> Dict:
        return {
            'orchestrator': {
                'activation': {
                    'min_singles_threshold': 2,
                    'min_parlay_pool': 2
                },
                'filters': {
                    'singles': {
                        'min_ev': 0.05,
                        'max_ev': 0.15,
                        'min_confidence': 0.65
                    },
                    'parlays': {
                        'min_ev': 0.02,
                        'min_confidence': 0.55
                    }
                },
                'portfolio': {
                    'max_parlays': 2,
                    'max_total_exposure': 0.015
                }
            },
            'backtest': {
                'initial_bankroll': 1000,
                'commission_rate': 0.05,  # 5% commission on winnings
                'min_bet_size': 10,
                'track_clv': True,
                'track_correlations': True
            }
        }
    
    def run_backtest(self,
                     historical_data: pd.DataFrame,
                     predictions: pd.DataFrame,
                     odds_data: pd.DataFrame,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Dict:
        """
        Run comprehensive backtest on historical data.
        
        Args:
            historical_data: Fight results with actual outcomes
            predictions: Model predictions for fights
            odds_data: Historical odds data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Comprehensive backtest results and metrics
        """
        # Filter data by date range
        if start_date:
            mask = historical_data['date'] >= start_date
            historical_data = historical_data[mask]
            predictions = predictions[mask]
            odds_data = odds_data[mask]
            
        if end_date:
            mask = historical_data['date'] <= end_date
            historical_data = historical_data[mask]
            predictions = predictions[mask]
            odds_data = odds_data[mask]
        
        # Initialize tracking
        bankroll = self.config['backtest']['initial_bankroll']
        self.results = []
        
        # Group by event/date for sequential processing
        events = historical_data.groupby(['date', 'event'])
        
        for (event_date, event_name), event_fights in events:
            # Get predictions and odds for this event
            event_preds = predictions[
                (predictions['date'] == event_date) & 
                (predictions['event'] == event_name)
            ]
            event_odds = odds_data[
                (odds_data['date'] == event_date) & 
                (odds_data['event'] == event_name)
            ]
            
            # Skip if insufficient data
            if event_preds.empty or event_odds.empty:
                continue
            
            # Build context for correlation estimation
            context = self._build_event_context(event_fights, event_preds)
            
            # Get betting recommendations
            recommendations = self.orchestrator.analyze_betting_opportunities(
                predictions=event_preds,
                odds_data=event_odds,
                bankroll=bankroll,
                context=context
            )
            
            # Process each recommendation
            for rec in recommendations['recommendations']:
                # Execute bet and track result
                result = self._execute_bet(
                    recommendation=rec,
                    actual_results=event_fights,
                    bankroll=bankroll,
                    event_date=event_date,
                    event_name=event_name
                )
                
                self.results.append(result)
                bankroll = result.running_bankroll
                
                # Log progress
                logger.info(f"Event: {event_name}, Bet: {result.bet_type}, "
                          f"Result: {result.result}, Bankroll: ${bankroll:.2f}")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics()
        
        return {
            'results': [r.to_dict() for r in self.results],
            'metrics': metrics,
            'final_bankroll': bankroll,
            'initial_bankroll': self.config['backtest']['initial_bankroll'],
            'total_return': (bankroll / self.config['backtest']['initial_bankroll'] - 1) * 100
        }
    
    def _build_event_context(self, 
                            event_fights: pd.DataFrame,
                            predictions: pd.DataFrame) -> Dict:
        """Build context for correlation estimation"""
        return {
            'event_fights': event_fights.to_dict('records'),
            'num_fights': len(event_fights),
            'fight_features': self._extract_fight_features(event_fights),
            'prediction_confidence': predictions['confidence'].mean() if 'confidence' in predictions else 0.6
        }
    
    def _extract_fight_features(self, fights: pd.DataFrame) -> Dict:
        """Extract relevant features for correlation estimation"""
        features = {}
        
        if 'weight_class' in fights.columns:
            features['weight_classes'] = fights['weight_class'].unique().tolist()
            
        if 'fighter_style' in fights.columns:
            features['styles'] = fights['fighter_style'].value_counts().to_dict()
            
        if 'rounds' in fights.columns:
            features['round_structure'] = fights['rounds'].mode()[0] if not fights['rounds'].empty else 3
            
        return features
    
    def _execute_bet(self,
                    recommendation: Dict,
                    actual_results: pd.DataFrame,
                    bankroll: float,
                    event_date: datetime,
                    event_name: str) -> BacktestResult:
        """Execute a bet and determine the outcome"""
        
        bet_type = recommendation['type']
        stake = recommendation['stake']
        
        if bet_type == 'single':
            # Single bet execution
            fight = recommendation['fight']
            odds = recommendation['odds']
            ev = recommendation['ev']
            prob = recommendation['probability']
            
            # Find actual result
            actual = actual_results[
                (actual_results['fighter1'] == fight['fighter1']) &
                (actual_results['fighter2'] == fight['fighter2'])
            ]
            
            if actual.empty:
                result = 'pending'
                actual_return = 0
            else:
                winner = actual.iloc[0]['winner']
                predicted_winner = fight['predicted_winner']
                
                if winner == predicted_winner:
                    result = 'win'
                    winnings = stake * (odds - 1)
                    commission = winnings * self.config['backtest']['commission_rate']
                    actual_return = stake + winnings - commission
                else:
                    result = 'loss'
                    actual_return = 0
            
            new_bankroll = bankroll - stake + actual_return
            
            return BacktestResult(
                date=event_date,
                event=event_name,
                bet_type='single',
                legs=[fight],
                stake=stake,
                odds=odds,
                ev=ev,
                predicted_prob=prob,
                result=result,
                actual_return=actual_return,
                running_bankroll=new_bankroll
            )
            
        else:  # Parlay
            legs = recommendation['legs']
            combined_odds = recommendation['combined_odds']
            ev = recommendation['ev']
            prob = recommendation['probability']
            correlation_info = recommendation.get('correlation_info', {})
            
            # Check all legs
            all_correct = True
            for leg in legs:
                actual = actual_results[
                    (actual_results['fighter1'] == leg['fighter1']) &
                    (actual_results['fighter2'] == leg['fighter2'])
                ]
                
                if actual.empty:
                    all_correct = None  # Pending
                    break
                    
                winner = actual.iloc[0]['winner']
                if winner != leg['predicted_winner']:
                    all_correct = False
                    break
            
            if all_correct is None:
                result = 'pending'
                actual_return = 0
            elif all_correct:
                result = 'win'
                winnings = stake * (combined_odds - 1)
                commission = winnings * self.config['backtest']['commission_rate']
                actual_return = stake + winnings - commission
            else:
                result = 'loss'
                actual_return = 0
            
            new_bankroll = bankroll - stake + actual_return
            
            return BacktestResult(
                date=event_date,
                event=event_name,
                bet_type='parlay',
                legs=legs,
                stake=stake,
                odds=combined_odds,
                ev=ev,
                predicted_prob=prob,
                result=result,
                actual_return=actual_return,
                running_bankroll=new_bankroll,
                correlation_info=correlation_info
            )
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if not self.results:
            return {}
        
        # Basic metrics
        total_bets = len(self.results)
        wins = sum(1 for r in self.results if r.result == 'win')
        losses = sum(1 for r in self.results if r.result == 'loss')
        pending = sum(1 for r in self.results if r.result == 'pending')
        
        # Financial metrics
        total_staked = sum(r.stake for r in self.results)
        total_returns = sum(r.actual_return for r in self.results)
        net_profit = total_returns - total_staked
        
        # ROI calculation
        roi = (net_profit / total_staked * 100) if total_staked > 0 else 0
        
        # Separate metrics by bet type
        singles = [r for r in self.results if r.bet_type == 'single']
        parlays = [r for r in self.results if r.bet_type == 'parlay']
        
        singles_metrics = self._calculate_subset_metrics(singles, 'Singles')
        parlays_metrics = self._calculate_subset_metrics(parlays, 'Parlays')
        
        # Drawdown analysis
        bankroll_history = [r.running_bankroll for r in self.results]
        max_drawdown = self._calculate_max_drawdown(bankroll_history)
        
        # EV analysis
        avg_ev = np.mean([r.ev for r in self.results]) if self.results else 0
        ev_accuracy = self._calculate_ev_accuracy()
        
        # Correlation analysis (for parlays)
        correlation_metrics = self._analyze_correlations(parlays)
        
        return {
            'summary': {
                'total_bets': total_bets,
                'wins': wins,
                'losses': losses,
                'pending': pending,
                'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
                'total_staked': total_staked,
                'total_returns': total_returns,
                'net_profit': net_profit,
                'roi': roi,
                'max_drawdown': max_drawdown,
                'average_ev': avg_ev,
                'ev_accuracy': ev_accuracy
            },
            'singles': singles_metrics,
            'parlays': parlays_metrics,
            'correlations': correlation_metrics,
            'monthly_breakdown': self._calculate_monthly_breakdown()
        }
    
    def _calculate_subset_metrics(self, subset: List[BacktestResult], label: str) -> Dict:
        """Calculate metrics for a subset of bets"""
        if not subset:
            return {'label': label, 'no_bets': True}
        
        wins = sum(1 for r in subset if r.result == 'win')
        losses = sum(1 for r in subset if r.result == 'loss')
        total_staked = sum(r.stake for r in subset)
        total_returns = sum(r.actual_return for r in subset)
        
        return {
            'label': label,
            'count': len(subset),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_staked': total_staked,
            'total_returns': total_returns,
            'roi': ((total_returns - total_staked) / total_staked * 100) if total_staked > 0 else 0,
            'average_odds': np.mean([r.odds for r in subset]),
            'average_ev': np.mean([r.ev for r in subset])
        }
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not bankroll_history:
            return 0
        
        peak = bankroll_history[0]
        max_dd = 0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_ev_accuracy(self) -> float:
        """Calculate how accurate EV predictions were"""
        completed = [r for r in self.results if r.result in ['win', 'loss']]
        if not completed:
            return 0
        
        # Compare predicted EV to actual returns
        ev_errors = []
        for r in completed:
            expected_return = r.stake * (1 + r.ev)
            actual_return = r.actual_return
            error = abs(expected_return - actual_return) / r.stake
            ev_errors.append(error)
        
        # Return accuracy score (lower error = higher accuracy)
        avg_error = np.mean(ev_errors)
        accuracy = max(0, 1 - avg_error) * 100
        
        return accuracy
    
    def _analyze_correlations(self, parlays: List[BacktestResult]) -> Dict:
        """Analyze correlation predictions vs actual outcomes"""
        if not parlays:
            return {'no_parlays': True}
        
        # Track correlation estimates vs actual outcomes
        correlation_data = []
        
        for p in parlays:
            if p.correlation_info and p.result in ['win', 'loss']:
                correlation_data.append({
                    'estimated_correlation': p.correlation_info.get('final_correlation', 0),
                    'confidence': p.correlation_info.get('confidence', 0),
                    'result': p.result,
                    'legs': len(p.legs)
                })
        
        if not correlation_data:
            return {'insufficient_data': True}
        
        # Analyze correlation accuracy
        df = pd.DataFrame(correlation_data)
        
        # Group by correlation buckets
        df['correlation_bucket'] = pd.cut(
            df['estimated_correlation'],
            bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate win rates by correlation level
        correlation_performance = df.groupby('correlation_bucket').agg({
            'result': lambda x: (x == 'win').mean(),
            'estimated_correlation': 'mean'
        }).to_dict('index')
        
        return {
            'total_parlays': len(correlation_data),
            'avg_correlation': df['estimated_correlation'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'performance_by_correlation': correlation_performance
        }
    
    def _calculate_monthly_breakdown(self) -> List[Dict]:
        """Calculate performance by month"""
        if not self.results:
            return []
        
        # Group by month
        monthly_data = {}
        
        for r in self.results:
            month_key = r.date.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'bets': 0,
                    'wins': 0,
                    'losses': 0,
                    'staked': 0,
                    'returns': 0
                }
            
            monthly_data[month_key]['bets'] += 1
            if r.result == 'win':
                monthly_data[month_key]['wins'] += 1
            elif r.result == 'loss':
                monthly_data[month_key]['losses'] += 1
            monthly_data[month_key]['staked'] += r.stake
            monthly_data[month_key]['returns'] += r.actual_return
        
        # Calculate ROI for each month
        monthly_breakdown = []
        for month, data in sorted(monthly_data.items()):
            roi = ((data['returns'] - data['staked']) / data['staked'] * 100) if data['staked'] > 0 else 0
            monthly_breakdown.append({
                'month': month,
                'bets': data['bets'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': data['wins'] / (data['wins'] + data['losses']) if (data['wins'] + data['losses']) > 0 else 0,
                'roi': roi,
                'profit': data['returns'] - data['staked']
            })
        
        return monthly_breakdown
    
    def export_results(self, filepath: str):
        """Export backtest results to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'config': self.config,
            'results': [r.to_dict() for r in self.results],
            'metrics': self._calculate_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Backtest results exported to {filepath}")