"""
Comprehensive metrics for UFC prediction evaluation.
Includes classification, calibration, and betting performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    log_loss, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for comprehensive metrics results."""
    classification_metrics: Dict[str, float]
    calibration_metrics: Dict[str, float]
    betting_metrics: Dict[str, float]
    market_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class UFCMetricsCalculator:
    """
    Comprehensive metrics calculator for UFC predictions.
    """
    
    def __init__(self, decimal_precision: int = 4):
        """
        Initialize metrics calculator.
        
        Args:
            decimal_precision: Number of decimal places for rounding
        """
        self.decimal_precision = decimal_precision
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds_data: Optional[pd.DataFrame] = None,
        stakes: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MetricsResult:
        """
        Calculate all metrics comprehensively.
        
        Args:
            y_true: True labels (0 or 1)
            y_prob: Predicted probabilities
            odds_data: Optional DataFrame with odds information
            stakes: Optional array of bet stakes
            metadata: Optional metadata dictionary
            
        Returns:
            MetricsResult with all calculated metrics
        """
        # Classification metrics
        classification_metrics = self.calculate_classification_metrics(y_true, y_prob)
        
        # Calibration metrics
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_prob)
        
        # Betting metrics (if odds provided)
        betting_metrics = {}
        market_metrics = {}
        
        if odds_data is not None:
            if stakes is not None:
                betting_metrics = self.calculate_betting_metrics(
                    y_true, y_prob, odds_data, stakes
                )
            
            if 'close_odds' in odds_data.columns and 'bet_odds' in odds_data.columns:
                market_metrics = self.calculate_market_metrics(
                    odds_data['bet_odds'].values,
                    odds_data['close_odds'].values,
                    stakes
                )
        
        return MetricsResult(
            classification_metrics=classification_metrics,
            calibration_metrics=calibration_metrics,
            betting_metrics=betting_metrics,
            market_metrics=market_metrics,
            metadata=metadata or {}
        )
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            'log_loss': self._safe_metric(log_loss, y_true, y_prob),
            'auc': self._safe_metric(roc_auc_score, y_true, y_prob),
            'accuracy': self._safe_metric(accuracy_score, y_true, y_pred),
            'precision': self._safe_metric(precision_score, y_true, y_pred, zero_division=0),
            'recall': self._safe_metric(recall_score, y_true, y_pred, zero_division=0),
            'f1': self._safe_metric(f1_score, y_true, y_pred, zero_division=0),
        }
        
        # Add confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        return self._round_metrics(metrics)
    
    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration-specific metrics."""
        metrics = {
            'brier_score': self.brier_score(y_true, y_prob),
            'ece': self.expected_calibration_error(y_true, y_prob, n_bins),
            'mce': self.maximum_calibration_error(y_true, y_prob, n_bins),
            'reliability': self.reliability_score(y_true, y_prob, n_bins),
            'resolution': self.resolution_score(y_true, y_prob),
            'uncertainty': self.uncertainty_score(y_true)
        }
        
        # Brier score decomposition
        brier_components = self.brier_score_decomposition(y_true, y_prob)
        metrics.update(brier_components)
        
        return self._round_metrics(metrics)
    
    def brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score."""
        return np.mean((y_prob - y_true) ** 2)
    
    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        Lower is better, 0 is perfect calibration.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        Maximum deviation in any bin.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def reliability_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate reliability score (lower is better).
        Measures deviation from perfect calibration.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        weighted_deviation = 0
        total_weight = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            n_in_bin = in_bin.sum()
            
            if n_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                expected_accuracy = bin_centers[i]
                deviation = (accuracy_in_bin - expected_accuracy) ** 2
                weighted_deviation += deviation * n_in_bin
                total_weight += n_in_bin
        
        return weighted_deviation / total_weight if total_weight > 0 else 0
    
    def resolution_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate resolution score (higher is better).
        Measures how much predictions deviate from base rate.
        """
        base_rate = y_true.mean()
        return np.mean((y_prob - base_rate) ** 2)
    
    def uncertainty_score(self, y_true: np.ndarray) -> float:
        """
        Calculate uncertainty score.
        Inherent uncertainty in the outcome distribution.
        """
        base_rate = y_true.mean()
        return base_rate * (1 - base_rate)
    
    def brier_score_decomposition(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Decompose Brier score into reliability, resolution, and uncertainty.
        Brier = Reliability - Resolution + Uncertainty
        """
        reliability = self.reliability_score(y_true, y_prob)
        resolution = self.resolution_score(y_true, y_prob)
        uncertainty = self.uncertainty_score(y_true)
        
        return {
            'brier_reliability': reliability,
            'brier_resolution': resolution,
            'brier_uncertainty': uncertainty,
            'brier_decomposed': reliability - resolution + uncertainty
        }
    
    def calculate_betting_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds_data: pd.DataFrame,
        stakes: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate betting performance metrics.
        
        Args:
            y_true: Actual outcomes
            y_prob: Predicted probabilities
            odds_data: DataFrame with 'odds' column
            stakes: Bet stakes for each prediction
            
        Returns:
            Dictionary of betting metrics
        """
        if 'odds' not in odds_data.columns:
            logger.warning("No 'odds' column in odds_data")
            return {}
        
        odds = odds_data['odds'].values
        
        # Calculate returns
        returns = np.where(y_true == 1, stakes * (odds - 1), -stakes)
        
        # Calculate metrics
        total_staked = stakes.sum()
        total_return = returns.sum()
        roi = (total_return / total_staked * 100) if total_staked > 0 else 0
        
        # Calculate edge
        edges = y_prob * odds - 1
        positive_edge_rate = (edges > 0).mean()
        
        # Calculate Sharpe ratio (simplified)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        metrics = {
            'total_staked': total_staked,
            'total_return': total_return,
            'profit': total_return - total_staked,
            'roi_percent': roi,
            'avg_stake': stakes.mean(),
            'n_bets': len(stakes),
            'win_rate': y_true.mean(),
            'positive_edge_rate': positive_edge_rate,
            'avg_edge': edges.mean(),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': (returns[returns > 0].sum() / abs(returns[returns < 0].sum()) 
                            if (returns < 0).any() else np.inf)
        }
        
        return self._round_metrics(metrics)
    
    def calculate_market_metrics(
        self,
        bet_odds: np.ndarray,
        close_odds: np.ndarray,
        stakes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate market-based metrics including CLV.
        
        Args:
            bet_odds: Odds at time of bet
            close_odds: Closing odds
            stakes: Optional bet stakes
            
        Returns:
            Dictionary of market metrics
        """
        # CLV calculations
        clv_results = self.clv_stats(bet_odds, close_odds, stakes)
        
        # Additional market metrics
        odds_movement = close_odds - bet_odds
        
        metrics = {
            **clv_results,
            'avg_odds_movement': odds_movement.mean(),
            'positive_movement_rate': (odds_movement > 0).mean(),
            'avg_bet_odds': bet_odds.mean(),
            'avg_close_odds': close_odds.mean()
        }
        
        return self._round_metrics(metrics)
    
    def clv_stats(
        self,
        reco_odds: np.ndarray,
        close_odds: np.ndarray,
        stakes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate Closing Line Value statistics.
        
        Args:
            reco_odds: Recommended/bet odds
            close_odds: Closing odds
            stakes: Optional stakes for weighted calculations
            
        Returns:
            Dictionary with CLV statistics
        """
        # Convert to implied probabilities
        reco_prob = 1 / reco_odds
        close_prob = 1 / close_odds
        
        # CLV as percentage
        clv_pct = (close_prob - reco_prob) / reco_prob * 100
        
        # CLV win rate (beat closing line)
        clv_win_rate = (clv_pct > 0).mean()
        
        if stakes is not None:
            # Weighted average CLV
            avg_clv = np.average(clv_pct, weights=stakes)
            
            # Dollar CLV
            clv_dollars = stakes * (close_odds - reco_odds) / reco_odds
            total_clv_dollars = clv_dollars.sum()
        else:
            avg_clv = clv_pct.mean()
            total_clv_dollars = 0
        
        return {
            'clv_win_rate': clv_win_rate,
            'avg_clv_percent': avg_clv,
            'median_clv_percent': np.median(clv_pct),
            'total_clv_dollars': total_clv_dollars,
            'clv_std': clv_pct.std()
        }
    
    def _safe_metric(self, metric_func, *args, **kwargs):
        """Safely calculate a metric, returning NaN on error."""
        try:
            return metric_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error calculating {metric_func.__name__}: {e}")
            return np.nan
    
    def _round_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Round numeric metrics to specified precision."""
        rounded = {}
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                rounded[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                if np.isfinite(value):
                    rounded[key] = round(value, self.decimal_precision)
                else:
                    rounded[key] = value
            else:
                rounded[key] = value
        return rounded
    
    def create_metrics_report(
        self,
        results: MetricsResult,
        output_format: str = 'dict'
    ) -> Union[Dict, pd.DataFrame, str]:
        """
        Create a formatted metrics report.
        
        Args:
            results: MetricsResult object
            output_format: 'dict', 'dataframe', or 'text'
            
        Returns:
            Formatted report in requested format
        """
        all_metrics = {
            **results.classification_metrics,
            **results.calibration_metrics,
            **results.betting_metrics,
            **results.market_metrics
        }
        
        if output_format == 'dict':
            return all_metrics
        
        elif output_format == 'dataframe':
            df = pd.DataFrame([all_metrics])
            return df.T.rename(columns={0: 'value'})
        
        elif output_format == 'text':
            lines = ["UFC Prediction Metrics Report", "=" * 40]
            
            sections = [
                ("Classification Metrics", results.classification_metrics),
                ("Calibration Metrics", results.calibration_metrics),
                ("Betting Metrics", results.betting_metrics),
                ("Market Metrics", results.market_metrics)
            ]
            
            for section_name, section_metrics in sections:
                if section_metrics:
                    lines.append(f"\n{section_name}:")
                    lines.append("-" * 30)
                    for key, value in section_metrics.items():
                        lines.append(f"{key:25s}: {value}")
            
            if results.metadata:
                lines.append("\nMetadata:")
                lines.append("-" * 30)
                for key, value in results.metadata.items():
                    lines.append(f"{key:25s}: {value}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown output format: {output_format}")


def calculate_portfolio_metrics(
    portfolio_results: pd.DataFrame,
    initial_bankroll: float = 1000
) -> Dict[str, float]:
    """
    Calculate portfolio-level metrics from multiple bets.
    
    Args:
        portfolio_results: DataFrame with columns 'date', 'stake', 'return'
        initial_bankroll: Starting bankroll
        
    Returns:
        Dictionary of portfolio metrics
    """
    # Sort by date
    portfolio_results = portfolio_results.sort_values('date')
    
    # Calculate cumulative returns
    portfolio_results['cumulative_return'] = portfolio_results['return'].cumsum()
    portfolio_results['bankroll'] = initial_bankroll + portfolio_results['cumulative_return']
    
    # Calculate metrics
    total_return = portfolio_results['return'].sum()
    total_staked = portfolio_results['stake'].sum()
    
    # ROI
    roi = (total_return / total_staked * 100) if total_staked > 0 else 0
    
    # Sharpe ratio (annualized, assuming daily returns)
    daily_returns = portfolio_results.groupby('date')['return'].sum()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # Maximum drawdown
    running_max = portfolio_results['bankroll'].expanding().max()
    drawdown_pct = (portfolio_results['bankroll'] - running_max) / running_max * 100
    max_drawdown_pct = drawdown_pct.min()
    
    # Win rate
    win_rate = (portfolio_results['return'] > 0).mean()
    
    # Calmar ratio (return / max drawdown)
    annual_return = roi * (365 / len(portfolio_results['date'].unique()))
    calmar = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
    
    return {
        'total_return': total_return,
        'total_staked': total_staked,
        'final_bankroll': portfolio_results['bankroll'].iloc[-1],
        'roi_percent': roi,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate': win_rate,
        'n_bets': len(portfolio_results),
        'avg_stake': portfolio_results['stake'].mean(),
        'calmar_ratio': calmar,
        'profit_factor': (portfolio_results[portfolio_results['return'] > 0]['return'].sum() / 
                         abs(portfolio_results[portfolio_results['return'] < 0]['return'].sum())
                         if (portfolio_results['return'] < 0).any() else np.inf)
    }