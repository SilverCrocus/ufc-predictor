"""
Monitoring dashboards and visualization for UFC prediction performance.
Generates static plots and JSON summaries for tracking model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceDashboard:
    """
    Creates performance monitoring dashboards for UFC predictions.
    """
    
    def __init__(
        self,
        output_dir: str = 'artifacts/monitoring',
        figure_size: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize dashboard generator.
        
        Args:
            output_dir: Directory to save outputs
            figure_size: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def create_calibration_plots(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Analysis"
    ) -> Dict[str, Any]:
        """
        Create calibration plots including reliability diagram.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            
        Returns:
            Dict with plot paths and metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # 1. Reliability Diagram
        ax = axes[0, 0]
        fraction_positives, mean_predicted = self._calculate_calibration_curve(
            y_true, y_prob, n_bins
        )
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean_predicted, fraction_positives, 'o-', label='Model')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram of predictions
        ax = axes[0, 1]
        ax.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label='Negative class', color='blue')
        ax.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label='Positive class', color='red')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Predictions')
        ax.legend()
        
        # 3. Calibration Error by Bin
        ax = axes[1, 0]
        calibration_errors = fraction_positives - mean_predicted
        ax.bar(range(len(calibration_errors)), calibration_errors)
        ax.set_xlabel('Bin')
        ax.set_ylabel('Calibration Error')
        ax.set_title('Calibration Error by Bin')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. ECE over confidence threshold
        ax = axes[1, 1]
        thresholds = np.linspace(0.5, 1.0, 20)
        eces = []
        for thresh in thresholds:
            mask = np.abs(y_prob - 0.5) * 2 >= (thresh - 0.5) * 2
            if mask.sum() > 0:
                ece = self._calculate_ece(y_true[mask], y_prob[mask], n_bins)
                eces.append(ece)
            else:
                eces.append(np.nan)
        
        ax.plot(thresholds, eces, 'o-')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('ECE')
        ax.set_title('ECE vs Confidence Level')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'calibration_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        metrics = {
            'ece': self._calculate_ece(y_true, y_prob, n_bins),
            'mce': self._calculate_mce(y_true, y_prob, n_bins),
            'brier_score': np.mean((y_prob - y_true) ** 2),
            'plot_path': str(plot_path)
        }
        
        return metrics
    
    def create_betting_performance_plots(
        self,
        backtest_result,
        title: str = "Betting Performance Analysis"
    ) -> Dict[str, Any]:
        """
        Create betting performance visualization.
        
        Args:
            backtest_result: BacktestResult object
            title: Plot title
            
        Returns:
            Dict with plot paths
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert bet history to DataFrame
        if backtest_result.bet_history:
            df = pd.DataFrame([
                {
                    'date': b.date,
                    'profit': b.profit,
                    'stake': b.stake,
                    'odds': b.bet_odds,
                    'bankroll': b.running_bankroll,
                    'outcome': b.outcome
                }
                for b in backtest_result.bet_history
            ])
        else:
            df = pd.DataFrame()
        
        if not df.empty:
            # 1. Equity Curve
            ax = axes[0, 0]
            ax.plot(df.index, df['bankroll'], linewidth=2)
            ax.set_xlabel('Bet Number')
            ax.set_ylabel('Bankroll')
            ax.set_title('Equity Curve')
            ax.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax = axes[0, 1]
            peak = df['bankroll'].expanding().max()
            drawdown = (df['bankroll'] - peak) / peak * 100
            ax.fill_between(df.index, 0, drawdown, color='red', alpha=0.3)
            ax.plot(df.index, drawdown, color='red', linewidth=1)
            ax.set_xlabel('Bet Number')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Drawdown Analysis')
            ax.grid(True, alpha=0.3)
            
            # 3. Rolling Sharpe Ratio
            ax = axes[0, 2]
            window = min(50, len(df) // 4)
            if window > 2:
                rolling_returns = df['profit'].rolling(window).mean()
                rolling_std = df['profit'].rolling(window).std()
                rolling_sharpe = rolling_returns / rolling_std.replace(0, np.nan)
                ax.plot(df.index[window:], rolling_sharpe[window:], linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Bet Number')
                ax.set_ylabel('Rolling Sharpe Ratio')
                ax.set_title(f'Rolling Sharpe ({window} bets)')
                ax.grid(True, alpha=0.3)
            
            # 4. Bet Size Distribution
            ax = axes[1, 0]
            ax.hist(df['stake'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Stake Size')
            ax.set_ylabel('Frequency')
            ax.set_title('Bet Size Distribution')
            ax.grid(True, alpha=0.3)
            
            # 5. Odds Distribution
            ax = axes[1, 1]
            ax.hist(df['odds'], bins=30, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Odds')
            ax.set_ylabel('Frequency')
            ax.set_title('Odds Distribution')
            ax.grid(True, alpha=0.3)
            
            # 6. Profit by Odds Range
            ax = axes[1, 2]
            odds_bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
            df['odds_bin'] = pd.cut(df['odds'], bins=odds_bins)
            profit_by_odds = df.groupby('odds_bin')['profit'].sum()
            ax.bar(range(len(profit_by_odds)), profit_by_odds.values)
            ax.set_xticks(range(len(profit_by_odds)))
            ax.set_xticklabels([str(b) for b in profit_by_odds.index], rotation=45)
            ax.set_xlabel('Odds Range')
            ax.set_ylabel('Total Profit')
            ax.set_title('Profit by Odds Range')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'betting_performance_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return {'plot_path': str(plot_path)}
    
    def create_model_comparison_plots(
        self,
        model_results: Dict[str, Dict[str, float]],
        title: str = "Model Comparison"
    ) -> Dict[str, Any]:
        """
        Create model comparison visualizations.
        
        Args:
            model_results: Dict of {model_name: metrics_dict}
            title: Plot title
            
        Returns:
            Dict with plot paths
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        models = list(model_results.keys())
        metrics_to_plot = ['accuracy', 'auc', 'brier_score', 'ece']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = [model_results[m].get(metric, 0) for m in models]
            bars = ax.bar(models, values)
            
            # Color bars based on performance (green=good, red=bad)
            if metric in ['brier_score', 'ece']:  # Lower is better
                colors = ['green' if v == min(values) else 'red' if v == max(values) else 'blue' 
                         for v in values]
            else:  # Higher is better
                colors = ['green' if v == max(values) else 'red' if v == min(values) else 'blue' 
                         for v in values]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
                bar.set_alpha(0.7)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'model_comparison_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return {'plot_path': str(plot_path)}
    
    def create_clv_analysis_plots(
        self,
        bet_history: List,
        title: str = "Closing Line Value Analysis"
    ) -> Dict[str, Any]:
        """
        Create CLV analysis visualizations.
        
        Args:
            bet_history: List of BetResult objects
            title: Plot title
            
        Returns:
            Dict with plot paths and metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        if bet_history:
            # Calculate CLV for each bet
            clv_data = []
            for bet in bet_history:
                if hasattr(bet, 'close_odds') and bet.close_odds > 0:
                    clv_pct = ((1/bet.bet_odds) - (1/bet.close_odds)) / (1/bet.bet_odds) * 100
                    clv_data.append({
                        'clv_pct': clv_pct,
                        'profit': bet.profit,
                        'odds': bet.bet_odds,
                        'beat_close': clv_pct > 0
                    })
            
            if clv_data:
                clv_df = pd.DataFrame(clv_data)
                
                # 1. CLV Distribution
                ax = axes[0, 0]
                ax.hist(clv_df['clv_pct'], bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('CLV %')
                ax.set_ylabel('Frequency')
                ax.set_title('CLV Distribution')
                ax.grid(True, alpha=0.3)
                
                # 2. CLV vs Profit
                ax = axes[0, 1]
                ax.scatter(clv_df['clv_pct'], clv_df['profit'], alpha=0.5)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('CLV %')
                ax.set_ylabel('Profit')
                ax.set_title('CLV vs Profit')
                ax.grid(True, alpha=0.3)
                
                # 3. Cumulative CLV
                ax = axes[1, 0]
                clv_df['cumulative_clv'] = clv_df['clv_pct'].cumsum()
                ax.plot(clv_df.index, clv_df['cumulative_clv'], linewidth=2)
                ax.set_xlabel('Bet Number')
                ax.set_ylabel('Cumulative CLV %')
                ax.set_title('Cumulative CLV Over Time')
                ax.grid(True, alpha=0.3)
                
                # 4. CLV Win Rate by Odds Range
                ax = axes[1, 1]
                odds_bins = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
                clv_df['odds_bin'] = pd.cut(clv_df['odds'], bins=odds_bins)
                clv_by_odds = clv_df.groupby('odds_bin')['beat_close'].mean() * 100
                ax.bar(range(len(clv_by_odds)), clv_by_odds.values)
                ax.set_xticks(range(len(clv_by_odds)))
                ax.set_xticklabels([str(b) for b in clv_by_odds.index], rotation=45)
                ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
                ax.set_xlabel('Odds Range')
                ax.set_ylabel('CLV Win Rate %')
                ax.set_title('CLV Win Rate by Odds')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'clv_analysis_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate CLV metrics
        if clv_data:
            clv_df = pd.DataFrame(clv_data)
            metrics = {
                'clv_win_rate': clv_df['beat_close'].mean(),
                'avg_clv_pct': clv_df['clv_pct'].mean(),
                'median_clv_pct': clv_df['clv_pct'].median(),
                'plot_path': str(plot_path)
            }
        else:
            metrics = {'clv_win_rate': 0, 'avg_clv_pct': 0, 'plot_path': str(plot_path)}
        
        return metrics
    
    def generate_summary_report(
        self,
        performance_metrics: Dict[str, float],
        output_format: str = 'json'
    ) -> str:
        """
        Generate summary report in specified format.
        
        Args:
            performance_metrics: Dict of metrics
            output_format: 'json' or 'html'
            
        Returns:
            Path to saved report
        """
        report_data = {
            'timestamp': self.timestamp,
            'metrics': performance_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        if output_format == 'json':
            report_path = self.output_dir / f'summary_{self.timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        elif output_format == 'html':
            report_path = self.output_dir / f'summary_{self.timestamp}.html'
            html_content = self._generate_html_report(report_data)
            with open(report_path, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def _calculate_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate calibration curve data."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        fraction_positives = []
        mean_predicted = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                fraction_positives.append(y_true[in_bin].mean())
                mean_predicted.append(y_prob[in_bin].mean())
        
        return np.array(fraction_positives), np.array(mean_predicted)
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int
    ) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UFC Predictor Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-good {{ color: green; font-weight: bold; }}
                .metric-bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>UFC Predictor Performance Report</h1>
            <p>Generated: {report_data['generated_at']}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in report_data['metrics'].items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            html += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html