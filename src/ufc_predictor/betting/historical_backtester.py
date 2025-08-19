"""
Historical Backtesting Engine for UFC Predictor
===============================================

Backtests the UFC prediction model using historical odds data.
Calculates ROI, win rate, and other performance metrics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from difflib import SequenceMatcher
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Result of a single bet in backtesting"""
    date: str
    event: str
    fighter_a: str
    fighter_b: str
    predicted_winner: str
    actual_winner: str
    bet_amount: float
    odds: float
    profit: float
    win: bool
    model_probability: float
    implied_probability: float
    expected_value: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FighterNameMatcher:
    """Matches fighter names between different data sources"""
    
    def __init__(self, threshold: float = 0.85):
        """
        Initialize the name matcher
        
        Args:
            threshold: Minimum similarity score for matching (0.0 to 1.0)
        """
        self.threshold = threshold
        self.name_cache = {}
    
    def normalize_name(self, name: str) -> str:
        """Normalize fighter name for matching"""
        # Remove common prefixes/suffixes
        name = name.lower().strip()
        name = name.replace("'", "").replace("-", " ")
        name = name.replace(".", "").replace(",", "")
        
        # Remove weight class indicators
        for term in ['jr', 'junior', 'sr', 'senior', 'ii', 'iii']:
            name = name.replace(f" {term}", "")
        
        return " ".join(name.split())  # Normalize spaces
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        # Check cache
        cache_key = f"{norm1}|{norm2}"
        if cache_key in self.name_cache:
            return self.name_cache[cache_key]
        
        # Calculate similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Bonus for exact last name match
        parts1 = norm1.split()
        parts2 = norm2.split()
        if parts1 and parts2 and parts1[-1] == parts2[-1]:
            similarity = min(1.0, similarity + 0.1)
        
        # Cache result
        self.name_cache[cache_key] = similarity
        return similarity
    
    def find_best_match(self, target_name: str, candidate_names: List[str]) -> Optional[Tuple[str, float]]:
        """
        Find the best matching name from candidates
        
        Args:
            target_name: Name to match
            candidate_names: List of candidate names
            
        Returns:
            Tuple of (best_match, similarity_score) or None
        """
        best_match = None
        best_score = 0
        
        for candidate in candidate_names:
            score = self.calculate_similarity(target_name, candidate)
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = candidate
        
        if best_match:
            return (best_match, best_score)
        return None


class HistoricalBacktester:
    """Main backtesting engine for UFC predictions"""
    
    def __init__(self, model_predictions: pd.DataFrame, 
                 historical_odds: pd.DataFrame,
                 fight_results: pd.DataFrame,
                 initial_bankroll: float = 1000):
        """
        Initialize the backtester
        
        Args:
            model_predictions: DataFrame with model predictions
            historical_odds: DataFrame with historical odds data
            fight_results: DataFrame with actual fight results
            initial_bankroll: Starting bankroll for simulation
        """
        self.predictions = model_predictions
        self.odds = historical_odds
        self.results = fight_results
        self.initial_bankroll = initial_bankroll
        self.name_matcher = FighterNameMatcher()
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and align data for backtesting"""
        # Ensure date columns are datetime
        if 'commence_time' in self.odds.columns:
            self.odds['date'] = pd.to_datetime(self.odds['commence_time']).dt.date
        
        if 'Event' in self.results.columns:
            # Extract date from event string
            self.results['date'] = self.results['Event'].apply(self._extract_date_from_event)
        
        # Create fight keys for matching
        self.odds['fight_key'] = self.odds.apply(
            lambda x: self._create_fight_key(x['fighter_a'], x['fighter_b']), axis=1
        )
        
        self.results['fight_key'] = self.results.apply(
            lambda x: self._create_fight_key(x['Fighter'], x['Opponent']), axis=1
        )
    
    def _extract_date_from_event(self, event_str: str) -> Optional[datetime]:
        """Extract date from UFC event string"""
        import re
        # Pattern: "Month. DD, YYYY"
        match = re.search(r'([A-Z][a-z]+)\. (\d{1,2}), (\d{4})', event_str)
        if match:
            month_str, day, year = match.groups()
            # Convert month abbreviation to number
            months = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = months.get(month_str[:3], 1)
            try:
                return datetime(int(year), month, int(day)).date()
            except:
                return None
        return None
    
    def _create_fight_key(self, fighter1: str, fighter2: str) -> str:
        """Create a normalized fight key for matching"""
        names = sorted([fighter1.lower().strip(), fighter2.lower().strip()])
        return f"{names[0]}_{names[1]}"
    
    def match_fights_with_odds(self) -> pd.DataFrame:
        """
        Match fights from results with historical odds
        
        Returns:
            DataFrame with matched fights and odds
        """
        matched_fights = []
        
        # Get unique fight dates from results
        fight_dates = self.results[self.results['date'].notna()]['date'].unique()
        
        for fight_date in fight_dates:
            # Get fights on this date
            date_fights = self.results[self.results['date'] == fight_date]
            
            # Get odds around this date (within 7 days before)
            odds_window = self.odds[
                (self.odds['date'] >= fight_date - timedelta(days=7)) &
                (self.odds['date'] <= fight_date)
            ]
            
            if odds_window.empty:
                continue
            
            # Match each fight
            for _, fight in date_fights.iterrows():
                # Find matching odds
                fighter_name = fight['Fighter']
                opponent_name = fight['Opponent']
                
                # Try exact match first
                exact_match = odds_window[
                    ((odds_window['fighter_a'] == fighter_name) & 
                     (odds_window['fighter_b'] == opponent_name)) |
                    ((odds_window['fighter_a'] == opponent_name) & 
                     (odds_window['fighter_b'] == fighter_name))
                ]
                
                if not exact_match.empty:
                    match_odds = exact_match.iloc[-1]  # Get most recent odds
                else:
                    # Try fuzzy matching
                    all_fighters = list(set(odds_window['fighter_a'].tolist() + 
                                          odds_window['fighter_b'].tolist()))
                    
                    fighter_match = self.name_matcher.find_best_match(fighter_name, all_fighters)
                    opponent_match = self.name_matcher.find_best_match(opponent_name, all_fighters)
                    
                    if fighter_match and opponent_match:
                        fuzzy_match = odds_window[
                            ((odds_window['fighter_a'] == fighter_match[0]) & 
                             (odds_window['fighter_b'] == opponent_match[0])) |
                            ((odds_window['fighter_a'] == opponent_match[0]) & 
                             (odds_window['fighter_b'] == fighter_match[0]))
                        ]
                        
                        if not fuzzy_match.empty:
                            match_odds = fuzzy_match.iloc[-1]
                        else:
                            continue
                    else:
                        continue
                
                # Create matched record
                matched_fight = {
                    'date': fight_date,
                    'event': fight['Event'],
                    'fighter': fight['Fighter'],
                    'opponent': fight['Opponent'],
                    'outcome': fight['Outcome'],
                    'fighter_odds': match_odds['fighter_a_odds'] if match_odds['fighter_a'] == fighter_name else match_odds['fighter_b_odds'],
                    'opponent_odds': match_odds['fighter_b_odds'] if match_odds['fighter_b'] == opponent_name else match_odds['fighter_a_odds'],
                    'bookmaker': match_odds.get('bookmaker', 'Unknown')
                }
                matched_fights.append(matched_fight)
        
        matched_df = pd.DataFrame(matched_fights)
        logger.info(f"Matched {len(matched_df)} fights with historical odds")
        
        return matched_df
    
    def run_backtest(self, matched_fights: pd.DataFrame, 
                     betting_strategy: str = 'kelly',
                     min_edge: float = 0.05) -> Dict[str, Any]:
        """
        Run backtesting simulation
        
        Args:
            matched_fights: DataFrame with matched fights and odds
            betting_strategy: 'kelly', 'flat', or 'proportional'
            min_edge: Minimum expected value to place bet
            
        Returns:
            Dictionary with backtest results and metrics
        """
        results = []
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        
        for _, fight in matched_fights.iterrows():
            # Get model prediction for this fight
            prediction = self._get_model_prediction(fight['fighter'], fight['opponent'])
            
            if prediction is None:
                continue
            
            # Calculate expected value
            win_prob = prediction['win_probability']
            odds = fight['fighter_odds'] if prediction['predicted_winner'] == fight['fighter'] else fight['opponent_odds']
            
            implied_prob = 1 / odds
            expected_value = (win_prob * (odds - 1)) - (1 - win_prob)
            
            # Check if we should bet
            if expected_value < min_edge:
                continue
            
            # Calculate bet size
            if betting_strategy == 'kelly':
                # Kelly Criterion: f = (p*b - q) / b
                # where p = win prob, q = loss prob, b = decimal odds - 1
                kelly_fraction = (win_prob * (odds - 1) - (1 - win_prob)) / (odds - 1)
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                bet_amount = bankroll * kelly_fraction * 0.25  # Conservative Kelly (1/4)
            elif betting_strategy == 'flat':
                bet_amount = min(bankroll * 0.05, 50)  # 5% or $50, whichever is smaller
            else:  # proportional
                bet_amount = bankroll * 0.02 * (1 + expected_value)  # 2% base, scaled by edge
            
            # Cap bet at 5% of bankroll
            bet_amount = min(bet_amount, bankroll * 0.05)
            
            if bet_amount < 1:
                continue
            
            # Determine if bet won
            actual_winner = fight['fighter'] if fight['outcome'] == 'win' else fight['opponent']
            bet_won = (prediction['predicted_winner'] == actual_winner)
            
            # Calculate profit
            if bet_won:
                profit = bet_amount * (odds - 1)
            else:
                profit = -bet_amount
            
            # Update bankroll
            bankroll += profit
            bankroll_history.append(bankroll)
            
            # Record result
            result = BacktestResult(
                date=str(fight['date']),
                event=fight['event'],
                fighter_a=fight['fighter'],
                fighter_b=fight['opponent'],
                predicted_winner=prediction['predicted_winner'],
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
        metrics = self._calculate_metrics(results, bankroll_history)
        
        return {
            'results': results,
            'metrics': metrics,
            'bankroll_history': bankroll_history
        }
    
    def _get_model_prediction(self, fighter: str, opponent: str) -> Optional[Dict]:
        """
        Get model prediction for a fight
        
        For now returns random prediction - integrate with actual model
        """
        # TODO: Integrate with actual model predictions
        # This is a placeholder that returns random predictions
        import random
        
        if random.random() > 0.5:
            return {
                'predicted_winner': fighter,
                'win_probability': 0.5 + random.random() * 0.4  # 50-90%
            }
        else:
            return {
                'predicted_winner': opponent,
                'win_probability': 0.5 + random.random() * 0.4
            }
    
    def _calculate_metrics(self, results: List[BacktestResult], 
                          bankroll_history: List[float]) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results"""
        if not results:
            return {
                'total_bets': 0,
                'roi': 0,
                'win_rate': 0,
                'profit': 0
            }
        
        df = pd.DataFrame([r.to_dict() for r in results])
        
        total_bets = len(df)
        total_staked = df['bet_amount'].sum()
        total_profit = df['profit'].sum()
        
        metrics = {
            'total_bets': total_bets,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
            'win_rate': (df['win'].sum() / total_bets * 100) if total_bets > 0 else 0,
            'avg_odds': df['odds'].mean(),
            'avg_bet_size': df['bet_amount'].mean(),
            'avg_expected_value': df['expected_value'].mean(),
            'final_bankroll': bankroll_history[-1] if bankroll_history else self.initial_bankroll,
            'bankroll_growth': ((bankroll_history[-1] - self.initial_bankroll) / self.initial_bankroll * 100) if bankroll_history else 0,
            'max_drawdown': self._calculate_max_drawdown(bankroll_history),
            'sharpe_ratio': self._calculate_sharpe_ratio(df),
            'profit_by_month': self._calculate_monthly_profits(df)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if len(bankroll_history) < 2:
            return 0
        
        peak = bankroll_history[0]
        max_dd = 0
        
        for value in bankroll_history[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe-like ratio for betting returns"""
        if df.empty or len(df) < 2:
            return 0
        
        # Group by date and calculate daily returns
        daily_returns = df.groupby('date')['profit'].sum()
        
        if len(daily_returns) < 2:
            return 0
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio
        sharpe = (mean_return / std_return) * np.sqrt(252)  # 252 trading days
        
        return sharpe
    
    def _calculate_monthly_profits(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate profits by month"""
        if df.empty:
            return {}
        
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['profit'].sum().to_dict()
        
        # Convert Period to string
        return {str(k): v for k, v in monthly.items()}
    
    def generate_report(self, backtest_results: Dict[str, Any], 
                       output_path: Path = None) -> str:
        """
        Generate comprehensive backtest report
        
        Args:
            backtest_results: Results from run_backtest
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        metrics = backtest_results['metrics']
        
        report = []
        report.append("=" * 60)
        report.append("UFC PREDICTOR HISTORICAL BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary metrics
        report.append("📊 SUMMARY METRICS")
        report.append("-" * 40)
        report.append(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        report.append(f"Final Bankroll: ${metrics['final_bankroll']:,.2f}")
        report.append(f"Total Profit: ${metrics['total_profit']:,.2f}")
        report.append(f"ROI: {metrics['roi']:.2f}%")
        report.append(f"Bankroll Growth: {metrics['bankroll_growth']:.2f}%")
        report.append("")
        
        # Betting statistics
        report.append("🎯 BETTING STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Bets: {metrics['total_bets']}")
        report.append(f"Win Rate: {metrics['win_rate']:.2f}%")
        report.append(f"Average Odds: {metrics['avg_odds']:.2f}")
        report.append(f"Average Bet Size: ${metrics['avg_bet_size']:.2f}")
        report.append(f"Average Expected Value: {metrics['avg_expected_value']:.3f}")
        report.append("")
        
        # Risk metrics
        report.append("⚠️ RISK METRICS")
        report.append("-" * 40)
        report.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append("")
        
        # Monthly breakdown
        if metrics['profit_by_month']:
            report.append("📅 MONTHLY PERFORMANCE")
            report.append("-" * 40)
            for month, profit in sorted(metrics['profit_by_month'].items()):
                sign = "+" if profit >= 0 else ""
                report.append(f"{month}: {sign}${profit:.2f}")
            report.append("")
        
        # Best and worst bets
        if backtest_results['results']:
            results_df = pd.DataFrame([r.to_dict() for r in backtest_results['results']])
            
            report.append("🏆 TOP 5 BEST BETS")
            report.append("-" * 40)
            best_bets = results_df.nlargest(5, 'profit')
            for _, bet in best_bets.iterrows():
                report.append(f"{bet['date']}: {bet['predicted_winner']} "
                            f"(${bet['profit']:.2f} @ {bet['odds']:.2f})")
            report.append("")
            
            report.append("😞 TOP 5 WORST BETS")
            report.append("-" * 40)
            worst_bets = results_df.nsmallest(5, 'profit')
            for _, bet in worst_bets.iterrows():
                report.append(f"{bet['date']}: {bet['predicted_winner']} "
                            f"(${bet['profit']:.2f} @ {bet['odds']:.2f})")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            # Also save detailed results as CSV
            if backtest_results['results']:
                csv_path = output_path.parent / "backtest_results.csv"
                results_df = pd.DataFrame([r.to_dict() for r in backtest_results['results']])
                results_df.to_csv(csv_path, index=False)
                
                # Save metrics as JSON
                json_path = output_path.parent / "backtest_metrics.json"
                with open(json_path, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
        
        return report_text


def main():
    """Example usage"""
    logger.info("Historical Backtester initialized")
    logger.info("Use with fetch_all_historical_odds.py to get data first")


if __name__ == "__main__":
    main()