"""
Profitable UFC Predictor

This module combines the existing UFC prediction system with profitability optimization
to identify the most profitable betting opportunities.

Integration points:
- Uses existing UFCPredictor for fight predictions
- Adds profitability analysis via ProfitabilityOptimizer
- Provides comprehensive betting recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ufc_predictor.core.prediction import UFCPredictor
from ufc_predictor.betting.profitability import ProfitabilityOptimizer, BettingOdds, BettingOpportunity


class ProfitableUFCPredictor:
    """
    Enhanced UFC predictor that combines prediction accuracy with profitability analysis.
    
    This class builds on your existing prediction models to identify the most
    profitable betting opportunities using expected value analysis and Kelly Criterion.
    """
    
    def __init__(self, 
                 winner_model_path: str,
                 winner_columns_path: str,
                 method_model_path: str = None,
                 method_columns_path: str = None,
                 fighters_data_path: str = None,
                 bankroll: float = 1000.0,
                 max_kelly_fraction: float = 0.05):
        """
        Initialize the profitable predictor.
        
        Args:
            winner_model_path: Path to winner prediction model
            winner_columns_path: Path to winner model columns
            method_model_path: Path to method prediction model (optional)
            method_columns_path: Path to method model columns (optional)
            fighters_data_path: Path to fighters database
            bankroll: Starting bankroll for betting analysis
            max_kelly_fraction: Maximum Kelly fraction for risk management
        """
        # Initialize base predictor
        self.predictor = UFCPredictor(
            model_path=winner_model_path,
            columns_path=winner_columns_path,
            fighters_data_path=fighters_data_path
        )
        
        # Initialize profitability optimizer
        self.profit_optimizer = ProfitabilityOptimizer(
            bankroll=bankroll,
            max_kelly_fraction=max_kelly_fraction
        )
        
        # Store method model paths for future use
        self.method_model_path = method_model_path
        self.method_columns_path = method_columns_path
        
        print(f"💰 Profitable UFC Predictor initialized with ${bankroll} bankroll")
    
    def predict_with_profitability(self, 
                                 fighter_a: str, 
                                 fighter_b: str, 
                                 odds_a: float, 
                                 odds_b: float, 
                                 sportsbook: str = "Generic") -> Dict[str, Any]:
        """
        Make a fight prediction with profitability analysis.
        
        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            odds_a: American odds for fighter A
            odds_b: American odds for fighter B
            sportsbook: Name of sportsbook (for tracking)
            
        Returns:
            Combined prediction and profitability analysis
        """
        # Get base prediction
        prediction = self.predictor.predict_fight_symmetrical(fighter_a, fighter_b)
        
        if 'error' in prediction:
            return prediction
        
        # Create odds object
        odds_data = BettingOdds(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            fighter_a_odds=odds_a,
            fighter_b_odds=odds_b,
            sportsbook=sportsbook,
            timestamp=datetime.now()
        )
        
        # Analyze betting opportunities
        opportunities = self.profit_optimizer.analyze_betting_opportunity(prediction, odds_data)
        
        # Combine results
        result = {
            **prediction,  # Include original prediction
            'odds_analysis': {
                'fighter_a_odds': odds_a,
                'fighter_b_odds': odds_b,
                'sportsbook': sportsbook,
                'market_prob_a': self.profit_optimizer.american_odds_to_probability(odds_a),
                'market_prob_b': self.profit_optimizer.american_odds_to_probability(odds_b)
            },
            'betting_opportunities': [],
            'recommendation': 'No profitable opportunities found'
        }
        
        if opportunities:
            # Add betting opportunities
            result['betting_opportunities'] = [
                {
                    'fighter': opp.fighter,
                    'expected_value': f"{opp.expected_value*100:.2f}%",
                    'recommended_bet': f"${opp.recommended_bet:.2f}",
                    'kelly_fraction': f"{opp.kelly_fraction*100:.2f}%",
                    'confidence_score': f"{opp.confidence_score*100:.1f}%",
                    'model_prob': f"{opp.model_prob*100:.1f}%",
                    'market_prob': f"{opp.market_prob*100:.1f}%"
                }
                for opp in opportunities
            ]
            
            # Best opportunity recommendation
            best_opp = max(opportunities, key=lambda x: x.expected_value)
            result['recommendation'] = f"Bet ${best_opp.recommended_bet:.2f} on {best_opp.fighter} (EV: {best_opp.expected_value*100:.2f}%)"
        
        return result
    
    def analyze_fight_card_profitability(self, 
                                       fight_card: List[Tuple[str, str, float, float]], 
                                       sportsbook: str = "Generic") -> Dict[str, Any]:
        """
        Analyze an entire fight card for profitability.
        
        Args:
            fight_card: List of tuples (fighter_a, fighter_b, odds_a, odds_b)
            sportsbook: Name of sportsbook
            
        Returns:
            Comprehensive fight card analysis with betting recommendations
        """
        print(f"🃏 Analyzing fight card profitability...")
        
        card_results = {
            'event_summary': {
                'total_fights': len(fight_card),
                'profitable_opportunities': 0,
                'total_recommended_bet': 0.0,
                'expected_profit': 0.0,
                'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'fight_analyses': [],
            'betting_opportunities': [],
            'risk_summary': {}
        }
        
        all_opportunities = []
        
        for i, (fighter_a, fighter_b, odds_a, odds_b) in enumerate(fight_card, 1):
            print(f"  Analyzing Fight {i}: {fighter_a} vs {fighter_b}")
            
            analysis = self.predict_with_profitability(fighter_a, fighter_b, odds_a, odds_b, sportsbook)
            card_results['fight_analyses'].append(analysis)
            
            if 'betting_opportunities' in analysis and analysis['betting_opportunities']:
                card_results['event_summary']['profitable_opportunities'] += 1
                all_opportunities.extend([
                    {
                        'fight_number': i,
                        'matchup': f"{fighter_a} vs {fighter_b}",
                        **opp
                    }
                    for opp in analysis['betting_opportunities']
                ])
        
        # Sort opportunities by expected value
        all_opportunities.sort(key=lambda x: float(x['expected_value'].replace('%', '')), reverse=True)
        card_results['betting_opportunities'] = all_opportunities
        
        # Calculate summary statistics
        if all_opportunities:
            total_recommended = sum(float(opp['recommended_bet'].replace('$', '')) for opp in all_opportunities)
            card_results['event_summary']['total_recommended_bet'] = total_recommended
            
            # Estimate expected profit (simplified)
            expected_profit = sum(
                float(opp['recommended_bet'].replace('$', '')) * 
                float(opp['expected_value'].replace('%', '')) / 100
                for opp in all_opportunities
            )
            card_results['event_summary']['expected_profit'] = expected_profit
        
        # Risk summary
        card_results['risk_summary'] = {
            'total_exposure': f"${card_results['event_summary']['total_recommended_bet']:.2f}",
            'bankroll_percentage': f"{(card_results['event_summary']['total_recommended_bet'] / self.profit_optimizer.bankroll) * 100:.1f}%",
            'expected_return': f"${card_results['event_summary']['expected_profit']:.2f}",
            'number_of_bets': len(all_opportunities)
        }
        
        return card_results
    
    def display_profitable_prediction(self, result: Dict[str, Any]):
        """Display a profitable prediction in a formatted way."""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n🥊 FIGHT ANALYSIS: {result['fighter_a']} vs {result['fighter_b']}")
        print("=" * 60)
        
        # Prediction results
        print(f"🏆 Predicted Winner: {result['predicted_winner']} ({result['confidence']})")
        print(f"📊 Win Probabilities:")
        for fighter in [result['fighter_a'], result['fighter_b']]:
            if f"{fighter}_win_probability" in result:
                print(f"   {fighter}: {result[f'{fighter}_win_probability']}")
        
        # Market analysis
        odds_analysis = result['odds_analysis']
        print(f"\n💰 MARKET ANALYSIS:")
        print(f"   Odds: {result['fighter_a']} ({odds_analysis['fighter_a_odds']}), {result['fighter_b']} ({odds_analysis['fighter_b_odds']})")
        print(f"   Market Probabilities: {result['fighter_a']} ({odds_analysis['market_prob_a']*100:.1f}%), {result['fighter_b']} ({odds_analysis['market_prob_b']*100:.1f}%)")
        print(f"   Sportsbook: {odds_analysis['sportsbook']}")
        
        # Betting opportunities
        if result['betting_opportunities']:
            print(f"\n💎 PROFITABLE OPPORTUNITIES:")
            for opp in result['betting_opportunities']:
                print(f"   🎯 {opp['fighter']}:")
                print(f"      Expected Value: {opp['expected_value']}")
                print(f"      Recommended Bet: {opp['recommended_bet']}")
                print(f"      Kelly Fraction: {opp['kelly_fraction']}")
                print(f"      Model Edge: {opp['model_prob']} vs Market {opp['market_prob']}")
        else:
            print(f"\n❌ No profitable betting opportunities found")
        
        print(f"\n📋 RECOMMENDATION: {result['recommendation']}")
    
    def display_card_analysis(self, card_results: Dict[str, Any]):
        """Display fight card analysis in a formatted way."""
        summary = card_results['event_summary']
        risk = card_results['risk_summary']
        
        print(f"\n🃏 FIGHT CARD PROFITABILITY ANALYSIS")
        print("=" * 60)
        
        print(f"📊 EVENT SUMMARY:")
        print(f"   Total Fights: {summary['total_fights']}")
        print(f"   Profitable Opportunities: {summary['profitable_opportunities']}")
        print(f"   Success Rate: {(summary['profitable_opportunities']/summary['total_fights'])*100:.1f}%")
        
        print(f"\n💰 BETTING SUMMARY:")
        print(f"   Total Recommended Bets: {risk['total_exposure']}")
        print(f"   Bankroll Exposure: {risk['bankroll_percentage']}")
        print(f"   Expected Profit: {risk['expected_return']}")
        print(f"   Number of Bets: {risk['number_of_bets']}")
        
        if card_results['betting_opportunities']:
            print(f"\n🎯 TOP OPPORTUNITIES:")
            for i, opp in enumerate(card_results['betting_opportunities'][:5], 1):  # Top 5
                print(f"   {i}. {opp['matchup']} - {opp['fighter']}")
                print(f"      EV: {opp['expected_value']}, Bet: {opp['recommended_bet']}")
        
        print(f"\n📋 Analysis completed at: {summary['analysis_timestamp']}")
    
    def update_bankroll(self, new_bankroll: float):
        """Update the current bankroll for analysis."""
        old_bankroll = self.profit_optimizer.bankroll
        self.profit_optimizer.bankroll = new_bankroll
        print(f"💰 Bankroll updated: ${old_bankroll:.2f} → ${new_bankroll:.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics from the profit optimizer."""
        return self.profit_optimizer.get_performance_metrics()


def create_profitable_predictor_from_latest() -> ProfitableUFCPredictor:
    """
    Create a ProfitableUFCPredictor using the latest trained models.
    
    Returns:
        Configured ProfitableUFCPredictor instance
    """
    # Use the same auto-detection logic from your existing notebook
    from pathlib import Path
    
    model_dir = Path('../model')
    
    # Find latest training directory
    training_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('training_')]
    
    if training_dirs:
        latest_training_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        
        # Look for model files
        winner_models = list(latest_training_dir.glob('ufc_winner_model_tuned_*.joblib'))
        if not winner_models:
            winner_models = list(latest_training_dir.glob('ufc_winner_model_*.joblib'))
        
        method_models = list(latest_training_dir.glob('ufc_method_model_*.joblib'))
        winner_cols_files = list(latest_training_dir.glob('winner_model_columns_*.json'))
        method_cols_files = list(latest_training_dir.glob('method_model_columns_*.json'))
        fighters_data_files = list(latest_training_dir.glob('ufc_fighters_engineered_*.csv'))
        
        if winner_models and winner_cols_files and fighters_data_files:
            winner_model_path = str(max(winner_models, key=lambda x: x.stat().st_mtime))
            winner_cols_path = str(max(winner_cols_files, key=lambda x: x.stat().st_mtime))
            fighters_data_path = str(max(fighters_data_files, key=lambda x: x.stat().st_mtime))
            
            method_model_path = str(max(method_models, key=lambda x: x.stat().st_mtime)) if method_models else None
            method_cols_path = str(max(method_cols_files, key=lambda x: x.stat().st_mtime)) if method_cols_files else None
            
            print(f"🔍 Using latest models from: {latest_training_dir.name}")
            
            return ProfitableUFCPredictor(
                winner_model_path=winner_model_path,
                winner_columns_path=winner_cols_path,
                method_model_path=method_model_path,
                method_columns_path=method_cols_path,
                fighters_data_path=fighters_data_path
            )
    
    # Fallback to standard locations
    print("⚠️  No training directories found, using standard model locations...")
    return ProfitableUFCPredictor(
        winner_model_path='../model/ufc_random_forest_model_tuned.joblib',
        winner_columns_path='../model/winner_model_columns.json',
        fighters_data_path='../model/ufc_fighters_engineered_corrected.csv'
    )


def demo_profitable_prediction():
    """Demonstrate the profitable prediction workflow."""
    print("🎯 Profitable UFC Prediction Demo")
    print("=" * 50)
    
    try:
        # Create predictor
        predictor = create_profitable_predictor_from_latest()
        
        # Example single fight analysis
        print("\n1. SINGLE FIGHT ANALYSIS")
        result = predictor.predict_with_profitability(
            fighter_a="Jon Jones",
            fighter_b="Stipe Miocic", 
            odds_a=-300,  # Jones favored
            odds_b=+250,  # Miocic underdog
            sportsbook="DraftKings"
        )
        
        predictor.display_profitable_prediction(result)
        
        # Example fight card analysis
        print("\n\n2. FIGHT CARD ANALYSIS")
        sample_card = [
            ("Jon Jones", "Stipe Miocic", -300, +250),
            ("Islam Makhachev", "Charles Oliveira", -180, +155),
            ("Sean O'Malley", "Marlon Vera", -120, +100),
        ]
        
        card_results = predictor.analyze_fight_card_profitability(sample_card, "DraftKings")
        predictor.display_card_analysis(card_results)
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        print("Make sure your models are trained and available.")
        return None


if __name__ == "__main__":
    demo_profitable_prediction() 