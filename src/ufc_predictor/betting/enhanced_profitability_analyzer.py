"""
Enhanced Profitability Analyzer

Integrates the optimal betting strategy framework with the existing TAB Australia
profitability analysis system. Provides comprehensive betting recommendations based
on the systematic strategy framework.

Usage:
    from ufc_predictor.betting.enhanced_profitability_analyzer import EnhancedProfitabilityAnalyzer
    
    analyzer = EnhancedProfitabilityAnalyzer(bankroll=1000)
    results = analyzer.analyze_with_optimal_strategy(predictions, fighter_data)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ufc_predictor.optimal_betting_strategy import (
    OptimalBettingStrategy, BettingOpportunity, BetType, RiskTier,
    StyleMatchup, WeightCuttingIndicators
)
from ufc_predictor.betting.tab_profitability import TABProfitabilityAnalyzer, TABOpportunity
from configs.model_config import PROJECT_ROOT

class EnhancedProfitabilityAnalyzer:
    """
    Enhanced profitability analyzer that combines optimal betting strategy
    with live TAB Australia odds scraping
    """
    
    def __init__(self, bankroll: float = 1000.0, use_live_odds: bool = True,
                 strategy_config_path: Optional[str] = None):
        """
        Initialize enhanced profitability analyzer
        
        Args:
            bankroll: Starting bankroll amount
            use_live_odds: Whether to scrape live odds
            strategy_config_path: Path to strategy configuration file
        """
        self.bankroll = bankroll
        
        # Initialize TAB analyzer for odds scraping
        self.tab_analyzer = TABProfitabilityAnalyzer(
            bankroll=bankroll,
            use_live_odds=use_live_odds
        )
        
        # Initialize optimal betting strategy
        if not strategy_config_path:
            strategy_config_path = PROJECT_ROOT / "config" / "betting_strategy_config.json"
            
        self.betting_strategy = OptimalBettingStrategy(
            initial_bankroll=bankroll,
            config_path=str(strategy_config_path)
        )
        
        # Track analysis results
        self.last_analysis = None
        self.historical_performance = []
        
    def analyze_with_optimal_strategy(self, model_predictions: Dict[str, float],
                                    fighter_data: Optional[Dict[str, Dict]] = None,
                                    event_info: Optional[Dict] = None) -> Dict:
        """
        Comprehensive analysis using optimal betting strategy framework
        
        Args:
            model_predictions: Fighter name -> win probability mapping
            fighter_data: Optional detailed fighter statistics for each fighter
            event_info: Optional event information (date, venue, etc.)
            
        Returns:
            Enhanced analysis results with optimal strategy recommendations
        """
        
        print("🚀 ENHANCED PROFITABILITY ANALYSIS")
        print("=" * 50)
        
        # Get basic TAB analysis for odds matching
        basic_results = self.tab_analyzer.analyze_predictions(model_predictions)
        
        if not basic_results.get('opportunities'):
            return {
                'status': 'no_opportunities',
                'message': 'No profitable opportunities found with current odds',
                'basic_results': basic_results
            }
        
        # Enhanced analysis with optimal strategy
        enhanced_opportunities = []
        existing_bets = []  # Track for correlation analysis
        
        print(f"\n🔍 APPLYING OPTIMAL STRATEGY FRAMEWORK")
        print("-" * 40)
        
        for tab_opp in basic_results['opportunities']:
            # Prepare data for optimal strategy analysis
            prediction_data = {
                'fighter': tab_opp.fighter,
                'opponent': tab_opp.opponent,
                'win_probability': tab_opp.model_prob,
                'confidence_score': self._calculate_model_confidence(tab_opp.model_prob)
            }
            
            odds_data = {
                'current_odds': tab_opp.american_odds,
                'event': tab_opp.event,
                'timing': 'closing'  # Default, could be enhanced with line movement data
            }
            
            # Get fighter statistics (use defaults if not provided)
            fighter_stats = self._get_fighter_stats(tab_opp.fighter, fighter_data)
            opponent_stats = self._get_fighter_stats(tab_opp.opponent, fighter_data)
            
            # Analyze with optimal strategy
            enhanced_opp = self.betting_strategy.analyze_betting_opportunity(
                prediction=prediction_data,
                odds_data=odds_data,
                fighter_stats=fighter_stats,
                opponent_stats=opponent_stats,
                existing_bets=existing_bets
            )
            
            if enhanced_opp:
                enhanced_opportunities.append(enhanced_opp)
                existing_bets.append(enhanced_opp)
                
                print(f"✅ {enhanced_opp.fighter}: {enhanced_opp.risk_tier.value.upper()} tier")
                print(f"   💰 EV: {enhanced_opp.expected_value:.1%} | Bet: ${enhanced_opp.adjusted_bet_size:.2f}")
                print(f"   🎯 Confidence: {enhanced_opp.confidence_score:.1%} | Style: {enhanced_opp.style_matchup.confidence_multiplier:.2f}x")
            else:
                print(f"❌ {tab_opp.fighter}: Filtered out by strategy framework")
        
        # Portfolio analysis
        portfolio_analysis = self._analyze_portfolio(enhanced_opportunities)
        
        # Multi-bet opportunities
        multi_bet_opportunities = self._analyze_multi_bet_opportunities(enhanced_opportunities)
        
        # Market timing recommendations
        timing_recommendations = self._get_timing_recommendations(enhanced_opportunities)
        
        # Compile enhanced results
        enhanced_results = {
            'status': 'success',
            'basic_tab_results': basic_results,
            'enhanced_opportunities': enhanced_opportunities,
            'portfolio_analysis': portfolio_analysis,
            'multi_bet_opportunities': multi_bet_opportunities,
            'timing_recommendations': timing_recommendations,
            'strategy_summary': self._create_strategy_summary(enhanced_opportunities),
            'risk_assessment': self._assess_portfolio_risk(enhanced_opportunities),
            'bankroll_allocation': self._calculate_bankroll_allocation(enhanced_opportunities)
        }
        
        self.last_analysis = enhanced_results
        return enhanced_results
    
    def _calculate_model_confidence(self, prob: float) -> float:
        """Calculate model confidence based on probability distance from 0.5"""
        return abs(prob - 0.5) * 2  # Maps 0.5 -> 0, 1.0/0.0 -> 1.0
    
    def _get_fighter_stats(self, fighter_name: str, fighter_data: Optional[Dict]) -> Dict:
        """Get fighter statistics with sensible defaults"""
        if not fighter_data or fighter_name not in fighter_data:
            # Return default stats if no data provided
            return {
                'td_avg': 1.0,
                'td_def': 0.70,
                'ko_percentage': 0.30,
                'recent_ko_losses': [],
                'third_round_performance': 0.60,
                'stance': 'Orthodox',
                'southpaw_experience': 0.50,
                'missed_weight_last_2': False,
                'struggled_at_weigh_ins': False,
                'same_day_cut_pct': 0.05,
                'looks_depleted': False,
                'professional_nutrition_team': False,
                'easy_cut_history': True
            }
        
        return fighter_data[fighter_name]
    
    def _analyze_portfolio(self, opportunities: List[BettingOpportunity]) -> Dict:
        """Analyze overall portfolio characteristics"""
        if not opportunities:
            return {'total_exposure': 0, 'risk_distribution': {}, 'expected_return': 0}
        
        total_exposure = sum(opp.adjusted_bet_size for opp in opportunities)
        expected_return = sum(opp.adjusted_bet_size * opp.expected_value for opp in opportunities)
        
        # Risk tier distribution
        risk_distribution = {}
        for tier in RiskTier:
            tier_ops = [o for o in opportunities if o.risk_tier == tier]
            risk_distribution[tier.value] = {
                'count': len(tier_ops),
                'total_stake': sum(o.adjusted_bet_size for o in tier_ops),
                'avg_ev': sum(o.expected_value for o in tier_ops) / len(tier_ops) if tier_ops else 0
            }
        
        return {
            'total_exposure': total_exposure,
            'total_exposure_pct': total_exposure / self.bankroll,
            'expected_return': expected_return,
            'expected_roi': expected_return / total_exposure if total_exposure > 0 else 0,
            'risk_distribution': risk_distribution,
            'average_confidence': sum(o.confidence_score for o in opportunities) / len(opportunities),
            'portfolio_kelly': sum(o.kelly_fraction for o in opportunities)
        }
    
    def _analyze_multi_bet_opportunities(self, single_opportunities: List[BettingOpportunity]) -> List[Dict]:
        """Identify profitable multi-bet combinations"""
        multi_bets = []
        
        if len(single_opportunities) < 2:
            return multi_bets
        
        # 2-leg combinations
        for i, opp1 in enumerate(single_opportunities):
            for opp2 in single_opportunities[i+1:]:
                # Skip if too correlated (same event with high penalty)
                if opp1.event == opp2.event:
                    correlation_penalty = 0.08  # Same event penalty
                    if correlation_penalty > 0.10:  # Threshold for multi-bet exclusion
                        continue
                
                # Calculate combined probability and payout
                combined_prob = opp1.model_prob * opp2.model_prob
                
                # Convert American odds to decimal for payout calculation
                decimal1 = (opp1.market_odds / 100) + 1 if opp1.market_odds > 0 else (100 / abs(opp1.market_odds)) + 1
                decimal2 = (opp2.market_odds / 100) + 1 if opp2.market_odds > 0 else (100 / abs(opp2.market_odds)) + 1
                
                combined_decimal = decimal1 * decimal2
                combined_ev = (combined_prob * (combined_decimal - 1)) - (1 - combined_prob)
                
                if combined_ev > 0.05:  # Minimum EV for multi-bets
                    stake = min(opp1.adjusted_bet_size, opp2.adjusted_bet_size) * 0.5  # Conservative sizing
                    
                    multi_bets.append({
                        'type': '2-leg parlay',
                        'fighters': [opp1.fighter, opp2.fighter],
                        'combined_prob': combined_prob,
                        'combined_decimal_odds': combined_decimal,
                        'expected_value': combined_ev,
                        'recommended_stake': stake,
                        'expected_profit': stake * combined_ev,
                        'correlation_penalty': correlation_penalty if opp1.event == opp2.event else 0
                    })
        
        # Sort by expected profit
        multi_bets.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        return multi_bets[:5]  # Top 5 multi-bet opportunities
    
    def _get_timing_recommendations(self, opportunities: List[BettingOpportunity]) -> Dict:
        """Generate market timing recommendations"""
        timing_recs = {
            'opening_line_bets': [],
            'closing_line_bets': [],
            'live_betting_candidates': [],
            'public_fade_opportunities': []
        }
        
        for opp in opportunities:
            # Recommend opening line for high-value underdogs
            if opp.market_odds > 150 and opp.expected_value > 0.15:
                timing_recs['opening_line_bets'].append({
                    'fighter': opp.fighter,
                    'reason': 'High-value underdog - bet early before sharp money',
                    'stake_split': '70% opening, 30% closing'
                })
            
            # Recommend closing line for favorites with model edge
            elif opp.market_odds < 0 and opp.model_prob > 0.70:
                timing_recs['closing_line_bets'].append({
                    'fighter': opp.fighter,
                    'reason': 'Strong favorite - wait for best closing line',
                    'stake_split': '30% opening, 70% closing'
                })
            
            # Live betting candidates (high-confidence style matchups)
            if opp.style_matchup.confidence_multiplier > 1.1:
                timing_recs['live_betting_candidates'].append({
                    'fighter': opp.fighter,
                    'reason': f'Style advantage may create live opportunities',
                    'watch_for': 'Early momentum or grappling success'
                })
        
        return timing_recs
    
    def _create_strategy_summary(self, opportunities: List[BettingOpportunity]) -> Dict:
        """Create high-level strategy summary"""
        if not opportunities:
            return {'message': 'No betting opportunities meet strategy criteria'}
        
        total_stake = sum(opp.adjusted_bet_size for opp in opportunities)
        total_expected_profit = sum(opp.adjusted_bet_size * opp.expected_value for opp in opportunities)
        
        return {
            'total_opportunities': len(opportunities),
            'total_recommended_stake': total_stake,
            'total_expected_profit': total_expected_profit,
            'portfolio_expected_roi': (total_expected_profit / total_stake) * 100 if total_stake > 0 else 0,
            'avg_confidence_score': sum(o.confidence_score for o in opportunities) / len(opportunities) * 100,
            'risk_tier_breakdown': {
                tier.value: len([o for o in opportunities if o.risk_tier == tier])
                for tier in RiskTier
            },
            'bankroll_utilization': (total_stake / self.bankroll) * 100
        }
    
    def _assess_portfolio_risk(self, opportunities: List[BettingOpportunity]) -> Dict:
        """Assess overall portfolio risk"""
        if not opportunities:
            return {'risk_level': 'none'}
        
        total_stake = sum(opp.adjusted_bet_size for opp in opportunities)
        bankroll_utilization = total_stake / self.bankroll
        
        # Calculate correlation risk
        same_event_pairs = 0
        total_pairs = len(opportunities) * (len(opportunities) - 1) // 2
        
        for i, opp1 in enumerate(opportunities):
            for opp2 in opportunities[i+1:]:
                if opp1.event == opp2.event:
                    same_event_pairs += 1
        
        correlation_risk = same_event_pairs / total_pairs if total_pairs > 0 else 0
        
        # Overall risk assessment
        if bankroll_utilization > 0.20 or correlation_risk > 0.50:
            risk_level = 'high'
        elif bankroll_utilization > 0.15 or correlation_risk > 0.30:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'bankroll_utilization': bankroll_utilization,
            'correlation_risk': correlation_risk,
            'max_single_bet_pct': max(o.adjusted_bet_size / self.bankroll for o in opportunities),
            'diversification_score': len(set(o.event for o in opportunities)) / len(opportunities)
        }
    
    def _calculate_bankroll_allocation(self, opportunities: List[BettingOpportunity]) -> Dict:
        """Calculate detailed bankroll allocation breakdown"""
        if not opportunities:
            return {'allocated': 0, 'reserved': self.bankroll}
        
        total_allocated = sum(opp.adjusted_bet_size for opp in opportunities)
        
        allocation_by_type = {}
        for bet_type in BetType:
            type_opportunities = [o for o in opportunities if o.bet_type == bet_type]
            allocation_by_type[bet_type.value] = {
                'count': len(type_opportunities),
                'total_stake': sum(o.adjusted_bet_size for o in type_opportunities),
                'percentage': sum(o.adjusted_bet_size for o in type_opportunities) / self.bankroll * 100
            }
        
        return {
            'total_allocated': total_allocated,
            'total_reserved': self.bankroll - total_allocated,
            'allocation_percentage': (total_allocated / self.bankroll) * 100,
            'allocation_by_type': allocation_by_type,
            'remaining_capacity': self.bankroll * 0.25 - total_allocated,  # Max 25% exposure
            'emergency_reserve': self.bankroll * 0.20  # Always keep 20% reserve
        }
    
    def get_betting_instructions(self, analysis_results: Optional[Dict] = None) -> List[str]:
        """Generate detailed betting instructions with strategy context"""
        if not analysis_results:
            analysis_results = self.last_analysis
            
        if not analysis_results or analysis_results.get('status') != 'success':
            return ["❌ No analysis results available or no opportunities found"]
        
        opportunities = analysis_results['enhanced_opportunities']
        timing_recs = analysis_results['timing_recommendations'] 
        
        instructions = [
            "🚀 OPTIMAL BETTING STRATEGY INSTRUCTIONS",
            "=" * 50,
            "",
            "📊 PORTFOLIO OVERVIEW:",
            f"   • Total opportunities: {len(opportunities)}",
            f"   • Total stake: ${sum(o.adjusted_bet_size for o in opportunities):.2f}",
            f"   • Expected profit: ${sum(o.adjusted_bet_size * o.expected_value for o in opportunities):.2f}",
            f"   • Portfolio ROI: {analysis_results['strategy_summary']['portfolio_expected_roi']:.1f}%",
            "",
            "💰 INDIVIDUAL BET RECOMMENDATIONS:"
        ]
        
        for i, opp in enumerate(opportunities, 1):
            instructions.extend([
                f"",
                f"   {i}. {opp.fighter} vs {opp.opponent}",
                f"      💵 Stake: ${opp.adjusted_bet_size:.2f} ({opp.risk_tier.value.upper()} tier)",
                f"      🎯 TAB Odds: {opp.market_odds:+d} | Expected Value: {opp.expected_value:.1%}",
                f"      📈 Model Prob: {opp.model_prob:.1%} | Confidence: {opp.confidence_score:.1%}",
                f"      ⚖️  Style Factor: {opp.style_matchup.confidence_multiplier:.2f}x | Weight Cut: {opp.weight_cutting.risk_multiplier:.2f}x"
            ])
        
        # Add timing recommendations
        if timing_recs['opening_line_bets'] or timing_recs['closing_line_bets']:
            instructions.extend([
                "",
                "⏰ MARKET TIMING STRATEGY:",
            ])
            
            for rec in timing_recs['opening_line_bets']:
                instructions.append(f"   📊 {rec['fighter']}: {rec['stake_split']} - {rec['reason']}")
                
            for rec in timing_recs['closing_line_bets']:
                instructions.append(f"   📊 {rec['fighter']}: {rec['stake_split']} - {rec['reason']}")
        
        # Add multi-bet opportunities
        multi_bets = analysis_results.get('multi_bet_opportunities', [])
        if multi_bets:
            instructions.extend([
                "",
                "🎰 MULTI-BET OPPORTUNITIES:"
            ])
            for mb in multi_bets[:3]:  # Top 3
                fighters = " + ".join(mb['fighters'])
                instructions.append(f"   • {fighters}: ${mb['recommended_stake']:.2f} for ${mb['expected_profit']:.2f} profit ({mb['expected_value']:.1%} EV)")
        
        # Add risk warnings
        risk_assessment = analysis_results['risk_assessment']
        if risk_assessment['risk_level'] != 'low':
            instructions.extend([
                "",
                f"⚠️  RISK WARNING: {risk_assessment['risk_level'].upper()} RISK PORTFOLIO",
                f"   • Bankroll utilization: {risk_assessment['bankroll_utilization']:.1%}",
                f"   • Correlation risk: {risk_assessment['correlation_risk']:.1%}",
                "   • Consider reducing position sizes if uncomfortable with risk level"
            ])
        
        instructions.extend([
            "",
            "📋 EXECUTION CHECKLIST:",
            "   1. 🔍 Verify all odds haven't moved significantly",
            "   2. 💰 Log into TAB Australia",
            "   3. 🥊 Navigate to UFC betting section",
            "   4. 📊 Place bets according to timing recommendations",
            "   5. 📝 Record all bet placements for tracking",
            "   6. 📈 Monitor line movements for additional opportunities",
            "",
            "🔔 IMPORTANT REMINDERS:",
            "   • Never bet more than the recommended amounts",
            "   • Verify fighter names match exactly in TAB system",
            "   • Set alerts for significant line movements",
            "   • Track results for model performance validation",
            "",
            f"💡 Strategy powered by optimal betting framework with {analysis_results['basic_tab_results']['odds_source'].upper()} TAB odds"
        ])
        
        return instructions
    
    def export_analysis_report(self, filepath: str, analysis_results: Optional[Dict] = None):
        """Export comprehensive analysis report"""
        if not analysis_results:
            analysis_results = self.last_analysis
            
        if not analysis_results:
            print("No analysis results to export")
            return
            
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'bankroll': self.bankroll,
            'analysis_results': analysis_results,
            'betting_instructions': self.get_betting_instructions(analysis_results)
        }
        
        # Convert BettingOpportunity objects to dictionaries for JSON serialization
        if 'enhanced_opportunities' in report_data['analysis_results']:
            opportunities = []
            for opp in report_data['analysis_results']['enhanced_opportunities']:
                opp_dict = {
                    'fighter': opp.fighter,
                    'opponent': opp.opponent,
                    'event': opp.event,
                    'bet_type': opp.bet_type.value,
                    'model_prob': opp.model_prob,
                    'market_odds': opp.market_odds,
                    'expected_value': opp.expected_value,
                    'kelly_fraction': opp.kelly_fraction,
                    'risk_tier': opp.risk_tier.value,
                    'recommended_bet_size': opp.recommended_bet_size,
                    'adjusted_bet_size': opp.adjusted_bet_size,
                    'confidence_score': opp.confidence_score,
                    'market_timing': opp.market_timing,
                    'correlation_penalty': opp.correlation_penalty,
                    'style_matchup_multiplier': opp.style_matchup.confidence_multiplier,
                    'weight_cutting_multiplier': opp.weight_cutting.risk_multiplier
                }
                opportunities.append(opp_dict)
            
            report_data['analysis_results']['enhanced_opportunities'] = opportunities
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        print(f"📄 Analysis report exported to {filepath}")
    
    def get_performance_dashboard(self) -> Dict:
        """Get performance dashboard data"""
        return self.betting_strategy.get_performance_summary()


def demonstrate_enhanced_analysis():
    """Demonstrate the enhanced profitability analysis"""
    print("🚀 Enhanced Profitability Analysis Demo")
    print("=" * 50)
    
    # Sample predictions
    sample_predictions = {
        'Ilia Topuria': 0.6953,
        'Charles Oliveira': 0.3047,
        'Alexandre Pantoja': 0.5489,
        'Kai Kara-France': 0.4511,
    }
    
    # Sample fighter data (in practice, this would come from your fighter database)
    sample_fighter_data = {
        'Ilia Topuria': {
            'td_avg': 0.5,
            'td_def': 0.85,
            'ko_percentage': 0.75,
            'recent_ko_losses': [],
            'third_round_performance': 0.80,
            'stance': 'Southpaw',
            'professional_nutrition_team': True,
            'easy_cut_history': True
        },
        'Charles Oliveira': {
            'td_avg': 1.8,
            'td_def': 0.60,
            'ko_percentage': 0.40,
            'recent_ko_losses': ['Gaethje KO'],
            'third_round_performance': 0.85,
            'stance': 'Orthodox',
            'southpaw_experience': 0.30
        }
    }
    
    # Initialize analyzer (use cached odds for demo)
    analyzer = EnhancedProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
    
    # Run enhanced analysis
    results = analyzer.analyze_with_optimal_strategy(
        model_predictions=sample_predictions,
        fighter_data=sample_fighter_data
    )
    
    # Display results
    if results['status'] == 'success':
        print(f"\n📊 ENHANCED ANALYSIS RESULTS:")
        print(f"✅ Found {len(results['enhanced_opportunities'])} optimal betting opportunities")
        
        # Show betting instructions
        instructions = analyzer.get_betting_instructions(results)
        for instruction in instructions:
            print(instruction)
    else:
        print(f"❌ Analysis failed: {results.get('message', 'Unknown error')}")
    
    return analyzer, results


if __name__ == "__main__":
    demonstrate_enhanced_analysis()