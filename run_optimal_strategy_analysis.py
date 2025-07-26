#!/usr/bin/env python3
"""
🚀 UFC Optimal Strategy Analysis - Enhanced Version

Advanced command-line tool for comprehensive UFC betting analysis using the
optimal betting strategy framework. Integrates model predictions, market analysis,
risk management, and portfolio optimization.

Usage:
    python run_optimal_strategy_analysis.py --bankroll 1000 --live-odds
    python run_optimal_strategy_analysis.py --sample --export results.json
    python run_optimal_strategy_analysis.py --config custom_config.json
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.enhanced_profitability_analyzer import EnhancedProfitabilityAnalyzer

def parse_predictions(prediction_string: str) -> Dict[str, float]:
    """Parse prediction string format: 'Fighter1:0.65,Fighter2:0.35'"""
    predictions = {}
    
    for pair in prediction_string.split(','):
        if ':' in pair:
            fighter, prob = pair.strip().split(':')
            predictions[fighter.strip()] = float(prob.strip())
    
    return predictions

def get_sample_predictions() -> Dict[str, float]:
    """Return sample predictions for testing"""
    return {
        'Ilia Topuria': 0.6953,
        'Charles Oliveira': 0.3047,
        'Alexandre Pantoja': 0.5489,
        'Kai Kara-France': 0.4511,
        'Joshua Van': 0.6356,
        'Brandon Royval': 0.3644,
        'Renato Moicano': 0.5212,
        'Beneil Dariush': 0.4788,
    }

def get_sample_fighter_data() -> Dict[str, Dict]:
    """Return sample fighter statistics for enhanced analysis"""
    return {
        'Ilia Topuria': {
            'td_avg': 0.5,
            'td_def': 0.85,
            'ko_percentage': 0.75,
            'recent_ko_losses': [],
            'third_round_performance': 0.80,
            'stance': 'Southpaw',
            'southpaw_experience': 0.70,
            'missed_weight_last_2': False,
            'struggled_at_weigh_ins': False,
            'same_day_cut_pct': 0.03,
            'looks_depleted': False,
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
            'southpaw_experience': 0.30,
            'missed_weight_last_2': True,
            'struggled_at_weigh_ins': False,
            'same_day_cut_pct': 0.08,
            'looks_depleted': False,
            'professional_nutrition_team': False,
            'easy_cut_history': False
        },
        'Alexandre Pantoja': {
            'td_avg': 2.5,
            'td_def': 0.75,
            'ko_percentage': 0.25,
            'recent_ko_losses': [],
            'third_round_performance': 0.90,
            'stance': 'Orthodox',
            'southpaw_experience': 0.60,
            'missed_weight_last_2': False,
            'professional_nutrition_team': True,
            'easy_cut_history': True
        },
        'Kai Kara-France': {
            'td_avg': 0.8,
            'td_def': 0.45,
            'ko_percentage': 0.60,
            'recent_ko_losses': ['Pantoja KO'],
            'third_round_performance': 0.40,
            'stance': 'Orthodox',
            'missed_weight_last_2': False,
            'professional_nutrition_team': False,
            'easy_cut_history': True
        },
        'Joshua Van': {
            'td_avg': 1.2,
            'td_def': 0.70,
            'ko_percentage': 0.45,
            'recent_ko_losses': [],
            'third_round_performance': 0.75,
            'stance': 'Orthodox',
            'professional_nutrition_team': True,
            'easy_cut_history': True
        },
        'Brandon Royval': {
            'td_avg': 2.0,
            'td_def': 0.55,
            'ko_percentage': 0.30,
            'recent_ko_losses': [],
            'third_round_performance': 0.80,
            'stance': 'Orthodox',
            'struggled_at_weigh_ins': True,
            'professional_nutrition_team': False,
            'easy_cut_history': False
        }
    }

def display_enhanced_results(results: Dict, show_details: bool = True):
    """Display enhanced analysis results in formatted output"""
    
    if results['status'] != 'success':
        print(f"❌ Analysis failed: {results.get('message', 'Unknown error')}")
        return
    
    opportunities = results['enhanced_opportunities']
    strategy_summary = results['strategy_summary']
    portfolio_analysis = results['portfolio_analysis']
    risk_assessment = results['risk_assessment']
    
    print("\n🚀 OPTIMAL STRATEGY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Strategy Summary
    print(f"\n📊 STRATEGY OVERVIEW:")
    print(f"   💰 Total opportunities: {strategy_summary['total_opportunities']}")
    print(f"   💵 Recommended stake: ${strategy_summary['total_recommended_stake']:.2f}")
    print(f"   📈 Expected profit: ${strategy_summary['total_expected_profit']:.2f}")
    print(f"   🎯 Portfolio ROI: {strategy_summary['portfolio_expected_roi']:.1f}%")
    print(f"   📊 Bankroll utilization: {strategy_summary['bankroll_utilization']:.1f}%")
    print(f"   ⭐ Avg confidence: {strategy_summary['avg_confidence_score']:.1f}%")
    
    # Risk Assessment
    print(f"\n⚖️  RISK ASSESSMENT:")
    print(f"   🚨 Risk level: {risk_assessment['risk_level'].upper()}")
    print(f"   📊 Correlation risk: {risk_assessment['correlation_risk']:.1%}")
    print(f"   💎 Diversification: {risk_assessment['diversification_score']:.1%}")
    print(f"   🔒 Max single bet: {risk_assessment['max_single_bet_pct']:.1%} of bankroll")
    
    # Individual Opportunities
    if show_details and opportunities:
        print(f"\n💰 BETTING OPPORTUNITIES:")
        print("-" * 50)
        
        for i, opp in enumerate(opportunities, 1):
            print(f"\n   {i}. {opp.fighter} vs {opp.opponent}")
            print(f"      💵 Stake: ${opp.adjusted_bet_size:.2f} ({opp.risk_tier.value.upper()})")
            print(f"      📊 TAB Odds: {opp.market_odds:+d} | EV: {opp.expected_value:.1%}")
            print(f"      🎯 Model: {opp.model_prob:.1%} | Confidence: {opp.confidence_score:.1%}")
            print(f"      ⚡ Style: {opp.style_matchup.confidence_multiplier:.2f}x | Weight: {opp.weight_cutting.risk_multiplier:.2f}x")
            
            if opp.correlation_penalty > 0:
                print(f"      ⚠️  Correlation penalty: {opp.correlation_penalty:.1%}")
    
    # Multi-bet opportunities
    multi_bets = results.get('multi_bet_opportunities', [])
    if multi_bets:
        print(f"\n🎰 MULTI-BET OPPORTUNITIES:")
        print("-" * 30)
        for i, mb in enumerate(multi_bets[:3], 1):
            fighters = " + ".join(mb['fighters'])
            print(f"   {i}. {fighters}")
            print(f"      💰 Stake: ${mb['recommended_stake']:.2f} | EV: {mb['expected_value']:.1%}")
            print(f"      📈 Profit: ${mb['expected_profit']:.2f} | Odds: {mb['combined_decimal_odds']:.2f}")
    
    # Portfolio Analysis
    if show_details:
        print(f"\n📈 PORTFOLIO ANALYSIS:")
        print(f"   📊 Total exposure: ${portfolio_analysis['total_exposure']:.2f} ({portfolio_analysis['total_exposure_pct']:.1%})")
        print(f"   💎 Expected return: ${portfolio_analysis['expected_return']:.2f}")
        print(f"   🎯 Portfolio Kelly: {portfolio_analysis['portfolio_kelly']:.3f}")
        
        print(f"\n   Risk Distribution:")
        for tier, data in portfolio_analysis['risk_distribution'].items():
            if data['count'] > 0:
                print(f"     • {tier.title()}: {data['count']} bets, ${data['total_stake']:.2f}, {data['avg_ev']:.1%} avg EV")

def display_timing_recommendations(results: Dict):
    """Display market timing recommendations"""
    timing_recs = results.get('timing_recommendations', {})
    
    if not any(timing_recs.values()):
        return
    
    print(f"\n⏰ MARKET TIMING STRATEGY:")
    print("-" * 30)
    
    for rec in timing_recs.get('opening_line_bets', []):
        print(f"   📊 {rec['fighter']}: {rec['stake_split']}")
        print(f"      Reason: {rec['reason']}")
    
    for rec in timing_recs.get('closing_line_bets', []):
        print(f"   📊 {rec['fighter']}: {rec['stake_split']}")
        print(f"      Reason: {rec['reason']}")
    
    for rec in timing_recs.get('live_betting_candidates', []):
        print(f"   🔴 {rec['fighter']}: Live betting candidate")
        print(f"      Watch for: {rec['watch_for']}")

def main():
    parser = argparse.ArgumentParser(
        description='🚀 UFC Optimal Strategy Analysis - Enhanced Profitability System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimal_strategy_analysis.py --sample --bankroll 1000
  python run_optimal_strategy_analysis.py --predictions "Topuria:0.7,Oliveira:0.3"
  python run_optimal_strategy_analysis.py --live-odds --export results.json
  python run_optimal_strategy_analysis.py --config custom_strategy.json
        """
    )
    
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1000,
        help='Your betting bankroll in AUD (default: 1000)'
    )
    
    parser.add_argument(
        '--live-odds',
        action='store_true',
        help='Use live TAB Australia odds scraping (slower but current)'
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Predictions in format "Fighter1:0.65,Fighter2:0.35"'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample predictions and fighter data for testing'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom strategy configuration JSON file'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export detailed results to JSON file'
    )
    
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Show minimal output (summary only)'
    )
    
    parser.add_argument(
        '--fighter-data',
        type=str,
        help='Path to JSON file with detailed fighter statistics'
    )
    
    args = parser.parse_args()
    
    # Header
    print("🚀 UFC OPTIMAL STRATEGY ANALYSIS")
    print("=" * 50)
    print(f"💳 Bankroll: ${args.bankroll:,.2f} AUD")
    print(f"🔄 Live odds: {'Enabled' if args.live_odds else 'Disabled'}")
    print(f"📊 Strategy: Enhanced Framework with Risk Management")
    print()
    
    # Get predictions
    predictions = None
    fighter_data = None
    
    if args.predictions:
        predictions = parse_predictions(args.predictions)
        print(f"✅ Using provided predictions ({len(predictions)} fighters)")
    elif args.sample:
        predictions = get_sample_predictions()
        fighter_data = get_sample_fighter_data()
        print(f"📝 Using sample predictions ({len(predictions)} fighters)")
    else:
        # Interactive mode
        print("📝 No predictions provided. Choose an option:")
        print("1. Use sample predictions with fighter data")
        print("2. Enter predictions manually")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            predictions = get_sample_predictions()
            fighter_data = get_sample_fighter_data()
            print(f"✅ Using sample data ({len(predictions)} fighters)")
        elif choice == '2':
            predictions = {}
            print("\nEnter predictions (press Enter with empty fighter name to finish):")
            while True:
                fighter = input("Fighter name: ").strip()
                if not fighter:
                    break
                try:
                    prob = float(input(f"Win probability for {fighter} (0.0-1.0): ").strip())
                    if 0.0 <= prob <= 1.0:
                        predictions[fighter] = prob
                        print(f"✅ Added {fighter}: {prob:.1%}")
                    else:
                        print("❌ Probability must be between 0.0 and 1.0")
                except ValueError:
                    print("❌ Invalid probability format")
        else:
            print("👋 Goodbye!")
            return
    
    if not predictions:
        print("❌ No predictions provided. Exiting.")
        return
    
    # Load custom fighter data if provided
    if args.fighter_data and Path(args.fighter_data).exists():
        try:
            with open(args.fighter_data, 'r') as f:
                fighter_data = json.load(f)
            print(f"📊 Loaded custom fighter data from {args.fighter_data}")
        except Exception as e:
            print(f"⚠️  Error loading fighter data: {e}. Using defaults.")
    
    print(f"\n🎯 Analyzing {len(predictions)} predictions with optimal strategy...")
    print("-" * 50)
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedProfitabilityAnalyzer(
            bankroll=args.bankroll,
            use_live_odds=args.live_odds,
            strategy_config_path=args.config
        )
        
        # Run enhanced analysis
        results = analyzer.analyze_with_optimal_strategy(
            model_predictions=predictions,
            fighter_data=fighter_data
        )
        
        # Display results
        display_enhanced_results(results, show_details=not args.minimal)
        
        # Show timing recommendations
        if not args.minimal:
            display_timing_recommendations(results)
        
        # Show betting instructions
        print("\n" + "="*60)
        print("📋 BETTING INSTRUCTIONS")
        print("="*60)
        
        instructions = analyzer.get_betting_instructions(results)
        for instruction in instructions:
            print(instruction)
        
        # Export results if requested
        if args.export:
            analyzer.export_analysis_report(args.export, results)
            print(f"\n💾 Detailed results exported to {args.export}")
        
        # Performance dashboard
        if not args.minimal:
            dashboard = analyzer.get_performance_dashboard()
            if dashboard.get('total_bets', 0) > 0:
                print(f"\n📊 PERFORMANCE DASHBOARD:")
                print(f"   📈 Total bets: {dashboard.get('total_bets', 0)}")
                print(f"   🎯 Win rate: {dashboard.get('win_rate', 'N/A')}")
                print(f"   💰 ROI: {dashboard.get('roi', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("💡 Try running with --sample for testing or check your configuration")

if __name__ == "__main__":
    main()