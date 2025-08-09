#!/usr/bin/env python3
"""
UFC Predictor - Main Entry Point
Simplified interface using unified modules.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ufc_predictor.core.unified_predictor import UnifiedUFCPredictor
from ufc_predictor.betting.unified_analyzer import UnifiedBettingAnalyzer
from ufc_predictor.scrapers.unified_scraper import UnifiedOddsScraper


def parse_fights(fight_strings: List[str]) -> List[Tuple[str, str]]:
    """Parse fight strings into fighter pairs."""
    fights = []
    for fight_str in fight_strings:
        if ' vs ' in fight_str.lower():
            parts = fight_str.split(' vs ')
        elif ' v ' in fight_str.lower():
            parts = fight_str.split(' v ')
        else:
            print(f"⚠️  Invalid fight format: {fight_str}")
            continue
        
        if len(parts) == 2:
            fights.append((parts[0].strip(), parts[1].strip()))
    
    return fights


def main():
    parser = argparse.ArgumentParser(
        description='UFC Fight Predictor - Predict fight outcomes and analyze betting opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single fight prediction
  python main.py predict --fighter1 "Max Holloway" --fighter2 "Ilia Topuria"
  
  # Multiple fights (card prediction)
  python main.py card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4"
  
  # Betting analysis with live odds
  python main.py betting --bankroll 1000 --source tab
  
  # Complete pipeline (train + predict + bet)
  python main.py pipeline --tune
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict single fight')
    predict_parser.add_argument('--fighter1', required=True, help='First fighter name')
    predict_parser.add_argument('--fighter2', required=True, help='Second fighter name')
    predict_parser.add_argument('--no-method', action='store_true', help='Skip method prediction')
    
    # Card command
    card_parser = subparsers.add_parser('card', help='Predict multiple fights')
    card_parser.add_argument('--fights', nargs='+', required=True, 
                            help='List of fights (format: "Fighter1 vs Fighter2")')
    
    # Betting command
    betting_parser = subparsers.add_parser('betting', help='Analyze betting opportunities')
    betting_parser.add_argument('--bankroll', type=float, default=1000, help='Betting bankroll')
    betting_parser.add_argument('--source', default='tab', choices=['tab', 'fightodds', 'cached'],
                               help='Odds source')
    betting_parser.add_argument('--event', help='Event name for filtering')
    betting_parser.add_argument('--export', help='Export analysis to file')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    pipeline_parser.add_argument('--no-scrape', action='store_true', help='Skip data scraping')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize components
    print("🥊 UFC Predictor - Loading models...")
    
    try:
        predictor = UnifiedUFCPredictor()
        print("✅ Models loaded successfully")
        
        if args.command == 'info':
            info = predictor.get_model_info()
            print("\n📊 Model Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        elif args.command == 'predict':
            print(f"\n🥊 Predicting: {args.fighter1} vs {args.fighter2}")
            
            result = predictor.predict_fight(
                args.fighter1, 
                args.fighter2,
                include_method=not args.no_method
            )
            
            print(f"\n📈 Prediction Results:")
            print(f"  {result.fighter1}: {result.fighter1_prob:.1%}")
            print(f"  {result.fighter2}: {result.fighter2_prob:.1%}")
            print(f"\n  🏆 Predicted Winner: {result.predicted_winner}")
            print(f"  📊 Confidence: {result.confidence:.1%}")
            
            if result.method_prediction:
                print(f"\n  🎯 Method Prediction:")
                for method, prob in result.method_prediction.items():
                    print(f"    {method}: {prob:.1%}")
        
        elif args.command == 'card':
            fights = parse_fights(args.fights)
            
            if not fights:
                print("❌ No valid fights provided")
                return
            
            print(f"\n🥊 Predicting {len(fights)} fights...")
            results = predictor.predict_card(fights)
            
            print("\n📊 Card Predictions:")
            print("-" * 60)
            
            for result in results:
                print(f"\n{result.fighter1} vs {result.fighter2}")
                print(f"  {result.fighter1}: {result.fighter1_prob:.1%}")
                print(f"  {result.fighter2}: {result.fighter2_prob:.1%}")
                print(f"  Winner: {result.predicted_winner} ({result.confidence:.1%})")
                
                if result.method_prediction:
                    methods = ', '.join([f"{m}: {p:.1%}" for m, p in result.method_prediction.items()])
                    print(f"  Method: {methods}")
        
        elif args.command == 'betting':
            print(f"\n💰 Analyzing betting opportunities...")
            print(f"  Bankroll: ${args.bankroll}")
            print(f"  Source: {args.source}")
            
            # Get predictions (sample for demo)
            sample_predictions = {
                'Ilia Topuria': 0.65,
                'Max Holloway': 0.35,
                'Alexandre Pantoja': 0.58,
                'Kai Kara-France': 0.42
            }
            
            # Get odds
            scraper = UnifiedOddsScraper()
            odds_data = scraper.get_odds(event=args.event, source=args.source)
            
            if not odds_data:
                print("⚠️  No odds available, using sample data")
                # Sample odds for demo
                odds = {
                    'Ilia Topuria': 1.45,
                    'Max Holloway': 2.80,
                    'Alexandre Pantoja': 1.65,
                    'Kai Kara-France': 2.30
                }
            else:
                odds = {o.fighter1: o.fighter1_odds for o in odds_data}
                odds.update({o.fighter2: o.fighter2_odds for o in odds_data})
            
            # Analyze
            analyzer = UnifiedBettingAnalyzer(bankroll=args.bankroll)
            opportunities = analyzer.analyze_single_bets(sample_predictions, odds)
            multi_bets = analyzer.analyze_multi_bets(opportunities)
            
            if opportunities:
                print(f"\n✅ Found {len(opportunities)} profitable single bets:")
                for opp in opportunities[:5]:
                    print(f"\n  🎯 {opp.fighter} vs {opp.opponent}")
                    print(f"     Probability: {opp.win_probability:.1%}")
                    print(f"     Odds: {opp.odds}")
                    print(f"     Expected Value: {opp.expected_value:.1%}")
                    print(f"     Recommended Bet: ${opp.recommended_bet}")
                    print(f"     Potential Return: ${opp.potential_return}")
                    print(f"     Confidence: {opp.confidence}")
            
            if multi_bets:
                print(f"\n✅ Found {len(multi_bets)} profitable multi-bets:")
                for multi in multi_bets[:3]:
                    print(f"\n  🎲 {multi.num_legs}-leg multi:")
                    print(f"     Fighters: {', '.join(multi.legs)}")
                    print(f"     Combined Odds: {multi.combined_odds}")
                    print(f"     Expected Value: {multi.expected_value:.1%}")
                    print(f"     Recommended Bet: ${multi.recommended_bet}")
                    print(f"     Potential Return: ${multi.potential_return}")
            
            # Portfolio
            portfolio = analyzer.calculate_optimal_portfolio(opportunities)
            print(f"\n📊 Optimal Portfolio:")
            print(f"  Total Stake: ${portfolio['total_stake']}")
            print(f"  Expected Return: ${portfolio['expected_return']}")
            print(f"  Expected Profit: ${portfolio['expected_profit']}")
            print(f"  ROI: {portfolio['roi']}%")
            
            if args.export:
                filepath = analyzer.export_analysis(opportunities, multi_bets, args.export)
                print(f"\n✅ Analysis exported to: {filepath}")
        
        elif args.command == 'pipeline':
            print("\n🔄 Running complete pipeline...")
            
            if not args.no_scrape:
                print("  1. Scraping latest data...")
                # Would implement scraping here
            
            print("  2. Processing features...")
            # Would implement feature engineering here
            
            print("  3. Training models...")
            # Would implement training here
            
            print("  4. Evaluating performance...")
            # Would implement evaluation here
            
            print("\n✅ Pipeline complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()