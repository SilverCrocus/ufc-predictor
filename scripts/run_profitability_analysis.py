#!/usr/bin/env python3
"""
🥊 UFC Profitability Analysis - Terminal Version

Quick command-line tool to analyze betting profitability with live TAB Australia odds.

Usage:
    python run_profitability_analysis.py
    python run_profitability_analysis.py --bankroll 500
    python run_profitability_analysis.py --no-live-odds  # Use cached odds
    python run_profitability_analysis.py --predictions "fighter1:0.65,fighter2:0.35"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from ufc_predictor.betting.tab_profitability import TABProfitabilityAnalyzer

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
        'Felipe Lima': 0.5702,
        'Payton Talbott': 0.4298,
    }

def main():
    parser = argparse.ArgumentParser(
        description='🥊 UFC Profitability Analysis with Live TAB Australia Odds',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1000,
        help='Your betting bankroll in AUD (default: 1000)'
    )
    
    parser.add_argument(
        '--no-live-odds',
        action='store_true',
        help='Use cached odds instead of scraping live (faster)'
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Predictions in format "Fighter1:0.65,Fighter2:0.35"'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample predictions for testing'
    )
    
    args = parser.parse_args()
    
    # Header
    print("🥊 UFC PROFITABILITY ANALYSIS")
    print("=" * 50)
    print(f"💳 Bankroll: ${args.bankroll:,.2f} AUD")
    print(f"🔄 Live odds: {'Disabled' if args.no_live_odds else 'Enabled'}")
    print()
    
    # Get predictions
    if args.predictions:
        predictions = parse_predictions(args.predictions)
        print(f"✅ Using provided predictions ({len(predictions)} fighters)")
    elif args.sample:
        predictions = get_sample_predictions()
        print(f"📝 Using sample predictions ({len(predictions)} fighters)")
    else:
        # Interactive mode
        print("📝 No predictions provided. Choose an option:")
        print("1. Use sample predictions")
        print("2. Enter predictions manually")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            predictions = get_sample_predictions()
            print(f"✅ Using sample predictions ({len(predictions)} fighters)")
        elif choice == '2':
            predictions = {}
            print("\nEnter predictions (press Enter with empty fighter name to finish):")
            while True:
                fighter = input("Fighter name: ").strip()
                if not fighter:
                    break
                try:
                    prob = float(input(f"Probability for {fighter} (0.0-1.0): ").strip())
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
    
    print(f"\n🎯 Analyzing {len(predictions)} predictions...")
    print("-" * 30)
    
    # Initialize analyzer
    analyzer = TABProfitabilityAnalyzer(
        bankroll=args.bankroll,
        use_live_odds=not args.no_live_odds
    )
    
    try:
        # Run analysis
        results = analyzer.analyze_predictions(predictions)
        
        # Display results
        analyzer._print_summary(results)
        
        # Show betting instructions
        print("\n" + "="*50)
        instructions = analyzer.get_betting_instructions(results)
        for instruction in instructions:
            print(instruction)
        
        # Debug info if needed
        if not results.get('opportunities', []):
            print("\n🔧 DEBUGGING INFO:")
            print("-" * 30)
            analyzer.debug_missing_fighters(predictions, analyzer.current_tab_odds)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("💡 Try running with --no-live-odds for faster testing")

if __name__ == "__main__":
    main() 