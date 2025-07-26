#!/usr/bin/env python3
"""
Demo: UFC Bet Tracking System

Demonstrates the complete bet tracking workflow with sample data.
Run this to see how the tracking system works with your prediction data.
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.bet_tracking import BetTracker, track_card_predictions, update_fight_results, get_performance_summary

def create_sample_predictions():
    """Create sample prediction data matching your notebook format"""
    
    sample_card_results = [
        {
            'fight': 'Robert Whittaker vs Reinier de Ridder',
            'predicted_winner': 'Robert Whittaker',
            'winner_confidence': '72.5%',
            'predicted_method': 'Decision',
            'method_probabilities': {
                'Decision': '55.2%',
                'KO/TKO': '32.1%',
                'Submission': '12.7%'
            }
        },
        {
            'fight': 'Petr Yan vs Marcus McGhee',
            'predicted_winner': 'Petr Yan',
            'winner_confidence': '89.3%',
            'predicted_method': 'KO/TKO',
            'method_probabilities': {
                'Decision': '25.8%',
                'KO/TKO': '68.4%',
                'Submission': '5.8%'
            }
        },
        {
            'fight': 'Sharaputdin Magomedov vs Marc-André Barriault',
            'predicted_winner': 'Sharaputdin Magomedov',
            'winner_confidence': '68.1%',
            'predicted_method': 'Decision',
            'method_probabilities': {
                'Decision': '58.9%',
                'KO/TKO': '31.2%',
                'Submission': '9.9%'
            }
        },
        {
            'fight': 'Asu Almabayev vs Jose Ochoa',
            'predicted_winner': 'Asu Almabayev',
            'winner_confidence': '76.8%',
            'predicted_method': 'Submission',
            'method_probabilities': {
                'Decision': '34.1%',
                'KO/TKO': '19.3%',
                'Submission': '46.6%'
            }
        },
        {
            'fight': 'Nikita Krylov vs Bogdan Guskov',
            'predicted_winner': 'Nikita Krylov',
            'winner_confidence': '81.7%',
            'predicted_method': 'KO/TKO',
            'method_probabilities': {
                'Decision': '28.9%',
                'KO/TKO': '59.4%',
                'Submission': '11.7%'
            }
        }
    ]
    
    return sample_card_results

def create_sample_profitability():
    """Create sample profitability analysis matching your TAB format"""
    
    from dataclasses import dataclass
    
    @dataclass
    class MockOpportunity:
        fighter: str
        opponent: str
        tab_decimal_odds: float
        american_odds: int
        expected_value: float
        recommended_bet: float
        market_prob: float
    
    opportunities = [
        MockOpportunity(
            fighter="Robert Whittaker",
            opponent="Reinier de Ridder", 
            tab_decimal_odds=1.85,
            american_odds=-118,
            expected_value=0.12,
            recommended_bet=2.14,
            market_prob=0.541
        ),
        MockOpportunity(
            fighter="Petr Yan",
            opponent="Marcus McGhee",
            tab_decimal_odds=1.35,
            american_odds=-286,
            expected_value=0.08,
            recommended_bet=1.71,
            market_prob=0.741
        ),
        MockOpportunity(
            fighter="Asu Almabayev", 
            opponent="Jose Ochoa",
            tab_decimal_odds=1.65,
            american_odds=-154,
            expected_value=0.15,
            recommended_bet=3.21,
            market_prob=0.606
        )
    ]
    
    return {
        'opportunities': opportunities,
        'total_expected_profit': 15.47,
        'total_opportunities': len(opportunities),
        'bankroll': 21.38
    }

def demo_tracking_workflow():
    """Demonstrate the complete tracking workflow"""
    
    print("🎯 UFC BET TRACKING SYSTEM DEMO")
    print("=" * 50)
    
    # Step 1: Record predictions
    print("\n📊 STEP 1: Recording Predictions")
    print("-" * 35)
    
    sample_predictions = create_sample_predictions()
    sample_profitability = create_sample_profitability()
    
    event_id = track_card_predictions(
        card_results=sample_predictions,
        profitability_results=sample_profitability,
        event_name="UFC Fight Night - Whittaker vs de Ridder",
        bankroll=21.38
    )
    
    print(f"✅ Recorded event: {event_id}")
    
    # Step 2: Show tracking status
    print(f"\n📋 STEP 2: Current Tracking Status")
    print("-" * 35)
    
    tracker = BetTracker()
    tracker.print_summary()
    
    # Step 3: Simulate fight results (in real use, this would be manual)
    print(f"\n🥊 STEP 3: Simulating Fight Results")
    print("-" * 35)
    
    # Simulate realistic results based on predictions
    fight_results = {
        "Robert Whittaker": {"winner": True, "method": "Decision", "round": "3"},
        "Reinier de Ridder": {"winner": False, "method": "Decision", "round": "3"}, 
        "Petr Yan": {"winner": True, "method": "TKO", "round": "2"},
        "Marcus McGhee": {"winner": False, "method": "TKO", "round": "2"},
        "Sharaputdin Magomedov": {"winner": False, "method": "KO", "round": "1"},  # Upset!
        "Marc-André Barriault": {"winner": True, "method": "KO", "round": "1"},
        "Asu Almabayev": {"winner": True, "method": "Submission", "round": "2"},
        "Jose Ochoa": {"winner": False, "method": "Submission", "round": "2"},
        "Nikita Krylov": {"winner": True, "method": "Decision", "round": "3"},  # Wrong method
        "Bogdan Guskov": {"winner": False, "method": "Decision", "round": "3"}
    }
    
    success = update_fight_results(event_id, fight_results)
    print(f"✅ Results updated: {success}")
    
    # Step 4: Performance analysis
    print(f"\n📈 STEP 4: Performance Analysis")
    print("-" * 35)
    
    performance = get_performance_summary()
    
    if 'error' not in performance:
        model = performance.get('model_performance', {})
        portfolio = performance.get('portfolio_performance', {})
        
        print(f"🎯 MODEL PERFORMANCE:")
        print(f"   Prediction Accuracy: {model.get('prediction_accuracy', 0):.1%}")
        print(f"   Method Accuracy: {model.get('method_accuracy', 0):.1%}")
        
        if portfolio.get('total_bets', 0) > 0:
            print(f"\n💰 PORTFOLIO PERFORMANCE:")
            print(f"   Total ROI: {portfolio.get('total_roi_percentage', 0):.1f}%")
            print(f"   Win Rate: {portfolio.get('win_rate', 0):.1%}")
            print(f"   Total Profit: ${portfolio.get('total_profit', 0):.2f}")
            print(f"   Ending Bankroll: ${portfolio.get('ending_bankroll', 0):.2f}")
    
    # Step 5: Export data
    print(f"\n📄 STEP 5: Data Export")
    print("-" * 35)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = f"performance_tracking/demo_export_{timestamp}.csv"
    
    success = tracker.export_to_csv(export_file, event_id)
    if success:
        print(f"✅ Event data exported to: {export_file}")
    
    # Generate performance report
    report_file = tracker.generate_performance_report()
    print(f"📊 Performance report: {report_file}")
    
    print(f"\n🎉 DEMO COMPLETE!")
    print("-" * 20)
    print("The bet tracking system is now ready for use with your real predictions.")
    print("Integration instructions are in: notebook_bet_tracking_integration.py")

def show_csv_structure():
    """Show the CSV structure with sample data"""
    
    print("\n📋 CSV STRUCTURE DEMO")
    print("=" * 30)
    
    try:
        tracker = BetTracker()
        df = pd.read_csv(tracker.tracking_file)
        
        if not df.empty:
            print(f"✅ Tracking file contains {len(df)} records")
            print(f"📊 Columns: {len(df.columns)}")
            print(f"🎯 Events: {df['event_id'].nunique()}")
            
            # Show column names
            print(f"\n📑 COLUMN STRUCTURE:")
            columns = df.columns.tolist()
            for i, col in enumerate(columns, 1):
                print(f"   {i:2d}. {col}")
            
            # Show sample record
            if len(df) > 0:
                print(f"\n📄 SAMPLE RECORD (latest):")
                latest = df.iloc[-1]
                key_fields = [
                    'event_name', 'fighter_name', 'predicted_winner', 
                    'predicted_probability', 'bet_placed', 'stake_amount',
                    'expected_value', 'actual_result'
                ]
                
                for field in key_fields:
                    if field in latest:
                        value = latest[field]
                        print(f"   {field}: {value}")
        
        else:
            print("📄 Tracking file is empty - run the demo first")
            
    except FileNotFoundError:
        print("📄 No tracking file found - run the demo first")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UFC Bet Tracking System Demo")
    parser.add_argument("--demo", action="store_true", help="Run full tracking demo")
    parser.add_argument("--structure", action="store_true", help="Show CSV structure")
    parser.add_argument("--summary", action="store_true", help="Show tracking summary")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_tracking_workflow()
    elif args.structure:
        show_csv_structure()
    elif args.summary:
        tracker = BetTracker()
        tracker.print_summary()
        
        # Also show performance if available
        performance = get_performance_summary()
        if 'error' not in performance:
            portfolio = performance.get('portfolio_performance', {})
            if portfolio.get('total_bets', 0) > 0:
                print(f"\n💰 QUICK PERFORMANCE:")
                print(f"   ROI: {portfolio.get('total_roi_percentage', 0):.1f}%")
                print(f"   Win Rate: {portfolio.get('win_rate', 0):.1%}")
                print(f"   Profit: ${portfolio.get('total_profit', 0):.2f}")
    else:
        print("🎯 UFC Bet Tracking System")
        print("=" * 30)
        print("Available commands:")
        print("  python demo_bet_tracking.py --demo      # Run full demo")
        print("  python demo_bet_tracking.py --structure # Show CSV structure") 
        print("  python demo_bet_tracking.py --summary   # Show tracking summary")
        print()
        print("📚 For notebook integration, see: notebook_bet_tracking_integration.py")