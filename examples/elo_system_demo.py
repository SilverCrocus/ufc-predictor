"""
UFC ELO System Demo

This example demonstrates the complete UFC ELO rating system, including:
1. Historical data processing
2. Multi-dimensional ELO training
3. Predictions and analysis
4. Integration with existing ML pipeline
5. Validation and benchmarking

Run this script to see the ELO system in action.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_elo_system import UFCELOSystem, ELOConfig
from multi_dimensional_elo import MultiDimensionalUFCELO
from elo_historical_processor import UFCHistoricalProcessor
from elo_integration import ELOIntegration
from elo_validation import ELOValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_sample_data():
    """Create realistic sample data for demonstration"""
    
    # Generate sample fight data
    fighters = [
        'Jon Jones', 'Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou',
        'Israel Adesanya', 'Kamaru Usman', 'Khabib Nurmagomedov', 'Conor McGregor',
        'Amanda Nunes', 'Valentina Shevchenko', 'Rose Namajunas', 'Weili Zhang'
    ]
    
    methods = ['U-DEC', 'M-DEC', 'S-DEC', 'KO', 'TKO', 'Submission']
    method_weights = [0.4, 0.1, 0.1, 0.15, 0.15, 0.1]  # Realistic distribution
    
    # Generate 200 sample fights over 5 years
    np.random.seed(42)  # For reproducible results
    
    fights_data = []
    current_date = datetime(2018, 1, 1)
    
    for i in range(200):
        fighter1, fighter2 = np.random.choice(fighters, 2, replace=False)
        winner = np.random.choice([fighter1, fighter2])
        method = np.random.choice(methods, p=method_weights)
        
        # Adjust round based on method
        if method in ['KO', 'TKO']:
            round_num = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        elif method == 'Submission':
            round_num = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:  # Decision
            round_num = np.random.choice([3, 5], p=[0.8, 0.2])
        
        # Add some title fights
        is_title = np.random.random() < 0.15  # 15% are title fights
        is_main = is_title or np.random.random() < 0.25  # Title fights + 25% of others
        
        fights_data.append({
            'Fighter': fighter1,
            'Opponent': fighter2,
            'Outcome': 'win' if winner == fighter1 else 'loss',
            'Method': method,
            'Round': round_num,
            'Event': f'UFC {250 + i}' if is_title else f'UFC Fight Night {i}',
            'Date': current_date.strftime('%Y-%m-%d'),
            'is_title_fight': is_title,
            'is_main_event': is_main
        })
        
        # Advance date (roughly one fight per week)
        current_date += timedelta(days=np.random.randint(5, 10))
    
    fights_df = pd.DataFrame(fights_data)
    
    # Generate sample fighter data
    fighter_weights = {
        'Jon Jones': 205, 'Daniel Cormier': 205, 'Stipe Miocic': 245, 
        'Francis Ngannou': 260, 'Israel Adesanya': 185, 'Kamaru Usman': 170,
        'Khabib Nurmagomedov': 155, 'Conor McGregor': 155,
        'Amanda Nunes': 135, 'Valentina Shevchenko': 125,
        'Rose Namajunas': 115, 'Weili Zhang': 115
    }
    
    fighters_data = []
    for fighter in fighters:
        # Generate realistic fighter stats
        wins = np.random.randint(15, 30)
        losses = np.random.randint(0, 5)
        
        fighters_data.append({
            'Name': fighter,
            'Weight (lbs)': fighter_weights.get(fighter, 185),
            'Wins': wins,
            'Losses': losses,
            'Draws': 0,
            'Height (inches)': np.random.randint(66, 78),
            'Reach (in)': np.random.randint(70, 80),
            'Age': np.random.randint(25, 35),
            'SLpM': np.random.uniform(2.0, 6.0),
            'Str. Acc.': np.random.uniform(0.35, 0.65),
            'TD Avg.': np.random.uniform(0.0, 3.0),
            'Sub. Avg.': np.random.uniform(0.0, 1.0)
        })
    
    fighters_df = pd.DataFrame(fighters_data)
    
    return fights_df, fighters_df


def demonstrate_basic_elo_system():
    """Demonstrate basic ELO system functionality"""
    
    print("\n" + "="*60)
    print("BASIC ELO SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize ELO system with custom configuration
    config = ELOConfig(
        initial_rating=1400,
        base_k_factor=32,
        rookie_k_factor=45,
        veteran_k_factor=24
    )
    
    elo_system = UFCELOSystem(config)
    
    # Simulate some fights manually
    example_fights = [
        {
            'fighter1_name': 'Jon Jones',
            'fighter2_name': 'Daniel Cormier',
            'winner_name': 'Jon Jones',
            'method': 'U-DEC',
            'fight_date': datetime(2015, 1, 3),
            'round_num': 5,
            'is_title_fight': True,
            'is_main_event': True,
            'fighter1_weight': 205,
            'fighter2_weight': 205
        },
        {
            'fighter1_name': 'Daniel Cormier',
            'fighter2_name': 'Stipe Miocic',
            'winner_name': 'Daniel Cormier',
            'method': 'KO',
            'fight_date': datetime(2018, 7, 7),
            'round_num': 1,
            'is_title_fight': True,
            'is_main_event': True,
            'fighter1_weight': 240,
            'fighter2_weight': 245
        },
        {
            'fighter1_name': 'Stipe Miocic',
            'fighter2_name': 'Francis Ngannou',
            'winner_name': 'Francis Ngannou',
            'method': 'KO',
            'fight_date': datetime(2021, 3, 27),
            'round_num': 2,
            'is_title_fight': True,
            'is_main_event': True,
            'fighter1_weight': 245,
            'fighter2_weight': 260
        }
    ]
    
    print("Processing example fights...")
    for fight in example_fights:
        result = elo_system.process_fight(**fight)
        
        print(f"\n{fight['fighter1_name']} vs {fight['fighter2_name']}")
        print(f"Winner: {fight['winner_name']} via {fight['method']}")
        print(f"Rating changes: {result['rating1_after'] - result['rating1_before']:.1f}, "
              f"{result['rating2_after'] - result['rating2_before']:.1f}")
    
    # Show current rankings
    print("\nCurrent ELO Rankings:")
    rankings = elo_system.get_rankings(top_n=6)
    for i, fighter in enumerate(rankings, 1):
        print(f"{i}. {fighter['name']}: {fighter['rating']:.1f} "
              f"(±{fighter['uncertainty']:.0f}, {fighter['fights_count']} fights)")
    
    # Make a prediction
    prediction = elo_system.predict_fight_outcome('Jon Jones', 'Francis Ngannou', include_uncertainty=True)
    print(f"\nPrediction - Jon Jones vs Francis Ngannou:")
    print(f"Jon Jones win probability: {prediction['fighter1_win_prob']:.3f}")
    print(f"Francis Ngannou win probability: {prediction['fighter2_win_prob']:.3f}")
    print(f"Confidence level: {prediction.get('confidence_level', 'medium')}")


def demonstrate_multi_dimensional_elo():
    """Demonstrate multi-dimensional ELO system"""
    
    print("\n" + "="*60)
    print("MULTI-DIMENSIONAL ELO SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize multi-dimensional ELO system
    multi_elo = MultiDimensionalUFCELO()
    
    # Process the same fights as above
    example_fights = [
        {
            'fighter1_name': 'Jon Jones',
            'fighter2_name': 'Daniel Cormier',
            'winner_name': 'Jon Jones',
            'method': 'U-DEC',
            'fight_date': datetime(2015, 1, 3),
            'round_num': 5,
            'is_title_fight': True,
            'is_main_event': True
        },
        {
            'fighter1_name': 'Conor McGregor',
            'fighter2_name': 'Jose Aldo',
            'winner_name': 'Conor McGregor',
            'method': 'KO',
            'fight_date': datetime(2015, 12, 12),
            'round_num': 1,
            'is_title_fight': True
        },
        {
            'fighter1_name': 'Khabib Nurmagomedov',
            'fighter2_name': 'Conor McGregor',
            'winner_name': 'Khabib Nurmagomedov',
            'method': 'Submission',
            'fight_date': datetime(2018, 10, 6),
            'round_num': 4,
            'is_title_fight': True
        }
    ]
    
    print("Processing fights with dimensional analysis...")
    for fight in example_fights:
        result = multi_elo.process_fight(**fight)
        
        print(f"\n{fight['fighter1_name']} vs {fight['fighter2_name']} - {fight['method']}")
        print(f"Dimensional impacts: {result['dimension_impacts']}")
        
        # Show rating changes across dimensions
        f1_changes = result['post_fight_ratings']['fighter1']
        f1_before = result['pre_fight_ratings']['fighter1']
        
        print(f"{fight['fighter1_name']} rating changes:")
        for dim in ['overall', 'striking', 'grappling', 'cardio']:
            change = f1_changes[dim] - f1_before[dim]
            print(f"  {dim.capitalize()}: {change:+.1f} (now {f1_changes[dim]:.1f})")
    
    # Get dimensional rankings
    print("\nStriking Rankings:")
    striking_rankings = multi_elo.get_dimensional_rankings('striking', top_n=5)
    for i, fighter in enumerate(striking_rankings, 1):
        print(f"{i}. {fighter['name']}: {fighter['rating']:.1f}")
    
    print("\nGrappling Rankings:")
    grappling_rankings = multi_elo.get_dimensional_rankings('grappling', top_n=5)
    for i, fighter in enumerate(grappling_rankings, 1):
        print(f"{i}. {fighter['name']}: {fighter['rating']:.1f}")
    
    # Make multi-dimensional prediction
    prediction = multi_elo.predict_fight_outcome('Conor McGregor', 'Khabib Nurmagomedov')
    print(f"\nMulti-dimensional prediction - Conor McGregor vs Khabib Nurmagomedov:")
    print(f"Win probability: {prediction['fighter1_win_prob']:.3f}")
    print(f"Dimensional advantages: {prediction['dimensional_advantages']}")
    print(f"Method predictions: {prediction['method_predictions']}")
    print(f"Style analysis: {prediction['fight_style_analysis']['matchup_narrative']}")


def demonstrate_historical_processing():
    """Demonstrate historical data processing"""
    
    print("\n" + "="*60)
    print("HISTORICAL DATA PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    fights_df, fighters_df = create_sample_data()
    
    print(f"Generated {len(fights_df)} sample fights and {len(fighters_df)} fighters")
    
    # Initialize ELO system and processor
    elo_system = MultiDimensionalUFCELO()
    processor = UFCHistoricalProcessor(elo_system)
    
    # Build ELO from historical data
    print("Building ELO system from historical data...")
    elo_system = processor.build_elo_from_history(
        fights_df, fighters_df,
        start_date=datetime(2018, 1, 1)
    )
    
    # Show final rankings
    print("\nFinal ELO Rankings (Top 10):")
    rankings = elo_system.get_rankings(top_n=10)
    for i, fighter in enumerate(rankings, 1):
        print(f"{i:2d}. {fighter['name']:<20} {fighter['rating']:6.1f} "
              f"(±{fighter['uncertainty']:3.0f}, {fighter['fights_count']:2d} fights)")
    
    # Show some predictions
    test_matchups = [
        ('Jon Jones', 'Francis Ngannou'),
        ('Amanda Nunes', 'Valentina Shevchenko'),
        ('Khabib Nurmagomedov', 'Kamaru Usman')
    ]
    
    print("\nSample Predictions:")
    for fighter1, fighter2 in test_matchups:
        try:
            prediction = elo_system.predict_fight_outcome(fighter1, fighter2)
            prob = prediction.get('fighter1_win_prob', 0.5)
            print(f"{fighter1} vs {fighter2}: {prob:.3f} / {1-prob:.3f}")
        except:
            print(f"{fighter1} vs {fighter2}: Not available")


def demonstrate_integration():
    """Demonstrate integration with ML pipeline"""
    
    print("\n" + "="*60)
    print("ELO-ML INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    fights_df, fighters_df = create_sample_data()
    
    # Initialize integration
    elo_integration = ELOIntegration(use_multi_dimensional=True)
    
    # Build ELO system
    print("Building integrated ELO system...")
    elo_integration.build_elo_from_data(fights_df, fighters_df)
    
    # Extract ELO features for sample matchups
    test_matchups = [('Jon Jones', 'Francis Ngannou'), ('Amanda Nunes', 'Valentina Shevchenko')]
    
    print("\nELO Features for Sample Matchups:")
    for fighter1, fighter2 in test_matchups:
        features = elo_integration.extract_elo_features(fighter1, fighter2)
        print(f"\n{fighter1} vs {fighter2}:")
        for feature, value in features.items():
            if isinstance(value, float):
                print(f"  {feature}: {value:.3f}")
            else:
                print(f"  {feature}: {value}")
    
    # Demonstrate enhanced dataset
    sample_fight_data = pd.DataFrame({
        'Fighter': ['Jon Jones', 'Amanda Nunes'],
        'Opponent': ['Francis Ngannou', 'Valentina Shevchenko'],
        'some_ml_feature': [1.5, 2.3]
    })
    
    enhanced_data = elo_integration.enhance_fight_dataset(sample_fight_data)
    print(f"\nEnhanced dataset shape: {enhanced_data.shape}")
    print(f"Added features: {[col for col in enhanced_data.columns if col.startswith('elo_')]}")
    
    # Make hybrid predictions
    print("\nHybrid Predictions:")
    for fighter1, fighter2 in test_matchups:
        prediction = elo_integration.predict_fight_hybrid(fighter1, fighter2)
        elo_prob = prediction['predictions']['elo']['fighter1_win_prob']
        print(f"{fighter1} vs {fighter2}: {elo_prob:.3f} (ELO only)")


def demonstrate_validation():
    """Demonstrate validation framework"""
    
    print("\n" + "="*60)
    print("VALIDATION FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Create larger sample for meaningful validation
    np.random.seed(42)
    fights_df, fighters_df = create_sample_data()
    
    # Double the data for better validation
    more_fights = fights_df.copy()
    more_fights['Date'] = pd.to_datetime(more_fights['Date']) + timedelta(days=365)
    extended_fights = pd.concat([fights_df, more_fights], ignore_index=True)
    
    # Initialize validator
    validator = ELOValidator()
    
    # Run cross-validation (with fewer folds for demo)
    print("Running cross-validation backtesting...")
    cv_results = validator.cross_validation_backtest(
        extended_fights, fighters_df, n_folds=3, min_train_period_days=180
    )
    
    print(f"Cross-validation results:")
    print(f"  Mean accuracy: {cv_results.get('accuracy_mean', 0):.3f} ± {cv_results.get('accuracy_std', 0):.3f}")
    print(f"  Mean Brier score: {cv_results.get('brier_score_mean', 0):.3f} ± {cv_results.get('brier_score_std', 0):.3f}")
    print(f"  Total predictions: {cv_results.get('total_predictions', 0)}")
    
    # Compare configurations
    print("\nComparing ELO configurations...")
    config_comparison = validator.compare_elo_configurations(extended_fights, fighters_df)
    
    print(f"Best configuration: {config_comparison.get('best_configuration', 'N/A')}")
    print(f"Best accuracy: {config_comparison.get('best_accuracy', 0):.3f}")
    
    # Benchmark against baselines
    print("\nBenchmarking against baselines...")
    benchmark_results = validator.benchmark_against_baseline(extended_fights, fighters_df)
    
    elo_acc = benchmark_results['elo_system']['accuracy']
    random_acc = benchmark_results['baselines']['random']['accuracy']
    record_acc = benchmark_results['baselines']['record_based']['accuracy']
    
    print(f"ELO system accuracy: {elo_acc:.3f}")
    print(f"Random baseline: {random_acc:.3f}")
    print(f"Record-based baseline: {record_acc:.3f}")
    print(f"Improvement vs random: {elo_acc - random_acc:.3f}")
    print(f"Improvement vs record-based: {elo_acc - record_acc:.3f}")


def main():
    """Run complete ELO system demonstration"""
    
    print("UFC ELO RATING SYSTEM COMPREHENSIVE DEMO")
    print("========================================")
    
    try:
        # Run all demonstrations
        demonstrate_basic_elo_system()
        demonstrate_multi_dimensional_elo()
        demonstrate_historical_processing()
        demonstrate_integration()
        demonstrate_validation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE - ALL SYSTEMS FUNCTIONAL")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("✓ Basic ELO rating system with UFC-specific adaptations")
        print("✓ Multi-dimensional ratings (striking, grappling, cardio)")
        print("✓ Historical data processing and batch training")
        print("✓ Integration with existing ML pipeline")
        print("✓ Comprehensive validation framework")
        print("✓ Method-specific adjustments and bonuses")
        print("✓ Uncertainty quantification")
        print("✓ Activity decay and champion handling")
        print("✓ Weight class management")
        
        print("\nNext Steps:")
        print("1. Load your actual UFC fight data")
        print("2. Run full historical processing")
        print("3. Validate on out-of-sample data")
        print("4. Integrate with your ML models")
        print("5. Deploy for live predictions")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()