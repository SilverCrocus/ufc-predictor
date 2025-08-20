#!/usr/bin/env python3
"""
Walk-Forward Validation Runner
Simple command-line interface for running walk-forward validation to address overfitting.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ufc_predictor.pipelines.enhanced_training_pipeline import EnhancedTrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Run Walk-Forward Validation for UFC Prediction Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run walk-forward validation with 3-month retraining
  python run_walk_forward_validation.py --retrain-months 3
  
  # Compare validation methods
  python run_walk_forward_validation.py --mode comparison
  
  # Quick validation run (6-month retraining, no tuning)
  python run_walk_forward_validation.py --retrain-months 6 --no-tune
  
  # Full production training with walk-forward validation
  python run_walk_forward_validation.py --production --tune --optimize
        """
    )
    
    # Validation options
    parser.add_argument('--mode', choices=['walk_forward', 'static_temporal', 'comparison'], 
                       default='walk_forward', 
                       help='Validation mode (default: walk_forward)')
    
    parser.add_argument('--retrain-months', type=int, default=3,
                       help='Retrain frequency in months (default: 3)')
    
    # Training options
    parser.add_argument('--tune', action='store_true', default=True,
                       help='Tune hyperparameters (default: True)')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Create optimized model (default: True)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Skip model optimization')
    
    parser.add_argument('--n-features', type=int, default=32,
                       help='Number of features for optimized model (default: 32)')
    
    # Mode options
    parser.add_argument('--production', action='store_true',
                       help='Production mode: train on ALL data')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory for results')
    
    args = parser.parse_args()
    
    # Handle conflicting arguments
    tune = args.tune and not args.no_tune
    optimize = args.optimize and not args.no_optimize
    
    print("🥊 UFC Walk-Forward Validation Runner")
    print("="*50)
    print(f"Validation mode: {args.mode}")
    print(f"Retrain frequency: {args.retrain_months} months")
    print(f"Hyperparameter tuning: {tune}")
    print(f"Model optimization: {optimize}")
    print(f"Production mode: {args.production}")
    print("="*50)
    
    try:
        # Initialize pipeline
        if args.output_dir:
            base_dir = Path(args.output_dir)
        else:
            base_dir = None
            
        pipeline = EnhancedTrainingPipeline(
            base_dir=base_dir,
            validation_mode=args.mode,
            production_mode=args.production
        )
        
        # Run enhanced pipeline
        results = pipeline.run_enhanced_pipeline(
            tune=tune,
            optimize=optimize,
            n_features=args.n_features,
            retrain_frequency_months=args.retrain_months
        )
        
        print("\n✅ Walk-forward validation completed successfully!")
        
        # Print key results
        if 'walk_forward_validation' in results:
            wf = results['walk_forward_validation']
            print(f"\n📊 Key Results:")
            print(f"  Mean test accuracy: {wf['mean_test_accuracy']:.4f} ± {wf['std_test_accuracy']:.4f}")
            print(f"  Mean overfitting gap: {wf['mean_overfitting_gap']:.4f}")
            print(f"  Model stability score: {wf['model_stability_score']:.4f}")
            print(f"  Total model retrains: {wf['total_retrains']}")
            
            if wf['mean_overfitting_gap'] < 0.10:
                print(f"  ✅ Overfitting is under control!")
            else:
                print(f"  ⚠️ High overfitting detected - consider model adjustments")
        
        if 'validation_comparison' in results:
            comp = results['validation_comparison']
            print(f"\n📈 Validation Comparison:")
            print(f"  Static split overfitting: {comp['static_overfitting_gap']:.4f}")
            print(f"  Walk-forward overfitting: {comp['walk_forward_overfitting_gap']:.4f}")
            print(f"  Improvement: {comp['overfitting_improvement']:.4f} ({comp['overfitting_improvement_pct']:.1f}%)")
            print(f"  Recommended method: {comp['recommended_method']}")
        
        # Print file locations
        print(f"\n📁 Results saved to:")
        print(f"  Training directory: {results['training_dir']}")
        if 'walk_forward_validation' in results:
            print(f"  Validation report: {results['walk_forward_validation']['validation_report_path']}")
        
        if optimize and 'optimized' in results:
            print(f"  Optimized model: {results['optimized']['location']}")
            print(f"  ⚡ {results['optimized']['speed_gain']} faster inference!")
        
    except Exception as e:
        print(f"❌ Error running walk-forward validation: {e}")
        raise


if __name__ == '__main__':
    main()