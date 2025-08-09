#!/usr/bin/env python3
"""
Evaluate the trained UFC models using the enhanced pipeline infrastructure.
This combines the trained models with enhanced evaluation metrics.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
import logging
import sys

# Add src to path
sys.path.insert(0, 'src')

# Import enhanced components
from ufc_predictor.evaluation.temporal_split import TemporalWalkForwardSplitter
from ufc_predictor.evaluation.calibration import UFCProbabilityCalibrator
from ufc_predictor.evaluation.metrics import UFCMetricsCalculator
from ufc_predictor.betting.staking import KellyStaking
from ufc_predictor.features.context_features import ContextFeatureGenerator
from ufc_predictor.features.matchup_features import MatchupFeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_latest_model():
    """Load the latest trained model and metadata."""
    model_dir = Path('model')
    
    # Find latest training directory
    training_dirs = [d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        # Try direct model files
        rf_model_path = model_dir / 'rf_tuned_model.pkl'
        if rf_model_path.exists():
            logger.info(f"Loading model from: {rf_model_path}")
            return joblib.load(rf_model_path), None
        raise FileNotFoundError("No trained models found")
    
    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest training: {latest_dir.name}")
    
    # Load metadata
    metadata_files = list(latest_dir.glob('training_metadata_*.json'))
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        if 'winner_models' in metadata:
            winner_acc = metadata['winner_models'].get('random_forest_tuned', {}).get('test_score', 0)
            logger.info(f"  Winner accuracy: {winner_acc:.2%}")
    else:
        metadata = None
    
    # Load model - look for joblib files
    model_files = list(latest_dir.glob('ufc_winner_model_*.joblib'))
    if not model_files:
        model_files = list(latest_dir.glob('random_forest_tuned_*.pkl'))
    if not model_files:
        model_files = list(latest_dir.glob('rf_tuned_*.pkl'))
    
    if model_files:
        model = joblib.load(model_files[0])
        logger.info(f"  Loaded model from: {model_files[0].name}")
        return model, metadata
    
    raise FileNotFoundError(f"No model found in {latest_dir}")

def load_fight_data():
    """Load the latest fight data with features."""
    # First try to load the engineered dataset from training
    model_dir = Path('model')
    training_dirs = [d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if training_dirs:
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        # Look for the engineered dataset
        dataset_files = list(latest_dir.glob('ufc_fight_dataset_with_diffs_*.csv'))
        if dataset_files:
            logger.info(f"Loading engineered dataset from: {dataset_files[0].name}")
            return pd.read_csv(dataset_files[0])
    
    # Try to find the latest processed data
    data_dir = Path('data')
    
    # Look for scrape directories
    scrape_dirs = [d for d in data_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('scrape_')]
    
    if scrape_dirs:
        latest_scrape = max(scrape_dirs, key=lambda x: x.stat().st_mtime)
        
        # Look for fights file
        fights_files = list(latest_scrape.glob('ufc_fights_*.csv'))
        fighters_files = list(latest_scrape.glob('ufc_fighters_engineered_*.csv'))
        
        if fights_files:
            logger.info(f"Loading fights from: {fights_files[0]}")
            fights_df = pd.read_csv(fights_files[0])
            
            # If we have engineered fighters, merge them
            if fighters_files:
                logger.info(f"Loading fighter features from: {fighters_files[0]}")
                fighters_df = pd.read_csv(fighters_files[0])
                # Would need to merge here based on fighter URLs
            
            return fights_df
    
    # Fallback to any CSV in data/
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        logger.info(f"Loading data from: {csv_files[0]}")
        return pd.read_csv(csv_files[0])
    
    raise FileNotFoundError("No fight data found")

def prepare_features(df):
    """Prepare features for model evaluation."""
    # Try to load the exact features the model was trained on
    model_dir = Path('model')
    training_dirs = [d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if training_dirs:
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        column_files = list(latest_dir.glob('*_columns.json'))
        if column_files:
            with open(column_files[0], 'r') as f:
                model_cols = json.load(f)
            logger.info(f"Using {len(model_cols)} features from model training")
            # Only use columns that exist in the dataframe
            available_cols = [col for col in model_cols if col in df.columns]
            logger.info(f"Found {len(available_cols)} of {len(model_cols)} features in dataset")
            return available_cols
    
    # Fallback: Get feature columns (exclude metadata)
    exclude_cols = {'Outcome', 'Winner', 'Loser', 'Fighter', 'Opponent', 
                   'Event', 'Method', 'Time', 'Date',  # Note: 'Round' is kept as a feature
                   'fighter_url', 'opponent_url', 'winner_url', 'loser_url'}
    
    # Find numeric columns
    numeric_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                numeric_cols.append(col)
    
    logger.info(f"Found {len(numeric_cols)} numeric features")
    return numeric_cols

def evaluate_with_enhanced_metrics(model, X_test, y_test, fight_dates=None):
    """Evaluate model with enhanced metrics including calibration."""
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Calibration metrics
    calibrator = UFCProbabilityCalibrator(method='isotonic')
    cal_df = pd.DataFrame({'prob_pred': y_pred_proba, 'winner': y_test})
    
    # Fit calibration
    calibrator.fit_isotonic_by_segment(cal_df)
    calibrated_probs = calibrator.apply_calibration(cal_df[['prob_pred']])
    
    # Calculate ECE manually
    def calculate_ece(y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    ece = calculate_ece(y_test, calibrated_probs)
    
    # Betting metrics (simplified)
    kelly = KellyStaking(kelly_fraction=0.25)
    
    # Simulate some betting scenarios
    bankroll = 1000
    total_profit = 0
    n_bets = 0
    
    for prob in y_pred_proba[y_pred_proba > 0.55]:  # Only bet on high confidence
        # Assume fair odds for simulation
        implied_odds = 1 / prob
        stake_rec = kelly.calculate_kelly_stake(prob, implied_odds * 1.05, bankroll)
        
        if stake_rec.bet_amount > 0:
            n_bets += 1
            # Simulate win/loss (simplified)
            if np.random.random() < prob:
                total_profit += stake_rec.bet_amount * (implied_odds - 1)
            else:
                total_profit -= stake_rec.bet_amount
    
    roi = (total_profit / bankroll) * 100 if n_bets > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'brier_score': brier,
        'ece_original': calculate_ece(y_test, y_pred_proba),
        'ece_calibrated': ece,
        'n_samples': len(y_test),
        'n_bets_placed': n_bets,
        'simulated_roi': roi,
        'avg_confidence': np.mean(y_pred_proba),
        'confidence_std': np.std(y_pred_proba),
        'calibration_improvement': (calculate_ece(y_test, y_pred_proba) - ece) / (calculate_ece(y_test, y_pred_proba) + 0.0001) * 100
    }
    
    return results

def run_temporal_evaluation(model, df, feature_cols):
    """Run temporal walk-forward evaluation."""
    
    # Initialize temporal splitter
    splitter = TemporalWalkForwardSplitter(
        date_col='Date' if 'Date' in df.columns else 'date',
        train_years=3,
        test_months=3,
        gap_days=14
    )
    
    # Prepare data
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    elif 'date' not in df.columns:
        # Create synthetic dates if missing
        df['date'] = pd.date_range(end='2024-12-31', periods=len(df), freq='D')
    
    # Get target
    if 'Outcome' in df.columns:
        outcome_map = {'W': 1, 'win': 1, 'L': 0, 'loss': 0}
        df['winner'] = df['Outcome'].map(outcome_map)
    
    # Filter valid data
    valid_mask = df['winner'].notna() & (df['winner'] != 0.5)  # Remove draws
    df_clean = df[valid_mask].copy()
    
    # Create folds
    try:
        folds = splitter.make_rolling_folds(df_clean, min_train_samples=100, min_test_samples=20)
        logger.info(f"Created {len(folds)} temporal folds")
    except Exception as e:
        logger.warning(f"Could not create temporal folds: {e}")
        return None
    
    # Evaluate each fold
    fold_results = []
    for i, fold in enumerate(folds[:5]):  # Limit to 5 folds for speed
        train_df = df_clean.iloc[fold.train_indices]
        test_df = df_clean.iloc[fold.test_indices]
        
        # Get features and target
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['winner'].astype(int)
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['winner'].astype(int)
        
        # Evaluate
        results = evaluate_with_enhanced_metrics(model, X_test, y_test)
        results['fold_id'] = i
        results['train_size'] = len(train_df)
        results['test_size'] = len(test_df)
        fold_results.append(results)
        
        logger.info(f"  Fold {i+1}: Accuracy={results['accuracy']:.2%}, ECE={results['ece_calibrated']:.3f}")
    
    return fold_results

def main():
    """Main evaluation function."""
    print("\n" + "="*70)
    print("ENHANCED EVALUATION OF TRAINED UFC MODELS")
    print("="*70)
    
    # Load model
    print("\n1. Loading trained model...")
    try:
        model, metadata = load_latest_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load data
    print("\n2. Loading fight data...")
    try:
        df = load_fight_data()
        print(f"✓ Loaded {len(df)} fights")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Prepare features
    print("\n3. Preparing features...")
    feature_cols = prepare_features(df)
    
    if len(feature_cols) < 10:
        print("✗ Too few features found. Need properly engineered dataset.")
        return
    
    # Simple evaluation on recent data
    print("\n4. Evaluating on recent fights...")
    
    # Use last 20% as test
    test_size = int(0.2 * len(df))
    test_df = df.tail(test_size)
    
    # Prepare test data
    X_test = test_df[feature_cols].fillna(0)
    
    # Get target
    if 'Outcome' in test_df.columns:
        outcome_map = {'W': 1, 'win': 1, 'L': 0, 'loss': 0}
        y_test = test_df['Outcome'].map(outcome_map).fillna(0).astype(int)
    else:
        y_test = test_df.get('winner', pd.Series([0]*len(test_df))).astype(int)
    
    # Evaluate
    results = evaluate_with_enhanced_metrics(model, X_test, y_test)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    print(f"   AUC Score: {results['auc']:.4f}")
    print(f"   Brier Score: {results['brier_score']:.4f}")
    
    print("\n📈 CALIBRATION METRICS:")
    print(f"   ECE (Original): {results['ece_original']:.4f}")
    print(f"   ECE (Calibrated): {results['ece_calibrated']:.4f}")
    print(f"   Calibration Improvement: {results['calibration_improvement']:.1f}%")
    
    print("\n💰 BETTING SIMULATION:")
    print(f"   Bets Placed: {results['n_bets_placed']} (high confidence only)")
    print(f"   Simulated ROI: {results['simulated_roi']:.1f}%")
    
    print("\n📉 CONFIDENCE DISTRIBUTION:")
    print(f"   Average Confidence: {results['avg_confidence']:.3f}")
    print(f"   Confidence Std Dev: {results['confidence_std']:.3f}")
    
    # Try temporal evaluation
    print("\n5. Running temporal walk-forward evaluation...")
    fold_results = run_temporal_evaluation(model, df, feature_cols)
    
    if fold_results:
        print("\n📅 TEMPORAL VALIDATION RESULTS:")
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_ece = np.mean([r['ece_calibrated'] for r in fold_results])
        print(f"   Average Accuracy: {avg_acc:.2%}")
        print(f"   Average ECE: {avg_ece:.4f}")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    
    # Summary
    print("\n🎯 KEY FINDINGS:")
    if results['accuracy'] > 0.70:
        print("   ✓ Model performance is EXCELLENT (>70%)")
    
    if results['ece_calibrated'] < 0.05:
        print("   ✓ Calibration is GOOD (ECE < 0.05)")
    elif results['ece_calibrated'] < 0.10:
        print("   ⚠ Calibration needs improvement (ECE > 0.05)")
    
    if results['simulated_roi'] > 0:
        print("   ✓ Positive ROI in simulation")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Focus on probability calibration")
    print("   2. Implement segment-specific calibrators")
    print("   3. Track CLV on real bets")
    print("   4. Use pessimistic Kelly staking")

if __name__ == "__main__":
    main()