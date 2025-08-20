#!/usr/bin/env python3
"""
Basic test script to demonstrate walk-forward validation functionality
This doesn't require all dependencies and shows the core concept.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'src')

def create_synthetic_ufc_data(n_samples=1000):
    """Create synthetic UFC fight data for testing"""
    np.random.seed(42)
    
    # Create dates spanning 5 years
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=i*3) for i in range(n_samples)]
    
    # Create features (simulating fighter stats)
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Create labels with some temporal pattern (newer fights slightly different)
    base_pattern = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    temporal_drift = np.linspace(0, 1, n_samples) * 0.5  # Simulate temporal drift
    noise = np.random.randn(n_samples) * 0.5
    
    y_continuous = base_pattern + temporal_drift + noise
    y = (y_continuous > y_continuous.mean()).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['winner'] = y
    df['date'] = dates
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def demonstrate_walk_forward_concept():
    """Demonstrate the concept of walk-forward validation"""
    
    print("=" * 80)
    print("WALK-FORWARD VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic UFC fight data...")
    df = create_synthetic_ufc_data(n_samples=1000)
    print(f"   ✓ Created {len(df)} synthetic fights from {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Set up walk-forward parameters
    initial_train_months = 24  # 2 years
    test_window_months = 3     # 3 months
    retrain_frequency_months = 6  # Retrain every 6 months
    
    print(f"\n2. Walk-Forward Configuration:")
    print(f"   - Initial training period: {initial_train_months} months")
    print(f"   - Test window: {test_window_months} months")
    print(f"   - Retrain frequency: {retrain_frequency_months} months")
    
    # Simulate walk-forward validation
    print("\n3. Walk-Forward Validation Process:")
    print("-" * 60)
    
    start_date = df['date'].min()
    initial_train_end = start_date + pd.DateOffset(months=initial_train_months)
    
    current_train_end = initial_train_end
    fold_num = 1
    results = []
    
    while current_train_end < df['date'].max() - pd.DateOffset(months=test_window_months):
        test_start = current_train_end
        test_end = test_start + pd.DateOffset(months=test_window_months)
        
        # Get training and test data
        train_data = df[df['date'] < current_train_end]
        test_data = df[(df['date'] >= test_start) & (df['date'] < test_end)]
        
        if len(test_data) == 0:
            break
            
        # Simulate model training and evaluation
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Train model
        X_train = train_data[[col for col in train_data.columns if col.startswith('feature_')]]
        y_train = train_data['winner']
        
        X_test = test_data[[col for col in test_data.columns if col.startswith('feature_')]]
        y_test = test_data['winner']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        overfitting_gap = train_acc - test_acc
        
        print(f"\n   Fold {fold_num}:")
        print(f"   Training: {df['date'].min().date()} to {current_train_end.date()}")
        print(f"   Testing:  {test_start.date()} to {test_end.date()}")
        print(f"   Training samples: {len(train_data):4d} | Test samples: {len(test_data):3d}")
        print(f"   Train Accuracy: {train_acc:.3f} | Test Accuracy: {test_acc:.3f}")
        print(f"   Overfitting Gap: {overfitting_gap:+.3f}")
        
        results.append({
            'fold': fold_num,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting_gap': overfitting_gap,
            'train_size': len(train_data),
            'test_size': len(test_data)
        })
        
        # Decide if we need to retrain
        if fold_num % (retrain_frequency_months // test_window_months) == 0:
            print(f"   >>> RETRAINING MODEL (scheduled retrain)")
        
        # Move forward
        current_train_end = test_end
        fold_num += 1
        
        if fold_num > 5:  # Limit to 5 folds for demonstration
            print(f"\n   ... (limiting to 5 folds for demonstration)")
            break
    
    # Summary statistics
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)
    
    avg_train_acc = results_df['train_acc'].mean()
    avg_test_acc = results_df['test_acc'].mean()
    avg_overfitting = results_df['overfitting_gap'].mean()
    
    print(f"\nAverage Performance Across {len(results_df)} Folds:")
    print(f"  • Average Training Accuracy:  {avg_train_acc:.3f}")
    print(f"  • Average Test Accuracy:      {avg_test_acc:.3f}")
    print(f"  • Average Overfitting Gap:    {avg_overfitting:+.3f}")
    
    # Compare with static split
    print("\n" + "-" * 60)
    print("COMPARISON: Static 80/20 Split")
    print("-" * 60)
    
    # Static split
    split_idx = int(len(df) * 0.8)
    train_static = df.iloc[:split_idx]
    test_static = df.iloc[split_idx:]
    
    X_train_static = train_static[[col for col in train_static.columns if col.startswith('feature_')]]
    y_train_static = train_static['winner']
    X_test_static = test_static[[col for col in test_static.columns if col.startswith('feature_')]]
    y_test_static = test_static['winner']
    
    model_static = RandomForestClassifier(n_estimators=10, random_state=42)
    model_static.fit(X_train_static, y_train_static)
    
    static_train_acc = accuracy_score(y_train_static, model_static.predict(X_train_static))
    static_test_acc = accuracy_score(y_test_static, model_static.predict(X_test_static))
    static_overfitting = static_train_acc - static_test_acc
    
    print(f"\nStatic Split Performance:")
    print(f"  • Training Accuracy:  {static_train_acc:.3f}")
    print(f"  • Test Accuracy:      {static_test_acc:.3f}")
    print(f"  • Overfitting Gap:    {static_overfitting:+.3f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    print(f"""
Walk-Forward Validation Benefits:
  
  1. MORE REALISTIC: Tests model on truly future data (no look-ahead bias)
  
  2. OVERFITTING DETECTION: Average gap of {avg_overfitting:+.3f} vs {static_overfitting:+.3f} (static)
     → Walk-forward often shows MORE realistic (worse) performance
  
  3. TEMPORAL STABILITY: Shows if model performance degrades over time
     → First fold test acc: {results_df.iloc[0]['test_acc']:.3f}
     → Last fold test acc:  {results_df.iloc[-1]['test_acc']:.3f}
  
  4. RETRAINING BENEFITS: Periodic retraining helps maintain performance
  
  5. PRODUCTION READY: Mimics real-world deployment scenarios
""")
    
    print("\n" + "=" * 60)
    print("✓ Walk-Forward Validation Demonstration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_walk_forward_concept()