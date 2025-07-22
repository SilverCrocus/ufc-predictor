#!/usr/bin/env python3
"""
UFC Model Comparison Tool
========================

Comprehensive comparison of different UFC prediction approaches including:
- Original Random Forest baseline
- Enhanced XGBoost/LightGBM models
- Feature engineering improvements
- ELO rating integration
- Ensemble methods

This script provides a direct comparison to help you understand which approach
works best for your UFC prediction system.

Usage:
    python compare_models.py                    # Full comparison
    python compare_models.py --quick            # Quick comparison with smaller dataset
    python compare_models.py --baseline-only    # Only test baseline vs XGBoost
    python compare_models.py --save-results     # Save detailed results to file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
from typing import Dict, List, Tuple

# Core ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Import our existing system
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.enhanced_modeling import EnhancedUFCPredictor, ELOSystem
from src.feature_engineering import engineer_features_final
from src.model_training import train_winner_models
from src.unified_config import config
from src.logging_config import get_logger

logger = get_logger(__name__)


class ModelComparisonFramework:
    """
    Framework for comprehensive comparison of different UFC prediction models
    """
    
    def __init__(self, data_path: Path = None, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {}
        self.models = {}
        self.data_loaded = False
        
        # Try to load real data
        self.data_path = data_path or config.paths.corrected_fighters_data
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare UFC fight data
        """
        try:
            # Try to load actual UFC data
            if self.data_path and self.data_path.exists():
                logger.info(f"Loading data from {self.data_path}")
                fighters_df = pd.read_csv(self.data_path)
                
                # Create synthetic fight outcomes for comparison
                # In real usage, this would come from actual fight results
                n_fights = min(2000 if self.quick_mode else 5000, len(fighters_df) // 2)
                
                # Create pairs of fighters for synthetic matchups
                X_features = []
                y_outcomes = []
                
                for i in range(0, min(n_fights * 2, len(fighters_df) - 1), 2):
                    fighter_a = fighters_df.iloc[i]
                    fighter_b = fighters_df.iloc[i + 1]
                    
                    # Create differential features
                    features = self._create_matchup_features(fighter_a, fighter_b)
                    X_features.append(features)
                    
                    # Synthetic outcome based on feature advantages
                    outcome = self._generate_realistic_outcome(features)
                    y_outcomes.append(outcome)
                
                X_df = pd.DataFrame(X_features)
                y_series = pd.Series(y_outcomes)
                
                logger.info(f"Created {len(X_df)} synthetic matchups from real fighter data")
                
            else:
                # Fallback to completely synthetic data
                logger.warning("No real data found, generating synthetic data")
                X_df, y_series = self._generate_synthetic_data()
            
            self.data_loaded = True
            return X_df, y_series
            
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            logger.info("Falling back to synthetic data generation")
            return self._generate_synthetic_data()
    
    def _create_matchup_features(self, fighter_a: pd.Series, fighter_b: pd.Series) -> Dict:
        """Create differential features between two fighters"""
        features = {}
        
        # Physical attribute differences
        if 'Height (inches)' in fighter_a:
            features['height_inches_diff'] = fighter_a['Height (inches)'] - fighter_b['Height (inches)']
        if 'Weight (lbs)' in fighter_a:
            features['weight_lbs_diff'] = fighter_a['Weight (lbs)'] - fighter_b['Weight (lbs)']
        if 'Reach (in)' in fighter_a:
            features['reach_in_diff'] = fighter_a['Reach (in)'] - fighter_b['Reach (in)']
        if 'Age' in fighter_a:
            features['age_diff'] = fighter_a['Age'] - fighter_b['Age']
        
        # Performance differences
        stats = ['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.', 'Wins', 'Losses']
        for stat in stats:
            if stat in fighter_a and not pd.isna(fighter_a[stat]) and not pd.isna(fighter_b[stat]):
                features[f'{stat.lower().replace(" ", "_").replace(".", "")}_diff'] = fighter_a[stat] - fighter_b[stat]
        
        # Fill missing values with 0
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features
    
    def _generate_realistic_outcome(self, features: Dict) -> int:
        """Generate realistic fight outcome based on features"""
        # Weight different advantages
        score = 0.0
        
        # Physical advantages
        score += features.get('height_inches_diff', 0) * 0.1
        score += features.get('reach_in_diff', 0) * 0.15
        score -= features.get('age_diff', 0) * 0.05  # Younger is better
        
        # Performance advantages  
        score += features.get('wins_diff', 0) * 0.3
        score -= features.get('losses_diff', 0) * 0.2
        score += features.get('slpm_diff', 0) * 0.1
        score += features.get('str_acc_diff', 0) * 2.0  # High weight for accuracy
        
        # Add some randomness for realistic upsets
        score += np.random.normal(0, 0.5)
        
        return 1 if score > 0 else 0
    
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic UFC-like data for testing"""
        n_samples = 1000 if self.quick_mode else 3000
        np.random.seed(42)  # Reproducible results
        
        feature_names = [
            'height_inches_diff', 'weight_lbs_diff', 'reach_in_diff', 'age_diff',
            'slpm_diff', 'str_acc_diff', 'sapm_diff', 'str_def_diff',
            'td_avg_diff', 'td_acc_diff', 'td_def_diff', 'sub_avg_diff',
            'wins_diff', 'losses_diff'
        ]
        
        # Generate realistic UFC-style distributions
        X = pd.DataFrame({
            'height_inches_diff': np.random.normal(0, 3, n_samples),
            'weight_lbs_diff': np.random.normal(0, 5, n_samples),
            'reach_in_diff': np.random.normal(0, 4, n_samples),
            'age_diff': np.random.normal(0, 4, n_samples),
            'slpm_diff': np.random.normal(0, 2, n_samples),
            'str_acc_diff': np.random.normal(0, 0.15, n_samples),
            'sapm_diff': np.random.normal(0, 1.5, n_samples),
            'str_def_diff': np.random.normal(0, 0.12, n_samples),
            'td_avg_diff': np.random.normal(0, 1.2, n_samples),
            'td_acc_diff': np.random.normal(0, 0.20, n_samples),
            'td_def_diff': np.random.normal(0, 0.15, n_samples),
            'sub_avg_diff': np.random.normal(0, 0.8, n_samples),
            'wins_diff': np.random.normal(0, 8, n_samples),
            'losses_diff': np.random.normal(0, 5, n_samples)
        })
        
        # Generate realistic outcomes
        y = []
        for idx in range(n_samples):
            features = X.iloc[idx].to_dict()
            outcome = self._generate_realistic_outcome(features)
            y.append(outcome)
        
        y = pd.Series(y)
        
        logger.info(f"Generated {n_samples} synthetic UFC matchups")
        logger.info(f"Outcome distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def test_baseline_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Test the original Random Forest baseline"""
        logger.info("🌲 Testing baseline Random Forest model...")
        
        start_time = time.time()
        
        # Use same parameters as original system
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=config.model.random_state,
            n_jobs=-1
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        training_time = time.time() - start_time
        
        # Get feature importance
        feature_importance = dict(zip(X_train.columns, rf_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        results = {
            'model_name': 'Random Forest (Baseline)',
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'top_features': top_features,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.models['baseline'] = rf_model
        self.results['baseline'] = results
        
        logger.info(f"Baseline RF: Accuracy={accuracy:.4f}, AUC={roc_auc:.4f}, Time={training_time:.2f}s")
        
        return results
    
    def test_enhanced_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Test the enhanced models (XGBoost, LightGBM, Ensemble)"""
        logger.info("🚀 Testing enhanced models...")
        
        # Initialize enhanced predictor
        predictor = EnhancedUFCPredictor(use_elo=False, use_advanced_features=True)
        
        # Train all models
        training_start = time.time()
        predictor.train_all_models(X_train, y_train)
        total_training_time = time.time() - training_start
        
        enhanced_results = {}
        
        # Test each model
        for model_name in ['xgboost', 'lightgbm', 'ensemble']:
            if model_name in predictor.models:
                start_time = time.time()
                
                # Make predictions
                y_pred = predictor.predict(X_test, model_name)
                y_pred_proba = predictor.predict_proba(X_test, model_name)[:, 1]
                
                prediction_time = time.time() - start_time
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Get feature importance if available
                top_features = []
                model = predictor.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                results = {
                    'model_name': model_name.title(),
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'training_time': predictor.performance[model_name].training_time,
                    'prediction_time': prediction_time,
                    'top_features': top_features,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                enhanced_results[model_name] = results
                self.results[model_name] = results
                
                logger.info(f"{model_name}: Accuracy={accuracy:.4f}, AUC={roc_auc:.4f}")
        
        self.models['enhanced_predictor'] = predictor
        
        logger.info(f"Enhanced models total training time: {total_training_time:.2f}s")
        
        return enhanced_results
    
    def run_comparison(self) -> Dict:
        """Run complete model comparison"""
        logger.info("🏁 Starting UFC Model Comparison")
        logger.info("=" * 60)
        
        # Load data
        X, y = self.load_data()
        
        # Split data temporally (not randomly) to avoid look-ahead bias
        split_idx = int(0.75 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Dataset: {len(X)} total samples")
        logger.info(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")
        
        # Test baseline model
        baseline_results = self.test_baseline_model(X_train, X_test, y_train, y_test)
        
        # Test enhanced models
        enhanced_results = self.test_enhanced_models(X_train, X_test, y_train, y_test)
        
        # Generate comparison summary
        self._generate_comparison_summary()
        
        return self.results
    
    def _generate_comparison_summary(self):
        """Generate and print comparison summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL COMPARISON SUMMARY")
        logger.info("=" * 60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': results['model_name'],
                'Accuracy': f"{results['accuracy']:.4f}",
                'ROC AUC': f"{results['roc_auc']:.4f}",
                'Training Time (s)': f"{results['training_time']:.2f}",
                'Improvement vs Baseline': f"{((results['accuracy'] - self.results['baseline']['accuracy']) * 100):+.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Performing Model: {best_model[1]['model_name']}")
        print(f"   Best Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   Improvement over baseline: {((best_model[1]['accuracy'] - self.results['baseline']['accuracy']) * 100):+.2f}%")
        
        # Feature importance comparison
        print(f"\n🔍 TOP PREDICTIVE FEATURES:")
        if 'top_features' in best_model[1] and best_model[1]['top_features']:
            for i, (feature, importance) in enumerate(best_model[1]['top_features'][:5], 1):
                print(f"   {i}. {feature}: {importance:.4f}")
    
    def save_results(self, save_path: Path = None):
        """Save detailed results to file"""
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"model_comparison_results_{timestamp}.json")
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'model_name': results['model_name'],
                'accuracy': float(results['accuracy']),
                'roc_auc': float(results['roc_auc']),
                'training_time': float(results['training_time']),
                'top_features': [(str(feat), float(imp)) for feat, imp in results.get('top_features', [])]
            }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        if not self.results:
            return ["Run comparison first using run_comparison()"]
        
        baseline_accuracy = self.results['baseline']['accuracy']
        
        # Find models that improve over baseline
        improved_models = []
        for model_name, results in self.results.items():
            if model_name != 'baseline' and results['accuracy'] > baseline_accuracy:
                improvement = (results['accuracy'] - baseline_accuracy) * 100
                improved_models.append((model_name, improvement))
        
        if improved_models:
            improved_models.sort(key=lambda x: x[1], reverse=True)
            best_model, best_improvement = improved_models[0]
            
            recommendations.append(f"✅ Switch to {self.results[best_model]['model_name']} for {best_improvement:+.2f}% accuracy improvement")
            
            if best_improvement > 5:
                recommendations.append(f"🚀 Significant improvement detected! {best_model} provides substantial gains")
            elif best_improvement > 2:
                recommendations.append(f"📈 Moderate improvement. Consider implementing {best_model} for better performance")
            else:
                recommendations.append(f"💡 Small but consistent improvement with {best_model}")
        else:
            recommendations.append("⚠️  No models significantly outperform the baseline Random Forest")
            recommendations.append("🔧 Consider more advanced feature engineering or data collection")
        
        # Training time recommendations
        fast_models = [name for name, results in self.results.items() 
                      if results['training_time'] < 10 and results['accuracy'] > baseline_accuracy * 0.98]
        
        if fast_models:
            recommendations.append(f"⚡ For production speed, consider: {', '.join(fast_models)}")
        
        return recommendations


def main():
    """Main entry point for model comparison"""
    parser = argparse.ArgumentParser(description="Compare UFC prediction models")
    parser.add_argument('--quick', action='store_true', help='Quick comparison with smaller dataset')
    parser.add_argument('--baseline-only', action='store_true', help='Compare only baseline vs XGBoost')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results to JSON file')
    parser.add_argument('--data-path', type=Path, help='Path to UFC data file')
    
    args = parser.parse_args()
    
    try:
        # Initialize comparison framework
        framework = ModelComparisonFramework(data_path=args.data_path, quick_mode=args.quick)
        
        # Run comparison
        results = framework.run_comparison()
        
        # Save results if requested
        if args.save_results:
            framework.save_results()
        
        # Generate recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        recommendations = framework.get_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {rec}")
        
        print(f"\n✅ Model comparison completed successfully!")
        
        # Return results for programmatic use
        return results
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()