"""
Enhanced Prediction Integration Example
======================================

This file demonstrates how the enhanced ML pipeline integrates with the existing
UFC prediction system while maintaining backward compatibility.

Key Integration Points:
1. Extended model training pipeline
2. Enhanced prediction service
3. Backward compatible API
4. Ensemble model management
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Existing imports
from ufc_predictor.core.prediction import predict_fight_symmetrical
from ufc_predictor.agent.services.prediction_service import UFCPredictionService, PredictionResult
from ufc_predictor.models.model_training import UFCModelTrainer

# Enhanced imports (new components)
try:
    from ufc_predictor.advanced_ensemble_methods import AdvancedEnsembleSystem
    from ufc_predictor.advanced_feature_engineering import AdvancedFeatureEngineer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logging.warning("Enhanced features not available, using baseline system")

logger = logging.getLogger(__name__)


class EnhancedModelTrainer(UFCModelTrainer):
    """
    Extended model trainer that creates ensemble models while maintaining 
    compatibility with existing training pipeline
    """
    
    def __init__(self, random_state: int = 42, enable_ensemble: bool = True):
        super().__init__(random_state)
        self.enable_ensemble = enable_ensemble and ENHANCED_FEATURES_AVAILABLE
        self.ensemble_models = {}
        self.feature_engineer = None
        
        if self.enable_ensemble:
            self.feature_engineer = AdvancedFeatureEngineer()
    
    def train_enhanced_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                              fighter_pairs: Optional[List[Tuple[str, str]]] = None,
                              tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Enhanced training pipeline that creates both baseline and ensemble models
        
        Args:
            X: Base feature DataFrame (existing 64 features)
            y: Target Series
            fighter_pairs: Optional fighter pairs for ELO features
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary containing all trained models and metadata
        """
        logger.info("Starting enhanced training pipeline...")
        
        # Step 1: Train baseline models (existing system)
        baseline_results = self._train_baseline_models(X, y, tune_hyperparameters)
        
        # Step 2: Create enhanced features if enabled
        if self.enable_ensemble:
            enhanced_X = self._create_enhanced_features(X, fighter_pairs)
            ensemble_results = self._train_ensemble_models(enhanced_X, y)
            
            # Combine results
            training_results = {
                **baseline_results,
                **ensemble_results,
                'enhanced_features_enabled': True,
                'total_features': len(enhanced_X.columns),
                'baseline_features': len(X.columns),
                'new_features': len(enhanced_X.columns) - len(X.columns)
            }
        else:
            training_results = {
                **baseline_results,
                'enhanced_features_enabled': False,
                'total_features': len(X.columns),
                'baseline_features': len(X.columns),
                'new_features': 0
            }
        
        logger.info(f"Enhanced training completed. Features: {training_results['total_features']}")
        return training_results
    
    def _train_baseline_models(self, X: pd.DataFrame, y: pd.Series, 
                             tune_hyperparameters: bool) -> Dict[str, Any]:
        """Train baseline Random Forest models (existing system)"""
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train baseline Random Forest
        if tune_hyperparameters:
            rf_model = self.tune_random_forest(X_train, y_train)
            model_name = 'random_forest_tuned'
        else:
            rf_model = self.train_random_forest(X_train, y_train)
            model_name = 'random_forest'
        
        # Evaluate baseline model
        baseline_results = self.evaluate_model(model_name, X_test, y_test, show_plots=False)
        
        return {
            'baseline_model': rf_model,
            'baseline_accuracy': baseline_results['accuracy'],
            'baseline_features': X.columns.tolist(),
            'test_data': (X_test, y_test)
        }
    
    def _create_enhanced_features(self, X: pd.DataFrame, 
                                fighter_pairs: Optional[List[Tuple[str, str]]]) -> pd.DataFrame:
        """Create enhanced features using advanced feature engineering"""
        
        if not self.feature_engineer:
            return X
            
        logger.info("Creating enhanced features...")
        enhanced_X = self.feature_engineer.create_all_features(
            X, fighter_pairs=fighter_pairs
        )
        
        logger.info(f"Enhanced features: {len(X.columns)} → {len(enhanced_X.columns)} "
                   f"(+{len(enhanced_X.columns) - len(X.columns)} new)")
        
        return enhanced_X
    
    def _train_ensemble_models(self, enhanced_X: pd.DataFrame, 
                             y: pd.Series) -> Dict[str, Any]:
        """Train ensemble models on enhanced features"""
        
        # Split enhanced data
        X_train, X_test, y_train, y_test = self.split_data(enhanced_X, y)
        
        # Initialize ensemble models
        ensemble_models = {}
        
        # 1. XGBoost Model
        try:
            import xgboost as xgb
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            ensemble_models['xgboost'] = xgb_model
            logger.info("XGBoost model trained successfully")
            
        except ImportError:
            logger.warning("XGBoost not available, skipping")
        
        # 2. Neural Network Model (simplified for example)
        try:
            from sklearn.neural_network import MLPClassifier
            
            nn_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
            
            nn_model.fit(X_train, y_train)
            ensemble_models['neural_network'] = nn_model
            logger.info("Neural Network model trained successfully")
            
        except Exception as e:
            logger.warning(f"Neural Network training failed: {e}")
        
        # 3. Create Ensemble System
        if len(ensemble_models) > 0:
            # Add baseline Random Forest to ensemble
            ensemble_models['random_forest'] = self.models.get('random_forest_tuned') or self.models.get('random_forest')
            
            ensemble_system = AdvancedEnsembleSystem()
            ensemble_system.add_models(ensemble_models)
            
            # Fit ensemble methods that require training
            ensemble_system.fit_ensembles(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_predictions = ensemble_system.predict_with_confidence(X_test)
            ensemble_accuracy = np.mean([
                p.predicted_class for p in ensemble_predictions
            ] == y_test.values)
            
            return {
                'ensemble_system': ensemble_system,
                'ensemble_models': ensemble_models,
                'ensemble_accuracy': ensemble_accuracy,
                'enhanced_features': enhanced_X.columns.tolist()
            }
        else:
            logger.warning("No ensemble models available")
            return {}


class EnhancedPredictionService(UFCPredictionService):
    """
    Enhanced prediction service that uses ensemble methods when available
    while maintaining complete backward compatibility
    """
    
    def __init__(self, betting_system: Dict[str, Any], ensemble_system=None):
        super().__init__(betting_system)
        self.ensemble_system = ensemble_system
        self.enhanced_features_available = ENHANCED_FEATURES_AVAILABLE and ensemble_system is not None
        
        if self.enhanced_features_available:
            self.feature_engineer = AdvancedFeatureEngineer()
            logger.info("Enhanced prediction service initialized with ensemble")
        else:
            logger.info("Enhanced prediction service using baseline models")
    
    def predict_event(self, odds_data: Dict[str, Dict], event_name: str, 
                     fight_contexts: Optional[Dict[str, Dict]] = None) -> 'PredictionAnalysis':
        """
        Enhanced event prediction with optional contextual features
        
        Args:
            odds_data: Dictionary of fight odds data (existing format)
            event_name: Name of the UFC event
            fight_contexts: Optional contextual information for enhanced features
            
        Returns:
            PredictionAnalysis with enhanced predictions if available
        """
        logger.info(f"Starting {'enhanced' if self.enhanced_features_available else 'baseline'} "
                   f"prediction analysis for {event_name}")
        
        if self.enhanced_features_available and fight_contexts:
            return self._predict_event_enhanced(odds_data, event_name, fight_contexts)
        else:
            # Fallback to original implementation
            return super().predict_event(odds_data, event_name)
    
    def _predict_event_enhanced(self, odds_data: Dict[str, Dict], event_name: str,
                              fight_contexts: Dict[str, Dict]) -> 'PredictionAnalysis':
        """Enhanced event prediction using ensemble methods"""
        
        from ufc_predictor.agent.services.prediction_service import PredictionAnalysis
        
        analysis = PredictionAnalysis(event_name)
        
        for fight_key, fight_odds in odds_data.items():
            fighter_a = fight_odds['fighter_a']
            fighter_b = fight_odds['fighter_b']
            fight_context = fight_contexts.get(fight_key, {})
            
            try:
                # Get enhanced prediction
                prediction_result = self._predict_single_fight_enhanced(
                    fight_key, fighter_a, fighter_b, fight_odds, fight_context
                )
                
                if prediction_result:
                    analysis.fight_predictions.append(prediction_result)
                    analysis.summary['successful_predictions'] += 1
                    
                    # Update opportunity counters (enhanced logic)
                    if prediction_result.is_upset_opportunity:
                        analysis.summary['upset_opportunities'] += 1
                    if prediction_result.is_high_confidence:
                        analysis.summary['high_confidence_picks'] += 1
                else:
                    analysis.summary['failed_predictions'] += 1
                    
            except Exception as e:
                logger.error(f"Enhanced prediction failed for {fighter_a} vs {fighter_b}: {e}")
                analysis.summary['failed_predictions'] += 1
        
        analysis.summary['total_fights'] = len(odds_data)
        return analysis
    
    def _predict_single_fight_enhanced(self, fight_key: str, fighter_a: str, fighter_b: str,
                                     odds_data: Dict, fight_context: Dict) -> Optional[PredictionResult]:
        """Enhanced single fight prediction with ensemble and contextual features"""
        
        try:
            # Get baseline prediction first
            baseline_prediction = self.betting_system['predict_function'](
                fighter_a, fighter_b,
                self.betting_system['fighters_df'],
                self.betting_system['winner_cols'],
                self.betting_system['method_cols'],
                self.betting_system['winner_model'],
                self.betting_system['method_model']
            )
            
            if 'error' in baseline_prediction:
                return None
            
            # Create enhanced features
            base_features = self._extract_base_features(fighter_a, fighter_b)
            enhanced_features = self.feature_engineer.create_all_features(
                pd.DataFrame([base_features]),
                fighter_pairs=[(fighter_a, fighter_b)]
            )
            
            # Add contextual features
            if fight_context:
                contextual_features = self._create_contextual_features(fight_context)
                for key, value in contextual_features.items():
                    enhanced_features[key] = value
            
            # Get ensemble prediction
            ensemble_predictions = self.ensemble_system.predict_with_confidence(
                enhanced_features,
                fighter_pairs=[(fighter_a, fighter_b)]
            )
            
            if ensemble_predictions:
                ensemble_pred = ensemble_predictions[0]
                
                # Create enhanced prediction result
                result = PredictionResult(fight_key, fighter_a, fighter_b)
                
                # Use ensemble prediction as primary
                result.model_prediction_a = 1 - ensemble_pred.prediction_probability
                result.model_prediction_b = ensemble_pred.prediction_probability
                
                # Enhanced confidence calculation
                result.confidence_score = ensemble_pred.confidence
                
                # Ensemble-specific analysis
                result.is_upset_opportunity = self._is_upset_opportunity_enhanced(
                    result, ensemble_pred
                )
                result.is_high_confidence = self._is_high_confidence_enhanced(
                    result, ensemble_pred
                )
                
                # Add ensemble metadata
                result.raw_prediction = {
                    **baseline_prediction,
                    'ensemble_probability': ensemble_pred.prediction_probability,
                    'ensemble_confidence': ensemble_pred.confidence,
                    'individual_predictions': ensemble_pred.individual_predictions,
                    'prediction_method': 'enhanced_ensemble'
                }
                
                # Market analysis (existing logic)
                self._add_market_analysis(result, odds_data)
                
                return result
            else:
                # Fallback to baseline if ensemble fails
                return self._create_baseline_result(fight_key, fighter_a, fighter_b, 
                                                  baseline_prediction, odds_data)
                
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return None
    
    def _extract_base_features(self, fighter_a: str, fighter_b: str) -> Dict[str, float]:
        """Extract base differential features for enhanced feature creation"""
        
        # Get fighter stats
        fighters_df = self.betting_system['fighters_df']
        
        fighter_a_stats = fighters_df[fighters_df['Name'] == fighter_a].iloc[0]
        fighter_b_stats = fighters_df[fighters_df['Name'] == fighter_b].iloc[0]
        
        # Create differential features (simplified for example)
        base_features = {}
        
        numerical_columns = ['Height (inches)', 'Weight (lbs)', 'Reach (in)', 'Age',
                           'SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 
                           'TD Acc.', 'TD Def.', 'Sub. Avg.', 'Wins', 'Losses']
        
        for col in numerical_columns:
            if col in fighter_a_stats and col in fighter_b_stats:
                diff_name = col.lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '') + '_diff'
                base_features[diff_name] = fighter_a_stats[col] - fighter_b_stats[col]
        
        return base_features
    
    def _create_contextual_features(self, fight_context: Dict) -> Dict[str, float]:
        """Create contextual features from fight context"""
        
        contextual_features = {}
        
        # Venue effects
        venue = fight_context.get('venue', 'las_vegas')
        if 'international' in venue.lower():
            contextual_features['international_venue'] = 1.0
        else:
            contextual_features['international_venue'] = 0.0
        
        # Title fight pressure
        contextual_features['title_fight'] = float(fight_context.get('is_title_fight', False))
        
        # Main event pressure
        contextual_features['main_event'] = float(fight_context.get('is_main_event', False))
        
        # Altitude factor
        altitude = fight_context.get('altitude', 0)
        contextual_features['altitude_factor'] = min(altitude / 5000.0, 1.0)  # Normalize
        
        return contextual_features
    
    def _is_upset_opportunity_enhanced(self, result: PredictionResult, 
                                     ensemble_pred) -> bool:
        """Enhanced upset opportunity detection using ensemble confidence"""
        
        # Use ensemble confidence in addition to probability difference
        prob_diff = abs(result.model_prediction_a - result.model_prediction_b)
        
        return (
            prob_diff > 0.15 and  # Significant probability difference
            ensemble_pred.confidence > 0.7 and  # High ensemble confidence
            max(result.expected_value_a, result.expected_value_b) > 0.08  # Good expected value
        )
    
    def _is_high_confidence_enhanced(self, result: PredictionResult,
                                   ensemble_pred) -> bool:
        """Enhanced high confidence detection using ensemble agreement"""
        
        return (
            ensemble_pred.confidence > 0.8 and  # High ensemble agreement
            max(result.model_prediction_a, result.model_prediction_b) > 0.65 and  # Strong prediction
            max(result.expected_value_a, result.expected_value_b) > 0.05  # Decent expected value
        )


def create_enhanced_betting_system(model_dir: str, enable_ensemble: bool = True) -> Dict[str, Any]:
    """
    Create enhanced betting system with ensemble capabilities
    
    Args:
        model_dir: Directory containing trained models
        enable_ensemble: Whether to load ensemble models
        
    Returns:
        Enhanced betting system dictionary
    """
    
    # Load baseline system (existing)
    baseline_system = _load_baseline_system(model_dir)
    
    if enable_ensemble and ENHANCED_FEATURES_AVAILABLE:
        # Try to load ensemble system
        ensemble_system = _load_ensemble_system(model_dir)
        
        if ensemble_system:
            baseline_system['ensemble_system'] = ensemble_system
            baseline_system['prediction_service_class'] = EnhancedPredictionService
            logger.info("Enhanced betting system created with ensemble")
        else:
            logger.info("Ensemble models not found, using baseline system")
    
    return baseline_system


def _load_baseline_system(model_dir: str) -> Dict[str, Any]:
    """Load baseline betting system (existing functionality)"""
    
    model_path = Path(model_dir)
    
    # Load fighters data
    fighters_df = pd.read_csv(model_path / 'ufc_fighters_engineered_corrected.csv')
    
    # Load models
    winner_model = joblib.load(model_path / 'ufc_random_forest_model_tuned.joblib')
    method_model = joblib.load(model_path / 'ufc_multiclass_model.joblib')
    
    # Load feature columns
    with open(model_path / 'winner_model_columns.json', 'r') as f:
        winner_cols = json.load(f)
    
    with open(model_path / 'method_model_columns.json', 'r') as f:
        method_cols = json.load(f)
    
    return {
        'fighters_df': fighters_df,
        'winner_model': winner_model,
        'method_model': method_model,
        'winner_cols': winner_cols,
        'method_cols': method_cols,
        'predict_function': predict_fight_symmetrical,
        'prediction_service_class': UFCPredictionService
    }


def _load_ensemble_system(model_dir: str) -> Optional[AdvancedEnsembleSystem]:
    """Load ensemble system if available"""
    
    try:
        model_path = Path(model_dir)
        
        # Check for ensemble models
        ensemble_models = {}
        
        # Load XGBoost if available
        xgb_path = model_path / 'ufc_xgboost_model.joblib'
        if xgb_path.exists():
            ensemble_models['xgboost'] = joblib.load(xgb_path)
        
        # Load Neural Network if available
        nn_path = model_path / 'ufc_neural_network_model.joblib'
        if nn_path.exists():
            ensemble_models['neural_network'] = joblib.load(nn_path)
        
        # Load baseline Random Forest
        rf_path = model_path / 'ufc_random_forest_model_tuned.joblib'
        if rf_path.exists():
            ensemble_models['random_forest'] = joblib.load(rf_path)
        
        if len(ensemble_models) > 1:
            ensemble_system = AdvancedEnsembleSystem()
            ensemble_system.add_models(ensemble_models)
            return ensemble_system
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error loading ensemble system: {e}")
        return None


# Example usage demonstrating backward compatibility
if __name__ == "__main__":
    
    print("🚀 Enhanced UFC Prediction System Example")
    print("=" * 50)
    
    # Example 1: Create enhanced betting system (automatically detects capabilities)
    try:
        betting_system = create_enhanced_betting_system('model/', enable_ensemble=True)
        
        print(f"✅ Betting system created")
        print(f"   Enhanced features: {ENHANCED_FEATURES_AVAILABLE}")
        print(f"   Ensemble available: {'ensemble_system' in betting_system}")
        
        # Example 2: Create prediction service (automatically uses best available)
        if 'ensemble_system' in betting_system:
            prediction_service = EnhancedPredictionService(
                betting_system, 
                ensemble_system=betting_system['ensemble_system']
            )
            print("   Using enhanced prediction service")
        else:
            prediction_service = UFCPredictionService(betting_system)
            print("   Using baseline prediction service")
        
        # Example 3: Make predictions (same interface, enhanced results when available)
        example_odds = {
            "main_event": {
                "fighter_a": "Jon Jones",
                "fighter_b": "Stipe Miocic",
                "fighter_a_decimal_odds": 1.5,
                "fighter_b_decimal_odds": 2.8
            }
        }
        
        # Optional contextual information (only used if enhanced features available)
        fight_contexts = {
            "main_event": {
                "venue": "Madison Square Garden",
                "is_title_fight": True,
                "is_main_event": True,
                "venue_country": "USA"
            }
        }
        
        # Make prediction (same API, enhanced internally)
        analysis = prediction_service.predict_event(
            example_odds, 
            "UFC Example Event",
            fight_contexts if 'ensemble_system' in betting_system else None
        )
        
        print(f"\n📊 Prediction Results:")
        print(f"   Successful predictions: {analysis.summary['successful_predictions']}")
        print(f"   Failed predictions: {analysis.summary['failed_predictions']}")
        
        if analysis.fight_predictions:
            fight_pred = analysis.fight_predictions[0]
            print(f"   Prediction method: {fight_pred.raw_prediction.get('prediction_method', 'baseline')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   This is expected if enhanced modules are not available")
        print("   The system will gracefully fall back to baseline functionality")