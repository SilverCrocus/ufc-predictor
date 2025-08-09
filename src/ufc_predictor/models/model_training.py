import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import logging
from pathlib import Path
from ..evaluation.calibration import UFCProbabilityCalibrator

logger = logging.getLogger(__name__)


class UFCModelTrainer:
    """Class for training UFC fight prediction models with stacking ensemble support."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.model_scores = {}
        self.feature_columns = None
        self.feature_selector = None  # Store feature selector for consistency
        self.X_test = None
        self.y_test = None
        
        # Ensemble configuration
        self.ensemble_weights = {
            'random_forest': 0.40,
            'xgboost': 0.35,
            'neural_network': 0.25
        }
        
        # Stacking ensemble components
        self.stacking_manager = None
        self.enable_stacking = False
        
        # Probability calibration components
        self.calibrators = {}
        self.enable_calibration = True
        self.calibration_method = 'platt'  # Default to Platt scaling for robustness
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Train a logistic regression model with feature scaling."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           n_estimators: int = 100) -> RandomForestClassifier:
        """Train a random forest model."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          n_iter: int = 100, cv: int = 3) -> RandomForestClassifier:
        """Tune random forest hyperparameters using RandomizedSearchCV."""
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        rf_random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_random_search.fit(X_train, y_train)
        best_model = rf_random_search.best_estimator_
        
        self.models['random_forest_tuned'] = best_model
        
        print("Best hyperparameters found:")
        print(rf_random_search.best_params_)
        
        return best_model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: Optional[pd.DataFrame] = None, 
                      y_val: Optional[pd.Series] = None) -> xgb.XGBClassifier:
        """Train XGBoost model with strict validation and error handling"""
        
        # Comprehensive input validation
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} != {len(y_train)}")
        
        if X_train.isnull().any().any():
            null_cols = X_train.columns[X_train.isnull().any()].tolist()
            raise ValueError(f"Training data contains null values in columns: {null_cols}")
        
        if y_train.isnull().any():
            raise ValueError("Target variable y_train contains null values")
        
        # Validate class distribution
        class_counts = y_train.value_counts()
        if len(class_counts) != 2:
            raise ValueError(f"XGBoost requires binary classification, got {len(class_counts)} classes: {class_counts}")
        
        min_class_size = class_counts.min()
        if min_class_size < 10:
            raise ValueError(f"Insufficient samples for minority class: {min_class_size} < 10")
        
        # Default XGBoost parameters optimized for UFC prediction
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'verbose': False,
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.01,
            'reg_lambda': 1.5,
            'enable_categorical': False,  # Explicit categorical handling
            'validate_parameters': True   # Enable parameter validation
        }
        
        # Prepare validation data with validation
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            # Validate validation data
            if len(X_val) != len(y_val):
                raise ValueError(f"Validation data length mismatch: {len(X_val)} != {len(y_val)}")
            
            if X_val.isnull().any().any():
                null_cols = X_val.columns[X_val.isnull().any()].tolist()
                raise ValueError(f"Validation data contains null values in columns: {null_cols}")
            
            if not set(X_val.columns).issubset(set(X_train.columns)):
                missing_cols = set(X_val.columns) - set(X_train.columns)
                raise ValueError(f"Validation data has columns not in training data: {missing_cols}")
            
            eval_set.append((X_val, y_val))
        
        try:
            model = xgb.XGBClassifier(**default_params)
            
            # Train with comprehensive error handling (early stopping removed for XGBoost 2.1+ compatibility)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Validate trained model
            if not hasattr(model, 'feature_importances_'):
                raise RuntimeError("XGBoost model training failed - no feature importances found")
            
            if model.n_estimators <= 0:
                raise RuntimeError(f"XGBoost model has invalid number of estimators: {model.n_estimators}")
            
            # Test prediction capability
            try:
                test_pred = model.predict_proba(X_train.iloc[:5])
                if test_pred.shape[1] != 2:
                    raise RuntimeError(f"XGBoost model predict_proba returned wrong shape: {test_pred.shape}")
            except Exception as e:
                raise RuntimeError(f"XGBoost model prediction test failed: {str(e)}")
            
            self.models['xgboost'] = model
            logger.info(f"XGBoost model trained successfully with {model.n_estimators} estimators")
            
            return model
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"XGBoost training failed: {str(e)}")
    
    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     n_iter: int = 50, cv: int = 3) -> xgb.XGBClassifier:
        """Tune XGBoost hyperparameters with strict validation and no fallbacks"""
        
        # Strict input validation - NO fallbacks
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} != {len(y_train)}")
        
        if n_iter < 1 or n_iter > 1000:
            raise ValueError(f"n_iter must be between 1 and 1000, got {n_iter}")
        
        if cv < 2 or cv > 10:
            raise ValueError(f"cv must be between 2 and 10, got {cv}")
        
        if X_train.isnull().any().any():
            raise ValueError("Training data contains null values")
        
        if y_train.isnull().any():
            raise ValueError("Target variable contains null values")
        
        # Ensure minimum samples for cross-validation
        min_samples_per_fold = len(X_train) // cv
        if min_samples_per_fold < 20:
            raise ValueError(f"Insufficient samples for {cv}-fold CV: {min_samples_per_fold} samples per fold < 20")
        
        logger.info(f"Starting XGBoost hyperparameter tuning: {n_iter} iterations, {cv}-fold CV")
        
        # XGBoost hyperparameter grid optimized for UFC fight prediction
        xgb_param_grid = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbose': False
        }
        
        xgb_random_search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(**base_params),
            param_distributions=xgb_param_grid,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=cv),  # Respects temporal order
            scoring='neg_log_loss',  # Better for probability calibration
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        try:
            logger.info(f"Starting XGBoost hyperparameter tuning with {n_iter} iterations")
            xgb_random_search.fit(X_train, y_train)
            
            # Validate results
            if not hasattr(xgb_random_search, 'best_estimator_') or xgb_random_search.best_estimator_ is None:
                raise RuntimeError("Hyperparameter search failed to find best estimator")
            
            best_model = xgb_random_search.best_estimator_
            best_score = -xgb_random_search.best_score_
            
            # Validate best model performance
            if best_score > 1.0:  # Log loss should be < 1.0 for decent performance
                raise RuntimeError(f"Hyperparameter tuning failed: best log loss {best_score:.4f} > 1.0")
            
            # Test best model prediction capability
            try:
                test_pred = best_model.predict_proba(X_train[:5])
                if test_pred.shape[1] != 2:
                    raise RuntimeError(f"Best model predict_proba returned wrong shape: {test_pred.shape}")
            except Exception as e:
                raise RuntimeError(f"Best model prediction test failed: {str(e)}")
            
            self.models['xgboost_tuned'] = best_model
            
            logger.info("XGBoost hyperparameter tuning completed successfully")
            logger.info(f"Best hyperparameters: {xgb_random_search.best_params_}")
            logger.info(f"Best cross-validation log loss: {best_score:.4f}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"XGBoost hyperparameter tuning failed: {str(e)}")
            # NO fallbacks - fail fast
            raise RuntimeError(f"XGBoost hyperparameter tuning failed: {str(e)}") from e
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, 
                      show_plots: bool = True) -> Dict[str, Any]:
        """Evaluate a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Handle scaling for logistic regression
        if model_name == 'logistic_regression' and model_name in self.scalers:
            X_test_processed = self.scalers[model_name].transform(X_test)
        else:
            X_test_processed = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the score for later retrieval
        self.model_scores[model_name] = accuracy
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=['Red Wins', 'Blue Wins']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n{model_name.replace('_', ' ').title()} Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:\n{results['classification_report']}")
        
        if show_plots:
            self._plot_confusion_matrix(results['confusion_matrix'], model_name)
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Red Wins', 'Blue Wins'], 
                   yticklabels=['Red Wins', 'Blue Wins'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
        plt.show()
    
    def get_feature_importance(self, model_name: str, feature_names: list, top_n: int = 15) -> pd.Series:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not support feature importance")
        
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        
        print(f"\nTop {top_n} Most Important Features for {model_name}:")
        print(importances.head(top_n))
        
        return importances
    
    def get_model_score(self, model_name: str, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> float:
        """Get the accuracy score for a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        # First check if we have stored scores from evaluation
        if model_name in self.model_scores:
            return self.model_scores[model_name]
        
        # Use provided test data or stored test data
        if X_test is None and hasattr(self, 'X_test') and self.X_test is not None:
            X_test = self.X_test
        if y_test is None and hasattr(self, 'y_test') and self.y_test is not None:
            y_test = self.y_test
            
        # If test data available, calculate score
        if X_test is not None and y_test is not None:
            model = self.models[model_name]
            
            # Handle scaling for logistic regression
            if model_name == 'logistic_regression' and model_name in self.scalers:
                X_test_processed = self.scalers[model_name].transform(X_test)
            else:
                X_test_processed = X_test
            
            y_pred = model.predict(X_test_processed)
            score = accuracy_score(y_test, y_pred)
            self.model_scores[model_name] = score  # Store for future use
            return score
        
        # NO FALLBACKS - require explicit test data or stored scores
        raise ValueError(
            f"Cannot retrieve score for model '{model_name}': no test data provided and no stored score found. "
            "Either provide X_test and y_test parameters or run evaluate_model() first."
        )

    def save_model(self, model_name: str, filepath: str, feature_columns: list = None):
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        joblib.dump(model, filepath)
        
        # Save feature columns if provided
        if feature_columns is not None:
            columns_filepath = filepath.replace('.joblib', '_columns.json')
            with open(columns_filepath, 'w') as f:
                json.dump(feature_columns, f)
            print(f"Feature columns saved to '{columns_filepath}'")
        
        # Save feature selector if exists
        if self.feature_selector is not None:
            selector_filepath = filepath.replace('.joblib', '_feature_selector.json')
            self.feature_selector.save_selection(selector_filepath)
            print(f"Feature selector saved to '{selector_filepath}'")
        
        # Save scaler if exists
        if model_name in self.scalers:
            scaler_filepath = filepath.replace('.joblib', '_scaler.joblib')
            joblib.dump(self.scalers[model_name], scaler_filepath)
            print(f"Scaler saved to '{scaler_filepath}'")
        
        print(f"Model '{model_name}' saved to '{filepath}'")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load a trained model from disk."""
        return joblib.load(filepath)
    
    @staticmethod
    def load_feature_columns(filepath: str) -> list:
        """Load feature columns from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def enable_stacking_ensemble(self, cv_splits: int = 5, 
                                temporal_validation: bool = True,
                                enable_optimization: bool = True):
        """Enable stacking ensemble for improved predictions"""
        try:
            from .stacking_ensemble import ProductionStackingManager, create_stacking_config
            
            # Create stacking configuration
            stacking_config = create_stacking_config(
                base_model_weights=self.ensemble_weights,
                cv_splits=cv_splits,
                temporal_validation=temporal_validation,
                enable_optimization=enable_optimization
            )
            
            # Initialize stacking manager
            self.stacking_manager = ProductionStackingManager(stacking_config)
            self.enable_stacking = True
            
            logger.info(f"Stacking ensemble enabled with {cv_splits} CV splits")
            
        except ImportError as e:
            logger.error(f"Failed to import stacking ensemble: {e}")
            raise ValueError("Stacking ensemble module not available")
        except Exception as e:
            logger.error(f"Failed to initialize stacking ensemble: {e}")
            raise
    
    def train_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the stacking ensemble using base models"""
        
        if not self.enable_stacking:
            raise ValueError("Stacking ensemble not enabled. Call enable_stacking_ensemble() first.")
        
        if not self.models:
            raise ValueError("No base models available. Train base models first.")
        
        logger.info("Training stacking ensemble...")
        
        # Get base models for stacking (prefer tuned versions)
        base_models = {}
        for model_type in ['random_forest', 'xgboost', 'logistic_regression']:
            tuned_name = f'{model_type}_tuned'
            if tuned_name in self.models:
                base_models[model_type] = self.models[tuned_name]
            elif model_type in self.models:
                base_models[model_type] = self.models[model_type]
        
        if len(base_models) < 2:
            raise ValueError(f"Need at least 2 base models for stacking, got {len(base_models)}")
        
        # Load base models into stacking manager
        self.stacking_manager.models = base_models
        self.stacking_manager.is_initialized = True
        
        # Fit stacking ensemble
        self.stacking_manager.fit_stacking_ensemble(X_train, y_train)
        
        logger.info("Stacking ensemble training completed")
    
    def predict_with_stacking(self, X: pd.DataFrame, 
                             fighter_pairs: Optional[list] = None) -> Dict[str, Any]:
        """Generate predictions using stacking ensemble"""
        
        if not self.enable_stacking:
            raise ValueError("Stacking ensemble not enabled")
        
        if not self.stacking_manager.is_stacking_fitted:
            raise ValueError("Stacking ensemble not fitted")
        
        # Generate fighter pairs if not provided
        if fighter_pairs is None:
            fighter_pairs = [(f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(len(X))]
        
        # Get stacking predictions
        stacking_results = self.stacking_manager.predict_stacking(
            X, fighter_pairs, enable_bootstrap=True
        )
        
        # Also get base model predictions for comparison
        base_predictions = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                base_predictions[model_name] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[model_name] = model.predict(X)
        
        return {
            'stacking_results': stacking_results,
            'base_predictions': base_predictions,
            'meta_learner': stacking_results[0].meta_learner_name if stacking_results else None,
            'oof_score': stacking_results[0].oof_validation_score if stacking_results else None
        }
    
    def evaluate_stacking_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate stacking ensemble performance"""
        
        if not self.enable_stacking:
            raise ValueError("Stacking ensemble not enabled")
        
        # Get stacking predictions
        prediction_results = self.predict_with_stacking(X_test)
        stacking_results = prediction_results['stacking_results']
        
        # Extract probabilities and classes
        stacked_probs = [r.stacked_probability for r in stacking_results]
        stacked_classes = [1 if p > 0.5 else 0 for p in stacked_probs]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, stacked_classes)
        auc = roc_auc_score(y_test, stacked_probs)
        log_loss_score = log_loss(y_test, stacked_probs)
        
        # Compare with base models
        base_predictions = prediction_results['base_predictions']
        base_scores = {}
        
        for model_name, preds in base_predictions.items():
            if len(preds.shape) > 1:
                preds = preds[:, 1] if preds.shape[1] > 1 else preds.flatten()
            
            base_classes = [1 if p > 0.5 else 0 for p in preds]
            base_accuracy = accuracy_score(y_test, base_classes)
            try:
                base_auc = roc_auc_score(y_test, preds)
            except ValueError:
                base_auc = 0.5
            
            base_scores[f'{model_name}_accuracy'] = base_accuracy
            base_scores[f'{model_name}_auc'] = base_auc
        
        results = {
            'stacking_accuracy': accuracy,
            'stacking_auc': auc,
            'stacking_log_loss': log_loss_score,
            'meta_learner': prediction_results['meta_learner'],
            'oof_validation_auc': prediction_results['oof_score'],
            **base_scores
        }
        
        # Log results
        logger.info("Stacking Ensemble Evaluation Results:")
        logger.info(f"  Meta-learner: {results['meta_learner']}")
        logger.info(f"  OOF Validation AUC: {results['oof_validation_auc']:.4f}")
        logger.info(f"  Test Accuracy: {results['stacking_accuracy']:.4f}")
        logger.info(f"  Test AUC: {results['stacking_auc']:.4f}")
        logger.info(f"  Test Log Loss: {results['stacking_log_loss']:.4f}")
        
        return results
    
    def save_stacking_ensemble(self, filepath: str):
        """Save stacking ensemble to disk"""
        
        if not self.enable_stacking:
            raise ValueError("Stacking ensemble not enabled")
        
        if not self.stacking_manager.is_stacking_fitted:
            raise ValueError("No fitted stacking ensemble to save")
        
        self.stacking_manager.save_stacking_ensemble(filepath)
    
    def load_stacking_ensemble(self, filepath: str):
        """Load stacking ensemble from disk"""
        
        if not self.models:
            raise ValueError("Base models must be loaded first")
        
        try:
            from .stacking_ensemble import ProductionStackingManager
            
            self.stacking_manager = ProductionStackingManager.load_stacking_ensemble(
                filepath, self.models
            )
            self.enable_stacking = True
            
            logger.info("Stacking ensemble loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import stacking ensemble: {e}")
            raise ValueError("Stacking ensemble module not available")
    
    def get_stacking_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of stacking ensemble"""
        
        if not self.enable_stacking:
            return {'stacking_enabled': False}
        
        summary = {'stacking_enabled': True}
        
        if self.stacking_manager:
            summary.update(self.stacking_manager.get_stacking_summary())
        
        return summary

    def fit_probability_calibration(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit probability calibration for all trained models using out-of-fold validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_split: Fraction of training data to use for calibration
            
        Returns:
            Dictionary of calibration results
        """
        if not self.models:
            raise ValueError("No trained models available. Train models first.")
        
        # Split training data for calibration validation
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=self.random_state
        )
        
        calibration_results = {}
        
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict_proba'):
                logger.info(f"Skipping calibration for {model_name} - no probability prediction")
                continue
            
            logger.info(f"Fitting calibration for {model_name}")
            
            try:
                # Get out-of-fold predictions for calibration
                if model_name == 'logistic_regression' and model_name in self.scalers:
                    X_cal_val_processed = self.scalers[model_name].transform(X_cal_val)
                else:
                    X_cal_val_processed = X_cal_val
                
                # Get uncalibrated probabilities
                y_prob_uncalibrated = model.predict_proba(X_cal_val_processed)[:, 1]
                
                # Check if calibration is needed
                calibrator = UFCProbabilityCalibrator(method=self.calibration_method)
                should_calibrate = calibrator.should_calibrate_model(
                    y_cal_val.values, 
                    y_prob_uncalibrated,
                    ece_threshold=0.05,
                    odds_included='odds' in str(X_train.columns).lower()
                )
                
                if should_calibrate:
                    # Create out-of-fold DataFrame for calibration
                    oof_df = pd.DataFrame({
                        'prob_pred': y_prob_uncalibrated,
                        'winner': y_cal_val.values
                    })
                    
                    # Fit calibration
                    cal_results = calibrator.fit_isotonic_by_segment(
                        oof_df, prob_col='prob_pred', target_col='winner'
                    )
                    
                    self.calibrators[model_name] = calibrator
                    calibration_results[model_name] = {
                        'calibrated': True,
                        'method': self.calibration_method,
                        'ece_improvement': cal_results['overall'].pre_calibration_ece - cal_results['overall'].post_calibration_ece,
                        'brier_improvement': cal_results['overall'].pre_calibration_brier - cal_results['overall'].post_calibration_brier
                    }
                    
                    logger.info(f"Calibration fitted for {model_name}: ECE improved by {calibration_results[model_name]['ece_improvement']:.4f}")
                else:
                    calibration_results[model_name] = {
                        'calibrated': False,
                        'reason': 'Already well-calibrated'
                    }
                    logger.info(f"Calibration skipped for {model_name}: already well-calibrated")
                    
            except Exception as e:
                logger.warning(f"Calibration failed for {model_name}: {str(e)}")
                calibration_results[model_name] = {
                    'calibrated': False,
                    'reason': f'Calibration failed: {str(e)}'
                }
        
        return calibration_results
    
    def get_calibrated_predictions(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated probability predictions for a model.
        
        Args:
            model_name: Name of the model
            X: Features to predict
            
        Returns:
            Calibrated probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Get raw predictions
        if model_name == 'logistic_regression' and model_name in self.scalers:
            X_processed = self.scalers[model_name].transform(X)
        else:
            X_processed = X
        
        raw_probs = model.predict_proba(X_processed)[:, 1]
        
        # Apply calibration if available
        if model_name in self.calibrators:
            prob_df = pd.DataFrame({'prob_pred': raw_probs})
            calibrated_probs = self.calibrators[model_name].apply_calibration(prob_df)
            return calibrated_probs
        else:
            return raw_probs
    
    def set_calibration_method(self, method: str):
        """
        Set the calibration method.
        
        Args:
            method: 'platt' for Platt scaling or 'isotonic' for isotonic regression
        """
        if method not in ['platt', 'isotonic']:
            raise ValueError(f"Method must be 'platt' or 'isotonic', got {method}")
        
        self.calibration_method = method
        logger.info(f"Calibration method set to: {method}")
    
    def save_calibrators(self, filepath: str):
        """Save calibrators to disk."""
        if not self.calibrators:
            logger.warning("No calibrators to save")
            return
        
        for model_name, calibrator in self.calibrators.items():
            cal_filepath = filepath.replace('.joblib', f'_calibrator_{model_name}.joblib')
            calibrator.save_calibrators(cal_filepath)
            logger.info(f"Calibrator for {model_name} saved to {cal_filepath}")
    
    def load_calibrators(self, filepath_pattern: str):
        """Load calibrators from disk."""
        for model_name in self.models.keys():
            cal_filepath = filepath_pattern.replace('.joblib', f'_calibrator_{model_name}.joblib')
            if Path(cal_filepath).exists():
                calibrator = UFCProbabilityCalibrator(method=self.calibration_method)
                calibrator.load_calibrators(cal_filepath)
                self.calibrators[model_name] = calibrator
                logger.info(f"Calibrator for {model_name} loaded from {cal_filepath}")
    
    def get_calibration_summary(self) -> pd.DataFrame:
        """Get summary of calibration status for all models."""
        summaries = []
        
        for model_name, model in self.models.items():
            summary = {
                'model': model_name,
                'has_proba': hasattr(model, 'predict_proba'),
                'calibrated': model_name in self.calibrators,
                'method': self.calibration_method if model_name in self.calibrators else 'none'
            }
            
            if model_name in self.calibrators:
                cal_results = self.calibrators[model_name].get_calibration_summary()
                if not cal_results.empty:
                    overall_result = cal_results[cal_results['segment'] == 'overall'].iloc[0]
                    summary['ece_improvement'] = overall_result['ece_improvement']
                    summary['brier_improvement'] = overall_result['brier_improvement']
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def train_complete_pipeline(X: pd.DataFrame, y: pd.Series, 
                          tune_hyperparameters: bool = True,
                          enable_ensemble: bool = True,
                          enable_stacking: bool = False,
                          enable_feature_selection: bool = False,
                          feature_selection_method: str = 'importance_based',
                          n_features: int = 32,
                          enable_calibration: bool = True,
                          calibration_method: str = 'platt') -> UFCModelTrainer:
    """
    Complete training pipeline for UFC prediction models including ensemble and stacking.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        tune_hyperparameters: Whether to tune hyperparameters
        enable_ensemble: Whether to train XGBoost for ensemble
        enable_stacking: Whether to train stacking ensemble
        enable_feature_selection: Whether to apply feature selection
        feature_selection_method: Method for feature selection
        n_features: Number of features to select
        enable_calibration: Whether to apply probability calibration
        calibration_method: 'platt' for Platt scaling, 'isotonic' for isotonic regression
        
    Returns:
        Trained UFCModelTrainer instance with optional feature selection
    """
    trainer = UFCModelTrainer()
    
    # Set calibration method
    if enable_calibration:
        trainer.set_calibration_method(calibration_method)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    print(f"Data split - Training: {X_train.shape}, Testing: {X_test.shape}")
    
    # Apply feature selection if enabled
    feature_selector = None
    if enable_feature_selection:
        print(f"\n🎯 FEATURE SELECTION")
        print(f"Method: {feature_selection_method}, Target features: {n_features}")
        
        from .feature_selection import UFCFeatureSelector
        feature_selector = UFCFeatureSelector(
            selection_method=feature_selection_method,
            n_features=n_features,
            random_state=trainer.random_state
        )
        
        # Fit and transform training data
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_test = feature_selector.transform(X_test)
        
        print(f"Feature selection complete: {X_train.shape[1]} features selected")
        print(f"Feature reduction: {X.shape[1]} -> {X_train.shape[1]} "
              f"({X_train.shape[1]/X.shape[1]:.1%})")
        
        # Store feature selector in trainer for later use
        trainer.feature_selector = feature_selector
        
        # Display top selected features
        selected_summary = feature_selector.get_selection_summary()
        print(f"Top 10 selected features:")
        for i, (feature, score) in enumerate(selected_summary['top_features'][:10], 1):
            print(f"  {i:2d}. {feature}: {score:.4f}")
    
    # Store test data for later use
    trainer.X_test = X_test
    trainer.y_test = y_test
    
    # Train logistic regression
    print("\nTraining Logistic Regression...")
    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model('logistic_regression', X_test, y_test)
    
    # Train random forest
    print("\nTraining Random Forest...")
    trainer.train_random_forest(X_train, y_train)
    trainer.evaluate_model('random_forest', X_test, y_test)
    trainer.get_feature_importance('random_forest', X.columns.tolist())
    
    # Train XGBoost for ensemble
    if enable_ensemble:
        print("\nTraining XGBoost for Ensemble...")
        # Split training data further for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=trainer.random_state
        )
        trainer.train_xgboost(X_train_split, y_train_split, X_val, y_val)
        trainer.evaluate_model('xgboost', X_test, y_test)
        trainer.get_feature_importance('xgboost', X.columns.tolist())
    
    # Tune hyperparameters if requested
    if tune_hyperparameters:
        print("\nTuning Random Forest hyperparameters...")
        trainer.tune_random_forest(X_train, y_train)
        trainer.evaluate_model('random_forest_tuned', X_test, y_test)
        
        if enable_ensemble:
            print("\nTuning XGBoost hyperparameters...")
            trainer.tune_xgboost(X_train, y_train)
            trainer.evaluate_model('xgboost_tuned', X_test, y_test)
    
    # Save feature columns for later use
    trainer.feature_columns = X.columns.tolist()
    
    # Fit probability calibration if enabled
    if enable_calibration:
        print("\n" + "="*60)
        print("📊 PROBABILITY CALIBRATION")
        print("="*60)
        print(f"Method: {calibration_method}")
        
        calibration_results = trainer.fit_probability_calibration(X_train, y_train)
        
        # Display calibration summary
        print("\nCalibration Results:")
        for model_name, result in calibration_results.items():
            if result['calibrated']:
                print(f"  ✅ {model_name}: {result['method']} calibration applied")
                print(f"     ECE improvement: {result['ece_improvement']:.4f}")
                print(f"     Brier improvement: {result['brier_improvement']:.4f}")
            else:
                print(f"  ⚪ {model_name}: {result['reason']}")
        
        # Show calibration summary table
        cal_summary = trainer.get_calibration_summary()
        if not cal_summary.empty:
            print(f"\nCalibration Summary:")
            print(cal_summary.to_string(index=False))
    
    # Train stacking ensemble if requested
    if enable_stacking:
        print("\n" + "="*60)
        print("🎯 STACKING ENSEMBLE TRAINING")
        print("="*60)
        
        # Enable stacking ensemble
        trainer.enable_stacking_ensemble(
            cv_splits=5,
            temporal_validation=True,
            enable_optimization=True
        )
        
        # Train stacking ensemble
        trainer.train_stacking_ensemble(X_train, y_train)
        
        # Evaluate stacking ensemble
        stacking_scores = trainer.evaluate_stacking_ensemble(X_test, y_test)
        
        print(f"Stacking ensemble complete!")
        print(f"Meta-learner: {stacking_scores['meta_learner']}")
        print(f"Stacking accuracy: {stacking_scores['stacking_accuracy']:.4f}")
        print(f"Stacking AUC: {stacking_scores['stacking_auc']:.4f}")
        print(f"OOF validation AUC: {stacking_scores['oof_validation_auc']:.4f}")
    
    # Print ensemble summary if enabled
    if enable_ensemble or enable_stacking:
        print("\n" + "="*60)
        print("🚀 ENSEMBLE TRAINING COMPLETE")
        print("="*60)
        print(f"Random Forest accuracy: {trainer.get_model_score('random_forest'):.4f}")
        if enable_ensemble:
            print(f"XGBoost accuracy: {trainer.get_model_score('xgboost'):.4f}")
        
        if tune_hyperparameters:
            print(f"Tuned Random Forest accuracy: {trainer.get_model_score('random_forest_tuned'):.4f}")
            if enable_ensemble:
                print(f"Tuned XGBoost accuracy: {trainer.get_model_score('xgboost_tuned'):.4f}")
        
        if enable_stacking:
            stacking_summary = trainer.get_stacking_summary()
            print(f"Stacking enabled: {stacking_summary.get('is_fitted', False)}")
        else:
            print(f"Ensemble weights: {trainer.ensemble_weights}")
        
        print("Ready for advanced ensemble prediction!")
    
    return trainer