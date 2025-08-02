"""
Production-Grade XGBoost Training with Strict Error Handling
============================================================

Replacement for existing model training with comprehensive validation,
strict error handling, and no fallback mechanisms.

Key Features:
- Comprehensive input validation and feature validation
- Strict XGBoost parameter validation
- Memory monitoring during training
- Detailed error reporting with context
- No silent failures or fallbacks
- Production-ready hyperparameter tuning
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import logging
from dataclasses import dataclass
import joblib
import time
import psutil
import gc
from pathlib import Path
import json

from .enhanced_error_handling import UFCPredictorError

logger = logging.getLogger(__name__)


class TrainingError(UFCPredictorError):
    """Training-specific errors"""
    pass


class ValidationError(UFCPredictorError):
    """Input validation errors"""
    pass


class HyperparameterError(UFCPredictorError):
    """Hyperparameter tuning errors"""
    pass


class TrainingResult(NamedTuple):
    """Structured training result"""
    model: xgb.XGBClassifier
    training_score: float
    validation_score: float
    test_score: float
    training_time_s: float
    feature_importance: Dict[str, float]
    model_params: Dict[str, Any]
    training_metadata: Dict[str, Any]


@dataclass
class XGBoostConfig:
    """XGBoost training configuration"""
    # Core XGBoost parameters
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    tree_method: str = 'hist'
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.01
    reg_lambda: float = 1.5
    
    # Training parameters
    early_stopping_rounds: int = 20
    verbose: bool = False
    random_state: int = 42
    n_jobs: int = -1
    
    # Validation parameters
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    
    # Resource limits
    max_memory_mb: int = 4096
    max_training_time_s: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.n_estimators <= 0:
            raise TrainingError(f"n_estimators must be positive, got {self.n_estimators}")
        
        if self.max_depth <= 0:
            raise TrainingError(f"max_depth must be positive, got {self.max_depth}")
        
        if not 0 < self.learning_rate <= 1:
            raise TrainingError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        
        if not 0 < self.subsample <= 1:
            raise TrainingError(f"subsample must be in (0, 1], got {self.subsample}")
        
        if not 0 < self.colsample_bytree <= 1:
            raise TrainingError(f"colsample_bytree must be in (0, 1], got {self.colsample_bytree}")
        
        if not 0 < self.validation_split < 1:
            raise TrainingError(f"validation_split must be in (0, 1), got {self.validation_split}")


class ProductionXGBoostTrainer:
    """Production-grade XGBoost trainer with strict error handling"""
    
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.memory_monitor = psutil.Process()
        self.training_history = []
        
        logger.info(f"Initialized ProductionXGBoostTrainer with config: {self.config}")
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   X_test: Optional[pd.DataFrame] = None,
                   y_test: Optional[pd.Series] = None) -> TrainingResult:
        """Train XGBoost model with comprehensive validation"""
        
        start_time = time.time()
        
        # Comprehensive input validation
        self._validate_training_inputs(X, y, X_test, y_test)
        
        # Memory check
        self._check_memory_usage("before training")
        
        logger.info(f"Starting XGBoost training: {len(X)} samples, {len(X.columns)} features")
        
        try:
            # Split training/validation data
            X_train, X_val, y_train, y_val = self._split_training_data(X, y)
            
            # Create and configure XGBoost model
            model = self._create_xgboost_model()
            
            # Prepare evaluation sets
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'validation']
            
            # Train model with monitoring
            model = self._train_with_monitoring(model, X_train, y_train, eval_set, eval_names)
            
            # Comprehensive evaluation
            training_score = self._evaluate_model(model, X_train, y_train, "training")
            validation_score = self._evaluate_model(model, X_val, y_val, "validation")
            
            test_score = None
            if X_test is not None and y_test is not None:
                test_score = self._evaluate_model(model, X_test, y_test, "test")
            
            # Extract feature importance
            feature_importance = self._get_feature_importance(model, X.columns)
            
            # Create training metadata
            training_time = time.time() - start_time
            metadata = self._create_training_metadata(X, y, training_time, model)
            
            # Memory check after training
            self._check_memory_usage("after training")
            
            result = TrainingResult(
                model=model,
                training_score=training_score,
                validation_score=validation_score,
                test_score=test_score,
                training_time_s=training_time,
                feature_importance=feature_importance,
                model_params=model.get_params(),
                training_metadata=metadata
            )
            
            self.training_history.append(result)
            
            logger.info(f"XGBoost training completed in {training_time:.1f}s - "
                       f"Train: {training_score:.4f}, Val: {validation_score:.4f}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"XGBoost training failed after {training_time:.1f}s: {str(e)}")
            
            if isinstance(e, (TrainingError, ValidationError)):
                raise
            else:
                raise TrainingError(f"XGBoost training failed: {str(e)}")
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           param_grid: Optional[Dict[str, List]] = None,
                           n_iter: int = 50, cv: int = 3) -> TrainingResult:
        """Hyperparameter tuning with strict validation"""
        
        start_time = time.time()
        
        # Input validation
        self._validate_training_inputs(X, y)
        
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        logger.info(f"Starting hyperparameter tuning: {n_iter} iterations, {cv}-fold CV")
        
        try:
            # Create base model
            base_model = self._create_xgboost_model()
            
            # Configure cross-validation
            cv_splitter = TimeSeriesSplit(n_splits=cv)
            
            # Configure RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring='neg_log_loss',
                n_jobs=1,  # Control parallelism at XGBoost level
                random_state=self.config.random_state,
                error_score='raise',  # Fail on any error
                verbose=1 if logger.isEnabledFor(logging.INFO) else 0
            )
            
            # Perform search with timeout monitoring
            search = self._perform_hyperparameter_search(search, X, y, start_time)
            
            # Validate search results
            if not hasattr(search, 'best_estimator_') or search.best_estimator_ is None:
                raise HyperparameterError("Hyperparameter search failed to find best estimator")
            
            best_model = search.best_estimator_
            best_score = -search.best_score_  # Convert back from negative
            
            logger.info(f"Best hyperparameters found: {search.best_params_}")
            logger.info(f"Best CV score: {best_score:.4f}")
            
            # Train final model on full dataset
            best_model.fit(X, y)
            
            # Evaluate final model
            final_score = self._evaluate_model(best_model, X, y, "final")
            feature_importance = self._get_feature_importance(best_model, X.columns)
            
            training_time = time.time() - start_time
            metadata = self._create_training_metadata(X, y, training_time, best_model)
            metadata.update({
                'hyperparameter_search': {
                    'best_params': search.best_params_,
                    'best_cv_score': best_score,
                    'n_iterations': n_iter,
                    'cv_folds': cv
                }
            })
            
            result = TrainingResult(
                model=best_model,
                training_score=final_score,
                validation_score=best_score,
                test_score=None,
                training_time_s=training_time,
                feature_importance=feature_importance,
                model_params=best_model.get_params(),
                training_metadata=metadata
            )
            
            self.training_history.append(result)
            
            logger.info(f"Hyperparameter tuning completed in {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Hyperparameter tuning failed after {training_time:.1f}s: {str(e)}")
            
            if isinstance(e, (HyperparameterError, TrainingError, ValidationError)):
                raise
            else:
                raise HyperparameterError(f"Hyperparameter tuning failed: {str(e)}")
    
    def _validate_training_inputs(self, X: pd.DataFrame, y: pd.Series,
                                X_test: Optional[pd.DataFrame] = None,
                                y_test: Optional[pd.Series] = None):
        """Comprehensive input validation"""
        
        # Validate X
        if X.empty:
            raise ValidationError("Training data X is empty")
        
        if len(X) < 10:
            raise ValidationError(f"Insufficient training samples: {len(X)} < 10")
        
        # Check for null values
        null_columns = X.columns[X.isnull().any()].tolist()
        if null_columns:
            null_counts = X[null_columns].isnull().sum().to_dict()
            raise ValidationError(f"Training data contains null values: {null_counts}")
        
        # Check for infinite values
        inf_columns = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
        if inf_columns:
            raise ValidationError(f"Training data contains infinite values: {inf_columns}")
        
        # Validate y
        if len(y) != len(X):
            raise ValidationError(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
        
        if y.isnull().any():
            raise ValidationError(f"Target variable contains {y.isnull().sum()} null values")
        
        # Check class distribution
        unique_classes = y.unique()
        if len(unique_classes) != 2:
            raise ValidationError(f"Expected binary classification, got {len(unique_classes)} classes: {unique_classes}")
        
        if not set(unique_classes).issubset({0, 1}):
            raise ValidationError(f"Target classes must be 0 and 1, got {unique_classes}")
        
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 5:
            raise ValidationError(f"Insufficient samples for minority class: {min_class_count} < 5")
        
        # Check class imbalance
        class_ratio = min_class_count / len(y)
        if class_ratio < 0.05:  # Less than 5% of samples
            logger.warning(f"Severe class imbalance detected: {class_ratio:.1%} minority class")
        
        # Validate test data if provided
        if X_test is not None:
            if X_test.empty:
                raise ValidationError("Test data X_test is empty")
            
            if set(X.columns) != set(X_test.columns):
                raise ValidationError("Training and test data have different features")
            
            if y_test is not None and len(X_test) != len(y_test):
                raise ValidationError(f"Test data length mismatch: X_test has {len(X_test)}, y_test has {len(y_test)}")
        
        # Check data types
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) != len(X.columns):
            non_numeric = list(set(X.columns) - set(numeric_columns))
            raise ValidationError(f"Non-numeric features found: {non_numeric}")
        
        logger.debug(f"Input validation passed: {len(X)} samples, {len(X.columns)} features, "
                    f"class distribution: {dict(class_counts)}")
    
    def _split_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split training data into train/validation sets"""
        
        from sklearn.model_selection import train_test_split
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=y  # Maintain class distribution
            )
            
            logger.debug(f"Data split: Train {len(X_train)}, Validation {len(X_val)}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            raise TrainingError(f"Failed to split training data: {str(e)}")
    
    def _create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with validated parameters"""
        
        try:
            params = {
                'objective': self.config.objective,
                'eval_metric': self.config.eval_metric,
                'tree_method': self.config.tree_method,
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'min_child_weight': self.config.min_child_weight,
                'gamma': self.config.gamma,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs,
                'verbose': self.config.verbose
            }
            
            model = xgb.XGBClassifier(**params)
            
            logger.debug(f"Created XGBoost model with parameters: {params}")
            
            return model
            
        except Exception as e:
            raise TrainingError(f"Failed to create XGBoost model: {str(e)}")
    
    def _train_with_monitoring(self, model: xgb.XGBClassifier, 
                             X_train: pd.DataFrame, y_train: pd.Series,
                             eval_set: List[Tuple], eval_names: List[str]) -> xgb.XGBClassifier:
        """Train model with memory and time monitoring"""
        
        start_time = time.time()
        
        try:
            fit_params = {
                'eval_set': eval_set,
                'verbose': self.config.verbose
            }
            
            if self.config.enable_early_stopping:
                fit_params['early_stopping_rounds'] = self.config.early_stopping_rounds
            
            # Monitor training
            class TrainingMonitor:
                def __init__(self, trainer, start_time):
                    self.trainer = trainer
                    self.start_time = start_time
                    self.last_check = start_time
                
                def __call__(self, env):
                    current_time = time.time()
                    
                    # Check training timeout
                    if current_time - self.start_time > self.trainer.config.max_training_time_s:
                        raise TrainingError(f"Training timeout: {current_time - self.start_time:.1f}s > {self.trainer.config.max_training_time_s}s")
                    
                    # Check memory periodically (every 10 iterations)
                    if env.iteration % 10 == 0:
                        self.trainer._check_memory_usage(f"iteration {env.iteration}")
            
            # Add training monitor
            monitor = TrainingMonitor(self, start_time)
            fit_params['callbacks'] = [monitor]
            
            # Train model
            model.fit(X_train, y_train, **fit_params)
            
            # Validate training completed successfully
            if not hasattr(model, 'n_estimators_') or model.n_estimators_ == 0:
                raise TrainingError("Model training failed - no estimators trained")
            
            training_time = time.time() - start_time
            logger.info(f"XGBoost training completed: {model.n_estimators_} estimators in {training_time:.1f}s")
            
            return model
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"XGBoost training failed after {training_time:.1f}s: {str(e)}")
            
            if isinstance(e, TrainingError):
                raise
            else:
                raise TrainingError(f"XGBoost model training failed: {str(e)}")
    
    def _perform_hyperparameter_search(self, search: RandomizedSearchCV, 
                                     X: pd.DataFrame, y: pd.Series, 
                                     start_time: float) -> RandomizedSearchCV:
        """Perform hyperparameter search with monitoring"""
        
        try:
            # Fit with timeout monitoring
            search.fit(X, y)
            
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.config.max_training_time_s:
                raise HyperparameterError(f"Hyperparameter search timeout: {elapsed_time:.1f}s")
            
            # Validate search completed
            if not hasattr(search, 'best_score_') or search.best_score_ is None:
                raise HyperparameterError("Hyperparameter search failed to complete")
            
            return search
            
        except Exception as e:
            if isinstance(e, HyperparameterError):
                raise
            else:
                raise HyperparameterError(f"Hyperparameter search failed: {str(e)}")
    
    def _evaluate_model(self, model: xgb.XGBClassifier, X: pd.DataFrame, 
                       y: pd.Series, dataset_name: str) -> float:
        """Comprehensive model evaluation"""
        
        try:
            # Get predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            logloss = log_loss(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            
            logger.info(f"{dataset_name.title()} metrics - "
                       f"Accuracy: {accuracy:.4f}, "
                       f"LogLoss: {logloss:.4f}, "
                       f"AUC: {auc:.4f}")
            
            return accuracy
            
        except Exception as e:
            raise TrainingError(f"Model evaluation failed on {dataset_name} set: {str(e)}")
    
    def _get_feature_importance(self, model: xgb.XGBClassifier, 
                              feature_names: pd.Index) -> Dict[str, float]:
        """Extract feature importance with validation"""
        
        try:
            if not hasattr(model, 'feature_importances_'):
                raise TrainingError("Model has no feature importance")
            
            importances = model.feature_importances_
            
            if len(importances) != len(feature_names):
                raise TrainingError(f"Feature importance length mismatch: {len(importances)} != {len(feature_names)}")
            
            # Create sorted importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            logger.debug(f"Top 5 features: {list(sorted_importance.keys())[:5]}")
            
            return sorted_importance
            
        except Exception as e:
            raise TrainingError(f"Failed to extract feature importance: {str(e)}")
    
    def _create_training_metadata(self, X: pd.DataFrame, y: pd.Series, 
                                training_time: float, model: xgb.XGBClassifier) -> Dict[str, Any]:
        """Create comprehensive training metadata"""
        
        memory_info = self.memory_monitor.memory_info()
        
        metadata = {
            'timestamp': time.time(),
            'training_time_s': training_time,
            'data_shape': X.shape,
            'class_distribution': dict(y.value_counts()),
            'feature_names': list(X.columns),
            'model_type': 'XGBClassifier',
            'xgboost_version': xgb.__version__,
            'n_estimators_trained': getattr(model, 'n_estimators_', 0),
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'config': {
                'objective': self.config.objective,
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'early_stopping_rounds': self.config.early_stopping_rounds
            }
        }
        
        return metadata
    
    def _check_memory_usage(self, context: str = ""):
        """Check memory usage and enforce limits"""
        memory_info = self.memory_monitor.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        logger.debug(f"Memory usage {context}: {memory_mb:.1f} MB")
        
        if memory_mb > self.config.max_memory_mb:
            raise TrainingError(f"Memory limit exceeded {context}: {memory_mb:.1f} MB > {self.config.max_memory_mb} MB")
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default hyperparameter grid for UFC prediction"""
        
        return {
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
    
    def save_model(self, model: xgb.XGBClassifier, model_path: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save model with metadata"""
        
        try:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"Model and metadata saved to {model_path}")
            else:
                logger.info(f"Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs"""
        
        if not self.training_history:
            return {'message': 'No training runs completed'}
        
        latest_run = self.training_history[-1]
        
        return {
            'total_training_runs': len(self.training_history),
            'latest_run': {
                'training_score': latest_run.training_score,
                'validation_score': latest_run.validation_score,
                'test_score': latest_run.test_score,
                'training_time_s': latest_run.training_time_s,
                'n_features': len(latest_run.feature_importance),
                'top_features': list(latest_run.feature_importance.keys())[:5]
            },
            'performance_trends': {
                'training_scores': [run.training_score for run in self.training_history],
                'validation_scores': [run.validation_score for run in self.training_history],
                'training_times': [run.training_time_s for run in self.training_history]
            }
        }


def create_production_xgboost_config(n_estimators: int = 300,
                                   max_depth: int = 6,
                                   learning_rate: float = 0.05,
                                   max_memory_mb: int = 4096) -> XGBoostConfig:
    """Factory function for production XGBoost configuration"""
    
    return XGBoostConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_memory_mb=max_memory_mb
    )


# Example usage for testing
if __name__ == "__main__":
    print("Production XGBoost Trainer Test")
    print("=" * 50)
    
    # Create test configuration
    config = create_production_xgboost_config(n_estimators=50, max_memory_mb=1024)
    trainer = ProductionXGBoostTrainer(config)
    
    # Create realistic test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create realistic binary target with some signal
    coefficients = np.random.randn(n_features) * 0.1
    linear_combination = X.dot(coefficients) + np.random.randn(n_samples) * 0.5
    y = pd.Series((linear_combination > 0).astype(int))
    
    # Test/train split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    try:
        # Test basic training
        result = trainer.train_model(X_train, y_train, X_test, y_test)
        print(f"✅ Training completed: Train {result.training_score:.3f}, "
              f"Val {result.validation_score:.3f}, Test {result.test_score:.3f}")
        
        # Test hyperparameter tuning (small grid for speed)
        small_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        }
        
        tuned_result = trainer.tune_hyperparameters(X_train, y_train, 
                                                   param_grid=small_grid, 
                                                   n_iter=4, cv=2)
        print(f"✅ Hyperparameter tuning completed: CV score {tuned_result.validation_score:.3f}")
        
        # Test model saving
        success = trainer.save_model(result.model, '/tmp/test_xgboost_model.joblib', 
                                   result.training_metadata)
        print(f"✅ Model saving: {success}")
        
        # Test training summary
        summary = trainer.get_training_summary()
        print(f"✅ Training summary: {summary['total_training_runs']} runs completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        import os
        for file in ['/tmp/test_xgboost_model.joblib', '/tmp/test_xgboost_model.json']:
            try:
                os.remove(file)
            except:
                pass
    
    print("✅ Production XGBoost trainer test completed")