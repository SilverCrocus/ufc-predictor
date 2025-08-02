#!/usr/bin/env python3
"""
Unit Tests for Production Ensemble Manager

Tests the production-grade ensemble manager with comprehensive validation
of all production fixes including error handling, memory management,
and input validation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from ufc_predictor.models.production_ensemble_manager import (
    ProductionEnsembleManager,
    EnsembleConfig,
    PredictionResult,
    EnsembleError,
    ValidationError,
    ModelLoadError,
    create_production_ensemble_config
)


class TestEnsembleConfig:
    """Test suite for EnsembleConfig validation"""
    
    def test_valid_config_creation(self):
        """Test successful config creation with valid parameters"""
        weights = {'model1': 0.6, 'model2': 0.4}
        config = EnsembleConfig(
            model_weights=weights,
            confidence_level=0.95,
            bootstrap_samples=100,
            max_memory_mb=2048
        )
        
        assert config.model_weights == weights
        assert config.confidence_level == 0.95
        assert config.bootstrap_samples == 100
        assert config.max_memory_mb == 2048
    
    def test_config_empty_weights(self):
        """Test config fails with empty weights"""
        with pytest.raises(EnsembleError, match="No model weights provided"):
            EnsembleConfig(model_weights={})
    
    def test_config_invalid_weight_sum(self):
        """Test config fails when weights don't sum to 1.0"""
        invalid_weights = {'model1': 0.3, 'model2': 0.5}  # Sum to 0.8
        with pytest.raises(EnsembleError, match="Model weights must sum to 1.0"):
            EnsembleConfig(model_weights=invalid_weights)
    
    def test_config_invalid_confidence_level(self):
        """Test config fails with invalid confidence level"""
        weights = {'model1': 1.0}
        
        with pytest.raises(EnsembleError, match="confidence_level must be between 0 and 1"):
            EnsembleConfig(model_weights=weights, confidence_level=1.5)
        
        with pytest.raises(EnsembleError, match="confidence_level must be between 0 and 1"):
            EnsembleConfig(model_weights=weights, confidence_level=0.0)
    
    def test_config_invalid_bootstrap_samples(self):
        """Test config fails with invalid bootstrap samples"""
        weights = {'model1': 1.0}
        
        with pytest.raises(EnsembleError, match="bootstrap_samples must be positive"):
            EnsembleConfig(model_weights=weights, bootstrap_samples=0)
        
        with pytest.raises(EnsembleError, match="bootstrap_samples must be positive"):
            EnsembleConfig(model_weights=weights, bootstrap_samples=-10)


class TestProductionEnsembleManager:
    """Test suite for ProductionEnsembleManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.weights = {'random_forest': 0.6, 'xgboost': 0.4}
        self.config = create_production_ensemble_config(
            model_weights=self.weights,
            bootstrap_samples=20,  # Small for fast tests
            max_memory_mb=1024,
            n_jobs=1  # Sequential for deterministic tests
        )
        self.manager = ProductionEnsembleManager(self.config)
        
        # Create test data
        np.random.seed(42)
        self.X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        self.fighter_pairs = [(f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(10)]
        
        # Create temporary directory for model files
        self.temp_dir = tempfile.mkdtemp()
        self.model_paths = {}
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_dummy_models(self):
        """Create dummy models for testing"""
        class DummyModel:
            def __init__(self, bias=0.0, fail_prediction=False):
                self.bias = bias
                self.fail_prediction = fail_prediction
            
            def predict_proba(self, X):
                if self.fail_prediction:
                    raise ValueError("Model prediction failed")
                
                np.random.seed(42)
                n_samples = len(X)
                prob_1 = np.clip(0.5 + self.bias + 0.1 * np.random.randn(n_samples), 0.01, 0.99)
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
            
            def predict(self, X):
                if self.fail_prediction:
                    raise ValueError("Model prediction failed")
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        
        models = {}
        for name in self.weights.keys():
            models[name] = DummyModel(bias=np.random.randn() * 0.1)
        
        return models
    
    def _save_models_to_temp(self, models):
        """Save models to temporary files"""
        for name, model in models.items():
            path = Path(self.temp_dir) / f'{name}_model.joblib'
            joblib.dump(model, path)
            self.model_paths[name] = str(path)
        return self.model_paths
    
    def test_initialization(self):
        """Test manager initialization"""
        assert self.manager.config == self.config
        assert self.manager.models == {}
        assert not self.manager.is_initialized
        assert self.manager.prediction_count == 0
        assert self.manager.error_count == 0
        assert self.manager.bootstrap_sampler is not None
    
    def test_load_models_success(self):
        """Test successful model loading"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        
        self.manager.load_models(model_paths)
        
        assert self.manager.is_initialized
        assert len(self.manager.models) == len(self.weights)
        assert set(self.manager.models.keys()) == set(self.weights.keys())
    
    def test_load_models_missing_file(self):
        """Test model loading fails with missing file"""
        missing_paths = {'random_forest': '/nonexistent/path/model.joblib'}
        
        with pytest.raises(ModelLoadError, match="Model file not found"):
            self.manager.load_models(missing_paths)
    
    def test_load_models_invalid_model(self):
        """Test model loading fails with invalid model"""
        # Save invalid object as model
        invalid_model = "not a model"
        path = Path(self.temp_dir) / 'invalid_model.joblib'
        joblib.dump(invalid_model, path)
        
        invalid_paths = {'random_forest': str(path)}
        
        with pytest.raises(ModelLoadError, match="no predict_proba or predict method"):
            self.manager.load_models(invalid_paths)
    
    def test_load_models_weight_mismatch(self):
        """Test model loading fails with weight mismatch"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        
        # Add extra model not in weights
        extra_model = models['random_forest']
        extra_path = Path(self.temp_dir) / 'extra_model.joblib'
        joblib.dump(extra_model, extra_path)
        model_paths['extra_model'] = str(extra_path)
        
        with pytest.raises(ModelLoadError, match="Model mismatch"):
            self.manager.load_models(model_paths)
    
    def test_load_feature_columns(self):
        """Test feature column loading"""
        feature_columns = ['feature_0', 'feature_1', 'feature_2']
        feature_path = Path(self.temp_dir) / 'features.json'
        with open(feature_path, 'w') as f:
            json.dump(feature_columns, f)
        
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        
        self.manager.load_models(model_paths, str(feature_path))
        
        assert self.manager.feature_columns == feature_columns
    
    def test_load_invalid_feature_columns(self):
        """Test loading invalid feature columns fails"""
        invalid_features = "not a list"
        feature_path = Path(self.temp_dir) / 'invalid_features.json'
        with open(feature_path, 'w') as f:
            json.dump(invalid_features, f)
        
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        
        with pytest.raises(ValidationError, match="Invalid feature columns format"):
            self.manager.load_models(model_paths, str(feature_path))
    
    def test_predict_fights_not_initialized(self):
        """Test prediction fails when models not loaded"""
        with pytest.raises(EnsembleError, match="Models not loaded"):
            self.manager.predict_fights(self.X_test, self.fighter_pairs)
    
    def test_predict_fights_success(self):
        """Test successful fight prediction"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        self.manager.load_models(model_paths)
        
        results = self.manager.predict_fights(self.X_test, self.fighter_pairs, enable_bootstrap=False)
        
        assert len(results) == len(self.fighter_pairs)
        assert self.manager.prediction_count == len(results)
        
        for result in results:
            assert isinstance(result, PredictionResult)
            assert 0 <= result.ensemble_probability <= 1
            assert result.fighter_a in [pair[0] for pair in self.fighter_pairs]
            assert result.fighter_b in [pair[1] for pair in self.fighter_pairs]
            assert isinstance(result.model_breakdown, dict)
            assert 0 <= result.data_quality_score <= 1
            assert result.processing_time_ms >= 0
    
    def test_predict_fights_with_bootstrap(self):
        """Test fight prediction with bootstrap confidence intervals"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        self.manager.load_models(model_paths)
        
        # Mock bootstrap sampler to avoid complexity
        with patch.object(self.manager, '_get_bootstrap_confidence_intervals') as mock_bootstrap:
            confidence_intervals = [(0.3, 0.7)] * len(self.fighter_pairs)
            uncertainty_scores = np.array([0.4] * len(self.fighter_pairs))
            mock_bootstrap.return_value = (confidence_intervals, uncertainty_scores)
            
            results = self.manager.predict_fights(self.X_test, self.fighter_pairs, enable_bootstrap=True)
            
            assert len(results) == len(self.fighter_pairs)
            for result in results:
                assert result.confidence_interval == (0.3, 0.7)
                assert result.uncertainty_score == 0.4
    
    def test_validate_prediction_inputs_empty_dataframe(self):
        """Test input validation fails with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="Input DataFrame X is empty"):
            self.manager._validate_prediction_inputs(empty_df, self.fighter_pairs)
    
    def test_validate_prediction_inputs_length_mismatch(self):
        """Test input validation fails with length mismatch"""
        wrong_pairs = [('A', 'B')]  # Only 1 pair for 10 rows of data
        
        with pytest.raises(ValidationError, match="Data length mismatch"):
            self.manager._validate_prediction_inputs(self.X_test, wrong_pairs)
    
    def test_validate_prediction_inputs_null_values(self):
        """Test input validation fails with null values"""
        null_df = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, 5, 6]})
        null_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F')]
        
        with pytest.raises(ValidationError, match="Input data contains null values"):
            self.manager._validate_prediction_inputs(null_df, null_pairs)
    
    def test_validate_prediction_inputs_infinite_values(self):
        """Test input validation fails with infinite values"""
        inf_df = pd.DataFrame({'col1': [1, np.inf, 3], 'col2': [4, 5, 6]})
        inf_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F')]
        
        with pytest.raises(ValidationError, match="Input data contains infinite values"):
            self.manager._validate_prediction_inputs(inf_df, inf_pairs)
    
    def test_validate_prediction_inputs_empty_fighter_names(self):
        """Test input validation fails with empty fighter names"""
        invalid_pairs = [('Fighter_A', ''), ('Fighter_C', 'Fighter_D')]
        test_df = pd.DataFrame(np.random.randn(2, 3))
        
        with pytest.raises(ValidationError, match="Empty fighter name"):
            self.manager._validate_prediction_inputs(test_df, invalid_pairs)
    
    def test_validate_prediction_inputs_same_fighter(self):
        """Test input validation fails with same fighter names"""
        same_pairs = [('Fighter_A', 'Fighter_A')]
        test_df = pd.DataFrame(np.random.randn(1, 3))
        
        with pytest.raises(ValidationError, match="Same fighter in pair"):
            self.manager._validate_prediction_inputs(test_df, same_pairs)
    
    def test_validate_prediction_inputs_non_numeric_columns(self):
        """Test input validation fails with non-numeric columns"""
        text_df = pd.DataFrame({'text_col': ['a', 'b'], 'num_col': [1, 2]})
        text_pairs = [('A', 'B'), ('C', 'D')]
        
        with pytest.raises(ValidationError, match="Non-numeric columns found"):
            self.manager._validate_prediction_inputs(text_df, text_pairs)
    
    def test_get_base_ensemble_predictions_success(self):
        """Test successful base ensemble prediction"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        self.manager.load_models(model_paths)
        
        predictions = self.manager._get_base_ensemble_predictions(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert all(0 <= p <= 1 for p in predictions)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_get_base_ensemble_predictions_model_failure(self):
        """Test ensemble prediction handles model failure"""
        # Create model that fails
        failing_models = {'random_forest': Mock(), 'xgboost': Mock()}
        failing_models['random_forest'].predict_proba.side_effect = ValueError("Model failed")
        
        self.manager.models = failing_models
        self.manager.is_initialized = True
        
        with pytest.raises(EnsembleError, match="Model 'random_forest' prediction failed"):
            self.manager._get_base_ensemble_predictions(self.X_test)
    
    def test_get_base_ensemble_predictions_invalid_output_shape(self):
        """Test ensemble prediction handles invalid model output shape"""
        invalid_model = Mock()
        invalid_model.predict_proba.return_value = np.array([[0.5]])  # Wrong shape
        
        self.manager.models = {'random_forest': invalid_model}
        self.manager.is_initialized = True
        
        with pytest.raises(EnsembleError, match="predict_proba returned insufficient classes"):
            self.manager._get_base_ensemble_predictions(self.X_test)
    
    def test_get_base_ensemble_predictions_nan_predictions(self):
        """Test ensemble prediction handles NaN predictions"""
        nan_model = Mock()
        nan_model.predict_proba.return_value = np.array([[0.5, np.nan]] * len(self.X_test))
        
        self.manager.models = {'random_forest': nan_model}
        self.manager.is_initialized = True
        
        with pytest.raises(EnsembleError, match="returned invalid predictions.*NaN"):
            self.manager._get_base_ensemble_predictions(self.X_test)
    
    def test_get_base_ensemble_predictions_out_of_range(self):
        """Test ensemble prediction handles out-of-range predictions"""
        out_of_range_model = Mock()
        out_of_range_model.predict_proba.return_value = np.array([[0.5, 1.5]] * len(self.X_test))
        
        self.manager.models = {'random_forest': out_of_range_model}
        self.manager.is_initialized = True
        
        with pytest.raises(EnsembleError, match="predictions outside.*range"):
            self.manager._get_base_ensemble_predictions(self.X_test)
    
    def test_calculate_data_quality_score(self):
        """Test data quality score calculation"""
        # Perfect data
        perfect_row = pd.Series([1, 2, 3, 4, 5])
        quality = self.manager._calculate_data_quality_score(perfect_row)
        assert 0.8 <= quality <= 1.0  # Should be high quality
        
        # Data with missing values
        missing_row = pd.Series([1, None, 3, 4, 5])
        quality_missing = self.manager._calculate_data_quality_score(missing_row)
        assert quality_missing < quality  # Should be lower quality
        
        # Data with extreme values
        extreme_row = pd.Series([1, 2, 3, 4, 1000])
        quality_extreme = self.manager._calculate_data_quality_score(extreme_row)
        assert 0.1 <= quality_extreme <= 1.0  # Should be within valid range
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        # Should not raise error under normal conditions
        self.manager._check_memory_usage("test")
        
        # Test with mock high memory usage
        with patch.object(self.manager.memory_monitor, 'memory_info') as mock_memory:
            # Simulate memory exceeding limit
            mock_memory.return_value.rss = self.config.max_memory_mb * 1024 * 1024 * 2  # 2x limit
            
            with pytest.raises(EnsembleError, match="Memory limit exceeded"):
                self.manager._check_memory_usage("test")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Initially no metrics
        metrics = self.manager.get_performance_metrics()
        assert metrics['total_predictions'] == 0
        assert metrics['total_errors'] == 0
        assert metrics['error_rate'] == 0
        
        # After successful prediction
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        self.manager.load_models(model_paths)
        
        self.manager.predict_fights(self.X_test, self.fighter_pairs, enable_bootstrap=False)
        
        metrics = self.manager.get_performance_metrics()
        assert metrics['total_predictions'] == len(self.fighter_pairs)
        assert metrics['avg_processing_time_ms'] > 0
        assert 'memory_usage_mb' in metrics
        assert 'models_loaded' in metrics
        assert 'configuration' in metrics
    
    def test_model_consistency_validation(self):
        """Test model consistency validation"""
        models = self._create_dummy_models()
        model_paths = self._save_models_to_temp(models)
        self.manager.load_models(model_paths)
        
        consistency = self.manager.validate_model_consistency(self.X_test)
        
        assert len(consistency) == len(self.weights)
        for model_name in self.weights.keys():
            assert model_name in consistency
            assert 'max_difference' in consistency[model_name]
            assert 'mean_difference' in consistency[model_name]
            assert 'is_consistent' in consistency[model_name]
            # Deterministic models should be consistent
            assert consistency[model_name]['is_consistent']
    
    def test_model_consistency_with_inconsistent_model(self):
        """Test model consistency with inconsistent model"""
        # Create inconsistent model
        class InconsistentModel:
            def __init__(self):
                self.call_count = 0
            
            def predict_proba(self, X):
                self.call_count += 1
                # Return different results each call
                prob_1 = np.random.rand(len(X))
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
        
        inconsistent_models = {'random_forest': InconsistentModel()}
        self.manager.models = inconsistent_models
        self.manager.is_initialized = True
        
        consistency = self.manager.validate_model_consistency(self.X_test)
        
        # Should detect inconsistency
        assert not consistency['random_forest']['is_consistent']
        assert consistency['random_forest']['max_difference'] > 1e-10


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_production_ensemble_config(self):
        """Test production config factory"""
        weights = {'model1': 0.7, 'model2': 0.3}
        config = create_production_ensemble_config(
            model_weights=weights,
            bootstrap_samples=200,
            max_memory_mb=8192,
            n_jobs=4
        )
        
        assert isinstance(config, EnsembleConfig)
        assert config.model_weights == weights
        assert config.bootstrap_samples == 200
        assert config.max_memory_mb == 8192
        assert config.n_jobs == 4


class TestPredictionResult:
    """Test PredictionResult structure"""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation and attributes"""
        result = PredictionResult(
            fighter_a="Jon Jones",
            fighter_b="Daniel Cormier",
            ensemble_probability=0.65,
            confidence_interval=(0.55, 0.75),
            uncertainty_score=0.20,
            model_breakdown={'rf': 0.6, 'xgb': 0.7},
            data_quality_score=0.85,
            processing_time_ms=150.5
        )
        
        assert result.fighter_a == "Jon Jones"
        assert result.fighter_b == "Daniel Cormier"
        assert result.ensemble_probability == 0.65
        assert result.confidence_interval == (0.55, 0.75)
        assert result.uncertainty_score == 0.20
        assert result.model_breakdown == {'rf': 0.6, 'xgb': 0.7}
        assert result.data_quality_score == 0.85
        assert result.processing_time_ms == 150.5


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    def test_complete_prediction_workflow(self):
        """Test complete prediction workflow from start to finish"""
        # Setup
        weights = {'random_forest': 0.6, 'xgboost': 0.4}
        config = create_production_ensemble_config(weights, bootstrap_samples=10, n_jobs=1)
        manager = ProductionEnsembleManager(config)
        
        # Create test data
        X_test = pd.DataFrame(np.random.randn(5, 10))
        fighter_pairs = [('Fighter_A', 'Fighter_B'), ('Fighter_C', 'Fighter_D'), 
                        ('Fighter_E', 'Fighter_F'), ('Fighter_G', 'Fighter_H'),
                        ('Fighter_I', 'Fighter_J')]
        
        # Create and save models
        class TestModel:
            def predict_proba(self, X):
                return np.column_stack([
                    np.random.beta(2, 3, len(X)), 
                    np.random.beta(3, 2, len(X))
                ])
        
        temp_dir = tempfile.mkdtemp()
        try:
            model_paths = {}
            for name in weights.keys():
                model = TestModel()
                path = Path(temp_dir) / f'{name}.joblib'
                joblib.dump(model, path)
                model_paths[name] = str(path)
            
            # Execute workflow
            manager.load_models(model_paths)
            results = manager.predict_fights(X_test, fighter_pairs, enable_bootstrap=False)
            
            # Validate results
            assert len(results) == 5
            assert all(isinstance(r, PredictionResult) for r in results)
            assert all(0 <= r.ensemble_probability <= 1 for r in results)
            
            # Check metrics
            metrics = manager.get_performance_metrics()
            assert metrics['total_predictions'] == 5
            assert metrics['total_errors'] == 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])