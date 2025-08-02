#!/usr/bin/env python3
"""
Unit Tests for Bootstrap Confidence Calculator

Tests the thread-safe bootstrap confidence interval implementation with
comprehensive validation of production fixes.
"""

import pytest
import numpy as np
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import psutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from ufc_predictor.utils.confidence_intervals import (
    BootstrapConfidenceCalculator, 
    ConfidenceIntervalResult,
    create_ufc_confidence_calculator,
    quick_confidence_intervals
)


class TestBootstrapConfidenceCalculator:
    """Test suite for BootstrapConfidenceCalculator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.calculator = BootstrapConfidenceCalculator(
            confidence_level=0.95,
            n_bootstrap=20,  # Small for fast tests
            n_jobs=1,
            random_state=42
        )
        
        # Create test data
        np.random.seed(42)
        self.X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Create dummy models
        self.models = self._create_test_models()
        self.weights = {'model1': 0.6, 'model2': 0.4}
    
    def _create_test_models(self):
        """Create dummy models for testing"""
        class TestModel:
            def __init__(self, bias=0.0):
                self.bias = bias
            
            def predict_proba(self, X):
                np.random.seed(42)
                n_samples = len(X)
                # Create deterministic probabilities for testing
                prob_1 = np.clip(0.5 + self.bias + 0.1 * np.random.randn(n_samples), 0.01, 0.99)
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
        
        return {
            'model1': TestModel(bias=0.1),
            'model2': TestModel(bias=-0.1)
        }
    
    def test_initialization_valid_parameters(self):
        """Test successful initialization with valid parameters"""
        calc = BootstrapConfidenceCalculator(
            confidence_level=0.95,
            n_bootstrap=100,
            n_jobs=2,
            random_state=42,
            max_memory_mb=1024
        )
        
        assert calc.confidence_level == 0.95
        assert calc.n_bootstrap == 100
        assert calc.random_state == 42
        assert calc.max_memory_mb == 1024
        assert calc.rng is not None
    
    def test_initialization_invalid_confidence_level(self):
        """Test initialization fails with invalid confidence level"""
        with pytest.raises(ValueError, match="Confidence level must be between"):
            BootstrapConfidenceCalculator(confidence_level=1.5)
        
        with pytest.raises(ValueError, match="Confidence level must be between"):
            BootstrapConfidenceCalculator(confidence_level=0.3)
    
    def test_initialization_invalid_bootstrap_samples(self):
        """Test initialization fails with invalid bootstrap samples"""
        with pytest.raises(ValueError, match="n_bootstrap must be between"):
            BootstrapConfidenceCalculator(n_bootstrap=5)
        
        with pytest.raises(ValueError, match="n_bootstrap must be between"):
            BootstrapConfidenceCalculator(n_bootstrap=20000)
    
    def test_initialization_insufficient_memory(self):
        """Test initialization fails with insufficient memory"""
        # Get available memory
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        excessive_memory = int(available_mb * 1.5)  # Request 150% of available
        
        with pytest.raises(RuntimeError, match="Requested memory.*exceeds safe limit"):
            BootstrapConfidenceCalculator(max_memory_mb=excessive_memory)
    
    def test_calculate_intervals_valid_inputs(self):
        """Test confidence interval calculation with valid inputs"""
        result = self.calculator.calculate_intervals(
            self.models, self.X_test, self.weights
        )
        
        assert isinstance(result, ConfidenceIntervalResult)
        assert len(result.predictions) == len(self.X_test)
        assert len(result.lower_bounds) == len(self.X_test)
        assert len(result.upper_bounds) == len(self.X_test)
        assert len(result.uncertainty_scores) == len(self.X_test)
        assert result.confidence_level == 0.95
        assert result.n_bootstrap == 20
        
        # Check confidence interval properties
        for i in range(len(result.predictions)):
            assert result.lower_bounds[i] <= result.predictions[i] <= result.upper_bounds[i]
            assert 0 <= result.lower_bounds[i] <= 1
            assert 0 <= result.upper_bounds[i] <= 1
    
    def test_calculate_intervals_empty_models(self):
        """Test calculation fails with empty models"""
        with pytest.raises(ValueError, match="Models dictionary cannot be empty"):
            self.calculator.calculate_intervals({}, self.X_test, {})
    
    def test_calculate_intervals_empty_data(self):
        """Test calculation fails with empty data"""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Feature matrix X cannot be empty"):
            self.calculator.calculate_intervals(self.models, empty_df, self.weights)
    
    def test_calculate_intervals_null_data(self):
        """Test calculation fails with null data"""
        null_df = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, 5, 6]})
        with pytest.raises(ValueError, match="Feature matrix X contains null values"):
            self.calculator.calculate_intervals(self.models, null_df, self.weights)
    
    def test_calculate_intervals_invalid_weights(self):
        """Test calculation fails with invalid weights"""
        invalid_weights = {'model1': 0.3, 'model2': 0.5}  # Sum to 0.8, not 1.0
        with pytest.raises(ValueError, match="Model weights must sum to 1.0"):
            self.calculator.calculate_intervals(self.models, self.X_test, invalid_weights)
    
    def test_thread_safety_sequential(self):
        """Test thread safety with sequential execution"""
        results = []
        
        for i in range(3):
            calc = BootstrapConfidenceCalculator(
                n_bootstrap=10, n_jobs=1, random_state=42
            )
            result = calc.calculate_intervals(self.models, self.X_test, self.weights)
            results.append(result.predictions)
        
        # Results should be consistent with same random seed
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10), \
                f"Results differ between runs: run 0 vs run {i}"
    
    def test_thread_safety_parallel(self):
        """Test thread safety with parallel execution"""
        def worker(worker_id):
            calc = BootstrapConfidenceCalculator(
                n_bootstrap=5, n_jobs=1, random_state=42 + worker_id
            )
            result = calc.calculate_intervals(self.models, self.X_test, self.weights)
            return result.predictions
        
        # Run multiple workers in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        # Each result should be valid (no shared state corruption)
        for i, result in enumerate(results):
            assert len(result) == len(self.X_test), \
                f"Worker {i} returned wrong length: {len(result)}"
            assert not np.any(np.isnan(result)), \
                f"Worker {i} returned NaN values"
            assert not np.any(np.isinf(result)), \
                f"Worker {i} returned Inf values"
    
    def test_memory_monitoring(self):
        """Test memory monitoring during calculation"""
        # Test with memory-constrained calculator
        calc = BootstrapConfidenceCalculator(
            n_bootstrap=10,
            n_jobs=1,
            max_memory_mb=512  # Low limit
        )
        
        # Should work with small data
        small_data = pd.DataFrame(np.random.randn(5, 3))
        result = calc.calculate_intervals(self.models, small_data, self.weights)
        assert isinstance(result, ConfidenceIntervalResult)
    
    def test_ensemble_prediction_consistency(self):
        """Test ensemble prediction consistency"""
        # Test that ensemble predictions are properly weighted
        single_model_weights = {'model1': 1.0, 'model2': 0.0}
        result_single = self.calculator.calculate_intervals(
            self.models, self.X_test, single_model_weights
        )
        
        # Get direct model1 predictions
        model1_preds = self.models['model1'].predict_proba(self.X_test)[:, 1]
        
        # Bootstrap should approximate direct prediction
        assert np.allclose(result_single.predictions, model1_preds, atol=0.1), \
            "Single model ensemble should approximate direct prediction"
    
    def test_confidence_level_coverage(self):
        """Test that confidence intervals have proper coverage"""
        # Use larger bootstrap for better statistics
        calc = BootstrapConfidenceCalculator(
            confidence_level=0.90,
            n_bootstrap=100,
            n_jobs=1,
            random_state=42
        )
        
        result = calc.calculate_intervals(self.models, self.X_test, self.weights)
        
        # Check that confidence intervals are narrower for 90% than 95%
        calc_95 = BootstrapConfidenceCalculator(
            confidence_level=0.95,
            n_bootstrap=100,
            n_jobs=1,
            random_state=42
        )
        
        result_95 = calc_95.calculate_intervals(self.models, self.X_test, self.weights)
        
        # 90% intervals should be narrower on average
        width_90 = np.mean(result.upper_bounds - result.lower_bounds)
        width_95 = np.mean(result_95.upper_bounds - result_95.lower_bounds)
        
        assert width_90 < width_95, \
            f"90% CI width ({width_90:.3f}) should be < 95% CI width ({width_95:.3f})"
    
    def test_bootstrap_worker_error_handling(self):
        """Test bootstrap worker error handling"""
        # Test with invalid model
        class BadModel:
            def predict_proba(self, X):
                raise ValueError("Model prediction failed")
        
        bad_models = {'bad_model': BadModel()}
        bad_weights = {'bad_model': 1.0}
        
        # Should fail with clear error message
        with pytest.raises(RuntimeError, match="Bootstrap sampling failed"):
            self.calculator.calculate_intervals(bad_models, self.X_test, bad_weights)
    
    def test_sequential_vs_parallel_consistency(self):
        """Test consistency between sequential and parallel execution"""
        # Sequential execution
        calc_seq = BootstrapConfidenceCalculator(
            n_bootstrap=20, n_jobs=1, random_state=42
        )
        result_seq = calc_seq.calculate_intervals(self.models, self.X_test, self.weights)
        
        # Parallel execution (if multiple cores available)
        if psutil.cpu_count() > 1:
            calc_par = BootstrapConfidenceCalculator(
                n_bootstrap=20, n_jobs=2, random_state=42
            )
            result_par = calc_par.calculate_intervals(self.models, self.X_test, self.weights)
            
            # Results should be similar (allowing for numerical differences)
            assert np.allclose(result_seq.predictions, result_par.predictions, rtol=0.1), \
                "Sequential and parallel results should be similar"
    
    def test_assessment_functionality(self):
        """Test prediction reliability assessment"""
        result = self.calculator.calculate_intervals(self.models, self.X_test, self.weights)
        
        assessment = self.calculator.assess_prediction_reliability(
            result, threshold=0.1
        )
        
        assert 'total_predictions' in assessment
        assert 'reliable_predictions' in assessment
        assert 'unreliable_predictions' in assessment
        assert 'reliability_rate' in assessment
        assert 'mean_uncertainty' in assessment
        
        assert assessment['total_predictions'] == len(self.X_test)
        assert (assessment['reliable_predictions'] + 
                assessment['unreliable_predictions']) == len(self.X_test)


class TestFactoryFunctions:
    """Test factory functions and utilities"""
    
    def test_create_ufc_confidence_calculator(self):
        """Test UFC-specific calculator factory"""
        calc = create_ufc_confidence_calculator(
            n_bootstrap=50,
            confidence_level=0.95,
            n_jobs=1
        )
        
        assert isinstance(calc, BootstrapConfidenceCalculator)
        assert calc.confidence_level == 0.95
        assert calc.n_bootstrap == 50
        assert calc.random_state == 42
    
    def test_quick_confidence_intervals(self):
        """Test quick confidence interval utility"""
        # Create test models
        class QuickModel:
            def predict_proba(self, X):
                return np.column_stack([
                    np.random.rand(len(X)),
                    np.random.rand(len(X))
                ])
        
        models = {'model': QuickModel()}
        X_test = pd.DataFrame(np.random.randn(5, 3))
        
        predictions, lower_bounds, upper_bounds = quick_confidence_intervals(
            models, X_test, n_bootstrap=10
        )
        
        assert len(predictions) == 5
        assert len(lower_bounds) == 5
        assert len(upper_bounds) == 5
        
        # Check ordering
        for i in range(5):
            assert lower_bounds[i] <= predictions[i] <= upper_bounds[i]


class TestPerformanceBenchmarks:
    """Performance benchmarks for bootstrap confidence intervals"""
    
    def test_single_prediction_performance(self):
        """Test single prediction performance benchmark"""
        calc = BootstrapConfidenceCalculator(
            n_bootstrap=25, n_jobs=1, random_state=42
        )
        
        # Single prediction data
        X_single = pd.DataFrame(np.random.randn(1, 50))
        models = {'model': TestBootstrapConfidenceCalculator()._create_test_models()['model1']}
        weights = {'model': 1.0}
        
        start_time = time.time()
        result = calc.calculate_intervals(models, X_single, weights)
        execution_time = (time.time() - start_time) * 1000
        
        # Should complete in under 2 seconds
        assert execution_time < 2000, \
            f"Single prediction took {execution_time:.0f}ms (target: <2000ms)"
        
        assert len(result.predictions) == 1
    
    def test_batch_prediction_performance(self):
        """Test batch prediction performance benchmark"""
        calc = BootstrapConfidenceCalculator(
            n_bootstrap=50, n_jobs=1, random_state=42
        )
        
        # Batch prediction data (100 samples)
        X_batch = pd.DataFrame(np.random.randn(100, 50))
        models = TestBootstrapConfidenceCalculator()._create_test_models()
        weights = {'model1': 0.6, 'model2': 0.4}
        
        start_time = time.time()
        result = calc.calculate_intervals(models, X_batch, weights)
        execution_time = (time.time() - start_time) * 1000
        
        # Should complete in under 10 seconds
        assert execution_time < 10000, \
            f"Batch prediction took {execution_time:.0f}ms (target: <10000ms)"
        
        assert len(result.predictions) == 100
    
    def test_memory_efficiency(self):
        """Test memory efficiency during bootstrap sampling"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        calc = BootstrapConfidenceCalculator(
            n_bootstrap=100, n_jobs=1, max_memory_mb=1024
        )
        
        X_test = pd.DataFrame(np.random.randn(200, 50))
        models = TestBootstrapConfidenceCalculator()._create_test_models()
        weights = {'model1': 0.6, 'model2': 0.4}
        
        result = calc.calculate_intervals(models, X_test, weights)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 512MB
        assert memory_increase < 512, \
            f"Memory increased by {memory_increase:.1f}MB (target: <512MB)"
        
        assert len(result.predictions) == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])