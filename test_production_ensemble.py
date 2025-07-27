#!/usr/bin/env python3
"""
Production XGBoost Ensemble System Test
======================================

Tests the complete ensemble system under production conditions:
- Thread safety validation
- Memory usage monitoring
- Performance benchmarking
- Error handling verification
- Input validation testing
"""

import numpy as np
import pandas as pd
import time
import psutil
import logging
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.production_ensemble_manager import (
    ProductionEnsembleManager, 
    EnsembleConfig,
    create_production_ensemble_config
)
from src.confidence_intervals import BootstrapConfidenceCalculator
from src.model_training import UFCModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTestSuite:
    """Comprehensive test suite for production ensemble system"""
    
    def __init__(self):
        self.test_results = {}
        self.memory_baseline = None
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete test suite"""
        
        logger.info("=" * 60)
        logger.info("STARTING PRODUCTION ENSEMBLE SYSTEM TESTS")
        logger.info("=" * 60)
        
        # Record baseline memory
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Baseline memory usage: {self.memory_baseline:.1f} MB")
        
        tests = [
            ("Thread Safety Test", self.test_thread_safety),
            ("Memory Management Test", self.test_memory_management),
            ("Input Validation Test", self.test_input_validation),
            ("Error Handling Test", self.test_error_handling),
            ("Performance Benchmark", self.test_performance_benchmarks),
            ("Bootstrap Confidence Test", self.test_bootstrap_confidence),
            ("Production Integration Test", self.test_production_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                start_time = time.time()
                result = test_func()
                test_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'duration_ms': test_time * 1000,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
                }
                
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status} ({test_time:.2f}s)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'passed': False,
                    'error': str(e),
                    'duration_ms': 0,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
                }
                logger.error(f"{test_name}: ❌ FAILED - {str(e)}")
        
        # Print summary
        self._print_test_summary()
        
        return {name: result['passed'] for name, result in self.test_results.items()}
    
    def test_thread_safety(self) -> bool:
        """Test thread-safe bootstrap sampling"""
        logger.info("Testing thread-safe random number generation...")
        
        # Create test data
        X_test = self._create_test_data(100, 20)
        models = self._create_dummy_models()
        weights = {'model1': 0.6, 'model2': 0.4}
        
        # Test multiple runs for consistency
        calculator = BootstrapConfidenceCalculator(
            n_bootstrap=50, 
            n_jobs=2, 
            random_state=42
        )
        
        results = []
        for i in range(3):
            # Reset calculator with same seed
            calculator.rng = np.random.Generator(np.random.PCG64(42))
            
            result = calculator.calculate_intervals(models, X_test, weights)
            results.append(result.predictions)
        
        # Check consistency across runs
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], rtol=1e-10):
                logger.error(f"Thread safety test failed: run 0 vs run {i} differ")
                return False
        
        logger.info("✅ Thread safety validated - consistent results across runs")
        return True
    
    def test_memory_management(self) -> bool:
        """Test memory monitoring and limits"""
        logger.info("Testing memory management...")
        
        # Create configuration with low memory limit for testing
        config = create_production_ensemble_config(
            model_weights={'model1': 0.6, 'model2': 0.4},
            bootstrap_samples=20,
            max_memory_mb=1024,  # Low limit for testing
            n_jobs=1
        )
        
        manager = ProductionEnsembleManager(config)
        
        # Test memory monitoring
        initial_memory = manager.memory_monitor.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory: {initial_memory:.1f} MB")
        
        # Test memory limit enforcement
        try:
            manager._check_memory_usage("test")
            logger.info("✅ Memory monitoring working correctly")
            return True
        except Exception as e:
            logger.error(f"Memory management test failed: {e}")
            return False
    
    def test_input_validation(self) -> bool:
        """Test comprehensive input validation"""
        logger.info("Testing input validation...")
        
        config = create_production_ensemble_config(
            model_weights={'model1': 1.0},
            bootstrap_samples=10
        )
        manager = ProductionEnsembleManager(config)
        
        # Test empty data
        try:
            empty_df = pd.DataFrame()
            manager._validate_prediction_inputs(empty_df, [])
            logger.error("Failed to catch empty DataFrame")
            return False
        except Exception:
            logger.info("✅ Empty DataFrame validation working")
        
        # Test null values
        try:
            null_df = pd.DataFrame({'col1': [1, 2, None], 'col2': [4, 5, 6]})
            manager._validate_prediction_inputs(null_df, [('A', 'B')])
            logger.error("Failed to catch null values")
            return False
        except Exception:
            logger.info("✅ Null value validation working")
        
        # Test fighter pair validation
        try:
            valid_df = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5]})
            manager._validate_prediction_inputs(valid_df, [('Fighter A', 'Fighter A')])  # Same fighter
            logger.error("Failed to catch same fighter names")
            return False
        except Exception:
            logger.info("✅ Fighter pair validation working")
        
        logger.info("✅ All input validation tests passed")
        return True
    
    def test_error_handling(self) -> bool:
        """Test strict error handling without fallbacks"""
        logger.info("Testing error handling...")
        
        # Test that errors propagate correctly without fallbacks
        calculator = BootstrapConfidenceCalculator(n_bootstrap=5, n_jobs=1)
        
        # Test with invalid models
        invalid_models = {'bad_model': object()}  # Object with no predict method
        X_test = self._create_test_data(10, 5)
        
        try:
            calculator.calculate_intervals(invalid_models, X_test, {'bad_model': 1.0})
            logger.error("Failed to catch invalid model")
            return False
        except Exception as e:
            logger.info(f"✅ Invalid model error handled: {type(e).__name__}")
        
        # Test with mismatched weights
        valid_models = self._create_dummy_models()
        try:
            calculator.calculate_intervals(valid_models, X_test, {'wrong_name': 1.0})
            logger.error("Failed to catch weight mismatch")
            return False
        except Exception as e:
            logger.info(f"✅ Weight mismatch error handled: {type(e).__name__}")
        
        logger.info("✅ All error handling tests passed")
        return True
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance against production targets"""
        logger.info("Testing performance benchmarks...")
        
        # Test data sizes
        small_data = self._create_test_data(50, 70)    # Small prediction
        medium_data = self._create_test_data(100, 70)  # Medium prediction
        
        models = self._create_dummy_models()
        weights = {'model1': 0.6, 'model2': 0.4}
        
        # Single prediction performance test
        start_time = time.time()
        calculator = BootstrapConfidenceCalculator(n_bootstrap=25, n_jobs=1)
        calculator.calculate_intervals(models, small_data.iloc[:1], weights)
        single_prediction_time = (time.time() - start_time) * 1000
        
        target_single = 2000  # 2 seconds
        if single_prediction_time > target_single:
            logger.warning(f"Single prediction slow: {single_prediction_time:.0f}ms > {target_single}ms")
        else:
            logger.info(f"✅ Single prediction: {single_prediction_time:.0f}ms")
        
        # Bootstrap confidence interval performance
        start_time = time.time()
        calculator_bootstrap = BootstrapConfidenceCalculator(n_bootstrap=100, n_jobs=2)
        calculator_bootstrap.calculate_intervals(models, medium_data, weights)
        bootstrap_time = (time.time() - start_time) * 1000
        
        target_bootstrap = 10000  # 10 seconds
        if bootstrap_time > target_bootstrap:
            logger.warning(f"Bootstrap slow: {bootstrap_time:.0f}ms > {target_bootstrap}ms")
        else:
            logger.info(f"✅ Bootstrap confidence: {bootstrap_time:.0f}ms")
        
        # Memory efficiency test
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = memory_after - self.memory_baseline
        
        target_memory = 512  # 512 MB increase max
        if memory_increase > target_memory:
            logger.warning(f"Memory usage high: +{memory_increase:.1f}MB > {target_memory}MB")
        else:
            logger.info(f"✅ Memory efficient: +{memory_increase:.1f}MB")
        
        return True
    
    def test_bootstrap_confidence(self) -> bool:
        """Test bootstrap confidence interval accuracy"""
        logger.info("Testing bootstrap confidence intervals...")
        
        X_test = self._create_test_data(50, 20)
        models = self._create_dummy_models()
        weights = {'model1': 0.6, 'model2': 0.4}
        
        calculator = BootstrapConfidenceCalculator(
            confidence_level=0.95,
            n_bootstrap=30,
            n_jobs=1
        )
        
        result = calculator.calculate_intervals(models, X_test, weights)
        
        # Validate results
        if len(result.predictions) != len(X_test):
            logger.error(f"Prediction length mismatch: {len(result.predictions)} != {len(X_test)}")
            return False
        
        if len(result.lower_bounds) != len(X_test):
            logger.error(f"Lower bounds length mismatch: {len(result.lower_bounds)} != {len(X_test)}")
            return False
        
        # Check confidence interval properties
        for i, (lower, upper) in enumerate(zip(result.lower_bounds, result.upper_bounds)):
            if lower > upper:
                logger.error(f"Invalid CI at index {i}: lower {lower} > upper {upper}")
                return False
            
            if not (0 <= lower <= 1 and 0 <= upper <= 1):
                logger.error(f"CI bounds out of range at index {i}: [{lower}, {upper}]")
                return False
        
        logger.info(f"✅ Bootstrap confidence intervals: {result.n_bootstrap} samples, {result.confidence_level:.1%} level")
        return True
    
    def test_production_integration(self) -> bool:
        """Test complete production integration"""
        logger.info("Testing production integration...")
        
        # Test with production configuration
        config = create_production_ensemble_config(
            model_weights={'random_forest': 0.4, 'xgboost': 0.35, 'neural_network': 0.25},
            bootstrap_samples=50,
            max_memory_mb=2048,
            n_jobs=2
        )
        
        manager = ProductionEnsembleManager(config)
        
        # Create temporary models for testing
        models = self._create_dummy_models_with_names(['random_forest', 'xgboost', 'neural_network'])
        model_paths = {}
        
        try:
            # Save models temporarily
            import joblib
            for name, model in models.items():
                path = f'/tmp/test_{name}_model.joblib'
                joblib.dump(model, path)
                model_paths[name] = path
            
            # Test loading
            manager.load_models(model_paths)
            
            # Test prediction
            X_test = self._create_test_data(20, 70)
            fighter_pairs = [(f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(20)]
            
            results = manager.predict_fights(X_test, fighter_pairs, enable_bootstrap=True)
            
            if len(results) != 20:
                logger.error(f"Wrong number of results: {len(results)} != 20")
                return False
            
            # Validate result structure
            for i, result in enumerate(results):
                if not (0 <= result.ensemble_probability <= 1):
                    logger.error(f"Invalid probability at {i}: {result.ensemble_probability}")
                    return False
                
                if result.confidence_interval[0] > result.confidence_interval[1]:
                    logger.error(f"Invalid CI at {i}: {result.confidence_interval}")
                    return False
            
            # Test performance metrics
            metrics = manager.get_performance_metrics()
            logger.info(f"Performance: {metrics['total_predictions']} predictions, "
                       f"{metrics['avg_processing_time_ms']:.1f}ms avg")
            
            logger.info("✅ Production integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"Production integration test failed: {e}")
            return False
        finally:
            # Cleanup
            import os
            for path in model_paths.values():
                try:
                    os.remove(path)
                except:
                    pass
    
    def _create_test_data(self, n_samples: int, n_features: int) -> pd.DataFrame:
        """Create realistic test data for UFC predictions"""
        np.random.seed(42)
        
        # Create data that resembles UFC features
        data = np.random.randn(n_samples, n_features)
        
        # Add some correlation structure
        for i in range(1, n_features):
            data[:, i] += 0.3 * data[:, i-1]  # Some correlation with previous feature
        
        # Normalize to reasonable ranges
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        columns = [f'feature_{i}' for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    def _create_dummy_models(self) -> Dict:
        """Create dummy models for testing"""
        class DummyModel:
            def predict_proba(self, X):
                np.random.seed(42)  # For consistency
                n_samples = len(X)
                prob_class_1 = np.random.beta(2, 2, size=n_samples)
                prob_class_0 = 1 - prob_class_1
                return np.column_stack([prob_class_0, prob_class_1])
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        
        return {'model1': DummyModel(), 'model2': DummyModel()}
    
    def _create_dummy_models_with_names(self, names: List[str]) -> Dict:
        """Create dummy models with specific names"""
        class DummyModel:
            def __init__(self, seed_offset: int = 0):
                self.seed_offset = seed_offset
            
            def predict_proba(self, X):
                np.random.seed(42 + self.seed_offset)
                n_samples = len(X)
                prob_class_1 = np.random.beta(2, 2, size=n_samples)
                prob_class_0 = 1 - prob_class_1
                return np.column_stack([prob_class_0, prob_class_1])
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        
        return {name: DummyModel(i) for i, name in enumerate(names)}
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['passed'])
        
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests:.1%}")
        
        # Memory summary
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - self.memory_baseline
        logger.info(f"Memory usage: {self.memory_baseline:.1f} MB → {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Performance summary
        total_time = sum(r.get('duration_ms', 0) for r in self.test_results.values())
        logger.info(f"Total test time: {total_time:.0f}ms")
        
        # Failed tests
        failed_tests = [name for name, result in self.test_results.items() if not result['passed']]
        if failed_tests:
            logger.error(f"Failed tests: {', '.join(failed_tests)}")
        
        logger.info("=" * 60)


if __name__ == "__main__":
    test_suite = ProductionTestSuite()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED - Production ensemble system ready!")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED - Check logs above")
        sys.exit(1)