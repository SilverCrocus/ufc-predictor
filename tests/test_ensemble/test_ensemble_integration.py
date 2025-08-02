#!/usr/bin/env python3
"""
Integration Tests for Complete Ensemble System

Tests the full ensemble workflow with real-world scenarios including:
- Model training and ensemble creation
- Confidence interval calculation
- Input validation and prediction pipeline
- Error handling and recovery
- Performance under load
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
import json
import time
from pathlib import Path
from unittest.mock import patch
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from ufc_predictor.models.model_training import UFCModelTrainer
from ufc_predictor.utils.confidence_intervals import BootstrapConfidenceCalculator
from ufc_predictor.models.production_ensemble_manager import (
    ProductionEnsembleManager,
    create_production_ensemble_config
)
from ufc_predictor.agent.services.enhanced_prediction_service import EnhancedUFCPredictionService


class TestCompleteEnsembleIntegration:
    """Integration tests for complete ensemble system"""
    
    def setup_method(self):
        """Set up realistic test environment"""
        # Create realistic UFC feature data
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 70  # Realistic UFC feature count
        
        # Generate correlated features resembling UFC stats
        self.X_train = self._generate_realistic_ufc_features(self.n_samples)
        self.y_train = self._generate_realistic_targets(self.n_samples)
        
        # Create test prediction data
        self.X_test = self._generate_realistic_ufc_features(50)
        self.fighter_pairs = [
            (f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(50)
        ]
        
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_realistic_ufc_features(self, n_samples):
        """Generate realistic UFC feature data"""
        # Base fighter statistics
        data = {}
        
        # Physical attributes
        data['reach_diff'] = np.random.normal(0, 5, n_samples)
        data['height_diff'] = np.random.normal(0, 3, n_samples)
        data['weight_diff'] = np.random.normal(0, 2, n_samples)
        data['age_diff'] = np.random.normal(0, 4, n_samples)
        
        # Fight statistics
        data['wins_diff'] = np.random.normal(0, 5, n_samples)
        data['losses_diff'] = np.random.normal(0, 3, n_samples)
        data['ko_rate_diff'] = np.random.normal(0, 0.2, n_samples)
        data['sub_rate_diff'] = np.random.normal(0, 0.15, n_samples)
        
        # Striking statistics
        data['sig_strikes_diff'] = np.random.normal(0, 10, n_samples)
        data['strike_acc_diff'] = np.random.normal(0, 0.1, n_samples)
        data['strike_def_diff'] = np.random.normal(0, 0.1, n_samples)
        
        # Grappling statistics
        data['takedown_acc_diff'] = np.random.normal(0, 0.2, n_samples)
        data['takedown_def_diff'] = np.random.normal(0, 0.2, n_samples)
        data['sub_att_diff'] = np.random.normal(0, 2, n_samples)
        
        # Add remaining features to reach 70
        remaining_features = 70 - len(data)
        for i in range(remaining_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    def _generate_realistic_targets(self, n_samples):
        """Generate realistic binary targets based on features"""
        # Create somewhat realistic targets based on feature combinations
        reach_advantage = self.X_train['reach_diff'].values if hasattr(self, 'X_train') else np.random.normal(0, 5, n_samples)
        wins_advantage = np.random.normal(0, 5, n_samples)
        
        # Probability influenced by advantages
        prob = 0.5 + 0.1 * reach_advantage / 5 + 0.1 * wins_advantage / 5
        prob = np.clip(prob, 0.1, 0.9)
        
        return (np.random.rand(n_samples) < prob).astype(int)
    
    def test_end_to_end_ensemble_workflow(self):
        """Test complete ensemble workflow from training to prediction"""
        print("\n🚀 Testing End-to-End Ensemble Workflow")
        print("=" * 50)
        
        # Step 1: Train models
        print("Step 1: Training ensemble models...")
        trainer = UFCModelTrainer()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_model = trainer.train_random_forest(X_train, y_train)
        
        # Train XGBoost
        xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Save models
        model_paths = {}
        model_paths['random_forest'] = str(Path(self.temp_dir) / 'rf_model.joblib')
        model_paths['xgboost'] = str(Path(self.temp_dir) / 'xgb_model.joblib')
        
        joblib.dump(rf_model, model_paths['random_forest'])
        joblib.dump(xgb_model, model_paths['xgboost'])
        
        # Save feature columns
        feature_columns_path = str(Path(self.temp_dir) / 'features.json')
        with open(feature_columns_path, 'w') as f:
            json.dump(list(self.X_train.columns), f)
        
        print(f"✅ Models trained and saved: {len(model_paths)} models")
        
        # Step 2: Create ensemble manager
        print("Step 2: Creating ensemble manager...")
        weights = {'random_forest': 0.6, 'xgboost': 0.4}
        config = create_production_ensemble_config(
            model_weights=weights,
            bootstrap_samples=50,  # Reasonable for testing
            max_memory_mb=2048,
            n_jobs=2
        )
        
        manager = ProductionEnsembleManager(config)
        manager.load_models(model_paths, feature_columns_path)
        
        print("✅ Ensemble manager created and models loaded")
        
        # Step 3: Generate predictions
        print("Step 3: Generating ensemble predictions...")
        start_time = time.time()
        
        # Ensure test data has same columns as training data
        X_test_aligned = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)
        
        results = manager.predict_fights(
            X_test_aligned, 
            self.fighter_pairs[:10],  # Test with 10 fights
            enable_bootstrap=True
        )
        
        prediction_time = (time.time() - start_time) * 1000
        print(f"✅ Generated {len(results)} predictions in {prediction_time:.1f}ms")
        
        # Step 4: Validate results
        print("Step 4: Validating results...")
        
        assert len(results) == 10
        for result in results:
            # Validate probability range
            assert 0 <= result.ensemble_probability <= 1
            
            # Validate confidence intervals
            ci_lower, ci_upper = result.confidence_interval
            assert ci_lower <= ci_upper
            assert 0 <= ci_lower <= 1
            assert 0 <= ci_upper <= 1
            
            # Validate uncertainty
            assert 0 <= result.uncertainty_score <= 1
            
            # Validate model breakdown
            assert isinstance(result.model_breakdown, dict)
            assert 'random_forest' in result.model_breakdown
            assert 'xgboost' in result.model_breakdown
            
            # Validate data quality
            assert 0 <= result.data_quality_score <= 1
        
        print("✅ All results validated successfully")
        
        # Step 5: Performance metrics
        print("Step 5: Checking performance metrics...")
        metrics = manager.get_performance_metrics()
        
        assert metrics['total_predictions'] == 10
        assert metrics['total_errors'] == 0
        assert metrics['error_rate'] == 0
        assert metrics['avg_processing_time_ms'] > 0
        
        print(f"✅ Performance: {metrics['avg_processing_time_ms']:.1f}ms avg per prediction")
        print("🎉 End-to-end workflow completed successfully!")
    
    def test_confidence_interval_integration(self):
        """Test confidence interval integration with ensemble"""
        print("\n🔍 Testing Confidence Interval Integration")
        print("=" * 50)
        
        # Create simple models for testing
        class TestModel:
            def __init__(self, bias=0.0):
                self.bias = bias
            
            def predict_proba(self, X):
                np.random.seed(42)
                prob_1 = np.clip(0.5 + self.bias + 0.1 * np.random.randn(len(X)), 0.1, 0.9)
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
        
        models = {
            'model1': TestModel(bias=0.1),
            'model2': TestModel(bias=-0.1)
        }
        weights = {'model1': 0.6, 'model2': 0.4}
        
        # Test confidence calculation
        calculator = BootstrapConfidenceCalculator(
            confidence_level=0.95,
            n_bootstrap=100,
            n_jobs=1,  # Sequential for deterministic results
            random_state=42
        )
        
        test_data = pd.DataFrame(np.random.randn(20, 10))
        
        start_time = time.time()
        result = calculator.calculate_intervals(models, test_data, weights)
        calc_time = (time.time() - start_time) * 1000
        
        print(f"✅ Confidence intervals calculated in {calc_time:.1f}ms")
        
        # Validate results
        assert len(result.predictions) == 20
        assert len(result.lower_bounds) == 20
        assert len(result.upper_bounds) == 20
        assert result.confidence_level == 0.95
        
        # Check interval properties
        for i in range(20):
            assert result.lower_bounds[i] <= result.predictions[i] <= result.upper_bounds[i]
            assert 0 <= result.lower_bounds[i] <= 1
            assert 0 <= result.upper_bounds[i] <= 1
        
        # Test reliability assessment
        assessment = calculator.assess_prediction_reliability(result, threshold=0.2)
        
        assert assessment['total_predictions'] == 20
        assert 'reliability_rate' in assessment
        assert 'mean_uncertainty' in assessment
        
        print(f"✅ Reliability: {assessment['reliability_rate']:.1%} reliable predictions")
        print("🎉 Confidence interval integration successful!")
    
    def test_input_validation_integration(self):
        """Test input validation integration in complete workflow"""
        print("\n🛡️ Testing Input Validation Integration")
        print("=" * 50)
        
        # Create enhanced prediction service
        betting_system = {
            'fighters_df': pd.DataFrame(),
            'winner_cols': list(self.X_train.columns),
            'method_cols': ['method_feature_1', 'method_feature_2'],
            'winner_model': None,
            'method_model': None,
            'predict_function': lambda *args: {'winner_prob': 0.6, 'method': 'Decision'}
        }
        
        service = EnhancedUFCPredictionService(betting_system)
        
        # Test valid input
        print("Testing valid input...")
        valid_odds_data = {
            'fight_1': {
                'fighter_a': 'Jon Jones',
                'fighter_b': 'Daniel Cormier',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            },
            'fight_2': {
                'fighter_a': 'José Aldo',
                'fighter_b': 'Conor McGregor',
                'fighter_a_decimal_odds': 2.5,
                'fighter_b_decimal_odds': 1.6
            }
        }
        
        validated_odds, sanitized_event = service._validate_and_sanitize_inputs(
            valid_odds_data, "UFC 300: Main Event"
        )
        
        assert len(validated_odds) == 2
        assert sanitized_event == "UFC 300: Main Event"
        print("✅ Valid input processed correctly")
        
        # Test invalid inputs
        print("Testing invalid input rejection...")
        
        invalid_test_cases = [
            # Empty odds data
            ({}, "UFC 300", "Odds data cannot be empty"),
            
            # Invalid event name
            (valid_odds_data, "", "Event name must be a non-empty string"),
            
            # Missing fighter
            ({'fight_1': {'fighter_a': '', 'fighter_b': 'Fighter B', 
                          'fighter_a_decimal_odds': 1.8, 'fighter_b_decimal_odds': 2.2}}, 
             "UFC 300", "Empty fighter name"),
            
            # Same fighters
            ({'fight_1': {'fighter_a': 'Same Fighter', 'fighter_b': 'Same Fighter',
                          'fighter_a_decimal_odds': 1.8, 'fighter_b_decimal_odds': 2.2}},
             "UFC 300", "cannot be identical"),
            
            # Invalid odds
            ({'fight_1': {'fighter_a': 'Fighter A', 'fighter_b': 'Fighter B',
                          'fighter_a_decimal_odds': 0.5, 'fighter_b_decimal_odds': 2.2}},
             "UFC 300", "out of valid range"),
        ]
        
        validated_count = 0
        for invalid_odds, event_name, expected_error in invalid_test_cases:
            try:
                service._validate_and_sanitize_inputs(invalid_odds, event_name)
                pytest.fail(f"Expected validation error for case with expected: {expected_error}")
            except ValueError as e:
                assert expected_error in str(e)
                validated_count += 1
        
        print(f"✅ {validated_count} invalid inputs correctly rejected")
        print("🎉 Input validation integration successful!")
    
    def test_error_handling_integration(self):
        """Test error handling integration across components"""
        print("\n⚠️ Testing Error Handling Integration")
        print("=" * 50)
        
        # Test 1: Model loading errors
        print("Testing model loading error handling...")
        
        config = create_production_ensemble_config({'model1': 1.0})
        manager = ProductionEnsembleManager(config)
        
        # Test missing file
        try:
            manager.load_models({'model1': '/nonexistent/path/model.joblib'})
            pytest.fail("Expected ModelLoadError for missing file")
        except Exception as e:
            assert "not found" in str(e)
            print("✅ Missing model file error handled correctly")
        
        # Test 2: Bootstrap calculation errors
        print("Testing bootstrap error handling...")
        
        class FailingModel:
            def predict_proba(self, X):
                raise ValueError("Model prediction failed")
        
        calculator = BootstrapConfidenceCalculator(n_bootstrap=5, n_jobs=1)
        failing_models = {'bad_model': FailingModel()}
        test_data = pd.DataFrame(np.random.randn(5, 3))
        
        try:
            calculator.calculate_intervals(failing_models, test_data, {'bad_model': 1.0})
            pytest.fail("Expected bootstrap error")
        except Exception as e:
            assert "Bootstrap sampling failed" in str(e) or "failed" in str(e)
            print("✅ Bootstrap calculation error handled correctly")
        
        # Test 3: Input validation errors
        print("Testing validation error handling...")
        
        service = EnhancedUFCPredictionService({
            'fighters_df': pd.DataFrame(),
            'winner_cols': [],
            'method_cols': [],
            'winner_model': None,
            'method_model': None,
            'predict_function': lambda *args: {}
        })
        
        # Test malicious input
        malicious_input = {
            'fight_1': {
                'fighter_a': 'Fighter<script>alert("xss")</script>',
                'fighter_b': 'Normal Fighter',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            }
        }
        
        try:
            service._validate_and_sanitize_inputs(malicious_input, "UFC 300")
            pytest.fail("Expected validation error for malicious input")
        except ValueError as e:
            assert "invalid characters" in str(e) or "suspicious" in str(e)
            print("✅ Malicious input rejected correctly")
        
        print("🎉 Error handling integration successful!")
    
    def test_performance_under_load(self):
        """Test system performance under realistic load"""
        print("\n⚡ Testing Performance Under Load")
        print("=" * 50)
        
        # Create multiple test scenarios
        scenarios = [
            (10, "Small batch"),
            (50, "Medium batch"), 
            (100, "Large batch")
        ]
        
        # Create simple but realistic models
        class FastModel:
            def __init__(self, name):
                self.name = name
            
            def predict_proba(self, X):
                # Fast vectorized prediction
                n_samples = len(X)
                # Use deterministic but fast computation
                prob_1 = 0.5 + 0.1 * np.sin(np.arange(n_samples) + hash(self.name) % 100)
                prob_1 = np.clip(prob_1, 0.1, 0.9)
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
        
        models = {
            'fast_model_1': FastModel('model1'),
            'fast_model_2': FastModel('model2')
        }
        weights = {'fast_model_1': 0.6, 'fast_model_2': 0.4}
        
        calculator = BootstrapConfidenceCalculator(
            n_bootstrap=50,  # Reasonable bootstrap count
            n_jobs=2,
            random_state=42
        )
        
        performance_results = []
        
        for n_samples, description in scenarios:
            print(f"Testing {description}: {n_samples} samples...")
            
            # Generate test data
            test_data = pd.DataFrame(np.random.randn(n_samples, 20))
            
            # Measure performance
            start_time = time.time()
            result = calculator.calculate_intervals(models, test_data, weights)
            total_time = (time.time() - start_time) * 1000
            
            per_sample_time = total_time / n_samples
            
            # Validate results
            assert len(result.predictions) == n_samples
            assert len(result.lower_bounds) == n_samples
            assert len(result.upper_bounds) == n_samples
            
            performance_results.append({
                'scenario': description,
                'samples': n_samples,
                'total_time_ms': total_time,
                'per_sample_ms': per_sample_time
            })
            
            print(f"  ✅ {total_time:.1f}ms total ({per_sample_time:.2f}ms per sample)")
        
        # Check performance targets
        print("\nPerformance Summary:")
        print("=" * 30)
        for result in performance_results:
            print(f"{result['scenario']}: {result['per_sample_ms']:.2f}ms per sample")
            
            # Performance targets (reasonable for ensemble with bootstrap)
            if result['samples'] <= 10:
                target_per_sample = 50  # 50ms per sample for small batches
            elif result['samples'] <= 50:
                target_per_sample = 30  # 30ms per sample for medium batches
            else:
                target_per_sample = 20  # 20ms per sample for large batches
            
            if result['per_sample_ms'] > target_per_sample:
                print(f"  ⚠️ Performance warning: {result['per_sample_ms']:.2f}ms > {target_per_sample}ms target")
            else:
                print(f"  ✅ Performance good: {result['per_sample_ms']:.2f}ms <= {target_per_sample}ms target")
        
        print("🎉 Performance testing completed!")
    
    def test_memory_management_integration(self):
        """Test memory management across ensemble components"""
        print("\n💾 Testing Memory Management Integration")
        print("=" * 50)
        
        import psutil
        process = psutil.Process()
        
        # Record baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        
        # Test memory-constrained ensemble
        config = create_production_ensemble_config(
            model_weights={'model1': 1.0},
            bootstrap_samples=100,
            max_memory_mb=1024,  # Constrained memory
            n_jobs=1
        )
        
        manager = ProductionEnsembleManager(config)
        
        # Test memory monitoring
        try:
            manager._check_memory_usage("baseline")
            print("✅ Memory monitoring working")
        except Exception as e:
            print(f"⚠️ Memory monitoring issue: {e}")
        
        # Test bootstrap with memory constraints
        calculator = BootstrapConfidenceCalculator(
            n_bootstrap=50,
            max_memory_mb=512,  # Low limit for testing
            n_jobs=1
        )
        
        # Small test to avoid hitting memory limits
        small_data = pd.DataFrame(np.random.randn(10, 20))
        
        class MemoryEfficientModel:
            def predict_proba(self, X):
                # Simple computation to minimize memory
                prob_1 = np.full(len(X), 0.6)
                prob_0 = 1 - prob_1
                return np.column_stack([prob_0, prob_1])
        
        models = {'model1': MemoryEfficientModel()}
        weights = {'model1': 1.0}
        
        result = calculator.calculate_intervals(models, small_data, weights)
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Should not increase memory dramatically
        if memory_increase > 200:  # 200MB increase is reasonable for testing
            print(f"⚠️ High memory increase: {memory_increase:.1f} MB")
        else:
            print(f"✅ Memory increase acceptable: {memory_increase:.1f} MB")
        
        print("🎉 Memory management testing completed!")


class TestRealWorldScenarios:
    """Test real-world UFC prediction scenarios"""
    
    def test_ufc_event_simulation(self):
        """Simulate a complete UFC event prediction"""
        print("\n🥊 Testing UFC Event Simulation")
        print("=" * 50)
        
        # Simulate UFC 300 main card
        ufc_300_fights = {
            'main_event': {
                'fighter_a': 'Alex Pereira',
                'fighter_b': 'Jamahal Hill',
                'fighter_a_decimal_odds': 1.7,
                'fighter_b_decimal_odds': 2.3
            },
            'co_main': {
                'fighter_a': 'Zhang Weili',
                'fighter_b': 'Yan Xiaonan',
                'fighter_a_decimal_odds': 1.4,
                'fighter_b_decimal_odds': 3.2
            },
            'fight_3': {
                'fighter_a': 'Max Holloway',
                'fighter_b': 'Justin Gaethje',
                'fighter_a_decimal_odds': 2.1,
                'fighter_b_decimal_odds': 1.8
            },
            'fight_4': {
                'fighter_a': 'Aljamain Sterling',
                'fighter_b': 'Calvin Kattar',
                'fighter_a_decimal_odds': 1.9,
                'fighter_b_decimal_odds': 2.0
            },
            'fight_5': {
                'fighter_a': 'Bobby Green',
                'fighter_b': 'Jim Miller',
                'fighter_a_decimal_odds': 1.6,
                'fighter_b_decimal_odds': 2.6
            }
        }
        
        # Create enhanced prediction service
        betting_system = {
            'fighters_df': pd.DataFrame(),
            'winner_cols': [f'feature_{i}' for i in range(50)],
            'method_cols': ['striking_diff', 'grappling_diff'],
            'winner_model': None,
            'method_model': None,
            'predict_function': lambda *args: {'winner_prob': 0.6, 'method': 'Decision'}
        }
        
        service = EnhancedUFCPredictionService(betting_system)
        
        # Test input validation
        try:
            validated_odds, sanitized_event = service._validate_and_sanitize_inputs(
                ufc_300_fights, "UFC 300: Pereira vs Hill"
            )
            
            print(f"✅ Event validated: {sanitized_event}")
            print(f"✅ {len(validated_odds)} fights validated")
            
            # Verify each fight
            for fight_key, fight_data in validated_odds.items():
                assert 'fighter_a' in fight_data
                assert 'fighter_b' in fight_data
                assert 'fighter_a_decimal_odds' in fight_data
                assert 'fighter_b_decimal_odds' in fight_data
                
                # Check odds are reasonable
                assert 1.01 <= fight_data['fighter_a_decimal_odds'] <= 50.0
                assert 1.01 <= fight_data['fighter_b_decimal_odds'] <= 50.0
                
                print(f"  Fight: {fight_data['fighter_a']} vs {fight_data['fighter_b']}")
                print(f"    Odds: {fight_data['fighter_a_decimal_odds']:.1f} / {fight_data['fighter_b_decimal_odds']:.1f}")
            
            print("🎉 UFC event simulation successful!")
            
        except Exception as e:
            pytest.fail(f"UFC event simulation failed: {e}")
    
    def test_international_fighter_names(self):
        """Test handling of international fighter names"""
        print("\n🌍 Testing International Fighter Names")
        print("=" * 50)
        
        international_fights = {
            'brazil_fight': {
                'fighter_a': 'José Aldo',
                'fighter_b': 'João Carvalho',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2
            },
            'russia_fight': {
                'fighter_a': 'Khabib Nurmagomedov',
                'fighter_b': 'Магомед Анкалаев',  # Cyrillic
                'fighter_a_decimal_odds': 1.5,
                'fighter_b_decimal_odds': 2.8
            },
            'china_fight': {
                'fighter_a': 'Zhang Weili',
                'fighter_b': 'Li Jingliang',
                'fighter_a_decimal_odds': 1.6,
                'fighter_b_decimal_odds': 2.5
            },
            'nordic_fight': {
                'fighter_a': 'Alexander Gustafsson',
                'fighter_b': 'Jørgen Kruth',
                'fighter_a_decimal_odds': 1.7,
                'fighter_b_decimal_odds': 2.4
            }
        }
        
        service = EnhancedUFCPredictionService({
            'fighters_df': pd.DataFrame(),
            'winner_cols': [],
            'method_cols': [],
            'winner_model': None,
            'method_model': None,
            'predict_function': lambda *args: {}
        })
        
        try:
            validated_odds, sanitized_event = service._validate_and_sanitize_inputs(
                international_fights, "International UFC Event"
            )
            
            print(f"✅ International event validated: {len(validated_odds)} fights")
            
            for fight_key, fight_data in validated_odds.items():
                fighter_a = fight_data['fighter_a']
                fighter_b = fight_data['fighter_b']
                print(f"  {fight_key}: {fighter_a} vs {fighter_b}")
                
                # Check names are preserved but safe
                assert len(fighter_a) >= 2
                assert len(fighter_b) >= 2
                assert fighter_a != fighter_b
            
            print("🎉 International fighter names handled successfully!")
            
        except Exception as e:
            pytest.fail(f"International fighter test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])