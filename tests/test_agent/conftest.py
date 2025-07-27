"""
Enhanced ML Pipeline Agent Test Configuration

Provides fixtures and configuration for testing the UFC Betting Agent
and Enhanced ML Pipeline components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import shared fixtures from main conftest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from conftest import (
        mock_fighter_data, mock_fight_data, mock_predictions, 
        mock_odds_data, performance_tester, error_simulator
    )
except ImportError:
    # Fallback for when shared fixtures aren't available
    @pytest.fixture
    def mock_fighter_data():
        return {"fighter": "test_data"}
    
    @pytest.fixture
    def mock_fight_data():
        return {"fight": "test_data"}
    
    @pytest.fixture
    def mock_predictions():
        return {"predictions": "test_data"}
    
    @pytest.fixture
    def mock_odds_data():
        return {"odds": "test_data"}
    
    @pytest.fixture
    def performance_tester():
        return lambda func: func()
    
    @pytest.fixture
    def error_simulator():
        return Mock()


# ============================================================================
# Agent Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_config():
    """Mock agent configuration for testing"""
    config = Mock()
    config.api.odds_api_key = "test_api_key_12345"
    config.betting.initial_bankroll = 1000.0
    config.agent.odds_storage_path = "test_odds"
    config.agent.backup_storage_path = "test_backup"
    config.agent.analysis_export_path = "test_analysis"
    config.agent.auto_execute_bets = False
    config.agent.enable_live_monitoring = False
    config.events = []
    return config


@pytest.fixture
def sample_event_config():
    """Sample event configuration for testing"""
    event = Mock()
    event.name = "UFC Test Event 300"
    event.target_fights = ["Fighter A vs Fighter B", "Fighter C vs Fighter D"]
    event.monitoring_enabled = True
    event.priority = "HIGH"
    return event


# ============================================================================
# Enhanced ML Pipeline Fixtures
# ============================================================================

@pytest.fixture
def mock_ensemble_config():
    """Mock ensemble configuration for Enhanced ML Pipeline"""
    return {
        'enable_ensemble': True,
        'model_weights': {
            'random_forest': 0.4,
            'xgboost': 0.35,
            'neural_network': 0.25
        },
        'confidence_intervals': True,
        'bootstrap_samples': 100,  # Reduced for testing
        'uncertainty_quantification': True,
        'data_quality_weighting': True
    }


@pytest.fixture
def mock_trained_models():
    """Mock trained models for testing"""
    models = {
        'random_forest': Mock(),
        'xgboost': Mock(),
        'neural_network': Mock()
    }
    
    # Configure mock predictions
    for model_name, model in models.items():
        model.predict_proba.return_value = [[0.3, 0.7]]  # Mock probability output
        model.predict.return_value = [1]  # Mock class prediction
        
    return models


@pytest.fixture
def mock_enhanced_predictor():
    """Mock enhanced ensemble predictor"""
    predictor = Mock()
    predictor.predict_with_confidence.return_value = {
        'ensemble_prediction': 0.65,
        'individual_predictions': {
            'random_forest': 0.6,
            'xgboost': 0.7,
            'neural_network': 0.65
        },
        'confidence_intervals': {'lower': 0.55, 'upper': 0.75},
        'uncertainty_score': 0.05,
        'model_weights': {'random_forest': 0.4, 'xgboost': 0.35, 'neural_network': 0.25}
    }
    return predictor


# ============================================================================
# Service Layer Fixtures
# ============================================================================

@pytest.fixture
def mock_hybrid_odds_service():
    """Mock Phase 2A hybrid odds service"""
    service = AsyncMock()
    service.fetch_event_odds.return_value = Mock(
        reconciled_data={
            "Fighter A vs Fighter B": {
                'fighter_a': 'Fighter A',
                'fighter_b': 'Fighter B',
                'fighter_a_decimal_odds': 1.8,
                'fighter_b_decimal_odds': 2.2,
                'data_sources': ['tab_australia'],
                'confidence_score': 0.85
            },
            "Fighter C vs Fighter D": {
                'fighter_a': 'Fighter C',
                'fighter_b': 'Fighter D',  
                'fighter_a_decimal_odds': 2.5,
                'fighter_b_decimal_odds': 1.6,
                'data_sources': ['odds_api', 'tab_australia'],
                'confidence_score': 0.95
            }
        },
        api_requests_used=1,
        confidence_score=0.90,
        fallback_activated=False
    )
    service.get_quota_status.return_value = {
        'quota_status': {
            'requests_used_today': 5,
            'requests_remaining_today': 11,
            'budget_remaining': 45.0
        },
        'hybrid_service_metrics': {
            'api_efficiency': 0.8
        }
    }
    service.health_check.return_value = {
        'overall_status': 'healthy',
        'components': {'quota_manager': 'healthy', 'api_client': 'healthy'}
    }
    return service


@pytest.fixture
def mock_prediction_service():
    """Mock prediction service with enhanced capabilities"""
    service = Mock()
    
    # Mock prediction result
    prediction_result = Mock()
    prediction_result.fight_key = "Fighter A vs Fighter B"
    prediction_result.fighter_a = "Fighter A"
    prediction_result.fighter_b = "Fighter B"
    prediction_result.model_prediction_a = 0.6
    prediction_result.model_prediction_b = 0.4
    prediction_result.model_favorite = "Fighter A"
    prediction_result.model_favorite_prob = 0.6
    prediction_result.predicted_method = "Decision"
    prediction_result.market_odds_a = 1.8
    prediction_result.market_odds_b = 2.2
    prediction_result.expected_value_a = 0.08
    prediction_result.expected_value_b = -0.12
    prediction_result.confidence_score = 0.75
    prediction_result.is_upset_opportunity = False
    prediction_result.is_high_confidence = True
    
    # Mock analysis result
    analysis = Mock()
    analysis.event_name = "Test Event"
    analysis.fight_predictions = [prediction_result]
    analysis.summary = {
        'total_fights': 1,
        'successful_predictions': 1,
        'failed_predictions': 0,
        'upset_opportunities': 0,
        'high_confidence_picks': 1,
        'method_breakdown': {'Decision': 1}
    }
    
    service.predict_event.return_value = analysis
    return service


@pytest.fixture
def mock_betting_service():
    """Mock betting service for recommendations"""
    service = Mock()
    
    # Mock betting recommendation
    bet = Mock()
    bet.fighter = "Fighter A"
    bet.opponent = "Fighter B" 
    bet.fight = "Fighter A vs Fighter B"
    bet.decimal_odds = 1.8
    bet.recommended_stake = 50.0
    bet.expected_value = 0.08
    bet.potential_profit = 40.0
    bet.is_upset_opportunity = False
    bet.is_high_confidence = True
    
    # Mock portfolio summary
    portfolio = Mock()
    portfolio.total_recommended_stake = 50.0
    portfolio.expected_return = 40.0
    portfolio.portfolio_ev = 0.08
    portfolio.bankroll_utilization = 0.05
    
    # Mock recommendations
    recommendations = Mock()
    recommendations.single_bets = [bet]
    recommendations.portfolio_summary = portfolio
    recommendations.bankroll_info = {'tier': 'STANDARD', 'amount': 1000.0}
    
    service.generate_betting_recommendations.return_value = recommendations
    return service


# ============================================================================
# Async Testing Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Provide event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_test_timeout():
    """Provide timeout for async tests"""
    return 30.0  # 30 second timeout for async operations


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_benchmark():
    """Enhanced performance benchmarking for ML operations"""
    def _benchmark(func, *args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return {
            'result': result,
            'execution_time': execution_time,
            'execution_time_ms': execution_time * 1000
        }
    return _benchmark


@pytest.fixture
def memory_profiler():
    """Memory usage profiling for ML operations"""
    def _profile_memory(func, *args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        return {
            'result': result,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta
        }
    return _profile_memory


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_event_analysis():
    """Sample complete event analysis result"""
    return {
        'event_name': 'UFC Test Event 300',
        'timestamp': datetime.now().isoformat(),
        'status': 'completed',
        'odds_result': {
            'status': 'success',
            'total_fights': 2,
            'csv_path': '/test/path/odds.csv',
            'api_requests_used': 1,
            'confidence_score': 0.90,
            'fallback_activated': False,
            'data_sources': ['tab_australia', 'odds_api']
        },
        'predictions_analysis': {
            'successful_predictions': 2,
            'failed_predictions': 0,
            'upset_opportunities': 0,
            'high_confidence_picks': 1
        },
        'betting_recommendations': {
            'total_bets': 1,
            'total_stake': 50.0,
            'expected_return': 40.0,
            'portfolio_ev': 0.08,
            'bankroll_utilization': 0.05
        },
        'export_path': '/test/path/analysis.json'
    }


@pytest.fixture
def large_fighter_dataset():
    """Large fighter dataset for performance testing"""
    import numpy as np
    
    n_fighters = 1000
    data = {
        'fighter_name': [f'Fighter_{i}' for i in range(n_fighters)],
        'wins': np.random.randint(0, 30, n_fighters),
        'losses': np.random.randint(0, 15, n_fighters),
        'height': np.random.normal(72, 4, n_fighters),  # inches
        'reach': np.random.normal(72, 4, n_fighters),   # inches
        'slpm': np.random.normal(4.0, 2.0, n_fighters), # strikes landed per minute
        'str_acc': np.random.normal(0.45, 0.1, n_fighters), # striking accuracy
        'td_avg': np.random.normal(2.0, 1.5, n_fighters),    # takedowns per fight
        'td_acc': np.random.normal(0.4, 0.15, n_fighters),   # takedown accuracy
        'td_def': np.random.normal(0.7, 0.15, n_fighters),   # takedown defense
        'sub_avg': np.random.normal(0.5, 0.5, n_fighters)   # submissions per fight
    }
    
    return pd.DataFrame(data)


# ============================================================================
# Mock Context Managers
# ============================================================================

@pytest.fixture
def mock_model_loading():
    """Mock model loading for testing without real models"""
    def _mock_loader():
        with patch('joblib.load') as mock_joblib, \
             patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:
            
            # Mock model loading
            mock_model = Mock()
            mock_model.predict_proba.return_value = [[0.4, 0.6]]
            mock_joblib.return_value = mock_model
            
            # Mock JSON loading
            mock_json.return_value = ['feature1', 'feature2', 'feature3']
            
            yield {
                'joblib': mock_joblib,
                'open': mock_open,
                'json': mock_json,
                'model': mock_model
            }
    
    return _mock_loader


# ============================================================================
# Test Markers and Categories
# ============================================================================

def pytest_configure(config):
    """Configure custom test markers"""
    config.addinivalue_line(
        "markers", "agent: mark test as agent-specific test"
    )
    config.addinivalue_line(
        "markers", "ensemble: mark test as ensemble ML test"
    )
    config.addinivalue_line(
        "markers", "phase2a: mark test as Phase 2A hybrid system test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as memory usage test"
    )


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def test_temp_dir():
    """Provide temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing"""
    def _create_mock_fs(files: Dict[str, str]):
        """Create mock file system with specified files and contents"""
        mock_files = {}
        for file_path, content in files.items():
            mock_files[file_path] = content
            
        def mock_exists(path):
            return str(path) in mock_files
            
        def mock_open_file(path, mode='r'):
            if str(path) in mock_files:
                from io import StringIO
                return StringIO(mock_files[str(path)])
            raise FileNotFoundError(f"Mock file not found: {path}")
        
        return {
            'exists': mock_exists,
            'open': mock_open_file,
            'files': mock_files
        }
    
    return _create_mock_fs