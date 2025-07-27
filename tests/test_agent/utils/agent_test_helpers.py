"""
Agent Test Helpers

Specialized utilities for testing UFC Betting Agent and Enhanced ML Pipeline components.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta


class AgentTestHelpers:
    """Helper utilities for agent testing"""
    
    @staticmethod
    def create_mock_betting_system(model_version: str = "test_v1.0") -> Dict[str, Any]:
        """Create mock betting system for testing"""
        mock_fighter_df = pd.DataFrame({
            'fighter_name': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
            'wins': [10, 8, 12, 5],
            'losses': [2, 3, 1, 4],
            'height': [72, 70, 74, 68],
            'reach': [72, 70, 74, 68],
            'slpm': [4.5, 3.2, 5.1, 2.8],
            'str_acc': [0.45, 0.52, 0.41, 0.38],
            'td_avg': [2.1, 0.8, 3.2, 1.5],
            'td_acc': [0.4, 0.3, 0.5, 0.35],
            'td_def': [0.7, 0.8, 0.6, 0.75],
            'sub_avg': [0.3, 0.1, 0.8, 0.2]
        })
        
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        mock_model.predict.return_value = np.array([1])
        
        winner_cols = ['height_diff', 'reach_diff', 'slpm_diff', 'str_acc_diff']
        method_cols = ['td_avg_diff', 'td_acc_diff', 'sub_avg_diff']
        
        def mock_predict_function(fighter_a, fighter_b, fighters_df, winner_cols, 
                                method_cols, winner_model, method_model):
            return {
                'win_probabilities': {
                    fighter_a: '60.5%',
                    fighter_b: '39.5%'
                },
                'predicted_method': 'Decision',
                'confidence': 0.75,
                'symmetrical_average': True
            }
        
        return {
            'fighters_df': mock_fighter_df,
            'winner_cols': winner_cols,
            'method_cols': method_cols,
            'winner_model': mock_model,
            'method_model': mock_model,
            'predict_function': mock_predict_function,
            'model_version': model_version,
            'bankroll': 1000.0,
            'api_key': 'test_api_key'
        }
    
    @staticmethod
    def create_mock_odds_data(fights: List[str]) -> Dict[str, Dict]:
        """Create mock odds data for testing"""
        odds_data = {}
        
        for i, fight in enumerate(fights):
            fighters = fight.split(' vs ')
            if len(fighters) != 2:
                continue
                
            fighter_a, fighter_b = fighters
            
            # Generate realistic odds
            favorite_odds = 1.4 + (i * 0.2)  # 1.4, 1.6, 1.8, etc.
            underdog_odds = 2.8 - (i * 0.1)  # 2.8, 2.7, 2.6, etc.
            
            odds_data[fight] = {
                'fighter_a': fighter_a,
                'fighter_b': fighter_b,
                'fighter_a_decimal_odds': favorite_odds,
                'fighter_b_decimal_odds': underdog_odds,
                'bookmakers': ['TAB Australia'],
                'commence_time': (datetime.now() + timedelta(days=7)).isoformat(),
                'api_timestamp': datetime.now().isoformat(),
                'source': 'test_data'
            }
        
        return odds_data
    
    @staticmethod
    def create_mock_hybrid_result(fights: List[str], api_requests_used: int = 1) -> Mock:
        """Create mock hybrid odds service result"""
        result = Mock()
        result.reconciled_data = AgentTestHelpers.create_mock_odds_data(fights)
        result.api_requests_used = api_requests_used
        result.confidence_score = 0.85 + (api_requests_used * 0.05)  # Higher confidence with API usage
        result.fallback_activated = api_requests_used == 0
        result.primary_source = Mock()
        result.primary_source.source_name = 'tab_australia' if api_requests_used == 0 else 'odds_api'
        result.validation_source = Mock() if api_requests_used > 0 else None
        
        return result
    
    @staticmethod
    def assert_prediction_quality(predictions: List[Any], min_confidence: float = 0.6):
        """Assert prediction quality meets minimum standards"""
        assert len(predictions) > 0, "No predictions generated"
        
        for prediction in predictions:
            # Check confidence scores
            assert hasattr(prediction, 'confidence_score'), "Missing confidence score"
            assert prediction.confidence_score >= min_confidence, \
                f"Low confidence: {prediction.confidence_score} < {min_confidence}"
            
            # Check probability bounds
            assert 0 <= prediction.model_prediction_a <= 1, "Invalid probability for fighter A"
            assert 0 <= prediction.model_prediction_b <= 1, "Invalid probability for fighter B"
            assert abs((prediction.model_prediction_a + prediction.model_prediction_b) - 1.0) < 0.01, \
                "Probabilities don't sum to 1"
    
    @staticmethod
    def assert_betting_recommendations_valid(recommendations: Any, max_bankroll_usage: float = 0.25):
        """Assert betting recommendations are valid and safe"""
        assert hasattr(recommendations, 'single_bets'), "Missing single bets"
        assert hasattr(recommendations, 'portfolio_summary'), "Missing portfolio summary"
        
        portfolio = recommendations.portfolio_summary
        
        # Check bankroll management
        assert portfolio.bankroll_utilization <= max_bankroll_usage, \
            f"Excessive bankroll usage: {portfolio.bankroll_utilization} > {max_bankroll_usage}"
        
        # Check expected value
        assert portfolio.portfolio_ev > 0, f"Negative expected value: {portfolio.portfolio_ev}"
        
        # Check individual bets
        for bet in recommendations.single_bets:
            assert bet.expected_value > 0, f"Negative EV bet: {bet.expected_value}"
            assert bet.recommended_stake > 0, f"Invalid stake: {bet.recommended_stake}"
    
    @staticmethod
    def simulate_quota_exhaustion(hybrid_service: AsyncMock):
        """Simulate API quota exhaustion for testing fallback behavior"""
        # Configure quota status to show exhaustion
        hybrid_service.get_quota_status.return_value = {
            'quota_status': {
                'requests_used_today': 16,
                'requests_remaining_today': 0,
                'budget_remaining': 5.0
            },
            'hybrid_service_metrics': {
                'api_efficiency': 0.0
            }
        }
        
        # Configure fetch to use fallback only
        hybrid_service.fetch_event_odds.return_value = AgentTestHelpers.create_mock_hybrid_result(
            ["Fighter A vs Fighter B"], api_requests_used=0
        )
    
    @staticmethod
    def create_performance_test_data(n_fights: int = 100) -> Tuple[List[str], Dict[str, Dict]]:
        """Create large dataset for performance testing"""
        fights = []
        for i in range(n_fights):
            fight = f"Fighter_{i*2} vs Fighter_{i*2+1}"
            fights.append(fight)
        
        odds_data = AgentTestHelpers.create_mock_odds_data(fights)
        return fights, odds_data


class AsyncTestHelpers:
    """Helpers for async testing"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 30.0):
        """Run async coroutine with timeout"""
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise AssertionError(f"Async operation timed out after {timeout} seconds")
    
    @staticmethod
    def create_async_mock_chain(*methods):
        """Create chain of async mocks for complex workflows"""
        mock = AsyncMock()
        current = mock
        
        for method in methods[:-1]:
            setattr(current, method, AsyncMock())
            current = getattr(current, method)
            
        # Last method returns a result
        if methods:
            setattr(current, methods[-1], AsyncMock(return_value={'status': 'success'}))
        
        return mock


class PerformanceTestHelpers:
    """Helpers for performance testing"""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time"""
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(coro) -> Tuple[Any, float]:
        """Measure async coroutine execution time"""
        import time
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def assert_performance_target(execution_time: float, target_time: float, operation_name: str):
        """Assert performance meets target"""
        assert execution_time <= target_time, \
            f"{operation_name} took {execution_time:.3f}s, target was {target_time:.3f}s"
    
    @staticmethod
    def profile_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile memory usage of function"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        memory_info = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta
        }
        
        return result, memory_info


class DataValidationHelpers:
    """Helpers for data validation testing"""
    
    @staticmethod
    def validate_ufc_fighter_data(fighter_data: pd.DataFrame):
        """Validate UFC fighter data meets expected constraints"""
        required_columns = ['fighter_name', 'wins', 'losses', 'height', 'reach']
        
        for col in required_columns:
            assert col in fighter_data.columns, f"Missing required column: {col}"
        
        # Validate data ranges
        assert fighter_data['wins'].min() >= 0, "Negative wins found"
        assert fighter_data['losses'].min() >= 0, "Negative losses found"
        assert fighter_data['height'].between(60, 84).all(), "Invalid height values"
        assert fighter_data['reach'].between(60, 90).all(), "Invalid reach values"
    
    @staticmethod
    def validate_odds_data(odds_data: Dict[str, Dict]):
        """Validate odds data structure and values"""
        for fight_key, fight_data in odds_data.items():
            # Check required fields
            required_fields = ['fighter_a', 'fighter_b', 'fighter_a_decimal_odds', 'fighter_b_decimal_odds']
            for field in required_fields:
                assert field in fight_data, f"Missing field {field} in {fight_key}"
            
            # Check odds are valid
            odds_a = fight_data['fighter_a_decimal_odds']
            odds_b = fight_data['fighter_b_decimal_odds']
            
            assert odds_a > 1.0, f"Invalid odds for fighter A: {odds_a}"
            assert odds_b > 1.0, f"Invalid odds for fighter B: {odds_b}"
            
            # Check implied probability makes sense (accounting for bookmaker margin)
            implied_prob_total = (1/odds_a) + (1/odds_b)
            assert 1.0 <= implied_prob_total <= 1.2, \
                f"Unusual implied probability total: {implied_prob_total}"
    
    @staticmethod
    def validate_prediction_symmetry(predict_func, fighter_a: str, fighter_b: str, **kwargs):
        """Validate prediction symmetry (Fighter A vs B == 1 - Fighter B vs A)"""
        pred_ab = predict_func(fighter_a, fighter_b, **kwargs)
        pred_ba = predict_func(fighter_b, fighter_a, **kwargs)
        
        # Extract probabilities (assuming percentage strings)
        prob_a_in_ab = float(pred_ab['win_probabilities'][fighter_a].replace('%', '')) / 100
        prob_a_in_ba = float(pred_ba['win_probabilities'][fighter_a].replace('%', '')) / 100
        
        # Should be approximately symmetric
        symmetry_error = abs(prob_a_in_ab - (1 - prob_a_in_ba))
        assert symmetry_error < 0.05, \
            f"Prediction asymmetry too high: {symmetry_error} (prob_a_in_ab={prob_a_in_ab}, prob_a_in_ba={prob_a_in_ba})"