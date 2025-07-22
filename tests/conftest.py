"""
Pytest Configuration and Fixtures for UFC Predictor Testing
==========================================================

This module provides shared test configuration, fixtures, and utilities
for the UFC predictor test suite. It includes mock data, test utilities,
and setup/teardown functionality.

Features:
- Mock fighter and fight data for consistent testing
- Test database fixtures
- Performance testing utilities
- Error simulation helpers
- Logging configuration for tests

Usage:
    # Fixtures are automatically available in test functions
    def test_feature_engineering(mock_fighter_data):
        result = engineer_features_final(mock_fighter_data)
        assert len(result) > 0
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import logging

# Configure logging for tests
logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture
def mock_fighter_data():
    """Mock fighter data for testing feature engineering"""
    return pd.DataFrame({
        'Name': ['Jon Jones', 'Stipe Miocic', 'Alexandre Pantoja', 'Kai Kara-France'],
        'Height': ['6\' 4"', '6\' 4"', '5\' 5"', '5\' 5"'],
        'Weight': ['205 lbs.', '241 lbs.', '125 lbs.', '125 lbs.'],
        'Reach': ['84.5"', '80"', '67"', '66"'],
        'STANCE': ['Orthodox', 'Orthodox', 'Orthodox', 'Switch'],
        'DOB': ['Jul 19, 1987', 'Aug 19, 1982', 'Aug 16, 1990', 'Mar 26, 1993'],
        'Record': ['27-1-0', '20-4-0', '26-5-0', '24-10-0'],
        'SLpM': ['4.29', '4.55', '4.82', '6.39'],
        'Str. Acc.': ['58%', '52%', '52%', '44%'],
        'SApM': ['2.22', '3.09', '2.95', '4.66'],
        'Str. Def': ['64%', '59%', '59%', '62%'],
        'TD Avg.': ['2.06', '2.45', '1.93', '1.77'],
        'TD Acc.': ['43%', '32%', '44%', '39%'],
        'TD Def.': ['95%', '68%', '64%', '70%'],
        'Sub. Avg.': ['0.6', '0.2', '1.2', '0.7'],
        'fighter_url': [
            'http://ufcstats.com/fighter-details/f4c49976c75c5ab2',
            'http://ufcstats.com/fighter-details/6c9c8aca88f9d0d7', 
            'http://ufcstats.com/fighter-details/a8d5c7b4e0f1a2b3',
            'http://ufcstats.com/fighter-details/d4e5f6c8a9b0c1d2'
        ]
    })


@pytest.fixture
def mock_fight_data():
    """Mock fight data for testing"""
    return pd.DataFrame({
        'Fighter': ['Jon Jones', 'Alexandre Pantoja'],
        'Opponent': ['Stipe Miocic', 'Kai Kara-France'],
        'Event': ['UFC 309', 'UFC 290'],
        'Outcome': ['W', 'W'],
        'Method': ['Decision - Unanimous', 'Submission - Rear Naked Choke'],
        'Time': ['5:00', '2:34'],
        'fighter_url': [
            'http://ufcstats.com/fighter-details/f4c49976c75c5ab2',
            'http://ufcstats.com/fighter-details/a8d5c7b4e0f1a2b3'
        ]
    })


@pytest.fixture
def mock_predictions():
    """Mock prediction data for testing profitability analysis"""
    return {
        'Jon Jones': 0.75,
        'Stipe Miocic': 0.25,
        'Alexandre Pantoja': 0.62,
        'Kai Kara-France': 0.38
    }


@pytest.fixture
def mock_odds_data():
    """Mock odds data for testing"""
    return {
        'Jon Jones vs Stipe Miocic': {
            'fighter_a': 'Jon Jones',
            'fighter_b': 'Stipe Miocic', 
            'fighter_a_decimal_odds': 1.33,
            'fighter_b_decimal_odds': 3.50,
            'fighter_a_american_odds': -300,
            'fighter_b_american_odds': 250
        },
        'Alexandre Pantoja vs Kai Kara-France': {
            'fighter_a': 'Alexandre Pantoja',
            'fighter_b': 'Kai Kara-France',
            'fighter_a_decimal_odds': 1.61,
            'fighter_b_decimal_odds': 2.63,
            'fighter_a_american_odds': -164,
            'fighter_b_american_odds': 163
        }
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Clean up after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_model_files(temp_directory):
    """Create mock model files for testing"""
    model_dir = temp_directory / "model"
    model_dir.mkdir(exist_ok=True)
    
    # Create mock joblib model file (empty file for testing)
    model_file = model_dir / "ufc_random_forest_model_tuned.joblib"
    model_file.touch()
    
    # Create mock column mapping file
    columns_file = model_dir / "winner_model_columns.json"
    mock_columns = ['blue_Height (inches)', 'blue_Weight (lbs)', 'red_Height (inches)', 'red_Weight (lbs)',
                   'height_inches_diff', 'weight_lbs_diff']
    with open(columns_file, 'w') as f:
        json.dump(mock_columns, f)
    
    # Create mock fighter data file
    fighter_data_file = model_dir / "ufc_fighters_engineered_corrected.csv"
    mock_fighter_data().to_csv(fighter_data_file, index=False)
    
    return {
        'model_file': str(model_file),
        'columns_file': str(columns_file),
        'fighter_data_file': str(fighter_data_file)
    }


@pytest.fixture
def mock_http_response():
    """Mock HTTP response for testing web scraping"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <body>
            <div class="fighter-name">Jon Jones</div>
            <div class="odds">1.33</div>
            <div class="fighter-name">Stipe Miocic</div>
            <div class="odds">3.50</div>
        </body>
    </html>
    """
    mock_response.json.return_value = {
        'data': [
            {
                'home_team': 'Jon Jones',
                'away_team': 'Stipe Miocic',
                'bookmakers': [{
                    'markets': [{
                        'outcomes': [
                            {'name': 'Jon Jones', 'price': 1.33},
                            {'name': 'Stipe Miocic', 'price': 3.50}
                        ]
                    }]
                }]
            }
        ]
    }
    return mock_response


class MockModel:
    """Mock model for testing prediction functions"""
    
    def __init__(self):
        self.classes_ = ['Decision', 'KO/TKO', 'Submission']
        self.feature_importances_ = np.random.random(10)
    
    def predict(self, X):
        """Mock prediction that returns consistent results"""
        if len(X) == 0:
            return np.array([])
        return np.array([1] * len(X))  # Always predict class 1
    
    def predict_proba(self, X):
        """Mock probability prediction"""
        if len(X) == 0:
            return np.array([]).reshape(0, 2)
        # Return consistent probabilities
        probs = np.array([[0.35, 0.65]] * len(X))
        return probs
    
    def fit(self, X, y):
        """Mock fit method"""
        return self
    
    def score(self, X, y):
        """Mock score method"""
        return 0.75  # Return consistent accuracy score


@pytest.fixture
def mock_trained_model():
    """Mock trained model for testing"""
    return MockModel()


class ErrorSimulator:
    """Helper class to simulate various error conditions"""
    
    @staticmethod
    def network_error():
        """Simulate network error"""
        import requests
        raise requests.ConnectionError("Mock network error")
    
    @staticmethod
    def timeout_error():
        """Simulate timeout error"""
        raise TimeoutError("Mock timeout error")
    
    @staticmethod
    def file_not_found_error():
        """Simulate file not found error"""
        raise FileNotFoundError("Mock file not found")
    
    @staticmethod
    def data_processing_error():
        """Simulate data processing error"""
        raise ValueError("Mock data processing error")


@pytest.fixture
def error_simulator():
    """Error simulation utilities"""
    return ErrorSimulator()


class PerformanceTester:
    """Helper class for performance testing"""
    
    def __init__(self):
        self.timing_results = {}
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.timing_results[func_name] = execution_time
        
        return result, execution_time
    
    def assert_performance(self, func_name: str, max_time: float):
        """Assert that function executed within time limit"""
        if func_name in self.timing_results:
            actual_time = self.timing_results[func_name]
            assert actual_time <= max_time, f"{func_name} took {actual_time:.2f}s, expected <= {max_time}s"
        else:
            pytest.fail(f"No timing data found for {func_name}")


@pytest.fixture
def performance_tester():
    """Performance testing utilities"""
    return PerformanceTester()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


# Helper functions for test data generation
def generate_large_fighter_dataset(num_fighters: int = 1000) -> pd.DataFrame:
    """Generate large mock fighter dataset for performance testing"""
    np.random.seed(42)  # For reproducible tests
    
    stances = ['Orthodox', 'Southpaw', 'Switch', 'Open Stance']
    
    data = {
        'Name': [f'Fighter_{i}' for i in range(num_fighters)],
        'Height': [f'{np.random.randint(5, 7)}\' {np.random.randint(0, 12)}"' for _ in range(num_fighters)],
        'Weight': [f'{np.random.randint(115, 265)} lbs.' for _ in range(num_fighters)],
        'Reach': [f'{np.random.randint(60, 85)}"' for _ in range(num_fighters)],
        'STANCE': [np.random.choice(stances) for _ in range(num_fighters)],
        'DOB': [f'Jan {np.random.randint(1, 28)}, {np.random.randint(1980, 2000)}' for _ in range(num_fighters)],
        'Record': [f'{np.random.randint(10, 30)}-{np.random.randint(0, 10)}-{np.random.randint(0, 3)}' for _ in range(num_fighters)],
        'SLpM': [f'{np.random.uniform(2.0, 8.0):.2f}' for _ in range(num_fighters)],
        'Str. Acc.': [f'{np.random.randint(35, 70)}%' for _ in range(num_fighters)],
        'SApM': [f'{np.random.uniform(1.0, 6.0):.2f}' for _ in range(num_fighters)],
        'Str. Def': [f'{np.random.randint(40, 80)}%' for _ in range(num_fighters)],
        'TD Avg.': [f'{np.random.uniform(0.0, 5.0):.2f}' for _ in range(num_fighters)],
        'TD Acc.': [f'{np.random.randint(20, 60)}%' for _ in range(num_fighters)],
        'TD Def.': [f'{np.random.randint(50, 95)}%' for _ in range(num_fighters)],
        'Sub. Avg.': [f'{np.random.uniform(0.0, 3.0):.1f}' for _ in range(num_fighters)],
        'fighter_url': [f'http://ufcstats.com/fighter-details/{i:08x}' for i in range(num_fighters)]
    }
    
    return pd.DataFrame(data)


def assert_dataframe_structure(df: pd.DataFrame, expected_columns: List[str], min_rows: int = 1):
    """Assert DataFrame has expected structure"""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    
    missing_columns = set(expected_columns) - set(df.columns)
    assert not missing_columns, f"Missing expected columns: {missing_columns}"


def assert_prediction_result(result: Dict[str, Any]):
    """Assert prediction result has expected structure"""
    required_keys = ['matchup', 'predicted_winner', 'winner_confidence', 'win_probabilities']
    missing_keys = set(required_keys) - set(result.keys())
    assert not missing_keys, f"Missing required keys in prediction result: {missing_keys}"
    
    assert 'vs' in result['matchup'], "Matchup should contain 'vs'"
    assert result['predicted_winner'] in result['win_probabilities'], "Predicted winner should be in probabilities"


# Global test configuration
@pytest.fixture(autouse=True)
def configure_test_environment():
    """Automatically configure test environment for all tests"""
    # Suppress warnings during tests
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Configure pandas for testing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    yield
    
    # Clean up after tests
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')