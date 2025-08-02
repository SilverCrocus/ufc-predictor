"""
Unit Tests for Feature Engineering Module
=========================================

This module contains comprehensive unit tests for the feature engineering
functionality, testing both the original and optimized implementations.

Tests cover:
- Individual parsing functions
- Complete feature engineering pipeline
- Error handling and edge cases
- Performance comparisons
- Data validation

Usage:
    pytest tests/test_feature_engineering.py -v
    pytest tests/test_feature_engineering.py::test_height_parsing -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

# Import the modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor.data.feature_engineering import (
    parse_height_to_inches,
    parse_weight_to_lbs,
    parse_reach_to_inches,
    parse_record,
    calculate_age,
    engineer_features_final,
    create_differential_features
)

from ufc_predictor.features.optimized_feature_engineering import (
    parse_height_vectorized,
    parse_weight_vectorized,
    parse_reach_vectorized,
    parse_record_vectorized,
    calculate_age_vectorized,
    engineer_features_final_optimized,
    create_differential_features_optimized
)

from conftest import assert_dataframe_structure


class TestIndividualParsers:
    """Test individual parsing functions"""
    
    def test_height_parsing_individual(self):
        """Test individual height parsing function"""
        # Test valid heights
        assert parse_height_to_inches('6\' 4"') == 76
        assert parse_height_to_inches('5\' 11"') == 71
        assert parse_height_to_inches('5\' 0"') == 60
        
        # Test edge cases
        assert parse_height_to_inches(None) is None
        assert parse_height_to_inches('') is None
        assert parse_height_to_inches('invalid') is None
    
    def test_height_parsing_vectorized(self):
        """Test vectorized height parsing"""
        heights = pd.Series(['6\' 4"', '5\' 11"', None, 'invalid', '5\' 0"'])
        result = parse_height_vectorized(heights)
        
        expected = pd.Series([76.0, 71.0, np.nan, np.nan, 60.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_weight_parsing_individual(self):
        """Test individual weight parsing function"""
        assert parse_weight_to_lbs('185 lbs.') == 185
        assert parse_weight_to_lbs('265 lbs.') == 265
        assert parse_weight_to_lbs(None) is None
        assert parse_weight_to_lbs('invalid') is None
    
    def test_weight_parsing_vectorized(self):
        """Test vectorized weight parsing"""
        weights = pd.Series(['185 lbs.', '265 lbs.', None, 'invalid'])
        result = parse_weight_vectorized(weights)
        
        expected = pd.Series([185.0, 265.0, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_reach_parsing_individual(self):
        """Test individual reach parsing function"""
        assert parse_reach_to_inches('84.5"') == 84.5
        assert parse_reach_to_inches('72"') == 72.0
        assert parse_reach_to_inches(None) is None
    
    def test_reach_parsing_vectorized(self):
        """Test vectorized reach parsing"""
        reaches = pd.Series(['84.5"', '72"', None, 'invalid"'])
        result = parse_reach_vectorized(reaches)
        
        expected = pd.Series([84.5, 72.0, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_record_parsing_individual(self):
        """Test individual record parsing function"""
        assert parse_record('27-1-0') == (27, 1, 0)
        assert parse_record('20-4-0') == (20, 4, 0)
        assert parse_record('15-3') == (15, 3, 0)  # No draws
        assert parse_record(None) == (None, None, None)
        assert parse_record('invalid') == (None, None, None)
    
    def test_record_parsing_vectorized(self):
        """Test vectorized record parsing"""
        records = pd.Series(['27-1-0', '20-4-0', '15-3', None, 'invalid'])
        result = parse_record_vectorized(records)
        
        expected_df = pd.DataFrame({
            'Wins': [27.0, 20.0, 15.0, np.nan, np.nan],
            'Losses': [1.0, 4.0, 3.0, np.nan, np.nan],
            'Draws': [0.0, 0.0, 0.0, np.nan, np.nan]
        })
        
        pd.testing.assert_frame_equal(result, expected_df, check_names=False)
    
    def test_age_calculation(self):
        """Test age calculation"""
        dobs = pd.Series(['Jul 19, 1987', 'Aug 19, 1982', 'Invalid Date'])
        result = calculate_age_vectorized(dobs, '2025-06-17')
        
        # Check that valid dates produce reasonable ages
        assert 35 <= result.iloc[0] <= 40  # Jon Jones born 1987
        assert 40 <= result.iloc[1] <= 45  # Stipe born 1982
        assert pd.isna(result.iloc[2])     # Invalid date


class TestFeatureEngineeringPipeline:
    """Test complete feature engineering pipeline"""
    
    def test_engineer_features_final_original(self, mock_fighter_data):
        """Test original feature engineering pipeline"""
        result = engineer_features_final(mock_fighter_data)
        
        # Validate structure
        expected_columns = ['Name', 'Height (inches)', 'Weight (lbs)', 'Reach (in)', 
                           'SLpM', 'Str. Acc.', 'SApM', 'Str. Def',
                           'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.',
                           'Wins', 'Losses', 'Draws', 'Age', 'fighter_url']
        
        assert_dataframe_structure(result, expected_columns, min_rows=4)
        
        # Check specific transformations
        assert result['Height (inches)'].iloc[0] == 76  # 6'4"
        assert result['Weight (lbs)'].iloc[0] == 205    # 205 lbs
        assert result['Str. Acc.'].iloc[0] == 0.58      # 58% -> 0.58
        assert result['Wins'].iloc[0] == 27             # From "27-1-0"
    
    def test_engineer_features_final_optimized(self, mock_fighter_data):
        """Test optimized feature engineering pipeline"""
        result = engineer_features_final_optimized(mock_fighter_data)
        
        # Should have same structure as original
        original_result = engineer_features_final(mock_fighter_data)
        
        # Check that column names match
        assert set(result.columns) == set(original_result.columns)
        
        # Check that key transformations match
        pd.testing.assert_series_equal(
            result['Height (inches)'], 
            original_result['Height (inches)'], 
            check_names=False
        )
        pd.testing.assert_series_equal(
            result['Weight (lbs)'], 
            original_result['Weight (lbs)'], 
            check_names=False
        )
    
    @pytest.mark.performance
    def test_performance_comparison(self, performance_tester, mock_fighter_data):
        """Test that optimized version is faster"""
        # Create larger dataset for meaningful performance test
        from conftest import generate_large_fighter_dataset
        large_data = generate_large_fighter_dataset(100)  # Smaller for CI
        
        # Time original implementation
        _, original_time = performance_tester.time_function(
            engineer_features_final, large_data.copy()
        )
        
        # Time optimized implementation
        _, optimized_time = performance_tester.time_function(
            engineer_features_final_optimized, large_data.copy()
        )
        
        # Optimized should be faster (allow some variance)
        improvement_ratio = original_time / optimized_time
        assert improvement_ratio > 1.2, f"Expected >20% improvement, got {improvement_ratio:.2f}x"
        
        print(f"Performance improvement: {improvement_ratio:.2f}x faster")
    
    def test_create_differential_features(self, mock_fighter_data):
        """Test differential features creation"""
        # Create mock merged fight data
        blue_fighters = mock_fighter_data.add_prefix('blue_')
        red_fighters = mock_fighter_data.add_prefix('red_')
        
        # Simple merge for testing
        fight_data = pd.concat([
            blue_fighters.iloc[:2].reset_index(drop=True),
            red_fighters.iloc[:2].reset_index(drop=True)
        ], axis=1)
        
        result = create_differential_features(fight_data)
        
        # Should have original columns plus differential columns
        assert len(result.columns) > len(fight_data.columns)
        
        # Check for specific differential columns
        diff_columns = [col for col in result.columns if col.endswith('_diff')]
        assert len(diff_columns) > 0, "Should create differential features"
        
        # Check that differentials are calculated correctly
        if 'blue_Height (inches)' in fight_data.columns and 'red_Height (inches)' in fight_data.columns:
            expected_diff = fight_data['blue_Height (inches)'] - fight_data['red_Height (inches)']
            if 'height_inches_diff' in result.columns:
                pd.testing.assert_series_equal(
                    result['height_inches_diff'], 
                    expected_diff, 
                    check_names=False
                )
    
    def test_create_differential_features_optimized(self, mock_fighter_data):
        """Test optimized differential features creation"""
        # Create the same test data as above
        blue_fighters = mock_fighter_data.add_prefix('blue_')
        red_fighters = mock_fighter_data.add_prefix('red_')
        fight_data = pd.concat([
            blue_fighters.iloc[:2].reset_index(drop=True),
            red_fighters.iloc[:2].reset_index(drop=True)
        ], axis=1)
        
        original_result = create_differential_features(fight_data)
        optimized_result = create_differential_features_optimized(fight_data)
        
        # Results should be identical
        assert set(original_result.columns) == set(optimized_result.columns)
        
        # Check specific differential columns match
        diff_columns = [col for col in original_result.columns if col.endswith('_diff')]
        for col in diff_columns:
            pd.testing.assert_series_equal(
                original_result[col], 
                optimized_result[col], 
                check_names=False
            )


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully without crashing
        try:
            result = engineer_features_final_optimized(empty_df)
            assert len(result) == 0
        except Exception as e:
            # Some exceptions are acceptable for empty data
            assert isinstance(e, (ValueError, KeyError))
    
    def test_missing_columns(self, mock_fighter_data):
        """Test handling of missing columns"""
        incomplete_data = mock_fighter_data[['Name', 'Height', 'Weight']].copy()
        
        # Should handle missing columns gracefully
        result = engineer_features_final_optimized(incomplete_data)
        assert len(result) > 0
        assert 'Name' in result.columns
    
    def test_all_null_values(self):
        """Test handling of all null values in a column"""
        df = pd.DataFrame({
            'Name': ['Fighter1', 'Fighter2'],
            'Height': [None, None],
            'Weight': [None, None],
            'Reach': [None, None],
            'Record': [None, None],
            'DOB': [None, None],
            'STANCE': [None, None],
            'SLpM': [None, None],
            'Str. Acc.': [None, None],
            'SApM': [None, None],
            'Str. Def': [None, None],
            'TD Avg.': [None, None],
            'TD Acc.': [None, None],
            'TD Def.': [None, None],
            'Sub. Avg.': [None, None],
            'fighter_url': ['url1', 'url2']
        })
        
        result = engineer_features_final_optimized(df)
        assert len(result) == 2  # Should still have 2 rows
        assert 'Name' in result.columns
    
    def test_malformed_data(self):
        """Test handling of malformed data"""
        df = pd.DataFrame({
            'Name': ['Fighter1'],
            'Height': ['invalid_height'],
            'Weight': ['not_a_weight'],
            'Reach': ['bad_reach'],
            'Record': ['not_a_record'],
            'DOB': ['not_a_date'],
            'STANCE': ['Unknown'],
            'SLpM': ['not_numeric'],
            'Str. Acc.': ['not_percent'],
            'SApM': ['not_numeric'],
            'Str. Def': ['not_percent'],
            'TD Avg.': ['not_numeric'],
            'TD Acc.': ['not_percent'],
            'TD Def.': ['not_percent'],
            'Sub. Avg.': ['not_numeric'],
            'fighter_url': ['url1']
        })
        
        # Should handle malformed data without crashing
        result = engineer_features_final_optimized(df)
        assert len(result) == 1
        
        # Malformed data should result in NaN values
        assert pd.isna(result['Height (inches)'].iloc[0])
        assert pd.isna(result['Weight (lbs)'].iloc[0])


class TestDataValidation:
    """Test data validation and consistency"""
    
    def test_output_data_types(self, mock_fighter_data):
        """Test that output has correct data types"""
        result = engineer_features_final_optimized(mock_fighter_data)
        
        # Numeric columns should be numeric
        numeric_columns = ['Height (inches)', 'Weight (lbs)', 'Reach (in)', 
                          'SLpM', 'Str. Acc.', 'Age', 'Wins', 'Losses', 'Draws']
        
        for col in numeric_columns:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should be numeric"
    
    def test_no_data_leakage(self, mock_fighter_data):
        """Test that no raw string data leaks into processed features"""
        result = engineer_features_final_optimized(mock_fighter_data)
        
        # These original columns should be removed
        removed_columns = ['Height', 'Weight', 'Reach', 'Record', 'DOB', 'STANCE']
        
        for col in removed_columns:
            assert col not in result.columns, f"Original column {col} should be removed"
    
    def test_stance_dummy_encoding(self, mock_fighter_data):
        """Test stance dummy variable encoding"""
        result = engineer_features_final_optimized(mock_fighter_data)
        
        # Should have stance dummy columns
        stance_columns = [col for col in result.columns if col.startswith('STANCE_')]
        assert len(stance_columns) > 0, "Should create stance dummy variables"
        
        # Each row should have exactly one stance = 1
        if stance_columns:
            stance_sums = result[stance_columns].sum(axis=1)
            assert all(stance_sums <= 1), "Each fighter should have at most one stance"
    
    def test_age_reasonableness(self, mock_fighter_data):
        """Test that calculated ages are reasonable"""
        result = engineer_features_final_optimized(mock_fighter_data)
        
        if 'Age' in result.columns:
            ages = result['Age'].dropna()
            assert all((18 <= age <= 50) for age in ages), "Ages should be reasonable for UFC fighters"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])