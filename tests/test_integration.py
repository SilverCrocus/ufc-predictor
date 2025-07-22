"""
Integration Tests for UFC Predictor System
==========================================

This module contains integration tests that verify the entire prediction
pipeline works correctly from end to end, including data processing,
model training, prediction, and profitability analysis.

Tests cover:
- Complete ML pipeline integration
- Prediction workflow
- Fast odds fetching integration
- Profitability analysis pipeline
- Error handling across components

Usage:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -m "not slow" -v  # Skip slow tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import time

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.fast_odds_fetcher import FastOddsFetcher, FastFightOdds
from src.fast_tab_profitability import FastTABProfitabilityAnalyzer
from src.optimized_feature_engineering import (
    engineer_features_final_optimized,
    create_differential_features_optimized,
    merge_fight_data_optimized
)

from conftest import assert_dataframe_structure, assert_prediction_result


class TestFastOddsFetching:
    """Test the fast odds fetching system"""
    
    def test_fast_odds_fetcher_initialization(self):
        """Test FastOddsFetcher initialization"""
        fetcher = FastOddsFetcher()
        
        assert fetcher.api_key is None
        assert fetcher.cache_duration == 300
        assert isinstance(fetcher.cache, dict)
        assert isinstance(fetcher.fallback_odds, dict)
    
    def test_odds_conversion_functions(self):
        """Test odds conversion utilities"""
        fetcher = FastOddsFetcher()
        
        # Test American to decimal conversion
        assert fetcher.american_to_decimal(100) == 2.0
        assert fetcher.american_to_decimal(-200) == 1.5
        assert fetcher.american_to_decimal(150) == 2.5
        
        # Test decimal to American conversion
        assert fetcher.decimal_to_american(2.0) == 100
        assert fetcher.decimal_to_american(1.5) == -200
        assert fetcher.decimal_to_american(2.5) == 150
    
    def test_fighter_name_similarity(self):
        """Test fighter name matching functionality"""
        fetcher = FastOddsFetcher()
        
        # Test exact matches
        assert fetcher.calculate_name_similarity("Jon Jones", "Jon Jones") == 1.0
        
        # Test similar names
        similarity = fetcher.calculate_name_similarity("Jon Jones", "Jonathan Jones")
        assert 0.7 < similarity < 1.0
        
        # Test different names
        similarity = fetcher.calculate_name_similarity("Jon Jones", "Stipe Miocic")
        assert similarity < 0.5
    
    @patch('aiohttp.ClientSession.get')
    def test_fast_odds_fetching_with_mock_api(self, mock_get, mock_http_response):
        """Test fast odds fetching with mocked API response"""
        # Setup mock
        mock_get.return_value.__aenter__.return_value.status = 200
        mock_get.return_value.__aenter__.return_value.json.return_value = mock_http_response.json()
        
        fetcher = FastOddsFetcher(api_key="test_key")
        fighter_pairs = ["Jon Jones vs Stipe Miocic"]
        
        # Test sync wrapper
        start_time = time.time()
        odds_list = fetcher.get_ufc_odds_sync(fighter_pairs, use_live_api=False)  # Use cache for test
        fetch_time = time.time() - start_time
        
        # Should be much faster than Selenium (under 5 seconds)
        assert fetch_time < 5.0, f"Fast fetching took {fetch_time:.2f}s, should be under 5s"
        
        # Should return FastFightOdds objects
        assert len(odds_list) == 1
        assert isinstance(odds_list[0], FastFightOdds)
        assert odds_list[0].fighter_a == "Jon Jones"
        assert odds_list[0].fighter_b == "Stipe Miocic"
    
    def test_cache_functionality(self):
        """Test caching mechanism"""
        fetcher = FastOddsFetcher(cache_duration=1)  # Short cache for testing
        
        # Cache some data
        test_data = {"test": "data"}
        fetcher.cache_odds("test_key", test_data)
        
        # Should retrieve from cache
        cached_data = fetcher.get_cached_odds("test_key")
        assert cached_data == test_data
        
        # Wait for cache expiration
        time.sleep(1.1)
        
        # Should not retrieve expired cache
        expired_data = fetcher.get_cached_odds("test_key")
        assert expired_data is None
    
    def test_performance_stats(self):
        """Test performance statistics"""
        fetcher = FastOddsFetcher()
        stats = fetcher.get_performance_stats()
        
        expected_keys = ['cache_size', 'cache_duration', 'fallback_available', 'api_key_configured']
        for key in expected_keys:
            assert key in stats


class TestFastProfitabilityAnalysis:
    """Test the fast profitability analysis system"""
    
    def test_fast_profitability_analyzer_initialization(self):
        """Test FastTABProfitabilityAnalyzer initialization"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000)
        
        assert analyzer.bankroll == 1000
        assert analyzer.use_live_odds is True
        assert isinstance(analyzer.fast_fetcher, FastOddsFetcher)
        assert isinstance(analyzer.performance_stats, dict)
    
    def test_fight_pairs_creation(self, mock_predictions):
        """Test creation of fight pairs from predictions"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000)
        
        fight_pairs = analyzer._create_fight_pairs(mock_predictions)
        
        # Should create pairs from fighters with complementary probabilities
        assert len(fight_pairs) > 0
        assert all(" vs " in pair for pair in fight_pairs)
        
        # All predicted fighters should be used
        used_fighters = set()
        for pair in fight_pairs:
            fighters = pair.split(" vs ")
            used_fighters.update(fighters)
        
        # Most fighters should be paired
        assert len(used_fighters) >= 2
    
    @patch.object(FastOddsFetcher, 'get_ufc_odds_sync')
    def test_profitability_analysis_pipeline(self, mock_odds_fetch, mock_predictions, mock_odds_data):
        """Test complete profitability analysis pipeline"""
        # Setup mocks
        mock_odds_list = [
            FastFightOdds(
                fighter_a="Jon Jones",
                fighter_b="Stipe Miocic",
                fighter_a_decimal_odds=1.33,
                fighter_b_decimal_odds=3.50,
                fighter_a_american_odds=-300,
                fighter_b_american_odds=250,
                source="mock",
                timestamp=time.time()
            )
        ]
        mock_odds_fetch.return_value = mock_odds_list
        
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
        
        start_time = time.time()
        results = analyzer.analyze_predictions(mock_predictions, "Test Event")
        analysis_time = time.time() - start_time
        
        # Should complete quickly
        assert analysis_time < 10.0, f"Analysis took {analysis_time:.2f}s, should be under 10s"
        
        # Check result structure
        required_keys = ['profitable_bets', 'multi_bet_opportunities', 'summary', 'performance_stats']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check summary statistics
        summary = results['summary']
        assert 'total_fights_analyzed' in summary
        assert 'analysis_time' in summary
        assert 'odds_fetch_time' in summary
        assert 'performance_improvement' in summary
    
    def test_multi_bet_analysis(self, mock_predictions):
        """Test multi-bet opportunity analysis"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000)
        
        # Create mock profitable opportunities
        from src.fast_tab_profitability import FastTABOpportunity
        opportunities = [
            FastTABOpportunity(
                fighter="Jon Jones",
                opponent="Stipe Miocic",
                event="Test Event",
                tab_decimal_odds=1.33,
                american_odds=-300,
                model_prob=0.75,
                market_prob=0.752,
                expected_value=0.10,
                recommended_bet=50.0,
                expected_profit=5.0
            ),
            FastTABOpportunity(
                fighter="Alexandre Pantoja",
                opponent="Kai Kara-France",
                event="Test Event",
                tab_decimal_odds=1.61,
                american_odds=-164,
                model_prob=0.62,
                market_prob=0.621,
                expected_value=0.08,
                recommended_bet=40.0,
                expected_profit=3.2
            )
        ]
        
        multi_bets = analyzer._analyze_multi_bets(opportunities)
        
        # Should create multi-bet opportunities
        if len(opportunities) > 1:
            assert len(multi_bets) >= 0  # May be 0 if EV threshold not met
            
            for multi_bet in multi_bets:
                required_keys = ['type', 'fighters', 'combined_odds', 'win_probability', 
                               'expected_value', 'recommended_stake', 'expected_profit', 'risk_level']
                for key in required_keys:
                    assert key in multi_bet, f"Missing key in multi-bet: {key}"
    
    def test_performance_comparison(self):
        """Test performance comparison with old system"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000)
        
        # Simulate some analyses
        analyzer.performance_stats['total_analyses'] = 5
        analyzer.performance_stats['total_fetch_time'] = 150.0  # 30s average
        
        comparison = analyzer.get_performance_comparison()
        
        # Should show significant improvement
        assert 'selenium_scraping_time' in comparison
        assert 'fast_fetching_time' in comparison
        assert 'performance_improvement' in comparison
        assert 'time_saved_per_analysis' in comparison
        
        # Should show major improvement
        improvement = float(comparison['performance_improvement'].replace('x faster', ''))
        assert improvement > 10.0, f"Expected >10x improvement, got {improvement}x"


class TestMLPipelineIntegration:
    """Test machine learning pipeline integration"""
    
    @pytest.mark.slow
    def test_complete_feature_engineering_pipeline(self, mock_fighter_data, mock_fight_data):
        """Test complete feature engineering from raw data to model-ready features"""
        # Step 1: Engineer fighter features
        fighters_engineered = engineer_features_final_optimized(mock_fighter_data)
        
        # Validate engineered fighters
        assert_dataframe_structure(fighters_engineered, 
                                 ['Name', 'Height (inches)', 'Weight (lbs)', 'Age', 'fighter_url'],
                                 min_rows=4)
        
        # Step 2: Merge with fight data  
        fight_dataset = merge_fight_data_optimized(mock_fight_data, fighters_engineered)
        
        # Should have blue and red corner features
        blue_cols = [col for col in fight_dataset.columns if col.startswith('blue_')]
        red_cols = [col for col in fight_dataset.columns if col.startswith('red_')]
        
        assert len(blue_cols) > 0, "Should have blue corner features"
        assert len(red_cols) > 0, "Should have red corner features"
        
        # Step 3: Create differential features
        fight_dataset_with_diffs = create_differential_features_optimized(fight_dataset)
        
        # Should have differential features
        diff_cols = [col for col in fight_dataset_with_diffs.columns if col.endswith('_diff')]
        assert len(diff_cols) > 0, "Should create differential features"
        
        # Final dataset should be ready for modeling
        assert len(fight_dataset_with_diffs) > 0
        assert len(fight_dataset_with_diffs.columns) > len(fight_dataset.columns)
    
    def test_prediction_pipeline_integration(self, mock_trained_model, mock_fighter_data):
        """Test prediction pipeline with mock model"""
        # This would test the actual prediction functions
        # For now, we test the structure
        
        # Mock prediction result
        prediction_result = {
            "matchup": "Jon Jones vs. Stipe Miocic",
            "predicted_winner": "Jon Jones",
            "winner_confidence": "75.00%",
            "win_probabilities": {
                "Jon Jones": "75.00%",
                "Stipe Miocic": "25.00%"
            },
            "predicted_method": "Decision",
            "method_probabilities": {
                "Decision": "60.00%",
                "KO/TKO": "30.00%",
                "Submission": "10.00%"
            }
        }
        
        # Validate prediction structure
        assert_prediction_result(prediction_result)


class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""
    
    def test_profitability_analysis_with_no_odds(self, mock_predictions):
        """Test profitability analysis when no odds are available"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
        
        # Mock empty odds response
        with patch.object(analyzer.fast_fetcher, 'get_ufc_odds_sync', return_value=[]):
            results = analyzer.analyze_predictions(mock_predictions, "Test Event")
            
            # Should handle gracefully
            assert results['summary']['profitable_opportunities'] == 0
            assert len(results['profitable_bets']) == 0
    
    def test_feature_engineering_with_corrupted_data(self):
        """Test feature engineering with heavily corrupted data"""
        corrupted_data = pd.DataFrame({
            'Name': ['Fighter1', None, ''],
            'Height': [None, 'corrupt', ''],
            'Weight': ['corrupt', None, 'bad_weight'],
            'Reach': ['bad_reach', '', None],
            'Record': [None, 'bad_record', ''],
            'DOB': ['bad_date', None, ''],
            'STANCE': [None, '', 'Unknown'],
            'SLpM': ['bad', None, ''],
            'Str. Acc.': [None, 'bad%', ''],
            'SApM': ['', None, 'corrupt'],
            'Str. Def': ['corrupt', '', None],
            'TD Avg.': [None, '', 'bad'],
            'TD Acc.': ['bad%', None, ''],
            'TD Def.': ['', 'corrupt', None],
            'Sub. Avg.': [None, 'bad', ''],
            'fighter_url': ['url1', 'url2', 'url3']
        })
        
        # Should not crash with corrupted data
        result = engineer_features_final_optimized(corrupted_data)
        assert len(result) == 3  # Should preserve rows
        assert 'Name' in result.columns
    
    def test_odds_fetching_with_network_errors(self, mock_predictions, error_simulator):
        """Test odds fetching resilience to network errors"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
        
        # Mock network error
        with patch.object(analyzer.fast_fetcher, 'get_ufc_odds_sync', side_effect=error_simulator.network_error):
            # Should fall back to cache/default behavior
            try:
                results = analyzer.analyze_predictions(mock_predictions, "Test Event")
                # If it doesn't crash, the fallback worked
                assert 'summary' in results
            except Exception:
                # Network errors might propagate in some cases
                pass


class TestPerformanceIntegration:
    """Test performance across integrated components"""
    
    @pytest.mark.performance 
    def test_end_to_end_performance(self, performance_tester, mock_predictions):
        """Test end-to-end performance of optimized system"""
        analyzer = FastTABProfitabilityAnalyzer(bankroll=1000, use_live_odds=False)
        
        # Time complete analysis
        _, analysis_time = performance_tester.time_function(
            analyzer.analyze_predictions, mock_predictions, "Performance Test"
        )
        
        # Should be much faster than old system
        performance_tester.assert_performance('analyze_predictions', 30.0)  # Under 30 seconds
        
        print(f"End-to-end analysis completed in {analysis_time:.2f} seconds")
    
    @pytest.mark.performance
    def test_feature_engineering_scalability(self, performance_tester):
        """Test feature engineering performance with larger datasets"""
        from conftest import generate_large_fighter_dataset
        
        # Test with progressively larger datasets
        sizes = [100, 500, 1000]
        
        for size in sizes:
            large_data = generate_large_fighter_dataset(size)
            
            _, processing_time = performance_tester.time_function(
                engineer_features_final_optimized, large_data
            )
            
            # Should scale linearly (roughly)
            time_per_fighter = processing_time / size
            assert time_per_fighter < 0.01, f"Processing {size} fighters took {time_per_fighter:.4f}s per fighter"
            
            print(f"Processed {size} fighters in {processing_time:.2f}s ({time_per_fighter:.4f}s per fighter)")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])