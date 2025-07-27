"""
Integration Tests for Complete Agent Workflow

End-to-end integration tests for the UFC Betting Agent with Phase 2A hybrid system,
testing complete workflows from event analysis to betting recommendations.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from pathlib import Path
import tempfile
import json

from tests.test_agent.utils.agent_test_helpers import AgentTestHelpers
from tests.test_agent.utils.async_test_utilities import AsyncTestRunner


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
class TestCompleteAgentWorkflow:
    """Test complete agent workflow integration"""
    
    async def test_end_to_end_analysis_workflow_phase2a(self, mock_agent_config, async_test_runner):
        """Test complete end-to-end analysis workflow with Phase 2A hybrid system"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        
        # Mock all external dependencies for integration test
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models') as mock_models, \
             patch('pandas.read_csv') as mock_csv, \
             patch('joblib.load') as mock_joblib, \
             patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:
            
            # Setup comprehensive mocks
            mock_models.return_value = {
                'version': 'integration_test_v1.0',
                'winner_model_path': Path('test_winner.pkl'),
                'method_model_path': Path('test_method.pkl'),
                'winner_cols_path': Path('test_winner_cols.json'),
                'method_cols_path': Path('test_method_cols.json'),
                'fighters_data_path': Path('test_fighters.csv')
            }
            
            # Create realistic test data
            betting_system = AgentTestHelpers.create_mock_betting_system()
            mock_csv.return_value = betting_system['fighters_df']
            mock_json.return_value = betting_system['winner_cols']
            
            mock_model = Mock()
            mock_model.predict_proba.return_value = [[0.35, 0.65]]  # Realistic probability
            mock_model.predict.return_value = [1]
            mock_joblib.return_value = mock_model
            
            # Initialize agent
            agent = UFCBettingAgent(mock_agent_config)
            
            # Initialize betting system
            assert agent.initialize_betting_system()
            assert agent.betting_system is not None
            assert agent.prediction_service is not None
            
            # Test realistic UFC event
            event_name = "UFC 300 - Championship Event"
            test_fights = [
                "Jon Jones vs Stipe Miocic",
                "Alex Pereira vs Khalil Rountree",
                "Zhang Weili vs Tatiana Suarez"
            ]
            
            # Mock Phase 2A hybrid service with realistic behavior
            original_hybrid_service = agent.hybrid_odds_service
            agent.hybrid_odds_service = AsyncMock()
            
            # Configure hybrid service to simulate realistic Phase 2A behavior
            hybrid_result = AgentTestHelpers.create_mock_hybrid_result(
                test_fights, 
                api_requests_used=1  # Should use API for CRITICAL priority event
            )
            agent.hybrid_odds_service.fetch_event_odds.return_value = hybrid_result
            agent.hybrid_odds_service.get_quota_status.return_value = {
                'quota_status': {
                    'requests_used_today': 6,
                    'requests_remaining_today': 10,
                    'budget_remaining': 44.0
                },
                'hybrid_service_metrics': {
                    'api_efficiency': 0.75
                }
            }
            
            # Run complete workflow with timeout protection
            with tempfile.TemporaryDirectory() as temp_dir:
                # Update config to use temp directory
                mock_agent_config.agent.analysis_export_path = temp_dir
                
                result = await async_test_runner.run_with_timeout(
                    agent.analyze_event(event_name, test_fights),
                    timeout=30.0
                )
            
            # Comprehensive validation of integration
            assert result['status'] == 'completed'
            assert result['event_name'] == event_name
            
            # Validate Phase 2A integration
            assert 'odds_result' in result
            odds_result = result['odds_result']
            assert odds_result['status'] == 'success'
            assert odds_result['total_fights'] == len(test_fights)
            assert 'api_requests_used' in odds_result
            assert 'confidence_score' in odds_result
            assert odds_result['confidence_score'] > 0.8  # High confidence expected
            
            # Validate predictions integration
            assert 'predictions_analysis' in result
            pred_analysis = result['predictions_analysis']
            assert pred_analysis['successful_predictions'] == len(test_fights)
            assert pred_analysis['failed_predictions'] == 0
            
            # Validate betting recommendations integration
            assert 'betting_recommendations' in result
            betting_recs = result['betting_recommendations']
            assert betting_recs['total_bets'] >= 0
            assert betting_recs['portfolio_ev'] > 0  # Should find positive EV opportunities
            
            # Validate export functionality
            assert 'export_path' in result
            export_path = Path(result['export_path'])
            assert export_path.exists()
            
            # Verify hybrid service was called correctly
            agent.hybrid_odds_service.fetch_event_odds.assert_called_once()
            call_args = agent.hybrid_odds_service.fetch_event_odds.call_args
            assert call_args[0][0] == event_name
            assert call_args[0][1] == test_fights
            # Should use CRITICAL priority for championship event
            assert call_args.kwargs['priority'].value == 'CRITICAL'
    
    async def test_multi_event_monitoring_workflow(self, mock_agent_config):
        """Test monitoring workflow for multiple events"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        from src.agent.config.agent_config import EventConfig
        
        # Setup agent with mocked dependencies
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models'), \
             patch('pandas.read_csv'), \
             patch('joblib.load'), \
             patch('builtins.open'), \
             patch('json.load'):
            
            agent = UFCBettingAgent(mock_agent_config)
            agent.initialize_betting_system()
            
            # Create multiple test events
            event_configs = []
            
            # High priority event
            high_priority_event = Mock()
            high_priority_event.name = "UFC 299 - Title Fight"
            high_priority_event.target_fights = ["Champion vs Challenger"]
            high_priority_event.monitoring_enabled = True
            event_configs.append(high_priority_event)
            
            # Medium priority event
            medium_priority_event = Mock()
            medium_priority_event.name = "UFC Fight Night"
            medium_priority_event.target_fights = ["Fighter A vs Fighter B", "Fighter C vs Fighter D"]
            medium_priority_event.monitoring_enabled = True
            event_configs.append(medium_priority_event)
            
            # Disabled event (should be skipped)
            disabled_event = Mock()
            disabled_event.name = "Disabled Event"
            disabled_event.target_fights = ["Fighter X vs Fighter Y"]
            disabled_event.monitoring_enabled = False
            event_configs.append(disabled_event)
            
            # Mock hybrid service for all events
            agent.hybrid_odds_service = AsyncMock()
            
            def mock_fetch_event_odds(event_name, target_fights, **kwargs):
                return AgentTestHelpers.create_mock_hybrid_result(target_fights, api_requests_used=1)
            
            agent.hybrid_odds_service.fetch_event_odds.side_effect = mock_fetch_event_odds
            agent.hybrid_odds_service.get_quota_status.return_value = {
                'quota_status': {
                    'requests_used_today': 3,
                    'requests_remaining_today': 13,
                    'budget_remaining': 47.0
                },
                'hybrid_service_metrics': {
                    'api_efficiency': 0.85
                }
            }
            
            # Mock export functionality
            with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
                
                # Run monitoring workflow
                monitoring_results = await agent.run_event_monitoring(event_configs)
            
            # Validate monitoring results
            assert 'events' in monitoring_results
            assert len(monitoring_results['events']) == 2  # Only enabled events
            
            # High priority event should be processed
            assert "UFC 299 - Title Fight" in monitoring_results['events']
            title_fight_result = monitoring_results['events']["UFC 299 - Title Fight"]
            assert title_fight_result['status'] == 'completed'
            
            # Medium priority event should be processed
            assert "UFC Fight Night" in monitoring_results['events']
            fight_night_result = monitoring_results['events']["UFC Fight Night"]
            assert fight_night_result['status'] == 'completed'
            
            # Disabled event should not be processed
            assert "Disabled Event" not in monitoring_results['events']
            
            # Validate quota summary
            assert 'quota_summary' in monitoring_results
            quota_summary = monitoring_results['quota_summary']
            assert quota_summary['api_requests_used'] == 3
            assert quota_summary['efficiency'] > 0.8
    
    async def test_error_recovery_workflow(self, mock_agent_config):
        """Test error recovery and graceful degradation"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models'), \
             patch('pandas.read_csv'), \
             patch('joblib.load'), \
             patch('builtins.open'), \
             patch('json.load'):
            
            agent = UFCBettingAgent(mock_agent_config)
            agent.initialize_betting_system()
            
            # Test scenario: API fails, should fallback to TAB scraper
            agent.hybrid_odds_service = AsyncMock()
            
            # First call fails (API error)
            # Second call succeeds (TAB fallback)
            api_failure_result = Mock()
            api_failure_result.reconciled_data = None
            api_failure_result.api_requests_used = 0
            api_failure_result.confidence_score = 0.0
            api_failure_result.fallback_activated = True
            
            fallback_success_result = AgentTestHelpers.create_mock_hybrid_result(
                ["Fighter A vs Fighter B"], 
                api_requests_used=0  # TAB only
            )
            
            agent.hybrid_odds_service.fetch_event_odds.side_effect = [
                api_failure_result,  # First attempt fails
                fallback_success_result  # Retry succeeds with fallback
            ]
            
            # Mock prediction and betting services to succeed
            agent.prediction_service = Mock()
            mock_analysis = Mock()
            mock_analysis.summary = {'successful_predictions': 1, 'failed_predictions': 0}
            agent.prediction_service.predict_event.return_value = mock_analysis
            
            agent.betting_service = Mock()
            mock_betting = Mock()
            mock_betting.single_bets = []
            mock_betting.portfolio_summary = Mock(total_recommended_stake=50.0)
            agent.betting_service.generate_betting_recommendations.return_value = mock_betting
            
            with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
                result = await agent.analyze_event("Error Recovery Test", ["Fighter A vs Fighter B"])
            
            # Should eventually succeed despite initial failure
            assert result['status'] == 'completed'
            assert result['odds_result']['fallback_activated'] is True
            assert result['odds_result']['api_requests_used'] == 0


@pytest.mark.integration
@pytest.mark.phase2a
@pytest.mark.asyncio
class TestPhase2AIntegrationWorkflow:
    """Test Phase 2A hybrid system integration workflows"""
    
    async def test_quota_aware_event_prioritization(self, mock_agent_config):
        """Test that Phase 2A system properly prioritizes events based on quota"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models'), \
             patch('pandas.read_csv'), \
             patch('joblib.load'), \
             patch('builtins.open'), \
             patch('json.load'):
            
            agent = UFCBettingAgent(mock_agent_config)
            agent.initialize_betting_system()
            
            # Mock hybrid service with low quota remaining
            agent.hybrid_odds_service = AsyncMock()
            agent.hybrid_odds_service.get_quota_status.return_value = {
                'quota_status': {
                    'requests_used_today': 14,  # Almost exhausted
                    'requests_remaining_today': 2,
                    'budget_remaining': 10.0
                }
            }
            
            # Track priority usage
            priority_calls = []
            
            def mock_fetch_with_priority_tracking(event_name, target_fights, priority=None, **kwargs):
                priority_calls.append((event_name, priority))
                return AgentTestHelpers.create_mock_hybrid_result(
                    target_fights, 
                    api_requests_used=1 if priority.value in ['CRITICAL', 'HIGH'] else 0
                )
            
            agent.hybrid_odds_service.fetch_event_odds.side_effect = mock_fetch_with_priority_tracking
            
            # Mock other services
            agent.prediction_service = Mock()
            agent.prediction_service.predict_event.return_value = Mock(
                summary={'successful_predictions': 1, 'failed_predictions': 0}
            )
            agent.betting_service = Mock()
            agent.betting_service.generate_betting_recommendations.return_value = Mock(
                single_bets=[], portfolio_summary=Mock(total_recommended_stake=50.0)
            )
            
            # Test different event types
            with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
                
                # Critical event - should use API despite low quota
                await agent.analyze_event("UFC 300 - Title Championship", ["Champion vs Challenger"])
                
                # Low priority event - should use fallback to preserve quota
                await agent.analyze_event("Training Camp Event", ["Prospect A vs Prospect B"])
            
            # Verify priority allocation
            assert len(priority_calls) == 2
            
            # Title fight should get CRITICAL priority
            title_call = next((call for call in priority_calls if "Title Championship" in call[0]), None)
            assert title_call is not None
            assert title_call[1].value == 'CRITICAL'
            
            # Training event should get LOW priority
            training_call = next((call for call in priority_calls if "Training Camp" in call[0]), None)
            assert training_call is not None
            assert training_call[1].value == 'LOW'
    
    async def test_hybrid_data_source_confidence_integration(self, mock_agent_config):
        """Test integration of data source confidence scoring"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models'), \
             patch('pandas.read_csv'), \
             patch('joblib.load'), \
             patch('builtins.open'), \
             patch('json.load'):
            
            agent = UFCBettingAgent(mock_agent_config)
            agent.initialize_betting_system()
            
            # Setup hybrid service with different data source scenarios
            agent.hybrid_odds_service = AsyncMock()
            
            test_fights = ["High Confidence Fighter vs Medium Confidence Fighter"]
            
            # Create result with mixed data sources and confidence scores
            hybrid_result = Mock()
            hybrid_result.reconciled_data = {
                test_fights[0]: {
                    'fighter_a': 'High Confidence Fighter',
                    'fighter_b': 'Medium Confidence Fighter',
                    'fighter_a_decimal_odds': 1.75,
                    'fighter_b_decimal_odds': 2.1,
                    'data_sources': ['odds_api', 'tab_australia'],  # Multiple sources
                    'confidence_score': 0.95,  # High confidence
                    'tab_validation': {
                        'odds_discrepancy_a': 0.02,  # Low discrepancy
                        'odds_discrepancy_b': 0.03
                    }
                }
            }
            hybrid_result.api_requests_used = 1
            hybrid_result.confidence_score = 0.95
            hybrid_result.fallback_activated = False
            
            agent.hybrid_odds_service.fetch_event_odds.return_value = hybrid_result
            
            # Mock prediction service to validate confidence integration
            prediction_calls = []
            
            def mock_predict_event(odds_data, event_name):
                prediction_calls.append(odds_data)
                
                # Create prediction result that reflects data confidence
                prediction_result = Mock()
                prediction_result.fight_key = test_fights[0]
                prediction_result.confidence_score = 0.92  # Should reflect high data confidence
                
                analysis = Mock()
                analysis.fight_predictions = [prediction_result]
                analysis.summary = {'successful_predictions': 1, 'failed_predictions': 0}
                
                return analysis
            
            agent.prediction_service = Mock()
            agent.prediction_service.predict_event.side_effect = mock_predict_event
            
            agent.betting_service = Mock()
            agent.betting_service.generate_betting_recommendations.return_value = Mock(
                single_bets=[], portfolio_summary=Mock(total_recommended_stake=75.0)
            )
            
            with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
                result = await agent.analyze_event("High Quality Data Event", test_fights)
            
            # Verify high confidence data integration
            assert result['status'] == 'completed'
            assert result['odds_result']['confidence_score'] == 0.95
            assert 'odds_api' in result['odds_result']['data_sources']
            assert 'tab_australia' in result['odds_result']['data_sources']
            
            # Verify prediction service received enriched odds data
            assert len(prediction_calls) == 1
            odds_data = prediction_calls[0]
            fight_data = odds_data[test_fights[0]]
            assert 'data_sources' in fight_data
            assert 'confidence_score' in fight_data
            assert fight_data['confidence_score'] == 0.95


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
class TestIntegrationPerformance:
    """Test integration performance under realistic conditions"""
    
    async def test_concurrent_event_analysis_performance(self, mock_agent_config):
        """Test performance with concurrent event analysis"""
        from src.agent.ufc_betting_agent import UFCBettingAgent
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models'), \
             patch('pandas.read_csv'), \
             patch('joblib.load'), \
             patch('builtins.open'), \
             patch('json.load'):
            
            agent = UFCBettingAgent(mock_agent_config)
            agent.initialize_betting_system()
            
            # Setup fast mocks for performance testing
            agent.hybrid_odds_service = AsyncMock()
            
            async def fast_fetch_odds(event_name, target_fights, **kwargs):
                # Simulate realistic fetch time
                await asyncio.sleep(0.1)  # 100ms per fetch
                return AgentTestHelpers.create_mock_hybrid_result(target_fights)
            
            agent.hybrid_odds_service.fetch_event_odds.side_effect = fast_fetch_odds
            
            # Mock other services for speed
            agent.prediction_service = Mock()
            agent.prediction_service.predict_event.return_value = Mock(
                summary={'successful_predictions': 1, 'failed_predictions': 0}
            )
            agent.betting_service = Mock()
            agent.betting_service.generate_betting_recommendations.return_value = Mock(
                single_bets=[], portfolio_summary=Mock(total_recommended_stake=50.0)
            )
            
            # Test concurrent analysis of multiple small events
            events = [
                ("Event 1", ["Fight 1A vs Fight 1B"]),
                ("Event 2", ["Fight 2A vs Fight 2B"]),
                ("Event 3", ["Fight 3A vs Fight 3B"]),
                ("Event 4", ["Fight 4A vs Fight 4B"]),
                ("Event 5", ["Fight 5A vs Fight 5B"])
            ]
            
            with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
                
                # Measure concurrent performance
                import time
                start_time = time.perf_counter()
                
                # Run all analyses concurrently
                tasks = [
                    agent.analyze_event(event_name, fights) 
                    for event_name, fights in events
                ]
                results = await asyncio.gather(*tasks)
                
                end_time = time.perf_counter()
                total_time = end_time - start_time
            
            # Verify all analyses completed successfully
            assert len(results) == 5
            for result in results:
                assert result['status'] == 'completed'
            
            # Performance assertion: concurrent should be faster than sequential
            # With 0.1s per fetch + processing overhead, should complete in < 2s total
            assert total_time < 2.0, \
                f"Concurrent analysis took {total_time:.2f}s, expected <2.0s"
            
            # Verify concurrent quota tracking
            # Should have made exactly 5 API calls (one per event)
            assert agent.hybrid_odds_service.fetch_event_odds.call_count == 5