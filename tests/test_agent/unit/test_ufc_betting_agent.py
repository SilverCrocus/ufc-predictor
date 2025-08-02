"""
Unit Tests for UFC Betting Agent

Comprehensive unit tests for the main UFC Betting Agent orchestrator,
including Phase 2A hybrid odds integration and Enhanced ML Pipeline components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import pandas as pd
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tests.test_agent.utils.agent_test_helpers import AgentTestHelpers, DataValidationHelpers
from tests.test_agent.utils.async_test_utilities import AsyncTestRunner


@pytest.mark.agent
class TestUFCBettingAgentInitialization:
    """Test UFC Betting Agent initialization and setup"""
    
    def test_agent_initialization_success(self, mock_agent_config):
        """Test successful agent initialization with proper configuration"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        assert agent.config == mock_agent_config
        assert agent.betting_system is None
        assert not agent.is_running
        assert agent.hybrid_odds_service is not None  # Phase 2A component
    
    def test_agent_initialization_with_invalid_config(self):
        """Test agent initialization handles invalid configuration"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        with pytest.raises((ValueError, AttributeError)):
            UFCBettingAgent(None)
    
    def test_hybrid_odds_service_initialization(self, mock_agent_config):
        """Test Phase 2A hybrid odds service initialization"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        # Should have hybrid service initialized
        assert agent.hybrid_odds_service is not None
        assert hasattr(agent.hybrid_odds_service, 'fetch_event_odds')
        assert hasattr(agent.hybrid_odds_service, 'get_quota_status')
    
    def test_betting_system_initialization_success(self, mock_agent_config):
        """Test successful betting system initialization"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models') as mock_models, \
             patch('pandas.read_csv') as mock_csv, \
             patch('joblib.load') as mock_joblib, \
             patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:
            
            # Setup mocks
            mock_models.return_value = {
                'version': 'test_v1.0',
                'winner_model_path': Path('test_winner.pkl'),
                'method_model_path': Path('test_method.pkl'),
                'winner_cols_path': Path('test_winner_cols.json'),
                'method_cols_path': Path('test_method_cols.json'),
                'fighters_data_path': Path('test_fighters.csv')
            }
            
            mock_csv.return_value = AgentTestHelpers.create_mock_betting_system()['fighters_df']
            mock_json.return_value = ['feature1', 'feature2', 'feature3']
            
            mock_model = Mock()
            mock_model.predict_proba.return_value = [[0.4, 0.6]]
            mock_joblib.return_value = mock_model
            
            # Initialize betting system
            result = agent.initialize_betting_system()
            
            assert result is True
            assert agent.betting_system is not None
            assert 'fighters_df' in agent.betting_system
            assert 'winner_model' in agent.betting_system
            assert 'model_version' in agent.betting_system
            assert agent.prediction_service is not None
    
    def test_betting_system_initialization_failure(self, mock_agent_config):
        """Test betting system initialization handles failures gracefully"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        with patch('src.agent.ufc_betting_agent.get_latest_trained_models') as mock_models:
            mock_models.side_effect = FileNotFoundError("Model files not found")
            
            result = agent.initialize_betting_system()
            
            assert result is False
            assert agent.betting_system is None


@pytest.mark.agent
@pytest.mark.asyncio
class TestUFCBettingAgentAnalysis:
    """Test event analysis workflows"""
    
    async def test_analyze_event_complete_workflow_phase2a(self, mock_agent_config, async_test_runner):
        """Test complete event analysis workflow with Phase 2A hybrid service"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        # Setup betting system
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        agent.prediction_service = Mock()
        agent.betting_service = Mock()
        
        # Setup hybrid odds service
        agent.hybrid_odds_service = AsyncMock()
        
        test_fights = ["Fighter A vs Fighter B", "Fighter C vs Fighter D"]
        hybrid_result = AgentTestHelpers.create_mock_hybrid_result(test_fights, api_requests_used=1)
        agent.hybrid_odds_service.fetch_event_odds.return_value = hybrid_result
        
        # Setup prediction service
        mock_predictions = Mock()
        mock_predictions.summary = {
            'successful_predictions': 2,
            'failed_predictions': 0,
            'upset_opportunities': 1,
            'high_confidence_picks': 1,
            'method_breakdown': {'Decision': 2}
        }
        agent.prediction_service.predict_event.return_value = mock_predictions
        
        # Setup betting service
        mock_betting = Mock()
        mock_betting.single_bets = []
        mock_betting.portfolio_summary = Mock(
            total_recommended_stake=100.0,
            expected_return=110.0,
            portfolio_ev=0.10,
            bankroll_utilization=0.10
        )
        agent.betting_service.generate_betting_recommendations.return_value = mock_betting
        
        # Run analysis with timeout
        with patch.object(agent, '_export_complete_analysis', return_value='/test/path/analysis.json'):
            result = await async_test_runner.run_with_timeout(
                agent.analyze_event("UFC Test Event 300", test_fights),
                timeout=10.0
            )
        
        # Verify Phase 2A integration
        assert result['status'] == 'completed'
        assert result['event_name'] == "UFC Test Event 300"
        assert 'odds_result' in result
        assert 'api_requests_used' in result['odds_result']
        assert 'confidence_score' in result['odds_result']
        assert 'fallback_activated' in result['odds_result']
        
        # Verify hybrid service was called with correct parameters
        agent.hybrid_odds_service.fetch_event_odds.assert_called_once()
        call_args = agent.hybrid_odds_service.fetch_event_odds.call_args
        assert call_args[0][0] == "UFC Test Event 300"  # event_name
        assert call_args[0][1] == test_fights  # target_fights
        assert 'priority' in call_args.kwargs  # Priority should be determined
    
    async def test_analyze_event_quota_exhaustion_fallback(self, mock_agent_config):
        """Test analysis handles API quota exhaustion with TAB fallback"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        agent.prediction_service = Mock()
        agent.betting_service = Mock()
        
        # Setup hybrid service with quota exhaustion
        agent.hybrid_odds_service = AsyncMock()
        test_fights = ["Fighter A vs Fighter B"]
        
        # Simulate quota exhaustion (fallback only)
        AgentTestHelpers.simulate_quota_exhaustion(agent.hybrid_odds_service)
        
        # Setup mocks for successful completion despite quota exhaustion
        mock_predictions = Mock()
        mock_predictions.summary = {'successful_predictions': 1, 'failed_predictions': 0}
        agent.prediction_service.predict_event.return_value = mock_predictions
        
        mock_betting = Mock()
        mock_betting.single_bets = []
        mock_betting.portfolio_summary = Mock(
            total_recommended_stake=50.0,
            expected_return=55.0,
            portfolio_ev=0.10,
            bankroll_utilization=0.05
        )
        agent.betting_service.generate_betting_recommendations.return_value = mock_betting
        
        with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
            result = await agent.analyze_event("Test Event", test_fights)
        
        # Should complete successfully even with quota exhaustion
        assert result['status'] == 'completed'
        assert result['odds_result']['api_requests_used'] == 0
        assert result['odds_result']['fallback_activated'] is True
    
    async def test_analyze_event_priority_determination(self, mock_agent_config):
        """Test event priority determination for quota allocation"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        # Test different event types
        test_cases = [
            ("UFC 300 - Title Fight", ["Champion vs Challenger"], "CRITICAL"),
            ("UFC Fight Night", ["Fighter A vs Fighter B", "Fighter C vs Fighter D"], "HIGH"),
            ("UFC Regular Event", ["Fighter A vs Fighter B"], "MEDIUM"),
            ("Test Event", ["Fighter A vs Fighter B"], "LOW")
        ]
        
        for event_name, fights, expected_priority in test_cases:
            priority = agent._determine_event_priority(event_name, fights)
            
            # Verify priority matches expectation
            assert priority.value == expected_priority, \
                f"Event '{event_name}' should have {expected_priority} priority, got {priority.value}"


@pytest.mark.agent
@pytest.mark.asyncio  
class TestUFCBettingAgentErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_analyze_event_without_betting_system(self, mock_agent_config):
        """Test analysis fails appropriately without betting system"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        with pytest.raises(RuntimeError, match="Betting system not initialized"):
            await agent.analyze_event("Test Event", ["Fighter A vs Fighter B"])
    
    async def test_analyze_event_odds_failure_recovery(self, mock_agent_config):
        """Test analysis handles odds fetching failure gracefully"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        
        # Setup failed hybrid service
        agent.hybrid_odds_service = AsyncMock()
        failed_result = Mock()
        failed_result.reconciled_data = None
        failed_result.api_requests_used = 0
        failed_result.confidence_score = 0.0
        failed_result.fallback_activated = True
        agent.hybrid_odds_service.fetch_event_odds.return_value = failed_result
        
        result = await agent.analyze_event("Test Event", ["Fighter A vs Fighter B"])
        
        assert result['status'] == 'failed'
        assert 'error' in result
        assert 'no data retrieved' in result['error'].lower()
    
    async def test_analyze_event_prediction_failure_handling(self, mock_agent_config):
        """Test analysis handles prediction service failures"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        
        # Setup successful odds but failing predictions
        agent.hybrid_odds_service = AsyncMock()
        test_fights = ["Fighter A vs Fighter B"]
        hybrid_result = AgentTestHelpers.create_mock_hybrid_result(test_fights)
        agent.hybrid_odds_service.fetch_event_odds.return_value = hybrid_result
        
        agent.prediction_service = Mock()
        agent.prediction_service.predict_event.side_effect = Exception("Prediction model error")
        
        result = await agent.analyze_event("Test Event", test_fights)
        
        assert result['status'] == 'failed'
        assert 'error' in result
        assert 'Prediction model error' in result['error']


@pytest.mark.agent
@pytest.mark.phase2a
class TestUFCBettingAgentPhase2AIntegration:
    """Test Phase 2A hybrid system integration"""
    
    async def test_system_status_includes_quota_info(self, mock_agent_config):
        """Test system status includes Phase 2A quota information"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        
        # Mock hybrid service with quota status
        agent.hybrid_odds_service = AsyncMock()
        agent.hybrid_odds_service.get_quota_status.return_value = {
            'quota_status': {
                'requests_used_today': 5,
                'requests_remaining_today': 11,
                'budget_remaining': 45.0
            },
            'hybrid_service_metrics': {
                'api_efficiency': 0.8
            }
        }
        
        status = await agent.get_system_status()
        
        assert 'quota_management' in status
        assert status['quota_management']['requests_remaining_today'] == 11
        assert status['quota_management']['budget_remaining'] == 45.0
        assert status['quota_management']['api_efficiency'] == 0.8
    
    async def test_health_check_includes_hybrid_components(self, mock_agent_config):
        """Test health check includes Phase 2A hybrid service components"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        
        # Mock hybrid service health check
        agent.hybrid_odds_service = AsyncMock()
        agent.hybrid_odds_service.health_check.return_value = {
            'overall_status': 'healthy',
            'components': {
                'quota_manager': 'healthy',
                'api_client': 'healthy',
                'tab_adapter': 'healthy'
            }
        }
        agent.hybrid_odds_service.get_quota_status.return_value = {
            'quota_status': {
                'requests_remaining_today': 10,
                'budget_remaining': 40.0
            }
        }
        
        health = await agent.health_check()
        
        assert 'hybrid_odds_service' in health['checks']
        assert health['checks']['hybrid_odds_service']['status'] == 'healthy'
        assert 'quota_management' in health['checks']
        assert health['checks']['quota_management']['status'] == 'healthy'


@pytest.mark.agent
@pytest.mark.performance
class TestUFCBettingAgentPerformance:
    """Test performance characteristics"""
    
    async def test_analysis_performance_target(self, mock_agent_config, performance_benchmark):
        """Test that event analysis meets performance targets"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        agent.betting_system = AgentTestHelpers.create_mock_betting_system()
        
        # Setup fast mocks
        agent.hybrid_odds_service = AsyncMock()
        test_fights = ["Fighter A vs Fighter B"]
        hybrid_result = AgentTestHelpers.create_mock_hybrid_result(test_fights)
        agent.hybrid_odds_service.fetch_event_odds.return_value = hybrid_result
        
        agent.prediction_service = Mock()
        mock_analysis = Mock()
        mock_analysis.summary = {'successful_predictions': 1, 'failed_predictions': 0}
        agent.prediction_service.predict_event.return_value = mock_analysis
        
        agent.betting_service = Mock()
        mock_betting = Mock()
        mock_betting.single_bets = []
        mock_betting.portfolio_summary = Mock(total_recommended_stake=50.0)
        agent.betting_service.generate_betting_recommendations.return_value = mock_betting
        
        # Benchmark the analysis
        with patch.object(agent, '_export_complete_analysis', return_value='/test/path'):
            
            async def analysis_operation():
                return await agent.analyze_event("Performance Test", test_fights)
            
            perf_result = performance_benchmark(lambda: asyncio.run(analysis_operation()))
        
        # Should complete within performance target (< 5 seconds for single fight)
        assert perf_result['execution_time_ms'] < 5000, \
            f"Analysis took {perf_result['execution_time_ms']:.1f}ms, target is <5000ms"


@pytest.mark.agent
class TestUFCBettingAgentDataValidation:
    """Test data validation and consistency"""
    
    def test_prediction_result_validation(self, mock_agent_config):
        """Test that prediction results meet quality standards"""
        from ufc_predictor.agent.ufc_betting_agent import UFCBettingAgent
        
        agent = UFCBettingAgent(mock_agent_config)
        
        # Test symmetrical prediction validation
        betting_system = AgentTestHelpers.create_mock_betting_system()
        predict_func = betting_system['predict_function']
        
        # Validate prediction symmetry
        DataValidationHelpers.validate_prediction_symmetry(
            predict_func, 
            "Fighter A", 
            "Fighter B",
            fighters_df=betting_system['fighters_df'],
            winner_cols=betting_system['winner_cols'],
            method_cols=betting_system['method_cols'],
            winner_model=betting_system['winner_model'],
            method_model=betting_system['method_model']
        )
    
    def test_odds_data_validation(self, mock_agent_config):
        """Test odds data validation meets UFC standards"""
        test_fights = ["Fighter A vs Fighter B", "Fighter C vs Fighter D"]
        odds_data = AgentTestHelpers.create_mock_odds_data(test_fights)
        
        # Should pass UFC odds validation
        DataValidationHelpers.validate_odds_data(odds_data)
    
    def test_fighter_data_validation(self, mock_agent_config):
        """Test fighter data meets UFC data standards"""
        betting_system = AgentTestHelpers.create_mock_betting_system()
        fighter_data = betting_system['fighters_df']
        
        # Should pass UFC fighter data validation
        DataValidationHelpers.validate_ufc_fighter_data(fighter_data)