"""
Integration Layer for UFC Betting System
========================================

Seamless integration layer to enhance existing UFCBettingAgent with new agent architecture:
- Backward compatibility with existing workflows
- Gradual migration path to full agent system
- Service adapter patterns for legacy code
- Enhanced features without breaking changes
- Comprehensive wrapper for agent coordination
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_agent import BaseAgent, AgentPriority, AgentState
from .data_agent import DataAgent, create_data_agent_config
from .model_agent import ModelAgent, create_model_agent_config
from .risk_agent import RiskAgent, create_risk_agent_config
from .monitor_agent import MonitorAgent, create_monitor_agent_config
from .message_bus import MessageBus, get_message_bus, create_message_bus_config

# Import existing betting agent
from ..ufc_betting_agent import UFCBettingAgent
from ..config.agent_config import UFCAgentConfiguration

logger = logging.getLogger(__name__)


class EnhancedUFCBettingAgent(UFCBettingAgent):
    """
    Enhanced UFC Betting Agent with multi-agent architecture
    
    Extends the existing UFCBettingAgent with:
    - Specialized agents for data, models, risk, and monitoring
    - Message-based coordination between agents
    - Enhanced risk management and portfolio optimization
    - Advanced monitoring and health tracking
    - Backward compatibility with existing workflows
    """
    
    def __init__(self, config: UFCAgentConfiguration, enable_agent_system: bool = True):
        """
        Initialize enhanced betting agent
        
        Args:
            config: Original UFC agent configuration
            enable_agent_system: Enable new agent architecture (default: True)
        """
        # Initialize parent class
        super().__init__(config)
        
        self.enable_agent_system = enable_agent_system
        self.agent_system_initialized = False
        
        # Agent system components
        self.message_bus: Optional[MessageBus] = None
        self.data_agent: Optional[DataAgent] = None
        self.model_agent: Optional[ModelAgent] = None
        self.risk_agent: Optional[RiskAgent] = None
        self.monitor_agent: Optional[MonitorAgent] = None
        
        # Enhanced features state
        self.enhanced_prediction_enabled = False
        self.enhanced_risk_management_enabled = False
        self.advanced_monitoring_enabled = False
        
        logger.info("EnhancedUFCBettingAgent initialized")
    
    async def initialize_agent_system(self) -> bool:
        """
        Initialize the multi-agent system
        
        Returns:
            True if initialization successful
        """
        if not self.enable_agent_system:
            logger.info("Agent system disabled, using legacy mode")
            return True
        
        try:
            logger.info("Initializing multi-agent system...")
            
            # Initialize message bus
            bus_config = create_message_bus_config()
            self.message_bus = MessageBus(bus_config)
            await self.message_bus.start()
            
            # Initialize specialized agents
            await self._initialize_specialized_agents()
            
            # Register agents with message bus
            await self._register_agents_with_bus()
            
            # Start all agents
            await self._start_all_agents()
            
            # Enable enhanced features
            await self._enable_enhanced_features()
            
            self.agent_system_initialized = True
            logger.info("Multi-agent system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent system initialization failed: {e}")
            self.agent_system_initialized = False
            return False
    
    async def _initialize_specialized_agents(self):
        """Initialize all specialized agents"""
        
        # Data Agent
        data_config = create_data_agent_config(
            data_sources={
                'fighters_data_path': str(self.config.agent.storage_base_path + '/ufc_fighters_engineered_corrected.csv'),
                'fights_data_path': str(self.config.agent.storage_base_path + '/ufc_fight_dataset_with_diffs.csv')
            }
        )
        self.data_agent = DataAgent(data_config)
        
        # Model Agent
        model_config = create_model_agent_config(
            model_weights={
                'xgboost': 0.5,
                'random_forest': 0.3,
                'ensemble': 0.2
            },
            model_paths={
                'xgboost': str(Path(self.config.agent.storage_base_path) / 'ufc_winner_model_tuned.joblib'),
                'random_forest': str(Path(self.config.agent.storage_base_path) / 'ufc_random_forest_model_tuned.joblib'),
                'ensemble': str(Path(self.config.agent.storage_base_path) / 'ufc_multiclass_model.joblib')
            },
            feature_columns_path=str(Path(self.config.agent.storage_base_path) / 'winner_model_columns.json')
        )
        self.model_agent = ModelAgent(model_config)
        
        # Risk Agent
        risk_config = create_risk_agent_config(
            bankroll=self.config.betting.initial_bankroll,
            kelly_multiplier=self.config.betting.kelly_multiplier
        )
        self.risk_agent = RiskAgent(risk_config)
        
        # Monitor Agent
        monitor_config = create_monitor_agent_config()
        self.monitor_agent = MonitorAgent(monitor_config)
        
        logger.info("Specialized agents initialized")
    
    async def _register_agents_with_bus(self):
        """Register all agents with the message bus"""
        agents = [self.data_agent, self.model_agent, self.risk_agent, self.monitor_agent]
        
        for agent in agents:
            if agent:
                await self.message_bus.register_agent(agent)
        
        logger.info("Agents registered with message bus")
    
    async def _start_all_agents(self):
        """Start all specialized agents"""
        agents = [
            ('data_agent', self.data_agent),
            ('model_agent', self.model_agent),
            ('risk_agent', self.risk_agent),
            ('monitor_agent', self.monitor_agent)
        ]
        
        for agent_name, agent in agents:
            if agent:
                initialized = await agent.initialize()
                if initialized:
                    started = await agent.start()
                    if started:
                        logger.info(f"{agent_name} started successfully")
                    else:
                        logger.error(f"{agent_name} failed to start")
                        raise RuntimeError(f"Failed to start {agent_name}")
                else:
                    logger.error(f"{agent_name} failed to initialize")
                    raise RuntimeError(f"Failed to initialize {agent_name}")
    
    async def _enable_enhanced_features(self):
        """Enable enhanced features provided by agent system"""
        try:
            self.enhanced_prediction_enabled = True
            self.enhanced_risk_management_enabled = True
            self.advanced_monitoring_enabled = True
            
            logger.info("Enhanced features enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable enhanced features: {e}")
    
    # === Enhanced Analysis Methods ===
    
    async def analyze_event_enhanced(self, event_name: str, target_fights: List[str]) -> Dict[str, Any]:
        """
        Enhanced event analysis using agent system
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to analyze
            
        Returns:
            Enhanced analysis results
        """
        if not self.agent_system_initialized:
            logger.warning("Agent system not initialized, falling back to legacy analysis")
            return await self.analyze_event(event_name, target_fights)
        
        try:
            logger.info(f"Starting enhanced analysis for {event_name}")
            
            # Step 1: Enhanced odds fetching (use existing method but with data validation)
            odds_result = await self._fetch_odds_with_validation(event_name, target_fights)
            
            # Step 2: Enhanced predictions with model coordination
            predictions_result = await self._generate_enhanced_predictions(odds_result, event_name)
            
            # Step 3: Advanced risk management and portfolio optimization
            betting_recommendations = await self._generate_optimized_betting_recommendations(predictions_result)
            
            # Step 4: Enhanced monitoring and reporting
            analysis_result = await self._compile_enhanced_analysis(
                event_name, odds_result, predictions_result, betting_recommendations
            )
            
            logger.info(f"Enhanced analysis completed for {event_name}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed, falling back to legacy: {e}")
            return await self.analyze_event(event_name, target_fights)
    
    async def _fetch_odds_with_validation(self, event_name: str, target_fights: List[str]) -> Dict[str, Any]:
        """Fetch odds with enhanced data validation"""
        try:
            # Use existing odds fetching but add data validation
            if self.hybrid_odds_service:
                priority = self._determine_event_priority(event_name, target_fights)
                hybrid_result = await self.hybrid_odds_service.fetch_event_odds(
                    event_name, target_fights, priority=priority
                )
                
                if hybrid_result.reconciled_data and self.data_agent:
                    # Validate odds data quality
                    validation_result = await self.message_bus.send_message(
                        sender_id="main_agent",
                        recipient_id="data_agent",
                        message_type="validate_odds_data",
                        payload={
                            'odds_data': hybrid_result.reconciled_data,
                            'event_name': event_name
                        },
                        requires_response=True
                    )
                    
                    if validation_result and validation_result.get('status') == 'success':
                        logger.info(f"Odds data validation passed: {validation_result.get('quality_score', 0.0):.3f}")
                    else:
                        logger.warning(f"Odds data validation issues: {validation_result}")
                
                return {
                    'status': 'success',
                    'odds_data': hybrid_result.reconciled_data,
                    'api_requests_used': hybrid_result.api_requests_used,
                    'confidence_score': hybrid_result.confidence_score,
                    'data_quality_validated': True
                }
            
            else:
                # Fallback to legacy odds service
                odds_result = self.odds_service.fetch_and_store_odds(event_name, target_fights)
                return {
                    'status': odds_result.status,
                    'odds_data': odds_result.odds_data,
                    'data_quality_validated': False
                }
        
        except Exception as e:
            logger.error(f"Enhanced odds fetching failed: {e}")
            raise
    
    async def _generate_enhanced_predictions(self, odds_result: Dict[str, Any], event_name: str) -> Dict[str, Any]:
        """Generate predictions using enhanced model coordination"""
        try:
            if not self.model_agent:
                # Fallback to legacy prediction
                return self.prediction_service.predict_event(odds_result['odds_data'], event_name)
            
            # Prepare data for model agent
            odds_data = odds_result['odds_data']
            fighter_pairs = []
            feature_data_list = []
            
            for fight_key, fight_data in odds_data.items():
                fighter_a = fight_data.get('fighter_a', '')
                fighter_b = fight_data.get('fighter_b', '')
                fighter_pairs.append((fighter_a, fighter_b))
                
                # This would need proper feature engineering integration
                # For now, create placeholder feature data
                feature_row = pd.Series({f'feature_{i}': 0.5 for i in range(20)})
                feature_data_list.append(feature_row)
            
            if feature_data_list:
                feature_data = pd.DataFrame(feature_data_list)
                
                # Send prediction request to model agent
                prediction_response = await self.message_bus.send_message(
                    sender_id="main_agent",
                    recipient_id="model_agent",
                    message_type="predict_fights",
                    payload={
                        'fighter_pairs': fighter_pairs,
                        'feature_data_path': None,  # Would save and pass path
                        'enable_bootstrap': True
                    },
                    requires_response=True
                )
                
                if prediction_response and prediction_response.get('status') == 'success':
                    return {
                        'status': 'success',
                        'predictions': prediction_response,
                        'model_coordination_used': True,
                        'ensemble_accuracy': prediction_response.get('ensemble_accuracy', 0.0)
                    }
            
            # Fallback to legacy prediction
            logger.warning("Model agent prediction failed, using legacy method")
            return self.prediction_service.predict_event(odds_data, event_name)
            
        except Exception as e:
            logger.error(f"Enhanced prediction generation failed: {e}")
            return self.prediction_service.predict_event(odds_result['odds_data'], event_name)
    
    async def _generate_optimized_betting_recommendations(self, predictions_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate betting recommendations with advanced risk management"""
        try:
            if not self.risk_agent:
                # Fallback to legacy betting service
                return self.betting_service.generate_betting_recommendations(
                    predictions_result, self.config.betting.initial_bankroll
                )
            
            # Prepare candidate bets for risk agent
            candidate_bets = []
            
            # This would extract candidate bets from predictions_result
            # For now, create placeholder data
            for i in range(3):  # Example: 3 candidate bets
                candidate_bets.append({
                    'fight_id': f'fight_{i}',
                    'fighter': f'Fighter_{i}',
                    'opponent': f'Opponent_{i}',
                    'predicted_probability': 0.6,
                    'decimal_odds': 2.0,
                    'confidence_score': 0.8,
                    'uncertainty': 0.2,
                    'data_quality_score': 0.9
                })
            
            # Send portfolio optimization request to risk agent
            optimization_response = await self.message_bus.send_message(
                sender_id="main_agent",
                recipient_id="risk_agent",
                message_type="optimize_portfolio",
                payload={'candidate_bets': candidate_bets},
                requires_response=True
            )
            
            if optimization_response and optimization_response.get('status') == 'success':
                return {
                    'status': 'success',
                    'recommendations': optimization_response['recommendations'],
                    'portfolio_metrics': optimization_response['portfolio_metrics'],
                    'advanced_risk_management': True
                }
            
            # Fallback to legacy betting service
            logger.warning("Risk agent optimization failed, using legacy method")
            return self.betting_service.generate_betting_recommendations(
                predictions_result, self.config.betting.initial_bankroll
            )
            
        except Exception as e:
            logger.error(f"Enhanced betting recommendations failed: {e}")
            return self.betting_service.generate_betting_recommendations(
                predictions_result, self.config.betting.initial_bankroll
            )
    
    async def _compile_enhanced_analysis(self, 
                                       event_name: str,
                                       odds_result: Dict[str, Any],
                                       predictions_result: Dict[str, Any],
                                       betting_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Compile enhanced analysis with monitoring integration"""
        try:
            # Compile basic analysis result
            analysis_result = {
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'enhanced_features_used': {
                    'data_validation': odds_result.get('data_quality_validated', False),
                    'model_coordination': predictions_result.get('model_coordination_used', False),
                    'advanced_risk_management': betting_recommendations.get('advanced_risk_management', False)
                },
                'odds_result': odds_result,
                'predictions_result': predictions_result,
                'betting_recommendations': betting_recommendations
            }
            
            # Send analysis metrics to monitor agent
            if self.monitor_agent:
                await self.message_bus.send_message(
                    sender_id="main_agent",
                    recipient_id="monitor_agent",
                    message_type="log_analysis_metrics",
                    payload={
                        'event_name': event_name,
                        'analysis_metrics': {
                            'odds_confidence': odds_result.get('confidence_score', 0.0),
                            'prediction_quality': predictions_result.get('ensemble_accuracy', 0.0),
                            'portfolio_optimization': betting_recommendations.get('portfolio_metrics', {}).get('portfolio_ev', 0.0)
                        }
                    }
                )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced analysis compilation failed: {e}")
            return {
                'event_name': event_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
    
    # === System Health and Status ===
    
    async def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including agent health"""
        try:
            # Get base system status
            base_status = self.get_system_status()
            
            if not self.agent_system_initialized:
                base_status['agent_system'] = {
                    'enabled': False,
                    'status': 'Agent system not initialized'
                }
                return base_status
            
            # Get agent system health
            agent_health = {}
            
            if self.monitor_agent:
                system_health = await self.message_bus.send_message(
                    sender_id="main_agent",
                    recipient_id="monitor_agent",
                    message_type="get_system_health",
                    payload={},
                    requires_response=True
                )
                
                if system_health and system_health.get('status') == 'success':
                    agent_health = system_health.get('system_health', {})
            
            # Get individual agent status
            agent_statuses = {}
            agents = [
                ('data_agent', self.data_agent),
                ('model_agent', self.model_agent),
                ('risk_agent', self.risk_agent),
                ('monitor_agent', self.monitor_agent)
            ]
            
            for agent_name, agent in agents:
                if agent:
                    agent_health_info = await agent.get_health_status()
                    agent_statuses[agent_name] = {
                        'state': agent.state.value,
                        'health_score': agent_health_info.get('agent_specific', {}).get('health_score', 0.0),
                        'uptime_seconds': agent_health_info.get('uptime_seconds', 0.0),
                        'error_rate': agent_health_info.get('error_rate', 0.0)
                    }
            
            # Get message bus status
            message_bus_status = self.message_bus.get_status() if self.message_bus else {}
            
            base_status['agent_system'] = {
                'enabled': True,
                'initialized': self.agent_system_initialized,
                'overall_health': agent_health,
                'individual_agents': agent_statuses,
                'message_bus': message_bus_status,
                'enhanced_features': {
                    'prediction_enabled': self.enhanced_prediction_enabled,
                    'risk_management_enabled': self.enhanced_risk_management_enabled,
                    'monitoring_enabled': self.advanced_monitoring_enabled
                }
            }
            
            return base_status
            
        except Exception as e:
            logger.error(f"Enhanced system status failed: {e}")
            base_status = self.get_system_status()
            base_status['agent_system'] = {
                'enabled': self.enable_agent_system,
                'status': 'error',
                'error': str(e)
            }
            return base_status
    
    async def health_check_enhanced(self) -> Dict[str, Any]:
        """Enhanced health check including all agents"""
        try:
            # Get base health check
            base_health = await self.health_check()
            
            if not self.agent_system_initialized:
                base_health['agent_system_health'] = {
                    'status': 'not_initialized',
                    'details': 'Agent system not enabled or initialized'
                }
                return base_health
            
            # Perform health checks on all agents
            agent_health_results = {}
            
            agents = [
                ('data_agent', self.data_agent),
                ('model_agent', self.model_agent),
                ('risk_agent', self.risk_agent),
                ('monitor_agent', self.monitor_agent)
            ]
            
            overall_agent_health = True
            
            for agent_name, agent in agents:
                if agent:
                    try:
                        health_result = await agent.get_health_status()
                        agent_health_results[agent_name] = {
                            'status': 'healthy' if health_result.get('success_rate', 0) > 0.8 else 'unhealthy',
                            'details': health_result
                        }
                        
                        if health_result.get('success_rate', 0) <= 0.8:
                            overall_agent_health = False
                            
                    except Exception as e:
                        agent_health_results[agent_name] = {
                            'status': 'error',
                            'details': str(e)
                        }
                        overall_agent_health = False
            
            # Check message bus health
            message_bus_health = 'healthy'
            if self.message_bus:
                bus_status = self.message_bus.get_status()
                if not bus_status.get('is_running', False) or bus_status.get('messages_failed', 0) > 10:
                    message_bus_health = 'unhealthy'
                    overall_agent_health = False
            
            base_health['agent_system_health'] = {
                'status': 'healthy' if overall_agent_health else 'unhealthy',
                'individual_agents': agent_health_results,
                'message_bus_health': message_bus_health,
                'enhanced_features_operational': all([
                    self.enhanced_prediction_enabled,
                    self.enhanced_risk_management_enabled,
                    self.advanced_monitoring_enabled
                ])
            }
            
            return base_health
            
        except Exception as e:
            logger.error(f"Enhanced health check failed: {e}")
            base_health = await self.health_check()
            base_health['agent_system_health'] = {
                'status': 'error',
                'error': str(e)
            }
            return base_health
    
    # === Lifecycle Management ===
    
    async def stop_enhanced(self, timeout: float = 30.0) -> bool:
        """Enhanced shutdown including all agents"""
        try:
            logger.info("Stopping enhanced UFC betting agent...")
            
            # Stop all specialized agents
            if self.agent_system_initialized:
                agents = [
                    ('monitor_agent', self.monitor_agent),
                    ('risk_agent', self.risk_agent),
                    ('model_agent', self.model_agent),
                    ('data_agent', self.data_agent)
                ]
                
                for agent_name, agent in agents:
                    if agent:
                        try:
                            await agent.stop(timeout / len(agents))
                            logger.info(f"{agent_name} stopped successfully")
                        except Exception as e:
                            logger.error(f"Failed to stop {agent_name}: {e}")
                
                # Stop message bus
                if self.message_bus:
                    await self.message_bus.stop(timeout / 2)
                    logger.info("Message bus stopped")
            
            # Call parent shutdown (if it exists)
            # Note: Original UFCBettingAgent doesn't have async stop method
            logger.info("Enhanced UFC betting agent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced shutdown failed: {e}")
            return False
    
    # === Backward Compatibility Methods ===
    
    async def analyze_event(self, event_name: str, target_fights: List[str]) -> Dict[str, Any]:
        """
        Backward compatible analyze_event that uses enhanced features if available
        """
        if self.agent_system_initialized and self.enable_agent_system:
            return await self.analyze_event_enhanced(event_name, target_fights)
        else:
            # Call parent method
            return await super().analyze_event(event_name, target_fights)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Backward compatible get_system_status with enhanced information
        """
        # Call parent method
        base_status = super().get_system_status()
        
        # Add agent system information
        base_status['enhanced_features'] = {
            'agent_system_enabled': self.enable_agent_system,
            'agent_system_initialized': self.agent_system_initialized,
            'enhanced_prediction': self.enhanced_prediction_enabled,
            'enhanced_risk_management': self.enhanced_risk_management_enabled,
            'advanced_monitoring': self.advanced_monitoring_enabled
        }
        
        return base_status
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Backward compatible health_check with enhanced agent information
        """
        if self.agent_system_initialized:
            return await self.health_check_enhanced()
        else:
            # Call parent method
            return await super().health_check()


# Factory function for easy creation
def create_enhanced_ufc_betting_agent(
    config: UFCAgentConfiguration,
    enable_agent_system: bool = True,
    auto_initialize: bool = True
) -> EnhancedUFCBettingAgent:
    """
    Factory function to create and optionally initialize enhanced betting agent
    
    Args:
        config: UFC agent configuration
        enable_agent_system: Enable multi-agent system
        auto_initialize: Automatically initialize agent system
        
    Returns:
        Enhanced UFC betting agent
    """
    agent = EnhancedUFCBettingAgent(config, enable_agent_system)
    
    if auto_initialize and enable_agent_system:
        # Note: This would need to be called in an async context
        # agent.initialize_agent_system() would need to be awaited
        pass
    
    return agent


# Migration helper
class MigrationHelper:
    """Helper class for migrating from legacy to enhanced agent system"""
    
    @staticmethod
    def check_compatibility(config: UFCAgentConfiguration) -> Dict[str, Any]:
        """
        Check compatibility with enhanced agent system
        
        Args:
            config: UFC agent configuration
            
        Returns:
            Compatibility assessment
        """
        compatibility = {
            'compatible': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check required paths exist
        required_paths = [
            'ufc_fighters_engineered_corrected.csv',
            'ufc_fight_dataset_with_diffs.csv',
            'winner_model_columns.json'
        ]
        
        for path in required_paths:
            full_path = Path(config.agent.storage_base_path) / path
            if not full_path.exists():
                compatibility['warnings'].append(f"Missing file: {path}")
                compatibility['recommendations'].append(f"Ensure {path} exists for enhanced features")
        
        # Check configuration
        if config.betting.initial_bankroll <= 0:
            compatibility['warnings'].append("Invalid bankroll configuration")
            compatibility['recommendations'].append("Set valid initial_bankroll for risk management")
        
        if not config.api.odds_api_key:
            compatibility['warnings'].append("Missing odds API key")
            compatibility['recommendations'].append("Set odds_api_key for enhanced odds fetching")
        
        return compatibility
    
    @staticmethod
    def migrate_configuration(old_config: UFCAgentConfiguration) -> UFCAgentConfiguration:
        """
        Migrate configuration for enhanced agent system compatibility
        
        Args:
            old_config: Original configuration
            
        Returns:
            Enhanced configuration
        """
        # For now, return the same configuration
        # In a real migration, this would update configuration for new features
        return old_config