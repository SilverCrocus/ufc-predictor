"""
UFC Betting Agent

Main orchestrator for the automated UFC betting agent.
Integrates all services and provides a complete automated betting workflow.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import json

# Add project root to path for imports
sys.path.append('.')
sys.path.append('..')

from .services.prediction_service import UFCPredictionService, PredictionAnalysis
from .services.odds_service import UFCOddsService, OddsResult
from .services.betting_service import UFCBettingService, BettingRecommendations
from .services.async_odds_client import AsyncUFCOddsAPIClient
from .services.hybrid_odds_service import HybridOddsService, create_hybrid_odds_service, RequestPriority
from .services.quota_manager import QuotaManager, create_quota_manager
from .config.agent_config import UFCAgentConfiguration, EventConfig
from ufc_predictor.betting.odds_api_integration import UFCOddsAPIClient

logger = logging.getLogger(__name__)


class UFCBettingAgent:
    """
    Main UFC betting agent orchestrator
    
    Coordinates all services to provide automated UFC betting analysis and recommendations.
    Extracted from Jupyter notebook workflow for production use.
    """
    
    def __init__(self, config: UFCAgentConfiguration):
        """
        Initialize the UFC betting agent
        
        Args:
            config: Complete agent configuration
        """
        self.config = config
        self.betting_system = None
        self.is_running = False
        
        # Initialize services
        self._initialize_services()
        
        logger.info("UFC Betting Agent initialized successfully")
    
    def _initialize_services(self):
        """Initialize all agent services with Phase 2A hybrid architecture"""
        # Initialize API client (sync for now, can be upgraded to async)
        self.odds_api_client = UFCOddsAPIClient(self.config.api.odds_api_key)
        
        # Initialize Phase 2A quota-aware hybrid odds service
        try:
            # Create quota manager with 500 monthly request limit
            daily_quota = 16  # ~500/month
            monthly_budget = 50.0  # $50 monthly budget
            
            self.hybrid_odds_service = create_hybrid_odds_service(
                self.config.api.odds_api_key,
                daily_quota=daily_quota,
                monthly_budget=monthly_budget
            )
            
            logger.info(f"Phase 2A Hybrid Odds Service initialized with {daily_quota} daily quota")
            
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid service, falling back to basic: {e}")
            # Fallback to original odds service
            self.hybrid_odds_service = None
            self.odds_service = UFCOddsService(
                self.odds_api_client,
                storage_base_path=self.config.agent.odds_storage_path
            )
        
        self.betting_service = UFCBettingService()
        
        # Prediction service will be initialized after betting system is loaded
        self.prediction_service = None
        
        logger.info("Agent services initialized with Phase 2A enhancements")
    
    def initialize_betting_system(self) -> bool:
        """
        Initialize the betting system with models and data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing betting system...")
            
            # Use the same initialization logic as notebook Cell 1
            from main import get_latest_trained_models
            import pandas as pd
            import joblib
            import json
            
            # Get model paths
            paths = get_latest_trained_models()
            
            # Load fighter data
            fighters_data_path = paths['fighters_data_path']
            if not fighters_data_path.exists():
                fighters_data_path = Path('ufc_fighters_engineered_corrected.csv')
            
            fighters_df = pd.read_csv(fighters_data_path)
            logger.info(f"Loaded {len(fighters_df):,} fighter records")
            
            # Load column configurations  
            with open(paths['winner_cols_path'], 'r') as f:
                winner_cols = json.load(f)
            
            with open(paths['method_cols_path'], 'r') as f:
                method_cols = json.load(f)
            
            # Load trained models
            winner_model = joblib.load(paths['winner_model_path'])
            method_model = joblib.load(paths['method_model_path'])
            
            logger.info(f"Models loaded from version: {paths['version']}")
            
            # Import prediction function
            from ufc_predictor.core.prediction import predict_fight_symmetrical
            
            # Create betting system dictionary
            self.betting_system = {
                'fighters_df': fighters_df,
                'winner_cols': winner_cols,
                'method_cols': method_cols,
                'winner_model': winner_model,
                'method_model': method_model,
                'odds_client': self.odds_api_client,
                'predict_function': predict_fight_symmetrical,
                'model_version': paths['version'],
                'bankroll': self.config.betting.initial_bankroll,
                'api_key': self.config.api.odds_api_key
            }
            
            # Initialize prediction service now that we have betting system
            self.prediction_service = UFCPredictionService(self.betting_system)
            
            logger.info("Betting system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize betting system: {str(e)}")
            return False
    
    async def analyze_event(self, event_name: str, target_fights: List[str]) -> Dict[str, Any]:
        """
        Complete analysis workflow for a UFC event
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to analyze
            
        Returns:
            Dict containing complete analysis results
        """
        if not self.betting_system:
            raise RuntimeError("Betting system not initialized. Call initialize_betting_system() first.")
        
        logger.info(f"Starting complete analysis for {event_name}")
        
        analysis_result = {
            'event_name': event_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'odds_result': None,
            'predictions_analysis': None,
            'betting_recommendations': None,
            'error': None
        }
        
        try:
            # Step 1: Fetch odds using Phase 2A hybrid approach
            logger.info("Step 1: Fetching odds with smart hybrid system...")
            
            hybrid_result = None
            odds_result = None
            
            if self.hybrid_odds_service:
                # Use Phase 2A hybrid approach with intelligent quota management
                priority = self._determine_event_priority(event_name, target_fights)
                
                hybrid_result = await self.hybrid_odds_service.fetch_event_odds(
                    event_name, target_fights, priority=priority
                )
                
                if hybrid_result.reconciled_data:
                    # Convert hybrid result to standard format for backward compatibility
                    odds_result = OddsResult(
                        status='success',
                        odds_data=hybrid_result.reconciled_data,
                        total_fights=len(hybrid_result.reconciled_data),
                        csv_path=None,  # Will be set after storing
                        fetch_timestamp=datetime.now().isoformat()
                    )
                    
                    # Store odds to CSV for compatibility (simplified path creation)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_filename = f'odds_{event_name}_{timestamp}.csv'
                    csv_path = Path(self.config.agent.odds_storage_path) / csv_filename
                    
                    # Create CSV manually since we're bypassing the original service
                    import pandas as pd
                    odds_df = pd.DataFrame.from_dict(hybrid_result.reconciled_data, orient='index')
                    odds_df.to_csv(csv_path, index=False)
                    odds_result.csv_path = str(csv_path)
                    
                    analysis_result['odds_result'] = {
                        'status': 'success',
                        'total_fights': len(hybrid_result.reconciled_data),
                        'csv_path': odds_result.csv_path,
                        'api_requests_used': hybrid_result.api_requests_used,
                        'confidence_score': hybrid_result.confidence_score,
                        'fallback_activated': hybrid_result.fallback_activated,
                        'data_sources': list(set(
                            fight.get('data_sources', ['unknown']) 
                            for fight in hybrid_result.reconciled_data.values() 
                            if isinstance(fight.get('data_sources'), list)
                        ))
                    }
                    
                    logger.info(
                        f"Phase 2A hybrid fetch: {len(hybrid_result.reconciled_data)} fights, "
                        f"{hybrid_result.api_requests_used} API requests, "
                        f"confidence: {hybrid_result.confidence_score:.2f}"
                    )
                    
                else:
                    analysis_result['status'] = 'failed'
                    analysis_result['error'] = "Hybrid odds fetching failed: no data retrieved"
                    return analysis_result
                    
            else:
                # Fallback to original odds service
                logger.info("Using fallback odds service")
                odds_result = self.odds_service.fetch_and_store_odds(event_name, target_fights)
                analysis_result['odds_result'] = {
                    'status': odds_result.status,
                    'total_fights': odds_result.total_fights,
                    'csv_path': odds_result.csv_path
                }
                
                if odds_result.status != 'success':
                    analysis_result['status'] = 'failed'
                    analysis_result['error'] = f"Odds fetching failed: {odds_result.error}"
                    return analysis_result
            
            # Step 2: Generate predictions
            logger.info("Step 2: Generating predictions...")
            
            # Use odds data from hybrid result or fallback
            if hybrid_result and hybrid_result.reconciled_data:
                odds_data = hybrid_result.reconciled_data
            else:
                odds_data = odds_result.odds_data
            
            predictions_analysis = self.prediction_service.predict_event(
                odds_data, event_name
            )
            analysis_result['predictions_analysis'] = {
                'successful_predictions': predictions_analysis.summary['successful_predictions'],
                'failed_predictions': predictions_analysis.summary['failed_predictions'],
                'upset_opportunities': predictions_analysis.summary['upset_opportunities'],
                'high_confidence_picks': predictions_analysis.summary['high_confidence_picks']
            }
            
            # Step 3: Generate betting recommendations
            logger.info("Step 3: Generating betting recommendations...")
            betting_recommendations = self.betting_service.generate_betting_recommendations(
                predictions_analysis, self.config.betting.initial_bankroll
            )
            analysis_result['betting_recommendations'] = {
                'total_bets': len(betting_recommendations.single_bets),
                'total_stake': betting_recommendations.portfolio_summary.total_recommended_stake,
                'expected_return': betting_recommendations.portfolio_summary.expected_return,
                'portfolio_ev': betting_recommendations.portfolio_summary.portfolio_ev,
                'bankroll_utilization': betting_recommendations.portfolio_summary.bankroll_utilization
            }
            
            # Step 4: Export analysis
            logger.info("Step 4: Exporting analysis...")
            export_path = await self._export_complete_analysis(
                event_name, odds_result, predictions_analysis, betting_recommendations
            )
            analysis_result['export_path'] = export_path
            
            analysis_result['status'] = 'completed'
            
            logger.info(
                f"Analysis completed for {event_name}: "
                f"{len(betting_recommendations.single_bets)} bets recommended, "
                f"${betting_recommendations.portfolio_summary.total_recommended_stake:.2f} total stake"
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed for {event_name}: {str(e)}")
            analysis_result['status'] = 'failed'
            analysis_result['error'] = str(e)
            return analysis_result
    
    async def _export_complete_analysis(self, event_name: str, odds_result: OddsResult,
                                      predictions_analysis: PredictionAnalysis,
                                      betting_recommendations: BettingRecommendations) -> str:
        """
        Export complete analysis to file
        
        Args:
            event_name: Name of the event
            odds_result: Odds fetching result
            predictions_analysis: Prediction analysis result
            betting_recommendations: Betting recommendations
            
        Returns:
            Path to exported analysis file
        """
        # Create analysis export
        analysis_dir = Path(self.config.agent.analysis_export_path)
        analysis_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = analysis_dir / f'complete_analysis_{event_name}_{timestamp}.json'
        
        # Compile comprehensive analysis data
        analysis_export = {
            'metadata': {
                'event_name': event_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'agent_version': '1.0.0',
                'bankroll_tier': betting_recommendations.bankroll_info['tier'],
                'bankroll_amount': betting_recommendations.bankroll_info['amount']
            },
            'odds_summary': {
                'status': odds_result.status,
                'total_fights': odds_result.total_fights,
                'csv_path': odds_result.csv_path,
                'fetch_timestamp': odds_result.fetch_timestamp
            },
            'predictions_summary': predictions_analysis.summary,
            'betting_summary': {
                'total_bets': len(betting_recommendations.single_bets),
                'total_stake': betting_recommendations.portfolio_summary.total_recommended_stake,
                'expected_return': betting_recommendations.portfolio_summary.expected_return,
                'portfolio_ev': betting_recommendations.portfolio_summary.portfolio_ev,
                'bankroll_utilization': betting_recommendations.portfolio_summary.bankroll_utilization
            },
            'recommended_bets': [
                {
                    'fighter': bet.fighter,
                    'opponent': bet.opponent,
                    'fight': bet.fight,
                    'decimal_odds': bet.decimal_odds,
                    'recommended_stake': bet.recommended_stake,
                    'expected_value': bet.expected_value,
                    'potential_profit': bet.potential_profit,
                    'is_upset_opportunity': bet.is_upset_opportunity,
                    'is_high_confidence': bet.is_high_confidence
                }
                for bet in betting_recommendations.single_bets
            ],
            'fight_predictions': [
                {
                    'fight_key': fight.fight_key,
                    'fighter_a': fight.fighter_a,
                    'fighter_b': fight.fighter_b,
                    'model_favorite': fight.model_favorite,
                    'model_favorite_prob': fight.model_favorite_prob,
                    'market_favorite': fight.market_favorite,
                    'market_favorite_prob': fight.market_favorite_prob,
                    'predicted_method': fight.predicted_method,
                    'confidence_score': fight.confidence_score,
                    'is_upset_opportunity': fight.is_upset_opportunity,
                    'is_high_confidence': fight.is_high_confidence
                }
                for fight in predictions_analysis.fight_predictions
            ]
        }
        
        # Save analysis
        with open(export_path, 'w') as f:
            json.dump(analysis_export, f, indent=2, default=str)
        
        logger.info(f"Complete analysis exported: {export_path}")
        return str(export_path)
    
    def _determine_event_priority(self, event_name: str, target_fights: List[str]) -> RequestPriority:
        """
        Determine API request priority based on event characteristics
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of target fights
            
        Returns:
            RequestPriority: Priority level for API quota allocation
        """
        event_lower = event_name.lower()
        
        # CRITICAL: Title fights, PPV main events, numbered events
        if any(keyword in event_lower for keyword in ['title', 'championship', 'main event']):
            return RequestPriority.CRITICAL
        
        # Check for numbered UFC events (UFC 300, UFC 295, etc.)
        import re
        if re.search(r'ufc\s*\d{3}', event_lower):
            return RequestPriority.CRITICAL
        
        # HIGH: Fight Night main cards, ranked fighters
        if 'fight night' in event_lower or len(target_fights) >= 5:
            return RequestPriority.HIGH
        
        # MEDIUM: Regular events with multiple fights
        if len(target_fights) >= 3:
            return RequestPriority.MEDIUM
        
        # LOW: Individual fights, testing, health checks
        return RequestPriority.LOW
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """
        Format analysis result summary for display
        
        Args:
            analysis_result: Result from analyze_event()
            
        Returns:
            Formatted summary text
        """
        if analysis_result['status'] == 'failed':
            return f"❌ Analysis failed: {analysis_result['error']}"
        
        if analysis_result['status'] != 'completed':
            return f"⏳ Analysis in progress: {analysis_result['status']}"
        
        odds = analysis_result['odds_result']
        preds = analysis_result['predictions_analysis']
        bets = analysis_result['betting_recommendations']
        
        summary = [
            f"✅ ANALYSIS COMPLETE: {analysis_result['event_name']}",
            f"=" * 60,
            f"📊 Odds: {odds['total_fights']} fights processed",
            f"🧠 Predictions: {preds['successful_predictions']} successful, {preds['failed_predictions']} failed",
            f"🚨 Opportunities: {preds['upset_opportunities']} upsets, {preds['high_confidence_picks']} high confidence",
            f"💰 Betting: {bets['total_bets']} bets recommended",
            f"💵 Total Stake: ${bets['total_stake']:.2f}",
            f"📈 Expected Return: ${bets['expected_return']:.2f}",
            f"🎯 Portfolio EV: {bets['portfolio_ev']:+.1%}",
            f"💳 Bankroll Usage: {bets['bankroll_utilization']:.1%}",
            f"📁 Export: {analysis_result.get('export_path', 'N/A')}"
        ]
        
        return "\n".join(summary)
    
    async def run_event_monitoring(self, event_configs: List[EventConfig]) -> Dict[str, Any]:
        """
        Run monitoring for multiple events
        
        Args:
            event_configs: List of events to monitor
            
        Returns:
            Dict with monitoring results
        """
        logger.info(f"Starting monitoring for {len(event_configs)} events")
        
        monitoring_results = {
            'start_time': datetime.now().isoformat(),
            'events': {},
            'total_opportunities': 0,
            'total_recommended_stake': 0.0
        }
        
        for event_config in event_configs:
            if not event_config.monitoring_enabled:
                continue
            
            try:
                logger.info(f"Analyzing event: {event_config.name}")
                analysis_result = await self.analyze_event(
                    event_config.name, event_config.target_fights
                )
                
                monitoring_results['events'][event_config.name] = analysis_result
                
                if analysis_result['status'] == 'completed':
                    bets = analysis_result['betting_recommendations']
                    monitoring_results['total_opportunities'] += bets['total_bets']
                    monitoring_results['total_recommended_stake'] += bets['total_stake']
                
            except Exception as e:
                logger.error(f"Failed to analyze event {event_config.name}: {str(e)}")
                monitoring_results['events'][event_config.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        monitoring_results['end_time'] = datetime.now().isoformat()
        
        # Add Phase 2A quota usage summary
        if self.hybrid_odds_service:
            try:
                quota_status = await self.hybrid_odds_service.get_quota_status()
                monitoring_results['quota_summary'] = {
                    'api_requests_used': quota_status['quota_status']['requests_used_today'],
                    'budget_used': quota_status['quota_status']['budget_remaining'] < 50.0,
                    'efficiency': quota_status['hybrid_service_metrics'].get('api_efficiency', 0)
                }
            except Exception as e:
                logger.warning(f"Failed to get quota summary: {e}")
        
        logger.info(
            f"Monitoring completed: {monitoring_results['total_opportunities']} opportunities, "
            f"${monitoring_results['total_recommended_stake']:.2f} total stake"
        )
        
        return monitoring_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status
        
        Returns:
            Dict with system status information
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'betting_system_initialized': self.betting_system is not None,
            'bankroll': self.config.betting.initial_bankroll,
            'bankroll_tier': None,
            'api_key_configured': bool(self.config.api.odds_api_key),
            'auto_execute_enabled': self.config.agent.auto_execute_bets,
            'monitoring_enabled': self.config.agent.enable_live_monitoring,
            'configured_events': len(self.config.events)
        }
        
        if self.betting_system:
            # Determine bankroll tier
            bankroll = self.config.betting.initial_bankroll
            if bankroll < 200:
                status['bankroll_tier'] = 'MICRO'
            elif bankroll < 1000:
                status['bankroll_tier'] = 'SMALL'
            else:
                status['bankroll_tier'] = 'STANDARD'
        
        # Add Phase 2A quota status if available
        if self.hybrid_odds_service:
            try:
                # Note: Synchronous call for now - can be made async later if needed
                quota_status = self.hybrid_odds_service.get_quota_status_sync()
                status['quota_management'] = {
                    'requests_used_today': quota_status.get('quota_status', {}).get('requests_used_today', 0),
                    'requests_remaining_today': quota_status.get('quota_status', {}).get('requests_remaining_today', 0),
                    'budget_remaining': quota_status.get('quota_status', {}).get('budget_remaining', 0),
                    'api_efficiency': quota_status.get('hybrid_service_metrics', {}).get('api_efficiency', 0)
                }
            except Exception as e:
                logger.warning(f"Failed to get quota status: {e}")
                status['quota_management'] = {'error': str(e)}
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check
        
        Returns:
            Dict with health check results
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # Check betting system
            health['checks']['betting_system'] = {
                'status': 'healthy' if self.betting_system else 'unhealthy',
                'details': 'Betting system initialized' if self.betting_system else 'Not initialized'
            }
            
            # Check Phase 2A hybrid service or fallback API connectivity
            if self.hybrid_odds_service:
                try:
                    hybrid_health = await self.hybrid_odds_service.health_check()
                    health['checks']['hybrid_odds_service'] = {
                        'status': hybrid_health['overall_status'],
                        'details': f"Components: {len(hybrid_health['components'])} checked"
                    }
                    
                    # Add quota status
                    quota_status = await self.hybrid_odds_service.get_quota_status()
                    health['checks']['quota_management'] = {
                        'status': 'healthy',
                        'details': (
                            f"API quota: {quota_status['quota_status']['requests_remaining_today']} remaining today, "
                            f"Budget: ${quota_status['quota_status']['budget_remaining']:.2f}"
                        )
                    }
                    
                except Exception as e:
                    health['checks']['hybrid_odds_service'] = {
                        'status': 'unhealthy',
                        'details': f"Hybrid service check failed: {str(e)}"
                    }
            else:
                # Fallback to original API check
                try:
                    api_status = self.odds_service.check_api_usage()
                    health['checks']['api_connectivity'] = {
                        'status': 'healthy' if api_status['connection'] else 'unhealthy',
                        'details': f"Available events: {api_status.get('available_events', 0)}"
                    }
                except Exception as e:
                    health['checks']['api_connectivity'] = {
                        'status': 'unhealthy',
                        'details': f"API check failed: {str(e)}"
                    }
            
            # Check storage paths
            storage_healthy = True
            for path_name in ['odds_storage_path', 'backup_storage_path', 'analysis_export_path']:
                path = Path(getattr(self.config.agent, path_name))
                if not path.exists():
                    storage_healthy = False
                    break
            
            health['checks']['storage'] = {
                'status': 'healthy' if storage_healthy else 'unhealthy',
                'details': 'All storage paths accessible' if storage_healthy else 'Storage path issues'
            }
            
            # Overall status
            unhealthy_checks = [
                check for check in health['checks'].values() 
                if check['status'] == 'unhealthy'
            ]
            
            if unhealthy_checks:
                health['overall_status'] = 'unhealthy'
            
        except Exception as e:
            health['overall_status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health


async def main():
    """Main entry point for the UFC betting agent"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        from .config.agent_config import load_configuration
        config = load_configuration()
        
        print(config.get_summary())
        
        # Initialize agent
        agent = UFCBettingAgent(config)
        
        # Initialize betting system
        if not agent.initialize_betting_system():
            print("❌ Failed to initialize betting system")
            return
        
        # Perform health check
        health = await agent.health_check()
        print(f"\n🏥 Health Check: {health['overall_status']}")
        
        # Run analysis for configured events
        if config.events:
            print(f"\n🔍 Running analysis for {len(config.events)} configured events...")
            monitoring_results = await agent.run_event_monitoring(config.events)
            
            print(f"\n📊 MONITORING RESULTS:")
            print(f"Events Analyzed: {len(monitoring_results['events'])}")
            print(f"Total Opportunities: {monitoring_results['total_opportunities']}")
            print(f"Total Recommended Stake: ${monitoring_results['total_recommended_stake']:.2f}")
        else:
            print("\n⚠️  No events configured for analysis")
            
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        print(f"❌ Agent failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())