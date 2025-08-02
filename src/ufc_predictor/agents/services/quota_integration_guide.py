"""
Quota Management Integration Guide

This module provides examples and utilities for integrating the quota management system
with existing services. Shows how to upgrade existing code to use the enhanced odds service
with minimal changes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .enhanced_odds_service import EnhancedUFCOddsService, HybridOddsResult
from .quota_manager import QuotaManager, RequestPriority
from .odds_service import UFCOddsService  # Legacy service
from .prediction_service import UFCPredictionService
from .betting_service import UFCBettingService

logger = logging.getLogger(__name__)


class QuotaAwareAgentService:
    """
    Example of how to integrate quota management into existing agent workflows
    
    This shows how to upgrade existing PredictionService and BettingService
    integrations to use the new quota-aware odds fetching.
    """
    
    def __init__(self, api_key: str, betting_system: Dict[str, Any]):
        """
        Initialize quota-aware agent service
        
        Args:
            api_key: The Odds API key
            betting_system: Betting system components for predictions
        """
        # Initialize enhanced services
        self.enhanced_odds_service = EnhancedUFCOddsService(api_key)
        self.prediction_service = UFCPredictionService(betting_system)
        self.betting_service = UFCBettingService()
        
        # Legacy fallback (for gradual migration)
        self.legacy_odds_service = None
        
        logger.info("Quota-aware agent service initialized")
    
    async def analyze_event_with_quota_management(self, event_name: str, 
                                                target_fights: List[str],
                                                bankroll: float,
                                                priority: RequestPriority = RequestPriority.HIGH) -> Dict[str, Any]:
        """
        Complete event analysis with intelligent quota management
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to analyze
            bankroll: Available bankroll for betting
            priority: Request priority for API quota allocation
            
        Returns:
            Complete analysis with predictions, betting recommendations, and quota info
        """
        logger.info(f"Starting quota-aware analysis for {event_name}")
        
        analysis_result = {
            'event_name': event_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'quota_info': {},
            'odds_result': None,
            'predictions': None,
            'betting_recommendations': None,
            'cost_analysis': {}
        }
        
        try:
            # Step 1: Check quota status before starting
            quota_status = await self.enhanced_odds_service.get_quota_status()
            analysis_result['quota_info']['pre_analysis'] = quota_status
            
            # Step 2: Fetch odds with quota management
            logger.info(f"Fetching odds with priority: {priority.name}")
            odds_result = await self.enhanced_odds_service.fetch_event_odds(
                event_name=event_name,
                target_fights=target_fights,
                priority=priority,
                enable_fallback=True
            )
            
            analysis_result['odds_result'] = {
                'status': odds_result.status,
                'data_sources': odds_result.data_sources,
                'confidence_score': odds_result.confidence_score,
                'cost_savings_usd': odds_result.cost_savings_usd,
                'api_quota_used': odds_result.api_quota_used,
                'fallback_reason': odds_result.fallback_reason,
                'fight_count': len(odds_result.odds_data) if odds_result.odds_data else 0
            }
            
            if odds_result.status in ['api_success', 'fallback_success', 'hybrid_success']:
                # Step 3: Generate predictions
                logger.info("Generating predictions with fetched odds")
                predictions_analysis = self.prediction_service.predict_event(
                    odds_result.odds_data, event_name
                )
                
                analysis_result['predictions'] = {
                    'total_fights': predictions_analysis.summary['total_fights'],
                    'successful_predictions': predictions_analysis.summary['successful_predictions'],
                    'upset_opportunities': predictions_analysis.summary['upset_opportunities'],
                    'high_confidence_picks': predictions_analysis.summary['high_confidence_picks'],
                    'method_breakdown': predictions_analysis.summary['method_breakdown']
                }
                
                # Step 4: Generate betting recommendations
                logger.info(f"Generating betting recommendations for ${bankroll:.2f} bankroll")
                betting_recommendations = self.betting_service.generate_betting_recommendations(
                    predictions_analysis, bankroll
                )
                
                analysis_result['betting_recommendations'] = {
                    'total_recommended_stake': betting_recommendations.portfolio_summary.total_recommended_stake,
                    'number_of_bets': betting_recommendations.portfolio_summary.number_of_bets,
                    'expected_return': betting_recommendations.portfolio_summary.expected_return,
                    'portfolio_ev': betting_recommendations.portfolio_summary.portfolio_ev,
                    'bankroll_utilization': betting_recommendations.portfolio_summary.bankroll_utilization,
                    'bankroll_tier': betting_recommendations.bankroll_info['tier']
                }
                
                analysis_result['status'] = 'success'
                
            else:
                analysis_result['status'] = 'failed'
                analysis_result['error'] = f"Odds fetching failed: {odds_result.fallback_reason}"
            
            # Step 5: Post-analysis quota status
            post_quota_status = await self.enhanced_odds_service.get_quota_status()
            analysis_result['quota_info']['post_analysis'] = post_quota_status
            
            # Step 6: Cost analysis
            analysis_result['cost_analysis'] = {
                'api_quota_used': odds_result.api_quota_used,
                'estimated_cost_usd': 0.01 if odds_result.api_quota_used else 0.0,
                'cost_savings_usd': odds_result.cost_savings_usd,
                'total_quota_used_today': post_quota_status['usage_summary']['daily_used'],
                'quota_remaining_today': post_quota_status['usage_summary']['daily_remaining']
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {event_name}: {str(e)}")
            analysis_result['status'] = 'error'
            analysis_result['error'] = str(e)
        
        logger.info(f"Analysis completed with status: {analysis_result['status']}")
        return analysis_result
    
    async def batch_analyze_multiple_events(self, events: List[Dict[str, Any]], 
                                          bankroll: float) -> Dict[str, Any]:
        """
        Analyze multiple events with intelligent priority management
        
        Args:
            events: List of event dictionaries with 'name', 'fights', and 'priority'
            bankroll: Available bankroll
            
        Returns:
            Batch analysis results with quota optimization
        """
        logger.info(f"Starting batch analysis for {len(events)} events")
        
        batch_result = {
            'total_events': len(events),
            'completed_events': 0,
            'failed_events': 0,
            'quota_exhausted': False,
            'total_cost_savings': 0.0,
            'event_results': {},
            'quota_summary': {}
        }
        
        # Sort events by priority
        sorted_events = sorted(events, key=lambda x: x.get('priority', RequestPriority.MEDIUM).value)
        
        for event in sorted_events:
            event_name = event['name']
            target_fights = event['fights']
            priority = event.get('priority', RequestPriority.MEDIUM)
            
            try:
                # Check if we should continue based on quota
                quota_forecast = await self.enhanced_odds_service.get_quota_forecast(hours_ahead=1)
                
                if quota_forecast['quota_exhaustion_risk'] == 'high' and priority.value > 2:
                    logger.warning(f"Skipping {event_name} due to high quota risk and low priority")
                    batch_result['event_results'][event_name] = {
                        'status': 'skipped',
                        'reason': 'quota_risk_management'
                    }
                    continue
                
                # Analyze event
                event_result = await self.analyze_event_with_quota_management(
                    event_name, target_fights, bankroll, priority
                )
                
                batch_result['event_results'][event_name] = event_result
                
                if event_result['status'] == 'success':
                    batch_result['completed_events'] += 1
                    batch_result['total_cost_savings'] += event_result['cost_analysis']['cost_savings_usd']
                else:
                    batch_result['failed_events'] += 1
                
                # Check for quota exhaustion
                if event_result.get('quota_info', {}).get('post_analysis', {}).get('quota_health', {}).get('quota_exhausted'):
                    batch_result['quota_exhausted'] = True
                    logger.warning("Quota exhausted during batch processing")
                    break
                
                # Brief pause between requests to be respectful
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {event_name}: {str(e)}")
                batch_result['event_results'][event_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                batch_result['failed_events'] += 1
        
        # Final quota summary
        final_quota_status = await self.enhanced_odds_service.get_quota_status()
        batch_result['quota_summary'] = final_quota_status
        
        logger.info(f"Batch analysis completed: {batch_result['completed_events']}/{len(events)} successful")
        return batch_result
    
    async def adaptive_priority_analysis(self, event_name: str, target_fights: List[str], 
                                       bankroll: float, event_time: datetime) -> Dict[str, Any]:
        """
        Adaptive analysis that adjusts priority based on event timing
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to analyze
            bankroll: Available bankroll
            event_time: When the event is scheduled
            
        Returns:
            Analysis result with adaptive priority management
        """
        # Calculate time until event
        time_until_event = (event_time - datetime.now()).total_seconds() / 3600  # hours
        
        # Adaptive priority based on event timing
        if time_until_event <= 2:  # Event starting within 2 hours
            priority = RequestPriority.CRITICAL
        elif time_until_event <= 24:  # Event within 24 hours
            priority = RequestPriority.HIGH
        elif time_until_event <= 168:  # Event within 1 week
            priority = RequestPriority.MEDIUM
        else:  # Event more than 1 week away
            priority = RequestPriority.LOW
        
        logger.info(f"Adaptive priority for {event_name}: {priority.name} "
                   f"(event in {time_until_event:.1f} hours)")
        
        # Use the adaptive priority for analysis
        return await self.analyze_event_with_quota_management(
            event_name, target_fights, bankroll, priority
        )
    
    async def quota_health_monitor(self) -> Dict[str, Any]:
        """
        Monitor quota health and provide recommendations
        
        Returns:
            Quota health report with recommendations
        """
        health_status = await self.enhanced_odds_service.health_check()
        quota_status = await self.enhanced_odds_service.get_quota_status()
        quota_forecast = await self.enhanced_odds_service.get_quota_forecast(hours_ahead=24)
        
        health_report = {
            'overall_health': health_status['overall_status'],
            'quota_health': quota_status['quota_health'],
            'current_usage': quota_status['usage_summary'],
            'forecast': quota_forecast,
            'recommendations': [],
            'alerts': []
        }
        
        # Generate recommendations based on status
        if quota_forecast['quota_exhaustion_risk'] == 'high':
            health_report['recommendations'].extend([
                "Consider activating fallback mode proactively",
                "Defer non-critical analysis requests",
                "Monitor usage closely for next 4-6 hours"
            ])
            health_report['alerts'].append("HIGH: Quota exhaustion risk detected")
        
        if quota_status['usage_summary']['daily_remaining'] < 50:
            health_report['recommendations'].append("Switch to fallback mode for non-critical requests")
            health_report['alerts'].append("WARNING: Low quota remaining for today")
        
        if quota_status['fallback_stats']['activations'] > 5:
            health_report['recommendations'].append("Consider upgrading API quota limits")
            health_report['alerts'].append("INFO: Frequent fallback usage detected")
        
        return health_report
    
    async def legacy_migration_helper(self, event_name: str, 
                                    target_fights: List[str]) -> Dict[str, Any]:
        """
        Helper function to gradually migrate from legacy odds service
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to analyze
            
        Returns:
            Comparison between legacy and enhanced services
        """
        migration_result = {
            'event_name': event_name,
            'enhanced_result': None,
            'legacy_result': None,
            'comparison': {},
            'migration_recommendation': ''
        }
        
        try:
            # Try enhanced service first
            enhanced_result = await self.enhanced_odds_service.fetch_event_odds(
                event_name, target_fights, RequestPriority.MEDIUM
            )
            migration_result['enhanced_result'] = {
                'status': enhanced_result.status,
                'data_sources': enhanced_result.data_sources,
                'confidence': enhanced_result.confidence_score,
                'cost_savings': enhanced_result.cost_savings_usd
            }
            
            # Try legacy service for comparison (if available)
            if self.legacy_odds_service:
                legacy_result = self.legacy_odds_service.fetch_and_store_odds(
                    event_name, target_fights
                )
                migration_result['legacy_result'] = {
                    'status': legacy_result.status,
                    'fight_count': legacy_result.total_fights
                }
            
            # Generate comparison and recommendation
            if enhanced_result.status in ['api_success', 'fallback_success']:
                migration_result['migration_recommendation'] = (
                    "Enhanced service successfully provided data. "
                    "Safe to migrate to enhanced service."
                )
            else:
                migration_result['migration_recommendation'] = (
                    "Enhanced service encountered issues. "
                    "Continue using legacy service until resolved."
                )
        
        except Exception as e:
            migration_result['error'] = str(e)
            migration_result['migration_recommendation'] = (
                "Error during migration test. "
                "Investigate before full migration."
            )
        
        return migration_result


# Example usage functions
async def example_basic_usage():
    """Example of basic quota-aware odds fetching"""
    
    # Initialize with your API key
    api_key = "your_api_key_here"
    
    # Mock betting system for predictions
    betting_system = {
        'fighters_df': None,  # Would be actual DataFrame
        'winner_cols': [],
        'method_cols': [],
        'winner_model': None,
        'method_model': None,
        'predict_function': lambda *args: {'error': 'Mock function'}
    }
    
    # Initialize service
    agent_service = QuotaAwareAgentService(api_key, betting_system)
    
    # Basic usage
    result = await agent_service.analyze_event_with_quota_management(
        event_name="UFC 307",
        target_fights=["Alex Pereira vs Khalil Rountree Jr"],
        bankroll=1000.0,
        priority=RequestPriority.HIGH
    )
    
    print(f"Analysis status: {result['status']}")
    print(f"Data sources used: {result['odds_result']['data_sources']}")
    print(f"Cost savings: ${result['cost_analysis']['cost_savings_usd']:.2f}")


async def example_batch_analysis():
    """Example of batch event analysis with quota management"""
    
    api_key = "your_api_key_here"
    betting_system = {}  # Mock system
    
    agent_service = QuotaAwareAgentService(api_key, betting_system)
    
    # Define multiple events with different priorities
    events = [
        {
            'name': 'UFC 307',
            'fights': ["Alex Pereira vs Khalil Rountree Jr"],
            'priority': RequestPriority.CRITICAL
        },
        {
            'name': 'UFC 308',
            'fights': ["Ilia Topuria vs Max Holloway"],
            'priority': RequestPriority.HIGH
        },
        {
            'name': 'UFC 309',
            'fights': ["Jon Jones vs Stipe Miocic"],
            'priority': RequestPriority.MEDIUM
        }
    ]
    
    # Batch analysis
    batch_result = await agent_service.batch_analyze_multiple_events(
        events=events,
        bankroll=2000.0
    )
    
    print(f"Completed: {batch_result['completed_events']}/{batch_result['total_events']}")
    print(f"Total cost savings: ${batch_result['total_cost_savings']:.2f}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_basic_usage())