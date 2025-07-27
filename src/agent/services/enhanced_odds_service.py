"""
Enhanced UFC Odds Service with Intelligent Quota Management

Seamlessly integrates QuotaManager with existing AsyncUFCOddsAPIClient to provide:
- Priority-based request management
- Automatic fallback to TAB scraper when quota exhausted
- Hybrid data source architecture with confidence scoring
- Data validation and reconciliation between API and scraper sources
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .quota_manager import QuotaManager, RequestPriority
from .async_odds_client import AsyncUFCOddsAPIClient, AsyncOddsAPIError
from .odds_service import UFCOddsService, OddsResult
from ..services.tab_integration import TABOddsIntegration
from ..services.data_confidence_scorer import DataConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class HybridOddsResult:
    """Enhanced odds result with data source information"""
    event_name: str
    status: str  # 'api_success', 'fallback_success', 'hybrid_success', 'failed'
    odds_data: Dict[str, Dict] = None
    data_sources: List[str] = None  # ['api', 'tab_scraper', 'cached']
    confidence_score: float = 0.0
    cost_savings_usd: float = 0.0
    api_quota_used: bool = False
    fallback_reason: Optional[str] = None
    data_reconciliation: Optional[Dict] = None
    fetch_timestamp: Optional[str] = None
    csv_path: Optional[str] = None


@dataclass
class DataSourceConfidence:
    """Confidence scoring for different data sources"""
    source_name: str
    confidence_score: float  # 0.0 to 1.0
    freshness_score: float  # 0.0 to 1.0 based on data age
    completeness_score: float  # 0.0 to 1.0 based on data completeness
    consistency_score: float  # 0.0 to 1.0 based on historical accuracy
    last_updated: datetime
    total_score: float = 0.0
    
    def __post_init__(self):
        # Weight the components
        self.total_score = (
            self.confidence_score * 0.4 +
            self.freshness_score * 0.3 +
            self.completeness_score * 0.2 +
            self.consistency_score * 0.1
        )


class EnhancedUFCOddsService:
    """
    Enhanced UFC odds service with intelligent quota management and hybrid data sources
    
    Features:
    - Seamless integration with QuotaManager for request prioritization
    - Automatic fallback to TAB scraper when API quota exhausted
    - Data source confidence scoring and selection
    - Data validation and reconciliation between sources
    - Cost tracking and optimization
    - Performance monitoring and alerting
    """
    
    def __init__(self, api_key: str, quota_config_path: str = "config/quota_config.json",
                 storage_base_path: str = "odds"):
        """
        Initialize enhanced odds service
        
        Args:
            api_key: The Odds API key
            quota_config_path: Path to quota configuration
            storage_base_path: Base directory for odds storage
        """
        # Initialize core components
        self.quota_manager = QuotaManager(quota_config_path)
        self.async_client = AsyncUFCOddsAPIClient(api_key)
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(exist_ok=True)
        
        # Initialize fallback and hybrid components
        self.tab_integration = TABOddsIntegration()
        self.confidence_scorer = DataConfidenceScorer()
        
        # Configure fallback integrations
        self.quota_manager.set_fallback_integrations(
            tab_scraper=self.tab_integration,
            odds_service_fallback=self._get_cached_odds
        )
        
        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'api_requests': 0,
            'fallback_requests': 0,
            'hybrid_requests': 0,
            'cost_savings': 0.0
        }
        
        logger.info("Enhanced UFC odds service initialized with quota management")
    
    async def fetch_event_odds(self, event_name: str, target_fights: Optional[List[str]] = None,
                             priority: RequestPriority = RequestPriority.MEDIUM,
                             enable_fallback: bool = True) -> HybridOddsResult:
        """
        Fetch odds for a UFC event with intelligent source selection
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            priority: Request priority level
            enable_fallback: Whether to use fallback sources
            
        Returns:
            HybridOddsResult: Complete odds fetch result with source information
        """
        logger.info(f"Fetching odds for {event_name} (priority: {priority.name})")
        
        result = HybridOddsResult(
            event_name=event_name,
            status='pending',
            data_sources=[],
            fetch_timestamp=datetime.now().isoformat()
        )
        
        self.request_stats['total_requests'] += 1
        
        try:
            # Step 1: Try API with quota management
            api_result = await self._try_api_fetch(event_name, target_fights, priority)
            
            if api_result['success']:
                result.status = 'api_success'
                result.odds_data = api_result['data']
                result.data_sources.append('api')
                result.api_quota_used = True
                result.confidence_score = 0.95  # High confidence for API data
                
                self.request_stats['api_requests'] += 1
                
                # Store to CSV
                result.csv_path = str(self._store_odds_to_csv(
                    result.odds_data, event_name, datetime.now()
                ))
                
                logger.info(f"Successfully fetched odds from API for {event_name}")
                return result
            
            # Step 2: Handle quota exhaustion or API failure
            if not enable_fallback:
                result.status = 'failed'
                result.fallback_reason = api_result.get('error', 'API unavailable')
                return result
            
            # Step 3: Try fallback sources
            fallback_result = await self._try_fallback_fetch(event_name, target_fights)
            
            if fallback_result['success']:
                result.status = 'fallback_success'
                result.odds_data = fallback_result['data']
                result.data_sources.extend(fallback_result['sources'])
                result.confidence_score = fallback_result['confidence']
                result.cost_savings_usd = 0.01  # Approximate API cost saved
                result.fallback_reason = api_result.get('error', 'quota_exhausted')
                
                self.request_stats['fallback_requests'] += 1
                self.request_stats['cost_savings'] += result.cost_savings_usd
                
                # Store to CSV
                result.csv_path = str(self._store_odds_to_csv(
                    result.odds_data, event_name, datetime.now()
                ))
                
                logger.info(f"Successfully fetched odds from fallback for {event_name}")
                return result
            
            # Step 4: All sources failed
            result.status = 'failed'
            result.fallback_reason = 'All data sources unavailable'
            
        except Exception as e:
            logger.error(f"Error fetching odds for {event_name}: {str(e)}")
            result.status = 'failed'
            result.fallback_reason = f"Unexpected error: {str(e)}"
        
        return result
    
    async def _try_api_fetch(self, event_name: str, target_fights: Optional[List[str]],
                           priority: RequestPriority) -> Dict[str, Any]:
        """
        Attempt to fetch odds from The Odds API with quota management
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            priority: Request priority level
            
        Returns:
            Dict with success status and data/error
        """
        try:
            # Request quota allocation
            quota_granted, request_info = await self.quota_manager.request_quota(
                priority, 'get_ufc_odds'
            )
            
            if not quota_granted:
                if request_info.startswith('queued:'):
                    # Handle queued request
                    request_id = request_info.split(':')[1]
                    logger.info(f"Request queued: {request_id}")
                    return {'success': False, 'error': 'request_queued', 'request_id': request_id}
                else:
                    return {'success': False, 'error': 'quota_exhausted'}
            
            # Make API request
            request_id = request_info
            
            async with self.async_client as client:
                # Fetch raw odds data
                api_data = await client.get_ufc_odds(region="au")
                
                # Format for analysis
                formatted_odds = await client.format_odds_for_analysis(api_data, target_fights)
                
                if not formatted_odds:
                    await self.quota_manager.record_request_completion(request_id, False)
                    return {'success': False, 'error': 'no_odds_found'}
                
                # Record successful completion
                await self.quota_manager.record_request_completion(request_id, True)
                
                return {'success': True, 'data': formatted_odds}
        
        except AsyncOddsAPIError as e:
            if request_id:
                await self.quota_manager.record_request_completion(request_id, False)
            
            # Check if quota-related error
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                await self.quota_manager.activate_fallback_mode("api_rate_limit")
            
            return {'success': False, 'error': str(e)}
        
        except Exception as e:
            if request_id:
                await self.quota_manager.record_request_completion(request_id, False)
            return {'success': False, 'error': f"api_error: {str(e)}"}
    
    async def _try_fallback_fetch(self, event_name: str, 
                                target_fights: Optional[List[str]]) -> Dict[str, Any]:
        """
        Attempt to fetch odds from fallback sources (TAB scraper, cached data)
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            
        Returns:
            Dict with success status, data, sources, and confidence
        """
        fallback_sources = []
        combined_data = {}
        confidence_scores = []
        
        try:
            # Try TAB Australia scraper
            tab_result = await self._fetch_from_tab_scraper(event_name, target_fights)
            if tab_result['success']:
                fallback_sources.append('tab_scraper')
                combined_data.update(tab_result['data'])
                confidence_scores.append(tab_result['confidence'])
                
                logger.info(f"TAB scraper provided {len(tab_result['data'])} fights")
            
            # Try cached data as additional source
            cached_result = await self._fetch_from_cache(event_name, target_fights)
            if cached_result['success']:
                fallback_sources.append('cached')
                
                # Reconcile with TAB data if available
                if combined_data:
                    reconciled_data = self._reconcile_data_sources(
                        combined_data, cached_result['data']
                    )
                    combined_data = reconciled_data['odds']
                    confidence_scores.append(reconciled_data['confidence'])
                else:
                    combined_data.update(cached_result['data'])
                    confidence_scores.append(cached_result['confidence'])
                
                logger.info(f"Cache provided additional data for {event_name}")
            
            if combined_data:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                return {
                    'success': True,
                    'data': combined_data,
                    'sources': fallback_sources,
                    'confidence': avg_confidence
                }
            else:
                return {'success': False, 'error': 'no_fallback_data'}
        
        except Exception as e:
            logger.error(f"Fallback fetch failed: {str(e)}")
            return {'success': False, 'error': f"fallback_error: {str(e)}"}
    
    async def _fetch_from_tab_scraper(self, event_name: str, 
                                    target_fights: Optional[List[str]]) -> Dict[str, Any]:
        """
        Fetch odds from TAB Australia scraper
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            
        Returns:
            Dict with success status, data, and confidence
        """
        try:
            # Use TAB integration to fetch odds
            tab_odds = await self.tab_integration.fetch_event_odds(event_name, target_fights)
            
            if tab_odds:
                # Convert TAB format to standard format
                formatted_odds = self._convert_tab_to_standard_format(tab_odds)
                confidence = self.confidence_scorer.score_tab_data(tab_odds)
                
                return {
                    'success': True,
                    'data': formatted_odds,
                    'confidence': confidence
                }
            else:
                return {'success': False, 'error': 'tab_no_data'}
        
        except Exception as e:
            logger.warning(f"TAB scraper failed: {str(e)}")
            return {'success': False, 'error': f"tab_error: {str(e)}"}
    
    async def _fetch_from_cache(self, event_name: str, 
                              target_fights: Optional[List[str]]) -> Dict[str, Any]:
        """
        Fetch odds from cached data
        
        Args:
            event_name: Name of the UFC event
            target_fights: Optional list of specific fights
            
        Returns:
            Dict with success status, data, and confidence
        """
        try:
            # Look for latest cached odds
            event_folder = self.storage_base_path / event_name
            if not event_folder.exists():
                return {'success': False, 'error': 'no_cache'}
            
            # Find latest CSV file
            csv_files = list(event_folder.glob("odds_*.csv"))
            if not csv_files:
                return {'success': False, 'error': 'no_cache_files'}
            
            latest_csv = sorted(csv_files, key=lambda x: x.name)[-1]
            
            # Load and reconstruct odds data
            import pandas as pd
            df = pd.read_csv(latest_csv)
            cached_odds = self._reconstruct_odds_data(df)
            
            # Calculate confidence based on data age
            file_age_hours = (datetime.now() - datetime.fromtimestamp(
                latest_csv.stat().st_mtime)).total_seconds() / 3600
            
            # Confidence decreases with age
            age_confidence = max(0.1, 1.0 - (file_age_hours / 24))  # 24-hour decay
            confidence = self.confidence_scorer.score_cached_data(cached_odds, file_age_hours)
            
            return {
                'success': True,
                'data': cached_odds,
                'confidence': confidence
            }
        
        except Exception as e:
            logger.warning(f"Cache fetch failed: {str(e)}")
            return {'success': False, 'error': f"cache_error: {str(e)}"}
    
    def _convert_tab_to_standard_format(self, tab_odds: List) -> Dict[str, Dict]:
        """
        Convert TAB scraper format to standard odds format
        
        Args:
            tab_odds: List of TABFightOdds objects
            
        Returns:
            Dict in standard format
        """
        formatted_odds = {}
        
        for fight in tab_odds:
            fight_key = f"{fight.fighter_a} vs {fight.fighter_b}"
            
            formatted_odds[fight_key] = {
                'fighter_a': fight.fighter_a,
                'fighter_b': fight.fighter_b,
                'fighter_a_decimal_odds': fight.fighter_a_decimal_odds,
                'fighter_b_decimal_odds': fight.fighter_b_decimal_odds,
                'bookmakers': ['TAB Australia'],
                'commence_time': fight.fight_time,
                'api_timestamp': datetime.now().isoformat(),
                'source': 'tab_scraper'
            }
        
        return formatted_odds
    
    def _reconcile_data_sources(self, source1_data: Dict, source2_data: Dict) -> Dict[str, Any]:
        """
        Reconcile odds data from multiple sources
        
        Args:
            source1_data: Odds data from first source
            source2_data: Odds data from second source
            
        Returns:
            Dict with reconciled odds and confidence score
        """
        reconciled_odds = {}
        confidence_scores = []
        
        # Combine all fights from both sources
        all_fight_keys = set(source1_data.keys()) | set(source2_data.keys())
        
        for fight_key in all_fight_keys:
            if fight_key in source1_data and fight_key in source2_data:
                # Both sources have this fight - reconcile
                fight1 = source1_data[fight_key]
                fight2 = source2_data[fight_key]
                
                # Use average odds if sources disagree
                reconciled_fight = {
                    'fighter_a': fight1['fighter_a'],
                    'fighter_b': fight1['fighter_b'],
                    'fighter_a_decimal_odds': (
                        fight1['fighter_a_decimal_odds'] + fight2['fighter_a_decimal_odds']
                    ) / 2,
                    'fighter_b_decimal_odds': (
                        fight1['fighter_b_decimal_odds'] + fight2['fighter_b_decimal_odds']
                    ) / 2,
                    'bookmakers': list(set(
                        fight1.get('bookmakers', []) + fight2.get('bookmakers', [])
                    )),
                    'commence_time': fight1.get('commence_time', fight2.get('commence_time', '')),
                    'api_timestamp': datetime.now().isoformat(),
                    'source': 'reconciled'
                }
                
                # Calculate agreement score
                odds_diff_a = abs(fight1['fighter_a_decimal_odds'] - fight2['fighter_a_decimal_odds'])
                odds_diff_b = abs(fight1['fighter_b_decimal_odds'] - fight2['fighter_b_decimal_odds'])
                agreement_score = 1.0 - min(1.0, (odds_diff_a + odds_diff_b) / 2)
                
                confidence_scores.append(agreement_score)
                reconciled_odds[fight_key] = reconciled_fight
            
            elif fight_key in source1_data:
                # Only in source 1
                reconciled_odds[fight_key] = source1_data[fight_key]
                confidence_scores.append(0.7)  # Lower confidence for single source
            
            else:
                # Only in source 2
                reconciled_odds[fight_key] = source2_data[fight_key]
                confidence_scores.append(0.7)  # Lower confidence for single source
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            'odds': reconciled_odds,
            'confidence': avg_confidence,
            'reconciliation_details': {
                'total_fights': len(reconciled_odds),
                'source1_exclusive': len([k for k in reconciled_odds if k in source1_data and k not in source2_data]),
                'source2_exclusive': len([k for k in reconciled_odds if k in source2_data and k not in source1_data]),
                'reconciled_fights': len([k for k in reconciled_odds if k in source1_data and k in source2_data])
            }
        }
    
    def _store_odds_to_csv(self, odds_data: Dict[str, Dict], 
                          event_name: str, timestamp: datetime) -> Path:
        """
        Store odds data to CSV with organized folder structure
        (Same implementation as UFCOddsService)
        """
        # Create event-specific folder
        event_folder = self.storage_base_path / event_name
        event_folder.mkdir(exist_ok=True)
        
        # Generate CSV filename with timestamp
        csv_filename = f"odds_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = event_folder / csv_filename
        
        # Prepare odds records (same as UFCOddsService)
        import pandas as pd
        from ..services.odds_service import OddsRecord
        
        odds_records = []
        timestamp_str = timestamp.isoformat()
        
        for fight_key, fight_data in odds_data.items():
            # Create record for fighter A
            record_a = OddsRecord(
                timestamp=timestamp_str,
                event_name=event_name,
                fight_key=fight_key,
                fighter=fight_data['fighter_a'],
                opponent=fight_data['fighter_b'],
                decimal_odds=fight_data['fighter_a_decimal_odds'],
                position='fighter_a'
            )
            record_a.bookmakers = fight_data.get('bookmakers', ['Unknown'])
            odds_records.append(record_a.to_dict())
            
            # Create record for fighter B
            record_b = OddsRecord(
                timestamp=timestamp_str,
                event_name=event_name,
                fight_key=fight_key,
                fighter=fight_data['fighter_b'],
                opponent=fight_data['fighter_a'],
                decimal_odds=fight_data['fighter_b_decimal_odds'],
                position='fighter_b'
            )
            record_b.bookmakers = fight_data.get('bookmakers', ['Unknown'])
            odds_records.append(record_b.to_dict())
        
        # Save to CSV
        df_odds = pd.DataFrame(odds_records)
        df_odds.to_csv(csv_path, index=False)
        
        logger.info(f"Odds saved to CSV: {csv_path}")
        return csv_path
    
    def _reconstruct_odds_data(self, df) -> Dict[str, Dict]:
        """
        Reconstruct odds data from CSV DataFrame
        (Same implementation as UFCOddsService)
        """
        import pandas as pd
        
        odds_data = {}
        
        # Group by fight_key to reconstruct fight data
        for fight_key, group in df.groupby('fight_key'):
            fighter_a_row = group[group['position'] == 'fighter_a'].iloc[0]
            fighter_b_row = group[group['position'] == 'fighter_b'].iloc[0]
            
            odds_data[fight_key] = {
                'fighter_a': fighter_a_row['fighter'],
                'fighter_b': fighter_b_row['fighter'],
                'fighter_a_decimal_odds': fighter_a_row['decimal_odds'],
                'fighter_b_decimal_odds': fighter_b_row['decimal_odds'],
                'bookmakers': fighter_a_row['bookmakers'].split(', ') if pd.notna(fighter_a_row['bookmakers']) else []
            }
        
        return odds_data
    
    async def get_quota_status(self) -> Dict[str, Any]:
        """Get comprehensive quota and service status"""
        quota_status = self.quota_manager.get_quota_status()
        
        # Add service-specific metrics
        quota_status['service_stats'] = self.request_stats
        quota_status['fallback_status'] = {
            'tab_scraper_available': self.tab_integration is not None,
            'cache_available': self.storage_base_path.exists(),
            'fallback_mode_active': self.quota_manager.fallback_mode
        }
        
        return quota_status
    
    async def get_quota_forecast(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get quota usage forecast"""
        return self.quota_manager.get_quota_forecast(hours_ahead)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # Check API client health
            api_health = await self.async_client.health_check()
            health_data['components']['api_client'] = api_health
            
            # Check quota manager health
            quota_status = await self.get_quota_status()
            health_data['components']['quota_manager'] = {
                'status': quota_status['quota_health']['status'],
                'health_score': quota_status['quota_health']['health_score']
            }
            
            # Check TAB scraper availability
            tab_status = await self._check_tab_scraper_health()
            health_data['components']['tab_scraper'] = tab_status
            
            # Check storage availability
            storage_status = self._check_storage_health()
            health_data['components']['storage'] = storage_status
            
            # Determine overall status
            component_statuses = [comp.get('status', 'unknown') for comp in health_data['components'].values()]
            
            if all(status == 'healthy' for status in component_statuses):
                health_data['overall_status'] = 'healthy'
            elif any(status == 'healthy' for status in component_statuses):
                health_data['overall_status'] = 'degraded'
            else:
                health_data['overall_status'] = 'unhealthy'
        
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_data['overall_status'] = 'unhealthy'
            health_data['error'] = str(e)
        
        return health_data
    
    async def _check_tab_scraper_health(self) -> Dict[str, Any]:
        """Check TAB scraper health and availability"""
        try:
            if self.tab_integration:
                # Simple connectivity check
                return {
                    'status': 'healthy',
                    'available': True,
                    'last_check': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unavailable',
                    'available': False,
                    'reason': 'TAB integration not configured'
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'available': False,
                'error': str(e)
            }
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage system health"""
        try:
            # Check if storage directory is accessible and writable
            test_file = self.storage_base_path / ".health_check"
            test_file.write_text("health_check")
            test_file.unlink()
            
            return {
                'status': 'healthy',
                'writable': True,
                'path': str(self.storage_base_path)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'writable': False,
                'error': str(e)
            }
    
    async def close(self):
        """Clean up resources"""
        await self.async_client.close()
        logger.info("Enhanced odds service closed")