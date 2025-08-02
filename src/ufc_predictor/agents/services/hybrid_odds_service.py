"""
Hybrid Odds Service for Phase 2A

Intelligent odds fetching that combines The Odds API with TAB scraping,
using quota management to maximize efficiency within the 500 request limit.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .quota_manager import QuotaManager, RequestPriority, create_quota_manager
from .async_odds_client import AsyncUFCOddsAPIClient
from .odds_service import UFCOddsService, OddsResult

logger = logging.getLogger(__name__)


@dataclass
class DataSourceResult:
    """Result from a data source with confidence scoring"""
    source_name: str
    odds_data: Dict[str, Dict]
    confidence_score: float  # 0.0 to 1.0
    fetch_timestamp: datetime
    latency_ms: int
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class HybridFetchResult:
    """Result from hybrid odds fetching"""
    primary_source: DataSourceResult
    validation_source: Optional[DataSourceResult] = None
    reconciled_data: Dict[str, Dict] = None
    confidence_score: float = 0.0
    api_requests_used: int = 0
    fallback_activated: bool = False


class TABScraperAdapter:
    """
    Adapter for TAB Australia scraper to work with hybrid system
    
    Wraps the existing TAB scraper to provide consistent interface
    """
    
    def __init__(self):
        """Initialize TAB scraper adapter"""
        # Import the existing TAB scraper
        try:
            import sys
            sys.path.append('../../webscraper')
            from tab_australia_scraper import TABAustraliaUFCScraper
            self.scraper = TABAustraliaUFCScraper()
            self.available = True
            logger.info("TAB scraper adapter initialized successfully")
        except ImportError as e:
            logger.warning(f"TAB scraper not available: {e}")
            self.scraper = None
            self.available = False
    
    async def fetch_ufc_odds(self, target_fights: List[str] = None) -> DataSourceResult:
        """
        Fetch UFC odds using TAB scraper
        
        Args:
            target_fights: Optional list of specific fights to fetch
            
        Returns:
            DataSourceResult with scraped odds data
        """
        start_time = datetime.now()
        
        if not self.available:
            return DataSourceResult(
                source_name="tab_australia",
                odds_data={},
                confidence_score=0.0,
                fetch_timestamp=start_time,
                latency_ms=0,
                success=False,
                error_message="TAB scraper not available"
            )
        
        try:
            # Use existing scraper functionality
            if hasattr(self.scraper, 'scrape_ufc_odds'):
                scraped_data = await self.scraper.scrape_ufc_odds(target_fights)
            else:
                # Fallback to synchronous scraping
                scraped_data = self.scraper.scrape_upcoming_ufc_fights()
            
            # Convert to standardized format
            formatted_odds = self._format_tab_data(scraped_data, target_fights)
            
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataSourceResult(
                source_name="tab_australia",
                odds_data=formatted_odds,
                confidence_score=0.85,  # TAB scraping is quite reliable
                fetch_timestamp=start_time,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"TAB scraper error: {str(e)}")
            
            return DataSourceResult(
                source_name="tab_australia",
                odds_data={},
                confidence_score=0.0,
                fetch_timestamp=start_time,
                latency_ms=latency,
                success=False,
                error_message=str(e)
            )
    
    def _format_tab_data(self, scraped_data: Any, target_fights: List[str] = None) -> Dict[str, Dict]:
        """
        Convert TAB scraped data to standardized odds format
        
        Args:
            scraped_data: Raw data from TAB scraper
            target_fights: Target fights to filter for
            
        Returns:
            Dictionary in standardized odds format
        """
        formatted_odds = {}
        
        # Handle different TAB scraper data formats
        if isinstance(scraped_data, list):
            for fight_data in scraped_data:
                if hasattr(fight_data, 'fighter_a') and hasattr(fight_data, 'fighter_b'):
                    fight_key = f"{fight_data.fighter_a} vs {fight_data.fighter_b}"
                    
                    # Filter for target fights if specified
                    if target_fights and not self._match_target_fight(fight_key, target_fights):
                        continue
                    
                    formatted_odds[fight_key] = {
                        'fighter_a': fight_data.fighter_a,
                        'fighter_b': fight_data.fighter_b,
                        'fighter_a_decimal_odds': float(fight_data.fighter_a_decimal_odds),
                        'fighter_b_decimal_odds': float(fight_data.fighter_b_decimal_odds),
                        'bookmakers': ['TAB Australia'],
                        'commence_time': getattr(fight_data, 'commence_time', ''),
                        'api_timestamp': datetime.now().isoformat(),
                        'source': 'tab_scraping'
                    }
        
        elif isinstance(scraped_data, dict):
            # Handle dictionary format
            for fight_key, fight_data in scraped_data.items():
                if target_fights and not self._match_target_fight(fight_key, target_fights):
                    continue
                    
                formatted_odds[fight_key] = {
                    'fighter_a': fight_data.get('fighter_a', ''),
                    'fighter_b': fight_data.get('fighter_b', ''),
                    'fighter_a_decimal_odds': float(fight_data.get('fighter_a_decimal_odds', 0)),
                    'fighter_b_decimal_odds': float(fight_data.get('fighter_b_decimal_odds', 0)),
                    'bookmakers': ['TAB Australia'],
                    'commence_time': fight_data.get('commence_time', ''),
                    'api_timestamp': datetime.now().isoformat(),
                    'source': 'tab_scraping'
                }
        
        return formatted_odds
    
    def _match_target_fight(self, fight_key: str, target_fights: List[str]) -> bool:
        """Check if fight matches any target fight using fuzzy matching"""
        from difflib import SequenceMatcher
        
        for target in target_fights:
            similarity = SequenceMatcher(None, fight_key.lower(), target.lower()).ratio()
            if similarity > 0.8:  # 80% similarity threshold
                return True
        return False


class HybridOddsService:
    """
    Hybrid odds service that intelligently combines API and scraping data sources
    
    Features:
    - Priority-based API usage with quota management
    - Automatic fallback to TAB scraper
    - Data source confidence scoring and reconciliation
    - Performance monitoring and optimization
    """
    
    def __init__(self, api_key: str, quota_manager: Optional[QuotaManager] = None,
                 storage_base_path: str = 'odds'):
        """
        Initialize hybrid odds service
        
        Args:
            api_key: The Odds API key
            quota_manager: Optional quota manager (creates default if None)
            storage_base_path: Base path for odds storage
        """
        self.api_key = api_key
        self.quota_manager = quota_manager or create_quota_manager()
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(exist_ok=True)
        
        # Initialize data sources
        self.api_client = AsyncUFCOddsAPIClient(api_key)
        self.tab_adapter = TABScraperAdapter()
        self.odds_service = UFCOddsService(None, str(storage_base_path))
        
        # Performance tracking
        self.fetch_history: List[HybridFetchResult] = []
        
        logger.info("Hybrid odds service initialized")
    
    async def fetch_event_odds(self, event_name: str, target_fights: List[str],
                              priority: RequestPriority = RequestPriority.MEDIUM,
                              enable_validation: bool = True) -> HybridFetchResult:
        """
        Fetch odds for an event using hybrid approach
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to fetch odds for
            priority: Priority level for API usage
            enable_validation: Whether to use API for validation
            
        Returns:
            HybridFetchResult with comprehensive odds data
        """
        logger.info(f"Fetching odds for {event_name} with {len(target_fights)} target fights")
        
        result = HybridFetchResult(
            primary_source=None,
            api_requests_used=0
        )
        
        # Step 1: Always try TAB scraper first (free and fast)
        logger.info("Fetching from TAB Australia scraper...")
        tab_result = await self.tab_adapter.fetch_ufc_odds(target_fights)
        
        if tab_result.success and tab_result.odds_data:
            result.primary_source = tab_result
            logger.info(f"TAB scraper successful: {len(tab_result.odds_data)} fights found")
            
            # Step 2: Decide if API validation is needed
            should_validate = (
                enable_validation and
                not self.quota_manager.should_use_fallback(priority) and
                self._should_validate_with_api(tab_result, priority)
            )
            
            if should_validate:
                logger.info("Requesting API validation...")
                api_result = await self._fetch_api_validation(event_name, target_fights, priority)
                
                if api_result and api_result.success:
                    result.validation_source = api_result
                    result.api_requests_used = 1
                    
                    # Reconcile data sources
                    result.reconciled_data = self._reconcile_data_sources(tab_result, api_result)
                    result.confidence_score = self._calculate_combined_confidence(tab_result, api_result)
                    
                    logger.info("API validation successful, data reconciled")
                else:
                    logger.warning("API validation failed, using TAB data only")
                    result.reconciled_data = tab_result.odds_data
                    result.confidence_score = tab_result.confidence_score
            else:
                logger.info("Skipping API validation (quota/priority constraints)")
                result.reconciled_data = tab_result.odds_data
                result.confidence_score = tab_result.confidence_score
        
        else:
            # Step 3: TAB failed, try API as primary source
            logger.warning("TAB scraper failed, trying API as primary source...")
            result.fallback_activated = True
            
            api_result = await self._fetch_api_primary(event_name, target_fights, priority)
            
            if api_result and api_result.success:
                result.primary_source = api_result
                result.reconciled_data = api_result.odds_data
                result.confidence_score = api_result.confidence_score
                result.api_requests_used = 1
                logger.info("API primary fetch successful")
            else:
                logger.error("Both TAB and API sources failed")
                result.primary_source = tab_result  # Include failed result for debugging
                result.reconciled_data = {}
                result.confidence_score = 0.0
        
        # Store result history
        self.fetch_history.append(result)
        
        return result
    
    def _should_validate_with_api(self, tab_result: DataSourceResult, 
                                 priority: RequestPriority) -> bool:
        """
        Determine if API validation is worth the quota cost
        
        Args:
            tab_result: TAB scraper result
            priority: Request priority
            
        Returns:
            True if API validation should be performed
        """
        # Always validate CRITICAL priority requests
        if priority == RequestPriority.CRITICAL:
            return True
        
        # Validate if we found high-value opportunities in TAB data
        if self._detect_high_value_opportunities(tab_result.odds_data):
            return True
        
        # Validate if TAB data seems suspicious or incomplete
        if self._detect_data_quality_issues(tab_result):
            return True
        
        # Skip validation for LOW priority or when quota is low
        if priority == RequestPriority.LOW:
            return False
        
        # Default: validate if quota allows
        quota_status = self.quota_manager.get_quota_status()
        return quota_status.requests_remaining_today > 5
    
    def _detect_high_value_opportunities(self, odds_data: Dict[str, Dict]) -> bool:
        """Detect if odds data contains high-value betting opportunities"""
        for fight_key, fight_data in odds_data.items():
            try:
                odds_a = fight_data.get('fighter_a_decimal_odds', 0)
                odds_b = fight_data.get('fighter_b_decimal_odds', 0)
                
                # Look for high odds (potential upsets) or significant discrepancies
                if odds_a > 3.0 or odds_b > 3.0:  # Potential upset opportunities
                    return True
                
                if abs(odds_a - odds_b) > 1.5:  # Significant favorite/underdog split
                    return True
                    
            except (ValueError, TypeError):
                continue
        
        return False
    
    def _detect_data_quality_issues(self, tab_result: DataSourceResult) -> bool:
        """Detect potential data quality issues that warrant API validation"""
        # Low confidence score
        if tab_result.confidence_score < 0.7:
            return True
        
        # High latency (potential scraping issues)
        if tab_result.latency_ms > 10000:  # 10 seconds
            return True
        
        # Very few fights found (potential scraping failure)
        if len(tab_result.odds_data) < 2:
            return True
        
        return False
    
    async def _fetch_api_validation(self, event_name: str, target_fights: List[str],
                                   priority: RequestPriority) -> Optional[DataSourceResult]:
        """Fetch API data for validation purposes"""
        # Request quota
        granted, request_id = await self.quota_manager.request_quota(
            priority, "get_ufc_odds_validation", f"{event_name}_{len(target_fights)}_fights"
        )
        
        if not granted:
            logger.info("API quota not granted for validation")
            return None
        
        start_time = datetime.now()
        
        try:
            # Fetch from API
            api_data = await self.api_client.get_event_odds(event_name, target_fights)
            
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record successful completion
            if request_id:
                self.quota_manager.record_request_completion(request_id, True, latency)
            
            return DataSourceResult(
                source_name="odds_api",
                odds_data=api_data,
                confidence_score=0.95,  # API data is highly reliable
                fetch_timestamp=start_time,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record failed completion
            if request_id:
                self.quota_manager.record_request_completion(request_id, False, latency)
            
            logger.error(f"API validation fetch failed: {str(e)}")
            return None
    
    async def _fetch_api_primary(self, event_name: str, target_fights: List[str],
                                priority: RequestPriority) -> Optional[DataSourceResult]:
        """Fetch API data as primary source"""
        # Request quota with higher priority since this is primary
        api_priority = RequestPriority.HIGH if priority == RequestPriority.CRITICAL else priority
        
        granted, request_id = await self.quota_manager.request_quota(
            api_priority, "get_ufc_odds_primary", f"{event_name}_{len(target_fights)}_fights"
        )
        
        if not granted:
            logger.warning("API quota not granted for primary fetch")
            return None
        
        start_time = datetime.now()
        
        try:
            # Fetch from API
            api_data = await self.api_client.get_event_odds(event_name, target_fights)
            
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record successful completion
            if request_id:
                self.quota_manager.record_request_completion(request_id, True, latency)
            
            return DataSourceResult(
                source_name="odds_api",
                odds_data=api_data,
                confidence_score=0.95,
                fetch_timestamp=start_time,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record failed completion
            if request_id:
                self.quota_manager.record_request_completion(request_id, False, latency)
            
            logger.error(f"API primary fetch failed: {str(e)}")
            return None
    
    def _reconcile_data_sources(self, tab_result: DataSourceResult,
                               api_result: DataSourceResult) -> Dict[str, Dict]:
        """
        Reconcile data from TAB and API sources
        
        Args:
            tab_result: TAB scraper result
            api_result: API result
            
        Returns:
            Reconciled odds data with best available information
        """
        reconciled = {}
        
        # Start with API data (higher confidence)
        api_fights = set(api_result.odds_data.keys())
        tab_fights = set(tab_result.odds_data.keys())
        
        # Use API data for fights found in both sources
        for fight_key in api_fights:
            reconciled[fight_key] = api_result.odds_data[fight_key].copy()
            reconciled[fight_key]['data_sources'] = ['odds_api']
            
            # Add TAB validation if available
            if fight_key in tab_fights:
                tab_fight = tab_result.odds_data[fight_key]
                reconciled[fight_key]['tab_validation'] = {
                    'tab_odds_a': tab_fight.get('fighter_a_decimal_odds'),
                    'tab_odds_b': tab_fight.get('fighter_b_decimal_odds'),
                    'odds_discrepancy_a': abs(
                        float(tab_fight.get('fighter_a_decimal_odds', 0)) - 
                        float(reconciled[fight_key]['fighter_a_decimal_odds'])
                    ),
                    'odds_discrepancy_b': abs(
                        float(tab_fight.get('fighter_b_decimal_odds', 0)) - 
                        float(reconciled[fight_key]['fighter_b_decimal_odds'])
                    )
                }
                reconciled[fight_key]['data_sources'].append('tab_australia')
        
        # Add TAB-only fights (lower confidence but still valuable)
        for fight_key in tab_fights - api_fights:
            reconciled[fight_key] = tab_result.odds_data[fight_key].copy()
            reconciled[fight_key]['data_sources'] = ['tab_australia']
            reconciled[fight_key]['confidence_adjustment'] = 0.85  # Lower confidence
        
        logger.info(
            f"Reconciled data: {len(api_fights)} API fights, "
            f"{len(tab_fights)} TAB fights, {len(reconciled)} total"
        )
        
        return reconciled
    
    def _calculate_combined_confidence(self, tab_result: DataSourceResult,
                                     api_result: DataSourceResult) -> float:
        """Calculate combined confidence score from multiple sources"""
        # Weighted average based on source reliability
        tab_weight = 0.3
        api_weight = 0.7
        
        combined_confidence = (
            tab_result.confidence_score * tab_weight +
            api_result.confidence_score * api_weight
        )
        
        # Bonus for data source agreement
        agreement_bonus = self._calculate_data_source_agreement(tab_result, api_result)
        combined_confidence = min(1.0, combined_confidence + agreement_bonus)
        
        return combined_confidence
    
    def _calculate_data_source_agreement(self, tab_result: DataSourceResult,
                                       api_result: DataSourceResult) -> float:
        """Calculate bonus confidence based on data source agreement"""
        if not tab_result.odds_data or not api_result.odds_data:
            return 0.0
        
        agreements = []
        
        for fight_key in set(tab_result.odds_data.keys()) & set(api_result.odds_data.keys()):
            try:
                tab_fight = tab_result.odds_data[fight_key]
                api_fight = api_result.odds_data[fight_key]
                
                tab_odds_a = float(tab_fight.get('fighter_a_decimal_odds', 0))
                api_odds_a = float(api_fight.get('fighter_a_decimal_odds', 0))
                
                tab_odds_b = float(tab_fight.get('fighter_b_decimal_odds', 0))
                api_odds_b = float(api_fight.get('fighter_b_decimal_odds', 0))
                
                # Calculate percentage differences
                if api_odds_a > 0 and api_odds_b > 0:
                    diff_a = abs(tab_odds_a - api_odds_a) / api_odds_a
                    diff_b = abs(tab_odds_b - api_odds_b) / api_odds_b
                    
                    # Consider it agreement if differences are < 10%
                    if diff_a < 0.1 and diff_b < 0.1:
                        agreements.append(1.0)
                    else:
                        agreements.append(max(0.0, 1.0 - max(diff_a, diff_b)))
                        
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        
        if agreements:
            avg_agreement = sum(agreements) / len(agreements)
            return avg_agreement * 0.1  # Up to 10% confidence bonus
        
        return 0.0
    
    async def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status and performance metrics"""
        quota_status = self.quota_manager.get_quota_status()
        performance_metrics = self.quota_manager.get_performance_metrics()
        
        return {
            'quota_status': {
                'requests_used_today': quota_status.requests_used_today,
                'requests_remaining_today': quota_status.requests_remaining_today,
                'requests_used_monthly': quota_status.requests_used_monthly,
                'budget_remaining': quota_status.budget_remaining_monthly
            },
            'performance_metrics': performance_metrics,
            'forecast': self.quota_manager.get_quota_forecast(),
            'hybrid_service_metrics': self._get_hybrid_metrics()
        }
    
    def _get_hybrid_metrics(self) -> Dict[str, Any]:
        """Get hybrid service specific metrics"""
        if not self.fetch_history:
            return {}
        
        total_fetches = len(self.fetch_history)
        api_requests_used = sum(r.api_requests_used for r in self.fetch_history)
        fallback_activations = sum(1 for r in self.fetch_history if r.fallback_activated)
        
        avg_confidence = sum(r.confidence_score for r in self.fetch_history) / total_fetches
        
        return {
            'total_hybrid_fetches': total_fetches,
            'api_requests_used': api_requests_used,
            'fallback_activation_rate': fallback_activations / total_fetches,
            'average_confidence_score': avg_confidence,
            'api_efficiency': api_requests_used / total_fetches if total_fetches > 0 else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check quota manager
        try:
            quota_status = self.quota_manager.get_quota_status()
            health['components']['quota_manager'] = {
                'status': 'healthy',
                'quota_remaining': quota_status.requests_remaining_today,
                'budget_remaining': quota_status.budget_remaining_monthly
            }
        except Exception as e:
            health['components']['quota_manager'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall_status'] = 'unhealthy'
        
        # Check API client
        try:
            api_health = await self.api_client.health_check()
            health['components']['api_client'] = api_health
            if api_health.get('status') != 'healthy':
                health['overall_status'] = 'degraded'
        except Exception as e:
            health['components']['api_client'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall_status'] = 'unhealthy'
        
        # Check TAB adapter
        health['components']['tab_adapter'] = {
            'status': 'healthy' if self.tab_adapter.available else 'unavailable',
            'available': self.tab_adapter.available
        }
        
        return health


# Factory function for easy initialization
def create_hybrid_odds_service(api_key: str, daily_quota: int = 16,
                              monthly_budget: float = 50.0) -> HybridOddsService:
    """
    Create a hybrid odds service with standard configuration
    
    Args:
        api_key: The Odds API key
        daily_quota: Daily API request quota
        monthly_budget: Monthly budget in USD
        
    Returns:
        Configured HybridOddsService instance
    """
    quota_manager = create_quota_manager(daily_quota, monthly_budget)
    return HybridOddsService(api_key, quota_manager)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_hybrid_service():
        """Test hybrid odds service functionality"""
        
        # Create hybrid service
        api_key = "test_api_key"  # Replace with actual key for testing
        hybrid_service = create_hybrid_odds_service(api_key)
        
        # Test event odds fetching
        event_name = "UFC_Test_Event"
        target_fights = ["Fighter A vs Fighter B", "Fighter C vs Fighter D"]
        
        print(f"Testing hybrid odds fetch for {event_name}...")
        
        result = await hybrid_service.fetch_event_odds(
            event_name, target_fights, RequestPriority.HIGH
        )
        
        print(f"Fetch result:")
        print(f"  Primary source: {result.primary_source.source_name if result.primary_source else None}")
        print(f"  API requests used: {result.api_requests_used}")
        print(f"  Confidence score: {result.confidence_score:.2f}")
        print(f"  Fights found: {len(result.reconciled_data or {})}")
        
        # Test quota status
        quota_status = await hybrid_service.get_quota_status()
        print(f"\nQuota status: {quota_status['quota_status']}")
        
        # Test health check
        health = await hybrid_service.health_check()
        print(f"\nHealth check: {health['overall_status']}")
    
    # Run test (commented out to avoid actual API calls)
    # asyncio.run(test_hybrid_service())