"""
Async UFC Odds API Client

High-performance async client for The Odds API with advanced features:
- Connection pooling and session management
- Circuit breaker pattern for reliability
- Rate limiting to respect API quotas
- Retry logic with exponential backoff
- Comprehensive error handling and logging
"""

import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class AsyncOddsAPIError(Exception):
    """Custom exception for async Odds API errors"""
    pass


@dataclass
class CircuitBreakerState:
    """Circuit breaker state management"""
    failure_count: int = 0
    failure_threshold: int = 5
    timeout_duration: int = 300  # 5 minutes
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN


@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    requests_remaining: int = 1000
    requests_used: int = 0
    reset_time: Optional[datetime] = None


class AsyncUFCOddsAPIClient:
    """
    High-performance async client for The Odds API with enterprise features
    
    Features:
    - Async/await support for non-blocking operations
    - Connection pooling for improved performance
    - Circuit breaker pattern for fault tolerance
    - Rate limiting to respect API quotas
    - Retry logic with exponential backoff
    - Comprehensive monitoring and logging
    """
    
    def __init__(self, api_key: str, max_connections: int = 10, 
                 request_timeout: int = 30):
        """
        Initialize the async Odds API client
        
        Args:
            api_key: Your The Odds API key (required)
            max_connections: Maximum concurrent connections
            request_timeout: Request timeout in seconds
            
        Raises:
            AsyncOddsAPIError: If API key is invalid
        """
        if not api_key or len(api_key) < 16:
            raise AsyncOddsAPIError(
                "Valid API key is required. Get your free key at: https://the-odds-api.com/"
            )
        
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "mma_mixed_martial_arts"
        self.request_timeout = request_timeout
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreakerState()
        
        # Rate limiting
        self.rate_limit = RateLimitInfo()
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_connections = max_connections
        
        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        
        logger.info(f"Async Odds API client initialized (key: {api_key[:8]}...)")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'UFC-Predictor-Agent/1.0'}
            )
            
            logger.info("Created new aiohttp session with connection pooling")
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Closed aiohttp session")
    
    def _check_circuit_breaker(self):
        """Check circuit breaker state before making requests"""
        if self.circuit_breaker.state == "OPEN":
            if (self.circuit_breaker.last_failure_time and 
                time.time() - self.circuit_breaker.last_failure_time > self.circuit_breaker.timeout_duration):
                self.circuit_breaker.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise AsyncOddsAPIError(
                    f"Circuit breaker is OPEN. Too many failures. "
                    f"Try again in {self.circuit_breaker.timeout_duration} seconds."
                )
    
    def _record_success(self):
        """Record successful request for circuit breaker"""
        if self.circuit_breaker.state == "HALF_OPEN":
            self.circuit_breaker.state = "CLOSED"
            self.circuit_breaker.failure_count = 0
            logger.info("Circuit breaker reset to CLOSED state")
        elif self.circuit_breaker.state == "CLOSED":
            self.circuit_breaker.failure_count = max(0, self.circuit_breaker.failure_count - 1)
    
    def _record_failure(self):
        """Record failed request for circuit breaker"""
        self.circuit_breaker.failure_count += 1
        self.circuit_breaker.last_failure_time = time.time()
        
        if self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold:
            self.circuit_breaker.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.circuit_breaker.failure_count} failures"
            )
    
    async def _make_request_with_retry(self, url: str, params: Dict[str, str], 
                                     max_retries: int = 3) -> Tuple[Dict, Dict[str, str]]:
        """
        Make HTTP request with retry logic and exponential backoff
        
        Args:
            url: Request URL
            params: Request parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_data, headers)
            
        Raises:
            AsyncOddsAPIError: If all retries fail
        """
        await self._ensure_session()
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                async with self.session.get(url, params=params) as response:
                    # Track latency
                    latency = time.time() - start_time
                    self.request_count += 1
                    self.total_latency += latency
                    
                    # Update rate limit info
                    self._update_rate_limit_info(response.headers)
                    
                    # Check for API errors
                    if response.status == 401:
                        raise AsyncOddsAPIError(
                            "Invalid API key. Check your key at: https://the-odds-api.com/account/"
                        )
                    elif response.status == 422:
                        raise AsyncOddsAPIError(
                            "Invalid parameters. Check region and sport settings."
                        )
                    elif response.status == 429:
                        raise AsyncOddsAPIError(
                            "Rate limit exceeded. You've used all your API requests."
                        )
                    elif response.status != 200:
                        error_text = await response.text()
                        raise AsyncOddsAPIError(
                            f"API error {response.status}: {error_text}"
                        )
                    
                    # Parse response
                    data = await response.json()
                    
                    logger.info(
                        f"API request successful (attempt {attempt + 1}, "
                        f"latency: {latency:.2f}s, remaining: {self.rate_limit.requests_remaining})"
                    )
                    
                    return data, dict(response.headers)
                    
            except aiohttp.ClientError as e:
                if attempt == max_retries:
                    raise AsyncOddsAPIError(f"Network error after {max_retries + 1} attempts: {str(e)}")
                
                # Exponential backoff
                wait_time = (2 ** attempt) + (asyncio.get_event_loop().time() % 1)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)
                
            except json.JSONDecodeError as e:
                raise AsyncOddsAPIError(f"Invalid JSON response: {str(e)}")
    
    def _update_rate_limit_info(self, headers: Dict[str, str]):
        """Update rate limit information from response headers"""
        try:
            self.rate_limit.requests_remaining = int(
                headers.get('x-requests-remaining', self.rate_limit.requests_remaining)
            )
            self.rate_limit.requests_used = int(
                headers.get('x-requests-used', self.rate_limit.requests_used)
            )
        except (ValueError, TypeError):
            logger.warning("Could not parse rate limit headers")
    
    async def get_ufc_odds(self, region: str = "au") -> List[Dict]:
        """
        Fetch current UFC odds from The Odds API (async version)
        
        Args:
            region: Betting region ('au' for Australia, 'us' for US, etc.)
            
        Returns:
            List of fight data with odds
            
        Raises:
            AsyncOddsAPIError: If API call fails
        """
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Check rate limit
        if self.rate_limit.requests_remaining <= 0:
            raise AsyncOddsAPIError(
                "No API requests remaining. Wait for quota reset or upgrade your plan."
            )
        
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': region,
            'markets': 'h2h',  # Head-to-head (moneyline)
            'oddsFormat': 'decimal'
        }
        
        logger.info(f"Fetching UFC odds async (region: {region})")
        
        try:
            data, headers = await self._make_request_with_retry(url, params)
            
            if not data:
                raise AsyncOddsAPIError(
                    "No UFC events found. Check if there are upcoming UFC fights."
                )
            
            self._record_success()
            
            logger.info(f"Successfully fetched {len(data)} UFC events")
            return data
            
        except AsyncOddsAPIError:
            self._record_failure()
            raise
        except Exception as e:
            self._record_failure()
            raise AsyncOddsAPIError(f"Unexpected error: {str(e)}")
    
    async def format_odds_for_analysis(self, api_data: List[Dict], 
                                     target_fights: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Format The Odds API data for UFC predictor analysis (async version)
        
        Args:
            api_data: Raw data from The Odds API
            target_fights: Optional list of specific fights to extract
            
        Returns:
            Dictionary formatted for profitability analysis
        """
        logger.info(f"Formatting odds for {len(api_data)} events")
        
        formatted_odds = {}
        
        for event in api_data:
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            
            if not home_team or not away_team:
                continue
            
            # Create fight key
            fight_key = f"{home_team} vs {away_team}"
            
            # Filter for target fights if specified
            if target_fights:
                fight_found = False
                for target in target_fights:
                    if self._fights_match(fight_key, target):
                        fight_found = True
                        break
                if not fight_found:
                    continue
            
            # Extract best odds from all bookmakers
            best_home_odds = None
            best_away_odds = None
            bookmaker_info = []
            
            for bookmaker in event.get('bookmakers', []):
                bm_name = bookmaker.get('title', 'Unknown')
                
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            name = outcome.get('name', '')
                            price = outcome.get('price', 0)
                            
                            if name == home_team:
                                if best_home_odds is None or price > best_home_odds:
                                    best_home_odds = price
                            elif name == away_team:
                                if best_away_odds is None or price > best_away_odds:
                                    best_away_odds = price
                        
                        bookmaker_info.append(bm_name)
            
            if best_home_odds and best_away_odds:
                formatted_odds[fight_key] = {
                    'fighter_a': home_team,
                    'fighter_b': away_team,
                    'fighter_a_decimal_odds': best_home_odds,
                    'fighter_b_decimal_odds': best_away_odds,
                    'bookmakers': list(set(bookmaker_info)),
                    'commence_time': event.get('commence_time', ''),
                    'api_timestamp': datetime.now().isoformat()
                }
                
                logger.debug(
                    f"Formatted fight: {fight_key} "
                    f"({best_home_odds:.2f} vs {best_away_odds:.2f})"
                )
        
        logger.info(f"Formatted {len(formatted_odds)} fights for analysis")
        return formatted_odds
    
    def _fights_match(self, api_fight: str, target_fight: str, threshold: float = 0.8) -> bool:
        """
        Check if an API fight matches a target fight using fuzzy matching
        
        Args:
            api_fight: Fight string from API
            target_fight: Target fight string to match
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if fights match above threshold
        """
        # Normalize fight strings
        api_normalized = api_fight.lower().replace(' vs ', ' vs. ').strip()
        target_normalized = target_fight.lower().replace(' vs ', ' vs. ').strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, api_normalized, target_normalized).ratio()
        
        return similarity >= threshold
    
    async def get_event_odds(self, event_name: str, target_fights: List[str], 
                           region: str = "au") -> Dict[str, Dict]:
        """
        Get odds for a specific UFC event
        
        Args:
            event_name: Name of the UFC event
            target_fights: List of fights to fetch odds for
            region: Betting region
            
        Returns:
            Formatted odds data for the event
            
        Raises:
            AsyncOddsAPIError: If API call fails or no odds found
        """
        logger.info(f"Fetching odds for event: {event_name}")
        
        # Fetch all UFC odds
        api_data = await self.get_ufc_odds(region=region)
        
        # Format for specific event
        event_odds = await self.format_odds_for_analysis(api_data, target_fights)
        
        if not event_odds:
            raise AsyncOddsAPIError(
                f"No odds found for {event_name}. Possible reasons:\n"
                f"1. Event fights not available yet\n"
                f"2. Event has already occurred\n"
                f"3. Fighter name mismatches\n"
                f"4. Bookmakers not offering odds yet"
            )
        
        logger.info(f"Successfully retrieved odds for {len(event_odds)} fights")
        return event_odds
    
    async def monitor_odds_changes(self, previous_odds: Dict[str, Dict], 
                                 current_odds: Dict[str, Dict], 
                                 threshold: float = 0.1) -> Dict[str, List]:
        """
        Monitor significant odds changes between two snapshots
        
        Args:
            previous_odds: Previous odds snapshot
            current_odds: Current odds snapshot
            threshold: Minimum change to report (decimal odds)
            
        Returns:
            Dict with 'significant_changes' and 'new_opportunities'
        """
        significant_changes = []
        new_opportunities = []
        
        for fight_key in current_odds:
            if fight_key not in previous_odds:
                continue
            
            current_fight = current_odds[fight_key]
            previous_fight = previous_odds[fight_key]
            
            # Check fighter A odds change
            curr_a = current_fight['fighter_a_decimal_odds']
            prev_a = previous_fight['fighter_a_decimal_odds']
            change_a = abs(curr_a - prev_a)
            
            if change_a >= threshold:
                change_data = {
                    'fighter': current_fight['fighter_a'],
                    'fight_key': fight_key,
                    'previous_odds': prev_a,
                    'current_odds': curr_a,
                    'change': change_a,
                    'change_percent': (change_a / prev_a) * 100
                }
                significant_changes.append(change_data)
                
                if curr_a > prev_a:  # Odds got worse for fighter (better value)
                    new_opportunities.append(change_data)
            
            # Check fighter B odds change
            curr_b = current_fight['fighter_b_decimal_odds']
            prev_b = previous_fight['fighter_b_decimal_odds']
            change_b = abs(curr_b - prev_b)
            
            if change_b >= threshold:
                change_data = {
                    'fighter': current_fight['fighter_b'],
                    'fight_key': fight_key,
                    'previous_odds': prev_b,
                    'current_odds': curr_b,
                    'change': change_b,
                    'change_percent': (change_b / prev_b) * 100
                }
                significant_changes.append(change_data)
                
                if curr_b > prev_b:  # Odds got worse for fighter (better value)
                    new_opportunities.append(change_data)
        
        logger.info(
            f"Found {len(significant_changes)} significant changes, "
            f"{len(new_opportunities)} new opportunities"
        )
        
        return {
            'significant_changes': significant_changes,
            'new_opportunities': new_opportunities
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get client performance statistics
        
        Returns:
            Dict with performance metrics
        """
        avg_latency = (self.total_latency / self.request_count) if self.request_count > 0 else 0
        
        return {
            'total_requests': self.request_count,
            'average_latency': avg_latency,
            'circuit_breaker_state': self.circuit_breaker.state,
            'failure_count': self.circuit_breaker.failure_count,
            'rate_limit_remaining': self.rate_limit.requests_remaining,
            'rate_limit_used': self.rate_limit.requests_used
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the API connection
        
        Returns:
            Dict with health status
        """
        try:
            # Make a lightweight request
            start_time = time.time()
            data = await self.get_ufc_odds(region="au")
            latency = time.time() - start_time
            
            return {
                'status': 'healthy',
                'latency': latency,
                'events_available': len(data),
                'rate_limit_remaining': self.rate_limit.requests_remaining,
                'circuit_breaker_state': self.circuit_breaker.state
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state
            }


# Utility functions for compatibility with existing code
async def get_ufc_event_odds_async(api_key: str, event_name: str, 
                                 target_fights: List[str]) -> Dict[str, Dict]:
    """
    Async version of get_ufc_304_odds for compatibility
    
    Args:
        api_key: The Odds API key
        event_name: Name of the UFC event
        target_fights: List of fights to fetch
        
    Returns:
        Formatted odds data for the event
    """
    async with AsyncUFCOddsAPIClient(api_key) as client:
        return await client.get_event_odds(event_name, target_fights)


def convert_to_profitability_format(odds_data: Dict[str, Dict]) -> Dict[str, float]:
    """
    Convert Odds API format to the format expected by profitability analysis
    (Same as sync version - no changes needed)
    
    Args:
        odds_data: Formatted odds data
        
    Returns:
        Dictionary mapping fighter names to decimal odds
    """
    fighter_odds = {}
    
    for fight_key, fight_data in odds_data.items():
        fighter_a = fight_data['fighter_a']
        fighter_b = fight_data['fighter_b']
        
        fighter_odds[fighter_a] = fight_data['fighter_a_decimal_odds']
        fighter_odds[fighter_b] = fight_data['fighter_b_decimal_odds']
    
    return fighter_odds