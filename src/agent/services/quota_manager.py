"""
API Quota Management System for Phase 2A

Intelligent allocation of 500 monthly API requests with priority-based distribution
and automatic fallback to scraping when quota is exhausted.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for API request allocation"""
    CRITICAL = "CRITICAL"    # Title fights, main events (60% allocation)
    HIGH = "HIGH"           # Co-main events, ranked fighters (20% allocation)  
    MEDIUM = "MEDIUM"       # Undercard value opportunities (10% allocation)
    LOW = "LOW"            # Health checks, discovery (10% allocation)


@dataclass
class QuotaAllocation:
    """Quota allocation configuration"""
    daily_limit: int = 16  # ~500/month = ~16/day
    monthly_limit: int = 500
    
    # Priority-based allocation (percentages of daily limit)
    priority_allocations: Dict[RequestPriority, float] = field(default_factory=lambda: {
        RequestPriority.CRITICAL: 0.60,  # 60% for critical requests
        RequestPriority.HIGH: 0.20,      # 20% for high priority
        RequestPriority.MEDIUM: 0.10,    # 10% for medium priority
        RequestPriority.LOW: 0.10        # 10% for low priority
    })
    
    # Cost management
    cost_per_request: float = 0.10  # $0.10 per request (estimated)
    monthly_budget_usd: float = 50.0


@dataclass
class RequestRecord:
    """Individual API request record"""
    request_id: str
    timestamp: datetime
    priority: RequestPriority
    endpoint: str
    fight_key: Optional[str] = None
    cost_usd: float = 0.10
    success: bool = True
    response_time_ms: Optional[int] = None
    quota_remaining_after: Optional[int] = None


@dataclass
class QuotaStatus:
    """Current quota status information"""
    requests_used_today: int = 0
    requests_remaining_today: int = 16
    requests_used_monthly: int = 0
    requests_remaining_monthly: int = 500
    
    cost_accumulated_monthly: float = 0.0
    budget_remaining_monthly: float = 50.0
    
    priority_usage: Dict[RequestPriority, int] = field(default_factory=dict)
    fallback_activations: int = 0
    last_reset_date: datetime = field(default_factory=datetime.now)


class QuotaManager:
    """
    Intelligent API quota management system
    
    Features:
    - Priority-based request allocation
    - Automatic fallback activation
    - Cost monitoring and budget protection
    - Usage forecasting and optimization
    - Request queuing and throttling
    """
    
    def __init__(self, config: QuotaAllocation, storage_path: str = "quota_data"):
        """
        Initialize quota manager
        
        Args:
            config: Quota allocation configuration
            storage_path: Path for persistent quota data storage
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Internal state
        self.status = QuotaStatus()
        self.request_history: List[RequestRecord] = []
        self.request_queue: asyncio.Queue = asyncio.Queue()
        
        # Load persistent state
        self._load_state()
        
        # Check if daily reset is needed
        self._check_daily_reset()
        
        logger.info(f"QuotaManager initialized with {self.config.daily_limit} daily requests")
    
    def _load_state(self):
        """Load persistent quota state from disk"""
        state_file = self.storage_path / "quota_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Restore status
                self.status.requests_used_today = data.get('requests_used_today', 0)
                self.status.requests_used_monthly = data.get('requests_used_monthly', 0)
                self.status.cost_accumulated_monthly = data.get('cost_accumulated_monthly', 0.0)
                self.status.fallback_activations = data.get('fallback_activations', 0)
                
                # Parse last reset date
                if 'last_reset_date' in data:
                    self.status.last_reset_date = datetime.fromisoformat(data['last_reset_date'])
                
                # Restore priority usage
                priority_usage = data.get('priority_usage', {})
                self.status.priority_usage = {
                    RequestPriority(k): v for k, v in priority_usage.items()
                }
                
                logger.info(f"Loaded quota state: {self.status.requests_used_today} used today")
                
            except Exception as e:
                logger.warning(f"Failed to load quota state: {e}, starting fresh")
                self.status = QuotaStatus()
    
    def _save_state(self):
        """Save persistent quota state to disk"""
        state_file = self.storage_path / "quota_state.json"
        
        try:
            data = {
                'requests_used_today': self.status.requests_used_today,
                'requests_used_monthly': self.status.requests_used_monthly,
                'cost_accumulated_monthly': self.status.cost_accumulated_monthly,
                'fallback_activations': self.status.fallback_activations,
                'last_reset_date': self.status.last_reset_date.isoformat(),
                'priority_usage': {k.value: v for k, v in self.status.priority_usage.items()}
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save quota state: {e}")
    
    def _check_daily_reset(self):
        """Check if daily quota should be reset"""
        today = datetime.now().date()
        last_reset_date = self.status.last_reset_date.date()
        
        if today > last_reset_date:
            # Reset daily counters
            self.status.requests_used_today = 0
            self.status.last_reset_date = datetime.now()
            
            # Check if monthly reset is needed
            if today.month != last_reset_date.month:
                self.status.requests_used_monthly = 0
                self.status.cost_accumulated_monthly = 0.0
                self.status.fallback_activations = 0
                self.status.priority_usage = {}
                
                logger.info("Monthly quota reset performed")
            else:
                logger.info("Daily quota reset performed")
            
            self._save_state()
    
    def _update_remaining_counts(self):
        """Update remaining request counts"""
        self.status.requests_remaining_today = (
            self.config.daily_limit - self.status.requests_used_today
        )
        self.status.requests_remaining_monthly = (
            self.config.monthly_limit - self.status.requests_used_monthly
        )
        self.status.budget_remaining_monthly = (
            self.config.monthly_budget_usd - self.status.cost_accumulated_monthly
        )
    
    async def request_quota(self, priority: RequestPriority, endpoint: str,
                          fight_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Request quota for an API call
        
        Args:
            priority: Request priority level
            endpoint: API endpoint being called
            fight_key: Optional fight identifier for tracking
            
        Returns:
            Tuple of (granted: bool, request_id: Optional[str])
        """
        self._check_daily_reset()
        
        # Generate request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{priority.value}"
        
        # Check if quota available
        if not self._can_grant_request(priority):
            logger.warning(f"Quota denied for {priority.value} request: {endpoint}")
            return False, None
        
        # Check budget constraints
        if self.status.cost_accumulated_monthly + self.config.cost_per_request > self.config.monthly_budget_usd:
            logger.warning(f"Budget limit reached, denying request: {endpoint}")
            return False, None
        
        # Grant quota
        self.status.requests_used_today += 1
        self.status.requests_used_monthly += 1
        self.status.cost_accumulated_monthly += self.config.cost_per_request
        
        # Update priority usage
        if priority not in self.status.priority_usage:
            self.status.priority_usage[priority] = 0
        self.status.priority_usage[priority] += 1
        
        # Update remaining counts
        self._update_remaining_counts()
        
        # Create request record
        request_record = RequestRecord(
            request_id=request_id,
            timestamp=datetime.now(),
            priority=priority,
            endpoint=endpoint,
            fight_key=fight_key,
            cost_usd=self.config.cost_per_request
        )
        
        self.request_history.append(request_record)
        
        # Save state
        self._save_state()
        
        logger.info(
            f"Quota granted: {priority.value} for {endpoint} "
            f"(Remaining today: {self.status.requests_remaining_today})"
        )
        
        return True, request_id
    
    def _can_grant_request(self, priority: RequestPriority) -> bool:
        """Check if request can be granted based on priority allocation"""
        
        # Always deny if no quota remaining
        if self.status.requests_remaining_today <= 0:
            return False
        
        # Get priority allocation for today
        priority_limit = int(self.config.daily_limit * self.config.priority_allocations[priority])
        priority_used = self.status.priority_usage.get(priority, 0)
        
        # Check if priority quota available
        if priority_used >= priority_limit:
            # Check if we can borrow from lower priority allocations
            return self._can_borrow_quota(priority)
        
        return True
    
    def _can_borrow_quota(self, priority: RequestPriority) -> bool:
        """Check if quota can be borrowed from lower priority allocations"""
        
        # Define priority hierarchy
        priority_hierarchy = [
            RequestPriority.CRITICAL,
            RequestPriority.HIGH,
            RequestPriority.MEDIUM,
            RequestPriority.LOW
        ]
        
        current_index = priority_hierarchy.index(priority)
        
        # Higher priority can borrow from lower priority
        for lower_priority in priority_hierarchy[current_index + 1:]:
            lower_limit = int(self.config.daily_limit * self.config.priority_allocations[lower_priority])
            lower_used = self.status.priority_usage.get(lower_priority, 0)
            
            if lower_used < lower_limit:
                logger.info(f"Borrowing quota from {lower_priority.value} for {priority.value}")
                return True
        
        return False
    
    def record_request_completion(self, request_id: str, success: bool,
                                response_time_ms: Optional[int] = None):
        """
        Record completion of an API request
        
        Args:
            request_id: Request identifier
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
        """
        for record in self.request_history:
            if record.request_id == request_id:
                record.success = success
                record.response_time_ms = response_time_ms
                record.quota_remaining_after = self.status.requests_remaining_today
                break
        
        logger.debug(f"Recorded completion for {request_id}: success={success}")
    
    def get_quota_status(self) -> QuotaStatus:
        """Get current quota status"""
        self._check_daily_reset()
        self._update_remaining_counts()
        return self.status
    
    def should_use_fallback(self, priority: RequestPriority) -> bool:
        """
        Determine if fallback should be used instead of API
        
        Args:
            priority: Request priority level
            
        Returns:
            True if fallback should be used
        """
        # Use fallback if quota exhausted
        if self.status.requests_remaining_today <= 0:
            return True
        
        # Use fallback if budget exhausted
        if self.status.budget_remaining_monthly < self.config.cost_per_request:
            return True
        
        # Use fallback if low priority and nearing limit
        if priority in [RequestPriority.MEDIUM, RequestPriority.LOW]:
            remaining_ratio = self.status.requests_remaining_today / self.config.daily_limit
            if remaining_ratio < 0.2:  # Less than 20% quota remaining
                return True
        
        return False
    
    def get_quota_forecast(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Forecast quota usage and exhaustion risk
        
        Args:
            hours_ahead: Hours to forecast ahead
            
        Returns:
            Dictionary with forecast information
        """
        # Calculate current usage rate
        hours_elapsed_today = (datetime.now() - self.status.last_reset_date).total_seconds() / 3600
        
        if hours_elapsed_today > 0:
            usage_rate_per_hour = self.status.requests_used_today / hours_elapsed_today
        else:
            usage_rate_per_hour = 0
        
        # Forecast usage
        forecasted_usage = usage_rate_per_hour * hours_ahead
        forecasted_remaining = max(0, self.status.requests_remaining_today - forecasted_usage)
        
        # Risk assessment
        exhaustion_risk = "low"
        if forecasted_remaining <= 5:
            exhaustion_risk = "high"
        elif forecasted_remaining <= 10:
            exhaustion_risk = "medium"
        
        return {
            'current_usage_rate_per_hour': usage_rate_per_hour,
            'forecasted_usage_in_period': forecasted_usage,
            'forecasted_remaining': forecasted_remaining,
            'quota_exhaustion_risk': exhaustion_risk,
            'recommended_action': self._get_recommended_action(exhaustion_risk),
            'cost_forecast_monthly': self.status.cost_accumulated_monthly + (forecasted_usage * self.config.cost_per_request)
        }
    
    def _get_recommended_action(self, exhaustion_risk: str) -> str:
        """Get recommended action based on exhaustion risk"""
        if exhaustion_risk == "high":
            return "activate_fallback_mode"
        elif exhaustion_risk == "medium":
            return "reduce_low_priority_requests"
        else:
            return "continue_normal_operation"
    
    def activate_fallback_mode(self, reason: str = "quota_exhaustion"):
        """
        Activate fallback mode
        
        Args:
            reason: Reason for activating fallback
        """
        self.status.fallback_activations += 1
        self._save_state()
        
        logger.warning(f"Fallback mode activated: {reason}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the quota system"""
        if not self.request_history:
            return {}
        
        # Calculate metrics
        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.request_history if r.success)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Response time metrics
        response_times = [r.response_time_ms for r in self.request_history if r.response_time_ms]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Cost efficiency
        cost_per_successful_request = (
            self.status.cost_accumulated_monthly / successful_requests 
            if successful_requests > 0 else 0
        )
        
        return {
            'total_requests': total_requests,
            'success_rate': success_rate,
            'average_response_time_ms': avg_response_time,
            'cost_per_successful_request': cost_per_successful_request,
            'quota_utilization_rate': self.status.requests_used_monthly / self.config.monthly_limit,
            'budget_utilization_rate': self.status.cost_accumulated_monthly / self.config.monthly_budget_usd,
            'fallback_activation_count': self.status.fallback_activations
        }
    
    def export_usage_report(self) -> Dict[str, Any]:
        """Export comprehensive usage report"""
        return {
            'quota_status': {
                'requests_used_today': self.status.requests_used_today,
                'requests_remaining_today': self.status.requests_remaining_today,
                'requests_used_monthly': self.status.requests_used_monthly,
                'requests_remaining_monthly': self.status.requests_remaining_monthly
            },
            'cost_analysis': {
                'cost_accumulated_monthly': self.status.cost_accumulated_monthly,
                'budget_remaining_monthly': self.status.budget_remaining_monthly,
                'cost_per_request': self.config.cost_per_request,
                'projected_monthly_cost': self.status.cost_accumulated_monthly * (30 / datetime.now().day)
            },
            'priority_breakdown': dict(self.status.priority_usage),
            'performance_metrics': self.get_performance_metrics(),
            'forecast': self.get_quota_forecast(),
            'configuration': {
                'daily_limit': self.config.daily_limit,
                'monthly_limit': self.config.monthly_limit,
                'monthly_budget_usd': self.config.monthly_budget_usd,
                'priority_allocations': {k.value: v for k, v in self.config.priority_allocations.items()}
            }
        }


# Factory function for easy initialization
def create_quota_manager(daily_limit: int = 16, monthly_budget: float = 50.0) -> QuotaManager:
    """
    Create a quota manager with standard configuration
    
    Args:
        daily_limit: Daily API request limit
        monthly_budget: Monthly budget in USD
        
    Returns:
        Configured QuotaManager instance
    """
    config = QuotaAllocation(
        daily_limit=daily_limit,
        monthly_limit=daily_limit * 30,  # Approximate monthly limit
        monthly_budget_usd=monthly_budget
    )
    
    return QuotaManager(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_quota_manager():
        """Test quota manager functionality"""
        
        # Create quota manager
        quota_manager = create_quota_manager(daily_limit=16, monthly_budget=50.0)
        
        # Test quota requests
        print("Testing quota requests...")
        
        # Critical request (should be granted)
        granted, request_id = await quota_manager.request_quota(
            RequestPriority.CRITICAL, "get_ufc_odds", "Fighter A vs Fighter B"
        )
        print(f"Critical request: granted={granted}, id={request_id}")
        
        # Record completion
        if request_id:
            quota_manager.record_request_completion(request_id, success=True, response_time_ms=1500)
        
        # Check status
        status = quota_manager.get_quota_status()
        print(f"Quota status: {status.requests_used_today}/{status.requests_remaining_today}")
        
        # Get forecast
        forecast = quota_manager.get_quota_forecast(hours_ahead=24)
        print(f"Forecast: {forecast}")
        
        # Test fallback decision
        should_fallback = quota_manager.should_use_fallback(RequestPriority.LOW)
        print(f"Should use fallback for LOW priority: {should_fallback}")
        
        # Export report
        report = quota_manager.export_usage_report()
        print(f"Usage report generated with {len(report)} sections")
    
    # Run test
    asyncio.run(test_quota_manager())