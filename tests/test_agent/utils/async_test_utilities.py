"""
Async Test Utilities

Specialized utilities for testing asynchronous components of the UFC Betting Agent
and Enhanced ML Pipeline.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor
import pytest


class AsyncTestRunner:
    """Advanced async test runner with timeout and resource management"""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_with_timeout(self, coro: Awaitable, timeout: Optional[float] = None) -> Any:
        """Run coroutine with timeout and proper cleanup"""
        timeout = timeout or self.default_timeout
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise AssertionError(
                f"Async operation timed out after {timeout} seconds. "
                f"Consider increasing timeout or optimizing the operation."
            )
        except Exception as e:
            raise AssertionError(f"Async operation failed: {str(e)}") from e
    
    async def run_concurrent_operations(self, operations: List[Awaitable], 
                                      max_concurrent: int = 5) -> List[Any]:
        """Run multiple async operations concurrently with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_operation(operation):
            async with semaphore:
                return await operation
        
        limited_ops = [limited_operation(op) for op in operations]
        return await asyncio.gather(*limited_ops)
    
    async def measure_async_performance(self, coro: Awaitable) -> Dict[str, Any]:
        """Measure performance metrics for async operation"""
        start_time = time.perf_counter()
        
        try:
            result = await coro
            end_time = time.perf_counter()
            
            return {
                'result': result,
                'execution_time': end_time - start_time,
                'success': True,
                'error': None
            }
        except Exception as e:
            end_time = time.perf_counter()
            
            return {
                'result': None,
                'execution_time': end_time - start_time,
                'success': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


class AsyncMockFactory:
    """Factory for creating complex async mocks"""
    
    @staticmethod
    def create_odds_service_mock(
        fetch_delay: float = 0.1,
        success_rate: float = 1.0,
        quota_remaining: int = 10
    ) -> AsyncMock:
        """Create mock hybrid odds service with configurable behavior"""
        
        async def mock_fetch_event_odds(event_name: str, target_fights: List[str], **kwargs):
            # Simulate fetch delay
            await asyncio.sleep(fetch_delay)
            
            # Simulate occasional failures
            if success_rate < 1.0:
                import random
                if random.random() > success_rate:
                    raise Exception(f"Mock fetch failure for {event_name}")
            
            # Create mock result
            reconciled_data = {}
            for fight in target_fights:
                fighters = fight.split(' vs ')
                if len(fighters) == 2:
                    reconciled_data[fight] = {
                        'fighter_a': fighters[0],
                        'fighter_b': fighters[1],
                        'fighter_a_decimal_odds': 1.8,
                        'fighter_b_decimal_odds': 2.2,
                        'data_sources': ['tab_australia']
                    }
            
            mock_result = Mock()
            mock_result.reconciled_data = reconciled_data
            mock_result.api_requests_used = 1 if quota_remaining > 0 else 0
            mock_result.confidence_score = 0.85
            mock_result.fallback_activated = quota_remaining <= 0
            
            return mock_result
        
        async def mock_get_quota_status():
            return {
                'quota_status': {
                    'requests_remaining_today': quota_remaining,
                    'budget_remaining': 45.0
                },
                'hybrid_service_metrics': {'api_efficiency': 0.8}
            }
        
        async def mock_health_check():
            return {
                'overall_status': 'healthy',
                'components': {'quota_manager': 'healthy'}
            }
        
        mock_service = AsyncMock()
        mock_service.fetch_event_odds = mock_fetch_event_odds
        mock_service.get_quota_status = mock_get_quota_status
        mock_service.health_check = mock_health_check
        
        return mock_service
    
    @staticmethod
    def create_agent_mock(
        analysis_delay: float = 0.5,
        initialization_success: bool = True
    ) -> Mock:
        """Create mock UFC Betting Agent with async capabilities"""
        
        async def mock_analyze_event(event_name: str, target_fights: List[str]):
            # Simulate analysis delay
            await asyncio.sleep(analysis_delay)
            
            return {
                'event_name': event_name,
                'status': 'completed',
                'odds_result': {'total_fights': len(target_fights), 'status': 'success'},
                'predictions_analysis': {'successful_predictions': len(target_fights)},
                'betting_recommendations': {'total_bets': len(target_fights), 'total_stake': 100.0}
            }
        
        async def mock_health_check():
            return {
                'overall_status': 'healthy',
                'checks': {'betting_system': {'status': 'healthy'}}
            }
        
        async def mock_get_system_status():
            return {
                'betting_system_initialized': initialization_success,
                'bankroll': 1000.0
            }
        
        mock_agent = Mock()
        mock_agent.analyze_event = mock_analyze_event
        mock_agent.health_check = mock_health_check
        mock_agent.get_system_status = mock_get_system_status
        mock_agent.initialize_betting_system.return_value = initialization_success
        
        return mock_agent


class AsyncTestPatterns:
    """Common async testing patterns"""
    
    @staticmethod
    async def test_concurrent_requests(
        async_func: Callable,
        request_params: List[Dict],
        max_concurrent: int = 5,
        expected_success_rate: float = 1.0
    ):
        """Test concurrent async requests with success rate validation"""
        
        tasks = [async_func(**params) for params in request_params]
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(task):
            async with semaphore:
                try:
                    result = await task
                    return {'success': True, 'result': result, 'error': None}
                except Exception as e:
                    return {'success': False, 'result': None, 'error': str(e)}
        
        limited_tasks = [limited_request(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks)
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = successful_requests / len(results)
        
        assert success_rate >= expected_success_rate, \
            f"Success rate {success_rate:.2f} below expected {expected_success_rate:.2f}"
        
        return results
    
    @staticmethod
    async def test_timeout_behavior(
        async_func: Callable,
        func_args: tuple = (),
        func_kwargs: Dict = None,
        timeout: float = 1.0
    ):
        """Test that async function respects timeout"""
        func_kwargs = func_kwargs or {}
        
        start_time = time.perf_counter()
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(async_func(*func_args, **func_kwargs), timeout=timeout)
        
        elapsed_time = time.perf_counter() - start_time
        
        # Should have timed out close to the specified timeout
        assert timeout <= elapsed_time <= timeout + 0.5, \
            f"Timeout behavior incorrect. Expected ~{timeout}s, got {elapsed_time:.2f}s"
    
    @staticmethod
    async def test_error_propagation(
        async_func: Callable,
        error_conditions: List[Dict],
        expected_exceptions: List[type]
    ):
        """Test that async function properly propagates different types of errors"""
        
        for error_condition, expected_exception in zip(error_conditions, expected_exceptions):
            with pytest.raises(expected_exception):
                await async_func(**error_condition)
    
    @staticmethod
    async def test_resource_cleanup(
        setup_func: Callable,
        async_operation: Callable,
        cleanup_verification: Callable
    ):
        """Test that async operations properly clean up resources"""
        
        # Setup resources
        resources = await setup_func()
        
        try:
            # Perform async operation
            await async_operation(resources)
        finally:
            # Verify cleanup (even if operation failed)
            cleanup_verification(resources)


class AsyncPerformanceTester:
    """Performance testing utilities for async operations"""
    
    @staticmethod
    async def benchmark_async_operation(
        async_func: Callable,
        iterations: int = 10,
        warmup_iterations: int = 2
    ) -> Dict[str, float]:
        """Benchmark async operation with warmup and statistics"""
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                await async_func()
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark runs
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                await async_func()
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                # Record failed attempts
                end_time = time.perf_counter()
                execution_times.append(float('inf'))  # Mark as failed
        
        # Calculate statistics
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if not valid_times:
            return {
                'mean': float('inf'),
                'median': float('inf'),
                'min': float('inf'),
                'max': float('inf'),
                'std': float('inf'),
                'success_rate': 0.0
            }
        
        return {
            'mean': sum(valid_times) / len(valid_times),
            'median': sorted(valid_times)[len(valid_times) // 2],
            'min': min(valid_times),
            'max': max(valid_times),
            'std': (sum((t - sum(valid_times)/len(valid_times))**2 for t in valid_times) / len(valid_times))**0.5,
            'success_rate': len(valid_times) / len(execution_times)
        }
    
    @staticmethod
    async def load_test_async_service(
        service_func: Callable,
        concurrent_users: int = 10,
        requests_per_user: int = 5,
        ramp_up_time: float = 1.0
    ) -> Dict[str, Any]:
        """Load test async service with gradual ramp-up"""
        
        async def user_simulation(user_id: int):
            """Simulate a single user's requests"""
            # Stagger user start times
            start_delay = (user_id / concurrent_users) * ramp_up_time
            await asyncio.sleep(start_delay)
            
            user_results = []
            
            for request_id in range(requests_per_user):
                start_time = time.perf_counter()
                try:
                    result = await service_func()
                    end_time = time.perf_counter()
                    
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'success': True,
                        'response_time': end_time - start_time,
                        'error': None
                    })
                except Exception as e:
                    end_time = time.perf_counter()
                    
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'success': False,
                        'response_time': end_time - start_time,
                        'error': str(e)
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            return user_results
        
        # Run all user simulations concurrently
        user_tasks = [user_simulation(user_id) for user_id in range(concurrent_users)]
        all_user_results = await asyncio.gather(*user_tasks)
        
        # Flatten results
        all_results = []
        for user_results in all_user_results:
            all_results.extend(user_results)
        
        # Calculate metrics
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(all_results),
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            'p99_response_time': sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0,
            'errors': [r['error'] for r in failed_requests if r['error']]
        }


# Pytest fixtures for async testing
@pytest.fixture
def async_test_runner():
    """Provide async test runner"""
    runner = AsyncTestRunner()
    yield runner
    runner.cleanup()


@pytest.fixture
def async_mock_factory():
    """Provide async mock factory"""
    return AsyncMockFactory()


@pytest.fixture
def async_performance_tester():
    """Provide async performance tester"""
    return AsyncPerformanceTester()