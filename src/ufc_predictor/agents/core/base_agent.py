"""
Base Agent Abstract Class
========================

Foundation for all UFC betting system agents providing:
- Standardized lifecycle management
- Health monitoring and diagnostics
- Event-driven communication
- Performance metrics tracking
- Error handling and recovery
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import psutil
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentPriority(Enum):
    """Agent execution priority levels"""
    CRITICAL = 1    # Model agents, risk management
    HIGH = 2        # Data processing, prediction generation
    MEDIUM = 3      # Monitoring, optimization
    LOW = 4         # Cleanup, maintenance


@dataclass
class AgentMetrics:
    """Agent performance and health metrics"""
    agent_id: str
    start_time: datetime
    last_activity: datetime
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_operation_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    @property
    def error_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.failed_operations / self.total_operations


@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    expires_at: Optional[datetime] = None


class AgentError(Exception):
    """Base exception for agent-related errors"""
    def __init__(self, message: str, agent_id: str = None, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code
        self.timestamp = datetime.now()


class BaseAgent(ABC):
    """
    Abstract base class for all UFC betting system agents
    
    Provides standardized:
    - Lifecycle management (initialize, start, stop, cleanup)
    - Health monitoring and metrics collection
    - Event-driven communication with other agents
    - Error handling and recovery mechanisms
    - Performance tracking and optimization
    """
    
    def __init__(self, agent_id: str, priority: AgentPriority = AgentPriority.MEDIUM):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for this agent
            priority: Execution priority level
        """
        self.agent_id = agent_id
        self.priority = priority
        self.state = AgentState.INITIALIZING
        self.metrics = AgentMetrics(
            agent_id=agent_id,
            start_time=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Set[str] = set()
        self.subscriptions: Set[str] = set()
        
        # Internal management
        self._stop_event = asyncio.Event()
        self._health_check_interval = 60  # seconds
        self._last_health_check = datetime.now()
        self._operation_times: List[float] = []
        self._max_operation_history = 100
        self._lock = asyncio.Lock()
        
        # Process monitoring
        self._process = psutil.Process()
        
        logger.info(f"Agent '{self.agent_id}' initialized with priority {self.priority.name}")
    
    # === Abstract Methods ===
    
    @abstractmethod
    async def _initialize_agent(self) -> bool:
        """
        Agent-specific initialization logic
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _start_agent(self) -> bool:
        """
        Agent-specific startup logic
        
        Returns:
            True if startup successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _stop_agent(self) -> bool:
        """
        Agent-specific shutdown logic
        
        Returns:
            True if shutdown successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """
        Process incoming message
        
        Args:
            message: Message to process
            
        Returns:
            Response payload if message requires response, None otherwise
        """
        pass
    
    @abstractmethod
    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Agent-specific health check
        
        Returns:
            Health status information
        """
        pass
    
    # === Lifecycle Management ===
    
    async def initialize(self) -> bool:
        """
        Initialize the agent
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state = AgentState.INITIALIZING
            logger.info(f"Initializing agent '{self.agent_id}'")
            
            # Agent-specific initialization
            success = await self._initialize_agent()
            
            if success:
                self.state = AgentState.READY
                logger.info(f"Agent '{self.agent_id}' initialized successfully")
            else:
                self.state = AgentState.ERROR
                logger.error(f"Agent '{self.agent_id}' initialization failed")
            
            return success
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics.error_count += 1
            logger.error(f"Agent '{self.agent_id}' initialization error: {e}")
            return False
    
    async def start(self) -> bool:
        """
        Start the agent
        
        Returns:
            True if successful, False otherwise
        """
        if self.state != AgentState.READY:
            logger.warning(f"Agent '{self.agent_id}' not ready for start (state: {self.state})")
            return False
        
        try:
            self.state = AgentState.RUNNING
            logger.info(f"Starting agent '{self.agent_id}'")
            
            # Start message processing
            self._message_task = asyncio.create_task(self._message_loop())
            self._health_task = asyncio.create_task(self._health_monitoring_loop())
            
            # Agent-specific startup
            success = await self._start_agent()
            
            if not success:
                self.state = AgentState.ERROR
                await self._cleanup_tasks()
                return False
            
            logger.info(f"Agent '{self.agent_id}' started successfully")
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics.error_count += 1
            logger.error(f"Agent '{self.agent_id}' start error: {e}")
            await self._cleanup_tasks()
            return False
    
    async def stop(self, timeout: float = 30.0) -> bool:
        """
        Stop the agent gracefully
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            True if successful, False otherwise
        """
        if self.state in [AgentState.STOPPED, AgentState.STOPPING]:
            return True
        
        try:
            self.state = AgentState.STOPPING
            logger.info(f"Stopping agent '{self.agent_id}'")
            
            # Signal stop
            self._stop_event.set()
            
            # Agent-specific shutdown
            agent_stop_success = await asyncio.wait_for(
                self._stop_agent(), timeout=timeout * 0.8
            )
            
            # Cleanup tasks
            await asyncio.wait_for(
                self._cleanup_tasks(), timeout=timeout * 0.2
            )
            
            self.state = AgentState.STOPPED
            logger.info(f"Agent '{self.agent_id}' stopped successfully")
            return agent_stop_success
            
        except asyncio.TimeoutError:
            self.state = AgentState.ERROR
            logger.error(f"Agent '{self.agent_id}' stop timeout")
            return False
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics.error_count += 1
            logger.error(f"Agent '{self.agent_id}' stop error: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause agent operations"""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            logger.info(f"Agent '{self.agent_id}' paused")
            return True
        return False
    
    async def resume(self) -> bool:
        """Resume agent operations"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            logger.info(f"Agent '{self.agent_id}' resumed")
            return True
        return False
    
    # === Communication ===
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register a message handler
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Agent '{self.agent_id}' registered handler for '{message_type}'")
    
    async def send_message(self, recipient_id: str, message_type: str, 
                          payload: Dict[str, Any], requires_response: bool = False,
                          timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Send message to another agent
        
        Args:
            recipient_id: Target agent ID
            message_type: Type of message
            payload: Message payload
            requires_response: Whether response is expected
            timeout: Response timeout
            
        Returns:
            Response payload if response required and received
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            requires_response=requires_response,
            expires_at=datetime.now() + timedelta(seconds=timeout) if requires_response else None
        )
        
        # This would integrate with agent registry/message bus
        # For now, just log the intent
        logger.debug(f"Agent '{self.agent_id}' sending '{message_type}' to '{recipient_id}'")
        
        # Implementation would depend on message bus architecture
        return None
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """
        Broadcast message to all subscribed agents
        
        Args:
            message_type: Type of message
            payload: Message payload
        """
        for subscriber_id in self.subscribers:
            await self.send_message(subscriber_id, message_type, payload)
    
    # === Health and Metrics ===
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status
        
        Returns:
            Health status information
        """
        # Update system metrics
        await self._update_system_metrics()
        
        # Get agent-specific health info
        agent_health = await self._perform_health_check()
        
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'priority': self.priority.name,
            'uptime_seconds': self.metrics.uptime_seconds,
            'success_rate': self.metrics.success_rate,
            'error_rate': self.metrics.error_rate,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'cpu_usage_percent': self.metrics.cpu_usage_percent,
            'last_activity': self.metrics.last_activity.isoformat(),
            'total_operations': self.metrics.total_operations,
            'avg_operation_time': self.metrics.avg_operation_time,
            'agent_specific': agent_health
        }
    
    async def get_metrics(self) -> AgentMetrics:
        """Get current metrics"""
        await self._update_system_metrics()
        return self.metrics
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str = None):
        """
        Context manager to track operation performance
        
        Args:
            operation_name: Name of operation for logging
        """
        start_time = time.time()
        operation_success = False
        
        try:
            self.metrics.total_operations += 1
            self.metrics.last_activity = datetime.now()
            yield
            operation_success = True
            self.metrics.successful_operations += 1
            
        except Exception as e:
            self.metrics.failed_operations += 1
            self.metrics.error_count += 1
            logger.error(f"Agent '{self.agent_id}' operation '{operation_name}' failed: {e}")
            raise
            
        finally:
            # Update timing metrics
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            # Keep only recent operation times
            if len(self._operation_times) > self._max_operation_history:
                self._operation_times = self._operation_times[-self._max_operation_history:]
            
            # Update average
            self.metrics.avg_operation_time = sum(self._operation_times) / len(self._operation_times)
            
            if operation_name:
                status = "completed" if operation_success else "failed"
                logger.debug(f"Agent '{self.agent_id}' operation '{operation_name}' {status} in {operation_time:.3f}s")
    
    # === Internal Methods ===
    
    async def _message_loop(self):
        """Main message processing loop"""
        logger.debug(f"Agent '{self.agent_id}' message loop started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                
                async with self.track_operation(f"process_message_{message.message_type}"):
                    response = await self._process_message(message)
                    
                    # Send response if required
                    if message.requires_response and response is not None:
                        await self.send_message(
                            message.sender_id,
                            f"{message.message_type}_response",
                            response
                        )
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, check stop event
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}' message processing error: {e}")
                self.metrics.error_count += 1
        
        logger.debug(f"Agent '{self.agent_id}' message loop stopped")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        logger.debug(f"Agent '{self.agent_id}' health monitoring started")
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if self._stop_event.is_set():
                    break
                
                await self._update_system_metrics()
                self._last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}' health monitoring error: {e}")
                self.metrics.error_count += 1
        
        logger.debug(f"Agent '{self.agent_id}' health monitoring stopped")
    
    async def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # Memory usage
            memory_info = self._process.memory_info()
            self.metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            self.metrics.cpu_usage_percent = self._process.cpu_percent()
            
        except Exception as e:
            logger.warning(f"Agent '{self.agent_id}' metrics update failed: {e}")
    
    async def _cleanup_tasks(self):
        """Cleanup async tasks"""
        tasks_to_cancel = []
        
        if hasattr(self, '_message_task'):
            tasks_to_cancel.append(self._message_task)
        
        if hasattr(self, '_health_task'):
            tasks_to_cancel.append(self._health_task)
        
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def __repr__(self) -> str:
        return f"BaseAgent(id='{self.agent_id}', state={self.state}, priority={self.priority})"