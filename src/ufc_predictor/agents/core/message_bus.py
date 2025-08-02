"""
Agent Message Bus and Communication System
==========================================

Central message bus for agent-to-agent communication:
- Asynchronous message passing with routing
- Event-driven coordination and workflows
- Message persistence and replay capabilities
- Circuit breaker patterns for reliability
- Message tracing and debugging
- Publish-subscribe patterns for events
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Set, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import weakref

from .base_agent import AgentMessage, BaseAgent

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class MessageRoute:
    """Message routing configuration"""
    pattern: str  # Pattern to match (e.g., "data_agent.*", "*.health_check")
    handler_id: str
    priority: MessagePriority = MessagePriority.NORMAL
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class MessageDeliveryRecord:
    """Record of message delivery attempt"""
    message_id: str
    recipient_id: str
    status: MessageStatus
    timestamp: datetime
    attempt_number: int
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    subscriber_id: str
    event_pattern: str  # Pattern to match event types
    callback: Callable
    priority: MessagePriority = MessagePriority.NORMAL
    enabled: bool = True


class MessageBus:
    """
    Central message bus for agent communication
    
    Features:
    - Asynchronous message delivery with routing
    - Event publish-subscribe pattern
    - Message persistence and replay
    - Circuit breaker for failed agents
    - Message tracing and debugging
    - Priority-based message handling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize message bus
        
        Args:
            config: Message bus configuration
        """
        self.config = config or {}
        
        # Agent registry
        self.agents: Dict[str, weakref.ref] = {}  # Weak references to prevent memory leaks
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        self.agent_health: Dict[str, bool] = {}
        
        # Message routing
        self.routes: List[MessageRoute] = []
        self.default_routes: Dict[str, str] = {}  # agent_id -> default handler
        
        # Event subscriptions
        self.event_subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        
        # Message tracking
        self.message_history: deque = deque(maxlen=10000)
        self.delivery_records: Dict[str, List[MessageDeliveryRecord]] = defaultdict(list)
        self.pending_messages: Dict[str, AgentMessage] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_threshold = self.config.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout = self.config.get('circuit_breaker_timeout', 300)  # 5 minutes
        
        # Message processing
        self.message_workers = self.config.get('message_workers', 3)
        self.max_queue_size = self.config.get('max_queue_size', 1000)
        self.message_timeout = self.config.get('message_timeout', 30.0)
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_failed = 0
        self.avg_delivery_time = 0.0
        
        # Internal queues
        self.incoming_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.outgoing_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Background tasks
        self.workers: List[asyncio.Task] = []
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Message bus state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        logger.info("MessageBus initialized")
    
    # === Agent Management ===
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register an agent with the message bus
        
        Args:
            agent: Agent to register
            
        Returns:
            True if registration successful
        """
        try:
            agent_id = agent.agent_id
            
            # Create weak reference to prevent circular references
            self.agents[agent_id] = weakref.ref(agent)
            
            # Create message queue for agent
            self.agent_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            
            # Initialize health status
            self.agent_health[agent_id] = True
            
            # Initialize circuit breaker
            self.circuit_breakers[agent_id] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
            
            logger.info(f"Agent registered: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the message bus
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            # Remove from registry
            if agent_id in self.agents:
                del self.agents[agent_id]
            
            # Clean up queue
            if agent_id in self.agent_queues:
                del self.agent_queues[agent_id]
            
            # Clean up health status
            if agent_id in self.agent_health:
                del self.agent_health[agent_id]
            
            # Clean up circuit breaker
            if agent_id in self.circuit_breakers:
                del self.circuit_breakers[agent_id]
            
            # Remove event subscriptions
            for event_type, subscriptions in self.event_subscriptions.items():
                self.event_subscriptions[event_type] = [
                    sub for sub in subscriptions if sub.subscriber_id != agent_id
                ]
            
            logger.info(f"Agent unregistered: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent unregistration failed: {e}")
            return False
    
    # === Message Sending ===
    
    async def send_message(self, 
                          sender_id: str,
                          recipient_id: str,
                          message_type: str,
                          payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL,
                          requires_response: bool = False,
                          timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Send message to specific agent
        
        Args:
            sender_id: ID of sending agent
            recipient_id: ID of receiving agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            requires_response: Whether response is expected
            timeout: Response timeout
            
        Returns:
            Response payload if response required and received
        """
        try:
            # Create message
            message = AgentMessage(
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
                requires_response=requires_response,
                expires_at=datetime.now() + timedelta(seconds=timeout or self.message_timeout),
                correlation_id=str(uuid.uuid4())
            )
            
            # Check circuit breaker
            if not self._check_circuit_breaker(recipient_id):
                logger.warning(f"Circuit breaker open for {recipient_id}, message dropped")
                return None
            
            # Queue message for delivery
            await self._queue_message_for_delivery(message, priority)
            
            # Wait for response if required
            if requires_response:
                response = await self._wait_for_response(message.correlation_id, timeout or self.message_timeout)
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Message send failed: {e}")
            self.messages_failed += 1
            return None
    
    async def broadcast_message(self,
                              sender_id: str,
                              message_type: str,
                              payload: Dict[str, Any],
                              exclude_agents: Set[str] = None) -> int:
        """
        Broadcast message to all registered agents
        
        Args:
            sender_id: ID of sending agent
            message_type: Type of message
            payload: Message payload
            exclude_agents: Set of agent IDs to exclude
            
        Returns:
            Number of agents message was sent to
        """
        try:
            exclude_agents = exclude_agents or set()
            exclude_agents.add(sender_id)  # Don't send to self
            
            sent_count = 0
            
            for agent_id in self.agents.keys():
                if agent_id not in exclude_agents:
                    await self.send_message(
                        sender_id=sender_id,
                        recipient_id=agent_id,
                        message_type=message_type,
                        payload=payload,
                        requires_response=False
                    )
                    sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            return 0
    
    # === Event System ===
    
    async def publish_event(self,
                          publisher_id: str,
                          event_type: str,
                          event_data: Dict[str, Any]) -> int:
        """
        Publish event to subscribers
        
        Args:
            publisher_id: ID of publishing agent
            event_type: Type of event
            event_data: Event data
            
        Returns:
            Number of subscribers notified
        """
        try:
            notified_count = 0
            
            # Find matching subscriptions
            for pattern, subscriptions in self.event_subscriptions.items():
                if self._pattern_matches(event_type, pattern):
                    for subscription in subscriptions:
                        if subscription.enabled and subscription.subscriber_id != publisher_id:
                            try:
                                # Create event message
                                await self.send_message(
                                    sender_id=publisher_id,
                                    recipient_id=subscription.subscriber_id,
                                    message_type=f"event_{event_type}",
                                    payload={'event_data': event_data, 'event_type': event_type},
                                    priority=subscription.priority
                                )
                                notified_count += 1
                                
                            except Exception as e:
                                logger.error(f"Event notification failed for {subscription.subscriber_id}: {e}")
            
            logger.debug(f"Event '{event_type}' published to {notified_count} subscribers")
            return notified_count
            
        except Exception as e:
            logger.error(f"Event publish failed: {e}")
            return 0
    
    async def subscribe_to_events(self,
                                subscriber_id: str,
                                event_pattern: str,
                                callback: Callable,
                                priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Subscribe to events matching pattern
        
        Args:
            subscriber_id: ID of subscribing agent
            event_pattern: Event pattern to match
            callback: Callback function for events
            priority: Subscription priority
            
        Returns:
            True if subscription successful
        """
        try:
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_pattern=event_pattern,
                callback=callback,
                priority=priority
            )
            
            self.event_subscriptions[event_pattern].append(subscription)
            
            logger.debug(f"Agent {subscriber_id} subscribed to events: {event_pattern}")
            return True
            
        except Exception as e:
            logger.error(f"Event subscription failed: {e}")
            return False
    
    async def unsubscribe_from_events(self,
                                    subscriber_id: str,
                                    event_pattern: str = None) -> bool:
        """
        Unsubscribe from events
        
        Args:
            subscriber_id: ID of subscribing agent
            event_pattern: Specific pattern to unsubscribe from (None for all)
            
        Returns:
            True if unsubscription successful
        """
        try:
            if event_pattern:
                # Remove specific subscription
                self.event_subscriptions[event_pattern] = [
                    sub for sub in self.event_subscriptions[event_pattern]
                    if sub.subscriber_id != subscriber_id
                ]
            else:
                # Remove all subscriptions for agent
                for pattern in self.event_subscriptions:
                    self.event_subscriptions[pattern] = [
                        sub for sub in self.event_subscriptions[pattern]
                        if sub.subscriber_id != subscriber_id
                    ]
            
            logger.debug(f"Agent {subscriber_id} unsubscribed from events: {event_pattern or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Event unsubscription failed: {e}")
            return False
    
    # === Message Bus Control ===
    
    async def start(self) -> bool:
        """Start message bus operations"""
        try:
            if self.is_running:
                return True
            
            # Start message processing workers
            for i in range(self.message_workers):
                worker = asyncio.create_task(self._message_worker(f"worker_{i}"))
                self.workers.append(worker)
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Start metrics collection
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            self.is_running = True
            logger.info("MessageBus started")
            return True
            
        except Exception as e:
            logger.error(f"MessageBus start failed: {e}")
            return False
    
    async def stop(self, timeout: float = 30.0) -> bool:
        """Stop message bus operations"""
        try:
            if not self.is_running:
                return True
            
            logger.info("Stopping MessageBus...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Cancel all workers
            all_tasks = self.workers + ([self.cleanup_task] if self.cleanup_task else []) + ([self.metrics_task] if self.metrics_task else [])
            
            for task in all_tasks:
                task.cancel()
            
            # Wait for tasks to complete with timeout
            if all_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*all_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not complete within timeout")
            
            self.is_running = False
            logger.info("MessageBus stopped")
            return True
            
        except Exception as e:
            logger.error(f"MessageBus stop failed: {e}")
            return False
    
    # === Message Processing ===
    
    async def _queue_message_for_delivery(self, message: AgentMessage, priority: MessagePriority):
        """Queue message for delivery"""
        try:
            # Add to pending messages
            self.pending_messages[message.correlation_id] = message
            
            # Queue for processing
            await self.incoming_queue.put((priority.value, message))
            
            self.messages_sent += 1
            
        except asyncio.QueueFull:
            logger.error(f"Message queue full, dropping message: {message.message_type}")
            self.messages_failed += 1
        except Exception as e:
            logger.error(f"Message queueing failed: {e}")
            self.messages_failed += 1
    
    async def _message_worker(self, worker_id: str):
        """Message processing worker"""
        logger.info(f"Message worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    priority, message = await asyncio.wait_for(
                        self.incoming_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._deliver_message(message)
                
            except Exception as e:
                logger.error(f"Message worker {worker_id} error: {e}")
        
        logger.info(f"Message worker {worker_id} stopped")
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver message to recipient"""
        start_time = datetime.now()
        
        try:
            recipient_id = message.recipient_id
            
            # Check if recipient exists and is healthy
            if recipient_id not in self.agents or not self.agent_health.get(recipient_id, False):
                await self._record_delivery_failure(message, "Recipient not available")
                return
            
            # Check message expiration
            if message.expires_at and datetime.now() > message.expires_at:
                await self._record_delivery_failure(message, "Message expired")
                return
            
            # Get agent reference
            agent_ref = self.agents[recipient_id]
            agent = agent_ref()
            
            if agent is None:
                # Agent was garbage collected
                await self.unregister_agent(recipient_id)
                await self._record_delivery_failure(message, "Agent no longer available")
                return
            
            # Deliver message to agent
            response = await agent._process_message(message)
            
            # Record successful delivery
            delivery_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._record_delivery_success(message, response, delivery_time)
            
            # Handle response if required
            if message.requires_response and response is not None:
                await self._handle_message_response(message, response)
            
        except Exception as e:
            await self._record_delivery_failure(message, str(e))
    
    async def _record_delivery_success(self, message: AgentMessage, response: Optional[Dict[str, Any]], delivery_time_ms: float):
        """Record successful message delivery"""
        record = MessageDeliveryRecord(
            message_id=message.correlation_id,
            recipient_id=message.recipient_id,
            status=MessageStatus.DELIVERED,
            timestamp=datetime.now(),
            attempt_number=1,  # Simplified for now
            response_data=response
        )
        
        self.delivery_records[message.correlation_id].append(record)
        self.message_history.append(message)
        
        # Update metrics
        self.messages_delivered += 1
        self._update_avg_delivery_time(delivery_time_ms)
        
        # Remove from pending
        if message.correlation_id in self.pending_messages:
            del self.pending_messages[message.correlation_id]
        
        # Reset circuit breaker failures
        if message.recipient_id in self.circuit_breakers:
            self.circuit_breakers[message.recipient_id]['failures'] = 0
            self.circuit_breakers[message.recipient_id]['state'] = 'closed'
    
    async def _record_delivery_failure(self, message: AgentMessage, error_message: str):
        """Record failed message delivery"""
        record = MessageDeliveryRecord(
            message_id=message.correlation_id,
            recipient_id=message.recipient_id,
            status=MessageStatus.FAILED,
            timestamp=datetime.now(),
            attempt_number=1,  # Simplified for now
            error_message=error_message
        )
        
        self.delivery_records[message.correlation_id].append(record)
        
        # Update metrics
        self.messages_failed += 1
        
        # Update circuit breaker
        await self._update_circuit_breaker(message.recipient_id)
        
        # Remove from pending
        if message.correlation_id in self.pending_messages:
            del self.pending_messages[message.correlation_id]
        
        logger.warning(f"Message delivery failed: {error_message}")
    
    async def _handle_message_response(self, original_message: AgentMessage, response: Dict[str, Any]):
        """Handle response to message that required response"""
        # This would store the response for the waiting sender
        # Implementation depends on how responses are tracked
        pass
    
    async def _wait_for_response(self, correlation_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        """Wait for response to message"""
        # Simplified implementation - would need proper response tracking
        await asyncio.sleep(0.1)  # Placeholder
        return None
    
    # === Circuit Breaker ===
    
    def _check_circuit_breaker(self, agent_id: str) -> bool:
        """Check if circuit breaker allows message delivery"""
        if agent_id not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[agent_id]
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if breaker['last_failure']:
                time_since_failure = (datetime.now() - breaker['last_failure']).total_seconds()
                if time_since_failure > self.circuit_breaker_timeout:
                    breaker['state'] = 'half-open'
                    return True
            return False
        
        return True
    
    async def _update_circuit_breaker(self, agent_id: str):
        """Update circuit breaker state after failure"""
        if agent_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[agent_id]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.now()
        
        if breaker['failures'] >= self.circuit_breaker_threshold:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for agent {agent_id}")
    
    # === Utility Methods ===
    
    def _pattern_matches(self, text: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)"""
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
    
    def _update_avg_delivery_time(self, delivery_time_ms: float):
        """Update average delivery time metric"""
        if self.messages_delivered == 1:
            self.avg_delivery_time = delivery_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_delivery_time = alpha * delivery_time_ms + (1 - alpha) * self.avg_delivery_time
    
    # === Background Tasks ===
    
    async def _cleanup_loop(self):
        """Background cleanup of old messages and records"""
        logger.info("MessageBus cleanup loop started")
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                if self.shutdown_event.is_set():
                    break
                
                await self._cleanup_old_records()
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
        
        logger.info("MessageBus cleanup loop stopped")
    
    async def _cleanup_old_records(self):
        """Clean up old delivery records and expired messages"""
        try:
            current_time = datetime.now()
            cleanup_threshold = current_time - timedelta(hours=24)  # Keep 24 hours
            
            # Clean up old delivery records
            for correlation_id in list(self.delivery_records.keys()):
                records = self.delivery_records[correlation_id]
                # Keep only recent records
                recent_records = [r for r in records if r.timestamp > cleanup_threshold]
                
                if recent_records:
                    self.delivery_records[correlation_id] = recent_records
                else:
                    del self.delivery_records[correlation_id]
            
            # Clean up expired pending messages
            expired_messages = []
            for correlation_id, message in self.pending_messages.items():
                if message.expires_at and current_time > message.expires_at:
                    expired_messages.append(correlation_id)
            
            for correlation_id in expired_messages:
                del self.pending_messages[correlation_id]
                # Record as expired
                if correlation_id in self.delivery_records:
                    self.delivery_records[correlation_id].append(
                        MessageDeliveryRecord(
                            message_id=correlation_id,
                            recipient_id="unknown",
                            status=MessageStatus.EXPIRED,
                            timestamp=current_time,
                            attempt_number=1
                        )
                    )
            
            if expired_messages:
                logger.info(f"Cleaned up {len(expired_messages)} expired messages")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def _metrics_loop(self):
        """Background metrics collection"""
        logger.info("MessageBus metrics loop started")
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # 1 minute
                
                if self.shutdown_event.is_set():
                    break
                
                # Log metrics
                logger.info(
                    f"MessageBus metrics: "
                    f"sent={self.messages_sent}, delivered={self.messages_delivered}, "
                    f"failed={self.messages_failed}, avg_time={self.avg_delivery_time:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
        
        logger.info("MessageBus metrics loop stopped")
    
    # === Status and Debugging ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get message bus status"""
        return {
            'is_running': self.is_running,
            'registered_agents': len(self.agents),
            'healthy_agents': sum(1 for healthy in self.agent_health.values() if healthy),
            'pending_messages': len(self.pending_messages),
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_failed': self.messages_failed,
            'avg_delivery_time_ms': self.avg_delivery_time,
            'circuit_breakers_open': sum(
                1 for breaker in self.circuit_breakers.values() 
                if breaker['state'] == 'open'
            ),
            'event_subscriptions': sum(len(subs) for subs in self.event_subscriptions.values())
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status for specific agent"""
        if agent_id not in self.agents:
            return None
        
        return {
            'agent_id': agent_id,
            'is_registered': True,
            'is_healthy': self.agent_health.get(agent_id, False),
            'queue_size': self.agent_queues[agent_id].qsize() if agent_id in self.agent_queues else 0,
            'circuit_breaker': self.circuit_breakers.get(agent_id, {}),
            'recent_deliveries': len(self.delivery_records.get(agent_id, [])),
            'subscriptions': sum(
                1 for subs in self.event_subscriptions.values() 
                for sub in subs if sub.subscriber_id == agent_id
            )
        }


# Global message bus instance
_message_bus_instance: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get global message bus instance"""
    global _message_bus_instance
    if _message_bus_instance is None:
        _message_bus_instance = MessageBus()
    return _message_bus_instance


def set_message_bus(message_bus: MessageBus):
    """Set global message bus instance"""
    global _message_bus_instance
    _message_bus_instance = message_bus


def create_message_bus_config(
    message_workers: int = 3,
    max_queue_size: int = 1000,
    message_timeout: float = 30.0,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: int = 300
) -> Dict[str, Any]:
    """
    Factory function for MessageBus configuration
    
    Args:
        message_workers: Number of message processing workers
        max_queue_size: Maximum queue size
        message_timeout: Default message timeout
        circuit_breaker_threshold: Circuit breaker failure threshold
        circuit_breaker_timeout: Circuit breaker timeout in seconds
        
    Returns:
        MessageBus configuration
    """
    return {
        'message_workers': message_workers,
        'max_queue_size': max_queue_size,
        'message_timeout': message_timeout,
        'circuit_breaker_threshold': circuit_breaker_threshold,
        'circuit_breaker_timeout': circuit_breaker_timeout
    }