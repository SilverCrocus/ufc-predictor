"""
Agent Registry and Dependency Injection System
==============================================

Centralized registry for agent discovery and dependency management:
- Service discovery and registration
- Dependency injection with lifecycle management
- Configuration management and hot-reloading
- Health monitoring and circuit breakers
- Plugin architecture for extensibility
- Factory patterns for agent creation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Type, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import inspect
import weakref
from pathlib import Path
import json
import yaml

from .base_agent import BaseAgent, AgentState, AgentPriority
from .message_bus import MessageBus

logger = logging.getLogger(__name__)


class DependencyScope(Enum):
    """Dependency injection scopes"""
    SINGLETON = "singleton"    # One instance for entire application
    TRANSIENT = "transient"    # New instance every time
    SCOPED = "scoped"         # One instance per scope/request


class ServiceLifecycle(Enum):
    """Service lifecycle states"""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceDescriptor:
    """Service registration descriptor"""
    service_id: str
    service_type: Type
    factory: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    scope: DependencyScope = DependencyScope.SINGLETON
    priority: AgentPriority = AgentPriority.MEDIUM
    config: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    health_check_interval: int = 60  # seconds
    auto_start: bool = True
    enabled: bool = True


@dataclass
class ServiceInstance:
    """Service instance wrapper"""
    descriptor: ServiceDescriptor
    instance: Any
    lifecycle_state: ServiceLifecycle
    created_at: datetime
    last_health_check: Optional[datetime] = None
    health_status: bool = True
    start_count: int = 0
    failure_count: int = 0
    dependencies_resolved: bool = False


class DependencyInjectionError(Exception):
    """Dependency injection related errors"""
    pass


class CircularDependencyError(DependencyInjectionError):
    """Circular dependency detected"""
    pass


class ServiceNotFoundError(DependencyInjectionError):
    """Service not found in registry"""
    pass


class AgentRegistry:
    """
    Central agent registry with dependency injection
    
    Features:
    - Service discovery and registration
    - Dependency injection with circular detection
    - Lifecycle management for all services
    - Configuration hot-reloading
    - Health monitoring and recovery
    - Plugin architecture support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent registry
        
        Args:
            config: Registry configuration
        """
        self.config = config or {}
        
        # Service registry
        self.services: Dict[str, ServiceDescriptor] = {}
        self.instances: Dict[str, ServiceInstance] = {}
        self.singletons: Dict[str, Any] = {}
        
        # Dependency management
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        self.resolution_order: List[str] = []
        
        # Lifecycle management
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.lifecycle_callbacks: Dict[str, List[Callable]] = {
            'before_start': [],
            'after_start': [],
            'before_stop': [],
            'after_stop': []
        }
        
        # Health monitoring
        self.health_check_enabled = self.config.get('health_check_enabled', True)
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        
        # Plugin system
        self.plugins: Dict[str, Any] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
        
        # Configuration management
        self.config_sources: List[str] = self.config.get('config_sources', [])
        self.hot_reload_enabled = self.config.get('hot_reload_enabled', False)
        self.last_config_reload = datetime.now()
        
        # Registry state
        self.is_started = False
        self.is_starting = False
        self.is_stopping = False
        self.startup_time: Optional[datetime] = None
        
        # Background tasks
        self.health_monitoring_task: Optional[asyncio.Task] = None
        self.config_reload_task: Optional[asyncio.Task] = None
        
        # Message bus integration
        self.message_bus: Optional[MessageBus] = None
        
        logger.info("AgentRegistry initialized")
    
    # === Service Registration ===
    
    def register_service(self, 
                        service_id: str,
                        service_type: Type,
                        factory: Optional[Callable] = None,
                        dependencies: List[str] = None,
                        scope: DependencyScope = DependencyScope.SINGLETON,
                        config: Dict[str, Any] = None,
                        **kwargs) -> bool:
        """
        Register a service with the registry
        
        Args:
            service_id: Unique service identifier
            service_type: Service class type
            factory: Optional factory function
            dependencies: List of dependency service IDs
            scope: Dependency injection scope
            config: Service configuration
            **kwargs: Additional service descriptor options
            
        Returns:
            True if registration successful
        """
        try:
            if service_id in self.services:
                logger.warning(f"Service {service_id} already registered, overwriting")
            
            descriptor = ServiceDescriptor(
                service_id=service_id,
                service_type=service_type,
                factory=factory,
                dependencies=dependencies or [],
                scope=scope,
                config=config or {},
                **kwargs
            )
            
            # Validate dependencies
            self._validate_dependencies(service_id, descriptor.dependencies)
            
            # Register service
            self.services[service_id] = descriptor
            
            # Update dependency graph
            self._update_dependency_graph(service_id, descriptor.dependencies)
            
            # Recalculate resolution order
            self._calculate_resolution_order()
            
            logger.info(f"Service registered: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service from the registry
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if unregistration successful
        """
        try:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not registered")
                return True
            
            # Stop service if running
            if service_id in self.instances:
                asyncio.create_task(self._stop_service(service_id))
            
            # Remove from registry
            del self.services[service_id]
            
            # Clean up dependency graph
            self._remove_from_dependency_graph(service_id)
            
            # Recalculate resolution order
            self._calculate_resolution_order()
            
            logger.info(f"Service unregistered: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Service unregistration failed: {e}")
            return False
    
    # === Service Discovery ===
    
    def get_service(self, service_id: str) -> Optional[Any]:
        """
        Get service instance by ID
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service instance or None
        """
        try:
            if service_id not in self.services:
                raise ServiceNotFoundError(f"Service {service_id} not registered")
            
            descriptor = self.services[service_id]
            
            # Handle different scopes
            if descriptor.scope == DependencyScope.SINGLETON:
                return self._get_singleton_instance(service_id)
            elif descriptor.scope == DependencyScope.TRANSIENT:
                return self._create_transient_instance(service_id)
            elif descriptor.scope == DependencyScope.SCOPED:
                # For now, treat scoped as singleton
                return self._get_singleton_instance(service_id)
            
        except Exception as e:
            logger.error(f"Service retrieval failed for {service_id}: {e}")
            return None
    
    def get_services_by_type(self, service_type: Type) -> List[Any]:
        """
        Get all services of a specific type
        
        Args:
            service_type: Service type to search for
            
        Returns:
            List of service instances
        """
        services = []
        
        for service_id, descriptor in self.services.items():
            if issubclass(descriptor.service_type, service_type):
                instance = self.get_service(service_id)
                if instance:
                    services.append(instance)
        
        return services
    
    def get_services_by_tag(self, tag: str) -> List[Any]:
        """
        Get all services with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of service instances
        """
        services = []
        
        for service_id, descriptor in self.services.items():
            if tag in descriptor.tags:
                instance = self.get_service(service_id)
                if instance:
                    services.append(instance)
        
        return services
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered services
        
        Returns:
            Dictionary of service information
        """
        service_list = {}
        
        for service_id, descriptor in self.services.items():
            instance_info = None
            if service_id in self.instances:
                instance = self.instances[service_id]
                instance_info = {
                    'lifecycle_state': instance.lifecycle_state.value,
                    'health_status': instance.health_status,
                    'created_at': instance.created_at.isoformat(),
                    'start_count': instance.start_count,
                    'failure_count': instance.failure_count
                }
            
            service_list[service_id] = {
                'service_type': descriptor.service_type.__name__,
                'scope': descriptor.scope.value,
                'priority': descriptor.priority.name,
                'dependencies': descriptor.dependencies,
                'tags': list(descriptor.tags),
                'enabled': descriptor.enabled,
                'auto_start': descriptor.auto_start,
                'instance': instance_info
            }
        
        return service_list
    
    # === Dependency Injection ===
    
    def _get_singleton_instance(self, service_id: str) -> Any:
        """Get or create singleton instance"""
        if service_id in self.singletons:
            return self.singletons[service_id]
        
        instance = self._create_service_instance(service_id)
        if instance:
            self.singletons[service_id] = instance
        
        return instance
    
    def _create_transient_instance(self, service_id: str) -> Any:
        """Create new transient instance"""
        return self._create_service_instance(service_id)
    
    def _create_service_instance(self, service_id: str) -> Any:
        """Create service instance with dependency injection"""
        try:
            descriptor = self.services[service_id]
            
            # Resolve dependencies first
            dependencies = self._resolve_dependencies(service_id)
            
            # Create instance
            if descriptor.factory:
                # Use factory function
                instance = descriptor.factory(dependencies, descriptor.config)
            else:
                # Use constructor
                if dependencies:
                    instance = descriptor.service_type(descriptor.config, **dependencies)
                else:
                    instance = descriptor.service_type(descriptor.config)
            
            # Store instance information
            service_instance = ServiceInstance(
                descriptor=descriptor,
                instance=instance,
                lifecycle_state=ServiceLifecycle.REGISTERED,
                created_at=datetime.now(),
                dependencies_resolved=True
            )
            
            self.instances[service_id] = service_instance
            
            logger.debug(f"Created instance for service: {service_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Instance creation failed for {service_id}: {e}")
            return None
    
    def _resolve_dependencies(self, service_id: str) -> Dict[str, Any]:
        """Resolve dependencies for a service"""
        descriptor = self.services[service_id]
        resolved_dependencies = {}
        
        for dep_id in descriptor.dependencies:
            if dep_id not in self.services:
                raise ServiceNotFoundError(f"Dependency {dep_id} not found for service {service_id}")
            
            dep_instance = self.get_service(dep_id)
            if dep_instance is None:
                raise DependencyInjectionError(f"Failed to resolve dependency {dep_id} for service {service_id}")
            
            resolved_dependencies[dep_id] = dep_instance
        
        return resolved_dependencies
    
    # === Lifecycle Management ===
    
    async def start_all_services(self) -> bool:
        """Start all registered services in dependency order"""
        if self.is_started or self.is_starting:
            return True
        
        try:
            self.is_starting = True
            logger.info("Starting all services...")
            
            # Execute before_start callbacks
            await self._execute_callbacks('before_start')
            
            # Start services in dependency order
            for service_id in self.resolution_order:
                descriptor = self.services[service_id]
                if descriptor.enabled and descriptor.auto_start:
                    success = await self._start_service(service_id)
                    if not success:
                        logger.error(f"Failed to start service {service_id}")
                        # Continue with other services
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Execute after_start callbacks
            await self._execute_callbacks('after_start')
            
            self.is_started = True
            self.is_starting = False
            self.startup_time = datetime.now()
            
            logger.info("All services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service startup failed: {e}")
            self.is_starting = False
            return False
    
    async def stop_all_services(self, timeout: float = 30.0) -> bool:
        """Stop all services in reverse dependency order"""
        if not self.is_started or self.is_stopping:
            return True
        
        try:
            self.is_stopping = True
            logger.info("Stopping all services...")
            
            # Execute before_stop callbacks
            await self._execute_callbacks('before_stop')
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop services in reverse order
            for service_id in reversed(self.resolution_order):
                if service_id in self.instances:
                    await self._stop_service(service_id, timeout / len(self.instances))
            
            # Execute after_stop callbacks
            await self._execute_callbacks('after_stop')
            
            self.is_started = False
            self.is_stopping = False
            
            logger.info("All services stopped")
            return True
            
        except Exception as e:
            logger.error(f"Service shutdown failed: {e}")
            self.is_stopping = False
            return False
    
    async def _start_service(self, service_id: str) -> bool:
        """Start individual service"""
        try:
            if service_id not in self.services:
                return False
            
            # Get or create instance
            instance = self.get_service(service_id)
            if not instance:
                return False
            
            service_instance = self.instances[service_id]
            service_instance.lifecycle_state = ServiceLifecycle.INITIALIZING
            
            # Initialize and start if it's an agent
            if isinstance(instance, BaseAgent):
                initialized = await instance.initialize()
                if initialized:
                    started = await instance.start()
                    if started:
                        service_instance.lifecycle_state = ServiceLifecycle.RUNNING
                        service_instance.start_count += 1
                        
                        # Register with message bus if available
                        if self.message_bus:
                            await self.message_bus.register_agent(instance)
                        
                        logger.info(f"Service started: {service_id}")
                        return True
                    else:
                        service_instance.lifecycle_state = ServiceLifecycle.FAILED
                        service_instance.failure_count += 1
                        return False
                else:
                    service_instance.lifecycle_state = ServiceLifecycle.FAILED
                    service_instance.failure_count += 1
                    return False
            else:
                # Non-agent service, just mark as running
                service_instance.lifecycle_state = ServiceLifecycle.RUNNING
                service_instance.start_count += 1
                logger.info(f"Service started: {service_id}")
                return True
            
        except Exception as e:
            logger.error(f"Service start failed for {service_id}: {e}")
            if service_id in self.instances:
                self.instances[service_id].lifecycle_state = ServiceLifecycle.FAILED
                self.instances[service_id].failure_count += 1
            return False
    
    async def _stop_service(self, service_id: str, timeout: float = 10.0) -> bool:
        """Stop individual service"""
        try:
            if service_id not in self.instances:
                return True
            
            service_instance = self.instances[service_id]
            instance = service_instance.instance
            
            service_instance.lifecycle_state = ServiceLifecycle.STOPPING
            
            # Stop if it's an agent
            if isinstance(instance, BaseAgent):
                stopped = await instance.stop(timeout)
                
                # Unregister from message bus
                if self.message_bus:
                    await self.message_bus.unregister_agent(service_id)
                
                if stopped:
                    service_instance.lifecycle_state = ServiceLifecycle.STOPPED
                    logger.info(f"Service stopped: {service_id}")
                    return True
                else:
                    service_instance.lifecycle_state = ServiceLifecycle.FAILED
                    return False
            else:
                # Non-agent service
                service_instance.lifecycle_state = ServiceLifecycle.STOPPED
                logger.info(f"Service stopped: {service_id}")
                return True
            
        except Exception as e:
            logger.error(f"Service stop failed for {service_id}: {e}")
            return False
    
    # === Dependency Graph Management ===
    
    def _validate_dependencies(self, service_id: str, dependencies: List[str]):
        """Validate service dependencies"""
        for dep_id in dependencies:
            if dep_id == service_id:
                raise CircularDependencyError(f"Service {service_id} cannot depend on itself")
            
            # Check if dependency creates a cycle
            if self._would_create_cycle(service_id, dep_id):
                raise CircularDependencyError(f"Adding dependency {dep_id} to {service_id} would create a cycle")
    
    def _would_create_cycle(self, service_id: str, dependency_id: str) -> bool:
        """Check if adding dependency would create a cycle"""
        if dependency_id not in self.dependency_graph:
            return False
        
        # DFS to check for cycle
        visited = set()
        
        def dfs(current: str) -> bool:
            if current == service_id:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            
            for dep in self.dependency_graph.get(current, set()):
                if dfs(dep):
                    return True
            
            return False
        
        return dfs(dependency_id)
    
    def _update_dependency_graph(self, service_id: str, dependencies: List[str]):
        """Update dependency graph"""
        # Update forward dependencies
        self.dependency_graph[service_id] = set(dependencies)
        
        # Update reverse dependencies
        for dep_id in dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = set()
            self.reverse_dependencies[dep_id].add(service_id)
    
    def _remove_from_dependency_graph(self, service_id: str):
        """Remove service from dependency graph"""
        # Remove forward dependencies
        if service_id in self.dependency_graph:
            dependencies = self.dependency_graph[service_id]
            for dep_id in dependencies:
                if dep_id in self.reverse_dependencies:
                    self.reverse_dependencies[dep_id].discard(service_id)
            del self.dependency_graph[service_id]
        
        # Remove reverse dependencies
        if service_id in self.reverse_dependencies:
            dependents = self.reverse_dependencies[service_id]
            for dependent_id in dependents:
                if dependent_id in self.dependency_graph:
                    self.dependency_graph[dependent_id].discard(service_id)
            del self.reverse_dependencies[service_id]
    
    def _calculate_resolution_order(self):
        """Calculate service resolution order using topological sort"""
        # Kahn's algorithm for topological sorting
        in_degree = {}
        
        # Initialize in-degree count
        for service_id in self.services:
            in_degree[service_id] = len(self.dependency_graph.get(service_id, set()))
        
        # Queue for services with no dependencies
        queue = [service_id for service_id, degree in in_degree.items() if degree == 0]
        resolution_order = []
        
        while queue:
            # Sort by priority for stable ordering
            queue.sort(key=lambda x: self.services[x].priority.value)
            current = queue.pop(0)
            resolution_order.append(current)
            
            # Update in-degree for dependents
            for dependent in self.reverse_dependencies.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(resolution_order) != len(self.services):
            remaining = set(self.services.keys()) - set(resolution_order)
            raise CircularDependencyError(f"Circular dependency detected among services: {remaining}")
        
        self.resolution_order = resolution_order
        logger.debug(f"Service resolution order: {self.resolution_order}")
    
    # === Health Monitoring ===
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if self.health_check_enabled:
            self.health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        
        if self.hot_reload_enabled:
            self.config_reload_task = asyncio.create_task(self._config_reload_loop())
    
    async def _stop_background_tasks(self):
        """Stop background monitoring tasks"""
        tasks = []
        
        if self.health_monitoring_task:
            tasks.append(self.health_monitoring_task)
        
        if self.config_reload_task:
            tasks.append(self.config_reload_task)
        
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        logger.info("Health monitoring started")
        
        while self.is_started:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self.is_started:
                    break
                
                await self._perform_health_checks()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
        
        logger.info("Health monitoring stopped")
    
    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        for service_id, service_instance in self.instances.items():
            try:
                if service_instance.lifecycle_state == ServiceLifecycle.RUNNING:
                    health_status = await self._check_service_health(service_id)
                    service_instance.health_status = health_status
                    service_instance.last_health_check = datetime.now()
                    
                    if not health_status and self.auto_recovery_enabled:
                        logger.warning(f"Service {service_id} unhealthy, attempting recovery")
                        await self._attempt_service_recovery(service_id)
            
            except Exception as e:
                logger.error(f"Health check failed for {service_id}: {e}")
                service_instance.health_status = False
    
    async def _check_service_health(self, service_id: str) -> bool:
        """Check health of individual service"""
        try:
            service_instance = self.instances[service_id]
            instance = service_instance.instance
            
            if isinstance(instance, BaseAgent):
                health_info = await instance.get_health_status()
                return health_info.get('overall_status') == 'healthy'
            else:
                # For non-agent services, assume healthy if running
                return service_instance.lifecycle_state == ServiceLifecycle.RUNNING
        
        except Exception as e:
            logger.error(f"Health check error for {service_id}: {e}")
            return False
    
    async def _attempt_service_recovery(self, service_id: str):
        """Attempt to recover unhealthy service"""
        try:
            logger.info(f"Attempting recovery for service: {service_id}")
            
            # Stop and restart the service
            await self._stop_service(service_id)
            await asyncio.sleep(1)  # Brief pause
            success = await self._start_service(service_id)
            
            if success:
                logger.info(f"Service {service_id} recovered successfully")
            else:
                logger.error(f"Service {service_id} recovery failed")
        
        except Exception as e:
            logger.error(f"Service recovery failed for {service_id}: {e}")
    
    # === Configuration Management ===
    
    async def _config_reload_loop(self):
        """Background configuration reloading"""
        logger.info("Configuration reload monitoring started")
        
        while self.is_started:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.is_started:
                    break
                
                await self._check_config_changes()
                
            except Exception as e:
                logger.error(f"Config reload monitoring error: {e}")
        
        logger.info("Configuration reload monitoring stopped")
    
    async def _check_config_changes(self):
        """Check for configuration file changes"""
        # Implementation would check file modification times
        # and reload configuration if needed
        pass
    
    # === Plugin System ===
    
    def register_plugin(self, plugin_id: str, plugin: Any) -> bool:
        """Register a plugin"""
        try:
            self.plugins[plugin_id] = plugin
            
            # Register plugin hooks if they exist
            if hasattr(plugin, 'get_hooks'):
                hooks = plugin.get_hooks()
                for hook_name, callback in hooks.items():
                    if hook_name not in self.plugin_hooks:
                        self.plugin_hooks[hook_name] = []
                    self.plugin_hooks[hook_name].append(callback)
            
            logger.info(f"Plugin registered: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Plugin registration failed: {e}")
            return False
    
    async def execute_plugin_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all callbacks for a plugin hook"""
        if hook_name in self.plugin_hooks:
            for callback in self.plugin_hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Plugin hook execution failed: {e}")
    
    # === Utility Methods ===
    
    async def _execute_callbacks(self, event: str):
        """Execute lifecycle callbacks"""
        for callback in self.lifecycle_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self)
                else:
                    callback(self)
            except Exception as e:
                logger.error(f"Callback execution failed for {event}: {e}")
    
    def add_lifecycle_callback(self, event: str, callback: Callable):
        """Add lifecycle callback"""
        if event in self.lifecycle_callbacks:
            self.lifecycle_callbacks[event].append(callback)
    
    def set_message_bus(self, message_bus: MessageBus):
        """Set message bus for agent communication"""
        self.message_bus = message_bus
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        return {
            'is_started': self.is_started,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'total_services': len(self.services),
            'running_services': len([
                inst for inst in self.instances.values()
                if inst.lifecycle_state == ServiceLifecycle.RUNNING
            ]),
            'failed_services': len([
                inst for inst in self.instances.values()
                if inst.lifecycle_state == ServiceLifecycle.FAILED
            ]),
            'healthy_services': len([
                inst for inst in self.instances.values()
                if inst.health_status
            ]),
            'resolution_order': self.resolution_order,
            'plugins_loaded': len(self.plugins),
            'health_monitoring_enabled': self.health_check_enabled,
            'config_hot_reload_enabled': self.hot_reload_enabled
        }


# Global registry instance
_registry_instance: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get global registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry()
    return _registry_instance


def set_registry(registry: AgentRegistry):
    """Set global registry instance"""
    global _registry_instance
    _registry_instance = registry


# Decorator for automatic service registration
def service(service_id: str = None, 
           dependencies: List[str] = None,
           scope: DependencyScope = DependencyScope.SINGLETON,
           **kwargs):
    """
    Decorator for automatic service registration
    
    Args:
        service_id: Service identifier (defaults to class name)
        dependencies: Service dependencies
        scope: Dependency injection scope
        **kwargs: Additional service descriptor options
    """
    def decorator(cls):
        actual_service_id = service_id or cls.__name__.lower()
        registry = get_registry()
        
        registry.register_service(
            service_id=actual_service_id,
            service_type=cls,
            dependencies=dependencies or [],
            scope=scope,
            **kwargs
        )
        
        return cls
    
    return decorator


def create_registry_config(
    health_check_enabled: bool = True,
    health_check_interval: int = 60,
    auto_recovery_enabled: bool = True,
    hot_reload_enabled: bool = False
) -> Dict[str, Any]:
    """
    Factory function for registry configuration
    
    Args:
        health_check_enabled: Enable health monitoring
        health_check_interval: Health check interval in seconds
        auto_recovery_enabled: Enable automatic service recovery
        hot_reload_enabled: Enable configuration hot reloading
        
    Returns:
        Registry configuration
    """
    return {
        'health_check_enabled': health_check_enabled,
        'health_check_interval': health_check_interval,
        'auto_recovery_enabled': auto_recovery_enabled,
        'hot_reload_enabled': hot_reload_enabled,
        'config_sources': []
    }