"""
Monitor Agent for UFC Betting System
====================================

Specialized agent for system monitoring, health tracking, and alert management:
- Real-time system health monitoring across all agents
- Performance metrics collection and analysis
- Alert management and notification routing
- Service dependency tracking
- Anomaly detection and automated recovery
- Dashboard data aggregation
"""

import asyncio
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import sqlite3
from collections import deque
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .base_agent import BaseAgent, AgentPriority, AgentMessage, AgentState

logger = logging.getLogger(__name__)


@dataclass
class SystemHealthMetrics:
    """System-wide health metrics"""
    timestamp: datetime
    overall_health_score: float  # 0-1, 1 = perfect health
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_latency_ms: float
    active_agents: int
    healthy_agents: int
    error_rate: float
    response_time_ms: float
    uptime_hours: float


@dataclass
class AgentHealthStatus:
    """Individual agent health status"""
    agent_id: str
    state: str
    health_score: float
    last_heartbeat: datetime
    response_time_ms: float
    error_count: int
    warning_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert configuration rule"""
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    severity: str   # "info", "warning", "error", "critical"
    cooldown_minutes: int = 15
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    rule_id: str
    message: str
    severity: str
    timestamp: datetime
    agent_id: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MonitorAgent(BaseAgent):
    """
    System monitoring and health management agent
    
    Responsibilities:
    - Monitor health of all system agents
    - Collect and aggregate performance metrics
    - Detect anomalies and system issues
    - Trigger alerts and notifications
    - Provide dashboard data and health reports
    - Coordinate system recovery procedures
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MonitorAgent
        
        Args:
            config: Monitor agent configuration
        """
        super().__init__(
            agent_id="monitor_agent",
            priority=AgentPriority.MEDIUM
        )
        
        self.config = config
        
        # Monitoring configuration
        self.monitoring_interval = config.get('monitoring_interval', 30)  # 30 seconds
        self.health_check_timeout = config.get('health_check_timeout', 10)  # 10 seconds
        self.metrics_retention_hours = config.get('metrics_retention_hours', 168)  # 7 days
        
        # Agent tracking
        self.monitored_agents: Set[str] = set()
        self.agent_health_status: Dict[str, AgentHealthStatus] = {}
        self.agent_last_seen: Dict[str, datetime] = {}
        self.agent_heartbeat_interval = config.get('agent_heartbeat_interval', 60)  # 60 seconds
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.metrics_db_path = config.get('metrics_db_path', 'monitoring_metrics.db')
        self.enable_metrics_persistence = config.get('enable_metrics_persistence', True)
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Notification configuration
        self.notification_config = config.get('notifications', {})
        self.enable_email_alerts = self.notification_config.get('email', {}).get('enabled', False)
        self.enable_webhook_alerts = self.notification_config.get('webhook', {}).get('enabled', False)
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detection_threshold = config.get('anomaly_detection_threshold', 2.0)  # Standard deviations
        
        # Dashboard data
        self.dashboard_data_cache: Dict[str, Any] = {}
        self.dashboard_cache_ttl = config.get('dashboard_cache_ttl', 30)  # 30 seconds
        self.last_dashboard_update = datetime.now()
        
        # Register message handlers
        self.register_message_handler('agent_heartbeat', self._handle_agent_heartbeat)
        self.register_message_handler('register_agent', self._handle_register_agent)
        self.register_message_handler('get_system_health', self._handle_get_system_health)
        self.register_message_handler('get_dashboard_data', self._handle_get_dashboard_data)
        self.register_message_handler('create_alert_rule', self._handle_create_alert_rule)
        self.register_message_handler('acknowledge_alert', self._handle_acknowledge_alert)
        
        logger.info("MonitorAgent initialized")
    
    async def _initialize_agent(self) -> bool:
        """Initialize monitor agent components"""
        try:
            # Initialize metrics database
            if self.enable_metrics_persistence:
                await self._initialize_metrics_database()
            
            # Load default alert rules
            await self._load_default_alert_rules()
            
            # Initialize performance baselines
            await self._initialize_performance_baselines()
            
            logger.info("MonitorAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MonitorAgent initialization failed: {e}")
            return False
    
    async def _start_agent(self) -> bool:
        """Start monitor agent operations"""
        try:
            # Start system monitoring
            self._system_monitoring_task = asyncio.create_task(
                self._system_monitoring_loop()
            )
            
            # Start agent health monitoring
            self._agent_monitoring_task = asyncio.create_task(
                self._agent_monitoring_loop()
            )
            
            # Start alert processing
            self._alert_processing_task = asyncio.create_task(
                self._alert_processing_loop()
            )
            
            # Start metrics persistence
            if self.enable_metrics_persistence:
                self._metrics_persistence_task = asyncio.create_task(
                    self._metrics_persistence_loop()
                )
            
            logger.info("MonitorAgent started successfully")
            return True
            
        except Exception as e:
            logger.error(f"MonitorAgent start failed: {e}")
            return False
    
    async def _stop_agent(self) -> bool:
        """Stop monitor agent operations"""
        try:
            # Cancel background tasks
            tasks_to_cancel = []
            
            if hasattr(self, '_system_monitoring_task'):
                tasks_to_cancel.append(self._system_monitoring_task)
            
            if hasattr(self, '_agent_monitoring_task'):
                tasks_to_cancel.append(self._agent_monitoring_task)
            
            if hasattr(self, '_alert_processing_task'):
                tasks_to_cancel.append(self._alert_processing_task)
            
            if hasattr(self, '_metrics_persistence_task'):
                tasks_to_cancel.append(self._metrics_persistence_task)
            
            for task in tasks_to_cancel:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Save final metrics
            if self.enable_metrics_persistence:
                await self._save_metrics_batch()
            
            logger.info("MonitorAgent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"MonitorAgent stop failed: {e}")
            return False
    
    async def _process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming messages"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message.payload)
        else:
            logger.warning(f"MonitorAgent: No handler for message type '{message.message_type}'")
            return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Monitor agent health check"""
        current_metrics = await self._collect_system_metrics()
        
        health_info = {
            'monitored_agents': len(self.monitored_agents),
            'healthy_agents': len([status for status in self.agent_health_status.values() if status.health_score > 0.8]),
            'active_alerts': len(self.active_alerts),
            'system_health_score': current_metrics.overall_health_score,
            'metrics_history_length': len(self.system_metrics_history),
            'last_system_check': current_metrics.timestamp.isoformat(),
            'database_enabled': self.enable_metrics_persistence
        }
        
        return health_info
    
    # === System Monitoring ===
    
    async def _collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect comprehensive system metrics"""
        try:
            timestamp = datetime.now()
            
            # System resource metrics
            cpu_usage = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network latency (simplified - could ping external service)
            network_latency = 0.0  # Would implement actual network check
            
            # Agent health metrics
            active_agents = len(self.monitored_agents)
            healthy_agents = len([
                status for status in self.agent_health_status.values()
                if status.health_score > 0.8
            ])
            
            # Error rate calculation
            total_errors = sum(status.error_count for status in self.agent_health_status.values())
            total_operations = len(self.agent_health_status) * 100  # Simplified
            error_rate = total_errors / total_operations if total_operations > 0 else 0.0
            
            # Response time calculation
            response_times = [status.response_time_ms for status in self.agent_health_status.values()]
            avg_response_time = np.mean(response_times) if response_times else 0.0
            
            # System uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime_hours = (timestamp - boot_time).total_seconds() / 3600
            
            # Overall health score calculation
            health_components = {
                'cpu_health': 1.0 - (cpu_usage / 100.0),
                'memory_health': 1.0 - (memory.percent / 100.0),
                'disk_health': 1.0 - (disk.percent / 100.0),
                'agent_health': healthy_agents / active_agents if active_agents > 0 else 1.0,
                'error_health': 1.0 - min(error_rate, 1.0)
            }
            
            overall_health_score = np.mean(list(health_components.values()))
            
            return SystemHealthMetrics(
                timestamp=timestamp,
                overall_health_score=overall_health_score,
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_latency_ms=network_latency,
                active_agents=active_agents,
                healthy_agents=healthy_agents,
                error_rate=error_rate,
                response_time_ms=avg_response_time,
                uptime_hours=uptime_hours
            )
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return SystemHealthMetrics(
                timestamp=datetime.now(),
                overall_health_score=0.0,
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_latency_ms=0.0,
                active_agents=0,
                healthy_agents=0,
                error_rate=1.0,
                response_time_ms=0.0,
                uptime_hours=0.0
            )
    
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any] = None):
        """Register agent for monitoring"""
        try:
            self.monitored_agents.add(agent_id)
            self.agent_last_seen[agent_id] = datetime.now()
            
            # Initialize health status
            self.agent_health_status[agent_id] = AgentHealthStatus(
                agent_id=agent_id,
                state="initializing",
                health_score=1.0,
                last_heartbeat=datetime.now(),
                response_time_ms=0.0,
                error_count=0,
                warning_count=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                custom_metrics=agent_info or {}
            )
            
            logger.info(f"Registered agent for monitoring: {agent_id}")
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
    
    async def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]):
        """Update agent health status"""
        try:
            if agent_id not in self.agent_health_status:
                await self.register_agent(agent_id)
            
            status = self.agent_health_status[agent_id]
            
            # Update basic metrics
            status.state = health_data.get('state', status.state)
            status.health_score = health_data.get('health_score', status.health_score)
            status.last_heartbeat = datetime.now()
            status.response_time_ms = health_data.get('response_time_ms', status.response_time_ms)
            status.error_count = health_data.get('error_count', status.error_count)
            status.warning_count = health_data.get('warning_count', status.warning_count)
            status.memory_usage_mb = health_data.get('memory_usage_mb', status.memory_usage_mb)
            status.cpu_usage_percent = health_data.get('cpu_usage_percent', status.cpu_usage_percent)
            
            # Update custom metrics
            if 'custom_metrics' in health_data:
                status.custom_metrics.update(health_data['custom_metrics'])
            
            self.agent_last_seen[agent_id] = datetime.now()
            
        except Exception as e:
            logger.error(f"Agent health update failed for {agent_id}: {e}")
    
    # === Alert Management ===
    
    async def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                condition="system_metrics.cpu_usage_percent > 80",
                severity="warning",
                cooldown_minutes=10,
                notification_channels=["email", "webhook"]
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                condition="system_metrics.memory_usage_percent > 85",
                severity="warning",
                cooldown_minutes=10,
                notification_channels=["email"]
            ),
            AlertRule(
                rule_id="agent_down",
                name="Agent Down",
                condition="agent_status.health_score < 0.1",
                severity="critical",
                cooldown_minutes=5,
                notification_channels=["email", "webhook"]
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                condition="system_metrics.error_rate > 0.1",
                severity="error",
                cooldown_minutes=15,
                notification_channels=["email"]
            ),
            AlertRule(
                rule_id="system_health_degraded",
                name="System Health Degraded",
                condition="system_metrics.overall_health_score < 0.7",
                severity="warning",
                cooldown_minutes=20,
                notification_channels=["email"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    async def evaluate_alert_rules(self, 
                                 system_metrics: SystemHealthMetrics,
                                 agent_statuses: Dict[str, AgentHealthStatus]):
        """Evaluate alert rules against current metrics"""
        try:
            current_time = datetime.now()
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule_id in self.alert_cooldowns:
                    cooldown_end = self.alert_cooldowns[rule_id] + timedelta(minutes=rule.cooldown_minutes)
                    if current_time < cooldown_end:
                        continue
                
                # Evaluate condition
                try:
                    # Create evaluation context
                    context = {
                        'system_metrics': system_metrics,
                        'agent_statuses': agent_statuses,
                        'datetime': datetime,
                        'len': len,
                        'sum': sum,
                        'max': max,
                        'min': min
                    }
                    
                    # For agent-specific rules, evaluate for each agent
                    if 'agent_status' in rule.condition:
                        for agent_id, agent_status in agent_statuses.items():
                            agent_context = context.copy()
                            agent_context['agent_status'] = agent_status
                            
                            if eval(rule.condition, {"__builtins__": {}}, agent_context):
                                await self._trigger_alert(rule, agent_id=agent_id, context=agent_context)
                    else:
                        # System-wide rule
                        if eval(rule.condition, {"__builtins__": {}}, context):
                            await self._trigger_alert(rule, context=context)
                
                except Exception as e:
                    logger.error(f"Alert rule evaluation failed for {rule_id}: {e}")
            
        except Exception as e:
            logger.error(f"Alert rule evaluation failed: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, agent_id: str = None, context: Dict[str, Any] = None):
        """Trigger an alert"""
        try:
            alert_id = f"{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create alert message
            if agent_id:
                message = f"Agent {agent_id}: {rule.name}"
            else:
                message = f"System: {rule.name}"
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                message=message,
                severity=rule.severity,
                timestamp=datetime.now(),
                agent_id=agent_id,
                metadata=context or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Set cooldown
            self.alert_cooldowns[rule.rule_id] = datetime.now()
            
            # Send notifications
            await self._send_alert_notifications(alert, rule.notification_channels)
            
            logger.warning(f"Alert triggered: {alert.message} ({alert.severity})")
            
        except Exception as e:
            logger.error(f"Alert trigger failed: {e}")
    
    async def _send_alert_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications to configured channels"""
        try:
            for channel in channels:
                if channel == "email" and self.enable_email_alerts:
                    await self._send_email_alert(alert)
                elif channel == "webhook" and self.enable_webhook_alerts:
                    await self._send_webhook_alert(alert)
                # Add more notification channels as needed
        
        except Exception as e:
            logger.error(f"Alert notification failed: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification"""
        try:
            email_config = self.notification_config.get('email', {})
            
            if not email_config.get('enabled'):
                return
            
            # Email sending implementation would go here
            # For now, just log the intent
            logger.info(f"Email alert sent: {alert.message}")
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert notification"""
        try:
            webhook_config = self.notification_config.get('webhook', {})
            
            if not webhook_config.get('enabled'):
                return
            
            # Webhook sending implementation would go here
            # For now, just log the intent
            logger.info(f"Webhook alert sent: {alert.message}")
            
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
    
    # === Dashboard Data ===
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Check cache validity
            cache_age = (datetime.now() - self.last_dashboard_update).total_seconds()
            if cache_age < self.dashboard_cache_ttl and self.dashboard_data_cache:
                return self.dashboard_data_cache
            
            # Collect fresh dashboard data
            current_metrics = await self._collect_system_metrics()
            
            # Recent metrics for charts
            recent_metrics = list(self.system_metrics_history)[-100:]  # Last 100 data points
            
            # Agent summary
            agent_summary = {}
            for agent_id, status in self.agent_health_status.items():
                agent_summary[agent_id] = {
                    'state': status.state,
                    'health_score': status.health_score,
                    'last_heartbeat': status.last_heartbeat.isoformat(),
                    'error_count': status.error_count,
                    'memory_usage_mb': status.memory_usage_mb
                }
            
            # Active alerts summary
            alerts_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity
                if severity not in alerts_by_severity:
                    alerts_by_severity[severity] = 0
                alerts_by_severity[severity] += 1
            
            # Performance trends
            if len(recent_metrics) > 1:
                health_trend = [m.overall_health_score for m in recent_metrics]
                cpu_trend = [m.cpu_usage_percent for m in recent_metrics]
                memory_trend = [m.memory_usage_percent for m in recent_metrics]
            else:
                health_trend = [current_metrics.overall_health_score]
                cpu_trend = [current_metrics.cpu_usage_percent]
                memory_trend = [current_metrics.memory_usage_percent]
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_overview': {
                    'overall_health_score': current_metrics.overall_health_score,
                    'active_agents': current_metrics.active_agents,
                    'healthy_agents': current_metrics.healthy_agents,
                    'active_alerts': len(self.active_alerts),
                    'uptime_hours': current_metrics.uptime_hours
                },
                'resource_usage': {
                    'cpu_percent': current_metrics.cpu_usage_percent,
                    'memory_percent': current_metrics.memory_usage_percent,
                    'disk_percent': current_metrics.disk_usage_percent,
                    'network_latency_ms': current_metrics.network_latency_ms
                },
                'agent_status': agent_summary,
                'alerts': {
                    'by_severity': alerts_by_severity,
                    'recent_alerts': [
                        {
                            'message': alert.message,
                            'severity': alert.severity,
                            'timestamp': alert.timestamp.isoformat(),
                            'agent_id': alert.agent_id
                        }
                        for alert in list(self.alert_history)[-10:]
                    ]
                },
                'trends': {
                    'health_score': health_trend,
                    'cpu_usage': cpu_trend,
                    'memory_usage': memory_trend,
                    'timestamps': [m.timestamp.isoformat() for m in recent_metrics]
                }
            }
            
            # Update cache
            self.dashboard_data_cache = dashboard_data
            self.last_dashboard_update = datetime.now()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard data collection failed: {e}")
            return {'error': str(e)}
    
    # === Background Tasks ===
    
    async def _system_monitoring_loop(self):
        """Main system monitoring loop"""
        logger.info("MonitorAgent system monitoring started")
        
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.system_metrics_history.append(metrics)
                
                # Evaluate alert rules
                await self.evaluate_alert_rules(metrics, self.agent_health_status)
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
        
        logger.info("MonitorAgent system monitoring stopped")
    
    async def _agent_monitoring_loop(self):
        """Agent health monitoring loop"""
        logger.info("MonitorAgent agent monitoring started")
        
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for stale agents
                stale_agents = []
                for agent_id, last_seen in self.agent_last_seen.items():
                    time_since_seen = (current_time - last_seen).total_seconds()
                    if time_since_seen > self.agent_heartbeat_interval * 2:  # 2x heartbeat interval
                        stale_agents.append(agent_id)
                
                # Update stale agent health
                for agent_id in stale_agents:
                    if agent_id in self.agent_health_status:
                        self.agent_health_status[agent_id].health_score = 0.0
                        self.agent_health_status[agent_id].state = "unresponsive"
                
                # Sleep until next check
                await asyncio.sleep(self.agent_heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Agent monitoring error: {e}")
                await asyncio.sleep(self.agent_heartbeat_interval)
        
        logger.info("MonitorAgent agent monitoring stopped")
    
    async def _alert_processing_loop(self):
        """Alert processing and cleanup loop"""
        logger.info("MonitorAgent alert processing started")
        
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Clean up old alerts (auto-resolve after 24 hours)
                expired_alerts = []
                for alert_id, alert in self.active_alerts.items():
                    alert_age = (current_time - alert.timestamp).total_seconds()
                    if alert_age > 24 * 3600:  # 24 hours
                        expired_alerts.append(alert_id)
                
                for alert_id in expired_alerts:
                    self.active_alerts[alert_id].resolved = True
                    del self.active_alerts[alert_id]
                
                # Clean up old cooldowns
                expired_cooldowns = []
                for rule_id, cooldown_time in self.alert_cooldowns.items():
                    if rule_id in self.alert_rules:
                        rule = self.alert_rules[rule_id]
                        cooldown_end = cooldown_time + timedelta(minutes=rule.cooldown_minutes)
                        if current_time > cooldown_end:
                            expired_cooldowns.append(rule_id)
                
                for rule_id in expired_cooldowns:
                    del self.alert_cooldowns[rule_id]
                
                # Sleep until next cleanup
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("MonitorAgent alert processing stopped")
    
    async def _metrics_persistence_loop(self):
        """Metrics persistence loop"""
        logger.info("MonitorAgent metrics persistence started")
        
        while not self._stop_event.is_set():
            try:
                # Save metrics batch every 5 minutes
                await asyncio.sleep(300)
                
                if self._stop_event.is_set():
                    break
                
                await self._save_metrics_batch()
                
            except Exception as e:
                logger.error(f"Metrics persistence error: {e}")
        
        logger.info("MonitorAgent metrics persistence stopped")
    
    # === Message Handlers ===
    
    async def _handle_agent_heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent heartbeat message"""
        try:
            agent_id = payload.get('agent_id')
            health_data = payload.get('health_data', {})
            
            if not agent_id:
                return {'status': 'error', 'error': 'Missing agent_id'}
            
            await self.update_agent_health(agent_id, health_data)
            
            return {'status': 'success', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_register_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration message"""
        try:
            agent_id = payload.get('agent_id')
            agent_info = payload.get('agent_info', {})
            
            if not agent_id:
                return {'status': 'error', 'error': 'Missing agent_id'}
            
            await self.register_agent(agent_id, agent_info)
            
            return {'status': 'success', 'registered': agent_id}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_system_health(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system health request"""
        try:
            current_metrics = await self._collect_system_metrics()
            
            return {
                'status': 'success',
                'system_health': {
                    'overall_health_score': current_metrics.overall_health_score,
                    'cpu_usage_percent': current_metrics.cpu_usage_percent,
                    'memory_usage_percent': current_metrics.memory_usage_percent,
                    'active_agents': current_metrics.active_agents,
                    'healthy_agents': current_metrics.healthy_agents,
                    'active_alerts': len(self.active_alerts),
                    'timestamp': current_metrics.timestamp.isoformat()
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_dashboard_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dashboard data request"""
        try:
            dashboard_data = await self.get_dashboard_data()
            
            return {
                'status': 'success',
                'dashboard_data': dashboard_data
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_create_alert_rule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alert rule creation"""
        try:
            rule_data = payload.get('rule_data', {})
            
            rule = AlertRule(
                rule_id=rule_data.get('rule_id'),
                name=rule_data.get('name'),
                condition=rule_data.get('condition'),
                severity=rule_data.get('severity', 'warning'),
                cooldown_minutes=rule_data.get('cooldown_minutes', 15),
                enabled=rule_data.get('enabled', True),
                notification_channels=rule_data.get('notification_channels', [])
            )
            
            self.alert_rules[rule.rule_id] = rule
            
            return {'status': 'success', 'rule_id': rule.rule_id}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_acknowledge_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alert acknowledgment"""
        try:
            alert_id = payload.get('alert_id')
            
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return {'status': 'success', 'acknowledged': alert_id}
            else:
                return {'status': 'error', 'error': 'Alert not found'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    # === Data Persistence ===
    
    async def _initialize_metrics_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.metrics_db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT PRIMARY KEY,
                    overall_health_score REAL,
                    cpu_usage_percent REAL,
                    memory_usage_percent REAL,
                    disk_usage_percent REAL,
                    active_agents INTEGER,
                    healthy_agents INTEGER,
                    error_rate REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    timestamp TEXT,
                    agent_id TEXT,
                    health_score REAL,
                    state TEXT,
                    memory_usage_mb REAL,
                    error_count INTEGER,
                    PRIMARY KEY (timestamp, agent_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Metrics database initialized")
            
        except Exception as e:
            logger.error(f"Metrics database initialization failed: {e}")
    
    async def _save_metrics_batch(self):
        """Save metrics batch to database"""
        try:
            if not self.enable_metrics_persistence:
                return
            
            conn = sqlite3.connect(self.metrics_db_path)
            cursor = conn.cursor()
            
            # Save recent system metrics
            recent_system_metrics = list(self.system_metrics_history)[-10:]  # Last 10 entries
            for metrics in recent_system_metrics:
                cursor.execute('''
                    INSERT OR REPLACE INTO system_metrics 
                    (timestamp, overall_health_score, cpu_usage_percent, memory_usage_percent,
                     disk_usage_percent, active_agents, healthy_agents, error_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp.isoformat(),
                    metrics.overall_health_score,
                    metrics.cpu_usage_percent,
                    metrics.memory_usage_percent,
                    metrics.disk_usage_percent,
                    metrics.active_agents,
                    metrics.healthy_agents,
                    metrics.error_rate
                ))
            
            # Save agent metrics
            timestamp = datetime.now().isoformat()
            for agent_id, status in self.agent_health_status.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO agent_metrics 
                    (timestamp, agent_id, health_score, state, memory_usage_mb, error_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    agent_id,
                    status.health_score,
                    status.state,
                    status.memory_usage_mb,
                    status.error_count
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Metrics batch save failed: {e}")
    
    # === Utility Methods ===
    
    async def _initialize_performance_baselines(self):
        """Initialize performance baselines for anomaly detection"""
        # This would load historical data to establish baselines
        # For now, use reasonable defaults
        self.performance_baselines = {
            'cpu_usage_baseline': 50.0,      # 50% CPU baseline
            'memory_usage_baseline': 60.0,   # 60% memory baseline
            'response_time_baseline': 100.0, # 100ms response time baseline
            'error_rate_baseline': 0.01      # 1% error rate baseline
        }
        
        logger.info("Performance baselines initialized")


def create_monitor_agent_config(
    monitoring_interval: int = 30,
    enable_metrics_persistence: bool = True,
    enable_email_alerts: bool = False,
    enable_webhook_alerts: bool = False
) -> Dict[str, Any]:
    """
    Factory function for MonitorAgent configuration
    
    Args:
        monitoring_interval: System monitoring interval in seconds
        enable_metrics_persistence: Enable database storage of metrics
        enable_email_alerts: Enable email notifications
        enable_webhook_alerts: Enable webhook notifications
        
    Returns:
        MonitorAgent configuration
    """
    return {
        'monitoring_interval': monitoring_interval,
        'health_check_timeout': 10,
        'metrics_retention_hours': 168,  # 7 days
        'agent_heartbeat_interval': 60,
        'enable_metrics_persistence': enable_metrics_persistence,
        'metrics_db_path': 'monitoring_metrics.db',
        'dashboard_cache_ttl': 30,
        'anomaly_detection_threshold': 2.0,
        'notifications': {
            'email': {
                'enabled': enable_email_alerts,
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': 'monitor@ufcbetting.com',
                'to_addresses': []
            },
            'webhook': {
                'enabled': enable_webhook_alerts,
                'url': '',
                'headers': {},
                'timeout': 10
            }
        }
    }