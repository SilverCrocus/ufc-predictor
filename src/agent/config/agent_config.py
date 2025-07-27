"""
UFC Betting Agent Configuration

Environment-based configuration management for the automated UFC betting agent.
Provides secure, flexible configuration with validation and type safety.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings"""
    odds_api_key: str
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    max_connections: int = 10
    request_timeout: int = 30
    rate_limit_buffer: int = 50  # Keep this many requests in reserve
    
    def __post_init__(self):
        """Validate API configuration"""
        if not self.odds_api_key or len(self.odds_api_key) < 16:
            raise ValueError("Invalid or missing Odds API key")


@dataclass 
class ModelConfig:
    """Model configuration settings"""
    model_version: str = "latest"
    prediction_cache_ttl: int = 3600  # 1 hour
    confidence_threshold: float = 0.6
    symmetrical_prediction: bool = True
    
    # Model paths (auto-detected if not specified)
    winner_model_path: Optional[str] = None
    method_model_path: Optional[str] = None
    fighters_data_path: Optional[str] = None


@dataclass
class BettingConfig:
    """Betting strategy configuration"""
    initial_bankroll: float
    
    # Risk management
    max_bet_percentage: float = 0.05  # 5% max single bet
    max_portfolio_exposure: float = 0.25  # 25% total exposure
    min_expected_value: float = 0.05  # 5% minimum EV
    
    # Kelly criterion settings
    use_fractional_kelly: bool = True
    kelly_multiplier: float = 0.25  # Quarter Kelly
    
    # Bankroll tiers (override automatic detection)
    force_tier: Optional[str] = None  # "MICRO", "SMALL", "STANDARD"
    
    def __post_init__(self):
        """Validate betting configuration"""
        if self.initial_bankroll <= 0:
            raise ValueError("Initial bankroll must be positive")
        if not 0 < self.max_bet_percentage <= 1:
            raise ValueError("Max bet percentage must be between 0 and 1")
        if not 0 < self.max_portfolio_exposure <= 1:
            raise ValueError("Max portfolio exposure must be between 0 and 1")


@dataclass
class AgentConfig:
    """Main agent configuration"""
    cycle_interval: int = 300  # 5 minutes
    max_concurrent_predictions: int = 5
    enable_live_monitoring: bool = True
    auto_execute_bets: bool = False  # Advisory only by default
    
    # Storage paths
    storage_base_path: str = "."
    odds_storage_path: str = "odds"
    backup_storage_path: str = "betting_backups"
    analysis_export_path: str = "analysis_exports"
    
    # Monitoring and alerting
    enable_discord_alerts: bool = False
    discord_webhook_url: Optional[str] = None
    enable_slack_alerts: bool = False
    slack_webhook_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/ufc_agent.log"


@dataclass
class EventConfig:
    """Event-specific configuration"""
    name: str
    target_fights: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    monitoring_enabled: bool = True
    custom_ev_threshold: Optional[float] = None


class UFCAgentConfiguration:
    """
    Central configuration manager for the UFC betting agent
    
    Loads configuration from environment variables, files, and defaults.
    Provides validation, type safety, and easy access to all settings.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_prefix: str = "UFC_AGENT"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to JSON config file
            env_prefix: Prefix for environment variables
        """
        self.env_prefix = env_prefix
        self.config_file = config_file
        
        # Load and validate configuration
        self._load_configuration()
        self._validate_configuration()
        
        logger.info(f"Configuration loaded successfully")
    
    def _load_configuration(self):
        """Load configuration from environment and files"""
        # Start with defaults
        config_data = self._get_default_config()
        
        # Override with file configuration if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                config_data.update(file_config)
                logger.info(f"Loaded configuration from file: {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config_data.update(env_config)
        
        # Create configuration objects
        self.api = APIConfig(**config_data.get('api', {}))
        self.model = ModelConfig(**config_data.get('model', {}))
        self.betting = BettingConfig(**config_data.get('betting', {}))
        self.agent = AgentConfig(**config_data.get('agent', {}))
        
        # Load events configuration
        self.events = []
        for event_data in config_data.get('events', []):
            self.events.append(EventConfig(**event_data))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'api': {
                'odds_api_key': '',
                'odds_api_base_url': 'https://api.the-odds-api.com/v4',
                'max_connections': 10,
                'request_timeout': 30,
                'rate_limit_buffer': 50
            },
            'model': {
                'model_version': 'latest',
                'prediction_cache_ttl': 3600,
                'confidence_threshold': 0.6,
                'symmetrical_prediction': True
            },
            'betting': {
                'initial_bankroll': 0.0,
                'max_bet_percentage': 0.05,
                'max_portfolio_exposure': 0.25,
                'min_expected_value': 0.05,
                'use_fractional_kelly': True,
                'kelly_multiplier': 0.25
            },
            'agent': {
                'cycle_interval': 300,
                'max_concurrent_predictions': 5,
                'enable_live_monitoring': True,
                'auto_execute_bets': False,
                'storage_base_path': '.',
                'odds_storage_path': 'odds',
                'backup_storage_path': 'betting_backups',
                'analysis_export_path': 'analysis_exports',
                'log_level': 'INFO',
                'log_to_file': True,
                'log_file_path': 'logs/ufc_agent.log'
            },
            'events': []
        }
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {'api': {}, 'model': {}, 'betting': {}, 'agent': {}}
        
        # API configuration
        if os.getenv(f'{self.env_prefix}_ODDS_API_KEY'):
            env_config['api']['odds_api_key'] = os.getenv(f'{self.env_prefix}_ODDS_API_KEY')
        
        # Betting configuration
        if os.getenv(f'{self.env_prefix}_INITIAL_BANKROLL'):
            try:
                env_config['betting']['initial_bankroll'] = float(
                    os.getenv(f'{self.env_prefix}_INITIAL_BANKROLL')
                )
            except ValueError:
                logger.warning("Invalid INITIAL_BANKROLL in environment")
        
        if os.getenv(f'{self.env_prefix}_MAX_BET_PERCENTAGE'):
            try:
                env_config['betting']['max_bet_percentage'] = float(
                    os.getenv(f'{self.env_prefix}_MAX_BET_PERCENTAGE')
                )
            except ValueError:
                logger.warning("Invalid MAX_BET_PERCENTAGE in environment")
        
        if os.getenv(f'{self.env_prefix}_MIN_EXPECTED_VALUE'):
            try:
                env_config['betting']['min_expected_value'] = float(
                    os.getenv(f'{self.env_prefix}_MIN_EXPECTED_VALUE')
                )
            except ValueError:
                logger.warning("Invalid MIN_EXPECTED_VALUE in environment")
        
        # Agent configuration
        if os.getenv(f'{self.env_prefix}_AUTO_EXECUTE_BETS'):
            env_config['agent']['auto_execute_bets'] = os.getenv(
                f'{self.env_prefix}_AUTO_EXECUTE_BETS'
            ).lower() in ('true', '1', 'yes')
        
        if os.getenv(f'{self.env_prefix}_LOG_LEVEL'):
            env_config['agent']['log_level'] = os.getenv(f'{self.env_prefix}_LOG_LEVEL').upper()
        
        # Discord/Slack webhooks
        if os.getenv(f'{self.env_prefix}_DISCORD_WEBHOOK'):
            env_config['agent']['discord_webhook_url'] = os.getenv(f'{self.env_prefix}_DISCORD_WEBHOOK')
            env_config['agent']['enable_discord_alerts'] = True
        
        if os.getenv(f'{self.env_prefix}_SLACK_WEBHOOK'):
            env_config['agent']['slack_webhook_url'] = os.getenv(f'{self.env_prefix}_SLACK_WEBHOOK')
            env_config['agent']['enable_slack_alerts'] = True
        
        return env_config
    
    def _validate_configuration(self):
        """Validate the complete configuration"""
        # Ensure required directories exist
        for path_attr in ['odds_storage_path', 'backup_storage_path', 'analysis_export_path']:
            path = getattr(self.agent, path_attr)
            full_path = Path(self.agent.storage_base_path) / path
            full_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure log directory exists
        if self.agent.log_to_file:
            log_path = Path(self.agent.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate model paths exist if specified
        if self.model.winner_model_path and not Path(self.model.winner_model_path).exists():
            logger.warning(f"Winner model path does not exist: {self.model.winner_model_path}")
        
        if self.model.method_model_path and not Path(self.model.method_model_path).exists():
            logger.warning(f"Method model path does not exist: {self.model.method_model_path}")
        
        if self.model.fighters_data_path and not Path(self.model.fighters_data_path).exists():
            logger.warning(f"Fighters data path does not exist: {self.model.fighters_data_path}")
    
    def get_event_config(self, event_name: str) -> Optional[EventConfig]:
        """
        Get configuration for a specific event
        
        Args:
            event_name: Name of the event
            
        Returns:
            EventConfig if found, None otherwise
        """
        for event in self.events:
            if event.name == event_name:
                return event
        return None
    
    def add_event(self, event_config: EventConfig):
        """
        Add a new event configuration
        
        Args:
            event_config: Event configuration to add
        """
        # Remove existing event with same name
        self.events = [e for e in self.events if e.name != event_config.name]
        self.events.append(event_config)
        
        logger.info(f"Added event configuration: {event_config.name}")
    
    def save_to_file(self, file_path: str):
        """
        Save current configuration to a file
        
        Args:
            file_path: Path to save configuration file
        """
        config_data = {
            'api': {
                'odds_api_key': self.api.odds_api_key[:8] + '...',  # Mask API key
                'odds_api_base_url': self.api.odds_api_base_url,
                'max_connections': self.api.max_connections,
                'request_timeout': self.api.request_timeout,
                'rate_limit_buffer': self.api.rate_limit_buffer
            },
            'model': {
                'model_version': self.model.model_version,
                'prediction_cache_ttl': self.model.prediction_cache_ttl,
                'confidence_threshold': self.model.confidence_threshold,
                'symmetrical_prediction': self.model.symmetrical_prediction
            },
            'betting': {
                'initial_bankroll': self.betting.initial_bankroll,
                'max_bet_percentage': self.betting.max_bet_percentage,
                'max_portfolio_exposure': self.betting.max_portfolio_exposure,
                'min_expected_value': self.betting.min_expected_value,
                'use_fractional_kelly': self.betting.use_fractional_kelly,
                'kelly_multiplier': self.betting.kelly_multiplier
            },
            'agent': {
                'cycle_interval': self.agent.cycle_interval,
                'max_concurrent_predictions': self.agent.max_concurrent_predictions,
                'enable_live_monitoring': self.agent.enable_live_monitoring,
                'auto_execute_bets': self.agent.auto_execute_bets,
                'log_level': self.agent.log_level,
                'log_to_file': self.agent.log_to_file
            },
            'events': [
                {
                    'name': event.name,
                    'target_fights': event.target_fights,
                    'monitoring_enabled': event.monitoring_enabled,
                    'custom_ev_threshold': event.custom_ev_threshold
                }
                for event in self.events
            ]
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Configuration saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_summary(self) -> str:
        """
        Get a summary of the current configuration
        
        Returns:
            Formatted configuration summary
        """
        summary = [
            "🔧 UFC BETTING AGENT CONFIGURATION",
            "=" * 50,
            f"💳 Bankroll: ${self.betting.initial_bankroll:,.2f}",
            f"🎯 Max Bet: {self.betting.max_bet_percentage:.1%}",
            f"📊 Min EV: {self.betting.min_expected_value:.1%}",
            f"🔑 API Key: {self.api.odds_api_key[:8]}...",
            f"🤖 Auto Execute: {'Yes' if self.agent.auto_execute_bets else 'No (Advisory Only)'}",
            f"📈 Live Monitoring: {'Enabled' if self.agent.enable_live_monitoring else 'Disabled'}",
            f"⏱️  Cycle Interval: {self.agent.cycle_interval}s",
            f"📁 Storage Path: {self.agent.storage_base_path}",
            f"📄 Log Level: {self.agent.log_level}",
            "",
            f"📅 Configured Events: {len(self.events)}"
        ]
        
        for event in self.events:
            summary.append(f"   • {event.name} ({len(event.target_fights)} fights)")
        
        return "\n".join(summary)


def load_configuration(config_file: Optional[str] = None) -> UFCAgentConfiguration:
    """
    Load UFC agent configuration
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Loaded and validated configuration
    """
    return UFCAgentConfiguration(config_file)


def create_sample_config_file(file_path: str):
    """
    Create a sample configuration file
    
    Args:
        file_path: Path where to create the sample file
    """
    sample_config = {
        "api": {
            "odds_api_key": "your_odds_api_key_here",
            "max_connections": 10,
            "request_timeout": 30
        },
        "betting": {
            "initial_bankroll": 100.0,
            "max_bet_percentage": 0.05,
            "min_expected_value": 0.05,
            "use_fractional_kelly": True,
            "kelly_multiplier": 0.25
        },
        "agent": {
            "cycle_interval": 300,
            "enable_live_monitoring": True,
            "auto_execute_bets": False,
            "log_level": "INFO"
        },
        "events": [
            {
                "name": "UFC_Fight_Night_Sample",
                "target_fights": [
                    "Fighter A vs Fighter B",
                    "Fighter C vs Fighter D"
                ],
                "monitoring_enabled": True
            }
        ]
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"Sample configuration created: {file_path}")
    except Exception as e:
        print(f"Failed to create sample config: {e}")


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config_file("agent_config_sample.json")
    
    # Test configuration loading
    try:
        config = load_configuration()
        print(config.get_summary())
    except Exception as e:
        print(f"Configuration test failed: {e}")