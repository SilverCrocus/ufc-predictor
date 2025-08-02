"""
Unified Configuration System for UFC Predictor
=============================================

This module consolidates all configuration settings across the UFC predictor
system, eliminating fragmentation and providing a centralized configuration
management system.

Features:
- Centralized configuration for all modules
- Environment-specific settings (dev/test/prod)
- Type validation and default values
- Configuration file loading and saving
- Dynamic configuration updates
- Backwards compatibility with existing configs

Usage:
    from ufc_predictor.utils.unified_config import config, ModelConfig, ScrapingConfig
    
    # Access configuration
    bankroll = config.profitability.default_bankroll
    model_path = config.model.tuned_model_path
    
    # Environment-specific configuration
    config.set_environment('production')
    
    # Update configuration
    config.profitability.max_bet_percentage = 0.03
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"


@dataclass
class PathConfig:
    """Centralized path configuration"""
    # Project structure
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "model")
    src_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "src")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    
    # Data files
    raw_fighters_data: Optional[Path] = None
    engineered_fighters_data: Optional[Path] = None
    corrected_fighters_data: Optional[Path] = None
    fights_data: Optional[Path] = None
    fight_dataset_with_diffs: Optional[Path] = None
    
    # Model files
    rf_model_path: Optional[Path] = None
    rf_tuned_model_path: Optional[Path] = None
    method_model_path: Optional[Path] = None
    winner_model_columns_path: Optional[Path] = None
    method_model_columns_path: Optional[Path] = None
    
    def __post_init__(self):
        """Set default paths after initialization"""
        if self.raw_fighters_data is None:
            self.raw_fighters_data = self.data_dir / "ufc_fighters_raw.csv"
        if self.engineered_fighters_data is None:
            self.engineered_fighters_data = self.data_dir / "ufc_fighters_engineered.csv"
        if self.corrected_fighters_data is None:
            self.corrected_fighters_data = self.model_dir / "ufc_fighters_engineered_corrected.csv"
        if self.fights_data is None:
            self.fights_data = self.data_dir / "ufc_fights.csv"
        if self.fight_dataset_with_diffs is None:
            self.fight_dataset_with_diffs = self.model_dir / "ufc_fight_dataset_with_diffs.csv"
        
        if self.rf_model_path is None:
            self.rf_model_path = self.model_dir / "ufc_random_forest_model.joblib"
        if self.rf_tuned_model_path is None:
            self.rf_tuned_model_path = self.model_dir / "ufc_random_forest_model_tuned.joblib"
        if self.method_model_path is None:
            self.method_model_path = self.model_dir / "ufc_multiclass_model.joblib"
        if self.winner_model_columns_path is None:
            self.winner_model_columns_path = self.model_dir / "winner_model_columns.json"
        if self.method_model_columns_path is None:
            self.method_model_columns_path = self.model_dir / "method_model_columns.json"
    
    def create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # Training parameters
    random_state: int = 42
    test_size: float = 0.2
    
    # Random Forest parameters
    rf_default_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Hyperparameter tuning grid
    rf_param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    })
    
    # Method model parameters
    method_param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    # Hyperparameter tuning settings
    hyperparameter_tuning: Dict[str, Any] = field(default_factory=lambda: {
        'n_iter': 100,
        'cv': 3,
        'verbose': 1,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Feature engineering settings
    percentage_columns: List[str] = field(default_factory=lambda: ['Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.'])
    numeric_columns: List[str] = field(default_factory=lambda: ['SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.'])
    reference_date: str = '2025-06-17'
    
    # Columns to drop for modeling
    columns_to_drop: List[str] = field(default_factory=lambda: [
        'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time',
        'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
        'blue_Name', 'red_Name', 'blue_is_winner'
    ])
    
    # Physical attribute parsing
    original_columns_to_drop: List[str] = field(default_factory=lambda: [
        'Height', 'Weight', 'Reach', 'Record', 'DOB', 'STANCE'
    ])


@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    # Base URLs
    ufcstats_base_url: str = 'http://ufcstats.com'
    ufcstats_fighters_url: str = 'http://ufcstats.com/statistics/fighters'
    tab_base_url: str = 'https://www.tab.com.au/sports/betting/UFC'
    tab_featured_fights_url: str = 'https://www.tab.com.au/sports/betting/UFC/competitions/UFC%20Featured%20Fights/tournaments/UFC%20Featured%20Fights'
    
    # Request settings
    delay_between_requests: float = 1.0  # seconds
    max_retries: int = 3
    timeout: int = 30  # seconds
    
    # Browser settings
    headless_mode: bool = True
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    # Selenium settings
    implicit_wait: int = 10
    page_load_timeout: int = 30
    script_timeout: int = 30
    
    # Stealth settings
    human_delay_min: float = 1.0
    human_delay_max: float = 3.0
    search_delay_min: float = 8.0
    search_delay_max: float = 15.0


@dataclass
class ProfitabilityConfig:
    """Profitability analysis configuration"""
    # Bankroll management
    default_bankroll: float = 1000.0
    max_bet_percentage_single: float = 0.05  # 5% for single bets
    max_bet_percentage_multi: float = 0.02   # 2% for multi-bets
    
    # Expected value thresholds
    min_expected_value_single: float = 0.05  # 5% minimum EV
    min_expected_value_multi: float = 0.15   # 15% minimum EV for multi-bets
    
    # Multi-bet settings
    max_multi_bet_legs: int = 4
    correlation_penalty: float = 0.08  # 8% penalty for same-event fights
    
    # Risk categories
    risk_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'LOW_RISK': {'min_probability': 0.4, 'max_legs': 2},
        'MEDIUM_RISK': {'min_probability': 0.2, 'max_legs': 3},
        'HIGH_RISK': {'min_probability': 0.1, 'max_legs': 4},
        'VERY_HIGH_RISK': {'min_probability': 0.0, 'max_legs': 6}
    })


@dataclass  
class OddsConfig:
    """Odds fetching and conversion configuration"""
    # API settings
    default_api_timeout: int = 10  # seconds
    cache_duration: int = 300  # 5 minutes
    max_cache_size: int = 1000
    
    # Odds conversion settings
    decimal_precision: int = 2
    probability_precision: int = 4
    
    # Name matching settings
    min_similarity_threshold: float = 0.7
    fuzzy_match_cutoff: float = 0.8
    
    # Supported bookmakers
    supported_bookmakers: List[str] = field(default_factory=lambda: [
        'tab', 'fanduel', 'draftkings', 'betmgm', 'caesars'
    ])
    
    # Rate limiting
    requests_per_minute: int = 60
    burst_requests: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Logging levels
    default_level: str = 'INFO'
    console_level: str = 'INFO'
    file_level: str = 'DEBUG'
    
    # File settings
    log_file_max_size: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5
    
    # Format settings
    console_format: str = '%(asctime)s | %(levelname)-8s | %(name)s | %(context)s %(message)s'
    file_format: str = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(context)s %(message)s'
    
    # Performance logging
    enable_performance_logging: bool = True
    performance_threshold: float = 1.0  # Log operations taking > 1 second


@dataclass
class DatabaseConfig:
    """Database configuration (for future use)"""
    # SQLite settings (default)
    sqlite_path: Optional[Path] = None
    enable_wal_mode: bool = True
    
    # PostgreSQL settings (optional)
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_database: Optional[str] = None
    postgres_username: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Connection settings
    connection_timeout: int = 30
    max_connections: int = 10


class UnifiedConfig:
    """Main configuration class that consolidates all settings"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self._config_file = None
        
        # Initialize all configuration sections
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.scraping = ScrapingConfig()
        self.profitability = ProfitabilityConfig()
        self.odds = OddsConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        
        # Create necessary directories
        self.paths.create_directories()
        
        # Load environment-specific overrides
        self._load_environment_config()
    
    def set_environment(self, environment: Union[Environment, str]):
        """Set the application environment"""
        if isinstance(environment, str):
            environment = Environment(environment)
        
        self.environment = environment
        self._load_environment_config()
        logger.info(f"Environment set to: {environment.value}")
    
    def _load_environment_config(self):
        """Load environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            self._configure_development()
        elif self.environment == Environment.TESTING:
            self._configure_testing()
        elif self.environment == Environment.PRODUCTION:
            self._configure_production()
    
    def _configure_development(self):
        """Configure for development environment"""
        self.logging.default_level = 'DEBUG'
        self.scraping.delay_between_requests = 0.5  # Faster for development
        self.odds.cache_duration = 60  # Shorter cache for development
        self.profitability.default_bankroll = 500.0  # Smaller test bankroll
    
    def _configure_testing(self):
        """Configure for testing environment"""
        self.logging.default_level = 'WARNING'
        self.scraping.delay_between_requests = 0.1  # Very fast for tests
        self.scraping.timeout = 5  # Short timeout for tests
        self.odds.cache_duration = 10  # Very short cache for tests
        self.profitability.default_bankroll = 100.0  # Small test bankroll
        
        # Use temporary directories for testing
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "ufc_predictor_test"
        self.paths.data_dir = temp_dir / "data"
        self.paths.model_dir = temp_dir / "model"
        self.paths.logs_dir = temp_dir / "logs"
        self.paths.cache_dir = temp_dir / "cache"
    
    def _configure_production(self):
        """Configure for production environment"""
        self.logging.default_level = 'INFO'
        self.scraping.delay_between_requests = 2.0  # Respectful delays
        self.scraping.headless_mode = True  # Always headless in production
        self.odds.cache_duration = 600  # Longer cache for production
        self.profitability.max_bet_percentage_single = 0.03  # More conservative
    
    def load_from_file(self, config_file: Union[str, Path]):
        """Load configuration from file (JSON or YAML)"""
        config_path = Path(config_file)
        self._config_file = config_path
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._apply_config_data(config_data)
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def save_to_file(self, config_file: Union[str, Path], format: str = 'json'):
        """Save current configuration to file"""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.to_dict()
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'yaml' or format.lower() == 'yml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to config objects"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        # Convert Path strings back to Path objects
                        if isinstance(getattr(section_obj, key), Path):
                            value = Path(value)
                        setattr(section_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'paths': asdict(self.paths),
            'model': asdict(self.model),
            'scraping': asdict(self.scraping),
            'profitability': asdict(self.profitability),
            'odds': asdict(self.odds),
            'logging': asdict(self.logging),
            'database': asdict(self.database)
        }
    
    def get_legacy_model_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for backwards compatibility"""
        return {
            # Paths (from old model_config.py)
            'PROJECT_ROOT': self.paths.project_root,
            'DATA_DIR': self.paths.data_dir,
            'MODEL_DIR': self.paths.model_dir,
            'SRC_DIR': self.paths.src_dir,
            'RAW_FIGHTERS_DATA': self.paths.raw_fighters_data,
            'ENGINEERED_FIGHTERS_DATA': self.paths.engineered_fighters_data,
            'CORRECTED_FIGHTERS_DATA': self.paths.corrected_fighters_data,
            'FIGHTS_DATA': self.paths.fights_data,
            'FIGHT_DATASET_WITH_DIFFS': self.paths.fight_dataset_with_diffs,
            'RF_MODEL_PATH': self.paths.rf_model_path,
            'RF_TUNED_MODEL_PATH': self.paths.rf_tuned_model_path,
            
            # Model settings
            'PERCENTAGE_COLUMNS': self.model.percentage_columns,
            'NUMERIC_COLUMNS': self.model.numeric_columns,
            'REFERENCE_DATE': self.model.reference_date,
            'RANDOM_STATE': self.model.random_state,
            'TEST_SIZE': self.model.test_size,
            'RF_DEFAULT_PARAMS': self.model.rf_default_params,
            'RF_PARAM_GRID': self.model.rf_param_grid,
            'COLUMNS_TO_DROP': self.model.columns_to_drop,
        }
    
    def get_legacy_webscraper_config(self) -> Dict[str, Any]:
        """Get configuration in legacy webscraper format"""
        return {
            'base_url': self.scraping.ufcstats_base_url,
            'fighters_url': self.scraping.ufcstats_fighters_url,
            'delay_between_requests': self.scraping.delay_between_requests,
            'max_retries': self.scraping.max_retries,
            'timeout': self.scraping.timeout
        }
    
    def validate(self) -> List[str]:
        """Validate configuration settings and return list of issues"""
        issues = []
        
        # Validate bankroll settings
        if self.profitability.default_bankroll <= 0:
            issues.append("Default bankroll must be positive")
        
        if not (0 < self.profitability.max_bet_percentage_single <= 1):
            issues.append("Max bet percentage for single bets must be between 0 and 1")
        
        # Validate model settings
        if self.model.test_size <= 0 or self.model.test_size >= 1:
            issues.append("Test size must be between 0 and 1")
        
        # Validate scraping settings
        if self.scraping.delay_between_requests < 0:
            issues.append("Delay between requests cannot be negative")
        
        # Validate odds settings
        if not (0 < self.odds.min_similarity_threshold <= 1):
            issues.append("Name similarity threshold must be between 0 and 1")
        
        return issues


# Global configuration instance
config = UnifiedConfig()

# Load configuration from environment variable or default file
config_file = os.getenv('UFC_PREDICTOR_CONFIG')
if config_file:
    config.load_from_file(config_file)
else:
    # Try to load from default locations
    possible_configs = [
        Path('ufc_predictor_config.json'),
        Path('config.json'),
        Path('ufc_predictor_config.yaml'),
        Path('config.yaml')
    ]
    
    for config_file in possible_configs:
        if config_file.exists():
            config.load_from_file(config_file)
            break

# Set environment from environment variable
env_name = os.getenv('UFC_PREDICTOR_ENV', 'development')
try:
    config.set_environment(Environment(env_name))
except ValueError:
    logger.warning(f"Invalid environment '{env_name}', using development")
    config.set_environment(Environment.DEVELOPMENT)

# Backwards compatibility exports
ModelConfig = config.model
ScrapingConfig = config.scraping  
ProfitabilityConfig = config.profitability

# Legacy exports for existing code
WEBSCRAPER_CONFIG = config.get_legacy_webscraper_config()


if __name__ == "__main__":
    print("🚀 UFC Predictor Unified Configuration System")
    print("=" * 60)
    
    # Show current configuration
    print(f"Environment: {config.environment.value}")
    print(f"Project root: {config.paths.project_root}")
    print(f"Default bankroll: ${config.profitability.default_bankroll}")
    print(f"Max bet percentage: {config.profitability.max_bet_percentage_single*100}%")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print(f"\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Configuration is valid")
    
    # Example: Save configuration
    config.save_to_file('example_config.json')
    print(f"\n💾 Example configuration saved to example_config.json")