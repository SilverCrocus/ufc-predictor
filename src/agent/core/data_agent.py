"""
Data Agent for UFC Betting System
=================================

Specialized agent for data management, validation, and quality assurance:
- Data ingestion and validation
- Feature engineering coordination
- Data quality scoring and monitoring
- Fighter data synchronization
- Statistical validation and outlier detection
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from .base_agent import BaseAgent, AgentPriority, AgentMessage
from ..validation.unified_validator import UnifiedValidator
from ..validation.dataframe_validator import DataFrameValidator
from ..validation.fighter_name_validator import FighterNameValidator

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness_score: float  # Percentage of non-null values
    consistency_score: float   # Consistency across data sources
    validity_score: float      # Valid data types and ranges
    timeliness_score: float    # Data freshness
    accuracy_score: float      # Statistical accuracy measures
    overall_score: float       # Weighted overall quality
    
    issues_found: List[str]    # List of data quality issues
    recommendations: List[str] # Recommendations for improvement


@dataclass
class DataValidationResult:
    """Result of data validation operation"""
    is_valid: bool
    validation_score: float
    error_count: int
    warning_count: int
    errors: List[str]
    warnings: List[str]
    metrics: DataQualityMetrics
    processed_records: int
    validation_time_ms: float


class DataAgent(BaseAgent):
    """
    Data management and quality assurance agent
    
    Responsibilities:
    - Validate incoming fighter and fight data
    - Coordinate feature engineering processes
    - Monitor data quality and detect anomalies
    - Synchronize data across multiple sources
    - Provide data quality metrics and recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataAgent
        
        Args:
            config: Data agent configuration
        """
        super().__init__(
            agent_id="data_agent",
            priority=AgentPriority.HIGH
        )
        
        self.config = config
        
        # Data paths and sources
        self.data_sources = config.get('data_sources', {})
        self.feature_engineering_config = config.get('feature_engineering', {})
        self.validation_config = config.get('validation', {})
        
        # Quality thresholds
        self.min_quality_score = config.get('min_quality_score', 0.8)
        self.quality_check_interval = config.get('quality_check_interval', 3600)  # 1 hour
        
        # Data caches
        self.fighters_cache: Optional[pd.DataFrame] = None
        self.fights_cache: Optional[pd.DataFrame] = None
        self.cache_expiry: Optional[datetime] = None
        self.cache_ttl = config.get('cache_ttl', 1800)  # 30 minutes
        
        # Validators
        self.unified_validator = None
        self.dataframe_validator = None
        self.fighter_name_validator = None
        
        # Quality monitoring
        self.quality_history: List[DataQualityMetrics] = []
        self.max_quality_history = 100
        
        # Register message handlers
        self.register_message_handler('validate_data', self._handle_validate_data)
        self.register_message_handler('get_data_quality', self._handle_get_data_quality)
        self.register_message_handler('refresh_data', self._handle_refresh_data)
        self.register_message_handler('engineer_features', self._handle_engineer_features)
        
        logger.info("DataAgent initialized")
    
    async def _initialize_agent(self) -> bool:
        """Initialize data agent components"""
        try:
            # Initialize validators
            self.unified_validator = UnifiedValidator()
            self.dataframe_validator = DataFrameValidator()
            self.fighter_name_validator = FighterNameValidator()
            
            # Load initial data
            await self._load_initial_data()
            
            # Perform initial quality assessment
            await self._perform_initial_quality_check()
            
            logger.info("DataAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"DataAgent initialization failed: {e}")
            return False
    
    async def _start_agent(self) -> bool:
        """Start data agent operations"""
        try:
            # Start periodic quality monitoring
            self._quality_monitoring_task = asyncio.create_task(
                self._quality_monitoring_loop()
            )
            
            logger.info("DataAgent started successfully")
            return True
            
        except Exception as e:
            logger.error(f"DataAgent start failed: {e}")
            return False
    
    async def _stop_agent(self) -> bool:
        """Stop data agent operations"""
        try:
            # Cancel quality monitoring
            if hasattr(self, '_quality_monitoring_task'):
                self._quality_monitoring_task.cancel()
                try:
                    await self._quality_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("DataAgent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"DataAgent stop failed: {e}")
            return False
    
    async def _process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming messages"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message.payload)
        else:
            logger.warning(f"DataAgent: No handler for message type '{message.message_type}'")
            return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Data agent health check"""
        health_info = {
            'data_cache_status': 'valid' if self._is_cache_valid() else 'expired',
            'fighters_cache_size': len(self.fighters_cache) if self.fighters_cache is not None else 0,
            'fights_cache_size': len(self.fights_cache) if self.fights_cache is not None else 0,
            'recent_quality_score': self.quality_history[-1].overall_score if self.quality_history else 0.0,
            'data_sources_available': await self._check_data_sources_availability()
        }
        
        return health_info
    
    # === Data Loading and Caching ===
    
    async def _load_initial_data(self):
        """Load initial fighter and fight data"""
        try:
            # Load fighter data
            fighters_path = self.data_sources.get('fighters_data_path')
            if fighters_path and Path(fighters_path).exists():
                self.fighters_cache = pd.read_csv(fighters_path)
                logger.info(f"Loaded {len(self.fighters_cache)} fighter records")
            
            # Load fight data
            fights_path = self.data_sources.get('fights_data_path')
            if fights_path and Path(fights_path).exists():
                self.fights_cache = pd.read_csv(fights_path)
                logger.info(f"Loaded {len(self.fights_cache)} fight records")
            
            self.cache_expiry = datetime.now() + timedelta(seconds=self.cache_ttl)
            
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            raise
    
    def _is_cache_valid(self) -> bool:
        """Check if data cache is still valid"""
        return (
            self.cache_expiry is not None and 
            datetime.now() < self.cache_expiry and
            self.fighters_cache is not None
        )
    
    async def refresh_data_cache(self) -> bool:
        """Refresh data cache from sources"""
        try:
            async with self.track_operation("refresh_data_cache"):
                await self._load_initial_data()
                logger.info("Data cache refreshed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to refresh data cache: {e}")
            return False
    
    # === Data Validation ===
    
    async def validate_fighter_data(self, data: pd.DataFrame) -> DataValidationResult:
        """
        Comprehensive fighter data validation
        
        Args:
            data: Fighter data DataFrame
            
        Returns:
            Detailed validation result
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with self.track_operation("validate_fighter_data"):
                # Basic DataFrame validation
                df_validation = self.dataframe_validator.validate_fighter_dataframe(data)
                if not df_validation['is_valid']:
                    errors.extend(df_validation['errors'])
                
                # Fighter name validation
                if 'fighter_name' in data.columns:
                    for idx, name in enumerate(data['fighter_name']):
                        name_validation = self.fighter_name_validator.validate_fighter_name(str(name))
                        if not name_validation['is_valid']:
                            errors.append(f"Row {idx}: {name_validation['error']}")
                
                # Statistical validation
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    # Check for outliers (beyond 3 standard deviations)
                    if len(data[col].dropna()) > 0:
                        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                        outliers = z_scores > 3
                        if outliers.any():
                            outlier_count = outliers.sum()
                            warnings.append(f"Column '{col}': {outlier_count} statistical outliers detected")
                
                # Calculate quality metrics
                quality_metrics = self._calculate_data_quality_metrics(data)
                
                # Determine overall validation result
                is_valid = len(errors) == 0 and quality_metrics.overall_score >= self.min_quality_score
                validation_score = quality_metrics.overall_score
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                result = DataValidationResult(
                    is_valid=is_valid,
                    validation_score=validation_score,
                    error_count=len(errors),
                    warning_count=len(warnings),
                    errors=errors,
                    warnings=warnings,
                    metrics=quality_metrics,
                    processed_records=len(data),
                    validation_time_ms=processing_time
                )
                
                logger.info(
                    f"Fighter data validation completed: "
                    f"{len(data)} records, score={validation_score:.3f}, "
                    f"{len(errors)} errors, {len(warnings)} warnings"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Fighter data validation failed: {e}")
            raise
    
    async def validate_fight_data(self, data: pd.DataFrame) -> DataValidationResult:
        """
        Comprehensive fight data validation
        
        Args:
            data: Fight data DataFrame
            
        Returns:
            Detailed validation result
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with self.track_operation("validate_fight_data"):
                # Check required columns
                required_columns = ['fighter_a', 'fighter_b', 'winner']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    errors.append(f"Missing required columns: {missing_columns}")
                
                # Validate fighter names exist
                if self.fighters_cache is not None and 'fighter_a' in data.columns:
                    known_fighters = set(self.fighters_cache['fighter_name'].str.lower())
                    
                    for idx, row in data.iterrows():
                        fighter_a = str(row['fighter_a']).lower()
                        fighter_b = str(row['fighter_b']).lower()
                        
                        if fighter_a not in known_fighters:
                            warnings.append(f"Row {idx}: Unknown fighter A '{row['fighter_a']}'")
                        
                        if fighter_b not in known_fighters:
                            warnings.append(f"Row {idx}: Unknown fighter B '{row['fighter_b']}'")
                        
                        if fighter_a == fighter_b:
                            errors.append(f"Row {idx}: Fighter A and B are the same '{row['fighter_a']}'")
                
                # Calculate quality metrics
                quality_metrics = self._calculate_data_quality_metrics(data)
                
                is_valid = len(errors) == 0 and quality_metrics.overall_score >= self.min_quality_score
                validation_score = quality_metrics.overall_score
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                result = DataValidationResult(
                    is_valid=is_valid,
                    validation_score=validation_score,
                    error_count=len(errors),
                    warning_count=len(warnings),
                    errors=errors,
                    warnings=warnings,
                    metrics=quality_metrics,
                    processed_records=len(data),
                    validation_time_ms=processing_time
                )
                
                logger.info(
                    f"Fight data validation completed: "
                    f"{len(data)} records, score={validation_score:.3f}, "
                    f"{len(errors)} errors, {len(warnings)} warnings"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Fight data validation failed: {e}")
            raise
    
    # === Data Quality Assessment ===
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        
        # Completeness: percentage of non-null values
        completeness_score = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        # Validity: check data types and ranges
        validity_issues = 0
        total_validations = 0
        
        for col in data.columns:
            total_validations += 1
            
            if data[col].dtype in ['object', 'string']:
                # Check for empty strings
                empty_strings = (data[col] == '').sum()
                if empty_strings > 0:
                    validity_issues += empty_strings / len(data)
            
            elif np.issubdtype(data[col].dtype, np.number):
                # Check for infinite values
                inf_values = np.isinf(data[col]).sum()
                if inf_values > 0:
                    validity_issues += inf_values / len(data)
                
                # Check for negative values where they shouldn't be
                if 'age' in col.lower() or 'wins' in col.lower() or 'fights' in col.lower():
                    negative_values = (data[col] < 0).sum()
                    if negative_values > 0:
                        validity_issues += negative_values / len(data)
        
        validity_score = max(0.0, 1.0 - (validity_issues / total_validations))
        
        # Consistency: check for duplicate records
        duplicate_ratio = data.duplicated().sum() / len(data)
        consistency_score = 1.0 - duplicate_ratio
        
        # Timeliness: assume current data is timely (could be enhanced with timestamp checks)
        timeliness_score = 1.0
        
        # Accuracy: statistical measures
        accuracy_score = 1.0  # Simplified - could include statistical tests
        
        # Overall weighted score
        weights = {
            'completeness': 0.3,
            'validity': 0.3,
            'consistency': 0.2,
            'timeliness': 0.1,
            'accuracy': 0.1
        }
        
        overall_score = (
            weights['completeness'] * completeness_score +
            weights['validity'] * validity_score +
            weights['consistency'] * consistency_score +
            weights['timeliness'] * timeliness_score +
            weights['accuracy'] * accuracy_score
        )
        
        # Identify issues and recommendations
        issues = []
        recommendations = []
        
        if completeness_score < 0.9:
            issues.append(f"Low completeness: {completeness_score:.1%}")
            recommendations.append("Review data collection processes to reduce missing values")
        
        if validity_score < 0.9:
            issues.append(f"Data validity issues detected: {validity_score:.1%}")
            recommendations.append("Implement stronger data validation at ingestion")
        
        if consistency_score < 0.95:
            issues.append(f"Duplicate records found: {duplicate_ratio:.1%}")
            recommendations.append("Implement deduplication processes")
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            timeliness_score=timeliness_score,
            accuracy_score=accuracy_score,
            overall_score=overall_score,
            issues_found=issues,
            recommendations=recommendations
        )
    
    # === Feature Engineering Coordination ===
    
    async def coordinate_feature_engineering(self, 
                                           fighter_data: pd.DataFrame,
                                           fight_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Coordinate feature engineering process
        
        Args:
            fighter_data: Fighter statistics DataFrame
            fight_data: Fight results DataFrame
            
        Returns:
            Feature engineering results and metadata
        """
        try:
            async with self.track_operation("coordinate_feature_engineering"):
                # Validate input data first
                fighter_validation = await self.validate_fighter_data(fighter_data)
                fight_validation = await self.validate_fight_data(fight_data)
                
                if not fighter_validation.is_valid:
                    raise ValueError(f"Fighter data validation failed: {fighter_validation.errors}")
                
                if not fight_validation.is_valid:
                    raise ValueError(f"Fight data validation failed: {fight_validation.errors}")
                
                # This would integrate with existing feature engineering
                # For now, return validation results and quality metrics
                return {
                    'status': 'success',
                    'fighter_validation': {
                        'score': fighter_validation.validation_score,
                        'warnings': fighter_validation.warnings
                    },
                    'fight_validation': {
                        'score': fight_validation.validation_score,
                        'warnings': fight_validation.warnings
                    },
                    'data_quality': {
                        'fighter_quality': fighter_validation.metrics.overall_score,
                        'fight_quality': fight_validation.metrics.overall_score
                    },
                    'recommendations': (
                        fighter_validation.metrics.recommendations +
                        fight_validation.metrics.recommendations
                    )
                }
                
        except Exception as e:
            logger.error(f"Feature engineering coordination failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # === Message Handlers ===
    
    async def _handle_validate_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation request"""
        data_type = payload.get('data_type')
        data_path = payload.get('data_path')
        
        if not data_path or not Path(data_path).exists():
            return {'status': 'error', 'error': 'Invalid data path'}
        
        try:
            data = pd.read_csv(data_path)
            
            if data_type == 'fighters':
                result = await self.validate_fighter_data(data)
            elif data_type == 'fights':
                result = await self.validate_fight_data(data)
            else:
                return {'status': 'error', 'error': f'Unknown data type: {data_type}'}
            
            return {
                'status': 'success',
                'validation_result': {
                    'is_valid': result.is_valid,
                    'validation_score': result.validation_score,
                    'error_count': result.error_count,
                    'warning_count': result.warning_count,
                    'errors': result.errors[:10],  # Limit for response size
                    'warnings': result.warnings[:10],
                    'overall_quality_score': result.metrics.overall_score
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_data_quality(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data quality request"""
        try:
            if not self.quality_history:
                return {'status': 'error', 'error': 'No quality data available'}
            
            latest_quality = self.quality_history[-1]
            
            return {
                'status': 'success',
                'current_quality': {
                    'overall_score': latest_quality.overall_score,
                    'completeness': latest_quality.completeness_score,
                    'validity': latest_quality.validity_score,
                    'consistency': latest_quality.consistency_score,
                    'issues': latest_quality.issues_found,
                    'recommendations': latest_quality.recommendations
                },
                'history_length': len(self.quality_history)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_refresh_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data refresh request"""
        try:
            success = await self.refresh_data_cache()
            return {
                'status': 'success' if success else 'error',
                'cache_size': {
                    'fighters': len(self.fighters_cache) if self.fighters_cache is not None else 0,
                    'fights': len(self.fights_cache) if self.fights_cache is not None else 0
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_engineer_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature engineering request"""
        try:
            if not self._is_cache_valid():
                await self.refresh_data_cache()
            
            result = await self.coordinate_feature_engineering(
                self.fighters_cache, self.fights_cache
            )
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    # === Background Tasks ===
    
    async def _quality_monitoring_loop(self):
        """Background quality monitoring"""
        logger.info("DataAgent quality monitoring started")
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.quality_check_interval)
                
                if self._stop_event.is_set():
                    break
                
                # Perform quality check if data is available
                if self.fighters_cache is not None:
                    quality_metrics = self._calculate_data_quality_metrics(self.fighters_cache)
                    
                    # Store quality history
                    self.quality_history.append(quality_metrics)
                    if len(self.quality_history) > self.max_quality_history:
                        self.quality_history = self.quality_history[-self.max_quality_history:]
                    
                    # Alert if quality drops significantly
                    if quality_metrics.overall_score < self.min_quality_score:
                        await self.broadcast_message(
                            'data_quality_alert',
                            {
                                'quality_score': quality_metrics.overall_score,
                                'issues': quality_metrics.issues_found,
                                'recommendations': quality_metrics.recommendations
                            }
                        )
                
            except Exception as e:
                logger.error(f"DataAgent quality monitoring error: {e}")
        
        logger.info("DataAgent quality monitoring stopped")
    
    async def _perform_initial_quality_check(self):
        """Perform initial data quality assessment"""
        if self.fighters_cache is not None:
            quality_metrics = self._calculate_data_quality_metrics(self.fighters_cache)
            self.quality_history.append(quality_metrics)
            
            logger.info(f"Initial data quality score: {quality_metrics.overall_score:.3f}")
            
            if quality_metrics.issues_found:
                logger.warning(f"Data quality issues: {quality_metrics.issues_found}")
    
    async def _check_data_sources_availability(self) -> Dict[str, bool]:
        """Check availability of configured data sources"""
        availability = {}
        
        for source_name, source_path in self.data_sources.items():
            if source_path:
                availability[source_name] = Path(source_path).exists()
            else:
                availability[source_name] = False
        
        return availability


def create_data_agent_config(
    data_sources: Dict[str, str],
    min_quality_score: float = 0.8,
    cache_ttl: int = 1800,
    quality_check_interval: int = 3600
) -> Dict[str, Any]:
    """
    Factory function for DataAgent configuration
    
    Args:
        data_sources: Paths to data sources
        min_quality_score: Minimum acceptable quality score
        cache_ttl: Cache time-to-live in seconds
        quality_check_interval: Quality check interval in seconds
        
    Returns:
        DataAgent configuration
    """
    return {
        'data_sources': data_sources,
        'min_quality_score': min_quality_score,
        'cache_ttl': cache_ttl,
        'quality_check_interval': quality_check_interval,
        'validation': {
            'check_outliers': True,
            'outlier_threshold': 3.0,
            'require_complete_records': False
        },
        'feature_engineering': {
            'enable_differential_features': True,
            'enable_statistical_features': True,
            'enable_temporal_features': True
        }
    }