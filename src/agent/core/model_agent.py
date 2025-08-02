"""
Model Agent for UFC Betting System
==================================

Specialized agent for model coordination and ensemble management:
- XGBoost ensemble coordination (integrates with existing ProductionEnsembleManager)
- Random Forest model management
- Future Neural Network model integration
- Model performance monitoring and comparison
- Adaptive model weighting and selection
- Model versioning and deployment coordination
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass
import json
import joblib
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent, AgentPriority, AgentMessage
from ...production_ensemble_manager import (
    ProductionEnsembleManager, 
    EnsembleConfig, 
    PredictionResult,
    create_production_ensemble_config
)
from ...thread_safe_bootstrap import BootstrapConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Individual model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    calibration_error: float
    prediction_time_ms: float
    memory_usage_mb: float
    last_updated: datetime


@dataclass
class EnsemblePerformanceMetrics:
    """Ensemble performance metrics"""
    ensemble_accuracy: float
    ensemble_auc_roc: float
    ensemble_calibration_error: float
    model_contributions: Dict[str, float]  # Actual contribution weights
    prediction_variance: float
    confidence_interval_coverage: float
    prediction_time_ms: float
    bootstrap_samples_used: int
    last_evaluation: datetime


@dataclass
class ModelPredictionRequest:
    """Request for model prediction"""
    request_id: str
    fighter_pairs: List[Tuple[str, str]]
    feature_data: pd.DataFrame
    enable_bootstrap: bool = True
    enable_uncertainty: bool = True
    model_subset: Optional[List[str]] = None  # Use specific models only
    priority: str = "normal"  # "low", "normal", "high", "critical"


@dataclass
class ModelPredictionResponse:
    """Response from model prediction"""
    request_id: str
    predictions: List[PredictionResult]
    ensemble_metrics: EnsemblePerformanceMetrics
    data_quality_score: float
    processing_time_ms: float
    models_used: List[str]
    warnings: List[str]
    status: str  # "success", "partial", "failed"


class ModelAgent(BaseAgent):
    """
    Model coordination and ensemble management agent
    
    Responsibilities:
    - Coordinate XGBoost, Random Forest, and future Neural Network models
    - Manage ensemble predictions with uncertainty quantification
    - Monitor model performance and drift
    - Adaptive model weighting based on recent performance
    - Model versioning and hot-swapping capabilities
    - Integration with existing ProductionEnsembleManager
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelAgent
        
        Args:
            config: Model agent configuration
        """
        super().__init__(
            agent_id="model_agent",
            priority=AgentPriority.CRITICAL
        )
        
        self.config = config
        
        # Model configuration
        self.model_weights = config.get('model_weights', {})
        self.model_paths = config.get('model_paths', {})
        self.feature_columns_path = config.get('feature_columns_path')
        
        # Ensemble configuration
        self.ensemble_config = None
        self.ensemble_manager: Optional[ProductionEnsembleManager] = None
        
        # Performance tracking
        self.model_performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.ensemble_performance_history: List[EnsemblePerformanceMetrics] = []
        self.max_performance_history = 100
        
        # Model management
        self.model_versions: Dict[str, str] = {}
        self.model_load_times: Dict[str, datetime] = {}
        self.model_validation_scores: Dict[str, float] = {}
        
        # Adaptive weighting
        self.enable_adaptive_weighting = config.get('enable_adaptive_weighting', True)
        self.weight_update_interval = config.get('weight_update_interval', 24)  # hours
        self.last_weight_update = datetime.now()
        
        # Performance monitoring
        self.performance_monitor_interval = config.get('performance_monitor_interval', 3600)  # 1 hour
        self.drift_detection_threshold = config.get('drift_detection_threshold', 0.05)
        
        # Prediction queue management
        self.prediction_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent_predictions = config.get('max_concurrent_predictions', 3)
        self.prediction_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_predictions)
        
        # Register message handlers
        self.register_message_handler('predict_fights', self._handle_predict_fights)
        self.register_message_handler('get_model_status', self._handle_get_model_status)
        self.register_message_handler('update_model_weights', self._handle_update_model_weights)
        self.register_message_handler('reload_models', self._handle_reload_models)
        self.register_message_handler('get_performance_metrics', self._handle_get_performance_metrics)
        
        logger.info("ModelAgent initialized")
    
    async def _initialize_agent(self) -> bool:
        """Initialize model agent components"""
        try:
            # Create ensemble configuration
            self.ensemble_config = create_production_ensemble_config(
                model_weights=self.model_weights,
                bootstrap_samples=self.config.get('bootstrap_samples', 100),
                max_memory_mb=self.config.get('max_memory_mb', 4096),
                n_jobs=self.config.get('n_jobs', -1)
            )
            
            # Initialize ensemble manager
            self.ensemble_manager = ProductionEnsembleManager(self.ensemble_config)
            
            # Load models
            await self._load_models()
            
            # Initial performance assessment
            await self._perform_initial_model_validation()
            
            logger.info("ModelAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"ModelAgent initialization failed: {e}")
            return False
    
    async def _start_agent(self) -> bool:
        """Start model agent operations"""
        try:
            # Start prediction processing
            self._prediction_processor_task = asyncio.create_task(
                self._prediction_processing_loop()
            )
            
            # Start performance monitoring
            self._performance_monitor_task = asyncio.create_task(
                self._performance_monitoring_loop()
            )
            
            # Start adaptive weight updates
            if self.enable_adaptive_weighting:
                self._weight_update_task = asyncio.create_task(
                    self._adaptive_weight_update_loop()
                )
            
            logger.info("ModelAgent started successfully")
            return True
            
        except Exception as e:
            logger.error(f"ModelAgent start failed: {e}")
            return False
    
    async def _stop_agent(self) -> bool:
        """Stop model agent operations"""
        try:
            # Cancel background tasks
            tasks_to_cancel = []
            
            if hasattr(self, '_prediction_processor_task'):
                tasks_to_cancel.append(self._prediction_processor_task)
            
            if hasattr(self, '_performance_monitor_task'):
                tasks_to_cancel.append(self._performance_monitor_task)
            
            if hasattr(self, '_weight_update_task'):
                tasks_to_cancel.append(self._weight_update_task)
            
            for task in tasks_to_cancel:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pool
            self.prediction_executor.shutdown(wait=True)
            
            logger.info("ModelAgent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"ModelAgent stop failed: {e}")
            return False
    
    async def _process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming messages"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message.payload)
        else:
            logger.warning(f"ModelAgent: No handler for message type '{message.message_type}'")
            return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Model agent health check"""
        health_info = {
            'ensemble_manager_initialized': self.ensemble_manager is not None,
            'models_loaded': list(self.ensemble_manager.models.keys()) if self.ensemble_manager else [],
            'total_models': len(self.model_weights),
            'prediction_queue_size': self.prediction_queue.qsize(),
            'recent_ensemble_performance': self.ensemble_performance_history[-1].ensemble_accuracy if self.ensemble_performance_history else 0.0,
            'adaptive_weighting_enabled': self.enable_adaptive_weighting,
            'last_weight_update': self.last_weight_update.isoformat()
        }
        
        # Check individual model health
        if self.ensemble_manager and self.ensemble_manager.is_initialized:
            try:
                # Quick consistency check
                test_data = pd.DataFrame(np.random.rand(5, 10), columns=[f'feature_{i}' for i in range(10)])
                consistency_results = self.ensemble_manager.validate_model_consistency(test_data)
                health_info['model_consistency'] = all(
                    result.get('is_consistent', False) for result in consistency_results.values()
                )
            except Exception as e:
                health_info['model_consistency'] = False
                health_info['consistency_error'] = str(e)
        
        return health_info
    
    # === Model Management ===
    
    async def _load_models(self):
        """Load models into ensemble manager"""
        try:
            # Validate model paths exist
            missing_models = []
            for model_name, model_path in self.model_paths.items():
                if not Path(model_path).exists():
                    missing_models.append(f"{model_name}: {model_path}")
            
            if missing_models:
                raise FileNotFoundError(f"Missing model files: {missing_models}")
            
            # Load models using ensemble manager
            self.ensemble_manager.load_models(
                self.model_paths, 
                self.feature_columns_path
            )
            
            # Record load times and versions
            for model_name in self.model_paths.keys():
                self.model_load_times[model_name] = datetime.now()
                # Extract version from path if available
                model_path = Path(self.model_paths[model_name])
                self.model_versions[model_name] = model_path.stem
            
            logger.info(f"Loaded {len(self.model_paths)} models successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    async def reload_models(self, new_model_paths: Optional[Dict[str, str]] = None,
                          new_weights: Optional[Dict[str, float]] = None) -> bool:
        """
        Reload models with optional new paths and weights
        
        Args:
            new_model_paths: Optional new model paths
            new_weights: Optional new model weights
            
        Returns:
            True if reload successful
        """
        try:
            async with self.track_operation("reload_models"):
                # Update configuration if provided
                if new_model_paths:
                    self.model_paths.update(new_model_paths)
                
                if new_weights:
                    # Validate weights sum to 1
                    total_weight = sum(new_weights.values())
                    if abs(total_weight - 1.0) > 1e-6:
                        raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
                    
                    self.model_weights.update(new_weights)
                    
                    # Update ensemble config
                    self.ensemble_config.model_weights = self.model_weights
                
                # Reload models
                await self._load_models()
                
                # Validate new models
                await self._perform_initial_model_validation()
                
                logger.info("Models reloaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return False
    
    # === Prediction Processing ===
    
    async def predict_fights(self, 
                           fighter_pairs: List[Tuple[str, str]],
                           feature_data: pd.DataFrame,
                           enable_bootstrap: bool = True,
                           model_subset: Optional[List[str]] = None) -> ModelPredictionResponse:
        """
        Generate fight predictions using ensemble
        
        Args:
            fighter_pairs: List of fighter pairs
            feature_data: Feature data for predictions
            enable_bootstrap: Enable bootstrap confidence intervals
            model_subset: Use only specific models
            
        Returns:
            Prediction response with results and metadata
        """
        request_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now()
        
        try:
            async with self.track_operation("predict_fights"):
                # Validate inputs
                if len(fighter_pairs) != len(feature_data):
                    raise ValueError(f"Fighter pairs ({len(fighter_pairs)}) and feature data ({len(feature_data)}) length mismatch")
                
                # Subset models if requested
                if model_subset:
                    # This would require modifications to ensemble manager
                    # For now, use all models
                    logger.warning(f"Model subset {model_subset} requested but not yet implemented")
                
                # Run prediction in thread pool to avoid blocking
                prediction_results = await asyncio.get_event_loop().run_in_executor(
                    self.prediction_executor,
                    self._run_ensemble_prediction,
                    feature_data,
                    fighter_pairs,
                    enable_bootstrap
                )
                
                # Calculate ensemble metrics
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                ensemble_metrics = self._calculate_ensemble_metrics(
                    prediction_results, processing_time
                )
                
                # Calculate data quality score
                data_quality_score = self._calculate_prediction_data_quality(feature_data)
                
                # Create response
                response = ModelPredictionResponse(
                    request_id=request_id,
                    predictions=prediction_results,
                    ensemble_metrics=ensemble_metrics,
                    data_quality_score=data_quality_score,
                    processing_time_ms=processing_time,
                    models_used=list(self.model_weights.keys()),
                    warnings=[],
                    status="success"
                )
                
                # Store performance metrics
                self.ensemble_performance_history.append(ensemble_metrics)
                if len(self.ensemble_performance_history) > self.max_performance_history:
                    self.ensemble_performance_history = self.ensemble_performance_history[-self.max_performance_history:]
                
                logger.info(
                    f"Prediction completed: {len(prediction_results)} fights, "
                    f"{processing_time:.1f}ms, quality={data_quality_score:.3f}"
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ModelPredictionResponse(
                request_id=request_id,
                predictions=[],
                ensemble_metrics=None,
                data_quality_score=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                models_used=[],
                warnings=[str(e)],
                status="failed"
            )
    
    def _run_ensemble_prediction(self, 
                               feature_data: pd.DataFrame,
                               fighter_pairs: List[Tuple[str, str]],
                               enable_bootstrap: bool) -> List[PredictionResult]:
        """Run ensemble prediction in thread pool"""
        if not self.ensemble_manager or not self.ensemble_manager.is_initialized:
            raise RuntimeError("Ensemble manager not initialized")
        
        return self.ensemble_manager.predict_fights(
            feature_data, fighter_pairs, enable_bootstrap
        )
    
    def _calculate_ensemble_metrics(self, 
                                  prediction_results: List[PredictionResult],
                                  processing_time_ms: float) -> EnsemblePerformanceMetrics:
        """Calculate ensemble performance metrics from predictions"""
        
        if not prediction_results:
            return EnsemblePerformanceMetrics(
                ensemble_accuracy=0.0,
                ensemble_auc_roc=0.0,
                ensemble_calibration_error=0.0,
                model_contributions=self.model_weights.copy(),
                prediction_variance=0.0,
                confidence_interval_coverage=0.0,
                prediction_time_ms=processing_time_ms,
                bootstrap_samples_used=self.ensemble_config.bootstrap_samples,
                last_evaluation=datetime.now()
            )
        
        # Calculate prediction variance
        probabilities = [result.ensemble_probability for result in prediction_results]
        prediction_variance = float(np.var(probabilities))
        
        # Calculate confidence interval coverage (simplified)
        ci_widths = [
            result.confidence_interval[1] - result.confidence_interval[0]
            for result in prediction_results
        ]
        avg_ci_width = np.mean(ci_widths)
        confidence_interval_coverage = 1.0 - (avg_ci_width / 2.0)  # Simplified metric
        
        return EnsemblePerformanceMetrics(
            ensemble_accuracy=0.0,  # Would need ground truth for this
            ensemble_auc_roc=0.0,   # Would need ground truth for this
            ensemble_calibration_error=0.0,  # Would need ground truth for this
            model_contributions=self.model_weights.copy(),
            prediction_variance=prediction_variance,
            confidence_interval_coverage=confidence_interval_coverage,
            prediction_time_ms=processing_time_ms,
            bootstrap_samples_used=self.ensemble_config.bootstrap_samples,
            last_evaluation=datetime.now()
        )
    
    def _calculate_prediction_data_quality(self, feature_data: pd.DataFrame) -> float:
        """Calculate data quality score for prediction input"""
        # Completeness
        completeness = 1.0 - (feature_data.isnull().sum().sum() / (len(feature_data) * len(feature_data.columns)))
        
        # Check for infinite/invalid values
        numeric_data = feature_data.select_dtypes(include=[np.number])
        invalid_ratio = (np.isinf(numeric_data) | np.isnan(numeric_data)).sum().sum() / (len(numeric_data) * len(numeric_data.columns))
        validity = 1.0 - invalid_ratio
        
        # Overall quality (weighted average)
        quality_score = 0.7 * completeness + 0.3 * validity
        
        return float(quality_score)
    
    # === Performance Monitoring ===
    
    async def _perform_initial_model_validation(self):
        """Perform initial model validation and performance assessment"""
        try:
            if not self.ensemble_manager or not self.ensemble_manager.is_initialized:
                logger.warning("Cannot validate models: ensemble manager not initialized")
                return
            
            # Create test data for validation
            test_data = pd.DataFrame(
                np.random.rand(10, len(self.ensemble_manager.feature_columns) if self.ensemble_manager.feature_columns else 20),
                columns=self.ensemble_manager.feature_columns or [f'feature_{i}' for i in range(20)]
            )
            
            # Test model consistency
            consistency_results = self.ensemble_manager.validate_model_consistency(test_data)
            
            for model_name, result in consistency_results.items():
                is_consistent = result.get('is_consistent', False)
                self.model_validation_scores[model_name] = 1.0 if is_consistent else 0.0
                
                if not is_consistent:
                    logger.warning(f"Model '{model_name}' failed consistency check")
            
            logger.info("Initial model validation completed")
            
        except Exception as e:
            logger.error(f"Initial model validation failed: {e}")
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        logger.info("ModelAgent performance monitoring started")
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.performance_monitor_interval)
                
                if self._stop_event.is_set():
                    break
                
                # Monitor ensemble manager performance
                if self.ensemble_manager:
                    performance_metrics = self.ensemble_manager.get_performance_metrics()
                    
                    # Check for performance degradation
                    if performance_metrics['error_rate'] > 0.1:  # 10% error rate threshold
                        await self.broadcast_message(
                            'model_performance_alert',
                            {
                                'error_rate': performance_metrics['error_rate'],
                                'avg_processing_time': performance_metrics['avg_processing_time_ms'],
                                'total_errors': performance_metrics['total_errors']
                            }
                        )
                
            except Exception as e:
                logger.error(f"ModelAgent performance monitoring error: {e}")
        
        logger.info("ModelAgent performance monitoring stopped")
    
    async def _adaptive_weight_update_loop(self):
        """Background adaptive weight updates based on performance"""
        logger.info("ModelAgent adaptive weight updates started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for update interval
                update_interval = self.weight_update_interval * 3600  # Convert hours to seconds
                await asyncio.sleep(update_interval)
                
                if self._stop_event.is_set():
                    break
                
                # Update weights based on recent performance
                await self._update_adaptive_weights()
                
            except Exception as e:
                logger.error(f"ModelAgent adaptive weight update error: {e}")
        
        logger.info("ModelAgent adaptive weight updates stopped")
    
    async def _update_adaptive_weights(self):
        """Update model weights based on recent performance"""
        try:
            # This is a simplified implementation
            # In practice, would use validation data and performance metrics
            
            if not self.model_performance_history:
                logger.info("No performance history available for weight updates")
                return
            
            # For now, just log the intent
            logger.info(f"Adaptive weight update triggered (weights remain: {self.model_weights})")
            self.last_weight_update = datetime.now()
            
            # Future implementation would:
            # 1. Evaluate recent model performance on validation set
            # 2. Calculate optimal weights using performance metrics
            # 3. Update ensemble configuration
            # 4. Notify other agents of weight changes
            
        except Exception as e:
            logger.error(f"Adaptive weight update failed: {e}")
    
    # === Message Handlers ===
    
    async def _handle_predict_fights(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fight prediction request"""
        try:
            # Extract request parameters
            fighter_pairs = payload.get('fighter_pairs', [])
            feature_data_path = payload.get('feature_data_path')
            enable_bootstrap = payload.get('enable_bootstrap', True)
            model_subset = payload.get('model_subset')
            
            # Load feature data
            if feature_data_path and Path(feature_data_path).exists():
                feature_data = pd.read_csv(feature_data_path)
            else:
                return {'status': 'error', 'error': 'Invalid feature data path'}
            
            # Generate predictions
            response = await self.predict_fights(
                fighter_pairs, feature_data, enable_bootstrap, model_subset
            )
            
            # Return simplified response for message
            return {
                'status': response.status,
                'request_id': response.request_id,
                'prediction_count': len(response.predictions),
                'processing_time_ms': response.processing_time_ms,
                'data_quality_score': response.data_quality_score,
                'ensemble_accuracy': response.ensemble_metrics.ensemble_accuracy if response.ensemble_metrics else 0.0
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_model_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model status request"""
        try:
            return {
                'status': 'success',
                'models_loaded': list(self.model_weights.keys()),
                'ensemble_initialized': self.ensemble_manager is not None and self.ensemble_manager.is_initialized,
                'model_weights': self.model_weights.copy(),
                'model_versions': self.model_versions.copy(),
                'model_load_times': {k: v.isoformat() for k, v in self.model_load_times.items()},
                'validation_scores': self.model_validation_scores.copy(),
                'adaptive_weighting_enabled': self.enable_adaptive_weighting,
                'last_weight_update': self.last_weight_update.isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_update_model_weights(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model weight update request"""
        try:
            new_weights = payload.get('weights', {})
            
            # Validate weights
            if not new_weights:
                return {'status': 'error', 'error': 'No weights provided'}
            
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                return {'status': 'error', 'error': f'Weights must sum to 1.0, got {total_weight}'}
            
            # Update weights
            self.model_weights.update(new_weights)
            self.ensemble_config.model_weights = self.model_weights
            
            logger.info(f"Model weights updated: {self.model_weights}")
            
            return {
                'status': 'success',
                'updated_weights': self.model_weights.copy()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_reload_models(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model reload request"""
        try:
            new_model_paths = payload.get('model_paths')
            new_weights = payload.get('weights')
            
            success = await self.reload_models(new_model_paths, new_weights)
            
            return {
                'status': 'success' if success else 'error',
                'models_loaded': list(self.model_weights.keys()) if success else [],
                'reload_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_get_performance_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance metrics request"""
        try:
            # Get ensemble manager metrics
            ensemble_metrics = {}
            if self.ensemble_manager:
                ensemble_metrics = self.ensemble_manager.get_performance_metrics()
            
            # Get recent ensemble performance
            recent_performance = None
            if self.ensemble_performance_history:
                recent = self.ensemble_performance_history[-1]
                recent_performance = {
                    'ensemble_accuracy': recent.ensemble_accuracy,
                    'prediction_variance': recent.prediction_variance,
                    'confidence_interval_coverage': recent.confidence_interval_coverage,
                    'last_evaluation': recent.last_evaluation.isoformat()
                }
            
            return {
                'status': 'success',
                'ensemble_manager_metrics': ensemble_metrics,
                'recent_ensemble_performance': recent_performance,
                'performance_history_length': len(self.ensemble_performance_history),
                'model_validation_scores': self.model_validation_scores.copy()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    # === Queue Processing ===
    
    async def _prediction_processing_loop(self):
        """Process prediction requests from queue"""
        logger.info("ModelAgent prediction processing started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for prediction request
                request = await asyncio.wait_for(
                    self.prediction_queue.get(), timeout=1.0
                )
                
                # Process request
                await self._process_prediction_request(request)
                
            except asyncio.TimeoutError:
                continue  # Normal timeout
            except Exception as e:
                logger.error(f"ModelAgent prediction processing error: {e}")
        
        logger.info("ModelAgent prediction processing stopped")
    
    async def _process_prediction_request(self, request: ModelPredictionRequest):
        """Process individual prediction request"""
        try:
            response = await self.predict_fights(
                request.fighter_pairs,
                request.feature_data,
                request.enable_bootstrap,
                request.model_subset
            )
            
            # Send response back (would integrate with message bus)
            logger.debug(f"Processed prediction request {request.request_id}: {response.status}")
            
        except Exception as e:
            logger.error(f"Failed to process prediction request {request.request_id}: {e}")


def create_model_agent_config(
    model_weights: Dict[str, float],
    model_paths: Dict[str, str],
    feature_columns_path: str,
    bootstrap_samples: int = 100,
    max_memory_mb: int = 4096,
    enable_adaptive_weighting: bool = True
) -> Dict[str, Any]:
    """
    Factory function for ModelAgent configuration
    
    Args:
        model_weights: Model ensemble weights
        model_paths: Paths to model files
        feature_columns_path: Path to feature columns JSON
        bootstrap_samples: Number of bootstrap samples
        max_memory_mb: Maximum memory usage
        enable_adaptive_weighting: Enable adaptive weight updates
        
    Returns:
        ModelAgent configuration
    """
    return {
        'model_weights': model_weights,
        'model_paths': model_paths,
        'feature_columns_path': feature_columns_path,
        'bootstrap_samples': bootstrap_samples,
        'max_memory_mb': max_memory_mb,
        'n_jobs': -1,
        'enable_adaptive_weighting': enable_adaptive_weighting,
        'weight_update_interval': 24,  # hours
        'performance_monitor_interval': 3600,  # seconds
        'drift_detection_threshold': 0.05,
        'max_concurrent_predictions': 3
    }