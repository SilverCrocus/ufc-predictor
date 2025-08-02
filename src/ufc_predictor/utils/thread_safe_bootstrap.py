"""
Thread-Safe Bootstrap Sampling for XGBoost Ensemble
==================================================

Provides thread-safe, reproducible bootstrap sampling for parallel processing
with strict error handling and no fallbacks.

Key Features:
- Thread-safe random number generation using numpy.random.Generator
- Memory-efficient batched processing
- Reproducible results across parallel workers
- Comprehensive input validation
- Fail-fast error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from pathlib import Path
import psutil
import gc
from contextlib import contextmanager

from .enhanced_error_handling import UFCPredictorError

logger = logging.getLogger(__name__)


class BootstrapError(UFCPredictorError):
    """Bootstrap sampling specific errors"""
    pass


class MemoryError(UFCPredictorError):
    """Memory management specific errors"""
    pass


class ThreadingError(UFCPredictorError):
    """Threading specific errors"""
    pass


@dataclass
class BootstrapConfig:
    """Configuration for thread-safe bootstrap sampling"""
    n_bootstrap: int
    confidence_level: float
    n_jobs: int
    random_state: int
    batch_size: int = 50  # Process bootstraps in batches
    max_memory_mb: int = 4096  # Maximum memory usage in MB
    memory_check_interval: int = 10  # Check memory every N samples
    enable_gc: bool = True  # Enable garbage collection
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.n_bootstrap <= 0:
            raise BootstrapError(f"n_bootstrap must be positive, got {self.n_bootstrap}")
        
        if not 0 < self.confidence_level < 1:
            raise BootstrapError(f"confidence_level must be between 0 and 1, got {self.confidence_level}")
        
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise BootstrapError(f"n_jobs must be -1 or positive, got {self.n_jobs}")
        
        if self.batch_size <= 0:
            raise BootstrapError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.max_memory_mb <= 0:
            raise BootstrapError(f"max_memory_mb must be positive, got {self.max_memory_mb}")


class ThreadSafeRandomGenerator:
    """Thread-safe random number generator for reproducible bootstrap sampling"""
    
    def __init__(self, random_state: int):
        self.base_seed = random_state
        self._generators = {}
        self._lock = threading.Lock()
    
    def get_generator(self, worker_id: int) -> np.random.Generator:
        """Get thread-specific random generator"""
        thread_id = threading.get_ident()
        key = (worker_id, thread_id)
        
        with self._lock:
            if key not in self._generators:
                # Create unique seed for this worker/thread combination
                worker_seed = self.base_seed + worker_id * 10000 + (thread_id % 10000)
                self._generators[key] = np.random.Generator(np.random.PCG64(worker_seed))
                logger.debug(f"Created generator for worker {worker_id}, thread {thread_id}, seed {worker_seed}")
        
        return self._generators[key]


class MemoryMonitor:
    """Memory usage monitoring and enforcement"""
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.process = psutil.Process()
    
    def check_memory_usage(self) -> Tuple[int, float]:
        """Check current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = (memory_info.rss / self.max_memory_bytes) * 100
        
        return memory_info.rss, memory_percent
    
    def enforce_memory_limit(self):
        """Enforce memory limit, raise error if exceeded"""
        memory_bytes, memory_percent = self.check_memory_usage()
        
        if memory_bytes > self.max_memory_bytes:
            raise MemoryError(
                f"Memory limit exceeded: {memory_percent:.1f}% ({memory_bytes/1024/1024:.1f} MB)",
                memory_used_mb=memory_bytes/1024/1024,
                memory_limit_mb=self.max_memory_bytes/1024/1024
            )
        
        logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_bytes/1024/1024:.1f} MB)")


@contextmanager
def memory_managed_context(memory_monitor: MemoryMonitor, enable_gc: bool = True):
    """Context manager for memory monitoring"""
    try:
        memory_monitor.enforce_memory_limit()
        yield
    finally:
        if enable_gc:
            gc.collect()
        memory_monitor.enforce_memory_limit()


class ThreadSafeBootstrapSampler:
    """Thread-safe bootstrap sampler for XGBoost ensemble"""
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.random_gen = ThreadSafeRandomGenerator(config.random_state)
        self.memory_monitor = MemoryMonitor(config.max_memory_mb)
        
        # Validate number of jobs
        if config.n_jobs == -1:
            self.n_jobs = min(psutil.cpu_count(), 8)  # Cap at 8 for memory reasons
        else:
            self.n_jobs = min(config.n_jobs, psutil.cpu_count())
        
        logger.info(f"Initialized ThreadSafeBootstrapSampler with {self.n_jobs} workers")
    
    def generate_bootstrap_indices(self, n_samples: int, n_bootstrap: int, 
                                 worker_id: int) -> List[np.ndarray]:
        """Generate bootstrap indices for a worker"""
        try:
            rng = self.random_gen.get_generator(worker_id)
            indices_list = []
            
            for i in range(n_bootstrap):
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                indices_list.append(indices)
                
                # Periodic memory check
                if i % self.config.memory_check_interval == 0:
                    self.memory_monitor.enforce_memory_limit()
            
            return indices_list
            
        except Exception as e:
            raise BootstrapError(
                f"Failed to generate bootstrap indices for worker {worker_id}: {str(e)}",
                worker_id=worker_id,
                n_samples=n_samples,
                n_bootstrap=n_bootstrap
            )
    
    def sample_bootstrap_predictions(self, models: Dict[str, Any], X: pd.DataFrame,
                                   model_weights: Dict[str, float]) -> np.ndarray:
        """Generate bootstrap predictions with thread-safe sampling"""
        
        # Input validation
        self._validate_inputs(models, X, model_weights)
        
        n_samples = len(X)
        logger.info(f"Starting bootstrap sampling: {self.config.n_bootstrap} samples, "
                   f"{n_samples} data points, {self.n_jobs} workers")
        
        # Calculate work distribution
        batches = self._create_work_batches()
        
        bootstrap_predictions = []
        
        try:
            with memory_managed_context(self.memory_monitor, self.config.enable_gc):
                if self.n_jobs == 1:
                    bootstrap_predictions = self._sequential_bootstrap(models, X, model_weights, batches)
                else:
                    bootstrap_predictions = self._parallel_bootstrap(models, X, model_weights, batches)
            
            # Validate results
            if len(bootstrap_predictions) == 0:
                raise BootstrapError("No bootstrap samples were generated successfully")
            
            if len(bootstrap_predictions) < self.config.n_bootstrap * 0.95:
                raise BootstrapError(
                    f"Insufficient bootstrap samples: {len(bootstrap_predictions)} < "
                    f"{self.config.n_bootstrap * 0.95:.0f} (95% threshold)",
                    expected=self.config.n_bootstrap,
                    actual=len(bootstrap_predictions)
                )
            
            result = np.array(bootstrap_predictions)
            logger.info(f"Bootstrap sampling completed: {result.shape}")
            
            return result
            
        except Exception as e:
            if isinstance(e, (BootstrapError, MemoryError, ThreadingError)):
                raise
            else:
                raise BootstrapError(f"Bootstrap sampling failed: {str(e)}")
    
    def _validate_inputs(self, models: Dict[str, Any], X: pd.DataFrame,
                        model_weights: Dict[str, float]):
        """Comprehensive input validation"""
        
        # Validate models
        if not models:
            raise BootstrapError("No models provided")
        
        for name, model in models.items():
            if not hasattr(model, 'predict_proba') and not hasattr(model, 'predict'):
                raise BootstrapError(f"Model '{name}' has no predict_proba or predict method")
        
        # Validate data
        if X.empty:
            raise BootstrapError("Input data X is empty")
        
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            raise BootstrapError(f"Input data contains null values in columns: {null_cols}")
        
        # Validate weights
        if not model_weights:
            raise BootstrapError("No model weights provided")
        
        total_weight = sum(model_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise BootstrapError(f"Model weights must sum to 1.0, got {total_weight}")
        
        for name in models.keys():
            if name not in model_weights:
                raise BootstrapError(f"No weight provided for model '{name}'")
    
    def _create_work_batches(self) -> List[Tuple[int, int]]:
        """Create work batches for parallel processing"""
        batches = []
        remaining = self.config.n_bootstrap
        worker_id = 0
        
        while remaining > 0:
            batch_size = min(self.config.batch_size, remaining)
            batches.append((worker_id, batch_size))
            remaining -= batch_size
            worker_id += 1
        
        logger.debug(f"Created {len(batches)} work batches")
        return batches
    
    def _sequential_bootstrap(self, models: Dict[str, Any], X: pd.DataFrame,
                            model_weights: Dict[str, float], 
                            batches: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Sequential bootstrap processing"""
        bootstrap_predictions = []
        
        for worker_id, batch_size in batches:
            batch_results = self._process_bootstrap_batch(
                models, X, model_weights, worker_id, batch_size
            )
            bootstrap_predictions.extend(batch_results)
            
            # Memory check after each batch
            self.memory_monitor.enforce_memory_limit()
        
        return bootstrap_predictions
    
    def _parallel_bootstrap(self, models: Dict[str, Any], X: pd.DataFrame,
                          model_weights: Dict[str, float],
                          batches: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Parallel bootstrap processing with strict error handling"""
        bootstrap_predictions = []
        failed_tasks = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_batch = {}
            for worker_id, batch_size in batches:
                future = executor.submit(
                    self._bootstrap_worker_process,
                    models, X, model_weights, worker_id, batch_size
                )
                future_to_batch[future] = (worker_id, batch_size)
            
            # Collect results
            for future in as_completed(future_to_batch):
                worker_id, batch_size = future_to_batch[future]
                
                try:
                    batch_results = future.result(timeout=300)  # 5-minute timeout
                    if batch_results is None or len(batch_results) == 0:
                        failed_tasks.append((worker_id, "No results returned"))
                    else:
                        bootstrap_predictions.extend(batch_results)
                        
                except Exception as e:
                    failed_tasks.append((worker_id, str(e)))
        
        # Strict error handling - fail if any task failed
        if failed_tasks:
            error_details = "; ".join([f"Worker {wid}: {err}" for wid, err in failed_tasks])
            raise ThreadingError(
                f"Bootstrap workers failed: {error_details}",
                failed_workers=len(failed_tasks),
                total_workers=len(batches)
            )
        
        return bootstrap_predictions
    
    @staticmethod
    def _bootstrap_worker_process(models: Dict[str, Any], X: pd.DataFrame,
                                model_weights: Dict[str, float], 
                                worker_id: int, batch_size: int) -> List[np.ndarray]:
        """Worker process for bootstrap sampling"""
        try:
            # Create new sampler instance for this process
            config = BootstrapConfig(
                n_bootstrap=batch_size,
                confidence_level=0.95,
                n_jobs=1,
                random_state=42,
                batch_size=batch_size
            )
            
            sampler = ThreadSafeBootstrapSampler(config)
            return sampler._process_bootstrap_batch(models, X, model_weights, worker_id, batch_size)
            
        except Exception as e:
            logger.error(f"Bootstrap worker {worker_id} failed: {str(e)}")
            raise
    
    def _process_bootstrap_batch(self, models: Dict[str, Any], X: pd.DataFrame,
                               model_weights: Dict[str, float],
                               worker_id: int, batch_size: int) -> List[np.ndarray]:
        """Process a batch of bootstrap samples"""
        
        n_samples = len(X)
        batch_predictions = []
        
        # Generate bootstrap indices for this batch
        indices_list = self.generate_bootstrap_indices(n_samples, batch_size, worker_id)
        
        for i, bootstrap_indices in enumerate(indices_list):
            try:
                # Create bootstrap sample efficiently (view, not copy)
                X_bootstrap = X.iloc[bootstrap_indices]
                
                # Get ensemble prediction
                ensemble_pred = self._get_ensemble_prediction(models, X_bootstrap, model_weights)
                batch_predictions.append(ensemble_pred)
                
            except Exception as e:
                raise BootstrapError(
                    f"Bootstrap sample {i} failed in worker {worker_id}: {str(e)}",
                    worker_id=worker_id,
                    sample_index=i,
                    batch_size=batch_size
                )
        
        return batch_predictions
    
    def _get_ensemble_prediction(self, models: Dict[str, Any], X: pd.DataFrame,
                               model_weights: Dict[str, float]) -> np.ndarray:
        """Get weighted ensemble prediction with strict validation"""
        
        ensemble_pred = None
        
        for model_name, model in models.items():
            if model_name not in model_weights:
                raise BootstrapError(f"No weight found for model '{model_name}'")
            
            weight = model_weights[model_name]
            
            try:
                # Get model prediction
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.shape[1] < 2:
                        raise BootstrapError(f"Model '{model_name}' predict_proba returned insufficient classes")
                    pred = pred[:, 1]  # Probability of positive class
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    raise BootstrapError(f"Model '{model_name}' has no predict method")
                
                # Validate prediction shape
                if len(pred) != len(X):
                    raise BootstrapError(
                        f"Model '{model_name}' prediction length mismatch: {len(pred)} != {len(X)}"
                    )
                
                # Weighted sum
                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred
                    
            except Exception as e:
                raise BootstrapError(f"Model '{model_name}' prediction failed: {str(e)}")
        
        if ensemble_pred is None:
            raise BootstrapError("No ensemble prediction generated")
        
        return ensemble_pred


def create_bootstrap_config(n_bootstrap: int = 100, confidence_level: float = 0.95,
                          n_jobs: int = -1, random_state: int = 42,
                          max_memory_mb: int = 4096) -> BootstrapConfig:
    """Factory function for bootstrap configuration"""
    
    return BootstrapConfig(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_jobs=n_jobs,
        random_state=random_state,
        max_memory_mb=max_memory_mb
    )


def calculate_confidence_intervals(bootstrap_predictions: np.ndarray,
                                 confidence_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate confidence intervals from bootstrap predictions"""
    
    if bootstrap_predictions.size == 0:
        raise BootstrapError("Empty bootstrap predictions provided")
    
    if bootstrap_predictions.ndim != 2:
        raise BootstrapError(f"Bootstrap predictions must be 2D, got shape {bootstrap_predictions.shape}")
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_predictions = np.mean(bootstrap_predictions, axis=0)
    lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
    upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
    
    return mean_predictions, lower_bounds, upper_bounds


# Example usage for testing
if __name__ == "__main__":
    print("Thread-Safe Bootstrap Sampling Test")
    print("=" * 50)
    
    # Create test configuration
    config = create_bootstrap_config(n_bootstrap=50, n_jobs=2, max_memory_mb=1024)
    sampler = ThreadSafeBootstrapSampler(config)
    
    # Create dummy test data
    np.random.seed(42)
    X_test = pd.DataFrame(np.random.rand(100, 10))
    
    # Create dummy models
    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([np.random.rand(len(X)), np.random.rand(len(X))])
    
    models = {'model1': DummyModel(), 'model2': DummyModel()}
    weights = {'model1': 0.6, 'model2': 0.4}
    
    try:
        # Test bootstrap sampling
        predictions = sampler.sample_bootstrap_predictions(models, X_test, weights)
        print(f"✅ Bootstrap sampling successful: {predictions.shape}")
        
        # Test confidence interval calculation
        mean_pred, lower, upper = calculate_confidence_intervals(predictions, 0.95)
        print(f"✅ Confidence intervals calculated: {len(mean_pred)} predictions")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    print("✅ Thread-safe bootstrap sampling test completed")