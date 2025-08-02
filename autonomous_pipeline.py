#!/usr/bin/env python3
"""
Autonomous UFC Predictor Pipeline
================================

This script orchestrates the complete autonomous workflow:
1. WebScraperAgent - Collects latest UFC data from multiple sources
2. ModelAgent - Trains ML models using the freshly scraped data

The pipeline runs end-to-end without user intervention, automatically
detecting completion and chaining the agents together.

Usage:
    python autonomous_pipeline.py [--debug] [--tune] [--max-retries 3]
"""

import asyncio
import sys
import argparse
import logging
import subprocess
import json
import os
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import what's available - fallback to direct model training if agents aren't available
try:
    from src.agent.core.data_agent import DataAgent, create_data_agent_config
    from src.agent.core.model_agent import ModelAgent, create_model_agent_config
    from src.agent.core.base_agent import AgentMessage
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Agent system not available ({e}), using direct integration mode")
    AGENTS_AVAILABLE = False

from config.model_config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the autonomous pipeline"""
    debug_mode: bool = False
    tune_hyperparameters: bool = True
    max_retries: int = 3
    scraping_timeout: int = 1800  # 30 minutes
    training_timeout: int = 3600   # 60 minutes
    data_validation_threshold: float = 0.8
    webscraper_script: str = "webscraper/fast_scraping.py"  # Use stable scraper first
    backup_scrapers: List[str] = None
    # Process termination settings
    graceful_termination_timeout: float = 10.0  # Time to wait for graceful shutdown
    heartbeat_interval: float = 60.0  # Heartbeat logging interval
    
    def __post_init__(self):
        if self.backup_scrapers is None:
            self.backup_scrapers = [
                "webscraper/scraping.py",            # Most reliable synchronous scraper  
                "webscraper/optimized_scraping.py"   # Move problematic async scraper to last
            ]


@dataclass
class PipelineResult:
    """Result of the autonomous pipeline execution"""
    success: bool
    scraping_completed: bool
    training_completed: bool
    data_quality_score: float
    model_accuracy: Dict[str, float]
    execution_time_seconds: float
    errors: List[str]
    warnings: List[str]
    artifacts_created: List[str]


class WebScraperAgent:
    """
    Agent wrapper for UFC data web scraping
    
    Orchestrates multiple scraping tools to collect:
    - Fighter statistics and profiles
    - Fight results and odds data
    - Event information
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.WebScraperAgent")
        
        # Circuit breaker for failure detection
        self.scraper_failures = {}  # Track failures per scraper
        self.failure_threshold = 2  # Fail after 2 consecutive failures
        self.success_reset_threshold = 1  # Reset failure count after 1 success
    
    def _is_scraper_circuit_open(self, scraper_script: str) -> bool:
        """Check if circuit breaker is open (scraper should be skipped)"""
        failure_count = self.scraper_failures.get(scraper_script, 0)
        return failure_count >= self.failure_threshold
    
    def _record_scraper_failure(self, scraper_script: str):
        """Record a scraper failure and update circuit breaker state"""
        self.scraper_failures[scraper_script] = self.scraper_failures.get(scraper_script, 0) + 1
        failure_count = self.scraper_failures[scraper_script]
        
        if failure_count >= self.failure_threshold:
            self.logger.warning(f"🚫 Circuit breaker OPEN for {scraper_script} ({failure_count} failures)")
        else:
            self.logger.info(f"⚠️ Failure #{failure_count} recorded for {scraper_script}")
    
    def _record_scraper_success(self, scraper_script: str):
        """Record a scraper success and reset circuit breaker if needed"""
        if scraper_script in self.scraper_failures:
            old_count = self.scraper_failures[scraper_script]
            
            # Full reset if circuit was open, partial reset otherwise
            if old_count >= self.failure_threshold:
                self.scraper_failures[scraper_script] = 0  # Full reset for opened circuits
                self.logger.info(f"✅ Circuit breaker FULLY RESET for {scraper_script} (failures: {old_count} → 0)")
            elif old_count > 0:
                self.scraper_failures[scraper_script] = max(0, old_count - self.success_reset_threshold)
                new_count = self.scraper_failures[scraper_script]
                self.logger.info(f"✅ Circuit breaker PARTIAL RESET for {scraper_script} (failures: {old_count} → {new_count})")
        
    async def execute_scraping(self) -> Dict[str, Any]:
        """
        Execute web scraping with robust subprocess management and timeout handling
        
        Returns:
            Dictionary with scraping results and metadata
        """
        self.logger.info("🕷️ Starting web scraping phase...")
        
        scrapers_to_try = [self.config.webscraper_script] + self.config.backup_scrapers
        scraping_result = None
        
        for attempt, scraper_script in enumerate(scrapers_to_try, 1):
            if not Path(scraper_script).exists():
                self.logger.warning(f"Scraper not found: {scraper_script}")
                continue
            
            # Check circuit breaker
            if self._is_scraper_circuit_open(scraper_script):
                self.logger.warning(f"🚫 Skipping {scraper_script} - circuit breaker is OPEN")
                continue
                
            self.logger.info(f"Attempt {attempt}: Running {scraper_script}")
            
            process = None
            try:
                # Create subprocess with proper signal handling
                process = await asyncio.create_subprocess_exec(
                    'python3', scraper_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # Enable proper signal handling
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
                
                # Verify process started successfully
                if process.returncode is not None:
                    self.logger.error(f"Process failed to start properly, return code: {process.returncode}")
                    continue
                
                try:
                    # Use enhanced monitoring with heartbeat logging
                    result = await self._monitor_subprocess_with_heartbeat(
                        process, 
                        timeout=self.config.scraping_timeout,
                        heartbeat_interval=self.config.heartbeat_interval
                    )
                    
                    if result is None:
                        self._record_scraper_failure(scraper_script)  # Record timeout as failure
                        continue  # Timeout occurred, try next scraper
                    
                    stdout, stderr = result
                    
                    if process.returncode == 0:
                        self.logger.info(f"✅ Scraping successful with {scraper_script}")
                        self._record_scraper_success(scraper_script)  # Record success
                        scraping_result = {
                            'status': 'success',
                            'scraper_used': scraper_script,
                            'attempt': attempt,
                            'stdout': stdout.decode('utf-8'),
                            'data_directory': self._detect_latest_scrape_directory()
                        }
                        break
                    else:
                        error_msg = stderr.decode('utf-8') if stderr else "No error output"
                        self.logger.error(f"Scraper failed with return code {process.returncode}: {error_msg}")
                        self._record_scraper_failure(scraper_script)  # Record failure
                
                except Exception as process_error:
                    self.logger.error(f"Process communication error: {process_error}")
                    self._record_scraper_failure(scraper_script)  # Record exception as failure
                    if process:
                        await self._terminate_process_group(process)
                    
            except Exception as e:
                self.logger.error(f"Scraping attempt {attempt} failed: {e}")
                self._record_scraper_failure(scraper_script)  # Record exception as failure
                if process:
                    await self._terminate_process_group(process)
                
        if not scraping_result:
            return {
                'status': 'failed',
                'error': 'All scraping attempts failed',
                'attempts': len(scrapers_to_try)
            }
            
        # Validate scraped data
        validation_result = await self._validate_scraped_data(
            scraping_result['data_directory']
        )
        scraping_result.update(validation_result)
        
        return scraping_result
    
    async def _terminate_process_group(self, process: asyncio.subprocess.Process, timeout: float = None):
        """
        Robustly terminate a subprocess and all its children
        
        Args:
            process: The subprocess to terminate
            timeout: Maximum time to wait for graceful termination (uses config default if None)
        """
        if not process or process.returncode is not None:
            return  # Process already terminated
        
        if timeout is None:
            timeout = self.config.graceful_termination_timeout
            
        self.logger.info(f"🔄 Attempting graceful termination of PID {process.pid}")
        
        try:
            # Step 1: Try graceful termination (SIGTERM)
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                try:
                    # Terminate the entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    self.logger.info("📤 Sent SIGTERM to process group")
                except (OSError, ProcessLookupError):
                    # Fallback to single process termination
                    process.terminate()
                    self.logger.info("📤 Sent SIGTERM to process")
            else:
                process.terminate()
                self.logger.info("📤 Sent SIGTERM to process")
            
            # Step 2: Wait for graceful termination
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
                self.logger.info("✅ Process terminated gracefully")
                return
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️ Process didn't terminate gracefully within {timeout}s")
            
            # Step 3: Force termination (SIGKILL)
            self.logger.info("🔨 Force terminating process with SIGKILL")
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                try:
                    # Kill the entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    self.logger.info("💀 Sent SIGKILL to process group")
                except (OSError, ProcessLookupError):
                    # Fallback to single process kill
                    process.kill()
                    self.logger.info("💀 Sent SIGKILL to process")
            else:
                process.kill()
                self.logger.info("💀 Sent SIGKILL to process")
            
            # Step 4: Final wait with timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                self.logger.info("✅ Process force terminated successfully")
            except asyncio.TimeoutError:
                self.logger.error("❌ Process still running after SIGKILL - may be zombie")
            
        except Exception as e:
            self.logger.error(f"❌ Error terminating process: {e}")
        
        # Step 5: Cleanup any remaining resources
        try:
            # Close stdin/stdout/stderr if they exist
            if hasattr(process, 'stdin') and process.stdin:
                process.stdin.close()
            if hasattr(process, 'stdout') and process.stdout:
                process.stdout.close()
            if hasattr(process, 'stderr') and process.stderr:
                process.stderr.close()
        except Exception as e:
            self.logger.warning(f"⚠️ Error closing process streams: {e}")
    
    async def _monitor_subprocess_with_heartbeat(self, process: asyncio.subprocess.Process, 
                                               timeout: float, heartbeat_interval: float = 30.0):
        """
        Monitor a subprocess with heartbeat logging and robust timeout handling
        
        Args:
            process: The subprocess to monitor
            timeout: Total timeout in seconds
            heartbeat_interval: Interval between heartbeat logs
            
        Returns:
            Tuple of (stdout, stderr) if successful, None if timeout/error
        """
        start_time = time.time()
        last_heartbeat = start_time
        
        # Create a task for process communication
        comm_task = asyncio.create_task(process.communicate())
        
        try:
            while not comm_task.done():
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if we've exceeded the timeout
                if elapsed >= timeout:
                    self.logger.error(f"⏰ Process timeout after {elapsed:.1f}s")
                    comm_task.cancel()
                    await self._terminate_process_group(process)
                    return None
                
                # Log heartbeat if interval has passed
                if current_time - last_heartbeat >= heartbeat_interval:
                    remaining = timeout - elapsed
                    
                    # Check if process is still alive
                    if process.returncode is None:
                        # Process still running - check if it's responsive
                        try:
                            # Try to get process status (non-blocking)
                            proc_status = "running"
                            if hasattr(os, 'kill'):
                                try:
                                    os.kill(process.pid, 0)  # Signal 0 just checks if process exists
                                except (OSError, ProcessLookupError):
                                    proc_status = "zombie/dead"
                        except:
                            proc_status = "unknown"
                        
                        self.logger.info(f"💓 Process heartbeat: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining (status: {proc_status})")
                    else:
                        self.logger.info(f"💓 Process heartbeat: {elapsed:.1f}s elapsed, process exited with code {process.returncode}")
                    
                    last_heartbeat = current_time
                
                # Wait a short time before checking again
                try:
                    await asyncio.wait_for(asyncio.shield(comm_task), timeout=1.0)
                    break  # Process completed
                except asyncio.TimeoutError:
                    continue  # Keep monitoring
            
            # Process completed normally
            stdout, stderr = await comm_task
            elapsed = time.time() - start_time
            self.logger.info(f"✅ Process completed in {elapsed:.1f}s")
            return stdout, stderr
            
        except asyncio.CancelledError:
            self.logger.warning("🛑 Process monitoring cancelled")
            await self._terminate_process_group(process)
            return None
        except Exception as e:
            self.logger.error(f"❌ Process monitoring error: {e}")
            await self._terminate_process_group(process)
            return None
    
    def _detect_latest_scrape_directory(self) -> Optional[str]:
        """Detect the most recently created scrape directory"""
        data_dir = Path('data')
        if not data_dir.exists():
            return None
            
        scrape_dirs = [d for d in data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('scrape_')]
        
        if not scrape_dirs:
            return None
            
        latest_dir = max(scrape_dirs, key=lambda x: x.stat().st_mtime)
        return str(latest_dir)
    
    async def _validate_scraped_data(self, data_directory: str) -> Dict[str, Any]:
        """Validate the quality and completeness of scraped data"""
        if not data_directory or not Path(data_directory).exists():
            return {
                'data_valid': False,
                'validation_error': 'Data directory not found'
            }
        
        data_dir = Path(data_directory)
        
        # Check for required files
        fighters_files = list(data_dir.glob('ufc_fighters_raw_*.csv'))
        fights_files = list(data_dir.glob('ufc_fights_*.csv'))
        
        if not fighters_files or not fights_files:
            return {
                'data_valid': False,
                'validation_error': 'Required data files not found',
                'fighters_files': len(fighters_files),
                'fights_files': len(fights_files)
            }
        
        # Basic data quality checks
        try:
            import pandas as pd
            
            latest_fighters = max(fighters_files, key=lambda x: x.stat().st_mtime)
            latest_fights = max(fights_files, key=lambda x: x.stat().st_mtime)
            
            fighters_df = pd.read_csv(latest_fighters)
            fights_df = pd.read_csv(latest_fights)
            
            # Quality metrics
            fighters_completeness = 1.0 - (fighters_df.isnull().sum().sum() / 
                                         (len(fighters_df) * len(fighters_df.columns)))
            fights_completeness = 1.0 - (fights_df.isnull().sum().sum() / 
                                       (len(fights_df) * len(fights_df.columns)))
            
            overall_quality = (fighters_completeness + fights_completeness) / 2
            
            return {
                'data_valid': overall_quality >= self.config.data_validation_threshold,
                'data_quality_score': overall_quality,
                'fighters_count': len(fighters_df),
                'fights_count': len(fights_df),
                'fighters_completeness': fighters_completeness,
                'fights_completeness': fights_completeness,
                'latest_fighters_file': str(latest_fighters),
                'latest_fights_file': str(latest_fights)
            }
            
        except Exception as e:
            return {
                'data_valid': False,
                'validation_error': f'Data validation failed: {e}'
            }


class AutonomousPipeline:
    """
    Main orchestrator for the autonomous UFC prediction pipeline
    
    Coordinates WebScraperAgent → DataAgent → ModelAgent workflow
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AutonomousPipeline")
        
        # Initialize agents
        self.webscraper_agent = WebScraperAgent(config)
        self.data_agent: Optional[DataAgent] = None
        self.model_agent: Optional[ModelAgent] = None
        
        # Pipeline state
        self.start_time = datetime.now()
        self.execution_log: List[str] = []
        self.artifacts_created: List[str] = []
        
    async def execute_pipeline(self) -> PipelineResult:
        """
        Execute the complete autonomous pipeline
        
        Returns:
            Comprehensive pipeline execution result
        """
        self.logger.info("🚀 Starting Autonomous UFC Predictor Pipeline")
        self._log_execution("Pipeline started")
        
        errors = []
        warnings = []
        scraping_completed = False
        training_completed = False
        data_quality_score = 0.0
        model_accuracy = {}
        
        try:
            # Phase 1: Web Scraping
            self.logger.info("=" * 60)
            self.logger.info("PHASE 1: WEB SCRAPING")
            self.logger.info("=" * 60)
            
            scraping_result = await self.webscraper_agent.execute_scraping()
            
            if scraping_result['status'] != 'success':
                errors.append(f"Scraping failed: {scraping_result.get('error', 'Unknown error')}")
                return self._create_result(
                    success=False, scraping_completed=False, training_completed=False,
                    data_quality_score=0.0, model_accuracy={}, errors=errors, warnings=warnings
                )
            
            scraping_completed = True
            data_quality_score = scraping_result.get('data_quality_score', 0.0)
            self._log_execution(f"Scraping completed with quality score: {data_quality_score:.3f}")
            
            # Phase 2: Data Validation and Processing
            self.logger.info("=" * 60)
            self.logger.info("PHASE 2: DATA VALIDATION")
            self.logger.info("=" * 60)
            
            data_validation_result = await self._initialize_and_validate_data(scraping_result)
            
            if not data_validation_result['success']:
                errors.extend(data_validation_result.get('errors', []))
                warnings.extend(data_validation_result.get('warnings', []))
                
                if not data_validation_result.get('can_proceed', False):
                    return self._create_result(
                        success=False, scraping_completed=True, training_completed=False,
                        data_quality_score=data_quality_score, model_accuracy={}, 
                        errors=errors, warnings=warnings
                    )
            
            # Phase 3: Model Training
            self.logger.info("=" * 60)
            self.logger.info("PHASE 3: MODEL TRAINING")
            self.logger.info("=" * 60)
            
            training_result = await self._execute_model_training()
            
            if training_result['status'] == 'success':
                training_completed = True
                model_accuracy = training_result.get('model_accuracy', {})
                self.artifacts_created.extend(training_result.get('artifacts', []))
                self._log_execution("Model training completed successfully")
            else:
                errors.extend(training_result.get('errors', []))
                warnings.extend(training_result.get('warnings', []))
            
            # Pipeline completion
            success = scraping_completed and training_completed and len(errors) == 0
            
            return self._create_result(
                success=success,
                scraping_completed=scraping_completed,
                training_completed=training_completed,
                data_quality_score=data_quality_score,
                model_accuracy=model_accuracy,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            errors.append(f"Pipeline execution error: {e}")
            
            return self._create_result(
                success=False, scraping_completed=scraping_completed,
                training_completed=training_completed, data_quality_score=data_quality_score,
                model_accuracy=model_accuracy, errors=errors, warnings=warnings
            )
        
        finally:
            # Cleanup
            await self._cleanup_pipeline()
    
    async def _initialize_and_validate_data(self, scraping_result: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize DataAgent and validate scraped data"""
        try:
            data_directory = scraping_result.get('data_directory')
            if not data_directory:
                return {
                    'success': False,
                    'can_proceed': False,
                    'errors': ['No data directory found from scraping']
                }
            
            if AGENTS_AVAILABLE:
                # Use DataAgent for validation
                data_sources = {
                    'fighters_data_path': scraping_result.get('latest_fighters_file'),
                    'fights_data_path': scraping_result.get('latest_fights_file')
                }
                
                data_agent_config = create_data_agent_config(
                    data_sources=data_sources,
                    min_quality_score=self.config.data_validation_threshold
                )
                
                # Initialize DataAgent
                self.data_agent = DataAgent(data_agent_config)
                await self.data_agent.initialize()
                await self.data_agent.start()
                
                # Validate data quality
                validation_message = AgentMessage(
                    sender_id='pipeline',
                    recipient_id='data_agent',
                    message_type='validate_data',
                    payload={
                        'data_type': 'fighters',
                        'data_path': data_sources['fighters_data_path']
                    }
                )
                
                validation_result = await self.data_agent._process_message(validation_message)
                
                if validation_result and validation_result.get('status') == 'success':
                    validation_info = validation_result['validation_result']
                    
                    return {
                        'success': validation_info['is_valid'],
                        'can_proceed': validation_info['validation_score'] >= self.config.data_validation_threshold,
                        'validation_score': validation_info['validation_score'],
                        'errors': validation_info.get('errors', []),
                        'warnings': validation_info.get('warnings', [])
                    }
                else:
                    return {
                        'success': False,
                        'can_proceed': False,
                        'errors': ['Data validation failed']
                    }
            else:
                # Fallback: Basic data validation without agents
                self.logger.info("Using basic data validation (agents not available)")
                
                # Use the existing validation from scraping_result
                data_valid = scraping_result.get('data_valid', False)
                quality_score = scraping_result.get('data_quality_score', 0.0)
                
                return {
                    'success': data_valid,
                    'can_proceed': quality_score >= self.config.data_validation_threshold,
                    'validation_score': quality_score,
                    'errors': [scraping_result.get('validation_error')] if scraping_result.get('validation_error') else [],
                    'warnings': []
                }
                
        except Exception as e:
            return {
                'success': False,
                'can_proceed': False,
                'errors': [f'Data validation error: {e}']
            }
    
    async def _execute_model_training(self) -> Dict[str, Any]:
        """Execute model training using main.py pipeline"""
        try:
            self.logger.info("Executing model training pipeline...")
            
            # Build command
            cmd = ['python3', 'main.py', '--mode', 'pipeline']
            if self.config.tune_hyperparameters:
                cmd.append('--tune')
            
            # Execute training pipeline with proper signal handling
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent,
                # Enable proper signal handling for training
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.training_timeout
                )
                
                if process.returncode == 0:
                    # Parse training output for accuracy metrics
                    stdout_str = stdout.decode('utf-8')
                    model_accuracy = self._parse_training_output(stdout_str)
                    
                    # Detect created artifacts
                    artifacts = self._detect_training_artifacts()
                    
                    return {
                        'status': 'success',
                        'model_accuracy': model_accuracy,
                        'artifacts': artifacts,
                        'stdout': stdout_str
                    }
                else:
                    stderr_str = stderr.decode('utf-8') if stderr else "No error output"
                    return {
                        'status': 'failed',
                        'errors': [f'Training failed with return code {process.returncode}: {stderr_str}'],
                        'stdout': stdout.decode('utf-8') if stdout else ""
                    }
                    
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ Training timeout after {self.config.training_timeout}s - forcefully terminating")
                await self._terminate_process_group(process)
                return {
                    'status': 'failed',
                    'errors': ['Training timeout - process terminated']
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [f'Training execution error: {e}']
            }
    
    def _parse_training_output(self, output: str) -> Dict[str, float]:
        """Parse model accuracy from training output"""
        accuracy_dict = {}
        
        # Look for accuracy patterns in output
        import re
        
        # Winner model accuracy
        winner_match = re.search(r'Winner Model.*?Accuracy.*?(\d+\.\d+)', output)
        if winner_match:
            accuracy_dict['winner_model'] = float(winner_match.group(1))
        
        # Method model accuracy
        method_match = re.search(r'Method Model.*?Accuracy.*?(\d+\.\d+)', output)
        if method_match:
            accuracy_dict['method_model'] = float(method_match.group(1))
        
        return accuracy_dict
    
    def _detect_training_artifacts(self) -> List[str]:
        """Detect files created during training"""
        artifacts = []
        model_dir = Path('model')
        
        # Look for recently created files
        cutoff_time = self.start_time - timedelta(minutes=5)
        
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > cutoff_time:
                    artifacts.append(str(file_path))
        
        return artifacts
    
    def _create_result(self, success: bool, scraping_completed: bool, 
                      training_completed: bool, data_quality_score: float,
                      model_accuracy: Dict[str, float], errors: List[str], 
                      warnings: List[str]) -> PipelineResult:
        """Create pipeline result object"""
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        return PipelineResult(
            success=success,
            scraping_completed=scraping_completed,
            training_completed=training_completed,
            data_quality_score=data_quality_score,
            model_accuracy=model_accuracy,
            execution_time_seconds=execution_time,
            errors=errors,
            warnings=warnings,
            artifacts_created=self.artifacts_created
        )
    
    def _log_execution(self, message: str):
        """Log execution step with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.execution_log.append(log_entry)
        self.logger.info(log_entry)
    
    async def _cleanup_pipeline(self):
        """Cleanup pipeline resources"""
        try:
            if AGENTS_AVAILABLE:
                if self.data_agent:
                    await self.data_agent.stop()
                if self.model_agent:
                    await self.model_agent.stop()
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")


def display_pipeline_results(result: PipelineResult):
    """Display comprehensive pipeline results"""
    print("\n" + "=" * 80)
    print("AUTONOMOUS PIPELINE EXECUTION RESULTS")
    print("=" * 80)
    
    # Overall status
    status_icon = "✅" if result.success else "❌"
    print(f"{status_icon} Overall Success: {result.success}")
    print(f"⏱️  Execution Time: {result.execution_time_seconds:.1f} seconds")
    
    # Phase results
    print(f"\n📊 PHASE RESULTS:")
    scraping_icon = "✅" if result.scraping_completed else "❌"
    training_icon = "✅" if result.training_completed else "❌"
    
    print(f"   {scraping_icon} Web Scraping: {'Completed' if result.scraping_completed else 'Failed'}")
    print(f"   {training_icon} Model Training: {'Completed' if result.training_completed else 'Failed'}")
    
    # Data quality
    print(f"\n📈 DATA QUALITY:")
    print(f"   Quality Score: {result.data_quality_score:.3f} / 1.000")
    
    # Model performance
    if result.model_accuracy:
        print(f"\n🏆 MODEL PERFORMANCE:")
        for model_name, accuracy in result.model_accuracy.items():
            print(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Artifacts created
    if result.artifacts_created:
        print(f"\n📁 ARTIFACTS CREATED: ({len(result.artifacts_created)} files)")
        for artifact in result.artifacts_created[:10]:  # Show first 10
            print(f"   {Path(artifact).name}")
        if len(result.artifacts_created) > 10:
            print(f"   ... and {len(result.artifacts_created) - 10} more files")
    
    # Errors and warnings
    if result.errors:
        print(f"\n❌ ERRORS ({len(result.errors)}):")
        for error in result.errors:
            print(f"   • {error}")
    
    if result.warnings:
        print(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    print("\n" + "=" * 80)
    
    # Success message
    if result.success:
        print("🎉 AUTONOMOUS PIPELINE COMPLETED SUCCESSFULLY!")
        print("Your UFC prediction models are ready for use.")
        print("\nNext steps:")
        print("• Test predictions: python main.py --mode predict")
        print("• Run profitability analysis: ./quick_analysis.sh")
        print("• View training results: python main.py --mode results")
    else:
        print("💥 PIPELINE EXECUTION FAILED")
        print("Please check the errors above and resolve issues before retrying.")
        
    print("=" * 80)


async def main():
    """Main entry point for autonomous pipeline"""
    parser = argparse.ArgumentParser(
        description="Autonomous UFC Predictor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autonomous_pipeline.py                    # Run with default settings
  python autonomous_pipeline.py --tune            # Enable hyperparameter tuning
  python autonomous_pipeline.py --debug           # Enable debug logging
  python autonomous_pipeline.py --max-retries 5   # Increase retry attempts
        """
    )
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--tune", action="store_true", 
                       help="Enable hyperparameter tuning")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for each phase")
    parser.add_argument("--scraping-timeout", type=int, default=1800,
                       help="Scraping timeout in seconds (default: 1800)")
    parser.add_argument("--training-timeout", type=int, default=3600,
                       help="Training timeout in seconds (default: 3600)")
    parser.add_argument("--graceful-timeout", type=float, default=10.0,
                       help="Graceful termination timeout in seconds (default: 10.0)")
    parser.add_argument("--heartbeat-interval", type=float, default=60.0,
                       help="Heartbeat logging interval in seconds (default: 60.0)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline configuration
    config = PipelineConfig(
        debug_mode=args.debug,
        tune_hyperparameters=args.tune,
        max_retries=args.max_retries,
        scraping_timeout=args.scraping_timeout,
        training_timeout=args.training_timeout,
        graceful_termination_timeout=args.graceful_timeout,
        heartbeat_interval=args.heartbeat_interval
    )
    
    # Execute pipeline
    pipeline = AutonomousPipeline(config)
    result = await pipeline.execute_pipeline()
    
    # Display results
    display_pipeline_results(result)
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Pipeline failed with unexpected error: {e}")
        sys.exit(1)