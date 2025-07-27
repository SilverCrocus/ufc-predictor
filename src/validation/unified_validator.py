"""
Unified Validation Suite for UFC Prediction System

Combines all validation components into a single, comprehensive
validation framework with strict error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .fighter_name_validator import FighterNameValidator, FighterNameValidationError
from .odds_validator import OddsValidator, OddsValidationError  
from .dataframe_validator import DataFrameValidator, DataFrameValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from comprehensive validation"""
    is_valid: bool
    validated_data: Optional[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    validation_timestamp: datetime
    validation_summary: Dict[str, Any]


class UFCValidationSuite:
    """
    Comprehensive validation suite for UFC prediction system
    
    Provides:
    - Unified validation interface
    - Strict error handling with NO fallbacks
    - Comprehensive logging and error reporting
    - Type safety enforcement
    - Security controls
    """
    
    def __init__(self, strict_mode: bool = True, enable_logging: bool = True):
        """
        Initialize validation suite
        
        Args:
            strict_mode: Enable strict validation (recommended for production)
            enable_logging: Enable detailed validation logging
        """
        self.strict_mode = strict_mode
        self.enable_logging = enable_logging
        
        # Initialize validators
        self.name_validator = FighterNameValidator(strict_mode=strict_mode)
        self.odds_validator = OddsValidator(strict_mode=strict_mode)
        self.dataframe_validator = DataFrameValidator(strict_mode=strict_mode)
        
        if enable_logging:
            logger.info(f"UFC Validation Suite initialized (strict_mode: {strict_mode})")
    
    def validate_prediction_input(self, 
                                fighter_a: str,
                                fighter_b: str,
                                feature_dataframe: pd.DataFrame,
                                model_columns: List[str],
                                odds_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate complete prediction input with strict error handling
        
        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name  
            feature_dataframe: ML feature DataFrame
            model_columns: Expected model feature columns
            odds_data: Optional betting odds data
            
        Returns:
            ValidationResult: Comprehensive validation result
            
        Raises:
            Various ValidationError types: If validation fails in strict mode
        """
        validation_start = datetime.now()
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            # 1. Validate fighter names
            self._log_validation_step("Validating fighter names")
            try:
                validated_fighter_a, validated_fighter_b = self.name_validator.validate_fighter_pair(
                    fighter_a, fighter_b
                )
                validated_data['fighter_a'] = validated_fighter_a
                validated_data['fighter_b'] = validated_fighter_b
                
            except FighterNameValidationError as e:
                error_msg = f"Fighter name validation failed: {e}"
                errors.append(error_msg)
                if self.strict_mode:
                    raise FighterNameValidationError(error_msg)
            
            # 2. Validate feature DataFrame
            self._log_validation_step("Validating feature DataFrame")
            try:
                validated_df = self.dataframe_validator.validate_feature_consistency(
                    feature_dataframe, model_columns
                )
                validated_data['features'] = validated_df
                
            except DataFrameValidationError as e:
                error_msg = f"DataFrame validation failed: {e}"
                errors.append(error_msg)
                if self.strict_mode:
                    raise DataFrameValidationError(error_msg)
            
            # 3. Validate odds data (if provided)
            if odds_data is not None:
                self._log_validation_step("Validating odds data")
                try:
                    validated_odds = self.odds_validator.validate_fight_odds(odds_data)
                    validated_data['odds'] = validated_odds
                    
                except OddsValidationError as e:
                    error_msg = f"Odds validation failed: {e}"
                    errors.append(error_msg)
                    if self.strict_mode:
                        raise OddsValidationError(error_msg)
            
            # 4. Cross-validation checks
            self._log_validation_step("Performing cross-validation checks")
            cross_validation_warnings = self._perform_cross_validation(validated_data)
            warnings.extend(cross_validation_warnings)
            
            # 5. Security checks
            self._log_validation_step("Performing security checks")
            security_warnings = self._perform_security_checks(validated_data)
            warnings.extend(security_warnings)
            
            # Determine overall validation result
            is_valid = len(errors) == 0
            
            # Create validation summary
            validation_summary = {
                'total_checks': 5,
                'passed_checks': 5 - len(errors),
                'error_count': len(errors),
                'warning_count': len(warnings),
                'strict_mode': self.strict_mode,
                'validation_duration_ms': (datetime.now() - validation_start).total_seconds() * 1000
            }
            
            if self.enable_logging:
                if is_valid:
                    logger.info(f"Validation completed successfully: {validation_summary}")
                else:
                    logger.error(f"Validation failed: {errors}")
            
            return ValidationResult(
                is_valid=is_valid,
                validated_data=validated_data if is_valid else None,
                errors=errors,
                warnings=warnings,
                validation_timestamp=validation_start,
                validation_summary=validation_summary
            )
            
        except Exception as e:
            # Re-raise specific validation errors
            if isinstance(e, (FighterNameValidationError, OddsValidationError, DataFrameValidationError)):
                raise
            
            # Handle unexpected errors
            error_msg = f"Unexpected validation error: {type(e).__name__}: {e}"
            logger.error(error_msg)
            
            if self.strict_mode:
                raise RuntimeError(error_msg)
            
            return ValidationResult(
                is_valid=False,
                validated_data=None,
                errors=[error_msg],
                warnings=warnings,
                validation_timestamp=validation_start,
                validation_summary={'validation_failed': True}
            )
    
    def validate_batch_predictions(self,
                                 fighter_pairs: List[Tuple[str, str]],
                                 feature_dataframes: List[pd.DataFrame],
                                 model_columns: List[str],
                                 odds_data_list: Optional[List[Dict[str, Any]]] = None) -> List[ValidationResult]:
        """
        Validate batch prediction inputs with individual error handling
        
        Args:
            fighter_pairs: List of (fighter_a, fighter_b) tuples
            feature_dataframes: List of feature DataFrames for each pair
            model_columns: Expected model columns
            odds_data_list: Optional list of odds data
            
        Returns:
            List[ValidationResult]: Individual validation results
            
        Raises:
            ValueError: If input lists have mismatched lengths
        """
        # Input validation
        if len(fighter_pairs) != len(feature_dataframes):
            raise ValueError(
                f"Mismatched input lengths: {len(fighter_pairs)} pairs, {len(feature_dataframes)} DataFrames"
            )
        
        if odds_data_list and len(odds_data_list) != len(fighter_pairs):
            raise ValueError(
                f"Mismatched input lengths: {len(fighter_pairs)} pairs, {len(odds_data_list)} odds data"
            )
        
        # Validate batch size
        if len(fighter_pairs) > 100:
            raise ValueError(f"Batch too large: {len(fighter_pairs)} (max: 100)")
        
        results = []
        
        for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
            try:
                feature_df = feature_dataframes[i]
                odds_data = odds_data_list[i] if odds_data_list else None
                
                result = self.validate_prediction_input(
                    fighter_a, fighter_b, feature_df, model_columns, odds_data
                )
                results.append(result)
                
            except Exception as e:
                # Create error result for this pair
                error_result = ValidationResult(
                    is_valid=False,
                    validated_data=None,
                    errors=[f"Batch item {i} failed: {e}"],
                    warnings=[],
                    validation_timestamp=datetime.now(),
                    validation_summary={'batch_index': i, 'failed': True}
                )
                results.append(error_result)
                
                if self.strict_mode:
                    logger.error(f"Batch validation failed at index {i}: {e}")
                    # In strict mode, fail fast
                    raise
        
        # Summary logging
        successful_count = sum(1 for r in results if r.is_valid)
        logger.info(f"Batch validation completed: {successful_count}/{len(results)} successful")
        
        return results
    
    def _perform_cross_validation(self, validated_data: Dict[str, Any]) -> List[str]:
        """Perform cross-validation checks between different data components"""
        warnings = []
        
        # Check if fighter names appear in features (if available)
        if 'features' in validated_data and 'fighter_a' in validated_data:
            features_df = validated_data['features']
            
            # Look for potential name leakage in feature columns
            fighter_a = validated_data['fighter_a'].lower()
            fighter_b = validated_data['fighter_b'].lower()
            
            for col in features_df.columns:
                col_lower = str(col).lower()
                if fighter_a in col_lower or fighter_b in col_lower:
                    warnings.append(f"Potential name leakage in feature column: {col}")
        
        # Check odds consistency (if available)
        if 'odds' in validated_data:
            odds_data = validated_data['odds']
            
            # Check if implied probabilities are reasonable
            if 'implied_prob_a' in odds_data and 'implied_prob_b' in odds_data:
                prob_a = odds_data['implied_prob_a']
                prob_b = odds_data['implied_prob_b']
                
                if abs(prob_a - prob_b) < 0.01:  # Very close probabilities
                    warnings.append("Very close implied probabilities (possible pick'em fight)")
        
        return warnings
    
    def _perform_security_checks(self, validated_data: Dict[str, Any]) -> List[str]:
        """Perform security-related checks"""
        warnings = []
        
        # Check for potential data poisoning indicators
        if 'features' in validated_data:
            features_df = validated_data['features']
            
            # Check for suspiciously perfect values
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = features_df[col].dropna()
                if len(col_data) > 0:
                    # Check for all identical values (suspicious)
                    if col_data.nunique() == 1:
                        warnings.append(f"All identical values in feature: {col}")
                    
                    # Check for extreme values
                    if np.any(np.abs(col_data) > 1e6):
                        warnings.append(f"Extremely large values in feature: {col}")
        
        return warnings
    
    def _log_validation_step(self, step_description: str):
        """Log validation step if logging is enabled"""
        if self.enable_logging:
            logger.debug(f"Validation step: {step_description}")


def create_strict_ufc_validator() -> UFCValidationSuite:
    """Create a strict UFC validator for production use"""
    return UFCValidationSuite(strict_mode=True, enable_logging=True)


def validate_complete_prediction_input(fighter_a: str,
                                     fighter_b: str,
                                     feature_dataframe: pd.DataFrame,
                                     model_columns: List[str],
                                     odds_data: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convenience function for complete prediction input validation
    
    Args:
        fighter_a: First fighter name
        fighter_b: Second fighter name
        feature_dataframe: ML feature DataFrame
        model_columns: Expected model columns
        odds_data: Optional odds data
        
    Returns:
        Tuple[Dict[str, Any], List[str]]: (validated_data, warnings)
        
    Raises:
        Various ValidationError types: If validation fails
    """
    validator = create_strict_ufc_validator()
    result = validator.validate_prediction_input(
        fighter_a, fighter_b, feature_dataframe, model_columns, odds_data
    )
    
    if not result.is_valid:
        # Combine all errors into a single exception
        error_msg = "; ".join(result.errors)
        raise RuntimeError(f"Validation failed: {error_msg}")
    
    return result.validated_data, result.warnings