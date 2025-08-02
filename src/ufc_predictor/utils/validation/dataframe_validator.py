"""
DataFrame Schema Validation for UFC Prediction System

Provides comprehensive validation for ML input DataFrames with strict
schema enforcement and feature integrity checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataFrameValidationError(Exception):
    """Raised when DataFrame validation fails"""
    pass


class UFCFeatureSchema:
    """
    Defines expected schema for UFC prediction features
    
    Based on the 70 engineered features from the feature engineering pipeline
    """
    
    # Expected feature categories
    FIGHTER_STATS = [
        'SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.',
        'Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.'
    ]
    
    PHYSICAL_ATTRIBUTES = [
        'Height_inches', 'Weight_lbs', 'Reach_inches', 'Age'
    ]
    
    RECORD_FEATURES = [
        'Wins', 'Losses', 'Draws', 'Win_Rate', 'Total_Fights'
    ]
    
    STANCE_FEATURES = [
        'STANCE_Orthodox', 'STANCE_Southpaw', 'STANCE_Switch', 'STANCE_Unknown'
    ]
    
    # Differential features (A vs B comparisons)
    DIFFERENTIAL_SUFFIX = '_diff'
    
    # Valid data types for each category
    FEATURE_DTYPES = {
        'float_features': FIGHTER_STATS + PHYSICAL_ATTRIBUTES + ['Win_Rate'],
        'int_features': RECORD_FEATURES[:-1] + ['Total_Fights'],  # Exclude Win_Rate
        'binary_features': STANCE_FEATURES
    }
    
    # Valid ranges for features
    FEATURE_RANGES = {
        'SLpM': (0.0, 15.0),           # Strikes landed per minute
        'SApM': (0.0, 20.0),           # Strikes absorbed per minute  
        'TD Avg.': (0.0, 10.0),        # Takedowns per 15 min
        'Sub. Avg.': (0.0, 5.0),       # Submission attempts per 15 min
        'Str. Acc.': (0.0, 1.0),       # Striking accuracy (percentage)
        'Str. Def': (0.0, 1.0),        # Striking defense (percentage)
        'TD Acc.': (0.0, 1.0),         # Takedown accuracy (percentage)
        'TD Def.': (0.0, 1.0),         # Takedown defense (percentage)
        'Height_inches': (60.0, 84.0), # 5'0" to 7'0"
        'Weight_lbs': (115.0, 265.0),  # Strawweight to Heavyweight
        'Reach_inches': (60.0, 84.0),  # Reasonable reach range
        'Age': (18.0, 50.0),           # Fighting age range
        'Wins': (0, 100),              # Career wins
        'Losses': (0, 50),             # Career losses
        'Draws': (0, 10),              # Career draws
        'Total_Fights': (0, 150),      # Total career fights
        'Win_Rate': (0.0, 1.0),        # Win percentage
    }


class DataFrameValidator:
    """
    Validates UFC prediction DataFrames with strict schema enforcement
    
    Ensures:
    - Correct column structure and types
    - Valid feature ranges
    - No data poisoning attempts
    - Consistent differential features
    - Proper missing value handling
    """
    
    def __init__(self, schema: UFCFeatureSchema = None, strict_mode: bool = True):
        """
        Initialize DataFrame validator
        
        Args:
            schema: Feature schema definition
            strict_mode: Apply strict validation rules
        """
        self.schema = schema or UFCFeatureSchema()
        self.strict_mode = strict_mode
        
    def validate_prediction_dataframe(self, df: pd.DataFrame, 
                                    expected_columns: Optional[List[str]] = None,
                                    check_differential: bool = True) -> pd.DataFrame:
        """
        Validate DataFrame for UFC prediction input
        
        Args:
            df: Input DataFrame to validate
            expected_columns: Expected column names (if None, inferred from schema)
            check_differential: Whether to validate differential features
            
        Returns:
            pd.DataFrame: Validated DataFrame
            
        Raises:
            DataFrameValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise DataFrameValidationError(
                f"Input must be pandas DataFrame, got {type(df).__name__}"
            )
        
        if df.empty:
            raise DataFrameValidationError("DataFrame cannot be empty")
        
        # Create a copy to avoid modifying original
        validated_df = df.copy()
        
        # Basic structure validation
        self._validate_basic_structure(validated_df)
        
        # Column validation
        if expected_columns:
            self._validate_columns(validated_df, expected_columns)
        else:
            self._validate_schema_columns(validated_df)
        
        # Data type validation
        self._validate_data_types(validated_df)
        
        # Range validation
        self._validate_feature_ranges(validated_df)
        
        # Missing value validation
        self._validate_missing_values(validated_df)
        
        # Differential feature validation
        if check_differential:
            self._validate_differential_features(validated_df)
        
        # Advanced integrity checks
        if self.strict_mode:
            self._validate_data_integrity(validated_df)
        
        logger.info(f"DataFrame validated successfully: {len(validated_df)} rows, "
                   f"{len(validated_df.columns)} columns")
        
        return validated_df
    
    def validate_feature_consistency(self, df: pd.DataFrame, 
                                   model_columns: List[str]) -> pd.DataFrame:
        """
        Validate that DataFrame features match trained model expectations
        
        Args:
            df: Input DataFrame
            model_columns: Expected columns from trained model
            
        Returns:
            pd.DataFrame: DataFrame with consistent feature set
            
        Raises:
            DataFrameValidationError: If features don't match
        """
        if not isinstance(model_columns, list):
            raise DataFrameValidationError(
                f"Model columns must be list, got {type(model_columns).__name__}"
            )
        
        if not model_columns:
            raise DataFrameValidationError("Model columns list cannot be empty")
        
        # Check for required columns
        missing_columns = set(model_columns) - set(df.columns)
        if missing_columns:
            raise DataFrameValidationError(
                f"Missing required model features: {sorted(missing_columns)}"
            )
        
        # Check for unexpected columns
        extra_columns = set(df.columns) - set(model_columns)
        if extra_columns and self.strict_mode:
            logger.warning(f"Extra columns will be ignored: {sorted(extra_columns)}")
        
        # Return DataFrame with exact model column order
        try:
            consistent_df = df[model_columns].copy()
        except KeyError as e:
            raise DataFrameValidationError(f"Failed to extract model features: {e}")
        
        # Validate the consistent DataFrame
        return self.validate_prediction_dataframe(
            consistent_df, 
            expected_columns=model_columns,
            check_differential=False  # Already checked above
        )
    
    def _validate_basic_structure(self, df: pd.DataFrame):
        """Validate basic DataFrame structure"""
        # Check dimensions
        if len(df) > 10000:  # Prevent excessive memory usage
            raise DataFrameValidationError(
                f"DataFrame too large: {len(df)} rows (max: 10000)"
            )
        
        if len(df.columns) > 500:  # Prevent feature explosion
            raise DataFrameValidationError(
                f"Too many columns: {len(df.columns)} (max: 500)"
            )
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            raise DataFrameValidationError(
                f"Duplicate column names: {duplicate_cols}"
            )
        
        # Check for invalid column names
        invalid_cols = [col for col in df.columns if not isinstance(col, str)]
        if invalid_cols:
            raise DataFrameValidationError(
                f"Invalid column names (must be strings): {invalid_cols}"
            )
    
    def _validate_columns(self, df: pd.DataFrame, expected_columns: List[str]):
        """Validate column names against expected list"""
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        if missing_columns:
            raise DataFrameValidationError(
                f"Missing expected columns: {sorted(missing_columns)}"
            )
        
        # In strict mode, check for exact match
        if self.strict_mode:
            extra_columns = actual_columns - expected_columns_set
            if extra_columns:
                raise DataFrameValidationError(
                    f"Unexpected columns: {sorted(extra_columns)}"
                )
    
    def _validate_schema_columns(self, df: pd.DataFrame):
        """Validate columns against UFC schema"""
        # Build expected column set from schema
        expected_base_features = (
            self.schema.FIGHTER_STATS + 
            self.schema.PHYSICAL_ATTRIBUTES + 
            self.schema.RECORD_FEATURES + 
            self.schema.STANCE_FEATURES
        )
        
        # Check for core features (allowing for differential variants)
        missing_core = []
        for feature in expected_base_features:
            # Look for either the base feature or its differential version
            has_base = feature in df.columns
            has_diff = f"{feature}{self.schema.DIFFERENTIAL_SUFFIX}" in df.columns
            
            if not (has_base or has_diff):
                missing_core.append(feature)
        
        if missing_core and self.strict_mode:
            raise DataFrameValidationError(
                f"Missing core UFC features: {sorted(missing_core)}"
            )
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate column data types"""
        type_errors = []
        
        for dtype_category, features in self.schema.FEATURE_DTYPES.items():
            for feature in features:
                # Check both base and differential versions
                for col_name in [feature, f"{feature}{self.schema.DIFFERENTIAL_SUFFIX}"]:
                    if col_name not in df.columns:
                        continue
                    
                    col_dtype = df[col_name].dtype
                    
                    if dtype_category == 'float_features':
                        if not pd.api.types.is_numeric_dtype(col_dtype):
                            type_errors.append(f"{col_name}: expected numeric, got {col_dtype}")
                    elif dtype_category == 'int_features':
                        if not pd.api.types.is_integer_dtype(col_dtype) and not pd.api.types.is_numeric_dtype(col_dtype):
                            type_errors.append(f"{col_name}: expected integer/numeric, got {col_dtype}")
                    elif dtype_category == 'binary_features':
                        if not pd.api.types.is_integer_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype):
                            type_errors.append(f"{col_name}: expected binary (0/1), got {col_dtype}")
        
        if type_errors:
            raise DataFrameValidationError(f"Data type validation failed: {'; '.join(type_errors)}")
    
    def _validate_feature_ranges(self, df: pd.DataFrame):
        """Validate feature values are within expected ranges"""
        range_errors = []
        
        for feature, (min_val, max_val) in self.schema.FEATURE_RANGES.items():
            # Check both base and differential versions
            for col_name in [feature, f"{feature}{self.schema.DIFFERENTIAL_SUFFIX}"]:
                if col_name not in df.columns:
                    continue
                
                col_data = df[col_name].dropna()  # Ignore NaN values for range checking
                
                if len(col_data) == 0:
                    continue
                
                # For differential features, allow negative values
                if col_name.endswith(self.schema.DIFFERENTIAL_SUFFIX):
                    # Differential range is roughly 2x the base range, centered at 0
                    diff_range = max_val - min_val
                    actual_min_val = -diff_range
                    actual_max_val = diff_range
                else:
                    actual_min_val = min_val
                    actual_max_val = max_val
                
                # Check for values outside range
                below_min = col_data < actual_min_val
                above_max = col_data > actual_max_val
                
                if below_min.any():
                    below_count = below_min.sum()
                    min_value = col_data.min()
                    range_errors.append(
                        f"{col_name}: {below_count} values below {actual_min_val} (min: {min_value})"
                    )
                
                if above_max.any():
                    above_count = above_max.sum()
                    max_value = col_data.max()
                    range_errors.append(
                        f"{col_name}: {above_count} values above {actual_max_val} (max: {max_value})"
                    )
        
        if range_errors:
            error_msg = f"Feature range validation failed: {'; '.join(range_errors)}"
            if self.strict_mode:
                raise DataFrameValidationError(error_msg)
            else:
                logger.warning(error_msg)
    
    def _validate_missing_values(self, df: pd.DataFrame):
        """Validate missing value patterns"""
        # Check for columns with excessive missing values
        missing_threshold = 0.5  # 50% missing is too much
        
        high_missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                high_missing_cols.append((col, missing_pct))
        
        if high_missing_cols:
            error_details = [f"{col}: {pct:.1%}" for col, pct in high_missing_cols]
            error_msg = f"Columns with excessive missing values: {'; '.join(error_details)}"
            
            if self.strict_mode:
                raise DataFrameValidationError(error_msg)
            else:
                logger.warning(error_msg)
        
        # Check for rows with all missing values
        all_missing_rows = df.isnull().all(axis=1).sum()
        if all_missing_rows > 0:
            raise DataFrameValidationError(
                f"Found {all_missing_rows} rows with all missing values"
            )
    
    def _validate_differential_features(self, df: pd.DataFrame):
        """Validate differential feature consistency"""
        # Find differential columns
        diff_cols = [col for col in df.columns if col.endswith(self.schema.DIFFERENTIAL_SUFFIX)]
        
        if not diff_cols:
            logger.debug("No differential features found")
            return
        
        # Check for mathematical consistency in differential features
        inconsistencies = []
        
        for diff_col in diff_cols:
            base_name = diff_col.replace(self.schema.DIFFERENTIAL_SUFFIX, '')
            
            # Look for corresponding base features
            fighter_a_col = f"{base_name}_a" if f"{base_name}_a" in df.columns else None
            fighter_b_col = f"{base_name}_b" if f"{base_name}_b" in df.columns else None
            
            if fighter_a_col and fighter_b_col:
                # Check if differential = A - B
                expected_diff = df[fighter_a_col] - df[fighter_b_col]
                actual_diff = df[diff_col]
                
                # Allow small floating point differences
                tolerance = 1e-6
                diff_mismatch = np.abs(expected_diff - actual_diff) > tolerance
                
                if diff_mismatch.any():
                    mismatch_count = diff_mismatch.sum()
                    inconsistencies.append(f"{diff_col}: {mismatch_count} inconsistent values")
        
        if inconsistencies:
            error_msg = f"Differential feature inconsistencies: {'; '.join(inconsistencies)}"
            if self.strict_mode:
                raise DataFrameValidationError(error_msg)
            else:
                logger.warning(error_msg)
    
    def _validate_data_integrity(self, df: pd.DataFrame):
        """Advanced data integrity checks for potential poisoning"""
        # Check for impossible value combinations
        integrity_errors = []
        
        # Win rate should match wins/(wins+losses) if available
        if all(col in df.columns for col in ['Wins', 'Losses', 'Win_Rate']):
            wins = df['Wins']
            losses = df['Losses']
            win_rate = df['Win_Rate']
            
            total_fights = wins + losses
            expected_win_rate = np.where(total_fights > 0, wins / total_fights, 0)
            
            # Allow small floating point differences
            tolerance = 0.01
            mismatch = np.abs(win_rate - expected_win_rate) > tolerance
            
            if mismatch.any():
                mismatch_count = mismatch.sum()
                integrity_errors.append(f"Win rate mismatch: {mismatch_count} rows")
        
        # Check for suspiciously perfect correlations (potential synthetic data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find perfect correlations (excluding diagonal)
            perfect_corr = (np.abs(corr_matrix) > 0.999) & (corr_matrix != 1.0)
            
            if perfect_corr.any().any():
                logger.warning("Suspiciously perfect correlations detected (potential synthetic data)")
        
        if integrity_errors:
            error_msg = f"Data integrity validation failed: {'; '.join(integrity_errors)}"
            raise DataFrameValidationError(error_msg)


# Factory function
def create_ufc_dataframe_validator(strict_mode: bool = True) -> DataFrameValidator:
    """Create configured UFC DataFrame validator"""
    return DataFrameValidator(strict_mode=strict_mode)


# Convenience function
def validate_ufc_prediction_dataframe(df: pd.DataFrame, 
                                     model_columns: Optional[List[str]] = None,
                                     strict_mode: bool = True) -> pd.DataFrame:
    """Quick validation for UFC prediction DataFrame"""
    validator = create_ufc_dataframe_validator(strict_mode=strict_mode)
    
    if model_columns:
        return validator.validate_feature_consistency(df, model_columns)
    else:
        return validator.validate_prediction_dataframe(df)