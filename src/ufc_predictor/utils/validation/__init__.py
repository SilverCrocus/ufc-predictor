"""
UFC Prediction System Validation Package

Provides comprehensive input validation and sanitization with strict
error handling and no fallback mechanisms.
"""

from .fighter_name_validator import (
    FighterNameValidator,
    FighterNameValidationError,
    create_ufc_name_validator,
    validate_ufc_fighter_name,
    validate_ufc_fighter_pair
)

from .odds_validator import (
    OddsValidator,
    OddsValidationError,
    create_ufc_odds_validator,
    validate_ufc_decimal_odds,
    validate_ufc_odds_pair
)

from .dataframe_validator import (
    DataFrameValidator,
    DataFrameValidationError,
    UFCFeatureSchema,
    create_ufc_dataframe_validator,
    validate_ufc_prediction_dataframe
)

from .unified_validator import (
    UFCValidationSuite,
    ValidationResult,
    create_strict_ufc_validator,
    validate_complete_prediction_input
)

__all__ = [
    # Fighter name validation
    'FighterNameValidator',
    'FighterNameValidationError',
    'create_ufc_name_validator',
    'validate_ufc_fighter_name',
    'validate_ufc_fighter_pair',
    
    # Odds validation
    'OddsValidator',
    'OddsValidationError',
    'create_ufc_odds_validator',
    'validate_ufc_decimal_odds',
    'validate_ufc_odds_pair',
    
    # DataFrame validation
    'DataFrameValidator',
    'DataFrameValidationError',
    'UFCFeatureSchema',
    'create_ufc_dataframe_validator',
    'validate_ufc_prediction_dataframe',
    
    # Unified validation
    'UFCValidationSuite',
    'ValidationResult',
    'create_strict_ufc_validator',
    'validate_complete_prediction_input'
]