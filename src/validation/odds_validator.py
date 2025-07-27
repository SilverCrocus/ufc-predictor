"""
Odds Data Validation and Sanitization

Provides comprehensive validation for betting odds data with strict
security controls and market integrity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)


class OddsValidationError(Exception):
    """Raised when odds validation fails"""
    pass


class OddsValidator:
    """
    Validates betting odds with strict security and integrity controls
    
    Handles:
    - Decimal odds validation (1.01 - 50.0)
    - Market integrity checks
    - Type safety enforcement
    - Probability conversion validation
    
    Prevents:
    - Market manipulation through invalid odds
    - Numeric overflow/underflow
    - Type confusion attacks
    - Invalid probability calculations
    """
    
    # Valid odds ranges for UFC betting
    MIN_DECIMAL_ODDS = 1.01  # ~99% probability
    MAX_DECIMAL_ODDS = 50.0  # ~2% probability
    
    # Precision for decimal calculations
    DECIMAL_PRECISION = 4
    
    # Maximum allowed vig (bookmaker margin) - 15%
    MAX_VIG_PERCENTAGE = 0.15
    
    def __init__(self, strict_mode: bool = True, allow_high_vig: bool = False):
        """
        Initialize odds validator
        
        Args:
            strict_mode: Apply stricter validation rules
            allow_high_vig: Allow higher than normal vig percentages
        """
        self.strict_mode = strict_mode
        self.allow_high_vig = allow_high_vig
        
    def validate_decimal_odds(self, odds: Union[float, int, str, Decimal]) -> Decimal:
        """
        Validate single decimal odds value
        
        Args:
            odds: Odds value to validate
            
        Returns:
            Decimal: Validated odds as precise decimal
            
        Raises:
            OddsValidationError: If odds fail validation
        """
        # Type checking and conversion
        try:
            if isinstance(odds, str):
                # Remove common formatting
                odds_clean = odds.strip().replace(',', '')
                odds_decimal = Decimal(odds_clean)
            elif isinstance(odds, (int, float)):
                if np.isnan(odds) or np.isinf(odds):
                    raise OddsValidationError(f"Odds cannot be NaN or infinite: {odds}")
                odds_decimal = Decimal(str(odds))
            elif isinstance(odds, Decimal):
                odds_decimal = odds
            else:
                raise OddsValidationError(
                    f"Invalid odds type: {type(odds).__name__}. Expected float, int, str, or Decimal"
                )
                
        except (InvalidOperation, ValueError, TypeError) as e:
            raise OddsValidationError(f"Cannot convert odds to decimal: {odds} - {e}")
        
        # Round to prevent precision issues
        try:
            odds_decimal = odds_decimal.quantize(
                Decimal('0.' + '0' * self.DECIMAL_PRECISION)
            )
        except InvalidOperation as e:
            raise OddsValidationError(f"Failed to quantize odds: {odds_decimal} - {e}")
        
        # Range validation
        if odds_decimal < Decimal(str(self.MIN_DECIMAL_ODDS)):
            raise OddsValidationError(
                f"Odds too low: {odds_decimal} (min: {self.MIN_DECIMAL_ODDS})"
            )
        
        if odds_decimal > Decimal(str(self.MAX_DECIMAL_ODDS)):
            raise OddsValidationError(
                f"Odds too high: {odds_decimal} (max: {self.MAX_DECIMAL_ODDS})"
            )
        
        # Additional strict mode checks
        if self.strict_mode:
            # Check for suspicious round numbers that might indicate manipulation
            if self._is_suspicious_round_number(odds_decimal):
                logger.warning(f"Suspicious round odds detected: {odds_decimal}")
        
        return odds_decimal
    
    def validate_odds_pair(self, odds_a: Union[float, int, str], 
                          odds_b: Union[float, int, str]) -> Tuple[Decimal, Decimal]:
        """
        Validate a pair of odds (e.g., for two fighters)
        
        Args:
            odds_a: Odds for fighter A
            odds_b: Odds for fighter B
            
        Returns:
            Tuple[Decimal, Decimal]: Validated odds pair
            
        Raises:
            OddsValidationError: If validation fails
        """
        try:
            validated_a = self.validate_decimal_odds(odds_a)
            validated_b = self.validate_decimal_odds(odds_b)
        except OddsValidationError as e:
            raise OddsValidationError(f"Odds pair validation failed: {e}")
        
        # Check for identical odds (suspicious)
        if validated_a == validated_b:
            if self.strict_mode:
                raise OddsValidationError(
                    f"Identical odds detected (suspicious): {validated_a}"
                )
            else:
                logger.warning(f"Identical odds detected: {validated_a}")
        
        # Calculate implied probabilities
        prob_a = self._odds_to_probability(validated_a)
        prob_b = self._odds_to_probability(validated_b)
        
        # Check total probability (market integrity)
        total_prob = prob_a + prob_b
        vig = float(total_prob - Decimal('1.0'))
        
        if vig < 0:
            raise OddsValidationError(
                f"Negative vig detected (arbitrage opportunity): {vig:.4f}"
            )
        
        if not self.allow_high_vig and vig > self.MAX_VIG_PERCENTAGE:
            raise OddsValidationError(
                f"Excessive vig: {vig:.4f} (max: {self.MAX_VIG_PERCENTAGE})"
            )
        
        logger.debug(f"Odds pair validated: {validated_a}/{validated_b}, vig: {vig:.4f}")
        return validated_a, validated_b
    
    def validate_fight_odds(self, fight_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete fight odds data structure
        
        Args:
            fight_data: Dictionary containing fight and odds information
            
        Returns:
            Dict[str, Any]: Validated and normalized fight data
            
        Raises:
            OddsValidationError: If validation fails
        """
        if not isinstance(fight_data, dict):
            raise OddsValidationError(
                f"Fight data must be dictionary, got {type(fight_data).__name__}"
            )
        
        # Required fields
        required_fields = ['fighter_a', 'fighter_b']
        for field in required_fields:
            if field not in fight_data:
                raise OddsValidationError(f"Missing required field: {field}")
            if not fight_data[field]:
                raise OddsValidationError(f"Empty required field: {field}")
        
        validated_data = fight_data.copy()
        
        # Validate odds if present
        odds_fields = ['odds_a', 'odds_b', 'decimal_odds_a', 'decimal_odds_b']
        odds_present = any(field in fight_data for field in odds_fields)
        
        if odds_present:
            # Determine which odds fields to use
            if 'decimal_odds_a' in fight_data and 'decimal_odds_b' in fight_data:
                odds_a_raw = fight_data['decimal_odds_a']
                odds_b_raw = fight_data['decimal_odds_b']
            elif 'odds_a' in fight_data and 'odds_b' in fight_data:
                odds_a_raw = fight_data['odds_a']
                odds_b_raw = fight_data['odds_b']
            else:
                raise OddsValidationError(
                    "Incomplete odds data: need both odds_a/odds_b or decimal_odds_a/decimal_odds_b"
                )
            
            # Validate the odds pair
            try:
                validated_odds_a, validated_odds_b = self.validate_odds_pair(odds_a_raw, odds_b_raw)
                
                # Store validated odds in consistent format
                validated_data['validated_odds_a'] = float(validated_odds_a)
                validated_data['validated_odds_b'] = float(validated_odds_b)
                
                # Calculate derived values
                validated_data['implied_prob_a'] = float(self._odds_to_probability(validated_odds_a))
                validated_data['implied_prob_b'] = float(self._odds_to_probability(validated_odds_b))
                
                total_prob = validated_data['implied_prob_a'] + validated_data['implied_prob_b']
                validated_data['market_vig'] = float(total_prob - 1.0)
                
            except OddsValidationError as e:
                raise OddsValidationError(f"Fight {validated_data.get('fight_key', 'unknown')}: {e}")
        
        # Validate additional numeric fields
        numeric_fields = ['event_timestamp', 'line_movement', 'volume']
        for field in numeric_fields:
            if field in fight_data:
                try:
                    validated_data[field] = self._validate_numeric_field(
                        fight_data[field], field
                    )
                except (ValueError, TypeError) as e:
                    raise OddsValidationError(f"Invalid {field}: {e}")
        
        return validated_data
    
    def validate_event_odds(self, event_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Validate odds data for an entire event
        
        Args:
            event_data: Dictionary of fight_key -> fight_data mappings
            
        Returns:
            Dict[str, Dict]: Validated event data
            
        Raises:
            OddsValidationError: If validation fails
        """
        if not isinstance(event_data, dict):
            raise OddsValidationError(
                f"Event data must be dictionary, got {type(event_data).__name__}"
            )
        
        if not event_data:
            raise OddsValidationError("Event data cannot be empty")
        
        if len(event_data) > 50:  # Prevent excessive event sizes
            raise OddsValidationError(
                f"Event too large: {len(event_data)} fights (max: 50)"
            )
        
        validated_event = {}
        failed_fights = []
        
        for fight_key, fight_data in event_data.items():
            try:
                # Validate fight key
                if not isinstance(fight_key, str) or not fight_key.strip():
                    raise OddsValidationError(f"Invalid fight key: {fight_key}")
                
                validated_fight = self.validate_fight_odds(fight_data)
                validated_event[fight_key] = validated_fight
                
            except OddsValidationError as e:
                failed_fights.append((fight_key, str(e)))
                logger.error(f"Fight validation failed: {fight_key} - {e}")
        
        # In strict mode, fail if any fights fail validation
        if self.strict_mode and failed_fights:
            error_summary = "; ".join([f"{key}: {error}" for key, error in failed_fights])
            raise OddsValidationError(
                f"Event validation failed for {len(failed_fights)} fights: {error_summary}"
            )
        
        logger.info(f"Event validated: {len(validated_event)} fights successful, "
                   f"{len(failed_fights)} failed")
        
        return validated_event
    
    def _odds_to_probability(self, odds: Decimal) -> Decimal:
        """Convert decimal odds to implied probability"""
        return Decimal('1') / odds
    
    def _validate_numeric_field(self, value: Any, field_name: str) -> float:
        """Validate a numeric field with appropriate constraints"""
        if pd.isna(value):
            return float('nan')
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {field_name} to numeric: {value}")
        
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            return float('nan')  # Allow NaN but not inf
        
        # Field-specific validation
        if field_name == 'volume' and numeric_value < 0:
            raise ValueError(f"Volume cannot be negative: {numeric_value}")
        
        return numeric_value
    
    def _is_suspicious_round_number(self, odds: Decimal) -> bool:
        """Check if odds are suspiciously round (potential manipulation indicator)"""
        # Check for exact round numbers like 2.00, 3.00, etc.
        return odds == odds.quantize(Decimal('1'))


# Factory function
def create_ufc_odds_validator(strict_mode: bool = True, 
                             allow_high_vig: bool = False) -> OddsValidator:
    """Create configured UFC odds validator"""
    return OddsValidator(strict_mode=strict_mode, allow_high_vig=allow_high_vig)


# Convenience functions
def validate_ufc_decimal_odds(odds: Union[float, int, str]) -> float:
    """Quick validation for single odds value"""
    validator = create_ufc_odds_validator()
    return float(validator.validate_decimal_odds(odds))


def validate_ufc_odds_pair(odds_a: Union[float, int, str], 
                          odds_b: Union[float, int, str]) -> Tuple[float, float]:
    """Quick validation for odds pair"""
    validator = create_ufc_odds_validator()
    validated_a, validated_b = validator.validate_odds_pair(odds_a, odds_b)
    return float(validated_a), float(validated_b)