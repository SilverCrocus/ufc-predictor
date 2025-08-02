"""
Fighter Name Validation and Sanitization

Provides secure validation for UFC fighter names while preserving
international characters and legitimate naming patterns.
"""

import re
import unicodedata
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FighterNameValidationError(Exception):
    """Raised when fighter name validation fails"""
    pass


class FighterNameValidator:
    """
    Validates and sanitizes UFC fighter names with strict security controls
    
    Supports:
    - International characters (José, Khabib, etc.)
    - Legitimate punctuation (O'Malley, McGregor)
    - Multiple names and nicknames
    
    Prevents:
    - SQL injection attempts
    - Script injection
    - Path traversal
    - Control characters
    """
    
    # Allowed characters: letters, spaces, hyphens, apostrophes, periods
    ALLOWED_PATTERN = re.compile(r"^[a-zA-Z\u00C0-\u017F\u0400-\u04FF\u0590-\u05FF\s'\-\.\"]+$")
    
    # Dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        re.compile(r"[<>\"'&]"),           # HTML/XML characters
        re.compile(r"[;|&$`]"),           # Shell injection
        re.compile(r"(script|javascript|vbscript)", re.IGNORECASE),  # Script injection
        re.compile(r"(union|select|insert|delete|drop|update)", re.IGNORECASE),  # SQL injection
        re.compile(r"[^\x20-\x7E\u00C0-\u017F\u0400-\u04FF\u0590-\u05FF]"),  # Control chars
        re.compile(r"\.\.\/|\.\.\\"),     # Path traversal
        re.compile(r"__.*__"),            # Python dunder methods
    ]
    
    # Maximum reasonable length for fighter names
    MAX_NAME_LENGTH = 100
    MIN_NAME_LENGTH = 2
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        
    def validate_fighter_name(self, name: str) -> str:
        """
        Validate and sanitize a single fighter name
        
        Args:
            name: Raw fighter name input
            
        Returns:
            str: Validated and normalized fighter name
            
        Raises:
            FighterNameValidationError: If name fails validation
        """
        if not isinstance(name, str):
            raise FighterNameValidationError(
                f"Fighter name must be string, got {type(name).__name__}"
            )
        
        # Check for empty or whitespace-only names
        if not name or not name.strip():
            raise FighterNameValidationError("Fighter name cannot be empty")
        
        name = name.strip()
        
        # Check length constraints
        if len(name) < self.MIN_NAME_LENGTH:
            raise FighterNameValidationError(
                f"Fighter name too short: {len(name)} chars (min: {self.MIN_NAME_LENGTH})"
            )
        
        if len(name) > self.MAX_NAME_LENGTH:
            raise FighterNameValidationError(
                f"Fighter name too long: {len(name)} chars (max: {self.MAX_NAME_LENGTH})"
            )
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(name):
                raise FighterNameValidationError(
                    f"Fighter name contains forbidden characters or patterns: {name}"
                )
        
        # Check against allowed pattern
        if not self.ALLOWED_PATTERN.match(name):
            raise FighterNameValidationError(
                f"Fighter name contains invalid characters: {name}"
            )
        
        # Normalize Unicode characters (prevent encoding attacks)
        try:
            normalized_name = unicodedata.normalize('NFC', name)
        except Exception as e:
            raise FighterNameValidationError(f"Failed to normalize fighter name: {e}")
        
        # Additional strict mode checks
        if self.strict_mode:
            # Check for excessive repetition (potential DoS)
            if self._has_excessive_repetition(normalized_name):
                raise FighterNameValidationError(
                    f"Fighter name has excessive character repetition: {normalized_name}"
                )
            
            # Check for suspicious number patterns
            if self._has_suspicious_numbers(normalized_name):
                raise FighterNameValidationError(
                    f"Fighter name contains suspicious number patterns: {normalized_name}"
                )
        
        logger.debug(f"Fighter name validated successfully: {normalized_name}")
        return normalized_name
    
    def validate_fighter_pair(self, fighter_a: str, fighter_b: str) -> Tuple[str, str]:
        """
        Validate a pair of fighter names
        
        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name
            
        Returns:
            Tuple[str, str]: Validated fighter names
            
        Raises:
            FighterNameValidationError: If validation fails
        """
        try:
            validated_a = self.validate_fighter_name(fighter_a)
            validated_b = self.validate_fighter_name(fighter_b)
        except FighterNameValidationError as e:
            raise FighterNameValidationError(f"Fighter pair validation failed: {e}")
        
        # Check for identical names
        if validated_a.lower() == validated_b.lower():
            raise FighterNameValidationError(
                f"Fighter names cannot be identical: {validated_a}"
            )
        
        return validated_a, validated_b
    
    def _has_excessive_repetition(self, name: str) -> bool:
        """Check for excessive character repetition (potential DoS attack)"""
        max_repetition = 3
        
        for i in range(len(name) - max_repetition):
            char = name[i]
            if all(name[j] == char for j in range(i, i + max_repetition + 1)):
                return True
        
        return False
    
    def _has_suspicious_numbers(self, name: str) -> bool:
        """Check for suspicious number patterns"""
        # More than 3 consecutive digits is suspicious for fighter names
        return bool(re.search(r'\d{4,}', name))
    
    def batch_validate_names(self, names: list) -> list:
        """
        Validate multiple fighter names in batch
        
        Args:
            names: List of fighter names to validate
            
        Returns:
            list: List of validated names
            
        Raises:
            FighterNameValidationError: If any name fails validation
        """
        if not isinstance(names, list):
            raise FighterNameValidationError(
                f"Names must be provided as list, got {type(names).__name__}"
            )
        
        if len(names) > 1000:  # Prevent excessive batch sizes
            raise FighterNameValidationError(
                f"Batch size too large: {len(names)} (max: 1000)"
            )
        
        validated_names = []
        failed_names = []
        
        for i, name in enumerate(names):
            try:
                validated_name = self.validate_fighter_name(name)
                validated_names.append(validated_name)
            except FighterNameValidationError as e:
                failed_names.append((i, name, str(e)))
        
        if failed_names:
            error_details = "; ".join([
                f"Index {i}: {name} - {error}" 
                for i, name, error in failed_names
            ])
            raise FighterNameValidationError(
                f"Batch validation failed for {len(failed_names)} names: {error_details}"
            )
        
        return validated_names


# Factory function for default validator
def create_ufc_name_validator(strict_mode: bool = True) -> FighterNameValidator:
    """Create a configured UFC fighter name validator"""
    return FighterNameValidator(strict_mode=strict_mode)


# Convenience functions
def validate_ufc_fighter_name(name: str, strict_mode: bool = True) -> str:
    """Quick validation for single fighter name"""
    validator = create_ufc_name_validator(strict_mode)
    return validator.validate_fighter_name(name)


def validate_ufc_fighter_pair(fighter_a: str, fighter_b: str, strict_mode: bool = True) -> Tuple[str, str]:
    """Quick validation for fighter pair"""
    validator = create_ufc_name_validator(strict_mode)
    return validator.validate_fighter_pair(fighter_a, fighter_b)