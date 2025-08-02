"""
Common Utilities for UFC Predictor System
=========================================

This module consolidates duplicate code found throughout the system,
providing centralized implementations of commonly used functions.

Key consolidations:
- Odds conversion functions (American ↔ Decimal ↔ Probability)
- Fighter name matching and similarity functions
- Date and time utilities
- File I/O helpers
- Data validation functions
- String processing utilities

Usage:
    from ufc_predictor.utils.common_utilities import OddsConverter, NameMatcher, FileUtils
    
    # Convert odds
    decimal = OddsConverter.american_to_decimal(-150)
    probability = OddsConverter.decimal_to_probability(2.5)
    
    # Match fighter names
    similarity = NameMatcher.calculate_similarity("Jon Jones", "Jonathan Jones")
    best_match = NameMatcher.find_best_match("Jon Jones", fighter_list)
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime, date
from difflib import SequenceMatcher
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class OddsConverter:
    """Centralized odds conversion utilities - eliminates duplication across modules"""
    
    @staticmethod
    def american_to_decimal(american_odds: Union[int, float]) -> float:
        """
        Convert American odds to decimal format
        
        Examples:
            +150 -> 2.50
            -150 -> 1.67
        """
        american_odds = float(american_odds)
        
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """
        Convert decimal odds to American format
        
        Examples:
            2.50 -> +150
            1.67 -> -150
        """
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    @staticmethod
    def american_to_probability(american_odds: Union[int, float]) -> float:
        """
        Convert American odds to implied probability
        
        Examples:
            +150 -> 0.40 (40%)
            -150 -> 0.60 (60%)
        """
        american_odds = float(american_odds)
        
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    @staticmethod
    def decimal_to_probability(decimal_odds: float) -> float:
        """
        Convert decimal odds to implied probability
        
        Example:
            2.50 -> 0.40 (40%)
        """
        return 1.0 / decimal_odds
    
    @staticmethod
    def probability_to_decimal(probability: float) -> float:
        """
        Convert probability to decimal odds
        
        Example:
            0.40 -> 2.50
        """
        if probability <= 0 or probability >= 1:
            raise ValueError("Probability must be between 0 and 1")
        
        return 1.0 / probability
    
    @staticmethod
    def probability_to_american(probability: float) -> int:
        """
        Convert probability to American odds
        
        Example:
            0.40 -> +150
        """
        decimal = OddsConverter.probability_to_decimal(probability)
        return OddsConverter.decimal_to_american(decimal)
    
    @staticmethod
    def calculate_expected_value(probability: float, decimal_odds: float) -> float:
        """
        Calculate expected value of a bet
        
        Formula: EV = (Probability × Decimal_Odds) - 1
        """
        return (probability * decimal_odds) - 1
    
    @staticmethod
    def calculate_kelly_fraction(probability: float, decimal_odds: float) -> float:
        """
        Calculate Kelly Criterion fraction for optimal bet sizing
        
        Formula: f = (bp - q) / b
        Where: b = decimal_odds - 1, p = probability, q = 1 - probability
        """
        if probability <= 0 or probability >= 1:
            return 0.0
        
        b = decimal_odds - 1
        p = probability
        q = 1 - probability
        
        kelly_fraction = (b * p - q) / b
        
        # Return 0 if Kelly fraction is negative (no bet)
        return max(0.0, kelly_fraction)


class NameMatcher:
    """Centralized fighter name matching utilities"""
    
    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """
        Calculate similarity score between two names using SequenceMatcher
        
        Returns:
            Float between 0.0 (no similarity) and 1.0 (identical)
        """
        if not name1 or not name2:
            return 0.0
        
        return SequenceMatcher(None, name1.lower().strip(), name2.lower().strip()).ratio()
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize fighter name for better matching
        
        - Converts to lowercase
        - Removes extra whitespace
        - Handles common name variations
        """
        if not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes that might interfere with matching
        name = re.sub(r'\b(jr\.?|sr\.?|iii|ii)\b', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove quotes and other punctuation that might interfere
        name = re.sub(r'["\']', '', name)
        
        return name.strip()
    
    @staticmethod
    def find_best_match(target_name: str, candidate_names: List[str], threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Find the best matching name from a list of candidates
        
        Args:
            target_name: Name to match
            candidate_names: List of potential matches
            threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of (best_match, similarity_score) or None if no match found
        """
        if not target_name or not candidate_names:
            return None
        
        normalized_target = NameMatcher.normalize_name(target_name)
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_names:
            if not candidate:
                continue
                
            normalized_candidate = NameMatcher.normalize_name(candidate)
            score = NameMatcher.calculate_similarity(normalized_target, normalized_candidate)
            
            if score > best_score and score >= threshold:
                best_match = candidate
                best_score = score
        
        return (best_match, best_score) if best_match else None
    
    @staticmethod
    def find_all_matches(target_name: str, candidate_names: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find all names that match above the threshold, sorted by similarity
        """
        if not target_name or not candidate_names:
            return []
        
        normalized_target = NameMatcher.normalize_name(target_name)
        matches = []
        
        for candidate in candidate_names:
            if not candidate:
                continue
                
            normalized_candidate = NameMatcher.normalize_name(candidate)
            score = NameMatcher.calculate_similarity(normalized_target, normalized_candidate)
            
            if score >= threshold:
                matches.append((candidate, score))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


@dataclass
class BetInfo:
    """Standardized bet information structure"""
    fighter: str
    opponent: str
    decimal_odds: float
    american_odds: int
    probability: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    expected_profit: float
    confidence: float = 1.0  # Name matching confidence
    source: str = "unknown"
    timestamp: Optional[datetime] = None


class BettingCalculator:
    """Centralized betting calculations and utilities"""
    
    @staticmethod
    def calculate_bet_info(
        fighter: str,
        opponent: str, 
        model_probability: float,
        decimal_odds: float,
        bankroll: float,
        max_bet_percentage: float = 0.05,
        source: str = "unknown"
    ) -> BetInfo:
        """
        Calculate complete bet information from basic inputs
        """
        american_odds = OddsConverter.decimal_to_american(decimal_odds)
        market_probability = OddsConverter.decimal_to_probability(decimal_odds)
        expected_value = OddsConverter.calculate_expected_value(model_probability, decimal_odds)
        kelly_fraction = OddsConverter.calculate_kelly_fraction(model_probability, decimal_odds)
        
        # Conservative position sizing - use smaller of Kelly and max percentage
        safe_fraction = min(kelly_fraction, max_bet_percentage)
        recommended_stake = bankroll * safe_fraction
        expected_profit = recommended_stake * expected_value
        
        return BetInfo(
            fighter=fighter,
            opponent=opponent,
            decimal_odds=decimal_odds,
            american_odds=american_odds,
            probability=model_probability,
            expected_value=expected_value,
            kelly_fraction=kelly_fraction,
            recommended_stake=max(0, recommended_stake),
            expected_profit=max(0, expected_profit),
            source=source,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def calculate_multi_bet_ev(
        individual_probabilities: List[float],
        individual_odds: List[float],
        correlation_penalty: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Calculate multi-bet expected value, combined probability, and combined odds
        
        Returns:
            Tuple of (combined_probability, combined_odds, expected_value)
        """
        if len(individual_probabilities) != len(individual_odds):
            raise ValueError("Probabilities and odds lists must have same length")
        
        # Calculate combined probability (all bets must win)
        combined_prob = 1.0
        for prob in individual_probabilities:
            combined_prob *= prob
        
        # Apply correlation penalty for same-event bets
        if correlation_penalty > 0:
            combined_prob *= (1 - correlation_penalty)
        
        # Calculate combined odds (multiply all odds)
        combined_odds = 1.0
        for odds in individual_odds:
            combined_odds *= odds
        
        # Calculate expected value
        expected_value = (combined_prob * combined_odds) - 1
        
        return combined_prob, combined_odds, expected_value


class DateTimeUtils:
    """Date and time utilities used across the system"""
    
    @staticmethod
    def parse_ufc_date(date_string: str) -> Optional[date]:
        """
        Parse UFC date strings in various formats
        
        Examples:
            "Jul 19, 1987" -> date(1987, 7, 19)
            "August 19, 1982" -> date(1982, 8, 19)
        """
        if not date_string or pd.isna(date_string):
            return None
        
        # Common UFC date formats
        formats = [
            '%b %d, %Y',    # Jul 19, 1987
            '%B %d, %Y',    # August 19, 1982
            '%m/%d/%Y',     # 07/19/1987
            '%Y-%m-%d'      # 1987-07-19
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_string.strip(), fmt)
                return parsed.date()
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date string: {date_string}")
        return None
    
    @staticmethod
    def calculate_age(birth_date: Union[str, date], reference_date: Union[str, date] = None) -> Optional[float]:
        """
        Calculate age from birth date
        
        Args:
            birth_date: Birth date as string or date object
            reference_date: Reference date (defaults to today)
            
        Returns:
            Age in years as float, or None if calculation fails
        """
        if isinstance(birth_date, str):
            birth_date = DateTimeUtils.parse_ufc_date(birth_date)
        
        if birth_date is None:
            return None
        
        if reference_date is None:
            reference_date = date.today()
        elif isinstance(reference_date, str):
            reference_date = DateTimeUtils.parse_ufc_date(reference_date)
        
        if reference_date is None:
            return None
        
        # Calculate age in years with decimal precision
        days_difference = (reference_date - birth_date).days
        age_years = days_difference / 365.25  # Account for leap years
        
        return round(age_years, 1)
    
    @staticmethod
    def format_timestamp(timestamp: Optional[datetime] = None) -> str:
        """Format timestamp for consistent display"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    
    @staticmethod
    def is_recent(timestamp: datetime, max_age_minutes: int = 60) -> bool:
        """Check if timestamp is recent (within specified minutes)"""
        age = datetime.now() - timestamp
        return age.total_seconds() / 60 <= max_age_minutes


class StringUtils:
    """String processing utilities used throughout the system"""
    
    @staticmethod
    def clean_numeric_string(value: str, remove_chars: str = None) -> str:
        """
        Clean numeric strings by removing specified characters
        
        Examples:
            "185 lbs." -> "185" (if remove_chars=" lbs.")
            "75%" -> "75" (if remove_chars="%")
        """
        if not value or pd.isna(value):
            return ""
        
        value = str(value).strip()
        
        if remove_chars:
            for char in remove_chars:
                value = value.replace(char, "")
        
        return value.strip()
    
    @staticmethod
    def parse_fight_record(record_string: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Parse fight record string like "27-1-0" into (wins, losses, draws)
        
        Returns:
            Tuple of (wins, losses, draws) or (None, None, None) if parsing fails
        """
        if not record_string or pd.isna(record_string):
            return (None, None, None)
        
        try:
            # Remove any extra text and split on hyphens
            clean_record = re.sub(r'[^\d-]', '', str(record_string))
            parts = clean_record.split('-')
            
            wins = int(parts[0]) if len(parts) > 0 and parts[0] else 0
            losses = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            draws = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            
            return (wins, losses, draws)
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not parse fight record '{record_string}': {e}")
            return (None, None, None)
    
    @staticmethod
    def parse_height_to_inches(height_string: str) -> Optional[float]:
        """
        Parse height string like "6' 4\"" to inches
        
        Returns:
            Height in inches or None if parsing fails
        """
        if not height_string or pd.isna(height_string):
            return None
        
        try:
            # Match patterns like "6' 4\"" or "6'4\""
            pattern = r"(\d+)'\s*(\d+)\"?"
            match = re.search(pattern, str(height_string))
            
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return feet * 12 + inches
            
        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse height '{height_string}': {e}")
        
        return None
    
    @staticmethod
    def standardize_method_name(method: str) -> Optional[str]:
        """
        Standardize fight method names to consistent categories
        
        Returns:
            Standardized method name or None if not recognized
        """
        if not method or pd.isna(method):
            return None
        
        method_upper = str(method).upper()
        
        if 'KO' in method_upper or 'TKO' in method_upper or 'KNOCKOUT' in method_upper:
            return 'KO/TKO'
        elif 'SUB' in method_upper or 'SUBMISSION' in method_upper:
            return 'Submission'  
        elif 'DEC' in method_upper or 'DECISION' in method_upper:
            return 'Decision'
        else:
            return None


class FileUtils:
    """File I/O utilities with error handling"""
    
    @staticmethod
    def safe_read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Safely read JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    def safe_write_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
        """Safely write JSON file with error handling"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def generate_cache_key(data: Any) -> str:
        """Generate consistent cache key from data"""
        data_str = str(data) if not isinstance(data, str) else data
        return hashlib.md5(data_str.encode()).hexdigest()
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def is_valid_probability(value: float) -> bool:
        """Check if value is a valid probability (0 <= p <= 1)"""
        return isinstance(value, (int, float)) and 0 <= value <= 1
    
    @staticmethod
    def is_valid_odds(decimal_odds: float) -> bool:
        """Check if decimal odds are valid (> 1.0)"""
        return isinstance(decimal_odds, (int, float)) and decimal_odds > 1.0
    
    @staticmethod
    def is_valid_bankroll(bankroll: float) -> bool:
        """Check if bankroll is valid (positive)"""
        return isinstance(bankroll, (int, float)) and bankroll > 0
    
    @staticmethod
    def validate_bet_inputs(
        probability: float, 
        decimal_odds: float, 
        bankroll: float
    ) -> List[str]:
        """
        Validate inputs for bet calculations
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not ValidationUtils.is_valid_probability(probability):
            errors.append("Probability must be between 0 and 1")
        
        if not ValidationUtils.is_valid_odds(decimal_odds):
            errors.append("Decimal odds must be greater than 1.0")
        
        if not ValidationUtils.is_valid_bankroll(bankroll):
            errors.append("Bankroll must be positive")
        
        return errors


# Convenience functions that combine utilities for common use cases
def calculate_complete_bet_analysis(
    fighter: str,
    opponent: str,
    model_probability: float,
    decimal_odds: float,
    bankroll: float,
    max_bet_percentage: float = 0.05
) -> Optional[BetInfo]:
    """
    Convenience function for complete bet analysis with validation
    """
    # Validate inputs
    errors = ValidationUtils.validate_bet_inputs(model_probability, decimal_odds, bankroll)
    if errors:
        logger.error(f"Invalid bet inputs for {fighter} vs {opponent}: {', '.join(errors)}")
        return None
    
    # Calculate bet info
    return BettingCalculator.calculate_bet_info(
        fighter, opponent, model_probability, decimal_odds, 
        bankroll, max_bet_percentage
    )


def convert_all_odds_formats(odds: Union[int, float], input_format: str = "american") -> Dict[str, Union[int, float]]:
    """
    Convert odds to all formats for display purposes
    
    Args:
        odds: Odds value
        input_format: "american", "decimal", or "probability"
        
    Returns:
        Dictionary with all odds formats
    """
    try:
        if input_format == "american":
            decimal = OddsConverter.american_to_decimal(odds)
            probability = OddsConverter.american_to_probability(odds)
            american = int(odds)
        elif input_format == "decimal":
            american = OddsConverter.decimal_to_american(odds)
            probability = OddsConverter.decimal_to_probability(odds)
            decimal = float(odds)
        elif input_format == "probability":
            decimal = OddsConverter.probability_to_decimal(odds)
            american = OddsConverter.probability_to_american(odds)
            probability = float(odds)
        else:
            raise ValueError(f"Invalid input format: {input_format}")
        
        return {
            "american": american,
            "decimal": round(decimal, 2),
            "probability": round(probability, 4),
            "percentage": f"{probability*100:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Failed to convert odds {odds} from {input_format}: {e}")
        return {}


if __name__ == "__main__":
    print("🚀 Common Utilities Demo")
    print("=" * 50)
    
    # Demo odds conversion
    print("📊 ODDS CONVERSION:")
    american_odds = -150
    decimal_odds = OddsConverter.american_to_decimal(american_odds)
    probability = OddsConverter.american_to_probability(american_odds)
    
    print(f"   American: {american_odds}")
    print(f"   Decimal: {decimal_odds:.2f}")
    print(f"   Probability: {probability:.3f} ({probability*100:.1f}%)")
    
    # Demo name matching
    print(f"\n🥊 NAME MATCHING:")
    target = "Jon Jones"
    candidates = ["Jonathan Jones", "John Jones", "Jon Bones Jones", "Stipe Miocic"]
    
    best_match = NameMatcher.find_best_match(target, candidates)
    if best_match:
        print(f"   Target: '{target}'")
        print(f"   Best match: '{best_match[0]}' (similarity: {best_match[1]:.3f})")
    
    # Demo betting calculation
    print(f"\n💰 BETTING CALCULATION:")
    bet_info = calculate_complete_bet_analysis(
        "Jon Jones", "Stipe Miocic",
        model_probability=0.70,
        decimal_odds=1.50,
        bankroll=1000.0
    )
    
    if bet_info:
        print(f"   Fighter: {bet_info.fighter}")
        print(f"   Model probability: {bet_info.probability:.2f}")
        print(f"   Decimal odds: {bet_info.decimal_odds}")
        print(f"   Expected value: {bet_info.expected_value:.3f}")
        print(f"   Recommended stake: ${bet_info.recommended_stake:.2f}")
        print(f"   Expected profit: ${bet_info.expected_profit:.2f}")
    
    print(f"\n✅ Common utilities demo completed")