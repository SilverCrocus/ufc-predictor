"""
Strict Validation Example for UFC Prediction System

Demonstrates how to use the comprehensive validation framework
with strict error handling and no fallback mechanisms.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import validation components
from src.validation import (
    validate_ufc_fighter_pair,
    validate_ufc_decimal_odds,
    validate_ufc_odds_pair, 
    validate_ufc_prediction_dataframe,
    create_strict_ufc_validator,
    FighterNameValidationError,
    OddsValidationError,
    DataFrameValidationError
)


def example_fighter_name_validation():
    """Demonstrate strict fighter name validation"""
    print("\n=== Fighter Name Validation Examples ===")
    
    # Valid cases
    valid_names = [
        ("Conor McGregor", "Nate Diaz"),
        ("José Aldo", "Max Holloway"),
        ("Khabib Nurmagomedov", "Tony Ferguson"),
        ("Jon \"Bones\" Jones", "Daniel Cormier")
    ]
    
    for fighter_a, fighter_b in valid_names:
        try:
            validated_a, validated_b = validate_ufc_fighter_pair(fighter_a, fighter_b, strict_mode=True)
            print(f"✓ Valid: {validated_a} vs {validated_b}")
        except FighterNameValidationError as e:
            print(f"✗ Validation failed: {e}")
    
    # Invalid cases that should be rejected
    invalid_cases = [
        ("", "Valid Fighter"),           # Empty name
        ("Valid Fighter", ""),           # Empty name
        ("Fighter<script>", "Normal"),   # Script injection attempt
        ("SELECT * FROM", "Normal"),     # SQL injection attempt
        ("Fighter'; DROP", "Normal"),    # SQL injection attempt
        ("Fighter\x00", "Normal"),       # Null byte injection
        ("A" * 150, "Normal"),          # Excessively long name
        ("Same Fighter", "Same Fighter") # Identical names
    ]
    
    for fighter_a, fighter_b in invalid_cases:
        try:
            validate_ufc_fighter_pair(fighter_a, fighter_b, strict_mode=True)
            print(f"✗ Should have failed: {fighter_a} vs {fighter_b}")
        except FighterNameValidationError:
            print(f"✓ Correctly rejected: {fighter_a[:20]}... vs {fighter_b[:20]}...")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")


def example_odds_validation():
    """Demonstrate strict odds validation"""
    print("\n=== Odds Validation Examples ===")
    
    # Valid odds cases
    valid_odds = [
        (1.50, 2.75),  # Normal odds
        (1.01, 15.0),  # Extreme favorite vs underdog
        (2.20, 1.65),  # Close fight
        ("1.85", "1.95") # String input
    ]
    
    for odds_a, odds_b in valid_odds:
        try:
            validated_a, validated_b = validate_ufc_odds_pair(odds_a, odds_b)
            print(f"✓ Valid odds: {validated_a} / {validated_b}")
        except OddsValidationError as e:
            print(f"✗ Validation failed: {e}")
    
    # Invalid odds cases
    invalid_odds = [
        (0.5, 2.0),     # Odds below minimum
        (1.5, 100.0),   # Odds above maximum
        (-1.5, 2.0),    # Negative odds
        (float('inf'), 2.0),  # Infinite odds
        (float('nan'), 2.0),  # NaN odds
        ("invalid", 2.0),     # Non-numeric string
        (1.5, 1.5),     # Identical odds (suspicious)
    ]
    
    for odds_a, odds_b in invalid_odds:
        try:
            validate_ufc_odds_pair(odds_a, odds_b)
            print(f"✗ Should have failed: {odds_a} / {odds_b}")
        except OddsValidationError:
            print(f"✓ Correctly rejected: {odds_a} / {odds_b}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")


def example_dataframe_validation():
    """Demonstrate strict DataFrame validation"""
    print("\n=== DataFrame Validation Examples ===")
    
    # Create valid test DataFrame
    valid_features = {
        'SLpM': [4.5, 3.2],
        'SApM': [2.1, 4.8], 
        'TD Avg.': [1.2, 0.8],
        'Sub. Avg.': [0.5, 1.1],
        'Str. Acc.': [0.65, 0.58],
        'Str. Def': [0.72, 0.61],
        'TD Acc.': [0.45, 0.38],
        'TD Def.': [0.85, 0.79],
        'Height_inches': [70, 72],
        'Weight_lbs': [155, 170],
        'Reach_inches': [68, 71],
        'Age': [28, 31],
        'Wins': [15, 12],
        'Losses': [3, 5],
        'Draws': [0, 1],
        'Total_Fights': [18, 18],
        'Win_Rate': [0.833, 0.667],
        'STANCE_Orthodox': [1, 0],
        'STANCE_Southpaw': [0, 1],
        'STANCE_Switch': [0, 0],
        'STANCE_Unknown': [0, 0]
    }
    
    # Valid DataFrame
    try:
        valid_df = pd.DataFrame(valid_features)
        validated_df = validate_ufc_prediction_dataframe(valid_df, strict_mode=True)
        print(f"✓ Valid DataFrame: {len(validated_df)} rows, {len(validated_df.columns)} columns")
    except DataFrameValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid DataFrames
    print("\nTesting invalid DataFrames...")
    
    # DataFrame with missing required features
    try:
        invalid_df = pd.DataFrame({'random_feature': [1, 2]})
        validate_ufc_prediction_dataframe(invalid_df, strict_mode=True)
        print("✗ Should have failed: missing features")
    except DataFrameValidationError:
        print("✓ Correctly rejected: missing required features")
    
    # DataFrame with out-of-range values
    try:
        invalid_features = valid_features.copy()
        invalid_features['Age'] = [150, 200]  # Impossible ages
        invalid_df = pd.DataFrame(invalid_features)
        validate_ufc_prediction_dataframe(invalid_df, strict_mode=True)
        print("✗ Should have failed: out-of-range values")
    except DataFrameValidationError:
        print("✓ Correctly rejected: out-of-range values")
    
    # DataFrame with wrong data types
    try:
        invalid_features = valid_features.copy()
        invalid_features['SLpM'] = ['text', 'invalid']  # String in numeric field
        invalid_df = pd.DataFrame(invalid_features)
        validate_ufc_prediction_dataframe(invalid_df, strict_mode=True)
        print("✗ Should have failed: wrong data types")
    except DataFrameValidationError:
        print("✓ Correctly rejected: wrong data types")


def example_unified_validation():
    """Demonstrate unified validation suite"""
    print("\n=== Unified Validation Suite Example ===")
    
    validator = create_strict_ufc_validator()
    
    # Prepare test data
    fighter_a = "Conor McGregor"
    fighter_b = "Nate Diaz"
    
    # Create valid feature DataFrame
    feature_data = {
        'SLpM_diff': [1.3],
        'SApM_diff': [-2.7],
        'TD Avg._diff': [0.4],
        'Sub. Avg._diff': [-0.6],
        'Str. Acc._diff': [0.07],
        'Str. Def_diff': [0.11],
        'TD Acc._diff': [0.07],
        'TD Def._diff': [0.06],
        'Height_inches_diff': [-2],
        'Weight_lbs_diff': [-15],
        'Reach_inches_diff': [-3],
        'Age_diff': [-3],
        'Win_Rate_diff': [0.166]
    }
    
    features_df = pd.DataFrame(feature_data)
    model_columns = list(feature_data.keys())
    
    # Valid odds data
    odds_data = {
        'fighter_a': fighter_a,
        'fighter_b': fighter_b,
        'decimal_odds_a': 1.85,
        'decimal_odds_b': 1.95
    }
    
    try:
        result = validator.validate_prediction_input(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            feature_dataframe=features_df,
            model_columns=model_columns,
            odds_data=odds_data
        )
        
        if result.is_valid:
            print(f"✓ Unified validation passed!")
            print(f"  Validated fighters: {result.validated_data['fighter_a']} vs {result.validated_data['fighter_b']}")
            print(f"  Feature shape: {result.validated_data['features'].shape}")
            print(f"  Odds validated: {result.validated_data['odds']['validated_odds_a']} / {result.validated_data['odds']['validated_odds_b']}")
            if result.warnings:
                print(f"  Warnings: {result.warnings}")
        else:
            print(f"✗ Unified validation failed: {result.errors}")
            
    except Exception as e:
        print(f"✗ Validation error: {e}")


def example_attack_scenarios():
    """Demonstrate protection against various attack scenarios"""
    print("\n=== Security Attack Scenarios ===")
    
    # SQL Injection attempts
    sql_attacks = [
        "'; DROP TABLE fighters; --",
        "1' OR '1'='1",
        "UNION SELECT * FROM users",
    ]
    
    print("Testing SQL injection protection:")
    for attack in sql_attacks:
        try:
            validate_ufc_fighter_pair(attack, "Normal Fighter", strict_mode=True)
            print(f"✗ SQL injection not blocked: {attack}")
        except FighterNameValidationError:
            print(f"✓ SQL injection blocked: {attack[:20]}...")
    
    # Script injection attempts
    script_attacks = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "vbscript:msgbox('xss')",
    ]
    
    print("\nTesting script injection protection:")
    for attack in script_attacks:
        try:
            validate_ufc_fighter_pair(attack, "Normal Fighter", strict_mode=True)
            print(f"✗ Script injection not blocked: {attack}")
        except FighterNameValidationError:
            print(f"✓ Script injection blocked: {attack[:20]}...")
    
    # Path traversal attempts
    path_attacks = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/shadow",
    ]
    
    print("\nTesting path traversal protection:")
    for attack in path_attacks:
        try:
            validate_ufc_fighter_pair(attack, "Normal Fighter", strict_mode=True)
            print(f"✗ Path traversal not blocked: {attack}")
        except FighterNameValidationError:
            print(f"✓ Path traversal blocked: {attack}")
    
    # Odds manipulation attempts
    odds_attacks = [
        (0.01, 1.5),    # Impossible odds
        (1.5, 1000.0),  # Extreme odds
        (-5, 2.0),      # Negative odds
    ]
    
    print("\nTesting odds manipulation protection:")
    for odds_a, odds_b in odds_attacks:
        try:
            validate_ufc_odds_pair(odds_a, odds_b)
            print(f"✗ Odds manipulation not blocked: {odds_a}/{odds_b}")
        except OddsValidationError:
            print(f"✓ Odds manipulation blocked: {odds_a}/{odds_b}")


def main():
    """Run all validation examples"""
    print("UFC Prediction System - Strict Validation Examples")
    print("=" * 60)
    
    try:
        example_fighter_name_validation()
        example_odds_validation()
        example_dataframe_validation()
        example_unified_validation()
        example_attack_scenarios()
        
        print("\n" + "=" * 60)
        print("All validation examples completed!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()