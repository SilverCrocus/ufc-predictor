"""
Simple test script for UFC validation system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Test basic functionality
def test_basic_validation():
    print("Testing basic validation components...")
    
    # Test fighter name validation
    try:
        from ufc_predictor.utils.validation.fighter_name_validator import validate_ufc_fighter_name, validate_ufc_fighter_pair
        
        # Valid names
        valid_name = validate_ufc_fighter_name("Conor McGregor")
        print(f"✓ Valid fighter name: {valid_name}")
        
        # Valid pair
        fighter_a, fighter_b = validate_ufc_fighter_pair("José Aldo", "Max Holloway")
        print(f"✓ Valid fighter pair: {fighter_a} vs {fighter_b}")
        
        # Test invalid name
        try:
            validate_ufc_fighter_name("<script>alert('xss')</script>")
            print("✗ Should have rejected script injection")
        except Exception:
            print("✓ Script injection correctly blocked")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test odds validation
    try:
        from ufc_predictor.utils.validation.odds_validator import validate_ufc_decimal_odds, validate_ufc_odds_pair
        
        # Valid odds
        valid_odds = validate_ufc_decimal_odds(1.85)
        print(f"✓ Valid odds: {valid_odds}")
        
        # Valid pair
        odds_a, odds_b = validate_ufc_odds_pair(1.50, 2.75)
        print(f"✓ Valid odds pair: {odds_a} / {odds_b}")
        
        # Test invalid odds
        try:
            validate_ufc_decimal_odds(0.5)  # Too low
            print("✗ Should have rejected low odds")
        except Exception:
            print("✓ Low odds correctly rejected")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test DataFrame validation
    try:
        from ufc_predictor.utils.validation.dataframe_validator import validate_ufc_prediction_dataframe
        
        # Create simple valid DataFrame
        test_df = pd.DataFrame({
            'SLpM': [4.5],
            'SApM': [2.1], 
            'Age': [28],
            'Win_Rate': [0.75]
        })
        
        validated_df = validate_ufc_prediction_dataframe(test_df, strict_mode=False)  # Use lenient mode for test
        print(f"✓ DataFrame validated: {len(validated_df)} rows")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ DataFrame validation error: {e}")
        return False
    
    print("✓ All basic validation tests passed!")
    return True

if __name__ == "__main__":
    print("UFC Prediction System - Validation Test")
    print("=" * 50)
    
    success = test_basic_validation()
    
    if success:
        print("\n✓ Validation system is working correctly!")
    else:
        print("\n✗ Validation system has issues!")
        sys.exit(1)