"""
Probability Calibration Configuration for UFC Predictor

This configuration file contains settings for probability calibration
based on research findings and best practices for MMA prediction models.
"""

# Calibration method selection
CALIBRATION_SETTINGS = {
    # Primary calibration method (recommended: 'platt' for robustness)
    'method': 'platt',  # 'platt' or 'isotonic'
    
    # Enable/disable calibration globally
    'enabled': True,
    
    # Minimum ECE threshold to trigger calibration
    'ece_threshold': 0.05,
    
    # Validation split for calibration fitting
    'validation_split': 0.2,
    
    # Minimum samples required for segment-specific calibration
    'min_samples_per_segment': 100,
    
    # Whether to use more conservative thresholds for models with odds features
    'conservative_with_odds': True
}

# Model-specific calibration settings
MODEL_SPECIFIC_SETTINGS = {
    'random_forest': {
        'method': 'platt',  # RF often well-calibrated, use simple Platt
        'priority': 'high'
    },
    'xgboost': {
        'method': 'platt',  # XGB can be overconfident, Platt works well
        'priority': 'high'
    },
    'logistic_regression': {
        'method': 'platt',  # Already fairly calibrated, light touch
        'priority': 'medium'
    }
}

# Segment-specific calibration (experimental)
SEGMENT_CALIBRATION = {
    'enabled': False,  # Start with overall calibration only
    'segment_columns': ['weight_class', 'title_fight'],  # Future implementation
    'min_samples_per_segment': 200
}

# Betting-specific calibration settings
BETTING_CALIBRATION = {
    # More aggressive calibration for betting applications
    'ece_threshold': 0.03,  # Lower threshold for betting
    
    # Kelly criterion specific settings
    'kelly_conservative_factor': 0.25,  # Use 25% of Kelly-optimal sizing
    'max_bet_percentage': 0.05,  # Never bet more than 5% of bankroll
    
    # Confidence intervals for betting decisions
    'confidence_intervals': True,
    'confidence_level': 0.95
}

# Evaluation metrics for calibration assessment
CALIBRATION_METRICS = {
    'primary_metrics': ['ece', 'brier_score'],
    'secondary_metrics': ['reliability', 'sharpness'],
    'visualization': {
        'reliability_diagram': True,
        'calibration_curve': True,
        'bins': 10
    }
}

# Research-based recommendations
RESEARCH_RECOMMENDATIONS = {
    'sample_size_21k': {
        'method': 'platt',
        'reasoning': 'Optimal for moderate sample sizes, less prone to overfitting'
    },
    'imbalanced_classes': {
        'method': 'platt',
        'reasoning': 'Handles variance differences better than isotonic regression'
    },
    'betting_optimization': {
        'optimize_for': 'calibration',
        'reasoning': '2024 research shows 69% higher returns with calibration-optimized models'
    }
}

def get_calibration_config(model_type: str = None, for_betting: bool = False) -> dict:
    """
    Get calibration configuration for a specific use case.
    
    Args:
        model_type: Specific model type ('random_forest', 'xgboost', etc.)
        for_betting: Whether this is for betting applications
        
    Returns:
        Configuration dictionary
    """
    config = CALIBRATION_SETTINGS.copy()
    
    # Apply model-specific settings
    if model_type and model_type in MODEL_SPECIFIC_SETTINGS:
        model_config = MODEL_SPECIFIC_SETTINGS[model_type]
        config.update(model_config)
    
    # Apply betting-specific settings
    if for_betting:
        config.update(BETTING_CALIBRATION)
    
    return config

def print_calibration_recommendations():
    """Print research-based calibration recommendations."""
    print("🎯 Probability Calibration Recommendations for UFC Predictor")
    print("=" * 65)
    print()
    print("Based on 2024 research and ~21,000 sample dataset analysis:")
    print()
    print("✅ RECOMMENDED APPROACH:")
    print("   • Method: Platt Scaling (sigmoid)")
    print("   • Reason: Optimal for moderate sample sizes, robust")
    print("   • Sample Size: Well-suited for your ~21K samples")
    print("   • Imbalanced Classes: Handles variance better than isotonic")
    print()
    print("📊 KEY RESEARCH FINDINGS:")
    print("   • Calibration-optimized models: +69% higher betting returns")
    print("   • Overconfident probabilities: Major cause of betting losses")
    print("   • Professional bettors: Use only 25% of Kelly-optimal sizing")
    print()
    print("⚙️  IMPLEMENTATION PRIORITIES:")
    print("   1. Enable Platt scaling for all models")
    print("   2. Validate with Expected Calibration Error (ECE)")
    print("   3. Test with conservative Kelly criterion")
    print("   4. Monitor reliability diagrams")
    print()
    print("💡 PRODUCTION SETTINGS:")
    print(f"   • ECE Threshold: {CALIBRATION_SETTINGS['ece_threshold']}")
    print(f"   • Validation Split: {CALIBRATION_SETTINGS['validation_split']}")
    print(f"   • Kelly Conservative Factor: {BETTING_CALIBRATION['kelly_conservative_factor']}")
    print(f"   • Max Bet Percentage: {BETTING_CALIBRATION['max_bet_percentage']}")

if __name__ == "__main__":
    print_calibration_recommendations()