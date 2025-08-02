"""Configuration settings for UFC predictor models."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
SRC_DIR = PROJECT_ROOT / "src"

# Data file paths
RAW_FIGHTERS_DATA = DATA_DIR / "ufc_fighters_raw.csv"
ENGINEERED_FIGHTERS_DATA = DATA_DIR / "ufc_fighters_engineered.csv"
CORRECTED_FIGHTERS_DATA = MODEL_DIR / "ufc_fighters_engineered_corrected.csv"
FIGHTS_DATA = DATA_DIR / "ufc_fights.csv"
FIGHT_DATASET_WITH_DIFFS = MODEL_DIR / "ufc_fight_dataset_with_diffs.csv"

# Model file paths
RF_MODEL_PATH = MODEL_DIR / "ufc_random_forest_model.joblib"
RF_TUNED_MODEL_PATH = MODEL_DIR / "ufc_random_forest_model_tuned.joblib"
XGBOOST_MODEL_PATH = MODEL_DIR / "ufc_xgboost_model.joblib"
XGBOOST_TUNED_MODEL_PATH = MODEL_DIR / "ufc_xgboost_model_tuned.joblib"
ENSEMBLE_MODEL_DIR = MODEL_DIR / "ensemble_models"
MODEL_COLUMNS_PATH = MODEL_DIR / "model_training_columns.json"

# Feature engineering settings
PERCENTAGE_COLUMNS = ['Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.']
NUMERIC_COLUMNS = ['SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.']
REFERENCE_DATE = '2025-06-17'

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest default parameters
RF_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Random Forest hyperparameter tuning grid
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Hyperparameter tuning settings
HYPERPARAMETER_TUNING = {
    'n_iter': 100,
    'cv': 3,
    'verbose': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Logistic Regression settings
LR_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000
}

# XGBoost default parameters optimized for UFC prediction
XGBOOST_DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist',  # Faster training
    'verbose': False,
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 1.5
}

# XGBoost hyperparameter tuning grid
XGBOOST_PARAM_GRID = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# Ensemble configuration for Phase 2A Enhanced ML Pipeline
ENSEMBLE_CONFIG = {
    'weights': {
        'random_forest': 0.40,
        'xgboost': 0.35,
        'neural_network': 0.25
    },
    'confidence_threshold': 0.6,
    'bootstrap_samples': 100,
    'enable_data_quality_scoring': True,
    'enable_confidence_intervals': True,
    'confidence_level': 0.95,
    'min_prediction_confidence': 0.5,
    'max_memory_mb': 4096,  # Memory limit in MB
    'max_failure_rate': 0.05,  # Maximum allowed failure rate (5%)
    'enable_strict_validation': True,  # Enable strict input validation
    'performance_targets': {
        'single_prediction_max_ms': 2000,  # Max 2 seconds per prediction
        'bootstrap_max_ms': 10000,  # Max 10 seconds for confidence intervals
        'memory_warning_threshold': 0.9  # Warn at 90% memory usage
    }
}

# Columns to drop for modeling
COLUMNS_TO_DROP = [
    'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time',
    'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
    'blue_Name', 'red_Name', 'blue_is_winner'
]

# Physical attribute parsing
ORIGINAL_COLUMNS_TO_DROP = ['Height', 'Weight', 'Reach', 'Record', 'DOB', 'STANCE']

# Webscraper settings (from webscraper/config.py)
WEBSCRAPER_CONFIG = {
    'base_url': 'http://ufcstats.com',
    'fighters_url': 'http://ufcstats.com/statistics/fighters',
    'delay_between_requests': 1,  # seconds
    'max_retries': 3,
    'timeout': 30  # seconds
}

# Display settings
DISPLAY_PRECISION = 2  # decimal places for percentages
TOP_FEATURES_TO_SHOW = 15

# Quota Management Settings (Phase 2A)
QUOTA_CONFIG = {
    'config_path': PROJECT_ROOT / 'config' / 'quota_config.json',
    'default_daily_limit': 500,
    'default_priority': 'MEDIUM',
    'enable_fallback': True,
    'cost_threshold_usd': 50.0
}

# Enhanced Odds Service Settings
ENHANCED_ODDS_CONFIG = {
    'storage_base_path': PROJECT_ROOT / 'odds',
    'enable_tab_fallback': True,
    'enable_cache_fallback': True,
    'confidence_threshold': 0.7,
    'max_cache_age_hours': 24
}

# Data Source Priorities (for hybrid system)
DATA_SOURCE_PRIORITIES = {
    'the_odds_api': 1,      # Highest priority
    'tab_scraper': 2,       # Fallback option
    'cached_data': 3,       # Last resort
    'manual_entry': 4       # Manual override
}