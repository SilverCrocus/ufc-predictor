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