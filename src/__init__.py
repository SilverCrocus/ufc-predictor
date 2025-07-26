"""UFC Predictor package for machine learning-based fight predictions."""

from .feature_engineering import (
    engineer_features_final,
    create_differential_features,
    merge_fight_data,
    prepare_modeling_data
)

from .model_training import UFCModelTrainer, train_complete_pipeline

from .prediction import UFCPredictor, create_predictor
from .ufc_fight_predictor import UFCFightPredictor, create_ufc_predictor

__version__ = "1.0.0"
__all__ = [
    "engineer_features_final",
    "create_differential_features", 
    "merge_fight_data",
    "prepare_modeling_data",
    "UFCModelTrainer",
    "train_complete_pipeline",
    "UFCPredictor",
    "create_predictor",
    "UFCFightPredictor",
    "create_ufc_predictor"
]