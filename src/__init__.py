"""UFC Predictor package for machine learning-based fight predictions."""

from .ufc_predictor.data.feature_engineering import (
    engineer_features_final,
    create_differential_features,
    merge_fight_data,
    prepare_modeling_data
)

from .ufc_predictor.models.model_training import UFCModelTrainer, train_complete_pipeline

from .ufc_predictor.core.prediction import UFCPredictor, create_predictor

__version__ = "1.0.0"
__all__ = [
    "engineer_features_final",
    "create_differential_features", 
    "merge_fight_data",
    "prepare_modeling_data",
    "UFCModelTrainer",
    "train_complete_pipeline",
    "UFCPredictor",
    "create_predictor"
]