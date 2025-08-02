"""
Enhanced UFC Prediction Models
=============================

This module implements advanced ML approaches to improve upon the existing Random Forest baseline.
Includes XGBoost, ensemble methods, enhanced feature engineering, and ELO rating integration.

Key Improvements:
- Gradient boosting with XGBoost/LightGBM
- Advanced feature engineering (interactions, time windows)
- ELO rating system integration
- Multi-model ensemble approaches
- Proper temporal validation to prevent data leakage

Usage:
    from ufc_predictor.models.enhanced_modeling import EnhancedUFCPredictor
    
    predictor = EnhancedUFCPredictor()
    predictor.train_models(X_train, y_train)
    predictions = predictor.predict(X_test)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import logging
from pathlib import Path

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Configuration and utilities
from ufc_predictor.utils.unified_config import config
from ufc_predictor.utils.logging_config import get_logger, UFCPredictorError
from ufc_predictor.utils.common_utilities import ValidationUtils

logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """Store model performance metrics"""
    accuracy: float
    roc_auc: float
    classification_report: str
    cross_val_scores: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    training_time: Optional[float] = None


@dataclass
class ELORating:
    """ELO rating for a fighter"""
    fighter_name: str
    rating: float = 1500.0
    k_factor: float = 32.0
    last_updated: Optional[datetime] = None
    fights_count: int = 0


class ELOSystem:
    """
    ELO rating system implementation for UFC fighters
    
    Based on professional MMA rating systems, provides dynamic ratings
    that update after each fight and decay over time for inactive fighters.
    """
    
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0, decay_months: int = 18):
        self.initial_rating = initial_rating
        self.base_k_factor = k_factor
        self.decay_months = decay_months
        self.ratings: Dict[str, ELORating] = {}
        
    def get_k_factor(self, fighter_rating: float, opponent_rating: float, fights_count: int) -> float:
        """
        Calculate dynamic K-factor based on rating difference and experience
        
        Higher K-factor for:
        - New fighters (fewer than 10 fights)
        - Significant rating differences
        - Upset victories
        """
        base_k = self.base_k_factor
        
        # Reduce K-factor for experienced fighters
        if fights_count > 10:
            base_k *= 0.8
        elif fights_count > 20:
            base_k *= 0.6
            
        # Increase K-factor for significant rating differences (potential upsets)
        rating_diff = abs(fighter_rating - opponent_rating)
        if rating_diff > 200:
            base_k *= 1.3
        elif rating_diff > 400:
            base_k *= 1.5
            
        return min(base_k, 50.0)  # Cap maximum K-factor
    
    def get_rating(self, fighter_name: str, reference_date: Optional[datetime] = None) -> float:
        """Get current ELO rating for a fighter with time decay"""
        if fighter_name not in self.ratings:
            return self.initial_rating
            
        rating_obj = self.ratings[fighter_name]
        
        # Apply time decay for inactive fighters
        if reference_date and rating_obj.last_updated:
            months_inactive = (reference_date - rating_obj.last_updated).days / 30.44
            if months_inactive > self.decay_months:
                # Decay toward mean (1500) over time
                decay_factor = min(months_inactive / (self.decay_months * 2), 0.3)
                decayed_rating = rating_obj.rating * (1 - decay_factor) + self.initial_rating * decay_factor
                return decayed_rating
                
        return rating_obj.rating
    
    def update_ratings(self, winner: str, loser: str, fight_date: Optional[datetime] = None) -> Tuple[float, float]:
        """
        Update ELO ratings after a fight
        
        Returns:
            Tuple of (winner_new_rating, loser_new_rating)
        """
        fight_date = fight_date or datetime.now()
        
        # Get current ratings
        winner_rating = self.get_rating(winner, fight_date)
        loser_rating = self.get_rating(loser, fight_date)
        
        # Initialize rating objects if needed
        if winner not in self.ratings:
            self.ratings[winner] = ELORating(winner, winner_rating)
        if loser not in self.ratings:
            self.ratings[loser] = ELORating(loser, loser_rating)
            
        winner_obj = self.ratings[winner]
        loser_obj = self.ratings[loser]
        
        # Calculate expected scores
        winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
        loser_expected = 1 - winner_expected
        
        # Get K-factors
        winner_k = self.get_k_factor(winner_rating, loser_rating, winner_obj.fights_count)
        loser_k = self.get_k_factor(loser_rating, winner_rating, loser_obj.fights_count)
        
        # Update ratings
        winner_new = winner_rating + winner_k * (1 - winner_expected)
        loser_new = loser_rating + loser_k * (0 - loser_expected)
        
        # Update rating objects
        winner_obj.rating = winner_new
        winner_obj.last_updated = fight_date
        winner_obj.fights_count += 1
        
        loser_obj.rating = loser_new
        loser_obj.last_updated = fight_date
        loser_obj.fights_count += 1
        
        logger.debug(f"ELO Update: {winner} {winner_rating:.0f} -> {winner_new:.0f}, "
                    f"{loser} {loser_rating:.0f} -> {loser_new:.0f}")
        
        return winner_new, loser_new
    
    def build_historical_ratings(self, fight_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build historical ELO ratings from fight data
        
        Args:
            fight_data: DataFrame with columns ['Date', 'Winner', 'Loser']
            
        Returns:
            DataFrame with ELO ratings at time of each fight
        """
        # Sort by date to process fights chronologically
        if 'Date' in fight_data.columns:
            fight_data = fight_data.sort_values('Date')
        
        elo_history = []
        
        for idx, fight in fight_data.iterrows():
            fight_date = pd.to_datetime(fight.get('Date', datetime.now()))
            winner = fight['Winner']
            loser = fight['Loser']
            
            # Get ratings before the fight
            winner_rating_before = self.get_rating(winner, fight_date)
            loser_rating_before = self.get_rating(loser, fight_date)
            
            # Update ratings
            winner_rating_after, loser_rating_after = self.update_ratings(winner, loser, fight_date)
            
            # Store historical record
            elo_history.append({
                'fight_id': idx,
                'date': fight_date,
                'winner': winner,
                'loser': loser,
                'winner_elo_before': winner_rating_before,
                'loser_elo_before': loser_rating_before,
                'winner_elo_after': winner_rating_after,
                'loser_elo_after': loser_rating_after,
                'elo_diff_before': winner_rating_before - loser_rating_before
            })
        
        return pd.DataFrame(elo_history)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for UFC prediction
    
    Creates interaction features, time windows, style matchup indicators,
    and integrates ELO ratings for enhanced model performance.
    """
    
    def __init__(self, elo_system: Optional[ELOSystem] = None):
        self.elo_system = elo_system or ELOSystem()
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial interaction features for key variables
        
        Focuses on meaningful interactions like:
        - Height advantage × Reach advantage
        - Age difference × Experience difference
        - Striking accuracy × Striking volume
        """
        # Select key features for interactions
        interaction_features = [
            'height_inches_diff', 'reach_in_diff', 'age_diff',
            'slpm_diff', 'str_acc_diff', 'sapm_diff', 'str_def_diff',
            'td_avg_diff', 'td_acc_diff', 'td_def_diff', 'wins_diff'
        ]
        
        # Filter to available columns
        available_features = [col for col in interaction_features if col in df.columns]
        
        if len(available_features) < 2:
            logger.warning("Insufficient features for interaction creation")
            return df
        
        # Create interaction features
        feature_subset = df[available_features]
        poly_features = self.poly_features.fit_transform(feature_subset)
        
        # Get feature names
        feature_names = self.poly_features.get_feature_names_out(available_features)
        
        # Create DataFrame with interaction features
        interactions_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Remove original features (they're already in df)
        original_features = set(available_features)
        interaction_only = interactions_df.drop(columns=[col for col in interactions_df.columns 
                                                       if col in original_features], errors='ignore')
        
        # Combine with original DataFrame
        enhanced_df = pd.concat([df, interaction_only], axis=1)
        
        logger.info(f"Created {len(interaction_only.columns)} interaction features")
        return enhanced_df
    
    def create_time_window_features(self, df: pd.DataFrame, fighter_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create time-based features like recent form, momentum indicators
        
        Features:
        - Last 3/5 fight win percentage
        - Recent performance trends
        - Time since last fight
        - Win/loss streaks
        """
        if fighter_history is None:
            logger.warning("No fighter history provided for time window features")
            return df
        
        # This would require historical fight data structured by fighter
        # For now, create placeholder momentum features based on existing data
        enhanced_df = df.copy()
        
        # Create momentum indicators from win/loss ratios
        if 'wins_diff' in df.columns and 'losses_diff' in df.columns:
            # Win percentage differential
            enhanced_df['win_pct_diff'] = (
                df['wins_diff'] / (df['wins_diff'].abs() + df['losses_diff'].abs() + 1)
            )
            
            # Recent form indicator (based on win/loss ratio)
            enhanced_df['form_indicator'] = np.tanh(df['wins_diff'] * 0.1)  # Sigmoid-like scaling
        
        return enhanced_df
    
    def add_elo_features(self, df: pd.DataFrame, fighter_names: Tuple[str, str] = None) -> pd.DataFrame:
        """
        Add ELO rating features to the dataset
        
        Args:
            df: Feature DataFrame
            fighter_names: Tuple of (fighter_a, fighter_b) names
        """
        enhanced_df = df.copy()
        
        if fighter_names and self.elo_system:
            fighter_a, fighter_b = fighter_names
            
            # Get current ELO ratings
            elo_a = self.elo_system.get_rating(fighter_a)
            elo_b = self.elo_system.get_rating(fighter_b)
            
            # Add ELO features
            enhanced_df['elo_a'] = elo_a
            enhanced_df['elo_b'] = elo_b
            enhanced_df['elo_diff'] = elo_a - elo_b
            enhanced_df['elo_favorite'] = np.where(elo_a > elo_b, elo_a, elo_b)
            enhanced_df['elo_underdog'] = np.where(elo_a < elo_b, elo_a, elo_b)
            
            logger.info(f"Added ELO features: {fighter_a}={elo_a:.0f}, {fighter_b}={elo_b:.0f}")
        
        return enhanced_df
    
    def create_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for fighting style matchups
        
        Based on fighter statistics, identify striker vs wrestler vs all-rounder
        and create style compatibility features.
        """
        enhanced_df = df.copy()
        
        # Define style indicators based on stats
        if all(col in df.columns for col in ['td_avg_diff', 'slpm_diff', 'sub_avg_diff']):
            # Striker indicator (high striking volume, low takedowns)
            enhanced_df['striker_advantage'] = df['slpm_diff'] - df['td_avg_diff'] * 0.5
            
            # Wrestler indicator (high takedowns, good takedown defense)
            enhanced_df['wrestler_advantage'] = df['td_avg_diff'] + df.get('td_def_diff', 0)
            
            # Submission specialist indicator
            enhanced_df['submission_advantage'] = df['sub_avg_diff'] + df.get('td_acc_diff', 0) * 0.3
            
            # Style mismatch indicator (striker vs wrestler is interesting)
            enhanced_df['style_mismatch'] = np.abs(enhanced_df['striker_advantage'] - 
                                                  enhanced_df['wrestler_advantage'])
        
        return enhanced_df


class EnhancedUFCPredictor:
    """
    Enhanced UFC prediction system with multiple advanced ML models
    
    Combines Random Forest baseline with XGBoost, LightGBM, and ensemble methods.
    Includes advanced feature engineering, ELO ratings, and proper temporal validation.
    """
    
    def __init__(self, use_elo: bool = True, use_advanced_features: bool = True):
        self.use_elo = use_elo
        self.use_advanced_features = use_advanced_features
        
        # Initialize components
        self.elo_system = ELOSystem() if use_elo else None
        self.feature_engineer = AdvancedFeatureEngineer(self.elo_system) if use_advanced_features else None
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.performance: Dict[str, ModelPerformance] = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 40,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': config.model.random_state,
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': config.model.random_state,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': config.model.random_state,
                    'verbose': -1
                }
            }
        }
    
    def prepare_features(self, X: pd.DataFrame, fighter_names: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Apply advanced feature engineering to input data
        """
        if not self.use_advanced_features:
            return X
        
        X_enhanced = X.copy()
        
        # Apply feature engineering
        X_enhanced = self.feature_engineer.create_interaction_features(X_enhanced)
        X_enhanced = self.feature_engineer.create_time_window_features(X_enhanced)
        X_enhanced = self.feature_engineer.create_style_features(X_enhanced)
        
        # Add ELO features if fighter names provided
        if fighter_names and len(fighter_names) == len(X_enhanced):
            for i, names in enumerate(fighter_names):
                if i < len(X_enhanced):
                    row_enhanced = self.feature_engineer.add_elo_features(
                        X_enhanced.iloc[i:i+1], names
                    )
                    X_enhanced.iloc[i] = row_enhanced.iloc[0]
        
        return X_enhanced
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> ModelPerformance:
        """
        Train a single model and return performance metrics
        """
        start_time = datetime.now()
        
        config = self.model_configs[model_name]
        model = config['model'](**config['params'])
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
        
        # Train on full dataset
        model.fit(X_train, y_train)
        
        # Calculate training performance
        y_pred = model.predict(X_train)
        y_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_train, y_pred)
        roc_auc = roc_auc_score(y_train, y_pred_proba) if y_pred_proba is not None else None
        class_report = classification_report(y_train, y_pred)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model
        self.models[model_name] = model
        
        performance = ModelPerformance(
            accuracy=accuracy,
            roc_auc=roc_auc or 0.0,
            classification_report=class_report,
            cross_val_scores=cv_scores,
            feature_importance=feature_importance,
            training_time=training_time
        )
        
        self.performance[model_name] = performance
        
        logger.info(f"Trained {model_name}: Accuracy={accuracy:.4f}, CV_Accuracy={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return performance
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> ModelPerformance:
        """
        Train ensemble model combining individual models
        """
        # Ensure individual models are trained
        individual_models = []
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name not in self.models:
                self.train_single_model(model_name, X_train, y_train)
            individual_models.append((model_name, self.models[model_name]))
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=individual_models,
            voting='soft'  # Use probability averaging
        )
        
        start_time = datetime.now()
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring='accuracy')
        
        y_pred = ensemble.predict(X_train)
        y_pred_proba = ensemble.predict_proba(X_train)[:, 1]
        
        accuracy = accuracy_score(y_train, y_pred)
        roc_auc = roc_auc_score(y_train, y_pred_proba)
        class_report = classification_report(y_train, y_pred)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store ensemble
        self.models['ensemble'] = ensemble
        
        performance = ModelPerformance(
            accuracy=accuracy,
            roc_auc=roc_auc,
            classification_report=class_report,
            cross_val_scores=cv_scores,
            training_time=training_time
        )
        
        self.performance['ensemble'] = performance
        
        logger.info(f"Trained ensemble: Accuracy={accuracy:.4f}, CV_Accuracy={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return performance
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, ModelPerformance]:
        """
        Train all models and return performance comparison
        """
        logger.info(f"Training all models on dataset: {X_train.shape}")
        
        # Train individual models
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            try:
                self.train_single_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        # Train ensemble
        try:
            self.train_ensemble(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train ensemble: {e}")
        
        return self.performance
    
    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble', 
                fighter_names: Optional[List[Tuple[str, str]]] = None) -> np.ndarray:
        """
        Make predictions using specified model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Prepare features
        X_prepared = self.prepare_features(X, fighter_names)
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X_prepared)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = 'ensemble',
                     fighter_names: Optional[List[Tuple[str, str]]] = None) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Prepare features
        X_prepared = self.prepare_features(X, fighter_names)
        
        # Get probabilities
        model = self.models[model_name]
        probabilities = model.predict_proba(X_prepared)
        
        return probabilities
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary for all trained models
        """
        summary_data = []
        
        for model_name, perf in self.performance.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': perf.accuracy,
                'ROC_AUC': perf.roc_auc,
                'CV_Mean': perf.cross_val_scores.mean(),
                'CV_Std': perf.cross_val_scores.std(),
                'Training_Time': perf.training_time
            })
        
        return pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)
    
    def save_models(self, save_dir: Optional[Path] = None) -> Path:
        """
        Save all trained models to disk
        """
        save_dir = save_dir or (config.paths.model_dir / f"enhanced_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = save_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save performance metrics
        perf_path = save_dir / "performance_summary.json"
        with open(perf_path, 'w') as f:
            perf_dict = {name: {
                'accuracy': perf.accuracy,
                'roc_auc': perf.roc_auc,
                'cv_scores': perf.cross_val_scores.tolist(),
                'training_time': perf.training_time
            } for name, perf in self.performance.items()}
            
            import json
            json.dump(perf_dict, f, indent=2)
        
        logger.info(f"Models saved to {save_dir}")
        return save_dir
    
    def load_models(self, load_dir: Path) -> None:
        """
        Load models from disk
        """
        model_files = list(load_dir.glob("*_model.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} from {model_file}")


def demonstrate_enhanced_models():
    """
    Demonstration function showing the enhanced modeling capabilities
    """
    logger.info("🚀 Enhanced UFC Prediction Models Demo")
    logger.info("=" * 60)
    
    try:
        # This would typically load real data
        # For demo, create synthetic data matching the expected structure
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic feature data
        feature_names = [
            'height_inches_diff', 'weight_lbs_diff', 'reach_in_diff', 'age_diff',
            'slpm_diff', 'str_acc_diff', 'sapm_diff', 'str_def_diff',
            'td_avg_diff', 'td_acc_diff', 'td_def_diff', 'sub_avg_diff',
            'wins_diff', 'losses_diff'
        ]
        
        X_demo = pd.DataFrame(
            np.random.randn(n_samples, len(feature_names)) * 2,
            columns=feature_names
        )
        
        # Create realistic target (slight edge to positive diffs)
        y_demo = ((X_demo['height_inches_diff'] * 0.3 + 
                  X_demo['reach_in_diff'] * 0.2 + 
                  X_demo['wins_diff'] * 0.4 +
                  np.random.randn(n_samples) * 0.5) > 0).astype(int)
        
        # Split data temporally (no random shuffle)
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X_demo[:split_idx], X_demo[split_idx:]
        y_train, y_test = y_demo[:split_idx], y_demo[split_idx:]
        
        # Initialize enhanced predictor
        predictor = EnhancedUFCPredictor(use_elo=True, use_advanced_features=True)
        
        # Train all models
        print("\n📊 Training Models...")
        performance = predictor.train_all_models(X_train, y_train)
        
        # Show performance summary
        print("\n📈 Performance Summary:")
        summary_df = predictor.get_performance_summary()
        print(summary_df.round(4))
        
        # Test predictions
        print("\n🎯 Testing Predictions...")
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'ensemble']:
            if model_name in predictor.models:
                test_pred = predictor.predict(X_test, model_name)
                test_accuracy = accuracy_score(y_test, test_pred)
                print(f"{model_name:12}: Test Accuracy = {test_accuracy:.4f}")
        
        print("\n✅ Enhanced models demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    demonstrate_enhanced_models()