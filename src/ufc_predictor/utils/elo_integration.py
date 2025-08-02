"""
ELO System Integration Module

This module integrates the UFC ELO rating system with the existing machine learning pipeline,
providing ELO-based features for the ML models and enabling hybrid predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
import logging

from .ufc_elo_system import UFCELOSystem, ELOConfig
from .multi_dimensional_elo import MultiDimensionalUFCELO
from .elo_historical_processor import UFCHistoricalProcessor
from configs.model_config import *


class ELOIntegration:
    """
    Integration class for combining ELO ratings with the existing ML pipeline
    """
    
    def __init__(self, 
                 config: ELOConfig = None,
                 use_multi_dimensional: bool = True,
                 elo_data_path: str = None):
        """
        Initialize ELO integration
        
        Args:
            config: ELO configuration
            use_multi_dimensional: Whether to use multi-dimensional ELO
            elo_data_path: Path to saved ELO data
        """
        self.config = config or ELOConfig()
        self.use_multi_dimensional = use_multi_dimensional
        
        # Initialize appropriate ELO system
        if use_multi_dimensional:
            self.elo_system = MultiDimensionalUFCELO(self.config)
        else:
            self.elo_system = UFCELOSystem(self.config)
        
        self.processor = UFCHistoricalProcessor(self.elo_system)
        self.logger = logging.getLogger(__name__)
        
        # Load existing ELO data if available
        if elo_data_path and Path(elo_data_path).exists():
            self.load_elo_data(elo_data_path)
        
        self.elo_features = [
            'elo_rating_diff',
            'elo_fighter1_rating',
            'elo_fighter2_rating',
            'elo_fighter1_uncertainty',
            'elo_fighter2_uncertainty',
            'elo_fighter1_fights_count',
            'elo_fighter2_fights_count',
            'elo_predicted_prob'
        ]
        
        if use_multi_dimensional:
            self.elo_features.extend([
                'elo_striking_diff',
                'elo_grappling_diff',
                'elo_cardio_diff',
                'elo_fighter1_ko_rate',
                'elo_fighter2_ko_rate',
                'elo_fighter1_sub_rate',
                'elo_fighter2_sub_rate',
                'elo_stylistic_advantage'
            ])
    
    def build_elo_from_data(self, 
                           fights_df: pd.DataFrame, 
                           fighters_df: pd.DataFrame = None,
                           save_path: str = None) -> None:
        """
        Build ELO system from historical UFC data
        
        Args:
            fights_df: Historical fights data
            fighters_df: Fighter information data
            save_path: Path to save ELO data
        """
        self.logger.info("Building ELO system from historical data...")
        
        # Process historical data
        self.elo_system = self.processor.build_elo_from_history(
            fights_df, fighters_df,
            start_date=datetime(2000, 1, 1)  # Start from UFC's modern era
        )
        
        # Save ELO data if path provided
        if save_path:
            self.save_elo_data(save_path)
        
        self.logger.info(f"ELO system built with {len(self.elo_system.fighters)} fighters")
    
    def load_elo_data(self, filepath: str) -> None:
        """Load ELO system from file"""
        try:
            self.elo_system.load_ratings(filepath)
            self.logger.info(f"ELO data loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading ELO data: {e}")
    
    def save_elo_data(self, filepath: str) -> None:
        """Save ELO system to file"""
        try:
            self.elo_system.save_ratings(filepath)
            self.logger.info(f"ELO data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving ELO data: {e}")
    
    def extract_elo_features(self, 
                            fighter1_name: str, 
                            fighter2_name: str) -> Dict[str, float]:
        """
        Extract ELO-based features for a fight matchup
        
        Args:
            fighter1_name: First fighter name
            fighter2_name: Second fighter name
            
        Returns:
            Dictionary of ELO features
        """
        features = {}
        
        # Get fighters from ELO system
        fighter1 = self.elo_system.fighters.get(fighter1_name)
        fighter2 = self.elo_system.fighters.get(fighter2_name)
        
        if fighter1 is None or fighter2 is None:
            # Return default values for unknown fighters
            return self._get_default_elo_features()
        
        # Basic ELO features
        features['elo_fighter1_rating'] = fighter1.current_rating
        features['elo_fighter2_rating'] = fighter2.current_rating
        features['elo_rating_diff'] = fighter1.current_rating - fighter2.current_rating
        features['elo_fighter1_uncertainty'] = fighter1.uncertainty
        features['elo_fighter2_uncertainty'] = fighter2.uncertainty
        features['elo_fighter1_fights_count'] = fighter1.fights_count
        features['elo_fighter2_fights_count'] = fighter2.fights_count
        
        # Predicted probability
        prediction = self.elo_system.predict_fight_outcome(
            fighter1_name, fighter2_name, include_uncertainty=False
        )
        features['elo_predicted_prob'] = prediction.get('fighter1_win_prob', 0.5)
        
        # Multi-dimensional features
        if self.use_multi_dimensional and hasattr(fighter1, 'striking_rating'):
            features['elo_striking_diff'] = fighter1.striking_rating - fighter2.striking_rating
            features['elo_grappling_diff'] = fighter1.grappling_rating - fighter2.grappling_rating
            features['elo_cardio_diff'] = fighter1.cardio_rating - fighter2.cardio_rating
            
            # Fighting style features
            total_fights1 = max(1, fighter1.fights_count)
            total_fights2 = max(1, fighter2.fights_count)
            
            features['elo_fighter1_ko_rate'] = (fighter1.ko_wins + fighter1.ko_losses) / total_fights1
            features['elo_fighter2_ko_rate'] = (fighter2.ko_wins + fighter2.ko_losses) / total_fights2
            features['elo_fighter1_sub_rate'] = (fighter1.submission_wins + fighter1.submission_losses) / total_fights1
            features['elo_fighter2_sub_rate'] = (fighter2.submission_wins + fighter2.submission_losses) / total_fights2
            
            # Stylistic advantage (simplified)
            striking_adv = abs(features['elo_striking_diff'])
            grappling_adv = abs(features['elo_grappling_diff'])
            features['elo_stylistic_advantage'] = max(striking_adv, grappling_adv)
        
        return features
    
    def _get_default_elo_features(self) -> Dict[str, float]:
        """Return default ELO features for unknown fighters"""
        features = {
            'elo_rating_diff': 0.0,
            'elo_fighter1_rating': self.config.initial_rating,
            'elo_fighter2_rating': self.config.initial_rating,
            'elo_fighter1_uncertainty': self.config.initial_uncertainty,
            'elo_fighter2_uncertainty': self.config.initial_uncertainty,
            'elo_fighter1_fights_count': 0,
            'elo_fighter2_fights_count': 0,
            'elo_predicted_prob': 0.5
        }
        
        if self.use_multi_dimensional:
            features.update({
                'elo_striking_diff': 0.0,
                'elo_grappling_diff': 0.0,
                'elo_cardio_diff': 0.0,
                'elo_fighter1_ko_rate': 0.2,  # Average KO rate
                'elo_fighter2_ko_rate': 0.2,
                'elo_fighter1_sub_rate': 0.1,  # Average submission rate
                'elo_fighter2_sub_rate': 0.1,
                'elo_stylistic_advantage': 0.0
            })
        
        return features
    
    def enhance_fight_dataset(self, 
                             fight_dataset: pd.DataFrame,
                             fighter1_col: str = 'Fighter',
                             fighter2_col: str = 'Opponent') -> pd.DataFrame:
        """
        Enhance existing fight dataset with ELO features
        
        Args:
            fight_dataset: Existing fight dataset
            fighter1_col: Column name for first fighter
            fighter2_col: Column name for second fighter
            
        Returns:
            Enhanced dataset with ELO features
        """
        self.logger.info("Enhancing fight dataset with ELO features...")
        
        enhanced_df = fight_dataset.copy()
        
        # Extract ELO features for each fight
        elo_feature_data = []
        
        for _, row in fight_dataset.iterrows():
            fighter1 = str(row[fighter1_col]).strip()
            fighter2 = str(row[fighter2_col]).strip()
            
            elo_features = self.extract_elo_features(fighter1, fighter2)
            elo_feature_data.append(elo_features)
        
        # Add ELO features to dataset
        elo_df = pd.DataFrame(elo_feature_data)
        enhanced_df = pd.concat([enhanced_df, elo_df], axis=1)
        
        self.logger.info(f"Added {len(self.elo_features)} ELO features to dataset")
        return enhanced_df
    
    def create_elo_predictor(self, 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series,
                            model_type: str = 'ensemble') -> object:
        """
        Create a hybrid predictor combining ELO and ML features
        
        Args:
            X_train: Training features (including ELO features)
            y_train: Training targets
            model_type: Type of model ('ensemble', 'elo_only', 'ml_only')
            
        Returns:
            Trained hybrid predictor
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        if model_type == 'elo_only':
            # Use only ELO features
            elo_cols = [col for col in X_train.columns if col in self.elo_features]
            X_train_elo = X_train[elo_cols]
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42))
            ])
            model.fit(X_train_elo, y_train)
            
        elif model_type == 'ml_only':
            # Use traditional ML features (exclude ELO)
            non_elo_cols = [col for col in X_train.columns if col not in self.elo_features]
            X_train_ml = X_train[non_elo_cols]
            
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_ml, y_train)
            
        else:  # ensemble
            # Combine ELO and ML predictions
            from sklearn.ensemble import VotingClassifier
            
            # ELO-based model
            elo_cols = [col for col in X_train.columns if col in self.elo_features]
            X_train_elo = X_train[elo_cols]
            
            elo_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            # Traditional ML model
            non_elo_cols = [col for col in X_train.columns if col not in self.elo_features]
            X_train_ml = X_train[non_elo_cols] if non_elo_cols else X_train
            
            ml_model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit individual models
            elo_model.fit(X_train_elo, y_train)
            ml_model.fit(X_train_ml, y_train)
            
            # Create ensemble
            model = VotingClassifier([
                ('elo', elo_model),
                ('ml', ml_model)
            ], voting='soft')
            
            model.fit(X_train, y_train)
        
        return model
    
    def predict_fight_hybrid(self, 
                            fighter1_name: str, 
                            fighter2_name: str,
                            ml_model: object = None,
                            traditional_features: pd.DataFrame = None) -> Dict:
        """
        Make hybrid prediction combining ELO and ML models
        
        Args:
            fighter1_name: First fighter name
            fighter2_name: Second fighter name
            ml_model: Trained ML model (optional)
            traditional_features: Traditional ML features (optional)
            
        Returns:
            Comprehensive prediction results
        """
        result = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'predictions': {}
        }
        
        # ELO-based prediction
        elo_prediction = self.elo_system.predict_fight_outcome(
            fighter1_name, fighter2_name, include_uncertainty=True
        )
        
        result['predictions']['elo'] = {
            'fighter1_win_prob': elo_prediction.get('fighter1_win_prob', 0.5),
            'confidence': elo_prediction.get('confidence_level', 'medium')
        }
        
        if self.use_multi_dimensional and 'method_predictions' in elo_prediction:
            result['predictions']['elo']['method_predictions'] = elo_prediction['method_predictions']
        
        # ML-based prediction (if model provided)
        if ml_model is not None and traditional_features is not None:
            try:
                # Extract ELO features and combine with traditional features
                elo_features = self.extract_elo_features(fighter1_name, fighter2_name)
                
                # Create feature vector
                feature_vector = pd.DataFrame([elo_features])
                if len(traditional_features) > 0:
                    # Combine with traditional features
                    combined_features = pd.concat([traditional_features, feature_vector], axis=1)
                else:
                    combined_features = feature_vector
                
                # Make ML prediction
                ml_prob = ml_model.predict_proba(combined_features)[0]
                
                result['predictions']['ml'] = {
                    'fighter1_win_prob': ml_prob[1],  # Assuming class 1 is win
                    'confidence': 'high' if max(ml_prob) > 0.7 else 'medium'
                }
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
                result['predictions']['ml'] = {'error': str(e)}
        
        # Ensemble prediction
        if 'elo' in result['predictions'] and 'ml' in result['predictions']:
            if 'error' not in result['predictions']['ml']:
                elo_prob = result['predictions']['elo']['fighter1_win_prob']
                ml_prob = result['predictions']['ml']['fighter1_win_prob']
                
                # Weighted ensemble (ELO gets slightly higher weight due to UFC-specific design)
                ensemble_prob = 0.6 * elo_prob + 0.4 * ml_prob
                
                result['predictions']['ensemble'] = {
                    'fighter1_win_prob': ensemble_prob,
                    'fighter2_win_prob': 1.0 - ensemble_prob,
                    'confidence': 'high' if abs(ensemble_prob - 0.5) > 0.2 else 'medium'
                }
        
        return result
    
    def get_elo_feature_importance(self, model: object, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance for ELO features in a trained model
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = X.columns
                for name, importance in zip(feature_names, model.feature_importances_):
                    if name in self.elo_features:
                        importances[name] = importance
            elif hasattr(model, 'coef_'):
                feature_names = X.columns
                coef_abs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                for name, coef in zip(feature_names, coef_abs):
                    if name in self.elo_features:
                        importances[name] = coef
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
        
        return importances
    
    def update_elo_with_new_fight(self, 
                                 fighter1_name: str,
                                 fighter2_name: str,
                                 winner_name: str,
                                 method: str,
                                 fight_date: datetime = None,
                                 **kwargs) -> Dict:
        """
        Update ELO ratings with a new fight result
        
        Args:
            fighter1_name: First fighter name
            fighter2_name: Second fighter name  
            winner_name: Winner name
            method: Victory method
            fight_date: Fight date (defaults to now)
            **kwargs: Additional fight parameters
            
        Returns:
            Fight processing results
        """
        if fight_date is None:
            fight_date = datetime.now()
        
        return self.elo_system.process_fight(
            fighter1_name=fighter1_name,
            fighter2_name=fighter2_name,
            winner_name=winner_name,
            method=method,
            fight_date=fight_date,
            **kwargs
        )


def main():
    """Example usage of ELO integration"""
    
    # Initialize integration
    elo_integration = ELOIntegration(use_multi_dimensional=True)
    
    # Example: Create sample data
    sample_fights = pd.DataFrame({
        'Fighter': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic'],
        'Opponent': ['Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'],
        'Outcome': ['win', 'win', 'loss'],
        'Method': ['U-DEC', 'KO', 'KO'],
        'Event': ['UFC 182', 'UFC 226', 'UFC 260'],
        'Date': ['2015-01-03', '2018-07-07', '2021-03-27']
    })
    
    sample_fighters = pd.DataFrame({
        'Name': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'],
        'Weight (lbs)': [205, 205, 240, 250]
    })
    
    # Build ELO system
    elo_integration.build_elo_from_data(sample_fights, sample_fighters)
    
    # Extract ELO features for a matchup
    features = elo_integration.extract_elo_features('Jon Jones', 'Stipe Miocic')
    print("ELO Features:", features)
    
    # Make hybrid prediction
    prediction = elo_integration.predict_fight_hybrid('Jon Jones', 'Stipe Miocic')
    print("Hybrid Prediction:", prediction)
    
    # Get ELO rankings
    rankings = elo_integration.elo_system.get_rankings(top_n=5)
    print("ELO Rankings:", rankings)


if __name__ == "__main__":
    main()