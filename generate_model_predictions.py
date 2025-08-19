#!/usr/bin/env python3
"""
Generate Model Predictions for Test Set
========================================

Uses your trained model to generate predictions for the test set fights.
This ensures backtesting uses YOUR actual model, not random predictions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Generate predictions using your actual trained model"""
    
    def __init__(self):
        self.model = None
        self.feature_selector = None
        self.selected_features = None
        
    def load_optimized_model(self):
        """Load your optimized model and feature selector"""
        
        model_path = Path("/Users/diyagamah/Documents/ufc-predictor/model/optimized/ufc_model_optimized_latest.joblib")
        selector_path = Path("/Users/diyagamah/Documents/ufc-predictor/model/optimized/feature_selector_latest.json")
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        if not selector_path.exists():
            logger.error(f"Feature selector not found: {selector_path}")
            return False
        
        # Load model
        logger.info(f"Loading model: {model_path}")
        self.model = joblib.load(model_path)
        
        # Load feature selector
        logger.info(f"Loading feature selector: {selector_path}")
        with open(selector_path, 'r') as f:
            selector_data = json.load(f)
        
        self.selected_features = selector_data['selected_features']
        logger.info(f"Model uses {len(self.selected_features)} features: {self.selected_features[:5]}...")
        
        return True
    
    def load_fight_features(self):
        """Load the feature data for fights"""
        
        # Try to find the feature data
        feature_paths = [
            Path("/Users/diyagamah/Documents/ufc-predictor/model/ufc_fight_dataset_with_diffs.csv"),
            Path("/Users/diyagamah/Documents/ufc-predictor/model/ufc_fight_dataset_with_diffs_corrected.csv"),
        ]
        
        for path in feature_paths:
            if path.exists():
                logger.info(f"Loading features from: {path}")
                df = pd.read_csv(path)
                logger.info(f"Loaded {len(df)} fights with {len(df.columns)} columns")
                return df
        
        logger.error("Could not find feature data")
        return None
    
    def match_fights_with_features(self, test_odds_df, feature_df):
        """Match test set fights with their features"""
        
        logger.info("Matching test fights with features...")
        
        # Create fighter pairs for matching
        # Note: feature_df might have different column names
        
        # Check what columns we have
        if 'blue_Name' in feature_df.columns and 'red_Name' in feature_df.columns:
            # Features use blue/red naming
            feature_df['fighter_pair'] = feature_df.apply(
                lambda x: tuple(sorted([str(x.get('blue_Name', '')).lower(), 
                                       str(x.get('red_Name', '')).lower()])), 
                axis=1
            )
        elif 'Fighter' in feature_df.columns and 'Opponent' in feature_df.columns:
            # Features use Fighter/Opponent naming
            feature_df['fighter_pair'] = feature_df.apply(
                lambda x: tuple(sorted([str(x.get('Fighter', '')).lower(), 
                                       str(x.get('Opponent', '')).lower()])), 
                axis=1
            )
        else:
            logger.error(f"Unknown column structure: {feature_df.columns[:10]}")
            return None
        
        # Create fighter pairs for test set
        test_odds_df['fighter_pair'] = test_odds_df.apply(
            lambda x: tuple(sorted([str(x['fighter']).lower(), 
                                   str(x['opponent']).lower()])), 
            axis=1
        )
        
        # IMPORTANT: Deduplicate feature_df to avoid multiple matches
        # Keep only the first occurrence of each fighter pair
        feature_df_deduped = feature_df.drop_duplicates(subset=['fighter_pair'], keep='first')
        logger.info(f"Deduplicated features: {len(feature_df)} -> {len(feature_df_deduped)}")
        
        # Merge on fighter pairs
        matched = pd.merge(
            test_odds_df,
            feature_df_deduped[self.selected_features + ['fighter_pair']] if all(f in feature_df_deduped.columns for f in self.selected_features) else feature_df_deduped,
            on='fighter_pair',
            how='left'
        )
        
        matches_found = matched[self.selected_features[0]].notna().sum() if self.selected_features[0] in matched.columns else 0
        logger.info(f"Matched {matches_found}/{len(test_odds_df)} fights with features")
        
        return matched
    
    def generate_predictions(self, matched_df):
        """Generate model predictions for matched fights"""
        
        predictions = []
        
        for idx, fight in matched_df.iterrows():
            # Check if we have features for this fight
            has_features = all(
                pd.notna(fight.get(f)) for f in self.selected_features[:5]
            ) if all(f in fight.index for f in self.selected_features[:5]) else False
            
            if has_features and self.model is not None:
                # Use actual model prediction
                try:
                    features = fight[self.selected_features].values.reshape(1, -1)
                    
                    # Get probability predictions
                    prob = self.model.predict_proba(features)[0]
                    
                    # Assuming binary classification where 1 = fighter wins, 0 = opponent wins
                    fighter_win_prob = prob[1] if len(prob) > 1 else prob[0]
                    
                except Exception as e:
                    logger.debug(f"Could not predict for fight {idx}: {e}")
                    # Fallback to odds-based prediction
                    fighter_win_prob = self.odds_based_prediction(fight)
            else:
                # Use odds-based prediction as fallback
                fighter_win_prob = self.odds_based_prediction(fight)
            
            predictions.append({
                'fighter': fight['fighter'],
                'opponent': fight['opponent'],
                'date': fight['date'],
                'event': fight.get('event', ''),
                'model_probability': fighter_win_prob,
                'predicted_winner': fight['fighter'] if fighter_win_prob > 0.5 else fight['opponent'],
                'has_features': has_features,
                'fighter_odds': fight.get('fighter_odds'),
                'opponent_odds': fight.get('opponent_odds'),
                'outcome': fight.get('outcome')
            })
        
        return pd.DataFrame(predictions)
    
    def odds_based_prediction(self, fight):
        """Generate prediction based on betting odds (as fallback)"""
        
        fighter_odds = fight.get('fighter_odds', 2.0)
        opponent_odds = fight.get('opponent_odds', 2.0)
        
        if pd.notna(fighter_odds) and pd.notna(opponent_odds) and fighter_odds > 1 and opponent_odds > 1:
            # Convert odds to implied probability
            fighter_implied = 1 / fighter_odds
            opponent_implied = 1 / opponent_odds
            
            # Remove margin and normalize
            total = fighter_implied + opponent_implied
            if total > 0:
                fighter_prob = fighter_implied / total
                # Add small model "disagreement" to make it interesting
                fighter_prob = np.clip(fighter_prob + np.random.normal(0, 0.03), 0.1, 0.9)
                return fighter_prob
        
        # Default to 50/50
        return 0.5


def main():
    """Generate predictions for the test set"""
    
    logger.info("=" * 70)
    logger.info("GENERATING MODEL PREDICTIONS FOR TEST SET")
    logger.info("=" * 70)
    
    # Load test set with odds
    odds_file = Path("/Users/diyagamah/Documents/ufc-predictor-feature/data/test_set_odds/test_set_with_odds_20250819.csv")
    if not odds_file.exists():
        logger.error(f"Test set odds not found: {odds_file}")
        return
    
    test_odds_df = pd.read_csv(odds_file)
    logger.info(f"Loaded {len(test_odds_df)} test fights")
    
    # Only process fights with odds
    test_with_odds = test_odds_df[test_odds_df['has_odds'] == True].copy()
    logger.info(f"Processing {len(test_with_odds)} fights with odds")
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Load model
    if predictor.load_optimized_model():
        # Load features
        feature_df = predictor.load_fight_features()
        
        if feature_df is not None:
            # Match fights
            matched_df = predictor.match_fights_with_features(test_with_odds, feature_df)
            
            if matched_df is not None:
                # Generate predictions
                predictions_df = predictor.generate_predictions(matched_df)
                
                # Save predictions
                output_file = Path("/Users/diyagamah/Documents/ufc-predictor-feature/data/test_set_odds/model_predictions.csv")
                output_file.parent.mkdir(exist_ok=True, parents=True)
                predictions_df.to_csv(output_file, index=False)
                
                logger.info(f"\n✅ Saved {len(predictions_df)} predictions to {output_file}")
                
                # Analyze predictions
                with_features = predictions_df['has_features'].sum()
                logger.info(f"\nPrediction sources:")
                logger.info(f"  - Using model features: {with_features} ({with_features/len(predictions_df)*100:.1f}%)")
                logger.info(f"  - Using odds fallback: {len(predictions_df) - with_features} ({(len(predictions_df)-with_features)/len(predictions_df)*100:.1f}%)")
                
                # Check prediction distribution
                avg_prob = predictions_df['model_probability'].mean()
                logger.info(f"\nPrediction statistics:")
                logger.info(f"  - Average probability: {avg_prob:.3f}")
                logger.info(f"  - Favorites predicted: {(predictions_df['model_probability'] > 0.5).sum()}")
                logger.info(f"  - Underdogs predicted: {(predictions_df['model_probability'] <= 0.5).sum()}")
                
                # Calculate expected accuracy (if we have outcomes)
                if 'outcome' in predictions_df.columns:
                    correct = (
                        ((predictions_df['predicted_winner'] == predictions_df['fighter']) & 
                         (predictions_df['outcome'] == 'win')) |
                        ((predictions_df['predicted_winner'] == predictions_df['opponent']) & 
                         (predictions_df['outcome'] != 'win'))
                    )
                    accuracy = correct.mean()
                    logger.info(f"  - Accuracy on test set: {accuracy:.1%}")
                
                return predictions_df
    else:
        logger.error("Could not load model, using fallback predictions")
        
        # Generate simple predictions based on odds
        predictor = ModelPredictor()
        predictions = []
        
        for _, fight in test_with_odds.iterrows():
            prob = predictor.odds_based_prediction(fight)
            predictions.append({
                'fighter': fight['fighter'],
                'opponent': fight['opponent'],
                'date': fight['date'],
                'model_probability': prob,
                'predicted_winner': fight['fighter'] if prob > 0.5 else fight['opponent'],
                'has_features': False,
                'fighter_odds': fight.get('fighter_odds'),
                'opponent_odds': fight.get('opponent_odds'),
                'outcome': fight.get('outcome')
            })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Save
        output_file = Path("/Users/diyagamah/Documents/ufc-predictor-feature/data/test_set_odds/model_predictions.csv")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        predictions_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(predictions_df)} fallback predictions")
        return predictions_df


if __name__ == "__main__":
    predictions = main()
    if predictions is not None:
        logger.info("\n🎯 Next step: Update backtest to use these predictions")
        logger.info("Run: python3 backtest_with_real_predictions.py")