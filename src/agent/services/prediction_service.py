"""
UFC Prediction Service

Professional prediction engine extracted from notebook workflow.
Provides comprehensive fight analysis with market comparison and opportunity identification.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PredictionResult:
    """Structured prediction result with market analysis"""
    
    def __init__(self, fight_key: str, fighter_a: str, fighter_b: str):
        self.fight_key = fight_key
        self.fighter_a = fighter_a
        self.fighter_b = fighter_b
        
        # Model predictions
        self.model_prediction_a: float = 0.0
        self.model_prediction_b: float = 0.0
        self.model_favorite: str = ""
        self.model_favorite_prob: float = 0.0
        self.predicted_method: str = ""
        
        # Market data
        self.market_odds_a: float = 0.0
        self.market_odds_b: float = 0.0
        self.market_prob_a: float = 0.0
        self.market_prob_b: float = 0.0
        self.market_favorite: str = ""
        self.market_favorite_prob: float = 0.0
        
        # Analysis
        self.expected_value_a: float = 0.0
        self.expected_value_b: float = 0.0
        self.confidence_score: float = 0.0
        self.is_upset_opportunity: bool = False
        self.is_high_confidence: bool = False
        
        # Raw prediction data
        self.raw_prediction: Dict = {}


class PredictionAnalysis:
    """Complete prediction analysis results"""
    
    def __init__(self, event_name: str):
        self.event_name = event_name
        self.timestamp = datetime.now().isoformat()
        self.fight_predictions: List[PredictionResult] = []
        self.summary = {
            'total_fights': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'upset_opportunities': 0,
            'high_confidence_picks': 0,
            'method_breakdown': {}
        }


class UFCPredictionService:
    """
    Professional UFC prediction service with market analysis capabilities
    
    Extracted from notebook Cell 3 for production use in automated agent.
    """
    
    def __init__(self, betting_system: Dict[str, Any]):
        """
        Initialize prediction service with trained models and data
        
        Args:
            betting_system: Dictionary containing models, data, and prediction function
        """
        self.betting_system = betting_system
        self.validate_system()
    
    def validate_system(self):
        """Validate that all required components are available"""
        required_keys = [
            'fighters_df', 'winner_cols', 'method_cols', 
            'winner_model', 'method_model', 'predict_function'
        ]
        
        for key in required_keys:
            if key not in self.betting_system:
                raise ValueError(f"Missing required component: {key}")
    
    def predict_event(self, odds_data: Dict[str, Dict], event_name: str) -> PredictionAnalysis:
        """
        Generate predictions for all fights in an event with market analysis
        
        Args:
            odds_data: Dictionary of fight odds data
            event_name: Name of the UFC event
            
        Returns:
            PredictionAnalysis: Complete analysis with predictions and market comparison
        """
        logger.info(f"Starting prediction analysis for {event_name}")
        
        analysis = PredictionAnalysis(event_name)
        
        logger.info(f"Generating predictions for {len(odds_data)} fights")
        
        for fight_key, fight_odds in odds_data.items():
            fighter_a = fight_odds['fighter_a']
            fighter_b = fight_odds['fighter_b']
            
            logger.debug(f"Predicting fight: {fighter_a} vs {fighter_b}")
            
            try:
                prediction_result = self._predict_single_fight(
                    fight_key, fighter_a, fighter_b, fight_odds
                )
                
                if prediction_result:
                    analysis.fight_predictions.append(prediction_result)
                    analysis.summary['successful_predictions'] += 1
                    
                    # Update opportunity counters
                    if prediction_result.is_upset_opportunity:
                        analysis.summary['upset_opportunities'] += 1
                    if prediction_result.is_high_confidence:
                        analysis.summary['high_confidence_picks'] += 1
                    
                    # Track method predictions
                    method = prediction_result.predicted_method
                    if method not in analysis.summary['method_breakdown']:
                        analysis.summary['method_breakdown'][method] = 0
                    analysis.summary['method_breakdown'][method] += 1
                    
                    logger.info(
                        f"Prediction complete: {prediction_result.model_favorite} "
                        f"({prediction_result.model_favorite_prob:.1%}) vs market favorite "
                        f"{prediction_result.market_favorite}"
                    )
                else:
                    analysis.summary['failed_predictions'] += 1
                    logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}")
                    
            except Exception as e:
                logger.error(f"Error predicting {fighter_a} vs {fighter_b}: {str(e)}")
                analysis.summary['failed_predictions'] += 1
        
        # Update summary totals
        analysis.summary['total_fights'] = len(odds_data)
        
        logger.info(
            f"Prediction analysis complete: {analysis.summary['successful_predictions']}"
            f"/{analysis.summary['total_fights']} successful predictions"
        )
        
        return analysis
    
    def _predict_single_fight(self, fight_key: str, fighter_a: str, fighter_b: str, 
                            odds_data: Dict) -> Optional[PredictionResult]:
        """
        Generate prediction for a single fight with market analysis
        
        Args:
            fight_key: Unique identifier for the fight
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            odds_data: Market odds data for the fight
            
        Returns:
            PredictionResult: Complete prediction with market analysis, or None if failed
        """
        try:
            # Generate model prediction using existing function
            prediction = self.betting_system['predict_function'](
                fighter_a, fighter_b,
                self.betting_system['fighters_df'],
                self.betting_system['winner_cols'],
                self.betting_system['method_cols'],
                self.betting_system['winner_model'],
                self.betting_system['method_model']
            )
            
            if 'error' in prediction:
                logger.warning(f"Model prediction failed: {prediction['error']}")
                return None
            
            # Create result object
            result = PredictionResult(fight_key, fighter_a, fighter_b)
            
            # Extract model probabilities
            result.model_prediction_a = float(
                prediction['win_probabilities'][fighter_a].replace('%', '')
            ) / 100
            result.model_prediction_b = float(
                prediction['win_probabilities'][fighter_b].replace('%', '')
            ) / 100
            
            # Determine model favorite
            if result.model_prediction_a > result.model_prediction_b:
                result.model_favorite = fighter_a
                result.model_favorite_prob = result.model_prediction_a
            else:
                result.model_favorite = fighter_b
                result.model_favorite_prob = result.model_prediction_b
            
            # Method prediction
            result.predicted_method = prediction['predicted_method']
            
            # Market data
            result.market_odds_a = odds_data['fighter_a_decimal_odds']
            result.market_odds_b = odds_data['fighter_b_decimal_odds']
            result.market_prob_a = 1 / result.market_odds_a
            result.market_prob_b = 1 / result.market_odds_b
            
            # Determine market favorite
            if result.market_odds_a < result.market_odds_b:
                result.market_favorite = fighter_a
                result.market_favorite_prob = result.market_prob_a
            else:
                result.market_favorite = fighter_b
                result.market_favorite_prob = result.market_prob_b
            
            # Calculate expected values
            result.expected_value_a = (result.model_prediction_a * result.market_odds_a) - 1
            result.expected_value_b = (result.model_prediction_b * result.market_odds_b) - 1
            
            # Analyze opportunities
            result.is_upset_opportunity = self._is_upset_opportunity(result)
            result.is_high_confidence = self._is_high_confidence(result)
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(result)
            
            # Store raw prediction
            result.raw_prediction = prediction
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single fight prediction: {str(e)}")
            return None
    
    def _is_upset_opportunity(self, result: PredictionResult) -> bool:
        """
        Determine if this is an upset opportunity
        
        Model favors different fighter than market with high confidence and good EV
        """
        return (
            result.market_favorite != result.model_favorite and 
            result.model_favorite_prob > 0.55 and
            max(result.expected_value_a, result.expected_value_b) > 0.08
        )
    
    def _is_high_confidence(self, result: PredictionResult) -> bool:
        """
        Determine if this is a high confidence pick
        
        Model and market agree on favorite with high model confidence and decent EV
        """
        return (
            result.model_favorite_prob > 0.7 and
            result.market_favorite == result.model_favorite and
            max(result.expected_value_a, result.expected_value_b) > 0.05
        )
    
    def _calculate_confidence_score(self, result: PredictionResult) -> float:
        """
        Calculate overall confidence score for the prediction
        
        Combines probability difference with market-model alignment
        """
        prob_diff = abs(result.model_prediction_a - result.model_prediction_b)
        market_model_alignment = 1.0 if result.market_favorite == result.model_favorite else 0.5
        confidence_score = (prob_diff + market_model_alignment) / 2
        
        return confidence_score
    
    def format_analysis_summary(self, analysis: PredictionAnalysis) -> str:
        """
        Format prediction analysis summary for display
        
        Args:
            analysis: Complete prediction analysis
            
        Returns:
            str: Formatted summary text
        """
        summary = analysis.summary
        
        output = [
            f"📈 PREDICTION ANALYSIS SUMMARY",
            f"=" * 40,
            f"Total Fights: {summary['total_fights']}",
            f"Successful Predictions: {summary['successful_predictions']}",
            f"Failed Predictions: {summary['failed_predictions']}",
            f"Upset Opportunities: {summary['upset_opportunities']}",
            f"High Confidence Picks: {summary['high_confidence_picks']}"
        ]
        
        if summary['method_breakdown']:
            output.append(f"\n🥊 Method Predictions:")
            for method, count in summary['method_breakdown'].items():
                percentage = (count / summary['successful_predictions']) * 100
                output.append(f"   {method}: {count} fights ({percentage:.1f}%)")
        
        return "\n".join(output)
    
    def format_comparison_table(self, analysis: PredictionAnalysis) -> str:
        """
        Format detailed fight comparison table
        
        Args:
            analysis: Complete prediction analysis
            
        Returns:
            str: Formatted comparison table
        """
        output = [
            f"📊 DETAILED FIGHT ANALYSIS",
            f"=" * 80
        ]
        
        for fight in analysis.fight_predictions:
            output.append(f"\n🥊 {fight.fight_key}")
            output.append(f"{'Fighter':<20} {'Model':<8} {'Market':<8} {'EV':<8} {'Status'}")
            output.append("-" * 60)
            
            # Fighter A
            status_a = ""
            if fight.model_favorite == fight.fighter_a:
                status_a += "⭐"
            if fight.expected_value_a > 0.05:
                status_a += "💎"
            
            model_a_str = f"{fight.model_prediction_a:.1%}".ljust(8)
            market_a_str = f"{fight.market_prob_a:.1%}".ljust(8)
            ev_a_str = f"{fight.expected_value_a:+.1%}".ljust(8)
            
            output.append(f"{fight.fighter_a:<20} {model_a_str} {market_a_str} {ev_a_str} {status_a}")
            
            # Fighter B
            status_b = ""
            if fight.model_favorite == fight.fighter_b:
                status_b += "⭐"
            if fight.expected_value_b > 0.05:
                status_b += "💎"
            
            model_b_str = f"{fight.model_prediction_b:.1%}".ljust(8)
            market_b_str = f"{fight.market_prob_b:.1%}".ljust(8)
            ev_b_str = f"{fight.expected_value_b:+.1%}".ljust(8)
            
            output.append(f"{fight.fighter_b:<20} {model_b_str} {market_b_str} {ev_b_str} {status_b}")
            
            # Additional info
            if fight.is_upset_opportunity:
                output.append(
                    f"   🚨 UPSET OPPORTUNITY - Model favors {fight.model_favorite}, "
                    f"Market favors {fight.market_favorite}"
                )
            if fight.is_high_confidence:
                output.append(f"   ⭐ HIGH CONFIDENCE - Strong model agreement with market")
        
        return "\n".join(output)
    
    def get_best_opportunities(self, analysis: PredictionAnalysis, 
                              min_ev: float = 0.05) -> List[Tuple[str, str, float]]:
        """
        Extract best betting opportunities from analysis
        
        Args:
            analysis: Complete prediction analysis
            min_ev: Minimum expected value threshold
            
        Returns:
            List of (fighter_name, fight_key, expected_value) tuples
        """
        opportunities = []
        
        for fight in analysis.fight_predictions:
            # Check fighter A
            if fight.expected_value_a >= min_ev:
                opportunities.append((fight.fighter_a, fight.fight_key, fight.expected_value_a))
            
            # Check fighter B
            if fight.expected_value_b >= min_ev:
                opportunities.append((fight.fighter_b, fight.fight_key, fight.expected_value_b))
        
        # Sort by expected value descending
        opportunities.sort(key=lambda x: x[2], reverse=True)
        
        return opportunities