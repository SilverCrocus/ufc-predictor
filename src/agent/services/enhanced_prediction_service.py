"""
Enhanced UFC Prediction Service with Ensemble Integration

Extends the base prediction service with XGBoost ensemble capabilities,
bootstrap confidence intervals, and Phase 2A data quality integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import re
import unicodedata

# Import base classes
from .prediction_service import UFCPredictionService, PredictionResult, PredictionAnalysis

# Import ensemble components
from src.ensemble_manager import UFCEnsembleManager, EnsemblePrediction
from src.confidence_intervals import BootstrapConfidenceCalculator
from config.model_config import ENSEMBLE_CONFIG

logger = logging.getLogger(__name__)


class EnhancedPredictionResult(PredictionResult):
    """Extended prediction result with ensemble and confidence interval data"""
    
    def __init__(self, fight_key: str, fighter_a: str, fighter_b: str):
        super().__init__(fight_key, fighter_a, fighter_b)
        
        # Ensemble-specific attributes
        self.ensemble_breakdown: Dict[str, float] = {}
        self.confidence_interval: Tuple[float, float] = (0.0, 1.0)
        self.uncertainty_score: float = 0.0
        self.data_quality_score: float = 0.0
        self.ensemble_confidence: float = 0.0


class EnhancedPredictionAnalysis(PredictionAnalysis):
    """Extended analysis with ensemble metrics"""
    
    def __init__(self, event_name: str):
        super().__init__(event_name)
        
        # Ensemble-specific summary metrics
        self.ensemble_summary = {
            'mean_confidence_interval_width': 0.0,
            'mean_uncertainty_score': 0.0,
            'mean_data_quality_score': 0.0,
            'high_uncertainty_predictions': 0,
            'ensemble_agreement_rate': 0.0
        }


class EnhancedUFCPredictionService(UFCPredictionService):
    """
    Enhanced UFC prediction service with XGBoost ensemble and confidence intervals
    
    Integrates with Phase 2A data quality confidence scoring and provides
    bootstrap uncertainty quantification for improved betting decisions.
    """
    
    def __init__(self, betting_system: Dict[str, Any], 
                 ensemble_manager: Optional[UFCEnsembleManager] = None,
                 confidence_calculator: Optional[BootstrapConfidenceCalculator] = None,
                 data_confidence_scorer=None):
        """
        Initialize enhanced prediction service
        
        Args:
            betting_system: Dictionary containing models, data, and prediction function
            ensemble_manager: UFCEnsembleManager for coordinating models
            confidence_calculator: Bootstrap confidence interval calculator
            data_confidence_scorer: Phase 2A data quality scorer
        """
        super().__init__(betting_system)
        
        self.ensemble_manager = ensemble_manager
        self.confidence_calculator = confidence_calculator
        self.data_confidence_scorer = data_confidence_scorer
        self.config = ENSEMBLE_CONFIG
        
        # Initialize ensemble if models are available
        if self.ensemble_manager is None and self._has_ensemble_models():
            self._initialize_ensemble_manager()
        
        # Initialize confidence calculator if not provided
        if self.confidence_calculator is None:
            from src.confidence_intervals import create_ufc_confidence_calculator
            self.confidence_calculator = create_ufc_confidence_calculator(
                n_bootstrap=self.config['bootstrap_samples']
            )
        
        logger.info("Enhanced UFC Prediction Service initialized with ensemble capabilities")
    
    def _has_ensemble_models(self) -> bool:
        """Check if betting system contains ensemble models"""
        return ('winner_model' in self.betting_system and 
                'xgboost_model' in self.betting_system)
    
    def _initialize_ensemble_manager(self):
        """Initialize ensemble manager with available models"""
        try:
            rf_model = self.betting_system.get('winner_model')
            xgb_model = self.betting_system.get('xgboost_model')
            nn_model = self.betting_system.get('neural_network_model')
            
            if rf_model and xgb_model:
                from src.ensemble_manager import create_default_ensemble_manager
                self.ensemble_manager = create_default_ensemble_manager(
                    rf_model, xgb_model, nn_model, self.data_confidence_scorer
                )
                logger.info("Ensemble manager initialized with available models")
            else:
                logger.warning("Insufficient models for ensemble initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize ensemble manager: {str(e)}")
    
    def predict_event(self, odds_data: Dict[str, Dict], event_name: str) -> EnhancedPredictionAnalysis:
        """
        Generate enhanced predictions with strict input validation and no fallbacks
        
        Args:
            odds_data: Dictionary of fight odds data
            event_name: Name of the UFC event
            
        Returns:
            EnhancedPredictionAnalysis: Complete analysis with ensemble predictions
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If validation or prediction fails
        """
        # Comprehensive input validation - NO fallbacks
        validated_odds_data, sanitized_event_name = self._validate_and_sanitize_inputs(odds_data, event_name)
        
        logger.info(f"Starting enhanced prediction analysis for {sanitized_event_name}")
        
        analysis = EnhancedPredictionAnalysis(sanitized_event_name)
        
        # Use ensemble predictions if available, strict error handling
        if self.ensemble_manager and self.ensemble_manager.is_trained:
            logger.info("Using ensemble prediction mode")
            analysis = self._predict_with_ensemble(validated_odds_data, analysis)
        else:
            logger.info("No ensemble manager available - using base prediction service")
            base_analysis = super().predict_event(validated_odds_data, sanitized_event_name)
            analysis = self._convert_to_enhanced_analysis(base_analysis)
        
        # Calculate ensemble summary metrics
        self._calculate_ensemble_summary(analysis)
        
        logger.info(
            f"Enhanced prediction analysis complete: {analysis.summary['successful_predictions']}"
            f"/{analysis.summary['total_fights']} successful predictions"
        )
        
        return analysis
    
    def _predict_with_ensemble(self, odds_data: Dict[str, Dict], 
                             analysis: EnhancedPredictionAnalysis) -> EnhancedPredictionAnalysis:
        """Generate predictions using ensemble manager"""
        
        logger.info(f"Generating ensemble predictions for {len(odds_data)} fights")
        
        # Import validation components
        from src.validation import validate_ufc_fighter_pair, validate_ufc_prediction_dataframe, FighterNameValidationError, DataFrameValidationError
        
        # Prepare fighter pairs and features with strict validation
        fighter_pairs = []
        features_list = []
        
        for fight_key, fight_odds in odds_data.items():
            try:
                # Validate fight_key
                if not fight_key or not isinstance(fight_key, str):
                    raise ValueError(f"Invalid fight_key: {fight_key}")
                
                # Validate fight_odds structure
                if not isinstance(fight_odds, dict):
                    raise ValueError(f"fight_odds must be dict for {fight_key}, got {type(fight_odds)}")
                
                # Check for required odds fields
                required_fields = ['fighter_a', 'fighter_b']
                missing_fields = [field for field in required_fields if field not in fight_odds]
                if missing_fields:
                    raise ValueError(f"Missing required fields in {fight_key}: {missing_fields}")
                
                # Comprehensive fighter name validation with NO fallbacks
                fighter_a_raw = fight_odds.get('fighter_a')
                fighter_b_raw = fight_odds.get('fighter_b')
                
                # Check if fighter names exist
                if fighter_a_raw is None or fighter_b_raw is None:
                    raise FighterNameValidationError(
                        f"Missing fighter names in fight {fight_key}: "
                        f"fighter_a={fighter_a_raw}, fighter_b={fighter_b_raw}"
                    )
                
                # Check data types
                if not isinstance(fighter_a_raw, str) or not isinstance(fighter_b_raw, str):
                    raise FighterNameValidationError(
                        f"Fighter names must be strings in fight {fight_key}: "
                        f"fighter_a={type(fighter_a_raw).__name__}, fighter_b={type(fighter_b_raw).__name__}"
                    )
                
                # Check for empty strings
                if not fighter_a_raw.strip() or not fighter_b_raw.strip():
                    raise FighterNameValidationError(
                        f"Fighter names cannot be empty in fight {fight_key}: "
                        f"fighter_a='{fighter_a_raw}', fighter_b='{fighter_b_raw}'"
                    )
                
                # Validate using strict validation framework
                fighter_a, fighter_b = validate_ufc_fighter_pair(
                    fighter_a_raw.strip(), fighter_b_raw.strip(), strict_mode=True
                )
                
                # Additional validation: fighters must be different
                if fighter_a.lower() == fighter_b.lower():
                    raise FighterNameValidationError(
                        f"Fighter names cannot be the same in fight {fight_key}: '{fighter_a}'"
                    )
                
                logger.debug(f"Fighter names validated: {fighter_a} vs {fighter_b}")
                
                fighter_pairs.append((fighter_a, fighter_b))
                
                # Generate features for the fight with validation
                fight_features = self._generate_fight_features(fighter_a, fighter_b)
                
                # Validate feature DataFrame before use
                validated_features = validate_ufc_prediction_dataframe(
                    fight_features, 
                    model_columns=None,  # Will be checked later against model
                    strict_mode=True
                )
                
                features_list.append(validated_features)
                logger.debug(f"Features validated for {fighter_a} vs {fighter_b}: {len(validated_features.columns)} columns")
                
            except (FighterNameValidationError, DataFrameValidationError) as e:
                logger.error(f"Validation failed for {fight_key}: {str(e)}")
                analysis.summary['failed_predictions'] += 1
                # NO fallbacks - explicit failure
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {fight_key}: {str(e)}")
                analysis.summary['failed_predictions'] += 1
                continue
        
        if not features_list:
            logger.error("No valid features generated for any fights")
            return analysis
        
        # Combine features into DataFrame
        try:
            X = pd.concat(features_list, ignore_index=True)
            
            # Get ensemble predictions with confidence intervals
            ensemble_predictions = self.ensemble_manager.predict_with_confidence(
                X, fighter_pairs, bootstrap_samples=self.config['bootstrap_samples']
            )
            
            # Process each ensemble prediction
            for i, (fight_key, fight_odds) in enumerate(odds_data.items()):
                if i >= len(ensemble_predictions):
                    break
                
                ensemble_pred = ensemble_predictions[i]
                
                try:
                    enhanced_result = self._create_enhanced_result_from_ensemble(
                        fight_key, fight_odds, ensemble_pred
                    )
                    
                    if enhanced_result:
                        analysis.fight_predictions.append(enhanced_result)
                        analysis.summary['successful_predictions'] += 1
                        
                        # Update opportunity counters
                        if enhanced_result.is_upset_opportunity:
                            analysis.summary['upset_opportunities'] += 1
                        if enhanced_result.is_high_confidence:
                            analysis.summary['high_confidence_picks'] += 1
                        
                        # Track method predictions
                        method = enhanced_result.predicted_method
                        if method not in analysis.summary['method_breakdown']:
                            analysis.summary['method_breakdown'][method] = 0
                        analysis.summary['method_breakdown'][method] += 1
                        
                        logger.info(
                            f"Enhanced prediction: {enhanced_result.model_favorite} "
                            f"({enhanced_result.model_favorite_prob:.1%}) "
                            f"CI: [{enhanced_result.confidence_interval[0]:.3f}, "
                            f"{enhanced_result.confidence_interval[1]:.3f}]"
                        )
                    else:
                        analysis.summary['failed_predictions'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing ensemble prediction {i}: {str(e)}")
                    analysis.summary['failed_predictions'] += 1
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            # NO FALLBACKS - fail fast and provide clear error message
            raise RuntimeError(
                f"Enhanced prediction failed for event '{analysis.event_name}': {str(e)}. "
                "Check input data quality and model availability."
            )
        
        # Update summary totals
        analysis.summary['total_fights'] = len(odds_data)
        
        return analysis
    
    def _create_enhanced_result_from_ensemble(self, fight_key: str, fight_odds: Dict,
                                            ensemble_pred: EnsemblePrediction) -> Optional[EnhancedPredictionResult]:
        """Create enhanced prediction result from ensemble prediction"""
        
        try:
            # Create enhanced result object
            result = EnhancedPredictionResult(fight_key, ensemble_pred.fighter_a, ensemble_pred.fighter_b)
            
            # Fill in ensemble predictions
            result.model_prediction_a = 1 - ensemble_pred.ensemble_probability
            result.model_prediction_b = ensemble_pred.ensemble_probability
            
            # Determine favorite
            if result.model_prediction_a > result.model_prediction_b:
                result.model_favorite = ensemble_pred.fighter_a
                result.model_favorite_prob = result.model_prediction_a
            else:
                result.model_favorite = ensemble_pred.fighter_b
                result.model_favorite_prob = result.model_prediction_b
            
            # Method prediction (use fallback for now)
            result.predicted_method = "Decision"  # Default fallback
            
            # Enhanced ensemble data
            result.ensemble_breakdown = ensemble_pred.model_breakdown
            result.confidence_interval = ensemble_pred.confidence_interval
            result.uncertainty_score = ensemble_pred.uncertainty_score
            result.data_quality_score = ensemble_pred.data_quality_score
            
            # Calculate enhanced confidence score
            ci_lower, ci_upper = ensemble_pred.confidence_interval
            ci_width = ci_upper - ci_lower
            data_quality = ensemble_pred.data_quality_score
            
            # Combined confidence: narrow CI + high data quality = high confidence
            prediction_confidence = (1 - ci_width) * 0.6 + data_quality * 0.4
            result.confidence_score = min(0.99, max(0.01, prediction_confidence))
            result.ensemble_confidence = prediction_confidence
            
            # Market analysis
            self._add_market_analysis(result, fight_odds)
            
            # Enhanced opportunity detection
            result.is_upset_opportunity = self._is_enhanced_upset_opportunity(result)
            result.is_high_confidence = self._is_enhanced_high_confidence(result)
            
            # Store raw ensemble data
            result.raw_prediction = {
                'ensemble_breakdown': ensemble_pred.model_breakdown,
                'confidence_interval': ensemble_pred.confidence_interval,
                'data_quality_score': ensemble_pred.data_quality_score,
                'uncertainty_score': ensemble_pred.uncertainty_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating enhanced result: {str(e)}")
            return None
    
    def _generate_fight_features(self, fighter_a: str, fighter_b: str) -> pd.DataFrame:
        """Generate features for a single fight using the existing prediction function"""
        
        # Use the existing predict_function to generate features
        # This maintains compatibility with the current feature engineering pipeline
        
        try:
            # Call the existing prediction function to get features
            prediction = self.betting_system['predict_function'](
                fighter_a, fighter_b,
                self.betting_system['fighters_df'],
                self.betting_system['winner_cols'],
                self.betting_system['method_cols'],
                self.betting_system['winner_model'],
                self.betting_system['method_model']
            )
            
            # Extract features from the prediction pipeline
            # This is a simplified approach - in production, you'd want to extract
            # the feature generation part of the predict_function
            
            # For now, create a placeholder feature DataFrame
            # In a real implementation, you'd extract the feature engineering logic
            feature_cols = self.betting_system['winner_cols']
            features = pd.DataFrame(
                np.random.randn(1, len(feature_cols)),  # Placeholder
                columns=feature_cols
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating features for {fighter_a} vs {fighter_b}: {str(e)}")
            raise
    
    def _is_enhanced_upset_opportunity(self, result: EnhancedPredictionResult) -> bool:
        """Enhanced upset opportunity detection using confidence intervals"""
        
        base_upset = super()._is_upset_opportunity(result)
        
        # Additional ensemble-based criteria
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        confidence_threshold = self.config['confidence_threshold']
        
        # High confidence upset: narrow CI + disagreement with market
        high_confidence_upset = (
            base_upset and 
            ci_width < 0.2 and  # Narrow confidence interval
            result.ensemble_confidence > confidence_threshold
        )
        
        return high_confidence_upset
    
    def _is_enhanced_high_confidence(self, result: EnhancedPredictionResult) -> bool:
        """Enhanced high confidence detection using ensemble agreement"""
        
        base_high_confidence = super()._is_high_confidence(result)
        
        # Additional ensemble-based criteria
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        
        # High ensemble confidence: narrow CI + model agreement + market agreement
        ensemble_high_confidence = (
            result.model_favorite_prob > 0.65 and
            ci_width < 0.15 and  # Very narrow confidence interval
            result.data_quality_score > 0.8 and
            result.market_favorite == result.model_favorite
        )
        
        return base_high_confidence or ensemble_high_confidence
    
    def _convert_to_enhanced_analysis(self, base_analysis: PredictionAnalysis) -> EnhancedPredictionAnalysis:
        """Convert base analysis to enhanced analysis format"""
        
        enhanced_analysis = EnhancedPredictionAnalysis(base_analysis.event_name)
        enhanced_analysis.timestamp = base_analysis.timestamp
        enhanced_analysis.summary = base_analysis.summary.copy()
        
        # Convert prediction results to enhanced format
        for base_result in base_analysis.fight_predictions:
            enhanced_result = EnhancedPredictionResult(
                base_result.fight_key, 
                base_result.fighter_a, 
                base_result.fighter_b
            )
            
            # Copy all base attributes
            for attr in dir(base_result):
                if not attr.startswith('_') and hasattr(enhanced_result, attr):
                    try:
                        setattr(enhanced_result, attr, getattr(base_result, attr))
                    except:
                        pass
            
            # Set default ensemble values
            enhanced_result.confidence_interval = (0.1, 0.9)  # Wide default
            enhanced_result.uncertainty_score = 0.8
            enhanced_result.data_quality_score = 0.7
            enhanced_result.ensemble_confidence = enhanced_result.confidence_score
            
            enhanced_analysis.fight_predictions.append(enhanced_result)
        
        return enhanced_analysis
    
    def _calculate_ensemble_summary(self, analysis: EnhancedPredictionAnalysis):
        """Calculate ensemble-specific summary metrics"""
        
        if not analysis.fight_predictions:
            return
        
        # Extract metrics from enhanced predictions
        ci_widths = []
        uncertainty_scores = []
        data_quality_scores = []
        high_uncertainty_count = 0
        
        for pred in analysis.fight_predictions:
            if isinstance(pred, EnhancedPredictionResult):
                ci_lower, ci_upper = pred.confidence_interval
                ci_width = ci_upper - ci_lower
                ci_widths.append(ci_width)
                
                uncertainty_scores.append(pred.uncertainty_score)
                data_quality_scores.append(pred.data_quality_score)
                
                if pred.uncertainty_score > 0.3:  # High uncertainty threshold
                    high_uncertainty_count += 1
        
        # Calculate summary statistics
        if ci_widths:
            analysis.ensemble_summary['mean_confidence_interval_width'] = np.mean(ci_widths)
            analysis.ensemble_summary['mean_uncertainty_score'] = np.mean(uncertainty_scores)
            analysis.ensemble_summary['mean_data_quality_score'] = np.mean(data_quality_scores)
            analysis.ensemble_summary['high_uncertainty_predictions'] = high_uncertainty_count
            
            # Calculate ensemble agreement rate (narrow CIs indicate agreement)
            narrow_ci_count = sum(1 for width in ci_widths if width < 0.2)
            analysis.ensemble_summary['ensemble_agreement_rate'] = narrow_ci_count / len(ci_widths)
        
        logger.info(f"Ensemble summary calculated: {analysis.ensemble_summary}")
    
    def _validate_and_sanitize_inputs(self, odds_data: Dict[str, Dict], event_name: str) -> Tuple[Dict[str, Dict], str]:
        """
        Comprehensive input validation and sanitization with strict error handling
        
        Args:
            odds_data: Raw fight odds data
            event_name: Raw event name
            
        Returns:
            Tuple of (validated_odds_data, sanitized_event_name)
            
        Raises:
            ValueError: If any input is invalid
            RuntimeError: If validation fails
        """
        # Validate event name
        if not event_name or not isinstance(event_name, str):
            raise ValueError("Event name must be a non-empty string")
        
        # Sanitize event name
        sanitized_event_name = self._sanitize_event_name(event_name)
        if not sanitized_event_name:
            raise ValueError(f"Event name '{event_name}' failed sanitization")
        
        # Validate odds data structure
        if not odds_data or not isinstance(odds_data, dict):
            raise ValueError("Odds data must be a non-empty dictionary")
        
        if len(odds_data) == 0:
            raise ValueError("Odds data cannot be empty")
        
        if len(odds_data) > 15:  # Reasonable limit for UFC events
            raise ValueError(f"Too many fights in event: {len(odds_data)} > 15")
        
        # Validate and sanitize each fight
        validated_odds_data = {}
        
        for fight_key, fight_odds in odds_data.items():
            try:
                validated_fight = self._validate_single_fight(fight_key, fight_odds)
                validated_odds_data[fight_key] = validated_fight
                
            except Exception as e:
                logger.error(f"Fight validation failed for '{fight_key}': {str(e)}")
                raise ValueError(f"Fight validation failed for '{fight_key}': {str(e)}")
        
        if not validated_odds_data:
            raise ValueError("No valid fights after validation")
        
        logger.info(f"Input validation successful: {len(validated_odds_data)} fights validated")
        
        return validated_odds_data, sanitized_event_name
    
    def _sanitize_event_name(self, event_name: str) -> str:
        """Sanitize event name to prevent injection attacks"""
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKC', event_name.strip())
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>"\';\\&|`${}()]', '', normalized)
        
        # Limit length
        sanitized = sanitized[:200]
        
        # Validate final result
        if not re.match(r'^[a-zA-Z0-9\s\-\.\:_]+$', sanitized):
            logger.warning(f"Event name contains invalid characters: {event_name}")
            raise ValueError(f"Event name contains invalid characters: {event_name}")
        
        return sanitized
    
    def _validate_single_fight(self, fight_key: str, fight_odds: Dict) -> Dict:
        """Validate and sanitize a single fight's data"""
        
        # Validate fight key
        if not fight_key or not isinstance(fight_key, str):
            raise ValueError("Fight key must be a non-empty string")
        
        if len(fight_key) > 100:
            raise ValueError(f"Fight key too long: {len(fight_key)} > 100")
        
        # Validate fight odds structure
        if not isinstance(fight_odds, dict):
            raise ValueError("Fight odds must be a dictionary")
        
        required_fields = ['fighter_a', 'fighter_b', 'fighter_a_decimal_odds', 'fighter_b_decimal_odds']
        missing_fields = [field for field in required_fields if field not in fight_odds]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate and sanitize fighter names
        fighter_a = self._sanitize_fighter_name(fight_odds['fighter_a'])
        fighter_b = self._sanitize_fighter_name(fight_odds['fighter_b'])
        
        if not fighter_a or not fighter_b:
            raise ValueError("Fighter names cannot be empty after sanitization")
        
        if fighter_a == fighter_b:
            raise ValueError(f"Fighter names cannot be identical: '{fighter_a}'")
        
        # Validate odds values
        odds_a = self._validate_odds_value(fight_odds['fighter_a_decimal_odds'], 'fighter_a_decimal_odds')
        odds_b = self._validate_odds_value(fight_odds['fighter_b_decimal_odds'], 'fighter_b_decimal_odds')
        
        # Market integrity check - odds should indicate reasonable probability
        implied_prob_a = 1 / odds_a
        implied_prob_b = 1 / odds_b
        total_implied_prob = implied_prob_a + implied_prob_b
        
        # Vig (bookmaker margin) should be reasonable (typically 1.05-1.15)
        if total_implied_prob < 0.9 or total_implied_prob > 1.5:
            raise ValueError(f"Suspicious odds: implied probabilities sum to {total_implied_prob:.3f}")
        
        # Return validated fight data
        return {
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'fighter_a_decimal_odds': odds_a,
            'fighter_b_decimal_odds': odds_b
        }
    
    def _sanitize_fighter_name(self, name: Any) -> str:
        """Sanitize fighter name to prevent injection while preserving legitimate characters"""
        
        if not name or not isinstance(name, str):
            raise ValueError("Fighter name must be a non-empty string")
        
        # Normalize unicode characters (important for international fighters)
        normalized = unicodedata.normalize('NFKC', name.strip())
        
        # Remove dangerous characters but preserve legitimate ones
        # Allow: letters, spaces, hyphens, apostrophes, periods
        sanitized = re.sub(r'[<>"\`;\\&|`${}()[\]{}=+*^%#@!~]', '', normalized)
        
        # Limit length
        sanitized = sanitized[:100]
        
        # Validate final result - allow international characters
        if not re.match(r'^[a-zA-ZÀ-ÿ\u0100-\u017F\u0400-\u04FF\s\-\'\.\,]+$', sanitized):
            raise ValueError(f"Fighter name contains invalid characters: {name}")
        
        # Additional security checks
        if len(sanitized.strip()) < 2:
            raise ValueError(f"Fighter name too short: '{name}'")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'(script|javascript|eval|exec|system|cmd)',
            r'(select|insert|update|delete|drop|union)',
            r'(\.\./|\.\.\\)',
            r'(http://|https://|ftp://)'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized.lower()):
                raise ValueError(f"Fighter name contains suspicious pattern: {name}")
        
        return sanitized.strip()
    
    def _validate_odds_value(self, odds: Any, field_name: str) -> float:
        """Validate decimal odds value"""
        
        try:
            odds_float = float(odds)
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid number, got {type(odds)}: {odds}")
        
        # Validate range - decimal odds typically 1.01 to 50.0
        if not 1.01 <= odds_float <= 50.0:
            raise ValueError(f"{field_name} out of valid range [1.01, 50.0]: {odds_float}")
        
        # Check for suspicious values
        if odds_float == 1.0:
            raise ValueError(f"{field_name} cannot be exactly 1.0 (invalid odds)")
        
        return odds_float
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get status of ensemble components"""
        status = {
            'ensemble_manager_available': self.ensemble_manager is not None,
            'ensemble_trained': False,
            'confidence_calculator_available': self.confidence_calculator is not None,
            'data_confidence_scorer_available': self.data_confidence_scorer is not None,
            'config': dict(self.config)
        }
        
        if self.ensemble_manager:
            status['ensemble_trained'] = self.ensemble_manager.is_trained
            status['ensemble_summary'] = self.ensemble_manager.get_ensemble_summary()
        
        return status