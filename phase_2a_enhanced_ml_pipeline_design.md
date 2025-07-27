# Phase 2A: Enhanced ML Pipeline Design Document

## Executive Summary

This document outlines the technical design for enhancing the existing UFC prediction system with advanced ensemble methods, confidence intervals, and data quality integration while maintaining backward compatibility with the current agent workflow.

## Current System Analysis

### Existing Architecture Assessment

**Core Components:**
- **Random Forest Model**: Primary winner prediction model with hyperparameter tuning
- **Method Prediction**: Multi-class classification for fight outcome methods
- **Symmetrical Prediction**: Averages predictions from both fighter perspectives to eliminate positional bias
- **Feature Engineering**: 64 differential features comparing fighter statistics
- **Agent Integration**: Production-ready prediction service with market analysis

**Current Strengths:**
1. **Robust Baseline**: Random Forest provides strong interpretable predictions
2. **Symmetrical Approach**: Eliminates blue/red corner bias effectively
3. **Production Ready**: Full agent integration with betting service
4. **Feature Rich**: Comprehensive 64-feature differential system
5. **Version Management**: Timestamped model artifacts and metadata

**Identified Enhancement Opportunities:**
1. **Single Model Limitation**: Only Random Forest for winner prediction
2. **No Confidence Quantification**: Predictions lack uncertainty estimates
3. **Limited Feature Interactions**: Basic differential features only
4. **Data Quality Blind**: No confidence weighting for different data sources
5. **No Contextual Features**: Missing venue, crowd, momentum indicators

### Current Prediction Workflow Analysis

**File**: `src/agent/services/prediction_service.py`
- Uses `predict_fight_symmetrical()` function from `src/prediction.py`
- Returns structured `PredictionResult` objects with market analysis
- Integrates with betting service for profitability calculations

**Integration Points:**
1. **Model Loading**: Auto-detects latest trained models
2. **Feature Creation**: Uses existing differential feature engineering
3. **Prediction Interface**: Standardized prediction function signature
4. **Result Formatting**: Structured output for agent consumption

## Enhanced ML Pipeline Architecture

### 1. Ensemble Model Integration

#### XGBoost Integration
```python
class XGBoostUFCModel:
    """XGBoost model optimized for UFC prediction"""
    
    def __init__(self, model_config):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        self.feature_importance = None
        
    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        """Fit with early stopping validation"""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.feature_importance = self.model.feature_importances_
        
    def predict_proba_with_uncertainty(self, X):
        """Predict with uncertainty quantification"""
        base_proba = self.model.predict_proba(X)
        
        # Calculate prediction uncertainty based on leaf node statistics
        uncertainty = self._calculate_prediction_uncertainty(X)
        
        return base_proba, uncertainty
```

#### Neural Network Integration
```python
class NeuralNetworkUFCModel:
    """Deep learning model for UFC prediction with uncertainty"""
    
    def __init__(self, input_dim, model_config):
        self.model = self._build_model(input_dim)
        self.dropout_model = self._build_dropout_model(input_dim)
        self.uncertainty_samples = 100
        
    def _build_model(self, input_dim):
        """Build main prediction model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def predict_with_monte_carlo_dropout(self, X):
        """Monte Carlo dropout for uncertainty estimation"""
        predictions = []
        
        for _ in range(self.uncertainty_samples):
            pred = self.dropout_model(X, training=True)
            predictions.append(pred.numpy())
            
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
```

### 2. Ensemble Fusion Strategy

#### Weighted Ensemble with Dynamic Weighting
```python
class EnhancedUFCEnsemble:
    """Enhanced ensemble combining RF, XGBoost, and Neural Network"""
    
    def __init__(self):
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'neural_network': None
        }
        self.weights = {
            'random_forest': 0.4,    # Proven baseline
            'xgboost': 0.35,         # Strong gradient boosting
            'neural_network': 0.25   # Deep learning insights
        }
        self.performance_history = defaultdict(list)
        
    def predict_with_confidence(self, X, fighter_pairs=None):
        """Generate ensemble predictions with confidence intervals"""
        individual_predictions = {}
        individual_uncertainties = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name == 'random_forest':
                pred_proba = model.predict_proba(X)
                uncertainty = self._bootstrap_uncertainty_rf(model, X)
            elif model_name == 'xgboost':
                pred_proba, uncertainty = model.predict_proba_with_uncertainty(X)
            else:  # neural_network
                pred_proba, uncertainty = model.predict_with_monte_carlo_dropout(X)
                
            individual_predictions[model_name] = pred_proba
            individual_uncertainties[model_name] = uncertainty
        
        # Weighted ensemble prediction
        ensemble_pred = self._weighted_average(individual_predictions)
        
        # Combined uncertainty using error propagation
        ensemble_uncertainty = self._combine_uncertainties(
            individual_uncertainties, self.weights
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            ensemble_pred, ensemble_uncertainty
        )
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': individual_predictions,
            'confidence_intervals': confidence_intervals,
            'uncertainty_score': ensemble_uncertainty,
            'model_weights': self.weights.copy()
        }
```

### 3. Bootstrap Confidence Intervals

#### Bootstrap Sampling Strategy
```python
class BootstrapConfidenceCalculator:
    """Calculate confidence intervals using bootstrap resampling"""
    
    def __init__(self, n_bootstrap_samples=1000, confidence_level=0.95):
        self.n_samples = n_bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_bootstrap_ci(self, model, X, y_true=None):
        """Calculate bootstrap confidence intervals for predictions"""
        bootstrap_predictions = []
        
        for i in range(self.n_samples):
            # Bootstrap sample indices
            n_samples = len(X)
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            
            X_bootstrap = X.iloc[bootstrap_indices]
            
            # Get prediction for bootstrap sample
            pred_proba = model.predict_proba(X_bootstrap)[:, 1]
            bootstrap_predictions.append(pred_proba)
        
        # Calculate percentile-based confidence intervals
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        return ci_lower, ci_upper
        
    def prediction_reliability_score(self, prediction, ci_lower, ci_upper):
        """Calculate reliability score based on confidence interval width"""
        ci_width = ci_upper - ci_lower
        
        # Normalize to 0-1 scale (narrower CI = higher reliability)
        max_width = 1.0  # Maximum possible CI width
        reliability = 1 - (ci_width / max_width)
        
        return np.clip(reliability, 0, 1)
```

### 4. Data Quality Integration

#### Data Source Confidence Weighting
```python
class DataQualityManager:
    """Manage data quality and source confidence weighting"""
    
    def __init__(self):
        self.source_confidence = {
            'api_official': 1.0,      # Official UFC API data
            'api_sherdog': 0.9,       # Sherdog API
            'scrape_ufcstats': 0.85,  # UFC Stats website
            'scrape_sherdog': 0.8,    # Sherdog scraping
            'manual_entry': 0.7       # Manual data entry
        }
        
        self.data_freshness_weights = {
            'days_0_7': 1.0,      # Data within 1 week
            'days_8_30': 0.95,    # Data within 1 month
            'days_31_90': 0.9,    # Data within 3 months
            'days_90_plus': 0.8   # Data older than 3 months
        }
        
    def calculate_data_confidence(self, fighter_data):
        """Calculate overall confidence score for fighter data"""
        source_score = self.source_confidence.get(
            fighter_data.get('data_source', 'scrape_ufcstats'), 0.8
        )
        
        # Calculate freshness score
        last_update = fighter_data.get('last_updated')
        if last_update:
            days_old = (datetime.now() - last_update).days
            if days_old <= 7:
                freshness_score = self.data_freshness_weights['days_0_7']
            elif days_old <= 30:
                freshness_score = self.data_freshness_weights['days_8_30']
            elif days_old <= 90:
                freshness_score = self.data_freshness_weights['days_31_90']
            else:
                freshness_score = self.data_freshness_weights['days_90_plus']
        else:
            freshness_score = 0.7  # Default for unknown freshness
            
        # Calculate completeness score
        expected_fields = ['wins', 'losses', 'height', 'reach', 'slpm', 'str_acc']
        available_fields = sum(1 for field in expected_fields 
                             if field in fighter_data and fighter_data[field] is not None)
        completeness_score = available_fields / len(expected_fields)
        
        # Combined confidence score
        overall_confidence = (
            source_score * 0.5 + 
            freshness_score * 0.3 + 
            completeness_score * 0.2
        )
        
        return overall_confidence
        
    def weight_predictions_by_confidence(self, predictions, fighter_confidences):
        """Weight predictions based on data quality confidence"""
        fighter_a_conf = fighter_confidences[0]
        fighter_b_conf = fighter_confidences[1]
        
        # Average confidence for the matchup
        matchup_confidence = (fighter_a_conf + fighter_b_conf) / 2
        
        # Adjust prediction confidence based on data quality
        adjusted_predictions = {}
        for model_name, pred in predictions.items():
            confidence_multiplier = 0.7 + (0.3 * matchup_confidence)
            adjusted_predictions[model_name] = {
                'prediction': pred['prediction'],
                'confidence': pred['confidence'] * confidence_multiplier,
                'data_quality_score': matchup_confidence
            }
            
        return adjusted_predictions
```

### 5. Enhanced Feature Engineering

#### Contextual Features
```python
class ContextualFeatureEngineer:
    """Create contextual features for enhanced prediction accuracy"""
    
    def __init__(self):
        self.venue_effects = {
            'las_vegas': {'crowd_factor': 0.1, 'altitude': 0.0},
            'denver': {'crowd_factor': 0.05, 'altitude': 0.15},
            'international': {'crowd_factor': 0.2, 'altitude': 0.0}
        }
        
    def create_contextual_features(self, base_features, fight_context):
        """Add contextual features to base feature set"""
        enhanced_features = base_features.copy()
        
        # Venue-specific features
        venue = fight_context.get('venue', 'las_vegas')
        venue_data = self.venue_effects.get(venue, self.venue_effects['las_vegas'])
        
        enhanced_features['venue_crowd_factor'] = venue_data['crowd_factor']
        enhanced_features['venue_altitude_factor'] = venue_data['altitude']
        
        # Title fight pressure
        enhanced_features['title_fight_pressure'] = float(
            fight_context.get('is_title_fight', False)
        )
        
        # Main event pressure
        enhanced_features['main_event_pressure'] = float(
            fight_context.get('is_main_event', False)
        )
        
        # International fighter advantage
        fighter_a_country = fight_context.get('fighter_a_country', 'USA')
        venue_country = fight_context.get('venue_country', 'USA')
        enhanced_features['home_country_advantage'] = float(
            fighter_a_country == venue_country
        )
        
        return enhanced_features
        
    def create_momentum_features(self, fighter_history):
        """Create momentum and recent form features"""
        momentum_features = {}
        
        # Recent win streak momentum
        recent_results = fighter_history[-5:]  # Last 5 fights
        win_momentum = sum(1 for result in recent_results if result == 'win') / len(recent_results)
        momentum_features['recent_win_momentum'] = win_momentum
        
        # Performance trend
        if len(fighter_history) >= 3:
            recent_performance = sum(recent_results[-3:]) / 3
            earlier_performance = sum(fighter_history[-6:-3]) / 3
            momentum_features['performance_trend'] = recent_performance - earlier_performance
        else:
            momentum_features['performance_trend'] = 0.0
            
        return momentum_features
```

#### Style Matchup Analysis
```python
class StyleMatchupAnalyzer:
    """Analyze fighting style compatibility and create matchup features"""
    
    def __init__(self):
        self.style_profiles = {
            'striker': {
                'slpm_weight': 1.0, 'str_acc_weight': 0.8, 
                'td_avg_weight': -0.3, 'sub_avg_weight': -0.2
            },
            'wrestler': {
                'td_avg_weight': 1.0, 'td_acc_weight': 0.8,
                'slpm_weight': -0.2, 'str_def_weight': 0.6
            },
            'grappler': {
                'sub_avg_weight': 1.0, 'td_avg_weight': 0.6,
                'td_def_weight': 0.7, 'slpm_weight': -0.1
            }
        }
        
    def classify_fighting_style(self, fighter_stats):
        """Classify fighter's primary fighting style"""
        style_scores = {}
        
        for style, weights in self.style_profiles.items():
            score = 0
            for stat, weight in weights.items():
                if stat in fighter_stats:
                    score += fighter_stats[stat] * weight
            style_scores[style] = score
            
        return max(style_scores.items(), key=lambda x: x[1])
        
    def create_matchup_features(self, fighter_a_stats, fighter_b_stats):
        """Create style-based matchup features"""
        # Classify both fighters
        style_a, score_a = self.classify_fighting_style(fighter_a_stats)
        style_b, score_b = self.classify_fighting_style(fighter_b_stats)
        
        matchup_features = {}
        
        # Style advantage matrix
        style_advantages = {
            ('striker', 'wrestler'): 0.1,
            ('wrestler', 'striker'): -0.1,
            ('grappler', 'striker'): 0.05,
            ('striker', 'grappler'): -0.05,
            ('wrestler', 'grappler'): 0.05,
            ('grappler', 'wrestler'): -0.05
        }
        
        matchup_key = (style_a, style_b)
        matchup_features['style_advantage'] = style_advantages.get(matchup_key, 0.0)
        
        # Style mismatch intensity
        if style_a != style_b:
            matchup_features['style_mismatch_intensity'] = abs(score_a - score_b)
        else:
            matchup_features['style_mismatch_intensity'] = 0.0
            
        return matchup_features
```

### 6. Backward Compatible Implementation Strategy

#### Enhanced Prediction Service
```python
class EnhancedPredictionService(UFCPredictionService):
    """Enhanced prediction service maintaining backward compatibility"""
    
    def __init__(self, betting_system, ensemble_config=None):
        super().__init__(betting_system)
        
        # Initialize enhanced components if available
        self.ensemble_system = None
        self.data_quality_manager = DataQualityManager()
        self.contextual_engineer = ContextualFeatureEngineer()
        
        if ensemble_config:
            self._initialize_ensemble_system(ensemble_config)
    
    def _initialize_ensemble_system(self, config):
        """Initialize ensemble system if models available"""
        try:
            from src.advanced_ensemble_methods import AdvancedEnsembleSystem
            
            models = {}
            
            # Load Random Forest (always available)
            models['random_forest'] = self.betting_system['winner_model']
            
            # Load XGBoost if available
            xgb_path = config.get('xgboost_model_path')
            if xgb_path and Path(xgb_path).exists():
                models['xgboost'] = joblib.load(xgb_path)
                
            # Load Neural Network if available
            nn_path = config.get('neural_network_path')
            if nn_path and Path(nn_path).exists():
                models['neural_network'] = tf.keras.models.load_model(nn_path)
                
            if len(models) > 1:
                self.ensemble_system = AdvancedEnsembleSystem()
                self.ensemble_system.add_models(models)
                logger.info(f"Initialized ensemble with {len(models)} models")
            else:
                logger.info("Using single Random Forest model (ensemble unavailable)")
                
        except ImportError:
            logger.warning("Advanced ensemble methods not available, using baseline")
    
    def predict_with_enhanced_features(self, fighter_a, fighter_b, fight_context=None):
        """Enhanced prediction with new features and confidence intervals"""
        
        # Use enhanced ensemble if available, otherwise fallback to original
        if self.ensemble_system:
            return self._predict_with_ensemble(fighter_a, fighter_b, fight_context)
        else:
            return self._predict_with_baseline(fighter_a, fighter_b)
    
    def _predict_with_ensemble(self, fighter_a, fighter_b, fight_context):
        """Enhanced prediction using ensemble methods"""
        
        # Get base features (existing system)
        base_prediction = self.betting_system['predict_function'](
            fighter_a, fighter_b,
            self.betting_system['fighters_df'],
            self.betting_system['winner_cols'],
            self.betting_system['method_cols'],
            self.betting_system['winner_model'],
            self.betting_system['method_model']
        )
        
        if 'error' in base_prediction:
            return base_prediction
            
        # Create enhanced features
        if fight_context:
            base_features = self._extract_features_from_prediction(base_prediction)
            enhanced_features = self.contextual_engineer.create_contextual_features(
                base_features, fight_context
            )
            
            # Get ensemble prediction with confidence
            ensemble_result = self.ensemble_system.predict_with_confidence(
                pd.DataFrame([enhanced_features]),
                fighter_pairs=[(fighter_a, fighter_b)]
            )[0]
            
            # Enhance base prediction with ensemble insights
            enhanced_prediction = base_prediction.copy()
            enhanced_prediction.update({
                'ensemble_probability': ensemble_result.prediction_probability,
                'confidence_interval': {
                    'lower': ensemble_result.prediction_probability - 0.1,  # Simplified
                    'upper': ensemble_result.prediction_probability + 0.1
                },
                'model_agreement': ensemble_result.confidence,
                'prediction_method': 'ensemble_enhanced'
            })
            
            return enhanced_prediction
        else:
            return base_prediction
    
    def _predict_with_baseline(self, fighter_a, fighter_b):
        """Fallback to original prediction method"""
        return self.betting_system['predict_function'](
            fighter_a, fighter_b,
            self.betting_system['fighters_df'],
            self.betting_system['winner_cols'],
            self.betting_system['method_cols'],
            self.betting_system['winner_model'],
            self.betting_system['method_model']
        )
```

### 7. Performance Optimization

#### Real-time Prediction Requirements
```python
class OptimizedPredictionCache:
    """Cache system for real-time prediction performance"""
    
    def __init__(self, cache_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.cache_size = cache_size
        self.ttl = ttl_seconds
        
    def get_cached_prediction(self, fighter_a, fighter_b, context_hash=None):
        """Get cached prediction if available and valid"""
        cache_key = self._create_cache_key(fighter_a, fighter_b, context_hash)
        
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            
            # Check if cache entry is still valid
            if (datetime.now() - timestamp).seconds < self.ttl:
                return cached_result
            else:
                del self.cache[cache_key]
                
        return None
        
    def cache_prediction(self, fighter_a, fighter_b, prediction, context_hash=None):
        """Cache prediction result"""
        cache_key = self._create_cache_key(fighter_a, fighter_b, context_hash)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            
        self.cache[cache_key] = (prediction, datetime.now())
```

## Implementation Roadmap

### Phase 1: Foundation Enhancement (Week 1-2)
1. **Model Training Pipeline Extension**
   - Add XGBoost model training to existing pipeline
   - Implement neural network training with TensorFlow/Keras
   - Create ensemble model storage and versioning

2. **Feature Engineering Enhancement**
   - Extend existing feature engineering with interaction features
   - Add temporal momentum features
   - Implement style matchup analysis

### Phase 2: Ensemble Integration (Week 3-4)
1. **Ensemble System Implementation**
   - Implement weighted voting ensemble
   - Add bootstrap confidence interval calculation
   - Create ensemble model fusion logic

2. **Data Quality Integration**
   - Implement data source confidence scoring
   - Add data freshness weighting
   - Create quality-adjusted prediction weighting

### Phase 3: Agent Integration (Week 5-6)
1. **Backward Compatible API**
   - Extend existing prediction service
   - Maintain original prediction function interface
   - Add optional enhanced features

2. **Performance Optimization**
   - Implement prediction caching
   - Optimize model loading and inference
   - Add monitoring and logging

### Phase 4: Testing and Validation (Week 7-8)
1. **Model Validation**
   - Cross-validation with historical data
   - A/B testing framework
   - Performance comparison analysis

2. **Production Deployment**
   - Gradual rollout strategy
   - Monitoring and alerting
   - Fallback mechanisms

## Technical Considerations

### Model Versioning Strategy
- Maintain existing timestamp-based versioning
- Add ensemble configuration metadata
- Implement model performance tracking

### Feature Compatibility
- Ensure new features are optional
- Maintain existing 64-feature baseline
- Gradual feature rollout strategy

### Error Handling
- Graceful degradation to baseline models
- Comprehensive logging and monitoring
- Clear error messages and fallbacks

### Performance Requirements
- Target: <500ms prediction latency
- Memory usage: <2GB for full ensemble
- CPU optimization for real-time inference

## Expected Improvements

### Prediction Accuracy
- **Target**: 5-10% improvement in win prediction accuracy
- **Confidence Intervals**: 95% CI for all predictions
- **Uncertainty Quantification**: Reliable confidence scores

### Data Quality Impact
- **Source Confidence**: Weighted predictions based on data quality
- **Freshness Scoring**: Recent data weighted higher
- **Missing Data Handling**: Graceful degradation strategies

### Feature Enhancement Impact
- **Contextual Awareness**: Venue, crowd, pressure factors
- **Style Matchups**: Fighting style compatibility analysis
- **Temporal Features**: Momentum and recent form indicators

## Risk Mitigation

### Technical Risks
1. **Model Complexity**: Start with simple ensemble, gradually add complexity
2. **Performance Impact**: Implement caching and optimization early
3. **Data Dependencies**: Ensure robust fallbacks for missing data

### Business Risks
1. **Accuracy Regression**: Maintain baseline model as fallback
2. **System Reliability**: Comprehensive testing and monitoring
3. **Agent Integration**: Gradual rollout with feature flags

## Conclusion

This enhanced ML pipeline design provides a clear path to significantly improve the UFC prediction system while maintaining backward compatibility and production reliability. The modular approach allows for gradual implementation and testing, ensuring robust enhancement of the existing successful system.

The ensemble approach with confidence intervals, data quality integration, and enhanced features will provide more accurate and reliable predictions for the automated betting agent while maintaining all existing functionality.