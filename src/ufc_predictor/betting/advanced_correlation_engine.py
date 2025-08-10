"""
Advanced Correlation Estimation Engine
Integrates multiple data sources for sophisticated parlay correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CorrelationSource:
    """Container for correlation estimates from different sources."""
    same_event: float
    feature_similarity: float
    residual_historical: float
    heuristic: float
    confidence: float


@dataclass
class HistoricalOutcome:
    """Container for historical fight outcome data."""
    event: str
    fighter: str
    opponent: str
    result: bool  # True for win, False for loss
    features: Dict[str, float]
    timestamp: datetime


class AdvancedCorrelationEngine:
    """
    Advanced correlation estimation engine that integrates multiple data sources
    including feature engineering, historical analysis, and betting market signals.
    """
    
    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        feature_weights: Optional[Dict[str, float]] = None,
        correlation_weights: Optional[List[float]] = None,
        min_historical_samples: int = 50
    ):
        """
        Initialize advanced correlation engine.
        
        Args:
            historical_data_path: Path to historical outcomes data
            feature_weights: Weights for different feature categories
            correlation_weights: Weights for blending correlation sources [same_event, feature, historical, heuristic]
            min_historical_samples: Minimum samples needed for historical analysis
        """
        self.feature_weights = feature_weights or self._default_feature_weights()
        self.correlation_weights = correlation_weights or [0.4, 0.3, 0.2, 0.1]
        self.min_historical_samples = min_historical_samples
        
        # Historical data storage
        self.historical_outcomes: List[HistoricalOutcome] = []
        self.residual_correlations: Dict[str, float] = {}
        self.feature_correlations: Dict[Tuple[str, str], float] = {}
        
        # Feature processing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        self.feature_names = []
        
        # Load historical data if provided
        if historical_data_path:
            self.load_historical_data(historical_data_path)
    
    def _default_feature_weights(self) -> Dict[str, float]:
        """Default weights for feature categories."""
        return {
            # Physical attributes
            'height': 0.08,
            'reach': 0.08,
            'age': 0.06,
            'weight': 0.05,
            
            # Fight metrics
            'striking_accuracy': 0.12,
            'takedown_accuracy': 0.10,
            'submission_attempts_avg': 0.08,
            'knockdown_avg': 0.10,
            
            # Performance metrics  
            'sig_strikes_per_min': 0.10,
            'takedowns_per_round': 0.08,
            'control_time_avg': 0.06,
            
            # Recent form
            'win_streak': 0.09,
            'recent_performance': 0.08,
            'activity_level': 0.05,
            
            # Categorical features
            'stance': 0.04,
            'camp': 0.06,
            'experience': 0.07
        }
    
    def estimate_comprehensive_correlation(
        self,
        fighter1: str,
        fighter2: str,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, CorrelationSource]:
        """
        Estimate correlation using all available methods and data sources.
        
        Args:
            fighter1: Name of first fighter
            fighter2: Name of second fighter
            fight1_features: Feature dict for first fight
            fight2_features: Feature dict for second fight
            context: Context information (event, timing, etc.)
            
        Returns:
            (final_correlation, source_breakdown)
        """
        correlation_sources = CorrelationSource(
            same_event=0.0,
            feature_similarity=0.0,
            residual_historical=0.0,
            heuristic=0.0,
            confidence=0.0
        )
        
        # 1. Same-event correlation
        if context.get('event') and fight1_features.get('event') == fight2_features.get('event'):
            correlation_sources.same_event = self._calculate_same_event_correlation(
                fight1_features, fight2_features, context
            )
        
        # 2. Feature similarity correlation
        correlation_sources.feature_similarity = self._calculate_feature_similarity_correlation(
            fight1_features, fight2_features
        )
        
        # 3. Historical residual correlation
        correlation_sources.residual_historical = self._calculate_historical_residual_correlation(
            fighter1, fighter2, fight1_features, fight2_features
        )
        
        # 4. Heuristic correlation
        correlation_sources.heuristic = self._calculate_heuristic_correlation(
            fight1_features, fight2_features, context
        )
        
        # Calculate confidence based on data availability
        correlation_sources.confidence = self._calculate_correlation_confidence(
            correlation_sources, fight1_features, fight2_features
        )
        
        # Blend correlations using weighted average
        correlations = [
            correlation_sources.same_event,
            correlation_sources.feature_similarity, 
            correlation_sources.residual_historical,
            correlation_sources.heuristic
        ]
        
        # Filter out zero correlations and adjust weights
        valid_indices = [i for i, corr in enumerate(correlations) if corr > 0]
        
        if valid_indices:
            valid_correlations = [correlations[i] for i in valid_indices]
            valid_weights = [self.correlation_weights[i] for i in valid_indices]
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            
            final_correlation = np.average(valid_correlations, weights=valid_weights)
        else:
            final_correlation = 0.05  # Default low correlation
        
        # Cap correlation based on confidence
        final_correlation *= correlation_sources.confidence
        final_correlation = np.clip(final_correlation, 0, 0.5)  # Cap at 50%
        
        return final_correlation, correlation_sources
    
    def _calculate_same_event_correlation(
        self,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate correlation for fights on the same event."""
        base_correlation = 0.15  # Base same-event correlation
        
        # Adjust for card position proximity
        pos1 = fight1_features.get('card_position', 0)
        pos2 = fight2_features.get('card_position', 0)
        
        if pos1 and pos2:
            position_diff = abs(pos1 - pos2)
            if position_diff <= 1:
                base_correlation += 0.05  # Adjacent fights
            elif position_diff >= 5:
                base_correlation -= 0.03  # Far apart
        
        # Adjust for event type
        event_type = context.get('event_type', 'regular')
        if event_type == 'pay_per_view':
            base_correlation += 0.02  # Higher stakes events
        elif event_type == 'fight_night':
            base_correlation -= 0.01
        
        # Venue and timing factors
        if context.get('international_venue'):
            base_correlation += 0.02  # International events can be correlated
        
        return max(0, base_correlation)
    
    def _calculate_feature_similarity_correlation(
        self,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any]
    ) -> float:
        """Calculate correlation based on fighter feature similarity."""
        try:
            # Extract numerical features
            features1 = self._extract_numerical_features(fight1_features)
            features2 = self._extract_numerical_features(fight2_features)
            
            if not features1 or not features2:
                return 0.05  # Default if no features
            
            # Ensure same features in both fights
            common_features = set(features1.keys()) & set(features2.keys())
            
            if len(common_features) < 5:  # Need minimum features
                return 0.05
            
            # Create feature vectors
            vec1 = np.array([features1[f] for f in sorted(common_features)])
            vec2 = np.array([features2[f] for f in sorted(common_features)])
            
            # Handle missing values
            mask = ~(np.isnan(vec1) | np.isnan(vec2))
            if mask.sum() < 3:
                return 0.05
            
            vec1_clean = vec1[mask]
            vec2_clean = vec2[mask]
            
            # Normalize vectors
            if np.std(vec1_clean) > 0 and np.std(vec2_clean) > 0:
                vec1_norm = (vec1_clean - np.mean(vec1_clean)) / np.std(vec1_clean)
                vec2_norm = (vec2_clean - np.mean(vec2_clean)) / np.std(vec2_clean)
                
                # Calculate cosine similarity
                cosine_sim = 1 - cosine(vec1_norm, vec2_norm)
                
                # Convert similarity to correlation estimate
                # High similarity suggests potential correlation
                correlation = max(0, cosine_sim * 0.20)  # Scale to max 20%
                
                return correlation
            
        except Exception as e:
            logger.debug(f"Error calculating feature correlation: {e}")
        
        return 0.05  # Default correlation
    
    def _extract_numerical_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Extract and weight numerical features from fight data."""
        numerical_features = {}
        
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Apply feature weight if available
                weight = self.feature_weights.get(feature_name, 1.0)
                numerical_features[feature_name] = float(value) * weight
        
        return numerical_features
    
    def _calculate_historical_residual_correlation(
        self,
        fighter1: str,
        fighter2: str,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any]
    ) -> float:
        """Calculate correlation based on historical prediction residuals."""
        
        if len(self.historical_outcomes) < self.min_historical_samples:
            return 0.0  # Insufficient data
        
        # Look for similar fighter archetypes
        archetype1 = self._classify_fighter_archetype(fight1_features)
        archetype2 = self._classify_fighter_archetype(fight2_features)
        
        correlation_key = tuple(sorted([archetype1, archetype2]))
        
        if correlation_key in self.residual_correlations:
            return self.residual_correlations[correlation_key]
        
        # Calculate empirical correlation from historical data
        empirical_correlation = self._calculate_empirical_correlation(
            archetype1, archetype2
        )
        
        # Cache result
        self.residual_correlations[correlation_key] = empirical_correlation
        
        return empirical_correlation
    
    def _classify_fighter_archetype(self, features: Dict[str, Any]) -> str:
        """Classify fighter into archetype based on features."""
        # Simplified archetype classification
        striking_accuracy = features.get('striking_accuracy', 0.5)
        takedown_accuracy = features.get('takedown_accuracy', 0.3)
        submission_attempts = features.get('submission_attempts_avg', 0.5)
        
        if takedown_accuracy > 0.6 or submission_attempts > 1.0:
            return 'grappler'
        elif striking_accuracy > 0.55:
            return 'striker'
        else:
            return 'balanced'
    
    def _calculate_empirical_correlation(self, archetype1: str, archetype2: str) -> float:
        """Calculate empirical correlation from historical outcomes."""
        
        # Group outcomes by archetype and event
        archetype_outcomes = {}
        
        for outcome in self.historical_outcomes:
            arch = self._classify_fighter_archetype(outcome.features)
            event_key = outcome.event
            
            if event_key not in archetype_outcomes:
                archetype_outcomes[event_key] = {}
            
            if arch not in archetype_outcomes[event_key]:
                archetype_outcomes[event_key][arch] = []
            
            archetype_outcomes[event_key][arch].append(outcome.result)
        
        # Find events with both archetypes
        correlation_data = []
        
        for event_key, event_outcomes in archetype_outcomes.items():
            if archetype1 in event_outcomes and archetype2 in event_outcomes:
                results1 = event_outcomes[archetype1]
                results2 = event_outcomes[archetype2]
                
                # Calculate win rates for this event
                win_rate1 = np.mean(results1)
                win_rate2 = np.mean(results2)
                
                correlation_data.append((win_rate1, win_rate2))
        
        if len(correlation_data) >= 10:  # Need sufficient data
            rates1, rates2 = zip(*correlation_data)
            correlation, _ = stats.pearsonr(rates1, rates2)
            
            return max(0, abs(correlation) * 0.15)  # Cap at 15%
        
        return 0.0
    
    def _calculate_heuristic_correlation(
        self,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate correlation using domain-specific heuristics."""
        correlation = 0.0
        
        # Weight class proximity correlation
        div1 = fight1_features.get('division', '')
        div2 = fight2_features.get('division', '')
        
        if div1 and div2:
            if div1 == div2:
                correlation += 0.03  # Same division
            elif self._are_adjacent_divisions(div1, div2):
                correlation += 0.01  # Adjacent divisions
        
        # Betting line correlation
        odds1 = fight1_features.get('odds', 2.0)
        odds2 = fight2_features.get('odds', 2.0)
        
        # Both heavy favorites
        if odds1 < 1.4 and odds2 < 1.4:
            correlation += 0.04
        
        # Both underdogs
        if odds1 > 2.8 and odds2 > 2.8:
            correlation += 0.02
        
        # Style matchup correlation
        if self._have_similar_styles(fight1_features, fight2_features):
            correlation += 0.02
        
        # Training camp correlation
        camp1 = fight1_features.get('camp', '')
        camp2 = fight2_features.get('camp', '')
        
        if camp1 and camp2 and camp1 == camp2:
            correlation += 0.08  # Same camp
        
        return min(correlation, 0.25)  # Cap at 25%
    
    def _are_adjacent_divisions(self, div1: str, div2: str) -> bool:
        """Check if two weight divisions are adjacent."""
        divisions = [
            'flyweight', 'bantamweight', 'featherweight', 'lightweight',
            'welterweight', 'middleweight', 'light_heavyweight', 'heavyweight'
        ]
        
        try:
            idx1 = divisions.index(div1.lower())
            idx2 = divisions.index(div2.lower())
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False
    
    def _have_similar_styles(self, fight1_features: Dict[str, Any], fight2_features: Dict[str, Any]) -> bool:
        """Check if fighters have similar fighting styles."""
        style1 = fight1_features.get('stance', '').lower()
        style2 = fight2_features.get('stance', '').lower()
        
        # Orthodox vs orthodox, southpaw vs southpaw
        if style1 and style2 and style1 == style2:
            return True
        
        # Check combat styles
        archetype1 = self._classify_fighter_archetype(fight1_features)
        archetype2 = self._classify_fighter_archetype(fight2_features)
        
        return archetype1 == archetype2
    
    def _calculate_correlation_confidence(
        self,
        sources: CorrelationSource,
        fight1_features: Dict[str, Any],
        fight2_features: Dict[str, Any]
    ) -> float:
        """Calculate confidence level in correlation estimate."""
        confidence_factors = []
        
        # Feature completeness
        feature_completeness = min(
            len([v for v in fight1_features.values() if v is not None]) / 20,
            len([v for v in fight2_features.values() if v is not None]) / 20
        )
        confidence_factors.append(min(feature_completeness, 1.0))
        
        # Historical data availability
        if len(self.historical_outcomes) >= self.min_historical_samples:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Source agreement (if multiple sources agree, higher confidence)
        source_values = [
            sources.same_event, sources.feature_similarity,
            sources.residual_historical, sources.heuristic
        ]
        non_zero_sources = [v for v in source_values if v > 0]
        
        if len(non_zero_sources) >= 2:
            # Check if sources are in agreement
            source_std = np.std(non_zero_sources)
            source_agreement = 1.0 - min(source_std / 0.1, 1.0)
            confidence_factors.append(source_agreement)
        else:
            confidence_factors.append(0.5)  # Single source reduces confidence
        
        # Return average confidence
        return np.mean(confidence_factors)
    
    def load_historical_data(self, filepath: str):
        """Load historical fight outcome data for residual analysis."""
        try:
            df = pd.read_csv(filepath)
            
            required_cols = ['event', 'fighter', 'opponent', 'result']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {filepath}")
                return
            
            self.historical_outcomes = []
            
            for _, row in df.iterrows():
                # Extract features (everything except required columns)
                features = {}
                for col in df.columns:
                    if col not in required_cols and col != 'timestamp':
                        features[col] = row[col]
                
                # Parse timestamp if available
                timestamp = datetime.now()
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    try:
                        timestamp = pd.to_datetime(row['timestamp'])
                    except:
                        pass
                
                outcome = HistoricalOutcome(
                    event=str(row['event']),
                    fighter=str(row['fighter']),
                    opponent=str(row['opponent']),
                    result=bool(row['result']),
                    features=features,
                    timestamp=timestamp
                )
                
                self.historical_outcomes.append(outcome)
            
            logger.info(f"✅ Loaded {len(self.historical_outcomes)} historical outcomes")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def update_residual_correlations(self, new_outcomes: List[HistoricalOutcome]):
        """Update residual correlation estimates with new outcome data."""
        self.historical_outcomes.extend(new_outcomes)
        
        # Clear cached correlations to force recalculation
        self.residual_correlations.clear()
        
        logger.info(f"Updated with {len(new_outcomes)} new outcomes")
    
    def get_correlation_diagnostics(self, fighter1: str, fighter2: str, 
                                  fight1_features: Dict, fight2_features: Dict,
                                  context: Dict) -> Dict[str, Any]:
        """Get detailed diagnostics for correlation estimation."""
        
        correlation, sources = self.estimate_comprehensive_correlation(
            fighter1, fighter2, fight1_features, fight2_features, context
        )
        
        return {
            'final_correlation': correlation,
            'source_breakdown': {
                'same_event': sources.same_event,
                'feature_similarity': sources.feature_similarity,
                'residual_historical': sources.residual_historical,
                'heuristic': sources.heuristic
            },
            'confidence': sources.confidence,
            'blend_weights': self.correlation_weights,
            'feature_count': len([v for v in {**fight1_features, **fight2_features}.values() if v is not None]),
            'historical_samples': len(self.historical_outcomes)
        }


if __name__ == "__main__":
    # Demo the advanced correlation engine
    print("🧠 Advanced Correlation Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = AdvancedCorrelationEngine()
    
    # Sample fight features
    fight1_features = {
        'event': 'UFC 309',
        'division': 'lightweight',
        'card_position': 1,
        'odds': 1.8,
        'striking_accuracy': 0.65,
        'takedown_accuracy': 0.45,
        'submission_attempts_avg': 0.8,
        'camp': 'American Top Team',
        'stance': 'orthodox',
        'age': 28,
        'reach': 74
    }
    
    fight2_features = {
        'event': 'UFC 309',
        'division': 'lightweight', 
        'card_position': 2,
        'odds': 2.2,
        'striking_accuracy': 0.62,
        'takedown_accuracy': 0.55,
        'submission_attempts_avg': 1.2,
        'camp': 'Team Alpha Male',
        'stance': 'orthodox',
        'age': 26,
        'reach': 72
    }
    
    context = {
        'event': 'UFC 309',
        'event_type': 'pay_per_view',
        'international_venue': False
    }
    
    # Estimate correlation
    correlation, sources = engine.estimate_comprehensive_correlation(
        "Fighter A", "Fighter B", fight1_features, fight2_features, context
    )
    
    # Get diagnostics
    diagnostics = engine.get_correlation_diagnostics(
        "Fighter A", "Fighter B", fight1_features, fight2_features, context
    )
    
    print(f"\n📊 CORRELATION ANALYSIS:")
    print(f"Final correlation: {correlation:.3f}")
    print(f"Confidence level: {sources.confidence:.3f}")
    
    print(f"\n🔍 SOURCE BREAKDOWN:")
    print(f"Same event: {sources.same_event:.3f}")
    print(f"Feature similarity: {sources.feature_similarity:.3f}")
    print(f"Historical residual: {sources.residual_historical:.3f}")
    print(f"Heuristic: {sources.heuristic:.3f}")
    
    print(f"\n📈 DIAGNOSTICS:")
    for key, value in diagnostics.items():
        if key != 'source_breakdown':
            print(f"{key}: {value}")
    
    print("\n✅ Advanced correlation engine ready!")