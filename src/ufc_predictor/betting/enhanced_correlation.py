"""
Enhanced Parlay Correlation Engine

Implements sophisticated multi-source correlation estimation from the multi-bet plan:
- Feature vector similarity (cosine similarity of normalized fighter vectors)
- Residual co-movement analysis from historical betting outcomes
- Blended correlation scoring with confidence weights
- Advanced heuristics for camp/division/teammate penalties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CorrelationComponent:
    """Individual correlation component with confidence score."""
    source: str
    value: float
    confidence: float
    weight: float
    metadata: Dict = None


class EnhancedCorrelationEngine:
    """
    Multi-source correlation engine implementing the sophisticated blending strategy.
    
    Sources (with default weights):
    1. Same-event heuristics (0.40) - Base correlation for fights on same card
    2. Feature similarity (0.30) - Cosine similarity of normalized fighter vectors  
    3. Historical residuals (0.20) - Empirical co-movement from past outcomes
    4. Advanced heuristics (0.10) - Camp, division, teammate penalties
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 fighter_features: Optional[pd.DataFrame] = None,
                 historical_results: Optional[pd.DataFrame] = None):
        """Initialize the enhanced correlation engine."""
        self.config = config or self._default_config()
        self.fighter_features = fighter_features
        self.historical_results = historical_results
        
        # Processed data caches
        self.feature_vectors = {}
        self.scaler = StandardScaler()
        self.residual_correlations = {}
        
        if fighter_features is not None:
            self._prepare_feature_vectors()
        
        if historical_results is not None:
            self._compute_residual_correlations()
    
    def _default_config(self) -> Dict:
        """Default configuration matching multi-bet plan specifications."""
        return {
            'correlation_sources': {
                'same_event': {'weight': 0.40, 'base_correlation': 0.15},
                'feature_similarity': {'weight': 0.30, 'scaling_factor': 0.25},
                'historical_residuals': {'weight': 0.20, 'min_observations': 10},
                'advanced_heuristics': {'weight': 0.10, 'penalties': {
                    'same_camp': 0.30,
                    'same_division': 0.15,
                    'teammates': 0.40,
                    'similar_style': 0.20
                }}
            },
            'blending': {
                'confidence_threshold': 0.50,
                'min_total_confidence': 1.0,
                'uncertainty_penalty': 0.05
            },
            'validation': {
                'max_correlation': 0.60,
                'min_correlation': 0.00,
                'sanity_check': True
            }
        }
    
    def estimate_correlation(self, 
                            bet1: Dict, 
                            bet2: Dict,
                            context: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Main correlation estimation with full component breakdown.
        
        Args:
            bet1: First betting opportunity
            bet2: Second betting opportunity  
            context: Additional context (event info, etc.)
            
        Returns:
            Tuple of (final_correlation, detailed_breakdown)
        """
        components = []
        
        # 1. Same-event correlation (base heuristic)
        same_event_comp = self._compute_same_event_correlation(bet1, bet2, context)
        components.append(same_event_comp)
        
        # 2. Feature similarity correlation
        feature_comp = self._compute_feature_similarity(bet1, bet2)
        components.append(feature_comp)
        
        # 3. Historical residual correlation
        residual_comp = self._compute_residual_correlation(bet1, bet2)
        components.append(residual_comp)
        
        # 4. Advanced heuristic penalties
        heuristic_comp = self._compute_heuristic_correlation(bet1, bet2)
        components.append(heuristic_comp)
        
        # 5. Blend components using confidence weights
        final_correlation, blend_info = self._blend_components(components)
        
        # 6. Apply validation and bounds
        final_correlation = self._validate_correlation(final_correlation, bet1, bet2)
        
        # Prepare detailed breakdown
        breakdown = {
            'final_correlation': final_correlation,
            'components': components,
            'blend_info': blend_info,
            'validation_applied': True
        }
        
        logger.debug(f"Correlation {bet1.get('fighter', 'F1')} <-> {bet2.get('fighter', 'F2')}: {final_correlation:.3f}")
        
        return final_correlation, breakdown
    
    def _compute_same_event_correlation(self, 
                                       bet1: Dict, 
                                       bet2: Dict,
                                       context: Optional[Dict]) -> CorrelationComponent:
        """Compute same-event correlation with position-based adjustments."""
        config = self.config['correlation_sources']['same_event']
        base_corr = config['base_correlation']
        
        # Check if same event
        same_event = (bet1.get('event') == bet2.get('event') or
                     (context and context.get('same_event', False)))
        
        if not same_event:
            return CorrelationComponent(
                source='same_event',
                value=0.02,  # Small base correlation for different events
                confidence=0.95,
                weight=config['weight']
            )
        
        # Position-based adjustments
        pos1 = bet1.get('card_position', 999)
        pos2 = bet2.get('card_position', 999)
        position_diff = abs(pos1 - pos2)
        
        # Closer fights have higher correlation
        if position_diff <= 1:
            correlation = base_corr + 0.05  # Adjacent fights
        elif position_diff <= 3:
            correlation = base_corr         # Close fights
        else:
            correlation = base_corr - 0.03  # Distant fights
        
        return CorrelationComponent(
            source='same_event',
            value=max(0.05, correlation),  # Minimum floor
            confidence=0.90,
            weight=config['weight'],
            metadata={'position_diff': position_diff, 'same_event': True}
        )
    
    def _compute_feature_similarity(self, bet1: Dict, bet2: Dict) -> CorrelationComponent:
        """Compute correlation from normalized fighter feature similarity."""
        config = self.config['correlation_sources']['feature_similarity']
        
        if self.fighter_features is None:
            return CorrelationComponent(
                source='feature_similarity',
                value=0.10,  # Default when no features
                confidence=0.20,
                weight=config['weight']
            )
        
        # Get feature vectors for both fighters
        fighter1 = bet1.get('fighter')
        fighter2 = bet2.get('fighter')
        
        vec1 = self.feature_vectors.get(fighter1)
        vec2 = self.feature_vectors.get(fighter2)
        
        if vec1 is None or vec2 is None:
            return CorrelationComponent(
                source='feature_similarity',
                value=0.08,  # Default for missing features
                confidence=0.30,
                weight=config['weight']
            )
        
        # Compute cosine similarity
        similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
        
        # Convert similarity to correlation estimate
        correlation = similarity * config['scaling_factor']
        
        # Confidence based on feature completeness
        confidence = min(0.85, 0.50 + (min(len(vec1), 20) / 40))  # Higher for more features
        
        return CorrelationComponent(
            source='feature_similarity',
            value=max(0.02, correlation),
            confidence=confidence,
            weight=config['weight'],
            metadata={'similarity': similarity, 'vec_dims': len(vec1)}
        )
    
    def _compute_residual_correlation(self, bet1: Dict, bet2: Dict) -> CorrelationComponent:
        """Compute correlation from historical residual co-movement."""
        config = self.config['correlation_sources']['historical_residuals']
        
        if not self.residual_correlations:
            return CorrelationComponent(
                source='historical_residuals',
                value=0.05,
                confidence=0.20,
                weight=config['weight']
            )
        
        # Create fighter pair key (order-independent)
        fighters = sorted([bet1.get('fighter'), bet2.get('fighter')])
        pair_key = f"{fighters[0]}_{fighters[1]}"
        
        # Also check for same-division correlations
        div1 = bet1.get('weight_class', 'unknown')
        div2 = bet2.get('weight_class', 'unknown')
        division_key = f"{min(div1, div2)}_{max(div1, div2)}" if div1 == div2 else None
        
        # Check specific pair correlation
        if pair_key in self.residual_correlations:
            pair_data = self.residual_correlations[pair_key]
            if pair_data['n_observations'] >= config['min_observations']:
                return CorrelationComponent(
                    source='historical_residuals',
                    value=abs(pair_data['correlation']),
                    confidence=min(0.80, pair_data['n_observations'] / 50),
                    weight=config['weight'],
                    metadata=pair_data
                )
        
        # Check division-level correlation
        if division_key and division_key in self.residual_correlations:
            div_data = self.residual_correlations[division_key]
            if div_data['n_observations'] >= config['min_observations']:
                return CorrelationComponent(
                    source='historical_residuals',
                    value=abs(div_data['correlation']) * 0.6,  # Discount for broader correlation
                    confidence=min(0.60, div_data['n_observations'] / 100),
                    weight=config['weight'],
                    metadata=div_data
                )
        
        # Default low correlation with low confidence
        return CorrelationComponent(
            source='historical_residuals',
            value=0.03,
            confidence=0.25,
            weight=config['weight']
        )
    
    def _compute_heuristic_correlation(self, bet1: Dict, bet2: Dict) -> CorrelationComponent:
        """Compute correlation from advanced heuristics (camps, teammates, etc.)."""
        config = self.config['correlation_sources']['advanced_heuristics']
        penalties = config['penalties']
        
        correlation = 0.02  # Base heuristic correlation
        applied_penalties = []
        
        # Same camp penalty
        camp1 = bet1.get('camp', bet1.get('fighter', '').split()[-1])  # Fallback to last name
        camp2 = bet2.get('camp', bet2.get('fighter', '').split()[-1])
        if camp1 and camp2 and camp1.lower() == camp2.lower():
            correlation += penalties['same_camp']
            applied_penalties.append('same_camp')
        
        # Same division correlation
        if (bet1.get('weight_class') == bet2.get('weight_class') and 
            bet1.get('weight_class') != 'unknown'):
            correlation += penalties['same_division']
            applied_penalties.append('same_division')
        
        # Teammate penalties (simplified check)
        fighter1_parts = set(bet1.get('fighter', '').lower().split())
        fighter2_parts = set(bet2.get('fighter', '').lower().split())
        if len(fighter1_parts.intersection(fighter2_parts)) > 0:
            correlation += penalties['teammates']
            applied_penalties.append('similar_names')
        
        # Fighting style similarity (simplified)
        style1 = bet1.get('fighting_style', 'unknown')
        style2 = bet2.get('fighting_style', 'unknown')
        if (style1 != 'unknown' and style2 != 'unknown' and 
            style1 == style2):
            correlation += penalties['similar_style']
            applied_penalties.append('similar_style')
        
        # Confidence based on number of heuristics applied
        confidence = 0.40 + (len(applied_penalties) * 0.15)
        
        return CorrelationComponent(
            source='advanced_heuristics',
            value=min(0.40, correlation),  # Cap maximum heuristic correlation
            confidence=min(0.80, confidence),
            weight=config['weight'],
            metadata={'penalties_applied': applied_penalties}
        )
    
    def _blend_components(self, components: List[CorrelationComponent]) -> Tuple[float, Dict]:
        """Blend correlation components using confidence-weighted averaging."""
        config = self.config['blending']
        
        # Filter components by confidence threshold
        valid_components = [c for c in components if c.confidence >= config['confidence_threshold']]
        
        if not valid_components:
            # Use all components if none meet threshold
            valid_components = components
        
        # Calculate total confidence-weighted value
        total_weighted_value = 0
        total_weight = 0
        
        for comp in valid_components:
            effective_weight = comp.weight * comp.confidence
            total_weighted_value += comp.value * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            final_correlation = 0.05  # Fallback default
        else:
            final_correlation = total_weighted_value / total_weight
        
        # Apply uncertainty penalty if total confidence is low
        total_confidence = sum(c.confidence for c in valid_components)
        if total_confidence < config['min_total_confidence']:
            uncertainty_factor = total_confidence / config['min_total_confidence']
            final_correlation *= uncertainty_factor
            final_correlation += config['uncertainty_penalty'] * (1 - uncertainty_factor)
        
        blend_info = {
            'n_components_used': len(valid_components),
            'total_confidence': total_confidence,
            'total_weight': total_weight,
            'uncertainty_applied': total_confidence < config['min_total_confidence']
        }
        
        return final_correlation, blend_info
    
    def _validate_correlation(self, correlation: float, bet1: Dict, bet2: Dict) -> float:
        """Apply validation and bounds to correlation estimate."""
        config = self.config['validation']
        
        # Apply bounds
        correlation = np.clip(correlation, config['min_correlation'], config['max_correlation'])
        
        # Sanity checks
        if config['sanity_check']:
            # Same fighter should have perfect correlation (impossible case, but safety)
            if bet1.get('fighter') == bet2.get('fighter'):
                correlation = 1.0
            
            # Fights involving same opponents should have higher correlation
            if (bet1.get('opponent') == bet2.get('fighter') or 
                bet1.get('fighter') == bet2.get('opponent')):
                correlation = max(correlation, 0.25)
        
        return correlation
    
    def _prepare_feature_vectors(self):
        """Prepare normalized feature vectors for all fighters."""
        if self.fighter_features is None:
            return
        
        # Select numerical features for similarity calculation
        numerical_cols = self.fighter_features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if '_diff' not in col]  # Avoid differential features
        
        if not feature_cols:
            logger.warning("No suitable feature columns found for similarity calculation")
            return
        
        # Prepare data matrix
        feature_matrix = self.fighter_features[feature_cols].fillna(0)
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_matrix)
        
        # Store feature vectors by fighter name
        if 'fighter' in self.fighter_features.columns:
            for i, fighter in enumerate(self.fighter_features['fighter']):
                self.feature_vectors[fighter] = normalized_features[i]
        
        logger.info(f"Prepared feature vectors for {len(self.feature_vectors)} fighters")
    
    def _compute_residual_correlations(self):
        """Compute empirical correlations from historical betting results."""
        if self.historical_results is None:
            return
        
        # Group results by event to find same-event outcomes
        if 'event' not in self.historical_results.columns:
            logger.warning("No event column in historical results")
            return
        
        events = self.historical_results.groupby('event')
        
        for event_name, event_df in events:
            if len(event_df) < 2:
                continue
            
            # Convert results to binary (1 for win, 0 for loss)
            results = []
            fighters = []
            
            for _, row in event_df.iterrows():
                result = 1 if row.get('actual_result') == 'WIN' else 0
                results.append(result)
                fighters.append(row.get('fighter'))
            
            # Compute pairwise correlations
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    # Fighter pair key
                    pair_key = f"{min(fighters[i], fighters[j])}_{max(fighters[i], fighters[j])}"
                    
                    if pair_key not in self.residual_correlations:
                        self.residual_correlations[pair_key] = {
                            'correlation': 0,
                            'outcomes': [],
                            'n_observations': 0
                        }
                    
                    # Store outcome pair
                    self.residual_correlations[pair_key]['outcomes'].append((results[i], results[j]))
                    self.residual_correlations[pair_key]['n_observations'] += 1
        
        # Calculate final correlations
        for pair_key, data in self.residual_correlations.items():
            if data['n_observations'] >= 3:  # Minimum observations
                outcomes = np.array(data['outcomes'])
                if outcomes.shape[0] > 1:
                    try:
                        corr, _ = pearsonr(outcomes[:, 0], outcomes[:, 1])
                        data['correlation'] = corr if not np.isnan(corr) else 0
                    except:
                        data['correlation'] = 0
        
        logger.info(f"Computed residual correlations for {len(self.residual_correlations)} pairs")
    
    def generate_correlation_report(self, bet1: Dict, bet2: Dict) -> str:
        """Generate detailed correlation analysis report."""
        correlation, breakdown = self.estimate_correlation(bet1, bet2)
        
        report = []
        report.append("=" * 60)
        report.append("CORRELATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Fighter 1: {bet1.get('fighter', 'Unknown')}")
        report.append(f"Fighter 2: {bet2.get('fighter', 'Unknown')}")
        report.append(f"Final Correlation: {correlation:.3f}")
        report.append("")
        
        report.append("COMPONENT BREAKDOWN:")
        report.append("-" * 40)
        
        for comp in breakdown['components']:
            report.append(f"{comp.source.upper()}:")
            report.append(f"  Value: {comp.value:.3f}")
            report.append(f"  Confidence: {comp.confidence:.3f}")
            report.append(f"  Weight: {comp.weight:.3f}")
            if comp.metadata:
                report.append(f"  Metadata: {comp.metadata}")
            report.append("")
        
        blend_info = breakdown['blend_info']
        report.append("BLENDING INFO:")
        report.append("-" * 40)
        report.append(f"Components used: {blend_info['n_components_used']}")
        report.append(f"Total confidence: {blend_info['total_confidence']:.2f}")
        report.append(f"Uncertainty applied: {blend_info['uncertainty_applied']}")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Enhanced Parlay Correlation Engine")
    print("Multi-source correlation estimation with confidence blending")