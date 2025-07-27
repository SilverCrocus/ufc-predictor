"""
Data Confidence Scoring System

Provides intelligent confidence scoring for different data sources in the
hybrid odds system, helping to determine which data source to trust and
how to weight different sources when reconciling data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for evaluating data quality"""
    completeness: float  # 0.0 to 1.0 - how complete is the data
    freshness: float     # 0.0 to 1.0 - how recent is the data
    consistency: float   # 0.0 to 1.0 - how consistent with historical patterns
    accuracy: float      # 0.0 to 1.0 - historical accuracy score
    reliability: float   # 0.0 to 1.0 - source reliability score
    total_score: float   # Weighted total score


class DataConfidenceScorer:
    """
    Intelligent confidence scoring system for multi-source odds data
    
    Evaluates data quality across multiple dimensions:
    - Completeness: How much of the expected data is present
    - Freshness: How recent the data is
    - Consistency: How consistent with historical patterns
    - Accuracy: Historical accuracy of the source
    - Reliability: Overall source reliability
    """
    
    def __init__(self):
        """Initialize the confidence scorer"""
        # Source reliability scores based on historical performance
        self.source_reliability = {
            'api': 0.95,           # High reliability for official API
            'tab_scraper': 0.85,   # Good reliability for TAB scraper
            'cached': 0.70,        # Lower reliability for cached data
            'manual': 0.60,        # Lowest for manual entry
            'reconciled': 0.90     # High for reconciled data
        }
        
        # Weights for different quality dimensions
        self.quality_weights = {
            'completeness': 0.25,
            'freshness': 0.30,
            'consistency': 0.20,
            'accuracy': 0.15,
            'reliability': 0.10
        }
        
        # Data completeness expectations
        self.expected_fields = [
            'fighter_a', 'fighter_b', 
            'fighter_a_decimal_odds', 'fighter_b_decimal_odds',
            'bookmakers'
        ]
        
        logger.info("Data confidence scorer initialized")
    
    def score_api_data(self, odds_data: Dict[str, Dict], 
                      response_time: float = None) -> float:
        """
        Score confidence for API data
        
        Args:
            odds_data: Formatted odds data from API
            response_time: API response time in seconds
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        metrics = DataQualityMetrics(
            completeness=self._calculate_completeness(odds_data),
            freshness=1.0,  # API data is always fresh
            consistency=self._calculate_api_consistency(odds_data),
            accuracy=self.source_reliability['api'],
            reliability=self.source_reliability['api'],
            total_score=0.0
        )
        
        # Apply response time penalty if available
        if response_time:
            response_penalty = min(0.1, response_time / 10)  # Penalty for slow responses
            metrics.freshness = max(0.0, metrics.freshness - response_penalty)
        
        metrics.total_score = self._calculate_weighted_score(metrics)
        
        logger.debug(f"API data confidence: {metrics.total_score:.3f}")
        return metrics.total_score
    
    def score_tab_data(self, tab_odds: List) -> float:
        """
        Score confidence for TAB scraper data
        
        Args:
            tab_odds: List of TABFightOdds objects
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Convert to standard format for analysis
        odds_data = {}
        for fight in tab_odds:
            fight_key = f"{fight.fighter_a} vs {fight.fighter_b}"
            odds_data[fight_key] = {
                'fighter_a': fight.fighter_a,
                'fighter_b': fight.fighter_b,
                'fighter_a_decimal_odds': fight.fighter_a_decimal_odds,
                'fighter_b_decimal_odds': fight.fighter_b_decimal_odds,
                'bookmakers': ['TAB Australia']
            }
        
        metrics = DataQualityMetrics(
            completeness=self._calculate_completeness(odds_data),
            freshness=0.9,  # Scraper data is recent but not real-time
            consistency=self._calculate_scraper_consistency(odds_data),
            accuracy=self.source_reliability['tab_scraper'],
            reliability=self.source_reliability['tab_scraper'],
            total_score=0.0
        )
        
        metrics.total_score = self._calculate_weighted_score(metrics)
        
        logger.debug(f"TAB scraper confidence: {metrics.total_score:.3f}")
        return metrics.total_score
    
    def score_cached_data(self, odds_data: Dict[str, Dict], 
                         age_hours: float) -> float:
        """
        Score confidence for cached data
        
        Args:
            odds_data: Cached odds data
            age_hours: Age of cached data in hours
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        metrics = DataQualityMetrics(
            completeness=self._calculate_completeness(odds_data),
            freshness=self._calculate_freshness_score(age_hours),
            consistency=self._calculate_cached_consistency(odds_data),
            accuracy=self.source_reliability['cached'],
            reliability=self.source_reliability['cached'],
            total_score=0.0
        )
        
        metrics.total_score = self._calculate_weighted_score(metrics)
        
        logger.debug(f"Cached data confidence: {metrics.total_score:.3f} (age: {age_hours:.1f}h)")
        return metrics.total_score
    
    def score_reconciled_data(self, source_scores: List[float], 
                            agreement_score: float) -> float:
        """
        Score confidence for reconciled data from multiple sources
        
        Args:
            source_scores: List of confidence scores from individual sources
            agreement_score: How well the sources agreed (0.0 to 1.0)
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Base score is average of source scores
        base_score = statistics.mean(source_scores) if source_scores else 0.5
        
        # Boost score based on agreement and multiple sources
        source_diversity_bonus = min(0.1, len(source_scores) * 0.03)  # Bonus for multiple sources
        agreement_bonus = agreement_score * 0.15  # Bonus for good agreement
        
        reconciled_score = min(1.0, base_score + source_diversity_bonus + agreement_bonus)
        
        logger.debug(f"Reconciled data confidence: {reconciled_score:.3f} "
                    f"(sources: {len(source_scores)}, agreement: {agreement_score:.3f})")
        
        return reconciled_score
    
    def _calculate_completeness(self, odds_data: Dict[str, Dict]) -> float:
        """
        Calculate data completeness score
        
        Args:
            odds_data: Odds data to evaluate
            
        Returns:
            Completeness score from 0.0 to 1.0
        """
        if not odds_data:
            return 0.0
        
        total_fields = 0
        present_fields = 0
        
        for fight_key, fight_data in odds_data.items():
            for field in self.expected_fields:
                total_fields += 1
                if field in fight_data and fight_data[field] is not None:
                    # Additional validation for odds fields
                    if 'odds' in field:
                        try:
                            odds_value = float(fight_data[field])
                            if 1.0 <= odds_value <= 100.0:  # Reasonable odds range
                                present_fields += 1
                        except (ValueError, TypeError):
                            pass
                    else:
                        present_fields += 1
        
        return present_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_freshness_score(self, age_hours: float) -> float:
        """
        Calculate freshness score based on data age
        
        Args:
            age_hours: Age of data in hours
            
        Returns:
            Freshness score from 0.0 to 1.0
        """
        if age_hours <= 0:
            return 1.0
        elif age_hours <= 1:
            return 0.95
        elif age_hours <= 4:
            return 0.85
        elif age_hours <= 12:
            return 0.70
        elif age_hours <= 24:
            return 0.50
        elif age_hours <= 48:
            return 0.30
        else:
            return 0.10
    
    def _calculate_api_consistency(self, odds_data: Dict[str, Dict]) -> float:
        """
        Calculate consistency score for API data
        
        Args:
            odds_data: API odds data
            
        Returns:
            Consistency score from 0.0 to 1.0
        """
        if not odds_data:
            return 0.0
        
        consistency_scores = []
        
        for fight_key, fight_data in odds_data.items():
            # Check odds rationality
            odds_a = fight_data.get('fighter_a_decimal_odds', 0)
            odds_b = fight_data.get('fighter_b_decimal_odds', 0)
            
            if odds_a > 0 and odds_b > 0:
                # Check if implied probabilities sum to reasonable range (1.0 to 1.1)
                implied_prob_sum = (1/odds_a) + (1/odds_b)
                if 0.95 <= implied_prob_sum <= 1.15:  # Allow for bookmaker margin
                    consistency_scores.append(0.9)
                elif 0.90 <= implied_prob_sum <= 1.20:
                    consistency_scores.append(0.7)
                else:
                    consistency_scores.append(0.3)
            else:
                consistency_scores.append(0.1)
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_scraper_consistency(self, odds_data: Dict[str, Dict]) -> float:
        """
        Calculate consistency score for scraper data
        
        Args:
            odds_data: Scraper odds data
            
        Returns:
            Consistency score from 0.0 to 1.0
        """
        # Similar to API consistency but with slightly lower baseline
        base_consistency = self._calculate_api_consistency(odds_data)
        
        # Scraper data might have slight inconsistencies due to parsing
        scraper_penalty = 0.05
        
        return max(0.0, base_consistency - scraper_penalty)
    
    def _calculate_cached_consistency(self, odds_data: Dict[str, Dict]) -> float:
        """
        Calculate consistency score for cached data
        
        Args:
            odds_data: Cached odds data
            
        Returns:
            Consistency score from 0.0 to 1.0
        """
        # Cached data consistency depends on when it was originally sourced
        base_consistency = self._calculate_api_consistency(odds_data)
        
        # Cached data is generally consistent with itself
        return base_consistency
    
    def _calculate_weighted_score(self, metrics: DataQualityMetrics) -> float:
        """
        Calculate weighted total score from individual metrics
        
        Args:
            metrics: Data quality metrics
            
        Returns:
            Weighted total score from 0.0 to 1.0
        """
        total_score = (
            metrics.completeness * self.quality_weights['completeness'] +
            metrics.freshness * self.quality_weights['freshness'] +
            metrics.consistency * self.quality_weights['consistency'] +
            metrics.accuracy * self.quality_weights['accuracy'] +
            metrics.reliability * self.quality_weights['reliability']
        )
        
        return max(0.0, min(1.0, total_score))
    
    def compare_sources(self, source_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare confidence scores across multiple sources
        
        Args:
            source_scores: Dict mapping source names to confidence scores
            
        Returns:
            Dict with comparison analysis
        """
        if not source_scores:
            return {'error': 'No sources to compare'}
        
        sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        best_source = sorted_sources[0]
        
        analysis = {
            'best_source': {
                'name': best_source[0],
                'score': best_source[1]
            },
            'source_ranking': sorted_sources,
            'score_spread': max(source_scores.values()) - min(source_scores.values()),
            'average_score': statistics.mean(source_scores.values()),
            'recommendation': self._get_source_recommendation(sorted_sources)
        }
        
        return analysis
    
    def _get_source_recommendation(self, sorted_sources: List[tuple]) -> str:
        """
        Get recommendation for source selection
        
        Args:
            sorted_sources: List of (source_name, score) tuples sorted by score
            
        Returns:
            Recommendation string
        """
        if not sorted_sources:
            return "No sources available"
        
        best_score = sorted_sources[0][1]
        
        if best_score >= 0.9:
            return f"Use {sorted_sources[0][0]} - excellent confidence"
        elif best_score >= 0.8:
            return f"Use {sorted_sources[0][0]} - good confidence"
        elif best_score >= 0.7:
            return f"Use {sorted_sources[0][0]} with caution - moderate confidence"
        elif len(sorted_sources) > 1:
            return "Consider reconciling multiple sources - low individual confidence"
        else:
            return "Data quality concerns - verify manually if possible"
    
    def update_source_reliability(self, source_name: str, accuracy_feedback: float):
        """
        Update source reliability based on accuracy feedback
        
        Args:
            source_name: Name of the data source
            accuracy_feedback: Accuracy score from 0.0 to 1.0
        """
        if source_name in self.source_reliability:
            # Use exponential moving average to update reliability
            alpha = 0.1  # Learning rate
            current_reliability = self.source_reliability[source_name]
            
            self.source_reliability[source_name] = (
                alpha * accuracy_feedback + (1 - alpha) * current_reliability
            )
            
            logger.info(f"Updated {source_name} reliability: "
                       f"{current_reliability:.3f} -> {self.source_reliability[source_name]:.3f}")
    
    def get_quality_report(self, source_name: str, odds_data: Dict[str, Dict], 
                          additional_info: Dict = None) -> Dict[str, Any]:
        """
        Generate detailed quality report for a data source
        
        Args:
            source_name: Name of the data source
            odds_data: Odds data to analyze
            additional_info: Additional information for scoring
            
        Returns:
            Detailed quality report
        """
        if source_name == 'api':
            confidence = self.score_api_data(odds_data, 
                                           additional_info.get('response_time') if additional_info else None)
        elif source_name == 'tab_scraper':
            # Convert format if needed
            confidence = self.score_tab_data([])  # Would need actual TABFightOdds objects
        elif source_name == 'cached':
            age_hours = additional_info.get('age_hours', 0) if additional_info else 0
            confidence = self.score_cached_data(odds_data, age_hours)
        else:
            confidence = 0.5  # Default score
        
        report = {
            'source_name': source_name,
            'overall_confidence': confidence,
            'data_completeness': self._calculate_completeness(odds_data),
            'source_reliability': self.source_reliability.get(source_name, 0.5),
            'fight_count': len(odds_data),
            'timestamp': datetime.now().isoformat(),
            'quality_breakdown': {
                'completeness': self._calculate_completeness(odds_data),
                'consistency': self._calculate_api_consistency(odds_data),
                'reliability': self.source_reliability.get(source_name, 0.5)
            }
        }
        
        if additional_info:
            report['additional_info'] = additional_info
        
        return report