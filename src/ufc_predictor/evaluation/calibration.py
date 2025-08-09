"""
Probability calibration for UFC fight predictions.
Implements isotonic regression and Platt scaling with segment-specific calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    calibrator: Any
    segment: Optional[str]
    method: str
    pre_calibration_ece: float
    post_calibration_ece: float
    pre_calibration_brier: float
    post_calibration_brier: float
    n_samples: int


class UFCProbabilityCalibrator:
    """
    Calibrates predicted probabilities for UFC fight predictions.
    Supports segment-specific calibration by division, rounds, etc.
    """
    
    def __init__(
        self,
        method: str = 'isotonic',
        segment_cols: Optional[List[str]] = None,
        min_samples_for_segment: int = 100
    ):
        """
        Initialize calibrator.
        
        Args:
            method: 'isotonic' or 'platt' (sigmoid)
            segment_cols: Columns to create segments (e.g., ['division', 'rounds'])
            min_samples_for_segment: Minimum samples required for segment-specific calibration
        """
        if method not in ['isotonic', 'platt', 'sigmoid']:
            raise ValueError(f"Method must be 'isotonic' or 'platt', got {method}")
        
        self.method = 'sigmoid' if method == 'platt' else method
        self.segment_cols = segment_cols or []
        self.min_samples_for_segment = min_samples_for_segment
        self.calibrators: Dict[str, Any] = {}
        self.calibration_results: Dict[str, CalibrationResult] = {}
        
    def fit_isotonic_by_segment(
        self,
        oof_df: pd.DataFrame,
        prob_col: str = 'prob_pred',
        target_col: str = 'winner',
        metadata_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, CalibrationResult]:
        """
        Fit isotonic calibration by segments.
        
        Args:
            oof_df: DataFrame with out-of-fold predictions
            prob_col: Column with predicted probabilities
            target_col: Column with true labels
            metadata_df: Optional DataFrame with segment information
            
        Returns:
            Dictionary of calibration results by segment
        """
        if metadata_df is not None:
            df = pd.concat([oof_df, metadata_df], axis=1)
        else:
            df = oof_df.copy()
        
        # Overall calibration
        logger.info("Fitting overall calibration")
        overall_result = self._fit_single_calibrator(
            df[prob_col].values,
            df[target_col].values,
            segment='overall'
        )
        self.calibrators['overall'] = overall_result.calibrator
        self.calibration_results['overall'] = overall_result
        
        # Segment-specific calibration if requested
        if self.segment_cols and all(col in df.columns for col in self.segment_cols):
            # Create segment identifier
            df['segment'] = df[self.segment_cols].astype(str).agg('_'.join, axis=1)
            
            for segment in df['segment'].unique():
                segment_mask = df['segment'] == segment
                n_samples = segment_mask.sum()
                
                if n_samples >= self.min_samples_for_segment:
                    logger.info(f"Fitting calibration for segment: {segment} ({n_samples} samples)")
                    
                    segment_result = self._fit_single_calibrator(
                        df.loc[segment_mask, prob_col].values,
                        df.loc[segment_mask, target_col].values,
                        segment=segment
                    )
                    
                    self.calibrators[segment] = segment_result.calibrator
                    self.calibration_results[segment] = segment_result
                else:
                    logger.warning(f"Insufficient samples for segment {segment}: {n_samples}")
        
        return self.calibration_results
    
    def _fit_single_calibrator(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        segment: str = 'overall'
    ) -> CalibrationResult:
        """Fit a single calibrator and evaluate performance."""
        # Calculate pre-calibration metrics
        pre_ece = self._calculate_ece(y_true, y_prob)
        pre_brier = self._calculate_brier_score(y_true, y_prob)
        
        # Fit calibrator based on method
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            # Reshape for sklearn
            calibrated_prob = calibrator.fit_transform(y_prob, y_true)
        else:  # sigmoid/platt
            calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            # Reshape for sklearn
            calibrator.fit(y_prob.reshape(-1, 1), y_true)
            calibrated_prob = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        
        # Calculate post-calibration metrics
        post_ece = self._calculate_ece(y_true, calibrated_prob)
        post_brier = self._calculate_brier_score(y_true, calibrated_prob)
        
        # Log improvement
        ece_improvement = (pre_ece - post_ece) / pre_ece * 100 if pre_ece > 0 else 0
        brier_improvement = (pre_brier - post_brier) / pre_brier * 100 if pre_brier > 0 else 0
        
        logger.info(f"Segment {segment}: ECE improved by {ece_improvement:.1f}%, Brier improved by {brier_improvement:.1f}%")
        
        return CalibrationResult(
            calibrator=calibrator,
            segment=segment,
            method=self.method,
            pre_calibration_ece=pre_ece,
            post_calibration_ece=post_ece,
            pre_calibration_brier=pre_brier,
            post_calibration_brier=post_brier,
            n_samples=len(y_true)
        )
    
    def apply_calibration(
        self,
        prob_df: pd.DataFrame,
        prob_col: str = 'prob_pred',
        metadata_df: Optional[pd.DataFrame] = None,
        fallback_to_overall: bool = True
    ) -> np.ndarray:
        """
        Apply calibration to new predictions.
        
        Args:
            prob_df: DataFrame with predicted probabilities
            prob_col: Column with probabilities
            metadata_df: Optional DataFrame with segment information
            fallback_to_overall: Use overall calibration if segment not found
            
        Returns:
            Calibrated probabilities
        """
        if not self.calibrators:
            raise ValueError("Calibrator not fitted. Call fit_isotonic_by_segment first.")
        
        probs = prob_df[prob_col].values
        calibrated_probs = np.zeros_like(probs)
        
        # If no segments, use overall calibration
        if not self.segment_cols or metadata_df is None:
            calibrator = self.calibrators.get('overall')
            if calibrator:
                if self.method == 'isotonic':
                    calibrated_probs = calibrator.transform(probs)
                else:
                    calibrated_probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            return calibrated_probs
        
        # Apply segment-specific calibration
        df = pd.concat([prob_df, metadata_df], axis=1)
        df['segment'] = df[self.segment_cols].astype(str).agg('_'.join, axis=1)
        
        for segment in df['segment'].unique():
            segment_mask = df['segment'] == segment
            segment_calibrator = self.calibrators.get(segment)
            
            if segment_calibrator is None and fallback_to_overall:
                segment_calibrator = self.calibrators.get('overall')
                logger.debug(f"Using overall calibration for segment {segment}")
            
            if segment_calibrator:
                segment_probs = probs[segment_mask]
                if self.method == 'isotonic':
                    calibrated_probs[segment_mask] = segment_calibrator.transform(segment_probs)
                else:
                    calibrated_probs[segment_mask] = segment_calibrator.predict_proba(
                        segment_probs.reshape(-1, 1)
                    )[:, 1]
            else:
                # No calibration available
                calibrated_probs[segment_mask] = segment_probs
                logger.warning(f"No calibration available for segment {segment}")
        
        return calibrated_probs
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score."""
        return np.mean((y_prob - y_true) ** 2)
    
    def should_calibrate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ece_threshold: float = 0.05,
        odds_included: bool = False
    ) -> bool:
        """
        Determine if calibration would be beneficial.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            ece_threshold: ECE threshold for calibration decision
            odds_included: Whether odds are included in features
            
        Returns:
            Whether calibration is recommended
        """
        # Models with odds features are often already well-calibrated
        if odds_included:
            ece_threshold *= 2  # Be more conservative
        
        current_ece = self._calculate_ece(y_true, y_prob)
        
        # Check if reliability diagram is close to diagonal
        reliability_score = self._compute_reliability_score(y_true, y_prob)
        
        should_calibrate = current_ece > ece_threshold or reliability_score > 0.1
        
        logger.info(f"ECE: {current_ece:.4f}, Reliability: {reliability_score:.4f}, Calibrate: {should_calibrate}")
        
        return should_calibrate
    
    def _compute_reliability_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute reliability score (deviation from diagonal in reliability diagram).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        deviations = []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                expected_accuracy = bin_centers[i]
                deviations.append(abs(accuracy_in_bin - expected_accuracy))
        
        return np.mean(deviations) if deviations else 0.0
    
    def save_calibrators(self, filepath: str):
        """Save fitted calibrators to disk."""
        if not self.calibrators:
            raise ValueError("No calibrators to save. Fit first.")
        
        save_dict = {
            'calibrators': self.calibrators,
            'method': self.method,
            'segment_cols': self.segment_cols,
            'calibration_results': self.calibration_results
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Saved calibrators to {filepath}")
    
    def load_calibrators(self, filepath: str):
        """Load calibrators from disk."""
        save_dict = joblib.load(filepath)
        
        self.calibrators = save_dict['calibrators']
        self.method = save_dict['method']
        self.segment_cols = save_dict['segment_cols']
        self.calibration_results = save_dict['calibration_results']
        
        logger.info(f"Loaded calibrators from {filepath}")
    
    def get_calibration_summary(self) -> pd.DataFrame:
        """Get summary of calibration results."""
        if not self.calibration_results:
            return pd.DataFrame()
        
        summaries = []
        for segment, result in self.calibration_results.items():
            summary = {
                'segment': segment,
                'method': result.method,
                'n_samples': result.n_samples,
                'pre_ece': result.pre_calibration_ece,
                'post_ece': result.post_calibration_ece,
                'ece_improvement': (result.pre_calibration_ece - result.post_calibration_ece) / 
                                 result.pre_calibration_ece * 100 if result.pre_calibration_ece > 0 else 0,
                'pre_brier': result.pre_calibration_brier,
                'post_brier': result.post_calibration_brier,
                'brier_improvement': (result.pre_calibration_brier - result.post_calibration_brier) / 
                                   result.pre_calibration_brier * 100 if result.pre_calibration_brier > 0 else 0
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries).sort_values('n_samples', ascending=False)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
) -> Dict[str, Any]:
    """
    Create reliability diagram data (for plotting).
    
    Returns:
        Dictionary with bin data for plotting
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracies.append(y_true[in_bin].mean())
            confidences.append(y_prob[in_bin].mean())
            counts.append(in_bin.sum())
        else:
            accuracies.append(np.nan)
            confidences.append(bin_centers[i])
            counts.append(0)
    
    return {
        'bin_centers': bin_centers,
        'accuracies': accuracies,
        'confidences': confidences,
        'counts': counts,
        'perfect_calibration': bin_centers,  # Diagonal line
        'title': title
    }