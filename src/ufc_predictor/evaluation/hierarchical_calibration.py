"""
Hierarchical Calibration System for UFC Predictions

Implements fallback strategy for segments with insufficient data:
1. Segment-specific (gender × weight class × rounds) if n >= 200
2. Gender-specific if 50 <= n < 200  
3. Overall calibration if n < 50

This addresses the sparse data problem in women's divisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
import json
from datetime import datetime


class HierarchicalCalibrator:
    """Hierarchical probability calibration with automatic fallback."""
    
    def __init__(self, 
                 min_samples_segment: int = 200,
                 min_samples_gender: int = 50,
                 method: str = 'isotonic'):
        """
        Initialize hierarchical calibrator.
        
        Args:
            min_samples_segment: Minimum samples for segment-specific calibration
            min_samples_gender: Minimum samples for gender-specific calibration
            method: 'isotonic' or 'platt' calibration
        """
        self.min_samples_segment = min_samples_segment
        self.min_samples_gender = min_samples_gender
        self.method = method
        
        # Storage for calibrators at different levels
        self.segment_calibrators = {}  # (gender, division, rounds) -> calibrator
        self.gender_calibrators = {}   # gender -> calibrator
        self.overall_calibrator = None
        
        # Track sample sizes for decision making
        self.segment_samples = {}
        self.gender_samples = {}
        self.overall_samples = 0
        
        # Performance tracking
        self.calibration_stats = {
            'segment_ece': {},
            'gender_ece': {},
            'overall_ece': None
        }
    
    def fit(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Fit calibrators at all hierarchy levels.
        
        Args:
            X: Features including gender, weight_class, rounds
            y_true: True binary outcomes (0/1)
            y_pred: Predicted probabilities
        """
        # Ensure we have required columns
        required = ['gender', 'weight_class', 'rounds']
        missing = [col for col in required if col not in X.columns]
        if missing:
            # Try to infer from other columns
            X = self._infer_metadata(X)
        
        # Fit overall calibrator (always)
        self.overall_calibrator = self._fit_calibrator(y_true, y_pred)
        self.overall_samples = len(y_true)
        self.calibration_stats['overall_ece'] = self._calculate_ece(y_true, y_pred)
        
        # Fit gender-level calibrators
        for gender in X['gender'].unique():
            mask = X['gender'] == gender
            if mask.sum() >= self.min_samples_gender:
                gender_y_true = y_true[mask]
                gender_y_pred = y_pred[mask]
                
                self.gender_calibrators[gender] = self._fit_calibrator(
                    gender_y_true, gender_y_pred
                )
                self.gender_samples[gender] = mask.sum()
                self.calibration_stats['gender_ece'][gender] = self._calculate_ece(
                    gender_y_true, gender_y_pred
                )
        
        # Fit segment-level calibrators
        segments = X.groupby(['gender', 'weight_class', 'rounds']).size()
        
        for (gender, weight_class, rounds), count in segments.items():
            if count >= self.min_samples_segment:
                segment_key = (gender, weight_class, rounds)
                mask = (
                    (X['gender'] == gender) & 
                    (X['weight_class'] == weight_class) & 
                    (X['rounds'] == rounds)
                )
                
                segment_y_true = y_true[mask]
                segment_y_pred = y_pred[mask]
                
                self.segment_calibrators[segment_key] = self._fit_calibrator(
                    segment_y_true, segment_y_pred
                )
                self.segment_samples[segment_key] = count
                self.calibration_stats['segment_ece'][segment_key] = self._calculate_ece(
                    segment_y_true, segment_y_pred
                )
        
        print(f"✅ Hierarchical Calibration Fitted:")
        print(f"   • Segments: {len(self.segment_calibrators)}")
        print(f"   • Genders: {len(self.gender_calibrators)}")
        print(f"   • Overall: Yes (n={self.overall_samples})")
    
    def predict_proba(self, X: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply hierarchical calibration to predictions.
        
        Args:
            X: Features including gender, weight_class, rounds
            y_pred: Uncalibrated predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        # Ensure metadata columns exist
        X = self._infer_metadata(X)
        
        calibrated = np.zeros_like(y_pred)
        calibration_levels = []
        
        for i, row in X.iterrows():
            # Try segment-specific first
            segment_key = (row['gender'], row['weight_class'], row['rounds'])
            
            if segment_key in self.segment_calibrators:
                calibrated[i] = self.segment_calibrators[segment_key].transform(
                    [y_pred[i]]
                )[0]
                calibration_levels.append('segment')
                
            # Fall back to gender-specific
            elif row['gender'] in self.gender_calibrators:
                calibrated[i] = self.gender_calibrators[row['gender']].transform(
                    [y_pred[i]]
                )[0]
                calibration_levels.append('gender')
                
            # Fall back to overall
            elif self.overall_calibrator is not None:
                calibrated[i] = self.overall_calibrator.transform(
                    [y_pred[i]]
                )[0]
                calibration_levels.append('overall')
                
            else:
                # No calibration available
                calibrated[i] = y_pred[i]
                calibration_levels.append('none')
        
        # Log calibration level distribution
        level_counts = pd.Series(calibration_levels).value_counts()
        print(f"📊 Calibration Levels Used:")
        for level, count in level_counts.items():
            print(f"   • {level}: {count} ({count/len(calibration_levels)*100:.1f}%)")
        
        return calibrated
    
    def _fit_calibrator(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit a single calibrator based on method."""
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # platt
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
        
        # Handle edge cases
        if len(np.unique(y_true)) == 1:
            # All same class - return identity calibrator
            return lambda x: x
        
        calibrator.fit(y_pred.reshape(-1, 1), y_true)
        return calibrator
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _infer_metadata(self, X: pd.DataFrame) -> pd.DataFrame:
        """Infer gender, weight_class, rounds from other columns."""
        X = X.copy()
        
        # Infer gender from fighter names or features
        if 'gender' not in X.columns:
            # Check for women's weight classes
            women_classes = ['strawweight', 'flyweight', 'bantamweight', 'featherweight']
            if 'weight_class' in X.columns:
                X['gender'] = X['weight_class'].apply(
                    lambda x: 'F' if any(w in str(x).lower() for w in women_classes) else 'M'
                )
            else:
                # Default to male (majority class)
                X['gender'] = 'M'
        
        # Infer weight class
        if 'weight_class' not in X.columns:
            if 'Weight_lbs' in X.columns:
                X['weight_class'] = pd.cut(
                    X['Weight_lbs'],
                    bins=[0, 125, 135, 145, 155, 170, 185, 205, 265, 400],
                    labels=['flyweight', 'bantamweight', 'featherweight', 'lightweight',
                           'welterweight', 'middleweight', 'light_heavyweight', 'heavyweight']
                )
            else:
                X['weight_class'] = 'unknown'
        
        # Infer rounds
        if 'rounds' not in X.columns:
            if 'Round' in X.columns:
                X['rounds'] = X['Round'].apply(lambda x: 5 if x > 3 else 3)
            else:
                X['rounds'] = 3  # Default to 3-round fights
        
        return X
    
    def get_calibration_report(self) -> Dict:
        """Generate comprehensive calibration report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'method': self.method,
            'hierarchy_stats': {
                'n_segments': len(self.segment_calibrators),
                'n_genders': len(self.gender_calibrators),
                'overall_samples': self.overall_samples
            },
            'ece_scores': self.calibration_stats,
            'segment_coverage': {}
        }
        
        # Calculate coverage by segment
        for segment, n_samples in self.segment_samples.items():
            report['segment_coverage'][str(segment)] = {
                'samples': n_samples,
                'ece': self.calibration_stats['segment_ece'].get(segment, None)
            }
        
        return report
    
    def save(self, filepath: str):
        """Save calibrator to disk."""
        save_dict = {
            'segment_calibrators': self.segment_calibrators,
            'gender_calibrators': self.gender_calibrators,
            'overall_calibrator': self.overall_calibrator,
            'segment_samples': self.segment_samples,
            'gender_samples': self.gender_samples,
            'overall_samples': self.overall_samples,
            'calibration_stats': self.calibration_stats,
            'min_samples_segment': self.min_samples_segment,
            'min_samples_gender': self.min_samples_gender,
            'method': self.method
        }
        
        joblib.dump(save_dict, filepath)
        print(f"✅ Calibrator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HierarchicalCalibrator':
        """Load calibrator from disk."""
        save_dict = joblib.load(filepath)
        
        calibrator = cls(
            min_samples_segment=save_dict['min_samples_segment'],
            min_samples_gender=save_dict['min_samples_gender'],
            method=save_dict['method']
        )
        
        calibrator.segment_calibrators = save_dict['segment_calibrators']
        calibrator.gender_calibrators = save_dict['gender_calibrators']
        calibrator.overall_calibrator = save_dict['overall_calibrator']
        calibrator.segment_samples = save_dict['segment_samples']
        calibrator.gender_samples = save_dict['gender_samples']
        calibrator.overall_samples = save_dict['overall_samples']
        calibrator.calibration_stats = save_dict['calibration_stats']
        
        return calibrator


def apply_hierarchical_calibration(df: pd.DataFrame, 
                                  model_path: str = None,
                                  calibrator_path: str = None) -> pd.DataFrame:
    """
    Apply hierarchical calibration to fight predictions.
    
    Args:
        df: DataFrame with predictions and metadata
        model_path: Path to trained model
        calibrator_path: Path to saved calibrator
        
    Returns:
        DataFrame with calibrated probabilities
    """
    # Load or create calibrator
    if calibrator_path and Path(calibrator_path).exists():
        calibrator = HierarchicalCalibrator.load(calibrator_path)
    else:
        # Fit new calibrator
        calibrator = HierarchicalCalibrator()
        
        # Use historical data for fitting
        # This would use your out-of-fold predictions
        calibrator.fit(df, df['actual'].values, df['predicted'].values)
        
        if calibrator_path:
            calibrator.save(calibrator_path)
    
    # Apply calibration
    df['calibrated_prob'] = calibrator.predict_proba(df, df['predicted'].values)
    
    # Calculate improvement
    original_ece = calibrator._calculate_ece(df['actual'].values, df['predicted'].values)
    calibrated_ece = calibrator._calculate_ece(df['actual'].values, df['calibrated_prob'].values)
    
    print(f"\n📊 Calibration Results:")
    print(f"   • Original ECE: {original_ece:.4f}")
    print(f"   • Calibrated ECE: {calibrated_ece:.4f}")
    print(f"   • Improvement: {(1 - calibrated_ece/original_ece)*100:.1f}%")
    
    return df


if __name__ == "__main__":
    # Demo with synthetic data
    print("🧪 Testing Hierarchical Calibration System")
    print("=" * 50)
    
    # Generate synthetic fight data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic metadata
    genders = np.random.choice(['M', 'F'], n_samples, p=[0.85, 0.15])
    weight_classes = np.random.choice(
        ['lightweight', 'welterweight', 'middleweight', 'heavyweight'],
        n_samples
    )
    rounds = np.random.choice([3, 5], n_samples, p=[0.8, 0.2])
    
    # Create DataFrame
    X = pd.DataFrame({
        'gender': genders,
        'weight_class': weight_classes,
        'rounds': rounds
    })
    
    # Generate synthetic predictions (intentionally miscalibrated)
    y_true = np.random.binomial(1, 0.5, n_samples)
    y_pred = np.clip(y_true * 0.7 + np.random.randn(n_samples) * 0.2, 0, 1)
    
    # Fit calibrator
    calibrator = HierarchicalCalibrator()
    calibrator.fit(X, y_true, y_pred)
    
    # Apply calibration
    y_calibrated = calibrator.predict_proba(X, y_pred)
    
    # Show report
    report = calibrator.get_calibration_report()
    print(f"\n📋 Calibration Report:")
    print(f"   • Segments calibrated: {report['hierarchy_stats']['n_segments']}")
    print(f"   • Overall ECE: {report['ece_scores']['overall_ece']:.4f}")
    
    print("\n✅ Hierarchical calibration system ready for integration!")