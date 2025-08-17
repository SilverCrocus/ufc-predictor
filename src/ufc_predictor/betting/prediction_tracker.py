#!/usr/bin/env python3
"""
UFC Model Prediction Tracker
Tracks ALL predictions (not just bets) to measure true model accuracy
Includes both raw and calibrated predictions for comparison
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import json

class PredictionTracker:
    """
    Comprehensive tracking system for model predictions.
    Tracks both raw and calibrated predictions to measure accuracy.
    """
    
    def __init__(self, project_path: str = '/Users/diyagamah/Documents/ufc-predictor'):
        self.project_path = Path(project_path)
        self.predictions_file = self.project_path / 'model_predictions_tracker.csv'
        self.summary_file = self.project_path / 'model_accuracy_summary.json'
        
        # Load existing predictions or create new
        self._load_or_create_tracker()
        
    def _load_or_create_tracker(self):
        """Load existing predictions or create new tracking file."""
        if self.predictions_file.exists():
            self.predictions_df = pd.read_csv(self.predictions_file)
            print(f"📂 Loaded {len(self.predictions_df)} existing predictions")
        else:
            self.predictions_df = pd.DataFrame(columns=[
                'prediction_id',
                'date_predicted',
                'event',
                'fight_date',
                'fighter1',
                'fighter2',
                'model_prob_raw',           # Raw model output
                'model_prob_calibrated',    # After temperature scaling
                'predicted_winner',          # Who model picks
                'confidence_level',          # HIGH/MEDIUM/LOW
                'method_prediction',         # Decision/KO/Sub
                'odds_fighter1',            # Market odds
                'odds_fighter2',
                'market_favorite',          # Who market favors
                'edge_raw',                 # Edge with raw probability
                'edge_calibrated',          # Edge with calibrated probability
                'bet_placed',               # Whether we actually bet
                'bet_amount',               # How much we bet (if any)
                'actual_winner',            # Filled after fight
                'actual_method',            # Actual finish method
                'prediction_correct',       # Was prediction right?
                'profit_loss',              # If we bet, how much we won/lost
                'notes'
            ])
            print("🆕 Created new prediction tracker")
    
    def track_prediction(self, 
                        fighter1: str,
                        fighter2: str,
                        model_prob_raw: float,
                        event: str = None,
                        odds: Dict = None,
                        bet_info: Dict = None,
                        method: str = None) -> str:
        """
        Track a single fight prediction.
        
        Args:
            fighter1: First fighter name
            fighter2: Second fighter name  
            model_prob_raw: Raw model probability for fighter1
            event: UFC event name
            odds: Dict with 'fighter1' and 'fighter2' decimal odds
            bet_info: Dict with 'placed', 'amount' if bet was made
            method: Predicted finish method
            
        Returns:
            prediction_id for future reference
        """
        # Generate unique ID
        timestamp = datetime.now()
        prediction_id = f"PRED_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(f'{fighter1}{fighter2}')%10000:04d}"
        
        # Calculate calibrated probability (mild T=1.1)
        model_prob_calibrated = self._apply_calibration(model_prob_raw)
        
        # Determine predicted winner
        predicted_winner = fighter1 if model_prob_raw > 0.5 else fighter2
        
        # Confidence level
        confidence = max(model_prob_raw, 1 - model_prob_raw)
        if confidence > 0.70:
            confidence_level = "HIGH"
        elif confidence > 0.60:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Calculate edges if odds provided
        edge_raw = None
        edge_calibrated = None
        market_favorite = None
        
        if odds:
            odds_f1 = odds.get('fighter1', 0)
            odds_f2 = odds.get('fighter2', 0)
            
            if odds_f1 > 0:
                edge_raw = (model_prob_raw * odds_f1) - 1
                edge_calibrated = (model_prob_calibrated * odds_f1) - 1
                
            # Determine market favorite
            if odds_f1 < odds_f2:
                market_favorite = fighter1
            elif odds_f2 < odds_f1:
                market_favorite = fighter2
            else:
                market_favorite = "EVEN"
        
        # Create prediction record
        prediction_record = {
            'prediction_id': prediction_id,
            'date_predicted': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'event': event or "Unknown Event",
            'fight_date': timestamp.strftime('%Y-%m-%d'),  # Update when known
            'fighter1': fighter1,
            'fighter2': fighter2,
            'model_prob_raw': model_prob_raw,
            'model_prob_calibrated': model_prob_calibrated,
            'predicted_winner': predicted_winner,
            'confidence_level': confidence_level,
            'method_prediction': method or "Decision",
            'odds_fighter1': odds.get('fighter1') if odds else None,
            'odds_fighter2': odds.get('fighter2') if odds else None,
            'market_favorite': market_favorite,
            'edge_raw': edge_raw,
            'edge_calibrated': edge_calibrated,
            'bet_placed': bet_info.get('placed', False) if bet_info else False,
            'bet_amount': bet_info.get('amount', 0) if bet_info else 0,
            'actual_winner': None,  # Fill after fight
            'actual_method': None,
            'prediction_correct': None,
            'profit_loss': None,
            'notes': f"Calibration adjustment: {(model_prob_calibrated - model_prob_raw)*100:+.1f}%"
        }
        
        # Add to dataframe
        self.predictions_df = pd.concat([
            self.predictions_df,
            pd.DataFrame([prediction_record])
        ], ignore_index=True)
        
        # Save immediately
        self.save()
        
        print(f"✅ Tracked: {fighter1} vs {fighter2}")
        print(f"   Raw: {model_prob_raw:.1%} → Calibrated: {model_prob_calibrated:.1%}")
        print(f"   Prediction: {predicted_winner} ({confidence_level} confidence)")
        
        return prediction_id
    
    def _apply_calibration(self, raw_prob: float, temperature: float = 1.1) -> float:
        """Apply conservative temperature scaling."""
        if raw_prob <= 0.01 or raw_prob >= 0.99:
            return raw_prob
            
        logit = np.log(raw_prob / (1 - raw_prob))
        calibrated_logit = logit / temperature
        return 1 / (1 + np.exp(-calibrated_logit))
    
    def update_result(self, 
                     prediction_id: str = None,
                     fighter1: str = None,
                     fighter2: str = None,
                     actual_winner: str = None,
                     actual_method: str = None) -> bool:
        """
        Update prediction with actual fight result.
        Can identify by prediction_id OR fighter names.
        """
        # Find the prediction
        if prediction_id:
            mask = self.predictions_df['prediction_id'] == prediction_id
        elif fighter1 and fighter2:
            mask = ((self.predictions_df['fighter1'] == fighter1) & 
                   (self.predictions_df['fighter2'] == fighter2)) | \
                  ((self.predictions_df['fighter1'] == fighter2) & 
                   (self.predictions_df['fighter2'] == fighter1))
            # Get most recent if multiple
            if mask.sum() > 1:
                mask = mask & (self.predictions_df['actual_winner'].isna())
        else:
            print("❌ Need prediction_id or fighter names")
            return False
        
        if mask.sum() == 0:
            print("❌ Prediction not found")
            return False
        
        # Update result
        idx = self.predictions_df[mask].index[-1]  # Most recent
        self.predictions_df.loc[idx, 'actual_winner'] = actual_winner
        self.predictions_df.loc[idx, 'actual_method'] = actual_method or "Unknown"
        
        # Check if prediction was correct
        predicted = self.predictions_df.loc[idx, 'predicted_winner']
        self.predictions_df.loc[idx, 'prediction_correct'] = (predicted == actual_winner)
        
        # Calculate profit/loss if bet was placed
        if self.predictions_df.loc[idx, 'bet_placed']:
            bet_amount = self.predictions_df.loc[idx, 'bet_amount']
            
            if predicted == actual_winner:
                # Won the bet
                if actual_winner == self.predictions_df.loc[idx, 'fighter1']:
                    odds = self.predictions_df.loc[idx, 'odds_fighter1']
                else:
                    odds = self.predictions_df.loc[idx, 'odds_fighter2']
                    
                profit = bet_amount * (odds - 1) if odds else 0
            else:
                # Lost the bet
                profit = -bet_amount
                
            self.predictions_df.loc[idx, 'profit_loss'] = profit
        
        # Save
        self.save()
        
        correct = self.predictions_df.loc[idx, 'prediction_correct']
        print(f"✅ Updated: {predicted} vs {actual_winner} - {'CORRECT' if correct else 'WRONG'}")
        
        return True
    
    def get_accuracy_stats(self, last_n: Optional[int] = None) -> Dict:
        """
        Calculate comprehensive accuracy statistics.
        
        Args:
            last_n: Only consider last N predictions (None = all)
        """
        # Filter to completed predictions
        completed = self.predictions_df[self.predictions_df['actual_winner'].notna()]
        
        if last_n:
            completed = completed.tail(last_n)
        
        if len(completed) == 0:
            return {"error": "No completed predictions to analyze"}
        
        stats = {
            'total_predictions': len(completed),
            'correct_predictions': completed['prediction_correct'].sum(),
            'accuracy_rate': completed['prediction_correct'].mean(),
            
            # Compare raw vs calibrated
            'avg_raw_confidence': completed['model_prob_raw'].apply(lambda x: max(x, 1-x)).mean(),
            'avg_calibrated_confidence': completed['model_prob_calibrated'].apply(lambda x: max(x, 1-x)).mean(),
            
            # By confidence level
            'high_confidence_accuracy': completed[completed['confidence_level'] == 'HIGH']['prediction_correct'].mean(),
            'medium_confidence_accuracy': completed[completed['confidence_level'] == 'MEDIUM']['prediction_correct'].mean(),
            'low_confidence_accuracy': completed[completed['confidence_level'] == 'LOW']['prediction_correct'].mean(),
            
            # Betting performance
            'bets_placed': completed['bet_placed'].sum(),
            'betting_roi': completed['profit_loss'].sum() / completed['bet_amount'].sum() if completed['bet_amount'].sum() > 0 else 0,
            
            # Favorites vs underdogs
            'favorite_predictions': (completed['predicted_winner'] == completed['market_favorite']).mean(),
            'underdog_success_rate': completed[completed['predicted_winner'] != completed['market_favorite']]['prediction_correct'].mean()
        }
        
        # Calibration analysis
        if len(completed) >= 10:
            # Brier score
            y_true = completed['prediction_correct'].astype(int)
            y_prob_raw = completed['model_prob_raw']
            y_prob_cal = completed['model_prob_calibrated']
            
            brier_raw = np.mean((y_prob_raw - y_true) ** 2)
            brier_cal = np.mean((y_prob_cal - y_true) ** 2)
            
            stats['brier_score_raw'] = brier_raw
            stats['brier_score_calibrated'] = brier_cal
            stats['calibration_improvement'] = (brier_raw - brier_cal) / brier_raw * 100
        
        return stats
    
    def save(self):
        """Save predictions to CSV."""
        self.predictions_df.to_csv(self.predictions_file, index=False)
        
    def generate_report(self):
        """Generate comprehensive accuracy report."""
        print("\n" + "="*70)
        print("📊 MODEL ACCURACY REPORT")
        print("="*70)
        
        stats = self.get_accuracy_stats()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"\n📈 Overall Performance:")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Correct: {stats['correct_predictions']}")
        print(f"   Accuracy: {stats['accuracy_rate']:.1%}")
        
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Avg Raw Confidence: {stats['avg_raw_confidence']:.1%}")
        print(f"   Avg Calibrated: {stats['avg_calibrated_confidence']:.1%}")
        
        print(f"\n📊 By Confidence Level:")
        print(f"   HIGH (>70%): {stats.get('high_confidence_accuracy', 0):.1%} accurate")
        print(f"   MEDIUM (60-70%): {stats.get('medium_confidence_accuracy', 0):.1%} accurate")
        print(f"   LOW (<60%): {stats.get('low_confidence_accuracy', 0):.1%} accurate")
        
        if stats['bets_placed'] > 0:
            print(f"\n💰 Betting Performance:")
            print(f"   Bets Placed: {stats['bets_placed']}")
            print(f"   ROI: {stats['betting_roi']:.1%}")
        
        if 'brier_score_raw' in stats:
            print(f"\n🌡️ Calibration Impact:")
            print(f"   Brier Score (Raw): {stats['brier_score_raw']:.4f}")
            print(f"   Brier Score (Calibrated): {stats['brier_score_calibrated']:.4f}")
            print(f"   Improvement: {stats['calibration_improvement']:.1f}%")
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to {self.summary_file}")


# Convenience functions for notebook use
def quick_track_prediction(fighter1: str, fighter2: str, prob: float, **kwargs) -> str:
    """Quick function to track a prediction from notebook."""
    tracker = PredictionTracker()
    return tracker.track_prediction(fighter1, fighter2, prob, **kwargs)

def update_fight_result(fighter1: str, fighter2: str, winner: str, method: str = None):
    """Quick function to update fight result."""
    tracker = PredictionTracker()
    return tracker.update_result(
        fighter1=fighter1, 
        fighter2=fighter2, 
        actual_winner=winner,
        actual_method=method
    )

def show_accuracy_report():
    """Display current accuracy statistics."""
    tracker = PredictionTracker()
    tracker.generate_report()


if __name__ == "__main__":
    # Demo the tracking system
    print("UFC Prediction Tracking System Demo")
    print("="*50)
    
    tracker = PredictionTracker()
    
    # Example predictions
    demo_fights = [
        ("Max Holloway", "Yair Rodriguez", 0.65, {"fighter1": 1.60, "fighter2": 2.40}),
        ("Amanda Nunes", "Valentina Shevchenko", 0.55, {"fighter1": 1.80, "fighter2": 2.10}),
        ("Dustin Poirier", "Justin Gaethje", 0.48, {"fighter1": 2.20, "fighter2": 1.70}),
    ]
    
    print("\nTracking example predictions:")
    for f1, f2, prob, odds in demo_fights:
        tracker.track_prediction(f1, f2, prob, odds=odds)
    
    print("\nCurrent tracking file:", tracker.predictions_file)
    print("Track all your predictions to measure true model accuracy!")