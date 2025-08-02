"""
UFC ELO Rating System
====================

Comprehensive ELO rating system specifically designed for UFC fight prediction.
Includes multi-dimensional ratings, method adjustments, and UFC-specific adaptations.

Features:
- Dynamic K-factors based on fighter experience and rating gaps
- Method of victory bonuses (KO > Decision > Submission adjustments)
- Title fight and main event multipliers
- Activity decay for inactive fighters
- Uncertainty quantification with confidence intervals
- Integration with existing ML pipeline

Usage:
    from ufc_predictor.utils.ufc_elo_system import UFCELOSystem
    
    elo_system = UFCELOSystem()
    elo_system.build_from_fight_history(fights_df)
    rating = elo_system.get_fighter_rating("Jon Jones")
    prediction = elo_system.predict_fight("Jon Jones", "Stipe Miocic")
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import math

# Import configuration
from ufc_predictor.utils.unified_config import config
from ufc_predictor.utils.logging_config import get_logger
from ufc_predictor.utils.common_utilities import DateTimeUtils, ValidationUtils

logger = get_logger(__name__)


@dataclass
class UFCFighterELO:
    """Complete ELO profile for a UFC fighter"""
    name: str
    
    # Core ratings
    overall_rating: float = 1400.0
    striking_rating: float = 1400.0
    grappling_rating: float = 1400.0
    cardio_rating: float = 1400.0
    
    # Uncertainty and confidence
    rating_deviation: float = 350.0  # TrueSkill-inspired uncertainty
    last_updated: Optional[datetime] = None
    
    # Experience tracking
    total_fights: int = 0
    ufc_fights: int = 0
    title_fights: int = 0
    main_events: int = 0
    
    # Performance tracking
    current_streak: int = 0
    win_percentage: float = 0.0
    finish_percentage: float = 0.0
    
    # Activity status
    last_fight_date: Optional[datetime] = None
    active_status: str = "active"  # active, semi_retired, retired
    
    def get_k_factor(self, opponent_rating: float, is_title_fight: bool = False, is_main_event: bool = False) -> float:
        """Calculate dynamic K-factor based on fighter profile and fight context"""
        base_k = 32.0
        
        # Experience adjustment
        if self.ufc_fights <= 5:
            base_k = 45.0  # Rookie fighters
        elif self.ufc_fights >= 20:
            base_k = 24.0  # Veterans
        elif self.title_fights >= 3:
            base_k = 20.0  # Champions
        
        # Rating gap adjustment (potential upset bonus)
        rating_gap = abs(self.overall_rating - opponent_rating)
        if rating_gap > 300:
            base_k *= 1.4  # Big upset potential
        elif rating_gap > 200:
            base_k *= 1.2  # Moderate upset potential
        
        # Context adjustments
        if is_title_fight:
            base_k *= 1.5  # Title fights matter more
        elif is_main_event:
            base_k *= 1.2  # Main events matter more
        
        # Uncertainty adjustment (higher uncertainty = higher K-factor)
        uncertainty_multiplier = min(2.0, self.rating_deviation / 200.0)
        base_k *= uncertainty_multiplier
        
        return min(base_k, 60.0)  # Cap maximum K-factor
    
    def calculate_activity_decay(self, current_date: datetime) -> float:
        """Calculate rating decay due to inactivity"""
        if not self.last_fight_date:
            return 0.0
        
        days_inactive = (current_date - self.last_fight_date).days
        
        if days_inactive < 365:
            return 0.0  # No decay for first year
        
        # 2% decay per year after first year
        years_inactive = (days_inactive - 365) / 365.25
        decay_factor = 0.02 * years_inactive
        
        return min(decay_factor, 0.15)  # Max 15% decay
    
    def get_effective_rating(self, rating_type: str = "overall", current_date: datetime = None) -> float:
        """Get rating with activity decay applied"""
        base_rating = getattr(self, f"{rating_type}_rating", self.overall_rating)
        
        if current_date:
            decay = self.calculate_activity_decay(current_date)
            return base_rating * (1 - decay)
        
        return base_rating


@dataclass
class UFCFightResult:
    """Structured fight result for ELO processing"""
    winner: str
    loser: str
    fight_date: datetime
    method: str = "Decision"
    round_finished: int = 3
    time_finished: str = "5:00"
    is_title_fight: bool = False
    is_main_event: bool = False
    event_name: str = ""
    weight_class: str = ""


class UFCELOSystem:
    """
    Comprehensive ELO rating system for UFC fighters
    
    Implements advanced ELO with UFC-specific adaptations:
    - Multi-dimensional ratings (overall, striking, grappling, cardio)
    - Method of victory adjustments
    - Title fight and context bonuses
    - Activity decay for inactive fighters
    - Uncertainty quantification
    """
    
    def __init__(self, initial_rating: float = 1400.0, use_multi_dimensional: bool = True):
        self.initial_rating = initial_rating
        self.use_multi_dimensional = use_multi_dimensional
        
        # Fighter database
        self.fighters: Dict[str, UFCFighterELO] = {}
        
        # Configuration parameters
        self.config = {
            'base_k_factor': 32.0,
            'rookie_k_factor': 45.0,
            'veteran_k_factor': 24.0,
            'champion_k_factor': 20.0,
            'rating_floor': 800.0,
            'rating_ceiling': 2800.0,
            'initial_uncertainty': 350.0,
            'min_uncertainty': 50.0,
            'max_uncertainty': 500.0,
            'activity_decay_rate': 0.02,
            'method_multipliers': {
                'KO': 1.3,
                'TKO': 1.25,
                'Submission': 1.2,
                'Decision - Unanimous': 1.0,
                'Decision - Majority': 0.95,
                'Decision - Split': 0.9,
                'DQ': 0.8,
                'No Contest': 0.0
            },
            'round_bonuses': {1: 8, 2: 5, 3: 3, 4: 2, 5: 1},
            'context_multipliers': {
                'title_fight': 1.5,
                'main_event': 1.2,
                'tournament_final': 1.3
            }
        }
        
        # Performance tracking
        self.rating_history = {}
        self.prediction_accuracy = []
        
    def get_fighter(self, fighter_name: str) -> UFCFighterELO:
        """Get or create fighter ELO profile"""
        if fighter_name not in self.fighters:
            self.fighters[fighter_name] = UFCFighterELO(
                name=fighter_name,
                overall_rating=self.initial_rating,
                striking_rating=self.initial_rating,
                grappling_rating=self.initial_rating,
                cardio_rating=self.initial_rating
            )
        
        return self.fighters[fighter_name]
    
    def standardize_method(self, method: str) -> str:
        """Standardize method names for consistent processing"""
        if not method or pd.isna(method):
            return "Decision - Unanimous"
        
        method_upper = str(method).upper()
        
        # KO variations
        if any(term in method_upper for term in ['KO', 'KNOCKOUT', 'KNOCKED OUT']):
            return 'KO'
        
        # TKO variations
        if any(term in method_upper for term in ['TKO', 'TECHNICAL KNOCKOUT', 'DOCTOR STOPPAGE', 'CORNER STOPPAGE']):
            return 'TKO'
        
        # Submission variations
        if any(term in method_upper for term in ['SUB', 'SUBMISSION', 'CHOKE', 'ARMBAR', 'TRIANGLE', 'KIMURA']):
            return 'Submission'
        
        # Decision variations
        if 'DECISION' in method_upper:
            if 'UNANIMOUS' in method_upper:
                return 'Decision - Unanimous'
            elif 'MAJORITY' in method_upper:
                return 'Decision - Majority'
            elif 'SPLIT' in method_upper:
                return 'Decision - Split'
            else:
                return 'Decision - Unanimous'  # Default
        
        # Other outcomes
        if any(term in method_upper for term in ['DQ', 'DISQUALIFICATION']):
            return 'DQ'
        
        if any(term in method_upper for term in ['NC', 'NO CONTEST']):
            return 'No Contest'
        
        return 'Decision - Unanimous'  # Default fallback
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula"""
        return 1.0 / (1.0 + 10**((rating_b - rating_a) / 400.0))
    
    def update_fighter_rating(self, fighter: UFCFighterELO, opponent: UFCFighterELO,
                            actual_score: float, expected_score: float,
                            k_factor: float, method_multiplier: float = 1.0) -> float:
        """Update a fighter's rating after a fight"""
        # Basic ELO update
        rating_change = k_factor * method_multiplier * (actual_score - expected_score)
        
        # Apply rating bounds
        new_rating = fighter.overall_rating + rating_change
        new_rating = max(self.config['rating_floor'], 
                        min(self.config['rating_ceiling'], new_rating))
        
        # Update uncertainty (decreases with more fights)
        if fighter.rating_deviation > self.config['min_uncertainty']:
            uncertainty_decrease = min(15.0, fighter.rating_deviation * 0.1)
            fighter.rating_deviation = max(
                self.config['min_uncertainty'],
                fighter.rating_deviation - uncertainty_decrease
            )
        
        old_rating = fighter.overall_rating
        fighter.overall_rating = new_rating
        
        return new_rating - old_rating
    
    def update_dimensional_ratings(self, winner: UFCFighterELO, loser: UFCFighterELO,
                                 method: str, k_factor: float) -> Dict[str, Tuple[float, float]]:
        """Update multi-dimensional ratings based on method of victory"""
        if not self.use_multi_dimensional:
            return {}
        
        method_impacts = {
            'striking': {'KO': 1.5, 'TKO': 1.3, 'Decision - Unanimous': 1.0, 'Decision - Split': 0.9, 'Submission': 0.7},
            'grappling': {'Submission': 1.5, 'Decision - Unanimous': 1.1, 'TKO': 0.8, 'KO': 0.7, 'Decision - Split': 1.0},
            'cardio': {'Decision - Unanimous': 1.3, 'Decision - Split': 1.2, 'Decision - Majority': 1.1, 'KO': 0.8, 'TKO': 0.9, 'Submission': 1.0}
        }
        
        dimensional_updates = {}
        
        for dimension in ['striking', 'grappling', 'cardio']:
            winner_rating = getattr(winner, f"{dimension}_rating")
            loser_rating = getattr(loser, f"{dimension}_rating")
            
            expected_winner = self.calculate_expected_score(winner_rating, loser_rating)
            expected_loser = 1.0 - expected_winner
            
            impact_multiplier = method_impacts[dimension].get(method, 1.0)
            dimensional_k = k_factor * 0.7  # Slightly lower K for dimensional ratings
            
            winner_change = dimensional_k * impact_multiplier * (1.0 - expected_winner)
            loser_change = dimensional_k * impact_multiplier * (0.0 - expected_loser)
            
            # Update ratings
            new_winner_rating = winner_rating + winner_change
            new_loser_rating = loser_rating + loser_change
            
            # Apply bounds
            new_winner_rating = max(self.config['rating_floor'], 
                                  min(self.config['rating_ceiling'], new_winner_rating))
            new_loser_rating = max(self.config['rating_floor'], 
                                 min(self.config['rating_ceiling'], new_loser_rating))
            
            setattr(winner, f"{dimension}_rating", new_winner_rating)
            setattr(loser, f"{dimension}_rating", new_loser_rating)
            
            dimensional_updates[dimension] = (winner_change, loser_change)
        
        return dimensional_updates
    
    def process_fight(self, fight_result: UFCFightResult) -> Dict[str, Any]:
        """Process a single fight and update fighter ratings"""
        winner = self.get_fighter(fight_result.winner)
        loser = self.get_fighter(fight_result.loser)
        
        # Get pre-fight ratings
        pre_winner_rating = winner.overall_rating
        pre_loser_rating = loser.overall_rating
        
        # Calculate expected scores
        expected_winner = self.calculate_expected_score(pre_winner_rating, pre_loser_rating)
        expected_loser = 1.0 - expected_winner
        
        # Determine method multiplier
        standardized_method = self.standardize_method(fight_result.method)
        method_multiplier = self.config['method_multipliers'].get(standardized_method, 1.0)
        
        # Add round bonus
        round_bonus = self.config['round_bonuses'].get(fight_result.round_finished, 0) / 100.0
        method_multiplier += round_bonus
        
        # Apply context multipliers
        context_multiplier = 1.0
        if fight_result.is_title_fight:
            context_multiplier *= self.config['context_multipliers']['title_fight']
        elif fight_result.is_main_event:
            context_multiplier *= self.config['context_multipliers']['main_event']
        
        # Get K-factors
        winner_k = winner.get_k_factor(pre_loser_rating, fight_result.is_title_fight, fight_result.is_main_event)
        loser_k = loser.get_k_factor(pre_winner_rating, fight_result.is_title_fight, fight_result.is_main_event)
        
        # Update overall ratings
        winner_change = self.update_fighter_rating(
            winner, loser, 1.0, expected_winner, 
            winner_k, method_multiplier * context_multiplier
        )
        loser_change = self.update_fighter_rating(
            loser, winner, 0.0, expected_loser, 
            loser_k, method_multiplier * context_multiplier
        )
        
        # Update dimensional ratings
        dimensional_updates = self.update_dimensional_ratings(winner, loser, standardized_method, (winner_k + loser_k) / 2)
        
        # Update fighter statistics
        self._update_fighter_stats(winner, loser, fight_result)
        
        # Store result
        fight_update = {
            'fight_date': fight_result.fight_date,
            'winner': fight_result.winner,
            'loser': fight_result.loser,
            'method': standardized_method,
            'expected_winner_prob': expected_winner,
            'winner_rating_change': winner_change,
            'loser_rating_change': loser_change,
            'winner_new_rating': winner.overall_rating,
            'loser_new_rating': loser.overall_rating,
            'method_multiplier': method_multiplier,
            'context_multiplier': context_multiplier,
            'dimensional_updates': dimensional_updates
        }
        
        return fight_update
    
    def _update_fighter_stats(self, winner: UFCFighterELO, loser: UFCFighterELO, fight_result: UFCFightResult):
        """Update fighter statistics after fight"""
        current_date = fight_result.fight_date
        
        # Update both fighters
        for fighter, is_winner in [(winner, True), (loser, False)]:
            fighter.total_fights += 1
            fighter.ufc_fights += 1
            fighter.last_fight_date = current_date
            
            if fight_result.is_title_fight:
                fighter.title_fights += 1
            if fight_result.is_main_event:
                fighter.main_events += 1
            
            # Update streaks
            if is_winner:
                fighter.current_streak = max(0, fighter.current_streak) + 1
            else:
                fighter.current_streak = min(0, fighter.current_streak) - 1
            
            # Update last updated timestamp
            fighter.last_updated = current_date
    
    def build_from_fight_history(self, fights_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build ELO ratings from historical fight data"""
        logger.info(f"Building ELO ratings from {len(fights_df)} fights")
        
        # Sort by date to process chronologically
        if 'Date' in fights_df.columns:
            fights_df = fights_df.sort_values('Date')
        
        processed_fights = []
        
        for idx, fight in fights_df.iterrows():
            try:
                # Parse fight date
                fight_date = pd.to_datetime(fight.get('Date', datetime.now()))
                
                # Create fight result
                fight_result = UFCFightResult(
                    winner=str(fight.get('Winner', fight.get('Fighter', ''))).strip(),
                    loser=str(fight.get('Loser', fight.get('Opponent', ''))).strip(),
                    fight_date=fight_date,
                    method=str(fight.get('Method', 'Decision')),
                    round_finished=int(fight.get('Round', 3)) if pd.notna(fight.get('Round')) else 3,
                    time_finished=str(fight.get('Time', '5:00')),
                    is_title_fight=self._detect_title_fight(fight),
                    is_main_event=self._detect_main_event(fight),
                    event_name=str(fight.get('Event', '')),
                    weight_class=str(fight.get('Weight_class', ''))
                )
                
                # Skip if missing essential data
                if not fight_result.winner or not fight_result.loser or fight_result.winner == fight_result.loser:
                    continue
                
                # Process fight
                fight_update = self.process_fight(fight_result)
                processed_fights.append(fight_update)
                
                if len(processed_fights) % 500 == 0:
                    logger.info(f"Processed {len(processed_fights)} fights")
                
            except Exception as e:
                logger.warning(f"Error processing fight at index {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_fights)} fights")
        logger.info(f"Created ELO profiles for {len(self.fighters)} fighters")
        
        return processed_fights
    
    def _detect_title_fight(self, fight_row) -> bool:
        """Detect if a fight is a title fight"""
        event_name = str(fight_row.get('Event', '')).lower()
        method = str(fight_row.get('Method', '')).lower()
        
        title_indicators = ['title', 'championship', 'belt', 'champion', 'interim']
        return any(indicator in event_name or indicator in method for indicator in title_indicators)
    
    def _detect_main_event(self, fight_row) -> bool:
        """Detect if a fight is a main event"""
        # This would need additional data, for now use heuristics
        event_name = str(fight_row.get('Event', '')).lower()
        
        main_event_indicators = ['main event', 'headliner', 'main card']
        return any(indicator in event_name for indicator in main_event_indicators)
    
    def predict_fight(self, fighter_a: str, fighter_b: str, fight_date: datetime = None) -> Dict[str, Any]:
        """Predict fight outcome using ELO ratings"""
        fighter_a_obj = self.get_fighter(fighter_a)
        fighter_b_obj = self.get_fighter(fighter_b)
        
        fight_date = fight_date or datetime.now()
        
        # Get effective ratings (with activity decay)
        rating_a = fighter_a_obj.get_effective_rating("overall", fight_date)
        rating_b = fighter_b_obj.get_effective_rating("overall", fight_date)
        
        # Calculate win probabilities
        prob_a_wins = self.calculate_expected_score(rating_a, rating_b)
        prob_b_wins = 1.0 - prob_a_wins
        
        # Multi-dimensional predictions
        dimensional_predictions = {}
        if self.use_multi_dimensional:
            for dimension in ['striking', 'grappling', 'cardio']:
                dim_rating_a = fighter_a_obj.get_effective_rating(dimension, fight_date)
                dim_rating_b = fighter_b_obj.get_effective_rating(dimension, fight_date)
                
                dim_prob_a = self.calculate_expected_score(dim_rating_a, dim_rating_b)
                dimensional_predictions[f"{dimension}_advantage_a"] = dim_prob_a
                dimensional_predictions[f"{dimension}_advantage_b"] = 1.0 - dim_prob_a
        
        # Confidence calculation
        rating_gap = abs(rating_a - rating_b)
        avg_uncertainty = (fighter_a_obj.rating_deviation + fighter_b_obj.rating_deviation) / 2
        confidence = min(0.95, rating_gap / (rating_gap + avg_uncertainty))
        
        prediction = {
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'fighter_a_rating': rating_a,
            'fighter_b_rating': rating_b,
            'fighter_a_win_prob': prob_a_wins,
            'fighter_b_win_prob': prob_b_wins,
            'rating_difference': rating_a - rating_b,
            'confidence': confidence,
            'predicted_winner': fighter_a if prob_a_wins > 0.5 else fighter_b,
            'prediction_strength': abs(prob_a_wins - 0.5) * 2,  # 0 to 1 scale
            'dimensional_predictions': dimensional_predictions,
            'fighter_a_uncertainty': fighter_a_obj.rating_deviation,
            'fighter_b_uncertainty': fighter_b_obj.rating_deviation
        }
        
        return prediction
    
    def get_top_fighters(self, n: int = 20, weight_class: str = None) -> List[Tuple[str, float]]:
        """Get top fighters by ELO rating"""
        fighter_ratings = []
        
        for name, fighter in self.fighters.items():
            if fighter.ufc_fights >= 3:  # Minimum fight threshold
                effective_rating = fighter.get_effective_rating("overall", datetime.now())
                fighter_ratings.append((name, effective_rating))
        
        # Sort by rating descending
        fighter_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return fighter_ratings[:n]
    
    def export_ratings(self, file_path: Path = None) -> pd.DataFrame:
        """Export current ELO ratings to DataFrame"""
        ratings_data = []
        
        for name, fighter in self.fighters.items():
            current_date = datetime.now()
            
            ratings_data.append({
                'fighter_name': name,
                'overall_rating': fighter.get_effective_rating("overall", current_date),
                'striking_rating': fighter.get_effective_rating("striking", current_date),
                'grappling_rating': fighter.get_effective_rating("grappling", current_date),
                'cardio_rating': fighter.get_effective_rating("cardio", current_date),
                'rating_deviation': fighter.rating_deviation,
                'total_fights': fighter.total_fights,
                'ufc_fights': fighter.ufc_fights,
                'title_fights': fighter.title_fights,
                'current_streak': fighter.current_streak,
                'last_fight_date': fighter.last_fight_date,
                'last_updated': fighter.last_updated,
                'active_status': fighter.active_status
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        if file_path:
            ratings_df.to_csv(file_path, index=False)
            logger.info(f"ELO ratings exported to {file_path}")
        
        return ratings_df
    
    def save_system(self, file_path: Path) -> None:
        """Save the complete ELO system"""
        import pickle
        
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"ELO system saved to {file_path}")
    
    @classmethod
    def load_system(cls, file_path: Path) -> 'UFCELOSystem':
        """Load a saved ELO system"""
        import pickle
        
        with open(file_path, 'rb') as f:
            system = pickle.load(f)
        
        logger.info(f"ELO system loaded from {file_path}")
        return system


if __name__ == "__main__":
    # Demonstration of UFC ELO system
    logger.info("🥊 UFC ELO Rating System Demo")
    
    try:
        # Initialize system
        elo_system = UFCELOSystem(use_multi_dimensional=True)
        
        # Create sample fight data
        sample_fights = pd.DataFrame([
            {'Winner': 'Jon Jones', 'Loser': 'Stipe Miocic', 'Date': '2024-03-01', 'Method': 'TKO', 'Round': 1},
            {'Winner': 'Francis Ngannou', 'Loser': 'Ciryl Gane', 'Date': '2024-02-01', 'Method': 'Decision', 'Round': 5},
            {'Winner': 'Tom Aspinall', 'Loser': 'Curtis Blaydes', 'Date': '2024-01-01', 'Method': 'KO', 'Round': 1}
        ])
        
        # Build ratings from sample data
        processed_fights = elo_system.build_from_fight_history(sample_fights)
        
        print(f"\n📈 Processed {len(processed_fights)} fights")
        
        # Show top fighters
        top_fighters = elo_system.get_top_fighters(10)
        print(f"\n🏆 Top 10 Fighters by ELO Rating:")
        for i, (name, rating) in enumerate(top_fighters, 1):
            print(f"{i:2d}. {name:<20} {rating:.0f}")
        
        # Make prediction
        prediction = elo_system.predict_fight("Jon Jones", "Francis Ngannou")
        print(f"\n🎯 Fight Prediction: {prediction['fighter_a']} vs {prediction['fighter_b']}")
        print(f"   {prediction['fighter_a']}: {prediction['fighter_a_win_prob']:.1%} (Rating: {prediction['fighter_a_rating']:.0f})")
        print(f"   {prediction['fighter_b']}: {prediction['fighter_b_win_prob']:.1%} (Rating: {prediction['fighter_b_rating']:.0f})")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        
        print(f"\n✅ UFC ELO System demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise