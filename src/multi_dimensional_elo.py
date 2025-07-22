"""
Multi-Dimensional ELO System for UFC

This module extends the base ELO system with multiple rating dimensions:
- Overall rating (general fighting ability)
- Striking rating (stand-up game)
- Grappling rating (takedowns, ground control, submissions)
- Cardio rating (endurance and late-round performance)

Each dimension is updated based on fight outcomes and methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
from .ufc_elo_system import UFCELOSystem, FighterELO, ELOConfig


@dataclass
class MultiDimFighterELO(FighterELO):
    """Extended fighter ELO with multiple dimensions"""
    striking_rating: float = field(default=1400)
    grappling_rating: float = field(default=1400)
    cardio_rating: float = field(default=1400)
    
    # Dimension-specific uncertainties
    striking_uncertainty: float = field(default=200)
    grappling_uncertainty: float = field(default=200)
    cardio_uncertainty: float = field(default=200)
    
    # Performance tracking
    striking_performances: List[float] = field(default_factory=list)
    grappling_performances: List[float] = field(default_factory=list)
    cardio_performances: List[float] = field(default_factory=list)
    
    # Method-specific statistics
    ko_wins: int = 0
    ko_losses: int = 0
    submission_wins: int = 0
    submission_losses: int = 0
    decision_wins: int = 0
    decision_losses: int = 0


class MultiDimensionalUFCELO(UFCELOSystem):
    """
    Multi-dimensional ELO system that tracks separate ratings for different fighting aspects
    """
    
    def __init__(self, config: ELOConfig = None):
        super().__init__(config)
        self.dimension_weights = {
            'striking': 0.4,
            'grappling': 0.3,
            'cardio': 0.2,
            'overall': 0.1  # Small adjustment factor
        }
    
    def initialize_fighter(self, name: str, weight: float = None, is_female: bool = False) -> MultiDimFighterELO:
        """Initialize a fighter with multi-dimensional ratings"""
        if name in self.fighters:
            return self.fighters[name]
        
        weight_class = self.get_weight_class(weight, is_female) if weight else None
        
        fighter = MultiDimFighterELO(
            name=name,
            current_rating=self.config.initial_rating,
            peak_rating=self.config.initial_rating,
            uncertainty=self.config.initial_uncertainty,
            fights_count=0,
            last_fight_date=None,
            weight_class_ratings={},
            active_weight_class=weight_class.value if weight_class else None,
            striking_rating=self.config.initial_rating,
            grappling_rating=self.config.initial_rating,
            cardio_rating=self.config.initial_rating
        )
        
        if weight_class:
            fighter.weight_class_ratings[weight_class.value] = self.config.initial_rating
        
        self.fighters[name] = fighter
        return fighter
    
    def get_method_dimension_impact(self, method: str) -> Dict[str, float]:
        """
        Determine which dimensions are most impacted by a victory method
        
        Returns:
            Dictionary mapping dimensions to impact multipliers
        """
        method_upper = method.upper()
        
        if 'KO' in method_upper or 'TKO' in method_upper:
            return {
                'striking': 1.5,  # High impact on striking
                'grappling': 0.8,  # Reduced impact on grappling
                'cardio': 1.0,    # Neutral impact on cardio
                'overall': 1.2    # Moderate overall impact
            }
        elif 'SUB' in method_upper:
            return {
                'striking': 0.8,  # Reduced striking impact
                'grappling': 1.6, # Very high grappling impact
                'cardio': 1.1,    # Slight cardio benefit
                'overall': 1.3    # Strong overall impact
            }
        elif 'DEC' in method_upper:
            return {
                'striking': 1.1,  # Slight striking benefit
                'grappling': 1.1, # Slight grappling benefit
                'cardio': 1.4,    # High cardio impact (went the distance)
                'overall': 1.0    # Neutral overall
            }
        else:  # DQ, NC, etc.
            return {
                'striking': 0.9,
                'grappling': 0.9,
                'cardio': 0.9,
                'overall': 0.9
            }
    
    def get_round_dimension_impact(self, round_num: int) -> Dict[str, float]:
        """
        Adjust dimension impact based on which round the fight ended
        """
        if round_num == 1:
            return {
                'striking': 1.2,  # Quick finish likely striking
                'grappling': 1.1, # Could be grappling
                'cardio': 0.8,    # Less cardio relevance
                'overall': 1.0
            }
        elif round_num == 2:
            return {
                'striking': 1.1,
                'grappling': 1.1,
                'cardio': 1.0,
                'overall': 1.0
            }
        elif round_num >= 3:
            return {
                'striking': 1.0,
                'grappling': 1.0,
                'cardio': 1.3,    # High cardio relevance
                'overall': 1.0
            }
        else:
            return {'striking': 1.0, 'grappling': 1.0, 'cardio': 1.0, 'overall': 1.0}
    
    def update_dimensional_rating(self, 
                                 fighter: MultiDimFighterELO,
                                 dimension: str,
                                 expected_score: float,
                                 actual_score: float,
                                 k_factor: float,
                                 impact_multiplier: float = 1.0) -> float:
        """Update a specific dimensional rating"""
        
        # Get current rating for dimension
        if dimension == 'striking':
            current_rating = fighter.striking_rating
            uncertainty = fighter.striking_uncertainty
        elif dimension == 'grappling':
            current_rating = fighter.grappling_rating
            uncertainty = fighter.grappling_uncertainty
        elif dimension == 'cardio':
            current_rating = fighter.cardio_rating
            uncertainty = fighter.cardio_uncertainty
        else:  # overall
            current_rating = fighter.current_rating
            uncertainty = fighter.uncertainty
        
        # Calculate rating change
        base_change = k_factor * impact_multiplier * (actual_score - expected_score)
        
        # Apply uncertainty boost for new fighters
        if fighter.fights_count < self.config.minimum_fights_for_stability:
            uncertainty_boost = 1.0 + (uncertainty / 200.0)
            base_change *= uncertainty_boost
        
        # Calculate new rating
        new_rating = current_rating + base_change
        
        # Apply bounds
        new_rating = max(self.config.minimum_rating, 
                        min(self.config.maximum_rating, new_rating))
        
        # Update the appropriate rating and uncertainty
        if dimension == 'striking':
            fighter.striking_rating = new_rating
            fighter.striking_uncertainty = max(
                self.config.min_uncertainty,
                fighter.striking_uncertainty - self.config.uncertainty_reduction_per_fight
            )
        elif dimension == 'grappling':
            fighter.grappling_rating = new_rating
            fighter.grappling_uncertainty = max(
                self.config.min_uncertainty,
                fighter.grappling_uncertainty - self.config.uncertainty_reduction_per_fight
            )
        elif dimension == 'cardio':
            fighter.cardio_rating = new_rating
            fighter.cardio_uncertainty = max(
                self.config.min_uncertainty,
                fighter.cardio_uncertainty - self.config.uncertainty_reduction_per_fight
            )
        else:  # overall
            fighter.current_rating = new_rating
            fighter.uncertainty = max(
                self.config.min_uncertainty,
                fighter.uncertainty - self.config.uncertainty_reduction_per_fight
            )
        
        return new_rating
    
    def calculate_composite_expected_score(self, 
                                         fighter1: MultiDimFighterELO, 
                                         fighter2: MultiDimFighterELO,
                                         predicted_method: str = None) -> float:
        """
        Calculate expected score using composite of dimensional ratings
        """
        # Get expected scores for each dimension
        striking_expected = self.calculate_expected_score(
            fighter1.striking_rating, fighter2.striking_rating
        )
        grappling_expected = self.calculate_expected_score(
            fighter1.grappling_rating, fighter2.grappling_rating
        )
        cardio_expected = self.calculate_expected_score(
            fighter1.cardio_rating, fighter2.cardio_rating
        )
        overall_expected = self.calculate_expected_score(
            fighter1.current_rating, fighter2.current_rating
        )
        
        # Adjust weights based on predicted method
        weights = self.dimension_weights.copy()
        if predicted_method:
            method_upper = predicted_method.upper()
            if 'KO' in method_upper or 'TKO' in method_upper:
                weights['striking'] *= 1.5
                weights['grappling'] *= 0.7
            elif 'SUB' in method_upper:
                weights['striking'] *= 0.7
                weights['grappling'] *= 1.5
            elif 'DEC' in method_upper:
                weights['cardio'] *= 1.3
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate composite expected score
        composite_expected = (
            weights['striking'] * striking_expected +
            weights['grappling'] * grappling_expected +
            weights['cardio'] * cardio_expected +
            weights['overall'] * overall_expected
        )
        
        return composite_expected
    
    def process_fight(self,
                      fighter1_name: str,
                      fighter2_name: str,
                      winner_name: str,
                      method: str,
                      fight_date: datetime,
                      round_num: int = None,
                      is_title_fight: bool = False,
                      is_main_event: bool = False,
                      fighter1_weight: float = None,
                      fighter2_weight: float = None) -> Dict:
        """Process fight with multi-dimensional updates"""
        
        # Initialize fighters
        fighter1 = self.initialize_fighter(fighter1_name, fighter1_weight)
        fighter2 = self.initialize_fighter(fighter2_name, fighter2_weight)
        
        # Apply activity decay
        self.apply_activity_decay(fighter1, fight_date)
        self.apply_activity_decay(fighter2, fight_date)
        
        # Store pre-fight ratings (all dimensions)
        pre_fight_ratings = {
            'fighter1': {
                'overall': fighter1.current_rating,
                'striking': fighter1.striking_rating,
                'grappling': fighter1.grappling_rating,
                'cardio': fighter1.cardio_rating
            },
            'fighter2': {
                'overall': fighter2.current_rating,
                'striking': fighter2.striking_rating,
                'grappling': fighter2.grappling_rating,
                'cardio': fighter2.cardio_rating
            }
        }
        
        # Calculate composite expected scores
        expected1 = self.calculate_composite_expected_score(fighter1, fighter2, method)
        expected2 = 1.0 - expected1
        
        # Determine actual scores
        if winner_name == fighter1_name:
            actual1, actual2 = 1.0, 0.0
        elif winner_name == fighter2_name:
            actual1, actual2 = 0.0, 1.0
        else:
            actual1, actual2 = 0.5, 0.5
        
        # Get K-factors
        k1 = self.get_k_factor(fighter1, is_title_fight)
        k2 = self.get_k_factor(fighter2, is_title_fight)
        
        # Get dimension impact multipliers
        method_impacts = self.get_method_dimension_impact(method)
        round_impacts = self.get_round_dimension_impact(round_num or 3)
        
        # Combine impacts
        combined_impacts = {}
        for dim in ['striking', 'grappling', 'cardio', 'overall']:
            combined_impacts[dim] = method_impacts[dim] * round_impacts[dim]
        
        # Apply main event bonus
        if is_main_event:
            for dim in combined_impacts:
                combined_impacts[dim] *= self.config.main_event_multiplier
        
        # Update all dimensional ratings
        new_ratings = {'fighter1': {}, 'fighter2': {}}
        
        for dimension in ['overall', 'striking', 'grappling', 'cardio']:
            # Update fighter 1
            new_rating1 = self.update_dimensional_rating(
                fighter1, dimension, expected1, actual1, k1, combined_impacts[dimension]
            )
            new_ratings['fighter1'][dimension] = new_rating1
            
            # Update fighter 2
            new_rating2 = self.update_dimensional_rating(
                fighter2, dimension, expected2, actual2, k2, combined_impacts[dimension]
            )
            new_ratings['fighter2'][dimension] = new_rating2
        
        # Update method-specific statistics
        if winner_name == fighter1_name:
            if 'KO' in method.upper() or 'TKO' in method.upper():
                fighter1.ko_wins += 1
                fighter2.ko_losses += 1
            elif 'SUB' in method.upper():
                fighter1.submission_wins += 1
                fighter2.submission_losses += 1
            elif 'DEC' in method.upper():
                fighter1.decision_wins += 1
                fighter2.decision_losses += 1
        elif winner_name == fighter2_name:
            if 'KO' in method.upper() or 'TKO' in method.upper():
                fighter2.ko_wins += 1
                fighter1.ko_losses += 1
            elif 'SUB' in method.upper():
                fighter2.submission_wins += 1
                fighter1.submission_losses += 1
            elif 'DEC' in method.upper():
                fighter2.decision_wins += 1
                fighter1.decision_losses += 1
        
        # Update fight counts and dates
        fighter1.fights_count += 1
        fighter2.fights_count += 1
        fighter1.last_fight_date = fight_date
        fighter2.last_fight_date = fight_date
        
        # Update peak ratings
        fighter1.peak_rating = max(fighter1.peak_rating, fighter1.current_rating)
        fighter2.peak_rating = max(fighter2.peak_rating, fighter2.current_rating)
        
        # Add to rating history
        fighter1.rating_history.append((fight_date, fighter1.current_rating, fighter2_name))
        fighter2.rating_history.append((fight_date, fighter2.current_rating, fighter1_name))
        
        # Championship updates
        if is_title_fight and winner_name in [fighter1_name, fighter2_name]:
            winner = fighter1 if winner_name == fighter1_name else fighter2
            loser = fighter2 if winner_name == fighter1_name else fighter1
            
            winner.is_champion = True
            winner.title_defenses = winner.title_defenses + 1 if winner.is_champion else 1
            loser.is_champion = False
            loser.title_defenses = 0
        
        # Create comprehensive fight record
        fight_record = {
            'date': fight_date,
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'winner': winner_name,
            'method': method,
            'round': round_num,
            'is_title_fight': is_title_fight,
            'is_main_event': is_main_event,
            'pre_fight_ratings': pre_fight_ratings,
            'post_fight_ratings': new_ratings,
            'expected_scores': {'fighter1': expected1, 'fighter2': expected2},
            'actual_scores': {'fighter1': actual1, 'fighter2': actual2},
            'k_factors': {'fighter1': k1, 'fighter2': k2},
            'dimension_impacts': combined_impacts
        }
        
        self.fight_history.append(fight_record)
        
        return fight_record
    
    def predict_fight_outcome(self, 
                             fighter1_name: str, 
                             fighter2_name: str,
                             include_uncertainty: bool = True,
                             predicted_method: str = None) -> Dict:
        """Enhanced prediction using multi-dimensional ratings"""
        
        if fighter1_name not in self.fighters or fighter2_name not in self.fighters:
            return {"error": "One or both fighters not found in ELO system"}
        
        fighter1 = self.fighters[fighter1_name]
        fighter2 = self.fighters[fighter2_name]
        
        # Calculate composite win probability
        overall_prob = self.calculate_composite_expected_score(
            fighter1, fighter2, predicted_method
        )
        
        # Calculate dimensional advantages
        striking_advantage = fighter1.striking_rating - fighter2.striking_rating
        grappling_advantage = fighter1.grappling_rating - fighter2.grappling_rating
        cardio_advantage = fighter1.cardio_rating - fighter2.cardio_rating
        
        # Predict likely method based on dimensional advantages
        method_predictions = {}
        
        # KO/TKO likelihood (based on striking advantage)
        ko_prob = 0.3 + max(0, striking_advantage / 1000)  # Base 30% chance
        ko_prob = min(0.7, max(0.1, ko_prob))  # Cap between 10% and 70%
        
        # Submission likelihood (based on grappling advantage)
        sub_prob = 0.15 + max(0, grappling_advantage / 1200)  # Base 15% chance
        sub_prob = min(0.4, max(0.05, sub_prob))  # Cap between 5% and 40%
        
        # Decision likelihood (remainder, influenced by cardio)
        decision_prob = 1.0 - ko_prob - sub_prob
        
        # Adjust for cardio advantage
        if cardio_advantage > 50:
            decision_prob *= 1.2
            ko_prob *= 0.9
        elif cardio_advantage < -50:
            decision_prob *= 0.8
            ko_prob *= 1.1
        
        # Normalize
        total_prob = ko_prob + sub_prob + decision_prob
        method_predictions = {
            'KO/TKO': ko_prob / total_prob,
            'Submission': sub_prob / total_prob,
            'Decision': decision_prob / total_prob
        }
        
        result = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'fighter1_win_prob': overall_prob,
            'fighter2_win_prob': 1.0 - overall_prob,
            'ratings': {
                'fighter1': {
                    'overall': fighter1.current_rating,
                    'striking': fighter1.striking_rating,
                    'grappling': fighter1.grappling_rating,
                    'cardio': fighter1.cardio_rating
                },
                'fighter2': {
                    'overall': fighter2.current_rating,
                    'striking': fighter2.striking_rating,
                    'grappling': fighter2.grappling_rating,
                    'cardio': fighter2.cardio_rating
                }
            },
            'dimensional_advantages': {
                'striking': striking_advantage,
                'grappling': grappling_advantage,
                'cardio': cardio_advantage
            },
            'method_predictions': method_predictions,
            'fight_style_analysis': self._analyze_fight_styles(fighter1, fighter2)
        }
        
        if include_uncertainty:
            # Enhanced uncertainty calculation
            combined_uncertainty = np.sqrt(
                fighter1.uncertainty**2 + fighter2.uncertainty**2 +
                fighter1.striking_uncertainty**2 + fighter2.striking_uncertainty**2 +
                fighter1.grappling_uncertainty**2 + fighter2.grappling_uncertainty**2 +
                fighter1.cardio_uncertainty**2 + fighter2.cardio_uncertainty**2
            ) / 4  # Average across dimensions
            
            result.update({
                'prediction_uncertainty': combined_uncertainty,
                'confidence_level': 'high' if combined_uncertainty < 75 else 
                                   'medium' if combined_uncertainty < 150 else 'low'
            })
        
        return result
    
    def _analyze_fight_styles(self, fighter1: MultiDimFighterELO, fighter2: MultiDimFighterELO) -> Dict:
        """Analyze fighting styles and matchup dynamics"""
        
        def get_style_profile(fighter: MultiDimFighterELO) -> str:
            """Determine fighter's primary style"""
            ratings = {
                'striking': fighter.striking_rating,
                'grappling': fighter.grappling_rating,
                'cardio': fighter.cardio_rating
            }
            
            max_rating = max(ratings.values())
            dominant_skill = [k for k, v in ratings.items() if v == max_rating][0]
            
            # Style classifications
            if dominant_skill == 'striking':
                if fighter.ko_wins > fighter.decision_wins:
                    return "Power Striker"
                else:
                    return "Technical Striker"
            elif dominant_skill == 'grappling':
                if fighter.submission_wins > fighter.decision_wins:
                    return "Submission Specialist"
                else:
                    return "Wrestler"
            else:  # cardio dominant
                return "Cardio Machine"
        
        fighter1_style = get_style_profile(fighter1)
        fighter2_style = get_style_profile(fighter2)
        
        # Analyze stylistic matchup
        matchup_analysis = ""
        if "Striker" in fighter1_style and "Wrestler" in fighter2_style:
            matchup_analysis = "Classic striker vs wrestler matchup"
        elif "Submission" in fighter1_style and "Striker" in fighter2_style:
            matchup_analysis = "Grappler looking to take it to the ground vs striker wanting to keep it standing"
        elif "Cardio" in fighter1_style or "Cardio" in fighter2_style:
            matchup_analysis = "Pace and endurance will be key factors"
        else:
            matchup_analysis = "Well-matched fighters with similar styles"
        
        return {
            'fighter1_style': fighter1_style,
            'fighter2_style': fighter2_style,
            'matchup_narrative': matchup_analysis
        }
    
    def get_dimensional_rankings(self, dimension: str, weight_class: str = None, top_n: int = 20) -> List[Dict]:
        """Get rankings for a specific dimension"""
        fighters_list = []
        
        for name, fighter in self.fighters.items():
            if weight_class is None or fighter.active_weight_class == weight_class:
                
                if dimension == 'overall':
                    rating = fighter.current_rating
                    uncertainty = fighter.uncertainty
                elif dimension == 'striking':
                    rating = fighter.striking_rating
                    uncertainty = fighter.striking_uncertainty
                elif dimension == 'grappling':
                    rating = fighter.grappling_rating
                    uncertainty = fighter.grappling_uncertainty
                elif dimension == 'cardio':
                    rating = fighter.cardio_rating
                    uncertainty = fighter.cardio_uncertainty
                else:
                    continue
                
                fighters_list.append({
                    'name': name,
                    'rating': rating,
                    'uncertainty': uncertainty,
                    'fights_count': fighter.fights_count,
                    'weight_class': fighter.active_weight_class,
                    'dimension': dimension
                })
        
        fighters_list.sort(key=lambda x: x['rating'], reverse=True)
        return fighters_list[:top_n]


def main():
    """Example usage of multi-dimensional ELO system"""
    
    # Initialize the system
    multi_elo = MultiDimensionalUFCELO()
    
    # Example fights with different methods
    fights = [
        {
            'fighter1': 'Jon Jones',
            'fighter2': 'Daniel Cormier',
            'winner': 'Jon Jones',
            'method': 'U-DEC',
            'date': datetime(2015, 1, 3),
            'round': 5,
            'is_title_fight': True
        },
        {
            'fighter1': 'Conor McGregor',
            'fighter2': 'Jose Aldo',
            'winner': 'Conor McGregor',
            'method': 'KO',
            'date': datetime(2015, 12, 12),
            'round': 1,
            'is_title_fight': True
        },
        {
            'fighter1': 'Demian Maia',
            'fighter2': 'Ben Askren',
            'winner': 'Demian Maia',
            'method': 'Submission',
            'date': datetime(2019, 10, 26),
            'round': 2
        }
    ]
    
    # Process fights
    for fight in fights:
        result = multi_elo.process_fight(**fight)
        print(f"Processed: {fight['fighter1']} vs {fight['fighter2']}")
        print(f"Method: {fight['method']}")
        print(f"Rating changes (overall): {result['post_fight_ratings']['fighter1']['overall'] - result['pre_fight_ratings']['fighter1']['overall']:.1f}")
        print()
    
    # Make a multi-dimensional prediction
    prediction = multi_elo.predict_fight_outcome('Jon Jones', 'Conor McGregor')
    print("Multi-dimensional prediction:")
    print(f"Overall win probability: {prediction['fighter1_win_prob']:.3f}")
    print(f"Dimensional advantages: {prediction['dimensional_advantages']}")
    print(f"Method predictions: {prediction['method_predictions']}")
    print(f"Style analysis: {prediction['fight_style_analysis']}")
    
    # Get dimensional rankings
    striking_rankings = multi_elo.get_dimensional_rankings('striking', top_n=3)
    print("\nTop strikers:")
    for i, fighter in enumerate(striking_rankings, 1):
        print(f"{i}. {fighter['name']}: {fighter['rating']:.1f}")


if __name__ == "__main__":
    main()