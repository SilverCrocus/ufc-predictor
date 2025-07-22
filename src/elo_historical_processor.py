"""
Historical Data Processor for UFC ELO System

This module processes historical UFC fight data to build and validate the ELO rating system.
It includes data cleaning, fight parsing, and batch processing capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import re
import logging
from pathlib import Path

from .ufc_elo_system import UFCELOSystem, ELOConfig


class UFCHistoricalProcessor:
    """
    Processes historical UFC fight data to build ELO ratings from scratch
    """
    
    def __init__(self, elo_system: UFCELOSystem = None):
        self.elo_system = elo_system or UFCELOSystem()
        self.logger = logging.getLogger(__name__)
        
        # Method mapping for consistency
        self.method_mapping = {
            'KO/TKO': 'TKO',
            'KO/TKOPunch': 'KO',
            'KO/TKOKick': 'KO',
            'KO/TKOElbow': 'KO',
            'KO/TKOKnee': 'KO',
            'Submission': 'Submission',
            'U-DEC': 'U-DEC',
            'M-DEC': 'M-DEC',
            'S-DEC': 'S-DEC',
            'DQ': 'DQ',
            'NC': 'NC'
        }
    
    def clean_fighter_name(self, name: str) -> str:
        """Clean and standardize fighter names"""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        
        # Remove common suffixes and prefixes
        suffixes_to_remove = [' Jr.', ' Sr.', ' III', ' II', ' IV']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        # Handle nickname patterns
        # Remove nicknames in quotes
        name = re.sub(r'\s*"[^"]*"\s*', ' ', name)
        
        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def parse_fight_time(self, time_str: str, round_num: int = None) -> Tuple[int, str]:
        """
        Parse fight time and round information
        
        Returns:
            Tuple of (round_number, time_string)
        """
        if pd.isna(time_str):
            return round_num or 1, "5:00"
        
        time_str = str(time_str).strip()
        
        # Handle different time formats
        if ':' in time_str:
            return round_num or 1, time_str
        elif time_str.replace('.', '').isdigit():
            # Convert decimal minutes to MM:SS
            minutes = float(time_str)
            mins = int(minutes)
            secs = int((minutes - mins) * 60)
            return round_num or 1, f"{mins}:{secs:02d}"
        
        return round_num or 1, "5:00"  # Default
    
    def standardize_method(self, method: str) -> str:
        """Standardize victory method names"""
        if pd.isna(method):
            return "U-DEC"
        
        method = str(method).strip()
        
        # Direct mapping
        if method in self.method_mapping:
            return self.method_mapping[method]
        
        # Pattern matching for variations
        method_upper = method.upper()
        
        if 'KO' in method_upper and 'TKO' not in method_upper:
            return 'KO'
        elif 'TKO' in method_upper:
            return 'TKO'
        elif 'SUB' in method_upper:
            return 'Submission'
        elif 'U-DEC' in method_upper or 'UNANIMOUS' in method_upper:
            return 'U-DEC'
        elif 'M-DEC' in method_upper or 'MAJORITY' in method_upper:
            return 'M-DEC'
        elif 'S-DEC' in method_upper or 'SPLIT' in method_upper:
            return 'S-DEC'
        elif 'DQ' in method_upper or 'DISQUALIF' in method_upper:
            return 'DQ'
        elif 'NC' in method_upper or 'NO CONTEST' in method_upper:
            return 'NC'
        
        # Default to unanimous decision
        return 'U-DEC'
    
    def parse_round_number(self, round_str: str) -> int:
        """Extract round number from string"""
        if pd.isna(round_str):
            return 3  # Default to round 3 for decisions
        
        round_str = str(round_str).strip()
        
        # Extract first number found
        numbers = re.findall(r'\d+', round_str)
        if numbers:
            return min(int(numbers[0]), 5)  # Cap at 5 rounds
        
        return 3  # Default
    
    def identify_title_fights(self, event_name: str, fighter1: str, fighter2: str) -> bool:
        """
        Identify if a fight was for a title based on event name and fighter names
        """
        if pd.isna(event_name):
            return False
        
        event_name = str(event_name).upper()
        
        # Title fight indicators
        title_indicators = [
            'TITLE', 'CHAMPIONSHIP', 'BELT', 'INTERIM',
            'VACANT', 'UNIFICATION', 'CHAMPION'
        ]
        
        return any(indicator in event_name for indicator in title_indicators)
    
    def identify_main_events(self, event_name: str, fighter1: str, fighter2: str) -> bool:
        """
        Identify main events based on event patterns and fighter prominence
        """
        if pd.isna(event_name):
            return False
        
        event_name = str(event_name).upper()
        
        # Main event indicators
        main_event_indicators = [
            'UFC ', 'UFC ON', 'UFC FIGHT NIGHT', 'THE ULTIMATE FIGHTER'
        ]
        
        # Check if it's a numbered UFC event (likely main event)
        if re.match(r'UFC \d+', event_name):
            return True
        
        return any(indicator in event_name for indicator in main_event_indicators)
    
    def process_fights_dataframe(self, fights_df: pd.DataFrame, 
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> List[Dict]:
        """
        Process a DataFrame of fights and extract ELO processing data
        
        Expected columns:
        - Fighter, Opponent, Outcome, Method, Event, Time, Round (optional)
        """
        processed_fights = []
        
        self.logger.info(f"Processing {len(fights_df)} fight records...")
        
        # Filter by date range if provided
        if 'Date' in fights_df.columns:
            fights_df['Date'] = pd.to_datetime(fights_df['Date'], errors='coerce')
            if start_date:
                fights_df = fights_df[fights_df['Date'] >= start_date]
            if end_date:
                fights_df = fights_df[fights_df['Date'] <= end_date]
        
        # Group fights by event to ensure proper processing order
        if 'Event' in fights_df.columns:
            fights_by_event = fights_df.groupby('Event')
        else:
            # Create single group if no event column
            fights_by_event = [('Unknown', fights_df)]
        
        for event_name, event_fights in fights_by_event:
            event_date = None
            
            # Try to extract date from event or use first available date
            if 'Date' in event_fights.columns:
                valid_dates = event_fights['Date'].dropna()
                if not valid_dates.empty:
                    event_date = valid_dates.iloc[0]
            
            if event_date is None:
                event_date = datetime.now()  # Fallback to current date
            
            # Process each fight in the event
            for _, fight_row in event_fights.iterrows():
                try:
                    fighter1 = self.clean_fighter_name(fight_row.get('Fighter', ''))
                    fighter2 = self.clean_fighter_name(fight_row.get('Opponent', ''))
                    outcome = str(fight_row.get('Outcome', 'loss')).lower()
                    
                    if not fighter1 or not fighter2:
                        continue
                    
                    # Determine winner
                    if outcome == 'win':
                        winner = fighter1
                    elif outcome == 'loss':
                        winner = fighter2
                    else:  # draw, nc, etc.
                        winner = None
                    
                    # Get method and round info
                    method = self.standardize_method(fight_row.get('Method', 'U-DEC'))
                    round_num = self.parse_round_number(fight_row.get('Round', '3'))
                    
                    # Fight context
                    is_title_fight = self.identify_title_fights(
                        str(event_name), fighter1, fighter2
                    )
                    is_main_event = self.identify_main_events(
                        str(event_name), fighter1, fighter2
                    ) or is_title_fight  # Title fights are always main events
                    
                    fight_data = {
                        'fighter1': fighter1,
                        'fighter2': fighter2,
                        'winner': winner,
                        'method': method,
                        'date': event_date,
                        'round': round_num,
                        'is_title_fight': is_title_fight,
                        'is_main_event': is_main_event,
                        'event': str(event_name)
                    }
                    
                    processed_fights.append(fight_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing fight: {e}")
                    continue
        
        # Sort fights by date for chronological processing
        processed_fights.sort(key=lambda x: x['date'])
        
        self.logger.info(f"Successfully processed {len(processed_fights)} fights")
        return processed_fights
    
    def process_fighters_dataframe(self, fighters_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Process fighter DataFrame to extract weight and demographic info
        
        Expected columns:
        - Name, Weight, Height, DOB, etc.
        """
        fighter_info = {}
        
        for _, fighter_row in fighters_df.iterrows():
            try:
                name = self.clean_fighter_name(fighter_row.get('Name', ''))
                if not name:
                    continue
                
                weight_str = fighter_row.get('Weight (lbs)', '') or fighter_row.get('Weight', '')
                weight = None
                
                if pd.notna(weight_str) and weight_str != '':
                    try:
                        # Extract numeric weight
                        weight_match = re.search(r'(\d+\.?\d*)', str(weight_str))
                        if weight_match:
                            weight = float(weight_match.group(1))
                    except:
                        pass
                
                # Determine gender (simple heuristic - can be improved)
                is_female = False
                if 'STANCE' in fighter_row:
                    # This is a simple heuristic - in practice, you'd want a better method
                    pass
                
                fighter_info[name] = {
                    'weight': weight,
                    'is_female': is_female
                }
                
            except Exception as e:
                self.logger.warning(f"Error processing fighter info: {e}")
                continue
        
        self.logger.info(f"Processed information for {len(fighter_info)} fighters")
        return fighter_info
    
    def build_elo_from_history(self, 
                              fights_df: pd.DataFrame,
                              fighters_df: pd.DataFrame = None,
                              start_date: datetime = None) -> UFCELOSystem:
        """
        Build ELO ratings from historical fight data
        
        Args:
            fights_df: DataFrame containing fight history
            fighters_df: Optional DataFrame with fighter information
            start_date: Optional start date for processing
            
        Returns:
            UFCELOSystem with processed historical ratings
        """
        self.logger.info("Building ELO system from historical data...")
        
        # Process fighter information if provided
        fighter_info = {}
        if fighters_df is not None:
            fighter_info = self.process_fighters_dataframe(fighters_df)
        
        # Process fights
        processed_fights = self.process_fights_dataframe(
            fights_df, start_date=start_date
        )
        
        # Initialize fighters with weight information
        for fight in processed_fights:
            for fighter_name in [fight['fighter1'], fight['fighter2']]:
                if fighter_name not in self.elo_system.fighters:
                    weight = None
                    is_female = False
                    
                    if fighter_name in fighter_info:
                        weight = fighter_info[fighter_name].get('weight')
                        is_female = fighter_info[fighter_name].get('is_female', False)
                    
                    self.elo_system.initialize_fighter(
                        fighter_name, weight=weight, is_female=is_female
                    )
        
        # Process fights chronologically
        processed_count = 0
        for fight_data in processed_fights:
            try:
                if fight_data['winner'] is not None:  # Skip draws/NCs for now
                    result = self.elo_system.process_fight(
                        fighter1_name=fight_data['fighter1'],
                        fighter2_name=fight_data['fighter2'],
                        winner_name=fight_data['winner'],
                        method=fight_data['method'],
                        fight_date=fight_data['date'],
                        round_num=fight_data['round'],
                        is_title_fight=fight_data['is_title_fight'],
                        is_main_event=fight_data['is_main_event']
                    )
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count} fights...")
                
            except Exception as e:
                self.logger.warning(f"Error processing fight: {e}")
                continue
        
        self.logger.info(f"ELO system built with {processed_count} processed fights")
        self.logger.info(f"Tracking {len(self.elo_system.fighters)} fighters")
        
        return self.elo_system
    
    def validate_elo_system(self, 
                           test_fights_df: pd.DataFrame,
                           prediction_threshold: float = 0.6) -> Dict:
        """
        Validate the ELO system against a test set of fights
        
        Args:
            test_fights_df: DataFrame with test fights
            prediction_threshold: Threshold for considering a prediction confident
            
        Returns:
            Validation metrics
        """
        self.logger.info("Validating ELO system...")
        
        test_fights = self.process_fights_dataframe(test_fights_df)
        
        correct_predictions = 0
        total_predictions = 0
        confident_correct = 0
        confident_total = 0
        
        calibration_data = {'predicted': [], 'actual': []}
        
        for fight in test_fights:
            if fight['winner'] is None:
                continue
            
            try:
                prediction = self.elo_system.predict_fight_outcome(
                    fight['fighter1'], fight['fighter2'], include_uncertainty=True
                )
                
                if 'error' in prediction:
                    continue
                
                # Determine actual outcome
                actual_winner = 1 if fight['winner'] == fight['fighter1'] else 0
                predicted_prob = prediction['fighter1_win_prob']
                
                # Record for calibration
                calibration_data['predicted'].append(predicted_prob)
                calibration_data['actual'].append(actual_winner)
                
                # Check prediction accuracy
                predicted_winner = 1 if predicted_prob > 0.5 else 0
                if predicted_winner == actual_winner:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Check confident predictions
                confidence = max(predicted_prob, 1 - predicted_prob)
                if confidence >= prediction_threshold:
                    confident_total += 1
                    if predicted_winner == actual_winner:
                        confident_correct += 1
                
            except Exception as e:
                self.logger.warning(f"Error in validation: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        confident_accuracy = confident_correct / confident_total if confident_total > 0 else 0
        
        # Calculate calibration (Brier score)
        if calibration_data['predicted']:
            predicted = np.array(calibration_data['predicted'])
            actual = np.array(calibration_data['actual'])
            brier_score = np.mean((predicted - actual) ** 2)
        else:
            brier_score = 1.0
        
        validation_results = {
            'total_fights_tested': total_predictions,
            'overall_accuracy': accuracy,
            'confident_predictions': confident_total,
            'confident_accuracy': confident_accuracy,
            'brier_score': brier_score,
            'calibration_quality': 'good' if brier_score < 0.2 else 'fair' if brier_score < 0.3 else 'poor'
        }
        
        self.logger.info(f"Validation complete: {accuracy:.3f} accuracy on {total_predictions} fights")
        
        return validation_results


def main():
    """Example usage of the historical processor"""
    
    # This would typically load from your actual data files
    # For demonstration, create sample data
    
    sample_fights = pd.DataFrame({
        'Fighter': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic'],
        'Opponent': ['Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'],
        'Outcome': ['win', 'win', 'loss'],
        'Method': ['U-DEC', 'KO', 'KO'],
        'Event': ['UFC 182', 'UFC 226', 'UFC 260'],
        'Round': [5, 1, 2],
        'Date': ['2015-01-03', '2018-07-07', '2021-03-27']
    })
    
    sample_fighters = pd.DataFrame({
        'Name': ['Jon Jones', 'Daniel Cormier', 'Stipe Miocic', 'Francis Ngannou'],
        'Weight (lbs)': [205, 205, 240, 250]
    })
    
    # Initialize processor
    processor = UFCHistoricalProcessor()
    
    # Build ELO system
    elo_system = processor.build_elo_from_history(
        sample_fights, sample_fighters,
        start_date=datetime(2010, 1, 1)
    )
    
    # Get rankings
    rankings = elo_system.get_rankings(top_n=10)
    print("Top fighters by ELO rating:")
    for i, fighter in enumerate(rankings, 1):
        print(f"{i}. {fighter['name']}: {fighter['rating']:.1f} (±{fighter['uncertainty']:.0f})")
    
    # Make a prediction
    prediction = elo_system.predict_fight_outcome('Jon Jones', 'Stipe Miocic')
    print(f"\nPrediction example: {prediction}")


if __name__ == "__main__":
    main()