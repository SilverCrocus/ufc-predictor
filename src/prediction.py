import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Any, Tuple
from IPython.display import display, HTML


class UFCPredictor:
    """Class for making UFC fight predictions using trained models."""
    
    def __init__(self, model_path: str = None, columns_path: str = None, fighters_data_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            columns_path: Path to the model columns JSON file
            fighters_data_path: Path to the fighters data CSV file
        """
        self.model = None
        self.model_columns = None
        self.fighters_data = None
        
        if model_path:
            self.load_model(model_path)
        if columns_path:
            self.load_columns(columns_path)
        if fighters_data_path:
            self.load_fighters_data(fighters_data_path)
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from '{model_path}'")
    
    def load_columns(self, columns_path: str):
        """Load model feature columns from JSON file."""
        with open(columns_path, 'r') as f:
            self.model_columns = json.load(f)
        print(f"Model columns loaded from '{columns_path}'")
    
    def load_fighters_data(self, fighters_data_path: str):
        """Load fighters data from CSV file."""
        self.fighters_data = pd.read_csv(fighters_data_path)
        print(f"Fighters data loaded from '{fighters_data_path}'")
    
    def _check_prerequisites(self):
        """Check if all required components are loaded."""
        if self.model is None:
            raise ValueError("Model not loaded. Use load_model() first.")
        if self.model_columns is None:
            raise ValueError("Model columns not loaded. Use load_columns() first.")
        if self.fighters_data is None:
            raise ValueError("Fighters data not loaded. Use load_fighters_data() first.")
    
    def _get_fighter_stats(self, fighter_name: str) -> pd.Series:
        """Get statistics for a specific fighter."""
        fighter_stats = self.fighters_data[self.fighters_data['Name'] == fighter_name]
        if fighter_stats.empty:
            raise ValueError(f"Fighter '{fighter_name}' not found in dataset.")
        return fighter_stats.iloc[0]
    
    def _create_differential_features(self, blue_stats: pd.Series, red_stats: pd.Series) -> Dict[str, float]:
        """Create differential features between blue and red corner fighters."""
        diff_features = {}
        
        for blue_col in blue_stats.index:
            if blue_col.startswith('blue_') and 'url' not in blue_col and 'Name' not in blue_col:
                red_col = blue_col.replace('blue_', 'red_')
                base_name = blue_col.replace('blue_', '')
                
                if red_col in red_stats.index:
                    diff_col_name = base_name.lower().replace(' ', '_').replace('.', '') + '_diff'
                    diff_features[diff_col_name] = blue_stats[blue_col] - red_stats[red_col]
        
        return diff_features
    
    def _get_prediction_from_perspective(self, fighter1_name: str, fighter2_name: str) -> float:
        """
        Get prediction probability from a single perspective (fighter1 = blue, fighter2 = red).
        
        Returns:
            Probability of fighter1 winning (blue corner win probability)
        """
        # Get fighter statistics
        fighter1_stats = self._get_fighter_stats(fighter1_name)
        fighter2_stats = self._get_fighter_stats(fighter2_name)
        
        # Create blue and red corner statistics
        blue_stats = fighter1_stats.add_prefix('blue_')
        red_stats = fighter2_stats.add_prefix('red_')
        
        # Create differential features
        diff_features = self._create_differential_features(blue_stats, red_stats)
        
        # Combine all features
        single_fight_data = {**blue_stats, **red_stats, **diff_features}
        
        # Create prediction DataFrame with correct column order
        prediction_df = pd.DataFrame([single_fight_data]).reindex(columns=self.model_columns, fill_value=0)
        
        # Get prediction probability
        return self.model.predict_proba(prediction_df)[0][1]  # Probability of blue corner winning
    
    def predict_fight_basic(self, fighter1_name: str, fighter2_name: str) -> Dict[str, Any]:
        """
        Make a basic fight prediction (fighter1 = blue corner, fighter2 = red corner).
        
        Args:
            fighter1_name: Name of the blue corner fighter
            fighter2_name: Name of the red corner fighter
            
        Returns:
            Dictionary containing prediction results
        """
        self._check_prerequisites()
        
        try:
            win_probability = self._get_prediction_from_perspective(fighter1_name, fighter2_name)
            
            result = {
                "blue_corner": fighter1_name,
                "red_corner": fighter2_name,
                "blue_win_probability": f"{win_probability*100:.2f}%",
                "red_win_probability": f"{(1-win_probability)*100:.2f}%",
                "predicted_winner": fighter1_name if win_probability > 0.5 else fighter2_name,
                "confidence": f"{max(win_probability, 1-win_probability)*100:.2f}%"
            }
            
            return result
            
        except ValueError as e:
            return {"error": str(e)}
    
    def predict_fight_symmetrical(self, fighter_a_name: str, fighter_b_name: str) -> Dict[str, Any]:
        """
        Make a symmetrical fight prediction by averaging both corner perspectives.
        
        Args:
            fighter_a_name: Name of first fighter
            fighter_b_name: Name of second fighter
            
        Returns:
            Dictionary containing symmetrical prediction results
        """
        self._check_prerequisites()
        
        try:
            # Prediction 1: A is blue corner, B is red corner
            prob_a_wins_as_blue = self._get_prediction_from_perspective(fighter_a_name, fighter_b_name)
            
            # Prediction 2: B is blue corner, A is red corner
            prob_b_wins_as_blue = self._get_prediction_from_perspective(fighter_b_name, fighter_a_name)
            
            # From the second perspective, probability of A winning is 1 - B's win probability
            prob_a_wins_as_red = 1 - prob_b_wins_as_blue
            
            # Average the probabilities
            final_prob_a_wins = (prob_a_wins_as_blue + prob_a_wins_as_red) / 2
            final_prob_b_wins = 1 - final_prob_a_wins
            
            result = {
                "fighter_a": fighter_a_name,
                "fighter_b": fighter_b_name,
                f"{fighter_a_name}_win_probability": f"{final_prob_a_wins*100:.2f}%",
                f"{fighter_b_name}_win_probability": f"{final_prob_b_wins*100:.2f}%",
                "predicted_winner": fighter_a_name if final_prob_a_wins > final_prob_b_wins else fighter_b_name,
                "confidence": f"{max(final_prob_a_wins, final_prob_b_wins)*100:.2f}%"
            }
            
            return result
            
        except ValueError as e:
            return {"error": str(e)}
    
    def predict_fight_card(self, fight_card: List[Tuple[str, str]], symmetrical: bool = True) -> List[Dict[str, Any]]:
        """
        Predict results for an entire fight card.
        
        Args:
            fight_card: List of tuples containing (fighter1, fighter2) matchups
            symmetrical: Whether to use symmetrical predictions
            
        Returns:
            List of prediction dictionaries
        """
        self._check_prerequisites()
        
        all_predictions = []
        print("--- Predicting Full Fight Card ---")
        
        for fighter_a, fighter_b in fight_card:
            print(f"Predicting: {fighter_a} vs. {fighter_b}...")
            
            if symmetrical:
                prediction = self.predict_fight_symmetrical(fighter_a, fighter_b)
            else:
                prediction = self.predict_fight_basic(fighter_a, fighter_b)
            
            all_predictions.append(prediction)
        
        print("\n--- Predictions Complete ---")
        return all_predictions
    
    def display_fight_card_results(self, predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Display fight card predictions in a formatted table.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with formatted results
        """
        results_list = []
        
        for p in predictions:
            if 'error' in p:
                results_list.append([p['error'], "", ""])
            else:
                # Handle both symmetrical and basic prediction formats
                if 'fighter_a' in p:  # Symmetrical format
                    fighter_a, fighter_b = p['fighter_a'], p['fighter_b']
                    winner = p['predicted_winner']
                    confidence = p['confidence']
                    matchup = f"{fighter_a} vs. {fighter_b}"
                else:  # Basic format
                    fighter_a, fighter_b = p['blue_corner'], p['red_corner']
                    winner = p['predicted_winner']
                    confidence = p['confidence']
                    matchup = f"{fighter_a} vs. {fighter_b}"
                
                results_list.append([matchup, f"**{winner}**", f"({confidence} confidence)"])
        
        results_df = pd.DataFrame(results_list, columns=['Matchup', 'Predicted Winner', 'Confidence'])
        
        # Display as HTML for better formatting
        display(HTML(results_df.to_html(index=False, justify='left', escape=False)))
        
        return results_df
    
    def get_available_fighters(self, search_term: str = None) -> List[str]:
        """
        Get list of available fighters, optionally filtered by search term.
        
        Args:
            search_term: Optional string to filter fighter names
            
        Returns:
            List of fighter names
        """
        if self.fighters_data is None:
            raise ValueError("Fighters data not loaded. Use load_fighters_data() first.")
        
        fighters = self.fighters_data['Name'].tolist()
        
        if search_term:
            fighters = [f for f in fighters if search_term.lower() in f.lower()]
        
        return sorted(fighters)


def create_predictor(model_path: str, columns_path: str, fighters_data_path: str) -> UFCPredictor:
    """
    Convenience function to create a fully loaded UFC predictor.
    
    Args:
        model_path: Path to the trained model file
        columns_path: Path to the model columns JSON file
        fighters_data_path: Path to the fighters data CSV file
        
    Returns:
        Fully loaded UFCPredictor instance
    """
    predictor = UFCPredictor()
    predictor.load_model(model_path)
    predictor.load_columns(columns_path)
    predictor.load_fighters_data(fighters_data_path)
    
    return predictor

def predict_fight_symmetrical(fighter1_name: str, fighter2_name: str, fighters_df: pd.DataFrame, 
                            winner_cols: List[str], method_cols: List[str], 
                            winner_model, method_model) -> Dict[str, Any]:
    """
    Standalone function for symmetrical fight prediction with method prediction.
    
    Args:
        fighter1_name: Name of first fighter
        fighter2_name: Name of second fighter  
        fighters_df: DataFrame containing fighter data
        winner_cols: List of winner model feature columns
        method_cols: List of method model feature columns
        winner_model: Trained winner prediction model
        method_model: Trained method prediction model
        
    Returns:
        Dictionary containing prediction results with win probabilities and method prediction
    """
    
    def normalize_name(name):
        return name.strip().lower()
    
    def get_fighter_stats(fighter_name: str) -> pd.Series:
        """Get statistics for a specific fighter."""
        fighter_name_normalized = normalize_name(fighter_name)
        fighters_df['Name_normalized'] = fighters_df['Name'].str.strip().str.lower()
        
        fighter_stats = fighters_df[fighters_df['Name_normalized'] == fighter_name_normalized]
        if fighter_stats.empty:
            raise ValueError(f"Fighter '{fighter_name}' not found in dataset.")
        return fighter_stats.iloc[0]
    
    def create_differential_features(blue_stats: pd.Series, red_stats: pd.Series) -> Dict[str, float]:
        """Create differential features between blue and red corner fighters."""
        diff_features = {}
        
        for blue_col in blue_stats.index:
            if blue_col.startswith('blue_') and 'url' not in blue_col and 'Name' not in blue_col:
                red_col = blue_col.replace('blue_', 'red_')
                base_name = blue_col.replace('blue_', '')
                
                if red_col in red_stats.index:
                    diff_col_name = base_name.lower().replace(' ', '_').replace('.', '') + '_diff'
                    diff_features[diff_col_name] = blue_stats[blue_col] - red_stats[red_col]
        
        return diff_features
    
    def get_prediction_from_perspective(fighter1: str, fighter2: str, model, columns) -> float:
        """Get prediction probability from single perspective (fighter1 = blue, fighter2 = red)."""
        # Get fighter statistics
        fighter1_stats = get_fighter_stats(fighter1)
        fighter2_stats = get_fighter_stats(fighter2)
        
        # Create blue and red corner statistics
        blue_stats = fighter1_stats.add_prefix('blue_')
        red_stats = fighter2_stats.add_prefix('red_')
        
        # Create differential features
        diff_features = create_differential_features(blue_stats, red_stats)
        
        # Combine all features
        single_fight_data = {**blue_stats, **red_stats, **diff_features}
        
        # Create prediction DataFrame with correct column order
        prediction_df = pd.DataFrame([single_fight_data]).reindex(columns=columns, fill_value=0)
        
        # Get prediction probability/probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(prediction_df)[0]
            if len(probs) == 2:  # Binary classification (winner model)
                return probs[1]  # Probability of blue corner winning
            else:  # Multi-class classification (method model)
                return probs
        else:
            return model.predict(prediction_df)[0]
    
    try:
        # Winner predictions from both perspectives
        # Prediction 1: fighter1 is blue corner, fighter2 is red corner
        prob_fighter1_wins_as_blue = get_prediction_from_perspective(fighter1_name, fighter2_name, winner_model, winner_cols)
        
        # Prediction 2: fighter2 is blue corner, fighter1 is red corner  
        prob_fighter2_wins_as_blue = get_prediction_from_perspective(fighter2_name, fighter1_name, winner_model, winner_cols)
        
        # From the second perspective, probability of fighter1 winning is 1 - fighter2's win probability
        prob_fighter1_wins_as_red = 1 - prob_fighter2_wins_as_blue
        
        # Average the probabilities for symmetrical prediction
        final_prob_fighter1_wins = (prob_fighter1_wins_as_blue + prob_fighter1_wins_as_red) / 2
        final_prob_fighter2_wins = 1 - final_prob_fighter1_wins
        
        # Method predictions (average from both perspectives)
        method_probs1 = get_prediction_from_perspective(fighter1_name, fighter2_name, method_model, method_cols)
        method_probs2 = get_prediction_from_perspective(fighter2_name, fighter1_name, method_model, method_cols)
        
        # Average the method probabilities
        avg_method_probs = (method_probs1 + method_probs2) / 2
        method_classes = method_model.classes_
        predicted_method = method_classes[np.argmax(avg_method_probs)]
        
        # Format the final result
        result = {
            "fight": f"{fighter1_name} vs {fighter2_name}",
            "predicted_winner": fighter1_name if final_prob_fighter1_wins > final_prob_fighter2_wins else fighter2_name,
            "winner_confidence": f"{max(final_prob_fighter1_wins, final_prob_fighter2_wins)*100:.2f}%",
            "win_probabilities": {
                fighter1_name: f"{final_prob_fighter1_wins*100:.2f}%",
                fighter2_name: f"{final_prob_fighter2_wins*100:.2f}%",
            },
            "predicted_method": predicted_method,
            "method_probabilities": {
                method_classes[i]: f"{avg_method_probs[i]*100:.2f}%" for i in range(len(method_classes))
            },
            "raw_winner_probs": [final_prob_fighter1_wins, final_prob_fighter2_wins],
            "raw_method_probs": avg_method_probs
        }
        
        return result
        
    except ValueError as e:
        return {"error": str(e)}