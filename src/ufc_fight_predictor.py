"""
UFC Fight Predictor Wrapper Class

This wrapper class provides the interface expected by the enhanced notebook
while using the existing UFC prediction system. It automatically detects and
loads the latest models and provides unified prediction methods.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
from datetime import datetime

try:
    from .prediction import predict_fight_symmetrical
except ImportError:
    from prediction import predict_fight_symmetrical


class UFCFightPredictor:
    """
    Wrapper class that provides the interface expected by the enhanced notebook
    while using the existing UFC prediction system.
    
    Features:
    - Auto-detects and loads latest models from training directories
    - Provides unified prediction interface
    - Handles fallbacks gracefully
    - Returns structured prediction results
    """
    
    def __init__(self, model_dir: str = None, auto_load: bool = True):
        """
        Initialize the UFC Fight Predictor.
        
        Args:
            model_dir: Directory containing models. If None, auto-detects latest.
            auto_load: Whether to automatically load models during initialization.
        """
        self.project_root = Path(__file__).parent.parent
        self.model_dir = Path(model_dir) if model_dir else self.project_root / "model"
        
        # Model components
        self.winner_model = None
        self.method_model = None
        self.winner_columns = None
        self.method_columns = None
        self.fighters_data = None
        
        # Model metadata
        self.model_accuracy = None
        self.model_timestamp = None
        self.training_metadata = None
        
        # Load status
        self.is_loaded = False
        self.load_errors = []
        
        if auto_load:
            self.load_models()
    
    def _find_latest_training_directory(self) -> Optional[Path]:
        """Find the most recent training directory."""
        training_pattern = str(self.model_dir / "training_*")
        training_dirs = glob.glob(training_pattern)
        
        if not training_dirs:
            return None
        
        # Sort by directory name (which includes timestamp)
        training_dirs.sort(reverse=True)
        return Path(training_dirs[0])
    
    def _find_latest_data_directory(self) -> Optional[Path]:
        """Find the most recent data directory."""
        data_dir = self.project_root / "data"
        scrape_pattern = str(data_dir / "scrape_*")
        scrape_dirs = glob.glob(scrape_pattern)
        
        if not scrape_dirs:
            return None
        
        # Sort by directory name (which includes timestamp)
        scrape_dirs.sort(reverse=True)
        return Path(scrape_dirs[0])
    
    def load_models(self) -> bool:
        """
        Load the latest available models.
        
        Returns:
            bool: True if models loaded successfully, False otherwise.
        """
        try:
            # Find latest training directory
            latest_training_dir = self._find_latest_training_directory()
            if not latest_training_dir:
                self.load_errors.append("No training directories found")
                return self._try_fallback_loading()
            
            self.model_timestamp = latest_training_dir.name.replace('training_', '')
            self.latest_model_dir = str(latest_training_dir)  # Add this for notebook compatibility
            
            # Load training metadata
            metadata_path = latest_training_dir / f"training_metadata_{self.model_timestamp}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.training_metadata = json.load(f)
                    # Get the best accuracy from tuned model
                    if 'winner_models' in self.training_metadata:
                        winner_models = self.training_metadata['winner_models']
                        if 'random_forest_tuned' in winner_models:
                            self.model_accuracy = winner_models['random_forest_tuned']['accuracy']
                        elif 'random_forest' in winner_models:
                            self.model_accuracy = winner_models['random_forest']['accuracy']
            
            # Load winner model (prefer tuned version)
            winner_model_path = latest_training_dir / f"ufc_winner_model_tuned_{self.model_timestamp}.joblib"
            if not winner_model_path.exists():
                winner_model_path = latest_training_dir / f"ufc_winner_model_{self.model_timestamp}.joblib"
            
            if winner_model_path.exists():
                self.winner_model = joblib.load(winner_model_path)
            else:
                self.load_errors.append(f"Winner model not found in {latest_training_dir}")
            
            # Load method model
            method_model_path = latest_training_dir / f"ufc_method_model_{self.model_timestamp}.joblib"
            if method_model_path.exists():
                self.method_model = joblib.load(method_model_path)
            else:
                self.load_errors.append(f"Method model not found in {latest_training_dir}")
            
            # Load winner model columns (prefer tuned version)
            winner_cols_path = latest_training_dir / f"ufc_winner_model_tuned_{self.model_timestamp}_columns.json"
            if not winner_cols_path.exists():
                winner_cols_path = latest_training_dir / f"ufc_winner_model_{self.model_timestamp}_columns.json"
            
            if winner_cols_path.exists():
                with open(winner_cols_path, 'r') as f:
                    self.winner_columns = json.load(f)
            else:
                self.load_errors.append(f"Winner columns not found in {latest_training_dir}")
            
            # Load method model columns
            method_cols_path = latest_training_dir / f"method_model_columns_{self.model_timestamp}.json"
            if method_cols_path.exists():
                with open(method_cols_path, 'r') as f:
                    self.method_columns = json.load(f)
            else:
                self.load_errors.append(f"Method columns not found in {latest_training_dir}")
            
            # Load fighters data (prefer from training directory)
            fighters_data_path = latest_training_dir / f"ufc_fighters_engineered_{self.model_timestamp}.csv"
            if fighters_data_path.exists():
                self.fighters_data = pd.read_csv(fighters_data_path)
            else:
                # Try to find from latest data directory
                latest_data_dir = self._find_latest_data_directory()
                if latest_data_dir:
                    data_timestamp = latest_data_dir.name.replace('scrape_', '').split('_')[0]
                    fighters_data_path = latest_data_dir / f"ufc_fighters_engineered_{data_timestamp}.csv"
                    if fighters_data_path.exists():
                        self.fighters_data = pd.read_csv(fighters_data_path)
                    else:
                        self.load_errors.append(f"Fighters data not found in {latest_data_dir}")
                else:
                    self.load_errors.append("No data directories found")
            
            # Check if we have all required components
            self.is_loaded = (self.winner_model is not None and 
                            self.method_model is not None and 
                            self.winner_columns is not None and 
                            self.method_columns is not None and 
                            self.fighters_data is not None)
            
            if self.is_loaded:
                print(f"Successfully loaded models from {latest_training_dir}")
                if self.model_accuracy:
                    print(f"Model accuracy: {self.model_accuracy:.3f}")
                return True
            else:
                return self._try_fallback_loading()
        
        except Exception as e:
            self.load_errors.append(f"Error loading models: {str(e)}")
            return self._try_fallback_loading()
    
    def _try_fallback_loading(self) -> bool:
        """Try to load models from standard locations as fallback."""
        try:
            fallback_paths = {
                'winner_model': self.model_dir / "ufc_random_forest_model_tuned.joblib",
                'method_model': self.model_dir / "ufc_multiclass_model.joblib", 
                'winner_columns': self.model_dir / "winner_model_columns.json",
                'method_columns': self.model_dir / "method_model_columns.json",
                'fighters_data': self.model_dir / "ufc_fighters_engineered_corrected.csv"
            }
            
            # Try alternative paths
            if not fallback_paths['winner_model'].exists():
                fallback_paths['winner_model'] = self.model_dir / "ufc_random_forest_model.joblib"
            
            # Load fallback models
            fallback_loaded = {}
            
            if fallback_paths['winner_model'].exists():
                self.winner_model = joblib.load(fallback_paths['winner_model'])
                fallback_loaded['winner_model'] = True
            
            if fallback_paths['method_model'].exists():
                self.method_model = joblib.load(fallback_paths['method_model'])
                fallback_loaded['method_model'] = True
            
            if fallback_paths['winner_columns'].exists():
                with open(fallback_paths['winner_columns'], 'r') as f:
                    self.winner_columns = json.load(f)
                fallback_loaded['winner_columns'] = True
            
            if fallback_paths['method_columns'].exists():
                with open(fallback_paths['method_columns'], 'r') as f:
                    self.method_columns = json.load(f)
                fallback_loaded['method_columns'] = True
            
            if fallback_paths['fighters_data'].exists():
                self.fighters_data = pd.read_csv(fallback_paths['fighters_data'])
                fallback_loaded['fighters_data'] = True
            
            self.is_loaded = len(fallback_loaded) >= 4  # Need at least 4 components
            
            if self.is_loaded:
                print("Loaded models from fallback locations")
                print(f"Components loaded: {list(fallback_loaded.keys())}")
                return True
            else:
                print(f"Fallback loading failed. Components loaded: {list(fallback_loaded.keys())}")
                return False
                
        except Exception as e:
            self.load_errors.append(f"Fallback loading failed: {str(e)}")
            return False
    
    def predict_fight(self, fighter_a: str, fighter_b: str) -> Dict[str, Any]:
        """
        Predict the outcome of a fight between two fighters.
        
        Args:
            fighter_a: Name of the first fighter
            fighter_b: Name of the second fighter
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            return {
                "error": "Models not loaded. Call load_models() first.",
                "load_errors": self.load_errors
            }
        
        try:
            # Use the existing symmetrical prediction function
            result = predict_fight_symmetrical(
                fighter_a, fighter_b, 
                self.fighters_data, 
                self.winner_columns, 
                self.method_columns,
                self.winner_model, 
                self.method_model
            )
            
            if "error" in result:
                return result
            
            # Convert to expected format
            predicted_winner = result["predicted_winner"]
            confidence = float(result["winner_confidence"].replace('%', '')) / 100
            
            # Extract method prediction
            method_prediction = result.get("predicted_method", "Decision")
            method_probs = result.get("raw_method_probs", [])
            
            # Format the result for notebook compatibility
            formatted_result = {
                "predicted_winner": predicted_winner,
                "confidence": confidence,
                "win_probabilities": {
                    fighter_a: float(result["win_probabilities"][fighter_a].replace('%', '')) / 100,
                    fighter_b: float(result["win_probabilities"][fighter_b].replace('%', '')) / 100,
                },
                "method_prediction": method_prediction,
                "method_probabilities": {},
                "fight": f"{fighter_a} vs {fighter_b}",
                "raw_data": result  # Include original result for debugging
            }
            
            # Convert method probabilities
            if "method_probabilities" in result:
                for method, prob_str in result["method_probabilities"].items():
                    formatted_result["method_probabilities"][method] = float(prob_str.replace('%', '')) / 100
            
            return formatted_result
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "fight": f"{fighter_a} vs {fighter_b}"
            }
    
    def predict_fight_card(self, fights: List[Union[Tuple[str, str], str]]) -> List[Dict[str, Any]]:
        """
        Predict outcomes for multiple fights.
        
        Args:
            fights: List of fight tuples (fighter_a, fighter_b) or strings "Fighter A vs Fighter B"
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded:
            return [{
                "error": "Models not loaded. Call load_models() first.",
                "load_errors": self.load_errors
            }]
        
        results = []
        
        for fight in fights:
            try:
                # Handle different input formats
                if isinstance(fight, tuple):
                    fighter_a, fighter_b = fight
                elif isinstance(fight, str):
                    if " vs " in fight:
                        fighter_a, fighter_b = fight.split(" vs ", 1)
                    elif " v " in fight:
                        fighter_a, fighter_b = fight.split(" v ", 1)
                    else:
                        results.append({"error": f"Invalid fight format: {fight}"})
                        continue
                else:
                    results.append({"error": f"Invalid fight type: {type(fight)}"})
                    continue
                
                # Clean fighter names
                fighter_a = fighter_a.strip()
                fighter_b = fighter_b.strip()
                
                # Make prediction
                prediction = self.predict_fight(fighter_a, fighter_b)
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    "error": f"Failed to process fight {fight}: {str(e)}"
                })
        
        return results
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            "is_loaded": self.is_loaded,
            "model_accuracy": self.model_accuracy,
            "model_timestamp": self.model_timestamp,
            "load_errors": self.load_errors,
            "has_winner_model": self.winner_model is not None,
            "has_method_model": self.method_model is not None,
            "fighters_count": len(self.fighters_data) if self.fighters_data is not None else 0,
            "training_metadata": self.training_metadata
        }
    
    def get_available_fighters(self, search_term: str = None, limit: int = None) -> List[str]:
        """
        Get list of available fighters.
        
        Args:
            search_term: Optional string to filter fighter names
            limit: Optional limit on number of results
            
        Returns:
            List of fighter names
        """
        if not self.is_loaded or self.fighters_data is None:
            return []
        
        fighters = self.fighters_data['Name'].tolist()
        
        if search_term:
            search_term = search_term.lower()
            fighters = [f for f in fighters if search_term in f.lower()]
        
        fighters = sorted(fighters)
        
        if limit:
            fighters = fighters[:limit]
        
        return fighters
    
    def search_fighter(self, name: str, limit: int = 10) -> List[str]:
        """
        Search for fighters by name.
        
        Args:
            name: Fighter name to search for
            limit: Maximum number of results
            
        Returns:
            List of matching fighter names
        """
        return self.get_available_fighters(search_term=name, limit=limit)
    
    def validate_fighters(self, fighter_a: str, fighter_b: str) -> Tuple[bool, List[str]]:
        """
        Validate that both fighters exist in the dataset.
        
        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name
            
        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []
        
        if not self.is_loaded or self.fighters_data is None:
            return False, ["Models not loaded"]
        
        available_fighters = set(self.fighters_data['Name'].str.lower())
        
        if fighter_a.lower() not in available_fighters:
            similar = self.search_fighter(fighter_a, limit=3)
            if similar:
                errors.append(f"Fighter '{fighter_a}' not found. Did you mean: {', '.join(similar[:3])}?")
            else:
                errors.append(f"Fighter '{fighter_a}' not found")
        
        if fighter_b.lower() not in available_fighters:
            similar = self.search_fighter(fighter_b, limit=3)
            if similar:
                errors.append(f"Fighter '{fighter_b}' not found. Did you mean: {', '.join(similar[:3])}?")
            else:
                errors.append(f"Fighter '{fighter_b}' not found")
        
        return len(errors) == 0, errors
    
    def quick_predict(self, fight_string: str) -> Dict[str, Any]:
        """
        Quick prediction from a fight string.
        
        Args:
            fight_string: String like "Fighter A vs Fighter B"
            
        Returns:
            Prediction dictionary
        """
        try:
            if " vs " in fight_string:
                fighter_a, fighter_b = fight_string.split(" vs ", 1)
            elif " v " in fight_string:
                fighter_a, fighter_b = fight_string.split(" v ", 1)
            else:
                return {"error": f"Invalid fight format: {fight_string}"}
            
            return self.predict_fight(fighter_a.strip(), fighter_b.strip())
            
        except Exception as e:
            return {"error": f"Quick prediction failed: {str(e)}"}
    
    def __repr__(self) -> str:
        """String representation of the predictor."""
        status = "Loaded" if self.is_loaded else "Not Loaded"
        accuracy = f" (Accuracy: {self.model_accuracy:.3f})" if self.model_accuracy else ""
        return f"UFCFightPredictor({status}{accuracy})"


# Convenience function for easy instantiation
def create_ufc_predictor(model_dir: str = None) -> UFCFightPredictor:
    """
    Create and load a UFC Fight Predictor.
    
    Args:
        model_dir: Optional model directory path
        
    Returns:
        UFCFightPredictor instance
    """
    return UFCFightPredictor(model_dir=model_dir, auto_load=True)


# Legacy compatibility
def load_latest_predictor() -> UFCFightPredictor:
    """Load predictor with latest models (legacy compatibility)."""
    return create_ufc_predictor()