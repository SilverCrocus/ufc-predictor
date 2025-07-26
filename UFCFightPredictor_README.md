# UFCFightPredictor Wrapper Class

The `UFCFightPredictor` class is a wrapper that provides the interface expected by the enhanced notebook while using the existing UFC prediction system. It bridges the gap between the sophisticated existing infrastructure and the simplified interface needed for enhanced analysis.

## Key Features

### 🔄 **Seamless Integration**
- Auto-detects and loads latest trained models
- Uses existing `predict_fight_symmetrical()` function under the hood  
- Maintains compatibility with current model training pipeline
- Graceful fallback to standard model locations when needed

### 🎯 **Simplified Interface**
- One-line initialization: `predictor = UFCFightPredictor()`
- Unified prediction methods that return structured results
- Built-in error handling and validation
- Fighter search and validation utilities

### 📊 **Enhanced Notebook Ready**
- Returns prediction results in the format expected by enhanced notebooks
- Provides model accuracy and metadata access
- Supports both single predictions and batch fight card analysis
- Includes confidence scores and method predictions

## Quick Start

```python
from src.ufc_fight_predictor import UFCFightPredictor

# Initialize (automatically loads latest models)
predictor = UFCFightPredictor()

# Check if loaded successfully
print(f"Status: {predictor}")
print(f"Model accuracy: {predictor.model_accuracy:.3f}")

# Single fight prediction
result = predictor.predict_fight("Jon Jones", "Daniel Cormier")
print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Method: {result['method_prediction']}")

# Fight card predictions
fights = ["Fighter1 vs Fighter2", "Fighter3 vs Fighter4"]
results = predictor.predict_fight_card(fights)
```

## Core Methods

### `__init__(model_dir=None, auto_load=True)`
Initialize the predictor with automatic model loading.

**Parameters:**
- `model_dir`: Optional path to model directory (auto-detects if None)
- `auto_load`: Whether to load models during initialization

### `predict_fight(fighter_a, fighter_b)`
Predict the outcome of a single fight.

**Returns:**
```python
{
    "predicted_winner": "Fighter Name",
    "confidence": 0.756,  # Float between 0-1
    "win_probabilities": {
        "Fighter A": 0.756,
        "Fighter B": 0.244
    },
    "method_prediction": "KO/TKO",
    "method_probabilities": {
        "Decision": 0.123,
        "KO/TKO": 0.654,
        "Submission": 0.223
    },
    "fight": "Fighter A vs Fighter B"
}
```

### `predict_fight_card(fights)`
Predict outcomes for multiple fights.

**Parameters:**
- `fights`: List of tuples `(fighter_a, fighter_b)` or strings `"Fighter A vs Fighter B"`

**Returns:** List of prediction dictionaries (same format as single prediction)

### Properties

- **`model_accuracy`**: Current model accuracy (float)
- **`is_loaded`**: Whether models are successfully loaded (bool)
- **`model_info`**: Detailed information about loaded models (dict)

## Utility Methods

### `search_fighter(name, limit=10)`
Search for fighters by name with fuzzy matching.

### `validate_fighters(fighter_a, fighter_b)`
Validate that both fighters exist in the dataset.

### `get_available_fighters(search_term=None, limit=None)`
Get list of available fighters with optional filtering.

### `quick_predict(fight_string)`
Quick prediction from a fight string like "Fighter A vs Fighter B".

## Auto-Detection Logic

The wrapper uses intelligent auto-detection to find the latest models:

1. **Latest Training Directory**: Searches for `model/training_YYYY-MM-DD_HH-MM/` directories
2. **Model Preference**: Uses tuned models when available, falls back to standard models
3. **Data Sources**: Loads fighter data from training directory or latest scrape directory
4. **Fallback System**: Falls back to standard model locations if latest models unavailable

## Error Handling

The wrapper includes robust error handling:

- **Graceful Fallbacks**: Automatically tries alternative model locations
- **Clear Error Messages**: Provides specific error information for debugging
- **Fighter Validation**: Suggests similar fighter names when exact matches not found
- **Load Status Tracking**: Maintains detailed information about what loaded successfully

## Integration with Existing System

The wrapper integrates seamlessly with your existing UFC prediction infrastructure:

- **Uses Existing Functions**: Calls `predict_fight_symmetrical()` from `src/prediction.py`
- **Model Compatibility**: Works with models trained by your existing pipeline
- **Data Compatibility**: Uses the same fighter data and feature engineering
- **Version Management**: Automatically detects and uses latest model versions

## Advanced Usage

### Batch Analysis
```python
# Analyze multiple fighters against a specific opponent
fighters = predictor.get_available_fighters(limit=10)
results = []

for fighter in fighters:
    result = predictor.predict_fight(fighter, "Jon Jones")
    if "error" not in result:
        results.append({
            'fighter': fighter,
            'win_probability': result['win_probabilities'][fighter],
            'method': result['method_prediction']
        })
```

### Model Information Access
```python
# Get detailed model information
info = predictor.model_info
print(f"Model timestamp: {info['model_timestamp']}")
print(f"Fighters in database: {info['fighters_count']}")
print(f"Training metadata available: {info['training_metadata'] is not None}")
```

## Files Created

- **`/Users/diyagamah/Documents/ufc-predictor/src/ufc_fight_predictor.py`**: Main wrapper class
- **Enhanced notebook integration**: Ready to use in Jupyter notebooks
- **Automatic imports**: Added to `src/__init__.py` for easy importing

## Example Output

```
Successfully loaded models from /path/to/model/training_2025-07-23_09-51
Model accuracy: 0.729
Status: UFCFightPredictor(Loaded (Accuracy: 0.729))

Predicting: Jon Jones vs Daniel Cormier
Predicted Winner: Jon Jones
Confidence: 75.6%
Method: Decision
Win Probabilities:
  Jon Jones: 75.6%
  Daniel Cormier: 24.4%
Method Probabilities:
  Decision: 45.2%
  KO/TKO: 32.1%
  Submission: 22.7%
```

The wrapper successfully bridges the enhanced notebook interface with your existing sophisticated UFC prediction system, providing a clean, simple API while maintaining all the powerful analysis capabilities underneath.