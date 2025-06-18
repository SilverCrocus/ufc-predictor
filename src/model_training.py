import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class UFCModelTrainer:
    """Class for training UFC fight prediction models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Train a logistic regression model with feature scaling."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           n_estimators: int = 100) -> RandomForestClassifier:
        """Train a random forest model."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          n_iter: int = 100, cv: int = 3) -> RandomForestClassifier:
        """Tune random forest hyperparameters using RandomizedSearchCV."""
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        rf_random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_random_search.fit(X_train, y_train)
        best_model = rf_random_search.best_estimator_
        
        self.models['random_forest_tuned'] = best_model
        
        print("Best hyperparameters found:")
        print(rf_random_search.best_params_)
        
        return best_model
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, 
                      show_plots: bool = True) -> Dict[str, Any]:
        """Evaluate a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Handle scaling for logistic regression
        if model_name == 'logistic_regression' and model_name in self.scalers:
            X_test_processed = self.scalers[model_name].transform(X_test)
        else:
            X_test_processed = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=['Red Wins', 'Blue Wins']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n{model_name.replace('_', ' ').title()} Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:\n{results['classification_report']}")
        
        if show_plots:
            self._plot_confusion_matrix(results['confusion_matrix'], model_name)
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Red Wins', 'Blue Wins'], 
                   yticklabels=['Red Wins', 'Blue Wins'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
        plt.show()
    
    def get_feature_importance(self, model_name: str, feature_names: list, top_n: int = 15) -> pd.Series:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not support feature importance")
        
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        
        print(f"\nTop {top_n} Most Important Features for {model_name}:")
        print(importances.head(top_n))
        
        return importances
    
    def save_model(self, model_name: str, filepath: str, feature_columns: list = None):
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        joblib.dump(model, filepath)
        
        # Save feature columns if provided
        if feature_columns is not None:
            columns_filepath = filepath.replace('.joblib', '_columns.json')
            with open(columns_filepath, 'w') as f:
                json.dump(feature_columns, f)
            print(f"Feature columns saved to '{columns_filepath}'")
        
        # Save scaler if exists
        if model_name in self.scalers:
            scaler_filepath = filepath.replace('.joblib', '_scaler.joblib')
            joblib.dump(self.scalers[model_name], scaler_filepath)
            print(f"Scaler saved to '{scaler_filepath}'")
        
        print(f"Model '{model_name}' saved to '{filepath}'")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load a trained model from disk."""
        return joblib.load(filepath)
    
    @staticmethod
    def load_feature_columns(filepath: str) -> list:
        """Load feature columns from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def train_complete_pipeline(X: pd.DataFrame, y: pd.Series, 
                          tune_hyperparameters: bool = True) -> UFCModelTrainer:
    """
    Complete training pipeline for UFC prediction models.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        tune_hyperparameters: Whether to tune random forest hyperparameters
        
    Returns:
        Trained UFCModelTrainer instance
    """
    trainer = UFCModelTrainer()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    print(f"Data split - Training: {X_train.shape}, Testing: {X_test.shape}")
    
    # Train logistic regression
    print("\nTraining Logistic Regression...")
    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model('logistic_regression', X_test, y_test)
    
    # Train random forest
    print("\nTraining Random Forest...")
    trainer.train_random_forest(X_train, y_train)
    trainer.evaluate_model('random_forest', X_test, y_test)
    trainer.get_feature_importance('random_forest', X.columns.tolist())
    
    # Tune random forest if requested
    if tune_hyperparameters:
        print("\nTuning Random Forest hyperparameters...")
        trainer.tune_random_forest(X_train, y_train)
        trainer.evaluate_model('random_forest_tuned', X_test, y_test)
    
    # Save feature columns for later use
    trainer.feature_columns = X.columns.tolist()
    
    return trainer