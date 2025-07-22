#!/usr/bin/env python3
"""
Create a simple method prediction model.
"""

import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def clean_method_column(df):
    """Clean the Method column for multiclass training."""
    def clean_method(method):
        if pd.isna(method):
            return None
        method_upper = str(method).upper()
        if method_upper.startswith('KO') or method_upper.startswith('TKO'):
            return 'KO/TKO'
        if method_upper.startswith('SUB'):
            return 'Submission'
        if 'DEC' in method_upper or method_upper == 'DECISION':
            return 'Decision'
        return None
    
    df_copy = df.copy()
    df_copy['Method_Cleaned'] = df_copy['Method'].apply(clean_method)
    return df_copy.dropna(subset=['Method_Cleaned'])

def create_method_model():
    """Create a simple method prediction model."""
    
    # Load the processed data
    fight_dataset = pd.read_csv('data/ufc_fights.csv')
    fighters_data = pd.read_csv('model/ufc_fighters_engineered_corrected.csv')
    
    # Merge the data (simulate the differential features creation)
    # For simplicity, we'll use a subset of features
    feature_columns = [
        'SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.',
        'Height (inches)', 'Weight (lbs)', 'Reach (inches)', 'Age', 'Wins', 'Losses', 'Draws',
        'Stance_Orthodox', 'Stance_Southpaw'
    ]
    
    # Create simple features (just use one fighter's stats for now)
    simple_data = []
    for _, fight in fight_dataset.iterrows():
        # Find fighter data
        fighter_stats = fighters_data[fighters_data['Name'] == fight['Fighter']].iloc[0]
        
        row = {}
        for col in feature_columns:
            if col in fighter_stats:
                row[col] = fighter_stats[col]
            else:
                row[col] = 0
        row['Method'] = fight['Method']
        simple_data.append(row)
    
    df = pd.DataFrame(simple_data)
    df_clean = clean_method_column(df)
    
    # Prepare features and target
    X = df_clean[feature_columns].fillna(0)
    y = df_clean['Method_Cleaned']
    
    print(f"Method dataset shape: {X.shape}")
    print(f"Method distribution: {y.value_counts().to_dict()}")
    
    if len(y.unique()) < 2:
        print("Not enough method variety, creating dummy model...")
        # Create a dummy model that always predicts Decision
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant='Decision')
    else:
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Split and train
    if len(df_clean) > 4:  # Need at least 4 samples for train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
    else:
        model.fit(X, y)
    
    # Save model
    model_path = "model/ufc_multiclass_model.joblib"
    joblib.dump(model, model_path)
    print(f"Method model saved to: {model_path}")
    
    # Save column names
    columns_path = "model/method_model_columns.json"
    with open(columns_path, 'w') as f:
        json.dump(list(X.columns), f)
    print(f"Method columns saved to: {columns_path}")

if __name__ == "__main__":
    create_method_model()