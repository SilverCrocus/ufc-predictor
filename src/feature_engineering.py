import pandas as pd
import numpy as np
from typing import Tuple


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean object columns by removing leading colons and extra spaces."""
    for col in df.select_dtypes(include=['object']).columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.lstrip(':').str.strip()
    return df


def parse_height_to_inches(height_str: str) -> float:
    """Convert height string like '5\' 11"' to inches."""
    if pd.isna(height_str):
        return None
    try:
        feet, inches = height_str.split("' ")
        return int(feet) * 12 + int(inches.replace('"', ''))
    except (ValueError, AttributeError):
        return None


def parse_weight_to_lbs(weight_str: str) -> float:
    """Convert weight string like '185 lbs.' to numeric value."""
    if pd.isna(weight_str):
        return None
    return pd.to_numeric(weight_str.replace(' lbs.', ''), errors='coerce')


def parse_reach_to_inches(reach_str: str) -> float:
    """Convert reach string like '76"' to numeric value."""
    if pd.isna(reach_str):
        return None
    return pd.to_numeric(reach_str.replace('"', ''), errors='coerce')


def parse_percentage_columns(df: pd.DataFrame, percent_cols: list) -> pd.DataFrame:
    """Convert percentage columns to decimal values."""
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce') / 100.0
    return df


def parse_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Convert columns to numeric values."""
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_record(record_str: str) -> Tuple[int, int, int]:
    """Parse fight record string like '15-3-1' to (wins, losses, draws)."""
    if pd.isna(record_str):
        return (None, None, None)
    try:
        parts = record_str.split('-')
        wins = int(parts[0])
        losses = int(parts[1])
        draws = int(parts[2].split(' ')[0]) if len(parts) > 2 else 0
        return (wins, losses, draws)
    except (ValueError, IndexError):
        return (None, None, None)


def calculate_age(dob: pd.Series, reference_date: str = '2025-06-17') -> pd.Series:
    """Calculate age from date of birth."""
    dob_parsed = pd.to_datetime(dob, format='%b %d, %Y', errors='coerce')
    current_date = pd.to_datetime(reference_date)
    return ((current_date - dob_parsed).dt.days / 365.25).round(1)


def create_stance_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables for fighter stance."""
    if 'STANCE' in df.columns:
        # Fix deprecation warning by avoiding chained assignment
        df = df.copy()
        df.loc[:, 'STANCE'] = df['STANCE'].fillna('Unknown')
        stance_dummies = pd.get_dummies(df['STANCE'], prefix='STANCE', dtype=int)
        df = pd.concat([df, stance_dummies], axis=1)
    return df


def engineer_features_final(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function that processes raw fighter data.
    
    Args:
        raw_df: Raw fighter DataFrame from web scraping
        
    Returns:
        Fully engineered DataFrame ready for modeling
    """
    df = raw_df.copy()
    
    # Clean text columns
    df = clean_object_columns(df)
    df.replace('--', pd.NA, inplace=True)
    
    # Parse physical attributes
    df['Height (inches)'] = df['Height'].apply(parse_height_to_inches)
    df['Weight (lbs)'] = df['Weight'].apply(parse_weight_to_lbs)
    df['Reach (in)'] = df['Reach'].apply(parse_reach_to_inches)
    
    # Parse percentage and numeric columns
    percent_cols = ['Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.']
    per_min_cols = ['SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.']
    
    df = parse_percentage_columns(df, percent_cols)
    df = parse_numeric_columns(df, per_min_cols)
    
    # Parse fight record
    record_data = df['Record'].apply(parse_record)
    df[['Wins', 'Losses', 'Draws']] = pd.DataFrame(record_data.tolist(), index=df.index)
    
    # Calculate age
    df['Age'] = calculate_age(df['DOB'])
    
    # Create stance dummies
    df = create_stance_dummies(df)
    
    # Drop original columns that were transformed
    cols_to_drop = ['Height', 'Weight', 'Reach', 'Record', 'DOB', 'STANCE']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return df


def create_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create differential features between blue and red corner fighters.
    
    Args:
        df: DataFrame with blue_ and red_ prefixed columns
        
    Returns:
        DataFrame with additional differential features
    """
    df_features = df.copy()
    blue_cols = [col for col in df_features.columns 
                 if col.startswith('blue_') and 'url' not in col and 'Name' not in col]
    
    for blue_col in blue_cols:
        red_col = blue_col.replace('blue_', 'red_')
        if red_col in df_features.columns:
            base_name = blue_col.replace('blue_', '')
            diff_col_name = base_name.lower().replace(' ', '_').replace('.', '') + '_diff'
            df_features[diff_col_name] = df_features[blue_col] - df_features[red_col]
    
    return df_features


def merge_fight_data(fights_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fight data with fighter statistics.
    
    Args:
        fights_df: DataFrame containing fight results
        fighters_df: DataFrame containing fighter statistics
        
    Returns:
        Merged DataFrame with blue and red corner statistics
    """
    # Create a copy to avoid modifying original data
    fights_copy = fights_df.copy()
    
    # Create name to URL mapping
    name_to_url_map = fighters_df.set_index('Name')['fighter_url'].to_dict()
    fights_copy['opponent_url'] = fights_copy['Opponent'].map(name_to_url_map)
    fights_copy = fights_copy.dropna(subset=['opponent_url', 'fighter_url'])
    
    # Create prefixed DataFrames
    blue_corner_stats = fighters_df.add_prefix('blue_')
    red_corner_stats = fighters_df.add_prefix('red_')
    
    # Merge data
    merged_df = pd.merge(fights_copy, blue_corner_stats, 
                        left_on='fighter_url', right_on='blue_fighter_url', how='left')
    fight_dataset = pd.merge(merged_df, red_corner_stats, 
                           left_on='opponent_url', right_on='red_fighter_url', how='left')
    
    return fight_dataset


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning modeling.
    
    Args:
        df: Fight dataset with features
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Filter to clear wins/losses only
    df_model_ready = df[df['Outcome'].isin(['win', 'loss'])].copy()
    
    # Create target variable
    df_model_ready['blue_is_winner'] = (df_model_ready['Outcome'] == 'win').astype(int)
    y = df_model_ready['blue_is_winner']
    
    # Select features
    cols_to_drop = [
        'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time',
        'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
        'blue_Name', 'red_Name', 'blue_is_winner'
    ]
    X = df_model_ready.drop(columns=cols_to_drop)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y