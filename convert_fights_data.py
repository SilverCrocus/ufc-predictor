#!/usr/bin/env python3
"""
Convert Winner/Loser fight format to Fighter/Opponent format expected by the system.
Each fight is represented twice: once from each fighter's perspective.
"""

import pandas as pd

def convert_fights_format(input_file, output_file):
    """Convert Winner/Loser format to Fighter/Opponent format."""
    # Read the original data
    df = pd.read_csv(input_file)
    
    # Create empty list for converted fights
    converted_fights = []
    
    for _, row in df.iterrows():
        # Fight from winner's perspective (win)
        fight_winner = {
            'Outcome': 'win',
            'Fighter': row['Winner'],
            'Opponent': row['Loser'],
            'Event': row['Event'],
            'Method': row['Method'],
            'Round': row['Round'],
            'Time': row['Time'],
            'fighter_url': row['winner_url'],
            'opponent_url': row['loser_url']
        }
        
        # Fight from loser's perspective (loss)
        fight_loser = {
            'Outcome': 'loss',
            'Fighter': row['Loser'],
            'Opponent': row['Winner'],
            'Event': row['Event'],
            'Method': row['Method'],
            'Round': row['Round'],
            'Time': row['Time'],
            'fighter_url': row['loser_url'],
            'opponent_url': row['winner_url']
        }
        
        converted_fights.extend([fight_winner, fight_loser])
    
    # Create DataFrame and save
    converted_df = pd.DataFrame(converted_fights)
    converted_df.to_csv(output_file, index=False)
    print(f"Converted {len(df)} fights to {len(converted_df)} fight records")
    print(f"Saved to: {output_file}")
    
    return converted_df

if __name__ == "__main__":
    convert_fights_format(
        'data/ufc_fights_sample.csv',
        'data/ufc_fights.csv'
    )