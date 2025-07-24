#!/usr/bin/env python3
"""
UFC Fighter ELO Ratings CSV Generator

This script builds ELO ratings for all UFC fighters from historical fight data
and exports them to a comprehensive CSV file for analysis.

Author: Generated with Claude Code
Date: July 2025
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import re
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ufc_elo_system import UFCELOSystem, UFCFightResult
except ImportError as e:
    print(f"Error importing ELO system: {e}")
    print("Make sure the src/ directory contains the ELO system files")
    sys.exit(1)

class ELORatingGenerator:
    def __init__(self):
        self.elo_system = None
        self.fights_df = None
        self.fighters_df = None
        self.processed_fights = 0
        
    def load_data(self):
        """Load historical fight data and current fighter data"""
        print("📊 Loading historical fight data...")
        
        # Try to find the most recent fight data
        fight_data_paths = [
            'data/scrape_2025-07-23_01-09/ufc_fights_2025-07-23.csv',
            'data/ufc_fights.csv',
            'data/ufc_fights_*.csv'
        ]
        
        fight_file = None
        for path in fight_data_paths:
            if os.path.exists(path):
                fight_file = path
                break
        
        if not fight_file:
            raise FileNotFoundError("Could not find fight data file")
            
        self.fights_df = pd.read_csv(fight_file)
        print(f"   ✅ Loaded {len(self.fights_df):,} fight records from {fight_file}")
        
        # Load current fighter data
        print("👤 Loading current fighter data...")
        fighter_data_paths = [
            'model/training_2025-07-23_09-51/ufc_fighters_engineered_2025-07-23_09-51.csv',
            'data/ufc_fighters_engineered_corrected.csv',
            'data/ufc_fighters.csv'
        ]
        
        fighter_file = None
        for path in fighter_data_paths:
            if os.path.exists(path):
                fighter_file = path
                break
                
        if not fighter_file:
            raise FileNotFoundError("Could not find fighter data file")
            
        self.fighters_df = pd.read_csv(fighter_file)
        print(f"   ✅ Loaded {len(self.fighters_df):,} fighter records from {fighter_file}")
        
    def parse_event_date(self, event_string):
        """Extract date from event string"""
        try:
            # Look for date patterns like "Dec. 04, 2010" or "Jan. 18, 2025"
            date_pattern = r'([A-Za-z]{3})\.?\s+(\d{1,2}),\s+(\d{4})'
            match = re.search(date_pattern, event_string)
            
            if match:
                month_str, day, year = match.groups()
                
                # Month abbreviation mapping
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                
                month = month_map.get(month_str, '01')
                date_str = f"{year}-{month}-{day.zfill(2)}"
                return datetime.strptime(date_str, "%Y-%m-%d")
            
            # Fallback: assume it's from 2020 if no date found
            return datetime(2020, 1, 1)
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not parse date from '{event_string}': {e}")
            return datetime(2020, 1, 1)
    
    def normalize_method(self, method):
        """Normalize fight method to standard categories"""
        if pd.isna(method):
            return "Decision"
            
        method = str(method).upper()
        
        if any(x in method for x in ['KO', 'TKO', 'KNOCKOUT', 'TECHNICAL KNOCKOUT']):
            return "KO/TKO"
        elif any(x in method for x in ['SUB', 'SUBMISSION', 'CHOKE']):
            return "Submission"
        elif any(x in method for x in ['DEC', 'DECISION']):
            return "Decision"
        else:
            return "Decision"  # Default
    
    def initialize_elo_system(self):
        """Initialize the ELO rating system"""
        print("⚡ Initializing UFC ELO System...")
        
        # Initialize with multi-dimensional ratings
        self.elo_system = UFCELOSystem(
            initial_rating=1400.0,
            use_multi_dimensional=True
        )
        
        print("   ✅ ELO system initialized with multi-dimensional ratings")
        
    def prepare_fights_dataframe(self):
        """Transform fight data to format expected by ELO system"""
        print("🔄 Transforming fight data for ELO system...")
        
        # Add date column to fights data
        print("   📅 Parsing event dates...")
        self.fights_df['Date'] = self.fights_df['Event'].apply(self.parse_event_date)
        self.fights_df['Method'] = self.fights_df['Method'].apply(self.normalize_method)
        
        # Transform to winner/loser format expected by ELO system
        transformed_fights = []
        
        for idx, fight in self.fights_df.iterrows():
            try:
                # Determine winner and loser based on outcome
                if fight['Outcome'] == 'win':
                    winner = fight['Fighter']
                    loser = fight['Opponent']
                else:  # loss
                    winner = fight['Opponent'] 
                    loser = fight['Fighter']
                
                # Create transformed fight record
                transformed_fight = {
                    'Winner': winner,
                    'Loser': loser,
                    'Date': fight['Date'],
                    'Method': fight['Method'],
                    'Round': fight['Round'] if pd.notna(fight['Round']) else 3,
                    'Time': fight['Time'] if pd.notna(fight['Time']) else '5:00',
                    'Event': fight['Event'],
                    'Weight_class': '',  # Not available in our data
                    'fighter_url': fight['fighter_url']
                }
                
                transformed_fights.append(transformed_fight)
                
            except Exception as e:
                print(f"   ⚠️  Warning: Could not transform fight {idx}: {e}")
                continue
        
        # Create new DataFrame with transformed data
        self.transformed_fights_df = pd.DataFrame(transformed_fights)
        
        # Sort by date (oldest first)
        self.transformed_fights_df = self.transformed_fights_df.sort_values('Date')
        
        print(f"   ✅ Transformed {len(self.transformed_fights_df):,} fights")
        print(f"   📊 Date range: {self.transformed_fights_df['Date'].min()} to {self.transformed_fights_df['Date'].max()}")
        
    def process_fights_chronologically(self):
        """Process all fights in chronological order to build ELO ratings"""
        print("🥊 Processing fights chronologically to build ELO ratings...")
        
        # Prepare the DataFrame in the format expected by ELO system
        self.prepare_fights_dataframe()
        
        # Use the ELO system's built-in method to process all fights
        processed_fight_updates = self.elo_system.build_from_fight_history(self.transformed_fights_df)
        
        self.processed_fights = len(processed_fight_updates)
        
        print(f"   ✅ Successfully processed {self.processed_fights:,} fights")
        print(f"   👥 ELO ratings generated for {len(self.elo_system.fighters):,} fighters")
        
    def export_to_csv(self, output_file='ufc_fighter_elo_ratings.csv'):
        """Export ELO ratings to CSV with fighter data integration"""
        print(f"💾 Exporting ELO ratings to {output_file}...")
        
        # Get ELO ratings data
        elo_data = []
        
        for fighter_name, fighter_elo in self.elo_system.fighters.items():
            elo_record = {
                'fighter_name': fighter_name,
                'elo_overall_rating': round(fighter_elo.overall_rating, 2),
                'elo_striking_rating': round(fighter_elo.striking_rating, 2),
                'elo_grappling_rating': round(fighter_elo.grappling_rating, 2),
                'elo_cardio_rating': round(fighter_elo.cardio_rating, 2),
                'elo_rating_deviation': round(fighter_elo.rating_deviation, 2),
                'elo_total_fights': fighter_elo.total_fights,
                'elo_ufc_fights': fighter_elo.ufc_fights,
                'elo_title_fights': fighter_elo.title_fights,
                'elo_main_events': fighter_elo.main_events,
                'elo_current_streak': fighter_elo.current_streak,
                'elo_win_percentage': round(fighter_elo.win_percentage, 3),
                'elo_finish_percentage': round(fighter_elo.finish_percentage, 3),
                'elo_last_fight_date': fighter_elo.last_fight_date.strftime('%Y-%m-%d') if fighter_elo.last_fight_date else None,
                'elo_last_updated': fighter_elo.last_updated.strftime('%Y-%m-%d %H:%M:%S') if fighter_elo.last_updated else None,
                'elo_active_status': fighter_elo.active_status
            }
            elo_data.append(elo_record)
        
        # Create ELO DataFrame
        elo_df = pd.DataFrame(elo_data)
        
        # Try to merge with existing fighter data if possible
        if self.fighters_df is not None and 'Name' in self.fighters_df.columns:
            print("   🔗 Merging with existing fighter data...")
            
            # Merge on fighter name
            merged_df = self.fighters_df.merge(
                elo_df, 
                left_on='Name', 
                right_on='fighter_name', 
                how='outer',
                indicator=True
            )
            
            # Report merge statistics
            merge_stats = merged_df['_merge'].value_counts()
            print(f"   📊 Merge results:")
            print(f"      Both datasets: {merge_stats.get('both', 0):,}")
            print(f"      Only fighter data: {merge_stats.get('left_only', 0):,}")  
            print(f"      Only ELO data: {merge_stats.get('right_only', 0):,}")
            
            # Drop merge indicator and duplicate name column
            final_df = merged_df.drop(['_merge', 'fighter_name'], axis=1, errors='ignore')
        else:
            print("   📊 Using ELO data only (no fighter data integration)")
            final_df = elo_df
        
        # Export to CSV
        final_df.to_csv(output_file, index=False)
        print(f"   ✅ Exported {len(final_df):,} fighter records to {output_file}")
        
        return final_df
    
    def show_top_fighters(self, df, n=20):
        """Display top fighters by ELO rating"""
        print(f"\n🏆 TOP {n} FIGHTERS BY ELO RATING:")
        print("=" * 80)
        
        if 'elo_overall_rating' in df.columns:
            top_fighters = df.nlargest(n, 'elo_overall_rating')
            
            for idx, fighter in top_fighters.iterrows():
                name = fighter.get('Name', fighter.get('fighter_name', 'Unknown'))
                overall = fighter.get('elo_overall_rating', 0)
                striking = fighter.get('elo_striking_rating', 0)
                grappling = fighter.get('elo_grappling_rating', 0)
                cardio = fighter.get('elo_cardio_rating', 0)
                fights = fighter.get('elo_total_fights', 0)
                
                print(f"{name:<25} | Overall: {overall:>6.0f} | Strike: {striking:>6.0f} | "
                      f"Grapple: {grappling:>6.0f} | Cardio: {cardio:>6.0f} | Fights: {fights:>3}")
        
        print("=" * 80)

def main():
    """Main execution function"""
    print("🥊 UFC Fighter ELO Ratings CSV Generator")
    print("=" * 50)
    
    try:
        # Initialize generator
        generator = ELORatingGenerator()
        
        # Load data
        generator.load_data()
        
        # Initialize ELO system
        generator.initialize_elo_system()
        
        # Process fights to build ratings
        generator.process_fights_chronologically()
        
        # Export to CSV
        output_file = 'ufc_fighter_elo_ratings.csv'
        result_df = generator.export_to_csv(output_file)
        
        # Show top fighters
        generator.show_top_fighters(result_df)
        
        print(f"\n🎯 SUCCESS: ELO ratings CSV generated!")
        print(f"   📄 File: {output_file}")
        print(f"   👥 Fighters: {len(result_df):,}")
        print(f"   🥊 Fights processed: {generator.processed_fights:,}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())