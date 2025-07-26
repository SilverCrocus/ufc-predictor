#!/usr/bin/env python3
"""
UFC Bet Tracking CSV Cleanup Script

This script removes duplicate betting recommendations from the CSV file.
Run this to clean up existing duplicated records.

Usage:
    python3 cleanup_duplicate_bets.py
"""

import sys
import os

# Add project path
project_path = '/Users/diyagamah/Documents/ufc-predictor'
if project_path not in sys.path:
    sys.path.append(project_path)

from src.bet_tracking import BetTracker

def main():
    """Clean up duplicate betting recommendations"""
    print("🧹 UFC Bet Tracking CSV Cleanup Script")
    print("=" * 50)
    
    try:
        # Initialize bet tracker
        tracker = BetTracker()
        
        print(f"📁 Target CSV file: {tracker.csv_path}")
        
        # Check if file exists
        if not os.path.exists(tracker.csv_path):
            print(f"❌ CSV file not found: {tracker.csv_path}")
            return
        
        # Remove duplicates
        duplicates_removed = tracker.remove_duplicate_recommendations()
        
        if duplicates_removed > 0:
            print(f"\n🎉 Successfully cleaned CSV file!")
            print(f"📊 Removed {duplicates_removed} duplicate betting recommendations")
            print(f"💾 Backup created before cleanup")
        else:
            print(f"\n✅ CSV file was already clean - no duplicates found")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)