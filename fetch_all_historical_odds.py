#!/usr/bin/env python3
"""
Fetch All Historical UFC Odds
==============================

Fetches ALL available historical UFC odds from The Odds API (June 2020 - present).
This will use your API credits efficiently and save data for backtesting.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ufc_predictor.betting.historical_odds_fetcher import (
    HistoricalOddsAPIClient,
    HistoricalOddsProcessor
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalOddsFetcher:
    """Main class for fetching all historical UFC odds"""
    
    def __init__(self, api_key: str):
        """Initialize the fetcher"""
        self.api_key = api_key
        self.client = HistoricalOddsAPIClient(api_key)
        self.data_dir = Path(__file__).parent / "data" / "historical_odds"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load progress if exists
        self.progress_file = self.data_dir / "fetch_progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self) -> dict:
        """Load fetch progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'last_fetched_date': None,
            'total_records': 0,
            'completed_ranges': []
        }
    
    def save_progress(self):
        """Save fetch progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def fetch_all_historical(self, start_date: datetime = None, 
                           end_date: datetime = None,
                           resume: bool = True):
        """
        Fetch all historical UFC odds data
        
        Args:
            start_date: Start date (defaults to June 2020)
            end_date: End date (defaults to today)
            resume: Whether to resume from last checkpoint
        """
        # Set default dates
        if start_date is None:
            start_date = datetime(2020, 6, 6)  # API historical data starts
        
        if end_date is None:
            end_date = datetime.now()
        
        # Check if resuming
        if resume and self.progress['last_fetched_date']:
            resume_date = datetime.fromisoformat(self.progress['last_fetched_date'])
            logger.info(f"📌 Resuming from {resume_date.date()}")
            start_date = resume_date + timedelta(days=1)
        
        logger.info("=" * 60)
        logger.info("🚀 FETCHING ALL UFC HISTORICAL ODDS")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Total days: {(end_date - start_date).days}")
        logger.info("This will use your API credits. Press Ctrl+C to stop anytime.")
        logger.info("=" * 60)
        
        # Fetch in monthly chunks
        all_records = []
        current_start = start_date
        
        try:
            while current_start < end_date:
                # Process one month at a time
                current_end = min(current_start + timedelta(days=30), end_date)
                
                logger.info(f"\n📅 Processing: {current_start.date()} to {current_end.date()}")
                
                # Fetch this chunk
                chunk_records = self.client.fetch_date_range(
                    current_start, 
                    current_end,
                    interval_hours=24  # Daily snapshots
                )
                
                if chunk_records:
                    all_records.extend(chunk_records)
                    logger.info(f"   ✅ Fetched {len(chunk_records)} records")
                    
                    # Save intermediate results
                    self.save_intermediate_results(all_records, current_end)
                    
                    # Update progress
                    self.progress['last_fetched_date'] = current_end.isoformat()
                    self.progress['total_records'] = len(all_records)
                    self.progress['completed_ranges'].append({
                        'start': current_start.isoformat(),
                        'end': current_end.isoformat(),
                        'records': len(chunk_records)
                    })
                    self.save_progress()
                
                # Check API limits
                if self.client.credits_remaining and self.client.credits_remaining < 100:
                    logger.warning(f"⚠️ Low API credits: {self.client.credits_remaining}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        break
                
                # Move to next chunk
                current_start = current_end + timedelta(days=1)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Fetch interrupted by user")
            logger.info(f"Progress saved. Run with --resume to continue from {current_start.date()}")
        
        except Exception as e:
            logger.error(f"❌ Error during fetch: {e}")
            logger.info(f"Progress saved. Run with --resume to continue from {current_start.date()}")
        
        finally:
            # Save final results
            if all_records:
                self.save_final_results(all_records, start_date, end_date)
    
    def save_intermediate_results(self, records: list, end_date: datetime):
        """Save intermediate results during fetching"""
        df = pd.DataFrame([r.to_dict() for r in records])
        
        # Save to timestamped file
        filename = f"historical_odds_partial_{end_date.strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"   💾 Saved intermediate: {filepath.name} ({len(df)} records)")
    
    def save_final_results(self, records: list, start_date: datetime, end_date: datetime):
        """Save final consolidated results"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SAVING FINAL RESULTS")
        logger.info("=" * 60)
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in records])
        
        # Process the data
        processor = HistoricalOddsProcessor(df)
        
        # Save main dataset
        main_file = self.data_dir / f"ufc_historical_odds_complete.csv"
        df.to_csv(main_file, index=False)
        logger.info(f"✅ Main dataset: {main_file}")
        logger.info(f"   Total records: {len(df)}")
        logger.info(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        # Generate and save analytics
        logger.info("\n📈 Generating analytics...")
        
        # Best odds per fight
        best_odds = processor.get_best_odds()
        best_odds_file = self.data_dir / "best_odds_per_fight.csv"
        best_odds.to_csv(best_odds_file, index=False)
        logger.info(f"✅ Best odds: {best_odds_file}")
        logger.info(f"   Unique fights: {len(best_odds)}")
        
        # Odds movements
        movements = processor.calculate_odds_movement()
        movements_file = self.data_dir / "odds_movements.csv"
        movements.to_csv(movements_file, index=False)
        logger.info(f"✅ Odds movements: {movements_file}")
        
        # Summary statistics
        self.generate_summary_stats(df)
    
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate summary statistics for the dataset"""
        stats = {
            'total_records': len(df),
            'unique_events': df['event_name'].nunique(),
            'unique_fights': len(df.groupby(['fighter_a', 'fighter_b'])),
            'bookmakers': df['bookmaker'].nunique(),
            'date_range': {
                'start': df['commence_time'].min(),
                'end': df['commence_time'].max()
            },
            'avg_bookmaker_margin': df['bookmaker_margin'].mean() if 'bookmaker_margin' in df else None
        }
        
        stats_file = self.data_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"✅ Statistics: {stats_file}")
        
        # Print summary
        logger.info("\n📊 Dataset Summary:")
        logger.info(f"   Total records: {stats['total_records']:,}")
        logger.info(f"   Unique events: {stats['unique_events']}")
        logger.info(f"   Unique fights: {stats['unique_fights']}")
        logger.info(f"   Bookmakers: {stats['bookmakers']}")


def get_api_key():
    """Get API key from environment or .env file"""
    api_key = os.getenv('UFC_ODDS_API_KEY')
    
    if not api_key:
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if 'UFC_ODDS_API_KEY' in line:
                        api_key = line.split('=')[1].strip().strip('"').strip("'")
                        break
    
    return api_key


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch all historical UFC odds')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--test', action='store_true', help='Test mode - fetch only 1 week')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("❌ No API key found!")
        logger.error("Please set UFC_ODDS_API_KEY environment variable")
        logger.error("Or create a .env file with: UFC_ODDS_API_KEY=your_key_here")
        return 1
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Test mode - just fetch 1 week
    if args.test:
        logger.info("🧪 TEST MODE - Fetching only 1 week of data")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
    
    # Initialize fetcher
    fetcher = HistoricalOddsFetcher(api_key)
    
    # Fetch the data
    fetcher.fetch_all_historical(
        start_date=start_date,
        end_date=end_date,
        resume=args.resume
    )
    
    logger.info("\n✅ Historical odds fetch complete!")
    logger.info(f"Data saved to: {fetcher.data_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())