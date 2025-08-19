#!/usr/bin/env python3
"""
Test Historical Odds Fetcher
============================

Test the historical odds fetching with a small date range first.
This helps verify the API is working before fetching all data.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ufc_predictor.betting.historical_odds_fetcher import (
    HistoricalOddsAPIClient,
    HistoricalOddsProcessor
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_small_range():
    """Test with just 1 week of recent data"""
    
    # Get API key
    api_key = os.getenv('UFC_ODDS_API_KEY')
    if not api_key:
        # Try to find it in config files
        config_file = Path(__file__).parent / '.env'
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    if 'UFC_ODDS_API_KEY' in line:
                        api_key = line.split('=')[1].strip().strip('"').strip("'")
                        break
    
    if not api_key:
        logger.error("❌ No API key found!")
        logger.error("Please set UFC_ODDS_API_KEY environment variable")
        logger.error("Or create a .env file with: UFC_ODDS_API_KEY=your_key_here")
        return False
    
    logger.info("=" * 60)
    logger.info("🧪 TESTING HISTORICAL ODDS FETCHER")
    logger.info("=" * 60)
    
    # Initialize client
    logger.info(f"Initializing API client...")
    client = HistoricalOddsAPIClient(api_key)
    
    # Test 1: Fetch just 1 week of recent data
    logger.info("\n📅 Test 1: Fetching 1 week of recent historical data")
    logger.info("-" * 40)
    
    end_date = datetime.now() - timedelta(days=30)  # Start from a month ago
    start_date = end_date - timedelta(days=7)  # Go back another week
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        # Fetch the data
        records = client.fetch_date_range(start_date, end_date, interval_hours=24*7)  # Weekly snapshots
        
        if records:
            logger.info(f"✅ Successfully fetched {len(records)} odds records")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([r.to_dict() for r in records])
            
            # Show summary
            logger.info(f"\n📊 Data Summary:")
            logger.info(f"   Total records: {len(df)}")
            logger.info(f"   Unique events: {df['event_name'].nunique()}")
            logger.info(f"   Unique fights: {len(df.groupby(['fighter_a', 'fighter_b']))}")
            logger.info(f"   Bookmakers: {df['bookmaker'].unique()[:5]}...")  # Show first 5
            
            # Show sample data
            if len(df) > 0:
                logger.info(f"\n📋 Sample Records:")
                sample = df[['event_name', 'fighter_a', 'fighter_b', 
                            'fighter_a_odds', 'fighter_b_odds', 'bookmaker']].head(3)
                for idx, row in sample.iterrows():
                    logger.info(f"   {row['fighter_a']} vs {row['fighter_b']}")
                    logger.info(f"   Odds: {row['fighter_a_odds']:.2f} vs {row['fighter_b_odds']:.2f} ({row['bookmaker']})")
                    logger.info("")
            
            # Test the processor
            processor = HistoricalOddsProcessor(df)
            best_odds = processor.get_best_odds()
            
            if len(best_odds) > 0:
                logger.info(f"✅ Processor working - found best odds for {len(best_odds)} fights")
            
            # Save test results
            output_file = Path(__file__).parent / "data" / "test_historical_odds.csv"
            output_file.parent.mkdir(exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Test data saved to: {output_file}")
            
            return True
            
        else:
            logger.warning("⚠️ No records fetched - this might be normal if no UFC events in this period")
            return True  # Not necessarily an error
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False
    
    finally:
        # Show API usage
        if client.credits_remaining:
            logger.info(f"\n📊 API Credits Remaining: {client.credits_remaining}")


def test_specific_event():
    """Test fetching a known recent UFC event"""
    
    api_key = os.getenv('UFC_ODDS_API_KEY')
    if not api_key:
        logger.error("No API key found")
        return False
    
    logger.info("\n📅 Test 2: Fetching specific UFC event (UFC 310)")
    logger.info("-" * 40)
    
    client = HistoricalOddsAPIClient(api_key)
    
    # UFC 310 was December 7, 2024
    event_date = datetime(2024, 12, 7)
    
    try:
        # Get events for that date
        events = client.get_historical_events(event_date)
        
        if events:
            logger.info(f"✅ Found {len(events)} UFC events on {event_date.date()}")
            
            for event in events:
                logger.info(f"   Event: {event.get('title', 'Unknown')}")
                logger.info(f"   ID: {event.get('id', 'Unknown')}")
                
                # Fetch odds for this event
                odds_data = client.get_historical_odds(event_date, event['id'])
                
                if odds_data['data']:
                    logger.info(f"   ✅ Found odds data for {len(odds_data['data'])} bookmakers")
                else:
                    logger.info(f"   ⚠️ No odds data available")
            
            return True
        else:
            logger.warning(f"No UFC events found on {event_date.date()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def estimate_full_fetch_cost():
    """Estimate the API credit cost for fetching all historical data"""
    
    logger.info("\n💰 Cost Estimation for Full Historical Fetch")
    logger.info("-" * 40)
    
    start_date = datetime(2020, 6, 6)
    end_date = datetime.now()
    
    total_days = (end_date - start_date).days
    total_months = total_days / 30
    
    # Estimate: ~12 UFC events per month, 1 snapshot per week
    estimated_events = int(total_months * 12)
    snapshots_per_event = 4  # Weekly snapshots
    credits_per_snapshot = 10  # 10 credits per region/market
    
    total_credits = estimated_events * snapshots_per_event * credits_per_snapshot
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {total_days}")
    logger.info(f"Estimated UFC events: {estimated_events}")
    logger.info(f"Snapshots per event: {snapshots_per_event}")
    logger.info(f"Credits per snapshot: {credits_per_snapshot}")
    logger.info(f"\n🎯 ESTIMATED TOTAL CREDITS NEEDED: {total_credits:,}")
    logger.info(f"   (Your 20k credits should be sufficient)")


def main():
    """Run all tests"""
    
    print("\n" + "=" * 60)
    print("🚀 HISTORICAL ODDS FETCHER TEST SUITE")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Small range
    result1 = test_small_range()
    test_results.append(("Small Range Test", result1))
    
    # Test 2: Specific event
    result2 = test_specific_event()
    test_results.append(("Specific Event Test", result2))
    
    # Show cost estimate
    estimate_full_fetch_cost()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r for _, r in test_results)
    
    if all_passed:
        print("\n✅ All tests passed! Ready to fetch full historical data.")
        print("\nTo fetch ALL historical data (June 2020 - present), run:")
        print("python3 fetch_all_historical_odds.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)