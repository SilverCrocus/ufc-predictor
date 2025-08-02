#!/usr/bin/env python3
"""
Test script for the Autonomous UFC Predictor Pipeline

This script validates that the pipeline can execute properly
without requiring actual web scraping or long training times.
"""

import asyncio
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_pipeline import AutonomousPipeline, PipelineConfig, display_pipeline_results


def create_mock_data():
    """Create mock UFC data for testing"""
    
    # Create temporary data directory
    temp_data_dir = Path('data') / f'scrape_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock fighters data - using formats expected by feature engineering
    fighters_data = {
        'Name': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
        'Wins': [10, 8, 12, 6],
        'Losses': [2, 4, 1, 8],
        'Height': ['6\' 0"', '5\' 8"', '6\' 2"', '5\' 10"'],  # Height format expected
        'Weight': ['185 lbs.', '170 lbs.', '205 lbs.', '155 lbs.'],  # Weight format expected
        'Reach': ['74"', '70"', '76"', '68"'],  # Reach format expected
        'Age': [28, 32, 25, 30],
        'Stance': ['Orthodox', 'Southpaw', 'Orthodox', 'Switch'],
        'SLpM': [4.2, 3.8, 5.1, 3.5],
        'SApM': [2.8, 3.2, 2.1, 4.1],
        'TDAvg': [2.1, 1.5, 3.2, 0.8],
        'TDAcc': ['45%', '38%', '62%', '25%'],  # Percentage format expected
        'TDDef': ['78%', '82%', '71%', '65%'],  # Percentage format expected
        'SubAvg': [0.5, 0.2, 1.1, 0.1]
    }
    
    fighters_df = pd.DataFrame(fighters_data)
    fighters_file = temp_data_dir / f'ufc_fighters_raw_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    fighters_df.to_csv(fighters_file, index=False)
    
    # Create mock fights data  
    fights_data = {
        'Fighter': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
        'Opponent': ['Fighter B', 'Fighter A', 'Fighter D', 'Fighter C'],
        'Outcome': ['Win', 'Loss', 'Win', 'Loss'],
        'Method': ['Decision', 'Decision', 'KO/TKO', 'KO/TKO'],
        'Time': ['5:00', '5:00', '2:34', '2:34'],
        'Event': ['UFC Test 1', 'UFC Test 1', 'UFC Test 2', 'UFC Test 2']
    }
    
    fights_df = pd.DataFrame(fights_data)
    fights_file = temp_data_dir / f'ufc_fights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    fights_df.to_csv(fights_file, index=False)
    
    return str(temp_data_dir), str(fighters_file), str(fights_file)


class MockWebScraperAgent:
    """Mock webscraper for testing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = print  # Simple logging
        
    async def execute_scraping(self):
        """Mock scraping that creates test data"""
        print("🕷️ Mock scraping: Creating test data...")
        
        try:
            # Create mock data
            data_dir, fighters_file, fights_file = create_mock_data()
            
            # Simulate validation
            import pandas as pd
            fighters_df = pd.read_csv(fighters_file)
            fights_df = pd.read_csv(fights_file)
            
            # Calculate quality score
            fighters_completeness = 1.0 - (fighters_df.isnull().sum().sum() / 
                                         (len(fighters_df) * len(fighters_df.columns)))
            fights_completeness = 1.0 - (fights_df.isnull().sum().sum() / 
                                       (len(fights_df) * len(fights_df.columns)))
            
            quality_score = (fighters_completeness + fights_completeness) / 2
            
            print(f"✅ Mock scraping completed: {len(fighters_df)} fighters, {len(fights_df)} fights")
            print(f"   Data quality score: {quality_score:.3f}")
            
            return {
                'status': 'success',
                'scraper_used': 'mock_scraper',
                'attempt': 1,
                'stdout': 'Mock scraping successful',
                'data_directory': data_dir,
                'data_valid': True,
                'data_quality_score': quality_score,
                'fighters_count': len(fighters_df),
                'fights_count': len(fights_df),
                'fighters_completeness': fighters_completeness,
                'fights_completeness': fights_completeness,
                'latest_fighters_file': fighters_file,
                'latest_fights_file': fights_file
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Mock scraping failed: {e}'
            }


async def test_pipeline():
    """Test the autonomous pipeline with mock data"""
    print("🧪 Testing Autonomous UFC Predictor Pipeline")
    print("=" * 60)
    
    # Create test configuration
    config = PipelineConfig(
        debug_mode=True,
        tune_hyperparameters=False,  # Disable for faster testing
        max_retries=1,
        scraping_timeout=60,
        training_timeout=300,  # 5 minutes for testing
        webscraper_script="mock_scraper"  # Will be ignored in our test
    )
    
    # Create pipeline with mock webscraper
    pipeline = AutonomousPipeline(config)
    pipeline.webscraper_agent = MockWebScraperAgent(config)
    
    try:
        # Execute pipeline
        result = await pipeline.execute_pipeline()
        
        # Display results
        display_pipeline_results(result)
        
        # Cleanup test data
        if result.scraping_completed:
            try:
                # Find and remove test data directories
                data_dir = Path('data')
                for test_dir in data_dir.glob('scrape_test_*'):
                    shutil.rmtree(test_dir)
                    print(f"🧹 Cleaned up test data: {test_dir}")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
        
        return result.success
        
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        return False


async def main():
    """Main test function"""
    print("Starting Autonomous Pipeline Test...")
    
    success = await test_pipeline()
    
    if success:
        print("\n🎉 AUTONOMOUS PIPELINE TEST PASSED!")
        print("The pipeline is ready for production use.")
        sys.exit(0)
    else:
        print("\n❌ AUTONOMOUS PIPELINE TEST FAILED!")
        print("Please check the errors above and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test failed with unexpected error: {e}")
        sys.exit(1)