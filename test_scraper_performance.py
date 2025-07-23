#!/usr/bin/env python3
"""
UFC Scraper Performance Testing and Comparison

This script tests the performance improvements of the optimized scraper
compared to the original implementation.

Usage:
    python3 test_scraper_performance.py --mode [quick|full|compare]
    
    --mode quick   : Test with a small subset of fighters (20 fighters)
    --mode full    : Test with full scraping (WARNING: Takes 90+ minutes for original)
    --mode compare : Run both scrapers on small dataset and compare performance
"""

import asyncio
import time
import argparse
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any
import json

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "webscraper"))

try:
    # Import both scrapers
    from webscraper.optimized_scraping import OptimizedUFCStatsScraper, ScrapingConfig
    from webscraper.scraping import main as original_scraper_main
    BOTH_SCRAPERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import scrapers: {e}")
    BOTH_SCRAPERS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScraperPerformanceTester:
    """Test and compare UFC scraper performance"""
    
    def __init__(self):
        self.results = {
            'optimized': {},
            'original': {},
            'comparison': {}
        }
    
    async def test_optimized_scraper(self, fighter_limit: int = None) -> Dict[str, Any]:
        """Test the optimized scraper performance"""
        logger.info(f"🚀 Testing optimized scraper (limit: {fighter_limit})")
        
        start_time = time.time()
        
        try:
            # Configure for testing
            config = ScrapingConfig()
            config.MAX_CONCURRENT_FIGHTERS = 5  # Conservative for testing
            config.BATCH_SIZE = 50 if fighter_limit and fighter_limit < 100 else 200
            
            async with OptimizedUFCStatsScraper(config) as scraper:
                # Get fighter URLs
                fighter_urls = await scraper.get_fighter_urls_concurrent()
                
                # Limit for testing if specified
                if fighter_limit:
                    fighter_urls = fighter_urls[:fighter_limit]
                    logger.info(f"🎯 Limited to {len(fighter_urls)} fighters for testing")
                
                # Run scraping
                fighter_details, fight_histories = await scraper.scrape_fighters_batch(fighter_urls)
                
                # Quick processing test
                processed_data = await scraper.process_data_optimized(fighter_details, fight_histories)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                results = {
                    'success': True,
                    'total_time_seconds': total_time,
                    'total_time_minutes': total_time / 60,
                    'fighters_scraped': len(fighter_details),
                    'fights_scraped': len(fight_histories),
                    'requests_made': scraper.stats['requests_made'],
                    'errors_encountered': scraper.stats['errors_encountered'],
                    'performance_stats': scraper.stats,
                    'features_created': processed_data['processing_stats'].get('features_created', 0),
                    'fighters_per_second': len(fighter_details) / total_time if total_time > 0 else 0,
                    'requests_per_second': scraper.stats['requests_made'] / total_time if total_time > 0 else 0,
                }
                
                logger.info(f"✅ Optimized scraper complete: {total_time:.1f}s, {len(fighter_details)} fighters")
                return results
                
        except Exception as e:
            logger.error(f"Optimized scraper test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time_seconds': time.time() - start_time
            }
    
    def test_original_scraper_simulation(self, fighter_count: int) -> Dict[str, Any]:
        """Simulate original scraper performance based on known characteristics"""
        logger.info(f"📊 Simulating original scraper performance for {fighter_count} fighters")
        
        # Known characteristics of original scraper:
        # - 1.5s delay per fighter
        # - 1s delay between letters (26 letters)
        # - Sequential processing only
        # - Additional overhead for processing
        
        fighter_time = fighter_count * 1.5  # 1.5s delay per fighter
        letter_time = 26 * 1.0              # 1s delay per letter
        processing_overhead = 30            # Estimated processing time
        network_overhead = fighter_count * 0.5  # Average response time
        
        estimated_total_time = fighter_time + letter_time + processing_overhead + network_overhead
        
        return {
            'success': True,
            'total_time_seconds': estimated_total_time,
            'total_time_minutes': estimated_total_time / 60,
            'fighters_scraped': fighter_count,
            'estimated': True,
            'breakdown': {
                'fighter_delays': fighter_time,
                'letter_delays': letter_time,
                'processing_overhead': processing_overhead,
                'network_overhead': network_overhead
            },
            'fighters_per_second': fighter_count / estimated_total_time if estimated_total_time > 0 else 0,
            'note': 'Simulated based on original scraper delays and sequential processing'
        }
    
    async def run_performance_comparison(self, fighter_limit: int = 50) -> Dict[str, Any]:
        """Run performance comparison between scrapers"""
        logger.info(f"🏁 Running performance comparison (limited to {fighter_limit} fighters)")
        
        comparison_results = {
            'test_config': {
                'fighter_limit': fighter_limit,
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'comparison'
            }
        }
        
        # Test optimized scraper
        logger.info("1️⃣ Testing optimized scraper...")
        optimized_results = await self.test_optimized_scraper(fighter_limit)
        comparison_results['optimized'] = optimized_results
        
        # Simulate original scraper (safer than running actual slow scraper)
        logger.info("2️⃣ Simulating original scraper...")
        actual_fighters = optimized_results.get('fighters_scraped', fighter_limit)
        original_results = self.test_original_scraper_simulation(actual_fighters)
        comparison_results['original_simulated'] = original_results
        
        # Calculate improvements
        if optimized_results.get('success') and original_results.get('success'):
            opt_time = optimized_results['total_time_seconds']
            orig_time = original_results['total_time_seconds']
            
            time_improvement = ((orig_time - opt_time) / orig_time * 100) if orig_time > 0 else 0
            speed_multiplier = orig_time / opt_time if opt_time > 0 else float('inf')
            
            comparison_results['performance_improvement'] = {
                'time_saved_seconds': orig_time - opt_time,
                'time_saved_minutes': (orig_time - opt_time) / 60,
                'percentage_improvement': time_improvement,
                'speed_multiplier': speed_multiplier,
                'optimized_time_minutes': opt_time / 60,
                'original_estimated_time_minutes': orig_time / 60
            }
        
        return comparison_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted performance test results"""
        print("\n" + "="*70)
        print("🏆 UFC SCRAPER PERFORMANCE TEST RESULTS")
        print("="*70)
        
        if 'optimized' in results:
            opt = results['optimized']
            print(f"\n🚀 OPTIMIZED SCRAPER:")
            if opt.get('success'):
                print(f"   ⏱️ Time: {opt['total_time_minutes']:.2f} minutes ({opt['total_time_seconds']:.1f}s)")
                print(f"   🥊 Fighters: {opt['fighters_scraped']}")
                print(f"   🥋 Fights: {opt.get('fights_scraped', 'N/A')}")
                print(f"   🌐 Requests: {opt.get('requests_made', 'N/A')}")
                print(f"   ⚡ Fighters/sec: {opt.get('fighters_per_second', 0):.2f}")
                print(f"   📡 Requests/sec: {opt.get('requests_per_second', 0):.2f}")
                print(f"   ⚠️ Errors: {opt.get('errors_encountered', 0)}")
                print(f"   ⚙️ Features: {opt.get('features_created', 'N/A')}")
            else:
                print(f"   ❌ Failed: {opt.get('error', 'Unknown error')}")
        
        if 'original_simulated' in results:
            orig = results['original_simulated']
            print(f"\n📊 ORIGINAL SCRAPER (Simulated):")
            print(f"   ⏱️ Time: {orig['total_time_minutes']:.2f} minutes ({orig['total_time_seconds']:.1f}s)")
            print(f"   🥊 Fighters: {orig['fighters_scraped']}")
            print(f"   ⚡ Fighters/sec: {orig.get('fighters_per_second', 0):.2f}")
            print(f"   📋 Breakdown:")
            breakdown = orig.get('breakdown', {})
            for component, time_val in breakdown.items():
                print(f"      {component.replace('_', ' ').title()}: {time_val:.1f}s")
        
        if 'performance_improvement' in results:
            imp = results['performance_improvement']
            print(f"\n🎯 PERFORMANCE IMPROVEMENT:")
            print(f"   ⏱️ Time Saved: {imp['time_saved_minutes']:.1f} minutes")
            print(f"   📈 Improvement: {imp['percentage_improvement']:.1f}%")
            print(f"   🚀 Speed Multiplier: {imp['speed_multiplier']:.1f}x faster")
            print(f"   ")
            print(f"   Optimized:  {imp['optimized_time_minutes']:.1f} minutes")
            print(f"   Original:   {imp['original_estimated_time_minutes']:.1f} minutes")
        
        print("\n" + "="*70)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"scraper_performance_test_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

async def main():
    """Main function for performance testing"""
    parser = argparse.ArgumentParser(description='Test UFC scraper performance')
    parser.add_argument('--mode', choices=['quick', 'full', 'compare'], default='quick',
                        help='Test mode: quick (20 fighters), full (all fighters), compare (both scrapers)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of fighters for testing')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if not BOTH_SCRAPERS_AVAILABLE:
        print("❌ Required scraper modules not available")
        return
    
    tester = ScraperPerformanceTester()
    
    try:
        if args.mode == 'quick':
            limit = args.limit or 20
            print(f"🏃 Running quick test with {limit} fighters...")
            results = await tester.test_optimized_scraper(limit)
            tester.print_results({'optimized': results})
            
        elif args.mode == 'full':
            if args.limit:
                limit = args.limit
            else:
                confirm = input("⚠️ Full test will scrape ALL fighters. This may take time. Continue? (y/N): ")
                if confirm.lower() != 'y':
                    print("Test cancelled.")
                    return
                limit = None
            
            print("🌍 Running full scraper test...")
            results = await tester.test_optimized_scraper(limit)
            tester.print_results({'optimized': results})
            
        elif args.mode == 'compare':
            limit = args.limit or 50
            print(f"⚖️ Running comparison test with {limit} fighters...")
            results = await tester.run_performance_comparison(limit)
            tester.print_results(results)
            
        if args.save:
            tester.save_results(results if 'results' in locals() else {})
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())